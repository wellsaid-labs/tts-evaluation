package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"text/tabwriter"
	"time"

	"github.com/spf13/cobra"
)

type requestPayload struct {
	SpeakerID      string `json:"speaker_id"`
	Text           string `json:"text"`
	ConsumerID     string `json:"consumerId"`
	ConsumerSource string `json:"consumerSource"`
}

// Send a request to the endpoint. Returns an error on a non-200 response.
func request(ep endpoint) error {
	client := http.Client{Timeout: 2 * time.Minute}

	payload := requestPayload{
		SpeakerID:      "4",
		Text:           "Lorem ipsum",
		ConsumerID:     "Test",
		ConsumerSource: "VerifyGKE",
	}

	buf, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	req := &http.Request{
		Method: http.MethodPost,
		URL:    ep.url,
		Body:   ioutil.NopCloser(bytes.NewBuffer(buf)),
		Header: http.Header{
			"Content-Type":   {"application/json"},
			"Accept-Version": {ep.model},
			"X-Api-Key":      {os.Getenv("API_KEY")},
		},
	}

	resp, err := client.Do(req)
	if err != nil {
		return err
	}

	if resp.StatusCode != 200 {
		return errors.New(resp.Status)
	}

	return nil
}

// An endpoint captures the URL (/stream vs /input_validated) and a model version.
type endpoint struct {
	url   *url.URL
	model string
}

type status struct {
	endpoint endpoint
	err      error
}

// A worker tests an endpoint at a time, reading from the provided channel and
// reports the outcome via the result channel.
func worker(endpoints <-chan endpoint, result chan<- status) {
	for ep := range endpoints {
		result <- status{ep, request(ep)}
	}
}

type endpointSpec struct {
	Paths []string `json:"paths"`
}

type modelConfig struct {
	Model    string       `json:"model"`
	Stream   endpointSpec `json:"stream"`
	Validate endpointSpec `json:"validate"`
}

// Returns a list of endpoints to test, per the provided path and hostname.
func readEndpoints(dir, host string) ([]endpoint, error) {
	if dir == "" || host == "" {
		return nil, errors.New("Usage: go run main.go <path> <host>, i.e. go run main.go ../deployments/staging staging.tts.wellsaidlabs.org")
	}

	path, err := filepath.Abs(dir)
	if err != nil {
		return nil, err
	}

	files, err := os.ReadDir(path)
	if err != nil {
		return nil, err
	}

	endpoints := []endpoint{}
	for _, file := range files {
		stat, err := file.Info()
		if err != nil {
			return nil, err
		}

		fp := filepath.Join(path, file.Name())
		if !stat.Mode().IsRegular() {
			return nil, fmt.Errorf("%s isn't a regular file", fp)
		}

		contents, err := os.ReadFile(fp)
		if err != nil {
			return nil, err
		}

		var conf modelConfig
		if err := json.Unmarshal(contents, &conf); err != nil {
			return nil, err
		}

		for _, s := range []endpointSpec{conf.Stream, conf.Validate} {
			for _, p := range s.Paths {
				endpoints = append(endpoints, endpoint{
					url: &url.URL{
						Scheme: "https",
						Host:   host,
						Path:   p,
					},
					model: conf.Model,
				})
			}
		}
	}

	return endpoints, nil
}

func runTests(dir, host string) error {
	fmt.Println("Reading endpoints...")
	endpoints, err := readEndpoints(dir, host)
	if err != nil {
		return err
	}

	fmt.Printf("Found %d endpoints to test\n", len(endpoints))
	queue := make(chan endpoint, len(endpoints))

	numWorkers := 20
	fmt.Printf("Starting %d workers...\n", numWorkers)
	results := make(chan status)
	for i := 0; i < numWorkers; i++ {
		go worker(queue, results)
	}

	fmt.Println("Enqueing work...")
	for _, ep := range endpoints {
		queue <- ep
	}
	close(queue)

	ok := []endpoint{}
	failures := []status{}
	for i := 0; i < len(endpoints); i++ {
		r := <-results
		if r.err != nil {
			fmt.Printf("%s %s %s\n", r.err, r.endpoint.model, r.endpoint.url)
			failures = append(failures, r)
			continue
		}

		fmt.Printf("200 OK %s %s\n", r.endpoint.model, r.endpoint.url)

		ok = append(ok, r.endpoint)
		continue
	}

	fmt.Println()

	fmt.Printf("DONE: %d succeeded, %d failed:\n", len(ok), len(failures))

	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 1, ' ', 0)
	for _, f := range failures {
		fmt.Fprintf(tw, "%s\t%s\t%s\n", f.err, f.endpoint.model, f.endpoint.url)
	}
	tw.Flush()

	fmt.Println("DONE")

	return nil
}

func newVerifyAllCommand() *cobra.Command {
	var host string

	cmd := cobra.Command{
		Use:   "all PATH",
		Short: "Verifies all endpoints once. PATH should be a directory containing endpoint definitions.",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runTests(args[0], host)
		},
	}

	cmd.Flags().StringVar(
		&host,
		"host",
		"staging.tts.wellsaidlabs.com",
		"The hostname to use when making requests",
	)

	return &cmd
}

func main() {
	root := cobra.Command{
		Use:   "verify",
		Short: "A small program for verifying a TTS cluster",
	}

	root.AddCommand(newVerifyAllCommand())

	if err := root.Execute(); err != nil {
		os.Exit(1)
	}
}
