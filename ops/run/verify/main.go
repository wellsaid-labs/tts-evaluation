package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"os/signal"
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

// Send a request to the endpoint and return the HTTP status code, string and an error if something
// failed while making the request.
func request(ctx context.Context, ep endpoint) (int, string, error) {
	client := http.Client{Timeout: 2 * time.Minute}

	payload := requestPayload{
		SpeakerID:      "4",
		Text:           "Lorem ipsum",
		ConsumerID:     "Test",
		ConsumerSource: "VerifyGKE",
	}

	buf, err := json.Marshal(payload)
	if err != nil {
		return 0, "", err
	}

	body := ioutil.NopCloser(bytes.NewBuffer(buf))
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, ep.url.String(), body)
	if err != nil {
		return 0, "", err
	}

	req.Header.Add("Content-Type", "application/json")
	req.Header.Add("Accept-Version", ep.model)
	req.Header.Add("X-Api-Key", os.Getenv("API_KEY"))

	resp, err := client.Do(req)
	if err != nil {
		return 0, "", err
	}
	defer resp.Body.Close()

	return resp.StatusCode, resp.Status, nil
}

// An endpoint captures the URL (/stream vs /input_validated) and a model version.
type endpoint struct {
	url   *url.URL
	model string
}

type outcome struct {
	endpoint   endpoint
	statusCode int
	status     string
	err        error
}

// A worker tests an endpoint at a time, reading from the provided channel and
// reports the outcome via the result channel.
func worker(endpoints <-chan endpoint, result chan<- outcome) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	for {
		select {
		case ep, more := <-endpoints:
			if !more {
				return
			}
			code, status, err := request(ctx, ep)
			result <- outcome{ep, code, status, err}
		}
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

func fill(queue chan<- endpoint, endpoints []endpoint) {
	for _, ep := range endpoints {
		queue <- ep
	}
}

type summary struct {
	ok       []outcome
	failures []outcome
}

func drain(ctx context.Context, size int, results <-chan outcome) summary {
	sum := summary{}
	for {
		select {
		case r, more := <-results:
			// The channel was closed
			if !more {
				return sum
			}

			if r.err != nil {
				fmt.Printf("%s %s %s\n", r.err, r.endpoint.model, r.endpoint.url)
			} else {
				fmt.Printf("%s %s %s\n", r.status, r.endpoint.model, r.endpoint.url)
			}

			if r.statusCode != 200 {
				sum.failures = append(sum.failures, r)
			} else {
				sum.ok = append(sum.ok, r)
			}

			// Nothing left
			if size == len(sum.ok)+len(sum.failures) {
				return sum
			}
		case <-ctx.Done():
			return sum
		}
	}
}

func runTests(ctx context.Context, dir, host, only string, repeat int) error {
	fmt.Println("Reading endpoints...")
	allEndpoints, err := readEndpoints(dir, host)
	if err != nil {
		return err
	}

	endpoints := []endpoint{}
	for _, e := range allEndpoints {
		if only == "" || e.model == only {
			endpoints = append(endpoints, e)
		}
	}

	fmt.Printf("Found %d endpoints to test\n", len(endpoints))
	queue := make(chan endpoint, len(endpoints))
	results := make(chan outcome, len(endpoints))

	numWorkers := 20
	if numWorkers > len(endpoints) {
		numWorkers = len(endpoints)
	}
	fmt.Printf("Starting %d workers...\n", numWorkers)
	for i := 0; i < numWorkers; i++ {
		go worker(queue, results)
	}

	numOk := 0
	failures := []outcome{}
	if repeat != 0 {
		interval := time.Duration(repeat * 1_000_000_000)
		fmt.Printf("Repeating indefinitely every %s...\n", interval.String())

		ticker := time.NewTicker(interval)
	INTERVAL:
		for {
			select {
			case <-ticker.C:
				fmt.Println("Starting test pass...")
				fill(queue, endpoints)

				s := drain(ctx, len(endpoints), results)
				numOk += len(s.ok)
				failures = append(failures, s.failures...)

				fmt.Printf("Repeating after %s...\n", interval.String())
				ticker.Reset(interval)
			case <-ctx.Done():
				ticker.Stop()
				break INTERVAL
			}
		}
	} else {
		fmt.Println("Starting test pass...")
		fill(queue, endpoints)

		s := drain(ctx, len(endpoints), results)
		numOk += len(s.ok)
		failures = append(failures, s.failures...)
	}

	close(queue)
	close(results)

	fmt.Println()

	suffix := ""
	if len(failures) > 0 {
		suffix = ":"
	}
	fmt.Printf("DONE: %d succeeded, %d failed%s\n", numOk, len(failures), suffix)

	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 1, ' ', 0)
	for _, f := range failures {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\n", f.status, f.endpoint.model, f.endpoint.url, f.err)
	}
	tw.Flush()

	return nil
}

func main() {
	var host, only string
	var repeat int

	cmd := cobra.Command{
		Use:   "verify",
		Short: "A small program for verifying a TTS cluster",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
			defer stop()

			done := make(chan error, 1)
			go func() {
				done <- runTests(ctx, args[0], host, only, repeat)
			}()

			select {
			case err := <-done:
				return err
			case <-ctx.Done():
				return nil
			}
		},
	}

	cmd.Flags().StringVar(
		&host,
		"host",
		"staging.tts.wellsaidlabs.com",
		"The hostname to use when making requests",
	)

	cmd.Flags().StringVar(
		&only,
		"only",
		"",
		"If specified, only endpoints belonging to the provided model version will be tested",
	)

	cmd.Flags().IntVar(
		&repeat,
		"repeat",
		0,
		"If set to a non-zero value,  repeat the tests indefinitely, waiting the specified amount of seconds between each iteration",
	)

	if err := cmd.Execute(); err != nil {
		os.Exit(1)
	}
}
