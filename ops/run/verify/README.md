# TTS Cluster Verification

This directory includes code for a small program that verifies a TTS cluster
by submitting a request to the stream and input validation endpoints for
every model deployed to it.

The intent is to use the script as a sanity check for validating that a deploy
or infrastructural adjustment didn't disrupt the functionality of services.

## Run

Run the tests like so:

```
API_KEY=XXX go run main.go ../deployments/staging
```

The hostname defaults to `staging.tts.wellsaidlabs.com`. You can change it
for testing production via the `--host` flag:

```
API_KEY=XXX go run main.go ../deployments/prod --host tts.wellsaidlabs.com
```

The tests make up to 20 requests at once. The actual request rate is probably
much slower than 20 rps in practice because of API latency.

By default all endpoints are tested. You can test a single endpoint
with the `--only` flag:

```
API_KEY=XXX go run main.go --only v10 ../deployments/staging
```

And you can repeat the tests indefinitely, every 10 seconds, like so:

```
API_KEY=XXX go run main.go --only v10 --repeat 10 \
  ../deployments/staging staging.tts.wellsaidlabs.com
```

Finally, for more information run:

```
go run main.go --help
```

## Why Go?

There are a lot of endpoints to test, doing so synchronously would take a lot
of time (and in that case a `bash` script would do).

Go's concurrency model makes a worker pool easy to construct. It'd be a bit
more work in other languages, like Python and Node, to do so the same.

