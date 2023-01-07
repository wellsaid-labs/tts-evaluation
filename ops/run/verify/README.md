# TTS Cluster Verification

This directory includes code for a small program that verifies a TTS cluster
by submitting real requests.

## Run

Run the tests like so:

```
API_KEY=XXX go run main.go ../deployments/staging
```

Change the host like so. You'll have to do this for production:

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

See usage info like so:

```
go run main.go --help
```

## Why Go?

There are a lot of endpoints to test, doing so synchronously would take a lot
of time (and in that case a `bash` script would do).

Go's concurrency model makes a worker pool easy to construct. It'd be a bit
more work in other languages, like Python and Node, to do so the same.

