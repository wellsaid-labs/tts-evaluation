# TTS Cluster Verification

This directory includes code for a small program that verifies a TTS cluster
by submitting sample requests to it.

Run the tests like so:

```
API_KEY=XXX go run main.go ../deployments/staging staging.tts.wellsaidlabs.com
```

The tests make up to 20 requests at once. The actual request rate is probably
much slower than 20 rps in practice because of API latency.

