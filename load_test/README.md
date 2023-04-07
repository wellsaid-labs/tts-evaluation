# Load Tests

This directory contains code for load tests for assessing the performance and
scalability of the TTS service.

The tests are executed via [k6](https://k6.io).

## Running the Tests

To run the tests:

1. Create an `*.env` file with the desired configuration variables, like so:

   ```env
   ORIGIN=https://staging.tts.wellsaidlabs.com
   API_KEY=XXXXXXXXXXXX
   API_KEY_LOCATION=<header|body>
   API_PATH_PREFIX=<api/text_to_speech|v1/tts>
   SKIP_VALIDATION_ENDPOINT=<true|false>
   MODEL_VERSION=<latest>
   ```

Where `ORIGIN` is the HTTP origin of the TTS service that you'd like to test,
and `API_KEY` is the API key that should be used by the tests. Reference the
[Kong Consumers](../ops/gateway/README.md) section of the gateway docs for
instructions on how to generate an api key.

The `API_KEY_LOCATION` variable defaults to `header`, authenticating via the
`X-Api-Key` header. This value is configurable in order to support our previous
infrastructure setup that authenticated via the `api_key` body parameter.

By default we chain the `input_validated` and `stream` endpoints to mimic real
usage. If you would like to test the stream endpoint in isolation set the
`SKIP_VALIDATION_ENDPOINT=true` env variable.

Set the `API_PATH_PREFIX` env variable accordingly:

- TTS service (tts.wellsaidlabs.com): `api/text_to_speech`
- Developer API (api.wellsaidlabs.com): `v1/tts`

The `MODEL_VERSION` variable specifies which TTS model version to request, defaulting to `latest`.

1. Build and run the tests like so (replacing `dev.env` with your environment
   configuration file):

   ```bash
   docker run --rm -it --env-file dev.env $(docker build -q .) | tee results_$(date +%Y_%m_%d_%H:%M).txt
   ```

   The results will be written both to stdout and to a a file named
   `results_YYYY_MM_DD_HH:MM.txt` in your working directory.

## Data

Most of the tests use sample data. Data lives in the `data/` directory. If you
add a dataset add a line here noting what the file contains and the
characteristics of the data.

### data/terms.txt

The WellSaidLabs terms of use:
<https://wellsaidlabs.com/legal?document=terms_of_service>.

Each line is a single paragraph of text. The character distribution per line is
as follows:

```txt
  23 0-100
  11 300-400
  11 200-300
   9 400-500
   7 100-200
   6 500-600
   3 600-700
   2 700-800
   1 900-1000
```

### data/actors.txt

A list of valid actor ids available in the `latest` model release. This is
helpful in accounting for variances in model output and better reflects API
usage.

```txt
   2
   3
   7
   12
```
