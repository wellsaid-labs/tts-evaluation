# Load Tests

This directory contains code for load tests for assessing the performance and scalability
of the TTS service.

The tests are executed via [k6](https://k6.io).

## Running the Tests

To run the tests:

1. Create an `*.env` file with the desired configuration variables, like so:

    ```env
    ORIGIN=https://ttsdev.wellsaidlabs.com
    API_KEY=XXXXXXXXXXXX
    API_KEY_LOCATION=<header|body>
    ```

  Where `ORIGIN` is the HTTP origin of the TTS service that you'd like to test, and `API_KEY`
  is the API key that should be used by the tests.

2. Build and run the tests like so:

    ```
    docker run --rm -it --env-file dev.env $(docker build -q .) | tee results_$(date +%Y_%m_%d_%H:%M).txt
    ```

   The results will be written both to stdout and to a a file named `results_YYYY_MM_DD_HH:MM.txt`
   in your working directory.

## Data

Most of the tests use sample data. Data lives in the `data/` directory. If you add a dataset
add a line here noting what the file contains and the characteristics of the data.

### data/terms.txt

The WellSaidLabs terms of use: https://wellsaidlabs.com/legal?document=terms_of_service.

Each line is a single paragraph of text. The character distribution per line is as
follows:

```
  11 300-400
  11 200-300
   9 400-500
   8 0-100
   6 500-600
   5 100-200
   3 600-700
   2 700-800
   1 1400-1500
```

