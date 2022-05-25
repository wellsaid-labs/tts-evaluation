/**
 * This file contains our load testing behavior and configuration following
 * the k6 scripting format. When running our load test, this script is passed
 * to the k6 command as an argument. The `default export` is a function that
 * is intended to mimic one use case of our API. Currently that is making a
 * single API call to the `input_validated` endpoint, followed by another
 * API call to the `stream` endpoint.
 *
 * @see https://k6.io/docs/using-k6/options/ `options`
 * @see https://k6.io/docs/using-k6/test-life-cycle `default export`
 */
import { check, sleep } from "k6";
import { SharedArray } from "k6/data";
import http from "k6/http";
import { Trend } from "k6/metrics";

const origin = __ENV.ORIGIN;
const apiKey = __ENV.API_KEY;
const apiKeyLocation = __ENV.API_KEY_LOCATION || "header"; // body || header
const hostHeader = __ENV.HOST;
const skipValidationEndpoint =
  __ENV.SKIP_VALIDATION_ENDPOINT === "true" || false;
const apiPathPrefix = __ENV.API_PATH_PREFIX || "v1/tts"; // v1/tts || api/text_to_speech

let characterLengthTrend = new Trend("character_length");

export let options = {
  scenarios: {
    ramping_request_rate: {
      // https://k6.io/docs/using-k6/scenarios/executors/ramping-arrival-rate/
      executor: "ramping-arrival-rate",
      startRate: 8,
      timeUnit: "1s",
      stages: [
        // Sample ramp up to 20iter/sec and back down
        { target: 1, duration: "30s" },
        { target: 5, duration: "2m" },
        { target: 10, duration: "2m" },
        { target: 20, duration: "1m" },
        { target: 5, duration: "1m" },
        { target: 1, duration: "30s" },
      ],
      preAllocatedVUs: 4,
      maxVUs: 256,
      gracefulStop: "60s",
    },
  },
  thresholds: {
    http_req_failed: ["rate<0.01"], // http errors should be less than 1%
    http_req_waiting: ["p(95)<30000"], // 95 percentile of requests should be below 30s
    checks: ["rate>0.98"],
  },
  minIterationDuration: "1s",
  // Recommended: https://k6.io/docs/using-k6/options/#discard-response-bodies
  discardResponseBodies: true,
};

const lines = new SharedArray("tos", function loadTosText() {
  return open("../data/terms.txt").split("\n");
});

const actors = new SharedArray("actors", function loadActors() {
  return open("../data/actors.txt").split("\n");
});

export default function main() {
  // __VU is the current virtual user number.
  // __ITER is a counter that's incremented everytime each VU executes the code defined here.
  // We use these two values to deterministically select a line of text, so that the tests
  // are consistent while still maintaining some degree of variability to better simulate "real"
  // traffic.
  const validateUrl = `${origin}/${apiPathPrefix}/input_validated`;
  const streamUrl = `${origin}/${apiPathPrefix}/stream`;

  const maxLineNo = lines.length - 1;
  const lineNo = Math.min(__VU + __ITER, (__VU + __ITER) % maxLineNo);
  const text = lines[lineNo];
  characterLengthTrend.add(text.length);

  const maxActorIdx = actors.length - 1;
  const actorIdx = Math.min(__VU + __ITER, (__VU + __ITER) % maxActorIdx);
  const actor = actors[actorIdx];

  const body = JSON.stringify({
    text,
    speaker_id: actor,
    api_key: apiKeyLocation === "body" ? apiKey : undefined,
  });
  const options = {
    headers: {
      "Content-Type": "application/json",
      "Accept-Version": "latest",
      "X-Api-Key": apiKeyLocation === "header" ? apiKey : undefined,
    },
    timeout: "5m",
  };
  if (hostHeader) options.headers["Host"] = hostHeader;

  // A: validate inputs
  if (!skipValidationEndpoint) {
    const validateResponse = http.post(validateUrl, body, options);
    check(validateResponse, {
      validate_200_response: (r) => {
        return r.status === 200;
      },
    });
  }

  // B: stream
  const streamResponse = http.post(streamUrl, body, options);
  check(streamResponse, {
    stream_200_response: (r) => {
      return r.status === 200;
    },
    received_audio_response: (r) => {
      return r.headers["Content-Type"] === "audio/mpeg";
    },
  });

  // Incoorperate some randomness, see: https://stackoverflow.com/a/61118956/2578619
  sleep(Math.floor(Math.random() * 4) + 1);
}
