import http from 'k6/http';
import { SharedArray } from 'k6/data';
import { Trend } from 'k6/metrics';
import { sleep, check } from 'k6';

const origin = __ENV.ORIGIN;
const apiKey = __ENV.API_KEY;
const apiKeyLocation = __ENV.API_KEY_LOCATION || 'header'; // body || header
const hostHeader = __ENV.HOST;

let characterLengthTrend = new Trend('character_length');

export let options = {
  scenarios: {
    ramping_request_rate: {
      executor: 'ramping-arrival-rate',
      startRate: 8,
      timeUnit: '1m',
      stages: [
        // Based off of real usage (on 1-min interval)
        { target: 10, duration: '1m' },
        { target: 30, duration: '10s' },
        { target: 30, duration: '1m' },
        { target: 25, duration: '1m' },
        { target: 50, duration: '10s' },
        { target: 50, duration: '1m' },
        { target: 10, duration: '10s' },
        { target: 10, duration: '1m' },
        { target: 30, duration: '10s' },
        { target: 30, duration: '1m' },
      ],
      preAllocatedVUs: 4,
      maxVUs: 64,
      gracefulStop: '60s',
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.01'],   // http errors should be less than 1%
    http_req_waiting: ['p(95)<30000'], // 95 percentile of requests should be below 30s
    checks: ['rate>0.98']
  },
  minIterationDuration: '1s',
  // Recommended: https://k6.io/docs/using-k6/options/#discard-response-bodies
  discardResponseBodies: true,
};

const lines = new SharedArray('tos', function loadTosText() {
  return open('../data/terms.txt').split('\n');
});

const actors = new SharedArray('actors', function loadActors() {
  return open('../data/actors.txt').split('\n');
});

export default function main() {
  // __VU is the current virtual user number.
  // __ITER is a counter that's incremented everytime each VU executes the code defined here.
  // We use these two values to deterministically select a line of text, so that the tests
  // are consistent while still maintaining some degree of variability to better simulate "real"
  // traffic.
  const url = `${origin}/api/text_to_speech/stream`;
  // text
  const maxLineNo = lines.length - 1;
  const lineNo = Math.min(__VU + __ITER, (__VU + __ITER) % maxLineNo);
  const text = lines[lineNo];
  characterLengthTrend.add(text.length);
  // actor
  const maxActorIdx = actors.length - 1;
  const actorIdx = Math.min(__VU + __ITER, (__VU + __ITER) % maxActorIdx);
  const actor = actors[actorIdx];
  // request
  const body = JSON.stringify({
    text,
    speaker_id: actor,
    api_key: apiKeyLocation === 'body' ? apiKey : undefined,
  });
  const options = {
    headers: {
      'Content-Type': 'application/json',
      'Accept-Version': 'latest',
      'X-Api-Key': apiKeyLocation === 'header' ? apiKey : undefined,
    },
    timeout: '5m',
  };
  if (hostHeader) options.headers['Host'] = hostHeader
  const streamResponse = http.post(url, body, options);
  check(streamResponse, {
    'received_200_response': (r) => {
      return r.status === 200
    },
    'received_audio_response': (r) => {
      return r.headers['Content-Type'] === 'audio/mpeg'
    },
  })
  // Incoorperate some randomness, see: https://stackoverflow.com/a/61118956/2578619
  sleep(Math.floor(Math.random() * 4) + 1);
}
