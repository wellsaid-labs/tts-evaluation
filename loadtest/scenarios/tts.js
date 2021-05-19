import http from 'k6/http';
import { SharedArray } from 'k6/data';
import { Trend } from 'k6/metrics';

const origin = __ENV.ORIGIN;
const apiKey = __ENV.API_KEY;

let characterLengthTrend = new Trend('character_length');

export let options = {
  stages: [
    { duration: '30s', target: 5 },
    { duration: '30s', target: 10 },
    { duration: '30s', target: 15 },
    { duration: '30s', target: 30 },
  ],
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
  const url = `${origin}/api/speech_synthesis/v1/text_to_speech/stream`;
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
    api_key: apiKey,
  });
  const options = {
    headers: { 'Content-Type': 'application/json' },
    timeout: '5m',
  };
  http.post(url, body, options);
}
