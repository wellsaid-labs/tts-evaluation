/**
 * This is a minimal testing module for `master.js`.
 *
 * Usage:
 *    $ node tests/service/test_master.js
 */
process.env['API_KEY_SUFFIX'] = '_SPEECH_API_KEY';
process.env['AUTOSCALE_WINDOW'] = '600000';
process.env['AUTOSCALE_LOOP'] = '5000';
process.env['AVAILABILITY_LOOP'] = '500';
process.env['EXTRA_WORKER_PODS'] = '1.0';
process.env['MINIMUM_WORKER_PODS'] = '1';
process.env['WORKER_NODE_POOL'] = 'workers-v2';
process.env['WORKER_NODE_PREFIX'] = 'speech-api-worker-pod';
process.env['WORKER_POD_PREFIX'] = 'speech-api-worker-node';

const assert = require('assert');
const master = require('../../src/service/master');

async function testEventLog() {
  console.log('Running `testEventLog`.');
  const maxTime = 1000;
  const eventLog = new master.EventLog(maxTime);
  eventLog.addEvent('a');
  assert.equal(eventLog.events.length, 1);
  await master.sleep(maxTime * 2);
  assert.equal(eventLog.events.length, 0);
}

async function testEventLogMaxTime() {
  console.log('Running `testEventLogMaxTime`.');
  const eventLog = new master.EventLog();
  eventLog.addEvent('a');
  assert.equal(eventLog.events.length, 1);
  await master.sleep(1000);
  assert.equal(eventLog.events.length, 1);
}

async function testSleep() {
  console.log('Running `testSleep`.');
  const now = Date.now();
  const sleepTime = 1000;
  await master.sleep(sleepTime);
  assert.ok(Date.now() - now >= sleepTime);
}

async function testRetry() {
  console.log('Running `testRetry`.');

  // Test basic case with no error.
  assert.equal(await master.retry(() => 1), 1);

  // Test case such that it always evaluates in an error.
  try {
    await master.retry(() => {
      throw "error";
    });
    assert.ok(false);
  } catch {
    assert.ok(true);
  }

  // Test case that mixes both.
  callCount = 0;
  assert.equal(await master.retry(() => {
    callCount += 1;
    if (callCount < 2) {
      throw "error";
    } else {
      return 2;
    }
  }), 2);
}

async function testAsyncFilter() {
  console.log('Running `testAsyncFilter`.');

  result = await master.asyncFilter([1, 2, 3, 1], async (i) => i == 1);
  expected = [1, 1];
  assert.ok(result.length == expected.length && expected.every(function (u, i) {
    return u === result[i];
  }));

  result = await master.asyncFilter([], async (i) => i == 1);
  expected = [];
  assert.ok(result.length == expected.length);
}

async function testPod() {
  console.log('Running `testPod`.');
  const pod = new master.Pod('podName', 'nodeName', '0.0.0.0', 8000);

  // The pod will not be able to connect to the internet.
  assert.ok(~(await pod.isReady()));
  assert.ok(~pod.isReserved());
  assert.ok(await pod.isDead());
  assert.ok(~(await pod.isAvailable()));

  // Pre-emtively releasing the pod fails.
  assert.throws(pod.release, Error);

  pod.reserve();
  assert.ok(pod.isReserved());
  // Double releasing pod fails.
  assert.throws(pod.reserve, Error);

  pod.release();
  assert.ok(~pod.isReserved());

  await pod.destroy();
  assert.ok(await pod.isDead());
  assert.throws(pod.reserve, Error);
  assert.throws(pod.release, Error);
}

async function testPodPoolGetNumShortTermPods() {
  console.log('Running `testPodPoolGetNumShortTermPods`.');
  // There are no jobs that need burst.
  assert.equal(master.PodPool.getNumShortTermPods(0, 5, 49000, 42000, 1), 0);

  // The current jobs will be finished before a new pod can be built.
  assert.equal(master.PodPool.getNumShortTermPods(10, 5, 49000, 7000, 1), 2);

  // New pods can be built to handle the backlog.
  assert.equal(master.PodPool.getNumShortTermPods(10, 5, 49000, 42000, 1), 10);
  assert.equal(master.PodPool.getNumShortTermPods(10, 0, 49000, 42000, 1), 10);
}

async function testPodPoolGetNumLongTermPods() {
  console.log('Running `testPodPoolGetNumLongTermPods`.');
  let eventLog = new master.EventLog();
  eventLog.addEvent(1);
  await master.sleep(500);
  eventLog.addEvent(2);
  await master.sleep(1000);
  eventLog.addEvent(3);
  await master.sleep(500);
  eventLog.addEvent(2);
  await master.sleep(250);
  eventLog.addEvent(1);
  assert.equal(master.PodPool.getNumLongTermPods(eventLog, 0, 0), 2);

  // Test min pods
  assert.equal(master.PodPool.getNumLongTermPods(eventLog, 3, 0), 3);

  // Test extra pods
  assert.equal(master.PodPool.getNumLongTermPods(eventLog, 0, 1.0), 4);

  // Test extra pods and min pods
  assert.equal(master.PodPool.getNumLongTermPods(eventLog, 3, 1.0), 4);
  assert.equal(master.PodPool.getNumLongTermPods(eventLog, 5, 1.0), 5);

  // Test one event
  eventLog = new master.EventLog();
  eventLog.addEvent(1);
  assert.equal(master.PodPool.getNumLongTermPods(eventLog, 0, 0), 1);
  assert.equal(master.PodPool.getNumLongTermPods(eventLog, 2, 0), 2);
  assert.equal(master.PodPool.getNumLongTermPods(eventLog, 2, 1.0), 2);
  assert.equal(master.PodPool.getNumLongTermPods(eventLog, 0, 1.0), 2);

  // Test no events
  assert.equal(master.PodPool.getNumLongTermPods(new master.EventLog(), 0, 0), 0);
}

async function main() {
  await testEventLog();
  await testEventLogMaxTime();
  await testSleep();
  await testRetry();
  await testAsyncFilter();
  await testPod();
  await testPodPoolGetNumShortTermPods();
  await testPodPoolGetNumLongTermPods();
}

main();
