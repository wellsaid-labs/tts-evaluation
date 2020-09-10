/**
 * This is a minimal testing module for `master.js`.
 *
 * Usage:
 *    $ node tests/service/test_master.js
 */
process.env['API_KEY_SUFFIX'] = '_SPEECH_API_KEY';
process.env['AUTOSCALE_WINDOW'] = '600000';
process.env['AUTOSCALE_LOOP'] = '5000';
process.env['EXTRA_WORKER_PODS'] = '1';
process.env['MINIMUM_WORKER_PODS'] = '1';
process.env['WORKER_NODE_POOL'] = 'workers-v2';
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
  assert.ok(~(await pod.isDead()));
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
  let eventLog = new master.EventLog(2000);
  eventLog.addEvent(1);
  await master.sleep(500);
  eventLog.addEvent(2);
  await master.sleep(1000);
  eventLog.addEvent(3);
  await master.sleep(500);
  eventLog.addEvent(2);
  await master.sleep(250);
  eventLog.addEvent(1);
  assert.equal(master.PodPool.getNumLongTermPods(eventLog, 0, 0, 0.5), 2);

  // Test min pods
  assert.equal(master.PodPool.getNumLongTermPods(eventLog, 3, 0, 0.5), 3);

  // Test extra pods
  assert.equal(master.PodPool.getNumLongTermPods(eventLog, 0, 1, 0.5), 3);

  // Test extra pods and min pods
  assert.equal(master.PodPool.getNumLongTermPods(eventLog, 3, 1, 0.5), 3);
  assert.equal(master.PodPool.getNumLongTermPods(eventLog, 5, 1, 0.5), 5);

  // Test one event
  eventLog = new master.EventLog(250);
  await master.sleep(500);
  eventLog.addEvent(1);
  assert.equal(master.PodPool.getNumLongTermPods(eventLog, 0, 0, 0.5), 0);
  assert.equal(master.PodPool.getNumLongTermPods(eventLog, 2, 0, 0.5), 2);
  assert.equal(master.PodPool.getNumLongTermPods(eventLog, 2, 1, 0.5), 2);
  assert.equal(master.PodPool.getNumLongTermPods(eventLog, 0, 1, 0.5), 0);

  // Test no events
  assert.equal(master.PodPool.getNumLongTermPods(new master.EventLog(), 0, 0, 0.5), 0);
}

async function testPodPoolGetNumLongTermPodsLastEvent() {
  console.log('Running `testPodPoolGetNumLongTermPodsLastEvent`.');
  let eventLog = new master.EventLog(1250);
  eventLog.addEvent(1);
  await master.sleep(500);
  eventLog.addEvent(2);
  // Ensure that time since the last event is accounted for.
  await master.sleep(1000);
  assert.equal(master.PodPool.getNumLongTermPods(eventLog, 0, 0, 0.5), 2);
}

async function testPodPoolGetNumLongTermPodsFirstEvent() {
  console.log('Running `testPodPoolGetNumLongTermPodsFirstEvent`.');
  let eventLog = new master.EventLog(1250);
  eventLog.addEvent(10);
  await master.sleep(500);
  eventLog.addEvent(2);
  // Without a prior event, the zero pods dominates.
  assert.equal(master.PodPool.getNumLongTermPods(eventLog, 0, 0, 0.5), 0);
}

async function testPodPoolGetNumLongTermPodsOldEvent() {
  console.log('Running `testPodPoolGetNumLongTermPodsOldEvent`.');
  let eventLog = new master.EventLog(250);
  eventLog.addEvent(10);
  await master.sleep(500);
  eventLog.addEvent(2);
  // An event before `maxTime` is considered if it's applicable.
  assert.equal(master.PodPool.getNumLongTermPods(eventLog, 0, 0, 0.5), 10);
}

async function testPodPoolGetNumLongTermPodsPercentile() {
  console.log('Running `testPodPoolGetNumLongTermPodsPercentile`.');
  let eventLog = new master.EventLog(1000);
  eventLog.addEvent(10);
  await master.sleep(50);
  eventLog.addEvent(0);
  // An event before `maxTime` is considered if it's applicable.
  assert.equal(master.PodPool.getNumLongTermPods(eventLog, 0, 0, 0.99), 10);
  assert.equal(master.PodPool.getNumLongTermPods(eventLog, 0, 0, 0.5), 0);
  assert.equal(master.PodPool.getNumLongTermPods(eventLog, 0, 0, 0.0), 0);
}


async function main() {
  await testEventLog();
  await testEventLogMaxTime();
  await testSleep();
  await testRetry();
  await testPod();
  await testPodPoolGetNumShortTermPods();
  await testPodPoolGetNumLongTermPods();
  await testPodPoolGetNumLongTermPodsLastEvent();
  await testPodPoolGetNumLongTermPodsFirstEvent();
  await testPodPoolGetNumLongTermPodsOldEvent();
  await testPodPoolGetNumLongTermPodsPercentile();
}

main();
