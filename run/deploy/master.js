/**
 * The goal of this master is to assign pods to complete work. It is optimized to handle two
 * primary use cases: burst traffic and consistent traffic.
 *
 * NOTE: This file is tested via `node tests/run/deploy/test_master.js`.
 *
 * TODO: Ensure that there are multiple copies of master running, in case of version upgrades.
 * TODO: Switch from `npm` to `yarn` for dependancy management.
 * TODO: Consider using HTTPS protocall between LoadBalancer and the container to protect the
 * sensitive information.
 * TODO: Consider creating a namespace in kubernetes, it's proper practice.
 * TODO: Investigate using preemtible nodes. It'll be more cost effective but it may preempt a
 * stream.
 * TODO: Track the audio hour cost and the price efficiency.
 * TODO: The workers managed by this master run on pods managed by Kubernetes. Those pods are
 *  executed on nodes that run on Google Cloud VMs. The provisioning of the VMs are managed by the
 *  GKE auto-scalar that is not configurable. Learn more:
 *  https://github.com/kubernetes/autoscaler/issues/966.
 *
 *  The GKE autoscalar has a number of inefficiencies that are not easily overcomable. For example,
 *  it may take up to 10 minutes trigger VM deletion and 15 - 60 seconds to trigger node creation.
 *  Learn more:
 *  - https://cloud.google.com/kubernetes-engine/docs/concepts/cluster-autoscaler
 *  - https://github.com/kubernetes/autoscaler/blob/master/cluster-autoscaler/FAQ.md
 *  - https://github.com/kubernetes/autoscaler/blob/master/cluster-autoscaler/proposals/scalability_tests.md
 *
 *  For this reason, it might be useful to not use Kubernetes for running workers. There is a
 *  relatively simple Google Cloud Python API
 *  (https://cloud.google.com/compute/docs/tutorials/python-guide) that can be used to create
 *  and manage instances. For a master node to create an instance, it must have Google Cloud API
 *  permissions set in the node-pool configurations.
 *
 *  Lastly, there might not be a concrete benefit to use Kubernetes for workers.
 * TODO: Consider implementing rate limiting to mitigate DDOS or brute force attacks.
 */
const AbortController = require('abort-controller');
const basicAuth = require('express-basic-auth');
const bodyParser = require('body-parser');
const Client = require('kubernetes-client').Client;
const cors = require('cors');
const express = require('express');
const fetch = require('node-fetch');
const helmet = require('helmet');
const path = require('path');
const Request = require('kubernetes-client/backends/request');
const uuidv4 = require('uuid/v4');

const {
  KubeConfig
} = require('kubernetes-client');

// TODO: The below constants should be parameterized and dynamically updated as the process runs.
// That'll provide a more accurate estimate.
// NOTE: The pod build times is a combination of:
// - ~49 seconds to provision an instance:
//   https://medium.com/google-cloud/understanding-and-profiling-gce-cold-boot-time-32c209fe86ab
// - ~15 seconds for the cluster-autoscalar to trigger provisioning:
//   https://github.com/kubernetes/autoscaler/blob/master/cluster-autoscaler/FAQ.md#what-are-the-service-level-objectives-for-cluster-autoscaler
// - ~30 seconds for the pod to start `gunicorn`, load checkpoints, and load docker container.
// NOTE: Practically, the build time is around 105 seconds.
const AVERAGE_POD_BUILD_TIME = 105 * 1000;
// NOTE: This average assumes 0.4x real time processing for an average clip length.
// These numbers were computed based on March 2021s mixpanel.
const AVERAGE_CHAR_PER_SECOND = 16.50;
const AVERAGE_CLIP_LENGTH_IN_CHAR = 309;
const AVERAGE_CLIP_LENGTH_IN_SEC = AVERAGE_CLIP_LENGTH_IN_CHAR / AVERAGE_CHAR_PER_SECOND;
const AVERAGE_JOB_TIME_IN_MILLI = 0.4 * AVERAGE_CLIP_LENGTH_IN_SEC * 1000;
// NOTE: At minimum, this should a minute long because at minimum GCP charges for a minute
// of usage. Learn more: https://cloud.google.com/compute/vm-instance-pricing
// NOTE: After the minute long, GCP charges for every second of usage.
// NOTE: Unfortunately, due to the auto-scaler, the minimum time a pod is online is 10 minutes.
// Learn more: https://cloud.google.com/compute/docs/autoscaler/understanding-autoscaler-decisions
const MINIMUM_POD_TIME_TO_LIVE = 10 * 60 * 1000;

const IS_PRODUCTION = process.env.NODE_ENV === 'production';

const API_KEYS = Object.entries(process.env).filter(item =>
  item[0].includes(process.env.API_KEY_SUFFIX)
);
const FRONTEND_USERS = Object.assign({}, ...API_KEYS.map(([k, v]) => ({
  [k.toLowerCase()]: v
})));

const logger = {
  log: (...arguments) => console.log(`[${(new Date()).toISOString()}]`, ...arguments),
  warn: (...arguments) => console.warn(`[${(new Date()).toISOString()}]`, ...arguments),
  error: (...arguments) => console.error(`[${(new Date()).toISOString()}]`, ...arguments),
}

/**
 * Get a `Client` in `THIS_POD_NAMESPACE` used to make requests to the Kubernetes API.
 *
 * TODO: Consider caching this client.
 */
async function getClient(namespace = process.env.THIS_POD_NAMESPACE) {
  const kubeconfig = new KubeConfig();
  kubeconfig.loadFromCluster();
  const backend = new Request({
    kubeconfig
  });
  const client = new Client({
    backend
  });
  await client.loadSpec();
  if (namespace) {
    return client.api.v1.namespaces(namespace);
  } else {
    return client.api.v1;
  }
}

/**
 * Time bounded log of events.
 *
 * @param {number} maxTime The maximum time to keep events around. This is defined in
 *    milliseconds.
 */
class EventLog {
  constructor(maxTime = Infinity) {
    this.events = [];
    this.timestamps = [];
    this.maxTime = maxTime;
    this.createdAt = Date.now();
    this.lastEvent = null; // The event before `maxTime`.
  }

  /**
   * Add an event to the event log.
   *
   * @param {*} event
   */
  addEvent(event) {
    this.events.push(event);
    this.timestamps.push(Date.now());
    if (this.maxTime != Infinity) {
      setTimeout(() => {
        this.lastEvent = this.events[0];
        this.events.shift();
        this.timestamps.shift();
      }, this.maxTime);
    }
  }
}


/**
 * Sleep for a number of milliseconds.
 *
 * @param {number} milliseconds Number of milliseconds to sleep.
 * @returns {Promise}
 */
function sleep(milliseconds) {
  return new Promise(resolve => setTimeout(resolve, milliseconds));
}

/**
 * Retries a function mulitple times.
 *
 * @param {Function} toTry Function to retry.
 * @param {number} retries Number of times to retry the function.
 * @param {number} delay Initial delay in milliseconds.
 * @returns {any}
 */
async function retry(toTry, {
  retries = 3,
  delay = 100,
} = {}) {
  for (let i = 0; i < retries; i++) {
    try {
      logger.log(`retry: Attempt #${i}.`);
      const result = await toTry();
      return result;
    } catch (error) {
      logger.warn(`retry: Caught this error: "${error}"`);
      if (i == retries - 1) {
        logger.error(`retry: Reached maximum retries ${retries}, throwing error.`);
        throw error;
      } else {
        await sleep(delay);
      }
    }
  }
}

class Pod {
  /**
   * `Pod` represents a worker with one Kubernetes Pod and Node attached.
   *
   * @param {string} name Name of the pod created.
   * @param {string} nodeName Name of the node created.
   * @param {string} ip IP address of this pod.
   * @param {number} port An exposed port on the pod that is accepting http requests.
   * @param {number} isReadyTTL The time in milliseconds the `isReady` promise is cached.
   */
  constructor(name, nodeName, ip, port, isReadyTTL = 5000) {
    this.name = name;
    this.nodeName = nodeName;
    this.freeSince = Date.now();
    this.createdAt = Date.now();
    this.isReadyCache = {
      contents: null,
      ttl: isReadyTTL,
      createdTime: null,
    }
    this.ip = ip;
    this.port = port;
    this.isDestroyed = false;
  }

  /**
   * Check if this Pod is ready for more requests.
   *
   * @param {string} name The pod name to request.
   * @param {string} host Host to make HTTP request to.
   * @param {number} port Port on host to query.
   * @param {number} timeout Timeout for the HTTP request.
   */
  static async isReady(name, host, port, timeout = 1000) {
    try {
      const abortController = new AbortController();
      setTimeout(() => abortController.abort(), timeout);

      const response = await fetch(`http://${host}:${port}/healthy`, {
        signal: abortController.signal,
        method: 'GET'
      });

      if (!response.ok) {
        const body = await response.text();
        logger.error(`[${name}] Pod.isReady Error ${response.status} ` +
          `"${response.statusText}":\n${body}`);
      } else {
        logger.log(`Pod.isReady: Pod ${name} is ready.`);
      }
      return response.ok;
    } catch (error) {
      logger.error(`[${name}] Pod.isReady Error:`, error);
      return false
    }
  }

  /**
   * Check if this Pod is ready for more requests.
   *
   * @param {boolean} update_cache If `true` `isReady` forcefully updates it's cache.
   */
  async isReady(update_cache = false, timeout = 1000) {
    if (this.isDestroyed) {
      throw new Error(`Pod.isReady Error: Pod ${this.name} has already been destroyed.`);
    }

    logger.log(`Pod.isReady: Checking if Pod ${this.name} is ready to serve requests.`);

    if (update_cache || !this.isReadyCache.createdTime ||
      Date.now() - this.isReadyCache.createdTime > this.isReadyCache.ttl) {
      this.isReadyCache.createdTime = Date.now();
      this.isReadyCache.contents = Pod.isReady(this.name, this.ip, this.port, timeout);
    } else {
      logger.log(`Pod.isReady: Using Pod ${this.name} \`isReadyCache\`.`);
    }

    // NOTE: This could be destroy during `await Pod.isReady`.
    const isReady = await this.isReadyCache.contents && !this.isDestroyed;
    logger.log(`Pod.isReady: Pod ${this.name} is ready - ${isReady}.`);
    return isReady;
  }

  /**
   * Get if `this` is available for work.
   */
  async isAvailable() {
    if (this.isDestroyed) {
      throw new Error(`Pod.isAvailable Error: Pod ${this.name} has already been destroyed.`);
    }

    // If `this` is reserved then this does not make a request for readiness due to the synchronous
    // implementation of the worker pods.
    if (this.isReserved()) {
      return false;
    }


    // NOTE: Recompute `this.isReserved` after `this.isReady` to ensure it's still not reserved.
    const isReady = await this.isReady();
    return isReady && !this.isReserved();
  }

  /**
   * Get if `this` is reserved and is unable to do more work.
   */
  isReserved() {
    if (this.isDestroyed) {
      throw new Error(`Pod.isReserved Error: Pod ${this.name} has already been destroyed.`);
    }

    return this.freeSince === undefined;
  }

  /**
   * Reserve `this` for a job.
   *
   * TODO: Consider implementing a leak pervention mechanism that cleans up pods reserved for more
   * than an hour. This can be implemented with `isReady` due to synchronous nature of the workers;
   * However, we'd need to timeout / abort mechanism. A pod that `isReady` is no longer occupied
   * with work, hence it is not reserved.
   */
  reserve() {
    if (this.isDestroyed) {
      throw new Error(`Pod.reserve Error: Pod ${this.name} has already been destroyed.`);
    }

    if (this.isReserved()) {
      throw new Error(
        `Pod.reserve Error: Pod ${this.name} is reserved, it cannot be reserved again.`);
    }

    logger.log(`Reserving pod ${this.name}.`);
    this.freeSince = undefined;
    logger.log(`Reserved pod ${this.name}.`);
    return this;
  }

  /**
   * Release `this` Pod from the job.
   */
  release() {
    logger.log(`Releasing pod ${this.name}.`);
    this.freeSince = Date.now();
    logger.log(`Released pod ${this.name}.`);
    return this;
  }

  /**
   * Return `true` if the `this` is no longer suitable for work.
   */
  async isDead() {
    // If `this` is reserved then this does not make a request for readiness due to the synchronous
    // implementation of the worker pods.
    // TODO: Retry a couple times before preemptively killing a pod.
    if (this.isDestroyed) {
      return true;
    }

    // TODO: Reserved `Pod`s are sometimes destroyed, investigate that.
    // NOTE: Only declare `isDead` iff `this` fails `isReady` check multiple times with a
    // large enough timeout.
    // NOTE: In November 2019, it was observed it could take up to 15 seconds to load the
    // checkpoints. This timeout was set to be double the expected response time, just in case.
    let isReady = await this.isReady(true, 30000);
    isReady = isReady || (!this.isDestroyed && await this.isReady(true, 30000));
    if (!isReady && !this.isDestroyed && this.isReserved()) {
      logger.warn(`Pod.destroy: Reserved Pod ${this.name} is already dead.`);
    }

    // NOTE: Ensure that after `await this.isReady` is hasn't died.
    return this.isDestroyed || !isReady;
  }

  static async destroy(podName, nodeName) {
    logger.log(`Pod.destroy: Deleting Pod ${podName} and Node ${nodeName}.`);

    try {
      await (await getClient()).pods(podName).delete();
      logger.log(`Pod.destroy: Deleted Pod ${podName}.`);
    } catch (error) { // TODO: Handle this scenario, try avoiding leaking a pod.
      logger.log(`Pod.destroy: Failed to delete Pod ${podName} due to error: ${error}`);
    }

    try {
      await (await getClient(null)).nodes(nodeName).delete();
      logger.log(`Pod.destroy: Deleted Node ${nodeName}.`);
    } catch (error) { // TODO: Handle this scenario, try avoiding leaking a node.
      logger.log(`Pod.destroy: Failed to delete Node ${nodeName} due to error: ${error}`);
    }
  }

  /**
   * Destroy `this`.
   */
  async destroy() {
    if (this.isDestroyed) {
      throw new Error(`Pod.destory Error: Pod ${this.name} has already been destroyed.`);
    }

    if (this.isReserved()) {
      throw new Error(`Pod.destroy Error: Pod ${this.name} is reserved, it cannot be destroyed.`);
    }

    logger.log(`Pod.destroy: Deleting Pod ${this.name}.`);
    this.isDestroyed = true;
    await Pod.destroy(this.name, this.nodeName);
  }

  /**
   * Create a `Pod` running an image, ready to recieve requests.
   *
   * NOTE: After `statusRetries` it's presumed the pod may no longer be relevant.
   * TODO: Instead of estimating pod usefulness via `statusRetries` instead there should be a
   * mechanism aborting pod builds once the work load decreases.
   * TODO: `statusLoop` and `statusRetries` default parameters should not be coupled via a
   * constant number.
   *
   * @param {string} podImage The Kubernetes image for the pod.
   * @param {int} statusRetries Number of status checks before giving up on `Pod` creation.
   * @param {number} statusLoop Length in milliseconds between pod status checks.
   * @throws Error if Pod is unavailable for work, after querying status `statusRetries` times.
   * @returns {Pod} Returns a `Pod`.
   */
  static async build(podImage, {
    statusRetries = Math.ceil((AVERAGE_POD_BUILD_TIME + MINIMUM_POD_TIME_TO_LIVE) / 2000),
    statusLoop = 2000,
  } = {}) {
    const name = `${process.env.WORKER_POD_PREFIX}-${uuidv4()}`;
    logger.log(`Pod.build: Creating pod named ${name}.`);

    // Included in the Pod environment, a list of API Keys defined by `API_KEY_SUFFIX`.
    let apiKeys = Object.entries(process.env);
    apiKeys = apiKeys.filter((item) => item[0].includes(process.env.API_KEY_SUFFIX));
    apiKeys = apiKeys.map(item => ({
      'name': item[0],
      'value': item[1],
    }));

    const podPort = parseInt(process.env.WORKER_POD_EXPOSED_PORT, 10);

    // Learn more about `body.spec.ownerReferences[].controller` key:
    // https://stackoverflow.com/questions/51068026/when-exactly-do-i-set-an-ownerreferences-controller-field-to-true
    // Note, kubernetes will ignore keys that are not proper sometimes.
    let info = await (await getClient()).pods.post({
      'body': {
        'apiVersion': 'v1',
        'kind': 'Pod',
        'metadata': {
          'name': name,
          'labels': {
            'run': process.env.WORKER_POD_PREFIX,
          },
          // NOTE: Once this Pod is delegated as an owner of a kubernetes resource it enables those
          // resources to be garbage collected if this pod dies.
          'ownerReferences': [{
            'apiVersion': 'v1',
            'blockOwnerDeletion': true,
            'controller': true,
            'kind': 'Pod',
            'name': process.env.THIS_POD_NAME,
            'uid': process.env.THIS_POD_UID,
          }],
        },
        'spec': {
          'restartPolicy': 'Never',
          'containers': [{
            'image': podImage,
            'name': process.env.WORKER_POD_PREFIX,
            'env': apiKeys,
            'resources': {
              'requests': {
                // NOTE: This is smaller than required purposefully to give room for any other
                // system pods.
                'memory': '4Gi',
                'cpu': '7250m'
              }
            },
            'ports': [{
              'containerPort': podPort,
              'protocol': 'TCP'
            }]
          }],
          'terminationGracePeriodSeconds': 600,
        }
      }
    });
    logger.log(`Pod.build: Pod ${name} manifest sent.`);

    try {
      await retry(async () => { // While Pod is not running or not ready, keep retrying.
        /**
         * Check if `Pod` has "been bound to a node, and all of the Containers have been created.";
         * otherwise, throw an error.
         */
        info = await (await getClient()).pods(name).get();
        // Learn more about pod status:
        // https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/
        if (info.body.status.phase == 'Running') {
          if (!(await Pod.isReady(name, info.body.status.podIP, podPort))) {
            throw new Error('Pod.build Error: Pod is not ready to recieve work.');
          }
        } else {
          throw new Error(
            `Pod.build Error: Not running, recieved:\n${JSON.stringify(info.body.status)}`);
        }
      }, {
        retries: statusRetries,
        delay: statusLoop,
      });
    } catch (error) {
      Pod.destroy(name, info.body.spec.nodeName);
      return null;
    }

    logger.log(`Pod.build: Pod ${name} is ready to recieve traffic at ${info.body.status.podIP}`);
    return new Pod(name, info.body.spec.nodeName, info.body.status.podIP, podPort);
  }
}

class PodPool {
  /**
   * This manages the scaling and reservation of pods for work.
   *
   * @param {string} podImage The Kubernetes Pod image to autoscale.
   * @param {number} scaleLoop This is the maximum time between recomputing the number of pods. This
   *   ensures that pods are scaled down quickly if pods are no longer in-use.
   * @param {number} scalingWindow This is the time we look into the past to compute the number of
   *   resources needed for the next time period.
   * @param {number} minPods The minimum worker pods to keep online always.
   */
  constructor(
    podImage,
    minPods = parseInt(process.env.MINIMUM_WORKER_PODS, 10),
    scaleLoop = parseInt(process.env.AUTOSCALE_LOOP, 10),
    scalingWindow = parseInt(process.env.AUTOSCALE_WINDOW, 10),
  ) {
    this.pods = [];
    this.numPodsBuilding = 0;
    this.podRequests = [];
    this.reservationLog = new EventLog(scalingWindow);
    this.waiting = [];
    this.podImage = podImage;
    this.minPods = minPods;

    // Learn more about `arguments` in a `class`:
    // https://stackoverflow.com/questions/48519484/uncaught-syntaxerror-unexpected-eval-or-arguments-in-strict-mode-window-gtag?rq=1
    this.logger = {
      log: (...args) => logger.log(`[${podImage}]`, ...args),
      warn: (...args) => logger.warn(`[${podImage}]`, ...args),
      error: (...args) => logger.error(`[${podImage}]`, ...args),
    }

    this.addExistingPods();
    this.loop(scaleLoop);
  }

  /**
   * On restart of this `Pod`, this retrieves any existing (running) worker's pods and adds them back to the
   * pool.
   */
  async addExistingPods() {
    let existingPods = (await (await getClient()).pods.get()).body.items;
    existingPods = existingPods.filter(pod =>
      pod.metadata.ownerReferences[0].uid == process.env.THIS_POD_UID &&
      pod.spec.containers[0].image == this.podImage);
    this.logger.log(`PodPool.addExistingPods: Found ${existingPods.length} existing pods.`);
    for (const existingPod of existingPods) {
      const podName = existingPod.metadata.name;
      const nodeName = existingPod.spec.nodeName;
      if (existingPod.status.phase == 'Running') {
        this.logger.log(`PodPool.addExistingPods: Adding existing Pod ${podName}.`);
        this.pods.push(
          new Pod(
            podName,
            nodeName,
            existingPod.status.podIP,
            existingPod.spec.containers[0].ports[0].containerPort));
      } else {
        Pod.destroy(podName, nodeName);
      }
    }
    this.clean();
  }

  /**
   * Start a loop to scale `this`.
   *
   * Typically, `this.scale` is triggered by activity like reserving or releasing pods. `this.loop`
   * covers the case where `PodPool` is not getting any activity and needs to be downscaled.
   *
   * @param {number} delay The number of milliseconds between loops.
   */
  loop(delay) {
    this.loopTimeout = setTimeout(() => {
      this.scale();
      this.loop(delay);
    }, delay);
  }

  /**
   * Wait until there are Pods ready.
   */
  waitTillReady() {
    return new Promise(async (resolve) => {
      this.logger.log(`PodPool.waitTillReady: Waiting till ready.`);
      this.waiting.push(resolve);
      this.fulfillPodRequests(); // Try to resolve right away
    });
  }

  /**
   * Take note of the number of reserved pods in `this.pods` after it's been changed.
   */
  logNumReservedPods() {
    const numReservedPods = this.pods.filter(p => p.isReserved()).length;
    this.logger.log(`PodPool.logNumReservedPods: There are ${numReservedPods} reserved pods.`);
    this.reservationLog.addEvent(numReservedPods);
  }

  /**
   * Check if `pod` is ready, and queue up `cleanPod` if it's not.
   */
  async isPodReady(pod, update_cache) {
    const isReady = await pod.isReady(update_cache);
    if (!isReady) {
      this.cleanPod(pod);
    }
    return isReady;
  }

  /**
   * Fulfill the next pod reservation request in `this.podRequests` and `this.waiting`.
   */
  async fulfillPodRequests() {
    // NOTE: This assumes that there are Pods ready if `this.pods.length > 0`.
    if (this.waiting.length > 0 && this.pods.length > 0) {
      this.waiting.map(resolve => resolve());
      this.waiting = [];
    }

    while (this.podRequests.length > 0 && this.pods.length > 0) {
      this.logger.log(`PodPool.fulfillPodRequests: Attempting to fulfill pod request.`);

      // NOTE: Always re-sort and re-filter because `pod.isReady()` can take some time.
      const available = this.pods.filter(p => !p.isReserved());
      if (available.length == 0) {
        break;
      }

      // NOTE: Sort most recently used first, allowing unused pods to stay unused.
      const pod = available.sort((a, b) => b.freeSince - a.freeSince)[0];
      // NOTE: Reserve preemptively before `await` during which `javascript` could run another
      // thread that'll reserve it.
      pod.reserve();
      if (await this.isPodReady(pod, true) && this.podRequests.length > 0) {
        this.podRequests.shift()[1](pod);
        this.logNumReservedPods();
      } else { // CASE: `pod` is dead.
        pod.release();
        this.logNumReservedPods();
      }
    }
  }

  /**
   * Reserve a pod for work.
   *
   * @param {function(Pod)} reservedPodCallback Callback called with a reserved Pod available for
   *  work.
   * @param {function} cancelPodRequest Callback this function to cancel the reservation.
   */
  reservePod(reservedPodCallback) {
    this.logger.log(`PodPool.reservePod: Requesting pod for work.`);
    const callback = (pod) => {
      this.logger.log(`PodPool.reservePod: Reserved ${pod.name} for work.`);
      return reservedPodCallback(pod);
    };
    this.podRequests.push([Date.now(), callback]);
    this.fulfillPodRequests(); // Attempt to immediately fulfill the pod request.
    return () => {
      if (this.podRequests.map(r => r[1]).includes(callback)) {
        this.logger.log(`PodPool.reservePod: Canceling one Pod request from ` +
          `${this.podRequests.length} Pod request(s).`);
        this.podRequests = this.podRequests.filter(r => r[1] !== callback);
        this.logger.log(`PodPool.reservePod: There are ${this.podRequests.length} Pod request(s).`);
      }
    }
  }

  /**
   * Release pod to the pool.
   *
   * @param {Pod} pod The pod to release.
   */
  releasePod(pod) {
    pod.release();
    this.logger.log(`PodPool.releasePod: Released ${pod.name}.`);
    this.logNumReservedPods();
    this.fulfillPodRequests();
  }

  /**
   * Recompute the number of pods needed and scale.
   *
   * This method should be called whenever the results might be affected.
   */
  async scale() {
    this.logger.log(`PodPool.scale: Removing dead pod(s).`);
    await this.clean();
    this.logger.log(`PodPool.scale: There are ${this.pods.length} pod(s).`);
    this.logger.log(`PodPool.scale: There are ${this.numPodsBuilding} pod(s) building.`);
    this.logger.log(`PodPool.scale: There are ${this.waiting.length} request(s) waiting.`);
    this.logger.log(`PodPool.scale: There are ${this.podRequests.length} job(s) outstanding.`);
    const shortTermNumPodsDesired = PodPool.getNumShortTermPods(
      this.podRequests.length, this.podRequests.map(r => r[0]));
    this.logger.log(`PodPool.scale: This desires short term ${shortTermNumPodsDesired} pod(s).`);
    const longTermNumPodsDesired = PodPool.getNumLongTermPods(this.reservationLog, this.minPods);
    this.logger.log(`PodPool.scale: This desires long term ${longTermNumPodsDesired} pod(s).`);
    const numPodsDesired = shortTermNumPodsDesired + longTermNumPodsDesired;
    this.logger.log(`PodPool.scale: This desires ${numPodsDesired} pod(s).`);
    if (numPodsDesired > this.pods.length + this.numPodsBuilding) {
      await this.upscale(numPodsDesired);
    } else if (numPodsDesired < this.pods.length) {
      await this.downscale(numPodsDesired);
    }
  }

  /**
   * Downscale up the number of pods in `this`.
   *
   * @param {number} numPods
   */
  async downscale(numPods) {
    this.logger.log(`PodPool.downscale: Attempting to downscale to ${numPods} pods.`);
    let toDestory = this.pods.filter(p => !p.isReserved());

    // Due to GCP pricing, it does not make sense to destroy pods prior to
    // `MINIMUM_POD_TIME_TO_LIVE`.
    toDestory = toDestory.filter(p => Date.now() - p.createdAt >= MINIMUM_POD_TIME_TO_LIVE);
    this.logger.log(`PodPool.downscale: ${toDestory.length} pods are eligable for destruction.`);

    // TODO: Consider sorting by creation time to benefit from Google's long usage discounts.
    toDestory = toDestory.sort((a, b) => a.freeSince - b.freeSince); // Sort by least recently used

    // Select pods to destroy
    let numPodsToDestory = Math.min(this.pods.length - numPods, toDestory.length);
    toDestory = toDestory.slice(0, numPodsToDestory);
    this.logger.log(`PodPool.downscale: Destorying ${toDestory.length} pods.`);

    // Destroy pods
    this.pods = this.pods.filter(p => !toDestory.includes(p));
    return Promise.all(toDestory.map(p => p.destroy()));
  }

  /**
   * Scale up the number of pods in `this`.
   *
   * @param {number} numPods
   * @param {number} maximumPods The maximum number of pods this can scale to.
   */
  async upscale(numPods, maximumPods = parseInt(process.env.MAXIMUM_PODS, 10)) {
    const numPodsToCreate = Math.min(maximumPods, numPods) -
      this.pods.length - this.numPodsBuilding;
    this.logger.log(`PodPool.upscale: Creating ${numPodsToCreate} pods.`);
    this.numPodsBuilding += numPodsToCreate;
    return Promise.all(Array.from({
      length: numPodsToCreate
    }, async () => {
      try {
        const pod = await Pod.build(this.podImage);
        if (pod != null) {
          this.logger.log(`PodPool.upscale: Adding Pod of type ` +
            `${Object.prototype.toString.call(pod)}`);
          this.pods.push(pod);
          this.fulfillPodRequests();
        }
      } catch (error) {
        this.logger.warn(`PodPool.upscale Error: Pod build failed: ${error}`);
      }
      this.numPodsBuilding -= 1;
    }));
  }

  /**
   * Attempt to remove `pod`.
   */
  async cleanPod(pod) {
    this.logger.log(`PodPool.clean: Maybe cleaning up Pod ${pod.name}.`);
    // NOTE: The pod could be reserved after `isDead` is evaluated but before it returns.
    // Run this example to return more:
    /**
    let bool = true;

    async function func() {
      console.log('func will return', bool)
      return bool;
    }

    async function main() {
      console.log('func returned', await func());
      console.log('actual value is', bool);
    }

    main()
    bool = false;
    */
    if (await pod.isDead() && this.pods.includes(pod)) {
      this.logger.log(`PodPool.clean: Cleaning up Pod ${pod.name}.`);
      if (pod.isReserved()) {
        this.logger.warn(`Pod.clean: Releasing Pod ${pod.name} before destroying.`);
        pod.release();
        this.logNumReservedPods();
      }
      this.pods = this.pods.filter(p => p !== pod);
      pod.destroy();
    }
  }

  /**
   * Remove any dead pods.
   */
  clean() {
    return Promise.all(this.pods.map(async (pod) => this.cleanPod(pod)));
  }

  /**
   * Get the number of pods that would have been needed to complete the outstanding jobs.
   *
   * @param {number} numJobsOutstanding The number of jobs that need to be completed.
   * @param {list} jobsTimes The time each job was created.
   * @param {number} averageJobTimeInMilli
   * @param {number} maxShortTermPods The maximum number of short term pods for
   *    getNumShortTermPods`.
   */
  static getNumShortTermPods(
    numJobsOutstanding,
    jobsTimes,
    averageJobTimeInMilli = AVERAGE_JOB_TIME_IN_MILLI,
    maxShortTermPods = 32,
  ) {
    if (jobsTimes.length != numJobsOutstanding) {
      throw new Error(`Pod.getNumShortTermPods Error: The number of jobs is ambigious.`);
    }

    if (numJobsOutstanding == 0) {
      return 0;
    }

    const elapsedTimeInMilli = Date.now() - Math.min(...jobsTimes);
    const totalOutstandingWorkInMilli = averageJobTimeInMilli * numJobsOutstanding;
    const podsDesired = Math.ceil(totalOutstandingWorkInMilli / elapsedTimeInMilli);
    return Math.min(podsDesired, numJobsOutstanding, maxShortTermPods);
  }

  /**
   * Get the number of pods consistently used over a period of time.
   *
   * Note this computes the weighted median of the number of pods reserved in the `reservationLog`.
   * This means that we'll have enough resources to fulfill customer needs immediately most of the
   * time as a baseline.
   *
   * @param {EventLog} reservationLog  A log of the number of reservations over a period of time.
   * @param {number} minPods The minimum worker pods to keep online always.
   * @param {number} extraPods The number of extra pods to keep online for any spill over.
   * @param {number} percentile The percentile median is computed at.
   * @returns {number} Get the number of pods to keep online.
   */
  static getNumLongTermPods(
    reservationLog,
    minPods,
    extraPods = parseInt(process.env.EXTRA_WORKER_PODS),
    percentile = parseFloat(process.env.COVERAGE),
  ) {
    const numEvents = reservationLog.events.length;
    logger.log(`PodPool.getNumLongTermPods: There are ${numEvents} events in \`reservationLog\`.`);
    if (numEvents == 0) {
      return minPods;
    }
    const now = Date.now();
    const timestamps = reservationLog.timestamps;

    const mapTimeCounter = new Map();
    const timeBeforeFirstEvent = reservationLog.maxTime - (now - timestamps[0]);
    mapTimeCounter.set(reservationLog.lastEvent == null ? 0 : reservationLog.lastEvent,
      timeBeforeFirstEvent);
    for (let i = 0; i < numEvents; i++) {
      const numReserved = reservationLog.events[i];
      const isLastEvent = i == numEvents - 1;
      const timeReserved = isLastEvent ? now - timestamps[i] : timestamps[i + 1] - timestamps[i];
      mapTimeCounter.set(numReserved, timeReserved +
        (mapTimeCounter.has(numReserved) ? mapTimeCounter.get(numReserved) : 0));
    }
    logger.log('PodPool.getNumLongTermPods: The distribution of reservations are in milliseconds:');
    logger.log(mapTimeCounter);

    const sorted = [...mapTimeCounter.entries()].sort((a, b) => a[0] - b[0]);
    const totalTime = sorted.map(s => s[1]).reduce((a, b) => a + b, 0);
    let timeCounter = 0;
    for (const [numReserved, time] of sorted) {
      timeCounter += time;
      if (timeCounter >= totalTime * percentile) {
        if (numReserved > 0 && numReserved >= minPods) {
          return numReserved + extraPods;
        } else {
          return minPods;
        }
      }
    }
  }
}

/**
 * Call `func` on the close of `request` or `response`.
 *
 * NOTE: `func` may be called multiple times; therefore, it must be idempotent.
 *
 * @param {http.IncomingMessage} request
 * @param {http.ServerResponse} response
 * @param {function} func
 */
function onClose(request, response, func) {
  // Handle canceled request
  // https://stackoverflow.com/questions/35198208/handling-cancelled-request-with-express-node-js-and-angular
  // NOTE: `http.IncomingMessage`: Handle events when the request has been aborted or the underlying
  // connection was closed.
  const requestPrefix = '[request (http.IncomingMessage)] ';
  request
    .on('aborted', () => func(`${requestPrefix}Emitted 'aborted' event.`, null))
    .on('close', () => func(`${requestPrefix}Emitted 'close' event.`, null))
    .on('error', error => func(`${requestPrefix}Emitted 'error' (${error}) event.`, error));

  // NOTE: `http.ServerResponse`: Handle events when the response has been sent or the underlying
  // connection was closed.
  const responsePrefix = '[response (http.ServerResponse)] ';
  response
    .on('close', () => func(`${responsePrefix}Emitted 'close' event.`, null))
    .on('finish', () => func(`${responsePrefix}Emitted 'finish' event.`, null))
    .on('error', error => func(`${responsePrefix}Emitted 'error' (${error}) event.`, error));
}

/**
 * Send `request` to `pod` and pass back the response to `response`.
 *
 * @param {string} prefix A string prefix printed with every log.
 * @param {PodPool} podPool
 * @param {Pod} pod The Pod to proxy the request to.
 * @param {express.Request} request
 * @param {express.Response} response
 */
function proxyRequestToPod(prefix, podPool, pod, request, response) {
  return new Promise(async (resolve, reject) => {
    const ttsAbortController = new AbortController();
    const handleClose = (message, error) => {
      logger.log(`${prefix}${message}`);
      ttsAbortController.abort();
      // NOTE: Node ignores any `resolve` or `reject` calls after the first call, learn more:
      // http://jsbin.com/gemepay/3/edit?js,console
      if (error == null) {
        resolve();
      } else {
        if (!pod.isDestroyed) {
          podPool.isPodReady(pod, true);
        }
        reject(error);
      }
    };
    onClose(request, response, handleClose);
    try {
      // WARNING: Do not log request body because it contains sensitive user information.
      const ttsEndPoint = `http://${pod.ip}:${pod.port}${request.url}`;
      logger.log(`${prefix}Sending request with to ${ttsEndPoint} headers:\n` +
        JSON.stringify(request.headers));
      const ttsResponse = await fetch(ttsEndPoint, {
        signal: ttsAbortController.signal,
        method: request.method,
        headers: request.headers,
        // Learn more:
        // https://stackoverflow.com/questions/47892127/succinct-concise-syntax-for-optional-object-keys-in-es6-es7
        // Learn more:
        // https://github.com/whatwg/fetch/issues/551
        ...(!['head', 'get'].includes(request.method.toLowerCase()) && {
          body: JSON.stringify(request.body)
        }),
      });

      // Stream response back
      response.writeHead(ttsResponse.status, ttsResponse.headers.raw());
      ttsResponse.body
        .on('data', chunk => response.write(chunk))
        .on('end', () => {
          logger.log(`${prefix}[ttsResponse.body] Emitted 'close' event.`);
          response.end();
        })
        .on('error', (error) =>
          handleClose(`[ttsResponse.body] Emitted 'error' (${error}) event.`, error));
    } catch (error) {
      handleClose(`[proxyRequestToPod] Caught error (${error}).`, error);
    }
  });
}

/**
 * Get a Pod Pool to serve requests based on the `Accept-Version` header.
 *
 * @param {express.Request} request
 * @param {express.Response} response
 */
function getPodPool(request, response) {
  const version = request.get('Accept-Version');
  logger.log(`getPodPool: Getting Pod Pool version ${version}.`);
  if (version && request.app.locals.podPools[version.toLowerCase()]) {
    return request.app.locals.podPools[version.toLowerCase()];
  } else if (version) {
    logger.log(`getPodPool: Version not found: ${version}.`);
    response.status(404);
    response.json({
      'code': 'NOT_FOUND',
      'message': 'Version not found.',
    });
    return;
  }
  return request.app.locals.podPools.latest;
}

/**
 * Reserve a Pod and proxy a request to the Pod.
 *
 * NOTE: `reservePodController` never sends parallel requests to the same Pod.
 *
 * @param {express.Request} request
 * @param {express.Response} response
 * @param {function} next
 */
function reservePodController(request, response, next) {
  let prefix = `reservePodController: `;
  logger.log(`${prefix}Got request.`);

  const podPool = getPodPool(request, response);
  if (response.headersSent) {
    return;
  }

  const cancelReservation = podPool.reservePod(async (pod) => {
    prefix = `reservePodController [${pod.name}]: `;
    try {
      await proxyRequestToPod(prefix, podPool, pod, request, response);
    } catch (error) {
      next(error);
    }
    logger.log(`${prefix}Releasing \`Pod\`.`);
    podPool.releasePod(pod);
  });

  onClose(request, response, (event, _) => {
    logger.log(`${prefix}${event}`);
    cancelReservation();
  });
}

const noReservationController = (() => {
  let podIndex = 0;

  /**
   * Return a Pod to query with Pod cycling.
   *
   * @returns {Pod}
   */
  async function getPod(podPool) {
    await podPool.waitTillReady();
    podIndex %= podPool.pods.length;
    logger.log(`noReservationController: Getting Pod ${podIndex} of ${podPool.pods.length}.`);
    const pod = podPool.pods[podIndex];
    logger.log(`noReservationController: Got Pod of type ${Object.prototype.toString.call(pod)}.`);
    podIndex += 1;
    logger.log(`noReservationController: Got Pod named ${pod.name}.`);
    if (await podPool.isPodReady(pod, true)) {
      return pod;
    } else {
      return getPod(podPool);
    }
  }

  /**
   * Proxy requests to a Pod regardless of it's reservation status.
   *
   * @param {express.Request} request
   * @param {express.Response} response
   * @param {function} next
   */
  return async (request, response, next) => {
    logger.log(`noReservationController: Got request.`);
    const podPool = getPodPool(request, response);
    if (response.headersSent) {
      return;
    }
    const pod = await getPod(podPool);
    try {
      await proxyRequestToPod(`noReservationController [${pod.name}]: `, podPool, pod, request, response);
    } catch (error) {
      next(error);
    }
  };
})();

const app = express();

app.use(bodyParser.json());
// NOTE: Recommened by:
// https://blog.risingstack.com/node-js-security-checklist/
// https://expressjs.com/en/advanced/best-practice-security.html
app.use(helmet());


// TODO: Setup more restrictive CORS options for security.
// NOTE: The reason CORS isn't setup is because it's nontrivial to white list the auto
// generated external IP address for our Kubernetes service.
// NOTE: Recommended by:
// https://expressjs.com/en/resources/middleware/cors.html
app.options('*', cors());
app.use(cors());

// NOTE: The reason we're not waiting for the `PodPool.waitTillReady` is because that would cause
// our resources to double with potentially two healthy masters during a transition.
app.get('/healthy', (_, response) => {
  response.send('ok');
});

// TODO: Consider remove `/api/*` so it is not redundant with the subdomain `api`.
app.all('/api/text_to_speech/stream', reservePodController);
app.all('/api/speech_synthesis/v1/text_to_speech/stream', reservePodController);
app.all('/api/*', noReservationController);

app.use(basicAuth({
  users: FRONTEND_USERS,
  challenge: true,
}));
app.use(express.static(path.join(__dirname, 'public')));
app.get('/', (_, response) => {
  response.sendFile('public/index.html', {
    root: __dirname
  });
});

// Catch-all error handler similar to:
// https://expressjs.com/en/guide/error-handling.html
// NOTE: Express requires all four parameters despite the `next` parameter being unused:
// https://stackoverflow.com/questions/51826711/express-js-error-handler-doesnt-work-if-i-remove-next-parameter
app.use((error, request, response, next) => {
  logger.error('Catch-all error handler:', error);

  if (response.headersSent) {
    next(error);
    return;
  }

  response.status(500);
  response.json({
    'code': 'INTERNAL_ERROR',
    'message': IS_PRODUCTION ? 'Internal error, please try again.' : error,
  });
});

if (require.main === module) {
  app.locals.podPools = {
    v1: new PodPool(process.env.V1_WORKER_POD_IMAGE, 0),
    v2: new PodPool(process.env.V2_WORKER_POD_IMAGE, 0),
    v3: new PodPool(process.env.V3_WORKER_POD_IMAGE, 0),
    v4: new PodPool(process.env.V4_WORKER_POD_IMAGE, 0),
    "lincoln.v1": new PodPool(process.env.LINCOLN_V1_WORKER_POD_IMAGE, 0),
    v5: new PodPool(process.env.V5_WORKER_POD_IMAGE, 0),
    v6: new PodPool(process.env.V6_WORKER_POD_IMAGE, 0),
    v7: new PodPool(process.env.V7_WORKER_POD_IMAGE, 0),
    v8: new PodPool(process.env.V8_WORKER_POD_IMAGE, 1),
    "uneeq.v1": new PodPool(process.env.UNEEQ_V1_WORKER_POD_IMAGE, 1),
    "v8.1": new PodPool(process.env.V8_1_WORKER_POD_IMAGE, 1),
    "veritone.v1": new PodPool(process.env.VERITONE_V1_WORKER_POD_IMAGE, 1),
    "super-hi-fi.v1": new PodPool(process.env.SUPER_HI_FI_V1_WORKER_POD_IMAGE, 1),
    v9: new PodPool(process.env.V9_WORKER_POD_IMAGE, 4),
  };
  app.locals.podPools.latest = app.locals.podPools.v9;

  const listener = app.listen(8000, '0.0.0.0', () => logger.log(`Listening on port ${8000}!`));

  function closeHTTPServer() {
    return new Promise(resolve => {
      listener.close(() => {
        console.log('HTTP server closed.');
        resolve();
      });
    });
  }

  // Shutdown `server` gracefully.
  // Learn more: https://expressjs.com/en/advanced/healthcheck-graceful-shutdown.html
  // Learn more: https://hackernoon.com/graceful-shutdown-in-nodejs-2f8f59d1c357
  process.on('SIGTERM', async () => {
    console.log('SIGTERM signal received, shutting down.');
    await closeHTTPServer();
    console.log('Exiting the process.');
    process.exit(0);
  });

  process.on('unhandledRejection', async (error) => {
    logger.error(`Caught unhandledRejection:`, error);
    await closeHTTPServer();
    console.log('Exiting the process.');
    process.exit(1);
  });
} else {
  module.exports = {
    EventLog,
    Pod,
    PodPool,
    sleep,
    retry,
  };
}
