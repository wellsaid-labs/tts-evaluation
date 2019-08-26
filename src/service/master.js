/**
 * The goal of this master is to assign pods to complete work. It is optimized to handle two
 * primary use cases: burst traffic and consistent traffic.
 *
 * TODO: Ensure that there are multiple copies of master running, in case of version upgrades.
 * TODO: Switch from `npm` to `yarn` for dependancy management.
 * TODO: Consider using HTTPS protocall between LoadBalancer and the container so that the API
 * Key is not sniffable?
 * TODO: Consider creating a namespace in kubernetes, it's proper practice.
 * TODO: Put a cache in front of the API frontend, ensuring it does not fail under a DDOS
 * attack.
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
 */
const AbortController = require('abort-controller');
const bodyParser = require('body-parser');
const Client = require('kubernetes-client').Client
const config = require('kubernetes-client').config
const express = require('express');
const fetch = require('node-fetch');
const uuidv4 = require('uuid/v4');

const APP = express();

// TODO: The below constants should be parameterized and dynamically updated as the process runs.
// That'll provide a more accurate estimate.
// NOTE: The pod build times is a combination of:
// - ~49 seconds to provision an instance:
//   https://medium.com/google-cloud/understanding-and-profiling-gce-cold-boot-time-32c209fe86ab
// - ~15 seconds for the cluster-autoscalar to trigger provisioning:
//   https://github.com/kubernetes/autoscaler/blob/master/cluster-autoscaler/FAQ.md#what-are-the-service-level-objectives-for-cluster-autoscaler
// - ~10 seconds for the pod to start `gunicorn`, load checkpoints, and load docker container.
// NOTE: Practically, the build time is around 85 seconds.
const AVERAGE_POD_BUILD_TIME = 85 * 1000;
// NOTE: This average assumes 5.5x real time processing for an average clip length of 7.147 seconds.
// These numbers were computed based on the first ~5500 clips generated on our website.
const AVERAGE_JOB_TIME = 5.5 * 7.147 * 1000;
// NOTE: At minimum, this should a minute long because at minimum GCP charges for a minute
// of usage. Learn more: https://cloud.google.com/compute/vm-instance-pricing
// NOTE: After the minute long, GCP charges for every second of usage.
// NOTE: Unfortunately, due to the auto-scaler, the minimum time a pod is online is 10 minutes.
// Learn more: https://cloud.google.com/compute/docs/autoscaler/understanding-autoscaler-decisions
const MINIMUM_POD_TIME_TO_LIVE = 10 * 60 * 1000;

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
  const client = new Client({
    config: config.getInCluster(),
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

/**
 * `Array.filter` with `async` support.
 *
 * Given node can change threads during `async` operations, the filter may not be accurate when
 * this function returns.
 *
 * @param {iterator} iterator Iterator to run filter on.
 * @param {Function} func Async function used to filter the iterator.
 * @returns {iterator} The filtered iterator.
 */
async function asyncFilter(iterator, func) {
  const promises = await Promise.all(iterator.map(element => func(element)));
  return iterator.filter((_, i) => promises[i]);
}


class Pod {
  /**
   * `Pod` represents a worker with one Kubernetes Pod and Node attached.
   *
   * @param {string} name Name of the pod created.
   * @param {string} nodeName Name of the node created.
   * @param {string} ip IP address of this pod.
   * @param {number} port An exposed port on the pod that is accepting http requests.
   */
  constructor(name, nodeName, ip, port) {
    this.name = name;
    this.nodeName = nodeName;
    this.freeSince = Date.now();
    this.createdAt = Date.now();
    this.ip = ip;
    this.port = port;
    this.isDestroyed = false;
  }

  /**
   * Check if this Pod is ready for more requests.
   *
   * @param {string} host Host to make HTTP request to.
   * @param {number} port Port on host to query.
   * @param {number} timeout Timeout for the HTTP request.
   */
  static async isReady(host, port, timeout = 1000) {
    try {
      const abortController = new AbortController();
      setTimeout(() => abortController.abort(), timeout);

      const response = await fetch(`http://${host}:${port}/healthy`, {
        signal: abortController.signal,
        method: 'GET',
        headers: {
          'Connection': 'keep-alive',
        },
      });

      if (!response.ok) {
        const body = await response.text();
        logger.error(`Pod.isReady Error "${response.statusText}":\n${body}`);
      }
      return response.ok;
    } catch (error) {
      logger.error(`Pod.isReady Error: ${error.message}`);
      return false
    }
  }

  async isReady() {
    if (this.isDestroyed) {
      throw `Pod.isReady Error: Pod ${this.name} has already been destroyed.`
    }

    return Pod.isReady(this.ip, this.port);
  }

  /**
   * Get if `this` is available for work.
   */
  async isAvailable() {
    if (this.isDestroyed) {
      throw `Pod.isAvailable Error: Pod ${this.name} has already been destroyed.`
    }

    // If `this` is reserved then this does not make a request for readiness due to the synchronous
    // implementation of the worker pods.
    return !this.isReserved() && (await this.isReady());
  }

  /**
   * Get if `this` is reserved and is unable to do more work.
   */
  isReserved() {
    if (this.isDestroyed) {
      throw `Pod.isReserved Error: Pod ${this.name} has already been destroyed.`
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
      throw `Pod.reserve Error: Pod ${this.name} has already been destroyed.`
    }

    if (this.isReserved()) {
      throw `Pod.reserve Error: Pod ${this.name} is reserved, it cannot be reserved again.`
    }

    logger.log(`Reserving pod ${this.name}.`);
    this.freeSince = undefined;
    return this;
  }

  /**
   * Release `this` Pod from the job.
   */
  release() {
    if (this.isDestroyed) {
      throw `Pod.release Error: Pod ${this.name} has already been destroyed.`
    }

    if (!this.isReserved()) {
      throw `Pod.release Error: Pod ${this.name} has not already been reserved.`
    }

    logger.log(`Releasing pod ${this.name}.`);
    this.freeSince = Date.now();
    return this;
  }

  /**
   * Return `true` if the `this` is no longer suitable for work.
   */
  async isDead() {
    // If `this` is reserved then this does not make a request for readiness due to the synchronous
    // implementation of the worker pods.
    // TODO: Retry a couple times before preemptively killing a pod.
    return this.isDestroyed || (!this.isReserved() && !(await this.isReady()));
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
    if (this.isReserved()) {
      throw `Pod.destroy Error: Pod ${this.name} is reserved, it cannot be destroyed.`
    }

    if (this.isDestroyed) {
      throw `Pod.destory Error: Pod ${this.name} has already been destroyed.`
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
   *
   * @param {int} statusRetries Number of status checks before giving up on `Pod` creation.
   * @param {number} statusLoop Length in milliseconds between pod status checks.
   * @param {number} timeToLive The maximum number of milliseconds this pod is allowed to live.
   *    At minimum, this should a minute long because at minimum GCP charages for a minute
   *    of usage. Learn more: https://cloud.google.com/compute/vm-instance-pricing
   * @throws Error if Pod is unavailable for work, after querying status `statusRetries` times.
   * @returns {Pod} Returns a `Pod`.
   */
  static async build({
    statusRetries = Math.ceil((AVERAGE_POD_BUILD_TIME + MINIMUM_POD_TIME_TO_LIVE) / 2000),
    statusLoop = 5000,
    timeToLive = Infinity,
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
            'image': process.env.WORKER_POD_IMAGE,
            'name': process.env.WORKER_POD_PREFIX,
            'env': [{
                'name': 'NUM_CPU_THREADS',
                'value': '8',
              },
              ...apiKeys,
            ],
            'resources': {
              'requests': {
                // NOTE: This is smaller than required purposefully to give room for any other
                // system pods.
                'memory': '3Gi',
                'cpu': '7250m'
              },
              'limits': {
                'memory': '5Gi'
              },
            },
            'ports': [{
              'containerPort': podPort,
              'protocol': 'TCP'
            }]
          }]
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
          if (!(await Pod.isReady(info.body.status.podIP, podPort))) {
            throw 'Pod.build Error: Pod is not ready to recieve work.';
          }
        } else {
          throw `Pod.build Error: Not running, recieved:\n${JSON.stringify(info.body.status)}`;
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
    return new Pod(name, info.body.spec.nodeName, info.body.status.podIP, podPort, timeToLive);
  }
}

class PodPool {
  /**
   * This manages the scaling and reservation of pods for work.
   *
   * @param {number} scaleLoop This is the maximum time between recomputing the number of pods. This
   *   ensures that pods are scaled down quickly if pods are no longer in-use.
   * @param {number} scalingWindow This is the time we look into the past to compute the number of
   *   resources needed for the next time period.
   */
  constructor(
    scaleLoop = parseInt(process.env.AUTOSCALE_LOOP, 10),
    scalingWindow = parseInt(process.env.AUTOSCALE_WINDOW, 10),
  ) {
    this.pods = [];
    this.numPodsBuilding = 0;
    this.podRequests = [];
    this.reservationLog = new EventLog(scalingWindow);
    this.loop(scaleLoop);
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
   * Take note of the number of reserved pods in `this.pods` after it's been changed.
   */
  logNumReservedPods() {
    const numReservedPods = this.pods.filter(p => p.isReserved()).length;
    logger.log(`PodPool.logNumReservedPods: There are ${numReservedPods} reserved pods.`);
    this.reservationLog.addEvent(numReservedPods);
    this.scale();
  }

  /**
   * Fulfill the next pod reservation request in `this.podRequests`.
   */
  async fufillPodRequests() {
    while (this.podRequests.length > 0) {
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
      if (await pod.isReady()) {
        this.podRequests.shift()(pod);
        this.logNumReservedPods();
      } else { // CASE: `pod` is dead.
        pod.release();
      }
    }
    this.scale();
  }

  /**
   * Reserve a pod for work.
   *
   * @returns {Pod} A reserved pod available to complete work.
   */
  async reservePod() {
    return new Promise((resolve) => {
      logger.log(`PodPool.reservePod: Requesting pod for work.`);
      this.podRequests.push((pod) => {
        logger.log(`PodPool.reservePod: Reserved ${pod.name} for work.`);
        resolve(pod);
      });
      this.fufillPodRequests(); // Attempt to immediately fufill the pod request.
    });
  }

  /**
   * Release pod to the pool.
   *
   * @param {Pod} pod The pod to release.
   */
  async releasePod(pod) {
    pod.release();
    logger.log(`PodPool.releasePod: Released ${pod.name}.`);
    this.logNumReservedPods();
    this.fufillPodRequests();
  }

  /**
   * Recompute the number of pods needed and scale.
   *
   * This method should be called whenever the results might be affected.
   */
  async scale() {
    await this.clean();
    logger.log(`PodPool.scale: There are ${this.pods.length} pod(s).`);
    logger.log(`PodPool.scale: There are ${this.numPodsBuilding} pod(s) building.`);
    const shortTermNumPodsDesired = PodPool.getNumShortTermPods(
      this.podRequests.length, this.pods.length);
    logger.log(`PodPool.scale: This desires short term ${shortTermNumPodsDesired} pod(s).`);
    const longTermNumPodsDesired = PodPool.getNumLongTermPods(this.reservationLog);
    logger.log(`PodPool.scale: This desires long term ${longTermNumPodsDesired} pod(s).`);
    const numPodsDesired = Math.max(shortTermNumPodsDesired, longTermNumPodsDesired);
    logger.log(`PodPool.scale: This desires ${numPodsDesired} pod(s).`);
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
    logger.log(`PodPool.downscale: Attempting to downscale to ${numPods} pods.`);
    let toDestory = this.pods.filter(p => !p.isReserved());

    // Due to GCP pricing, it does not make sense to destroy pods prior to
    // `MINIMUM_POD_TIME_TO_LIVE`.
    toDestory = toDestory.filter(p => Date.now() - p.createdAt >= MINIMUM_POD_TIME_TO_LIVE);
    logger.log(`PodPool.downscale: ${toDestory.length} pods are eligable for destruction.`);

    toDestory = toDestory.sort((a, b) => a.freeSince - b.freeSince); // Sort by least recently used

    // Select pods to destroy
    let numPodsToDestory = Math.min(this.pods.length - numPods, toDestory.length);
    toDestory = toDestory.slice(0, numPodsToDestory);
    logger.log(`PodPool.downscale: Destorying ${toDestory.length} pods.`);

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
    logger.log(`PodPool.upscale: Creating ${numPodsToCreate} pods.`);
    this.numPodsBuilding += numPodsToCreate;
    return Promise.all(Array.from({
      length: numPodsToCreate
    }, async () => {
      try {
        const pod = await Pod.build();
        if (pod != null) {
          this.pods.push(pod);
          this.fufillPodRequests();
        }
      } catch (error) {
        throw `PodPool.upscale Error: Pod build failed: ${error}`;
      }
      this.numPodsBuilding -= 1;
    }));
  }

  /**
   * Remove any dead pods.
   */
  async clean() {
    const deadPods = await asyncFilter(this.pods, p => p.isDead());
    logger.log(`PodPool.clean: There are ${deadPods.length} dead pod(s).`);
    this.pods = this.pods.filter(p => !deadPods.includes(p));
    return Promise.all(deadPods.map(p => p.destroy()));
  }

  /**
   * Get the number of pods needed to complete the outstanding jobs.
   *
   * @param {number} numJobsOutstanding The number of jobs that need to be completed.
   * @param {number} numPods The number of pods existing to complete jobs.
   * @param {number} averagePodBuildTime The average time in milliseconds it takes for a pod to come
   *    online.
   * @param {number} averageJobTime The average time in milliseconds it takes for a job to finish.
   * @param {number} minJobsPerPod The minimum work a pod should do before going offline. This
   *    ensures that pods are price efficient.
   */
  static getNumShortTermPods(
    numJobsOutstanding,
    numPods,
    averagePodBuildTime = AVERAGE_POD_BUILD_TIME,
    averageJobTime = AVERAGE_JOB_TIME,
    minJobsPerPod = Math.floor(MINIMUM_POD_TIME_TO_LIVE / AVERAGE_JOB_TIME),
  ) {
    logger.log(`PodPool.getNumShortTermPods: There are ${numJobsOutstanding} jobs outstanding.`);
    const jobsCompletedDuringPodBuildTime = Math.floor(averagePodBuildTime / averageJobTime);
    if (numJobsOutstanding / jobsCompletedDuringPodBuildTime > numPods) {
      // Get the number of pods to build to finish the outstanding jobs as quickly as possible.
      const jobsUnaccountedFor = numJobsOutstanding - numPods * jobsCompletedDuringPodBuildTime;
      const numPodsToBuild = Math.ceil(jobsUnaccountedFor / minJobsPerPod);
      return numPods + numPodsToBuild;
    }

    // Get the number of pods needed so that new pods do not need to be built.
    return Math.ceil(numJobsOutstanding / jobsCompletedDuringPodBuildTime);
  }

  /**
   * Get the number of pods consistently used over a period of time.
   *
   * Note this computes the weighted median of the number of pods reserved in the `reservationLog`.
   * This means that we'll have enough resources to fufill customer needs immediately most of the
   * time as a baseline.
   *
   * @param {EventLog} reservationLog  A log of the number of reservations over a period of time.
   * @param {number} minPods The minimum worker pods to keep online always.
   * @param {number} extraPods The percentage of extra pods to keep online for any spill over.
   * @returns {number} Get the number of pods to keep online.
   */
  static getNumLongTermPods(
    reservationLog,
    minPods = parseInt(process.env.MINIMUM_WORKER_PODS, 10),
    extraPods = parseFloat(process.env.EXTRA_WORKER_PODS),
  ) {
    const numEvents = reservationLog.events.length;
    logger.log(`PodPool.getNumLongTermPods: There are ${numEvents} events in  \`reservationLog\`.`);
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
      if (timeCounter >= totalTime / 2) {
        return Math.max(minPods, Math.ceil(numReserved * (1 + extraPods)));
      }
    }
  }
}

APP.use(bodyParser.json());

APP.get('/styles.css', (_, response) => {
  response.sendFile('styles.css', {
    root: __dirname
  });
});

APP.get('/reset.css', (_, response) => {
  response.sendFile('reset.css', {
    root: __dirname
  });
});

APP.get('/script.js', (_, response) => {
  response.sendFile('script.js', {
    root: __dirname
  });
});

APP.get('/', (_, response) => {
  response.sendFile('index.html', {
    root: __dirname
  });
});

APP.get('/favicon.ico', (_, response) => {
  response.sendFile('favicon.ico', {
    root: __dirname
  });
});

APP.get('/healthy', (_, response) => {
  response.send('ok');
});

// TODO: Consider remove `/api/*` so it is not redudant with the subdomain `api`.

APP.all('/api/*', async (request, response, next) => {
  logger.log(`/api/*: Got request.`);
  let pod;
  try {
    pod = await request.app.locals.podPool.reservePod();
  } catch (error) {
    next(error);
    return;
  }

  const ttsAbortController = new AbortController();
  const prefix = `/api/* [${pod.name}]: `;
  let isPodReserved = true; // Ensure that a pod is not released twice.

  function exit(message) {
    if (message) {
      logger.log(`${prefix}${message}`);
    }
    ttsAbortController.abort();
    response.end();
    if (isPodReserved) {
      request.app.locals.podPool.releasePod(pod);
      isPodReserved = false;
    }
  }

  try {
    const ttsEndPoint = `http://${pod.ip}:${pod.port}${request.url}`;
    // NOTE: We do not log the body because it contains sensitive information.
    logger.log(`${prefix}Sending request with to ${ttsEndPoint} headers:\n` +
      JSON.stringify(request.headers));
    const ttsResponse = await fetch(ttsEndPoint, {
      signal: ttsAbortController.signal,
      method: request.method,
      headers: request.headers,
      body: JSON.stringify(request.body),
    });

    // Handle canceled request
    // https://stackoverflow.com/questions/35198208/handling-cancelled-request-with-express-node-js-and-angular
    const requestType = request.constructor.name
    request
      // NOTE: `http.IncomingMessage`: Handle events when the request has been aborted or the
      // underlying connection was closed.
      .on('aborted', () => exit(`\`request\` (${requestType}) emitted 'aborted' event.`))
      .on('close', () => exit(`\`request\` (${requestType}) emitted 'close' event.`))
      .on('finish', () => exit(`\`request\` (${requestType}) emitted 'finish' event.`))
      .on('error', error => exit(`\`request\` (${requestType}) emitted 'error' (${error}) event.`));

    const responeType = response.constructor.name
    response
      // NOTE: `http.ServerResponse`: Handle events when the response has been sent or the
      // underlying connection was closed.
      .on('close', () => exit(`\`response\` (${responeType}) emitted 'close' event.`))
      .on('finish', () => exit(`\`response\` (${responeType}) emitted 'finish' event.`))
      .on('error', error => exit(
        `\`response\` (${responeType}) emitted 'error' (${error}) event.`));

    // Stream response back
    response.writeHead(ttsResponse.status, ttsResponse.headers.raw());
    const ttsResponseType = ttsResponse.body.constructor.name;
    ttsResponse.body
      .on('data', chunk => response.write(chunk))
      // NOTE: `stream.Readable`: Handle events when there is no more data to be read.
      .on('end', () => exit(`\`ttsResponse.body\` (${ttsResponseType}) emitted 'end' event.`))
      .on('close', () => exit(`\`ttsResponse.body\` (${ttsResponseType}) emitted 'close' event.`))
      .on('error', (error) =>
        exit(`\`ttsResponse.body\` (${ttsResponseType}) emitted 'error' (${error}) event.`));
  } catch (error) { // Catch and clean up after any other error
    exit(`${prefix}Error: ${error}`);
    next(error);
  }
});

if (require.main === module) {
  APP.locals.podPool = new PodPool();
  const listener = APP.listen(8000, '0.0.0.0', () => logger.log(`Listening on port ${8000}!`));

  // Shutdown `server` gracefully.
  // Learn more: https://expressjs.com/en/advanced/healthcheck-graceful-shutdown.html
  // Learn more: https://hackernoon.com/graceful-shutdown-in-nodejs-2f8f59d1c357
  process.on('SIGTERM', () => {
    console.log('SIGTERM signal received, shutting down.');
    listener.close(() => {
      console.log('HTTP server closed.');
      clearTimeout(APP.locals.podPool.loopTimeout);
      // NOTE: There needs to be at least 1 Pod for GKE to function.
      APP.locals.podPool.clean();
      APP.locals.podPool.downscale(1);
    });
  });
} else {
  module.exports = {
    EventLog,
    Pod,
    PodPool,
    sleep,
    retry,
    asyncFilter,
  };
}
