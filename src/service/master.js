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
 * TODO: Rewrite these dependancies with conda, and without pyenv or pip.
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
// NOTE: This average is based off of the below data:
// https://medium.com/google-cloud/understanding-and-profiling-gce-cold-boot-time-32c209fe86ab
const AVERAGE_POD_BUILD_TIME = 49 * 1000;
// NOTE: This average assumes 6x real time processing for an average clip length of 7 seconds.
const AVERAGE_JOB_TIME = 6 * 7 * 1000;
// NOTE: At minimum, this should a minute long because at minimum GCP charges for a minute
// of usage. Learn more: https://cloud.google.com/compute/vm-instance-pricing
// NOTE: After the minute long, GCP charges for every second of usage.
const MINIMUM_POD_TIME_TO_LIVE = 60 * 1000 + 999;

const logger = {
  log: (...arguments) => console.log(`[${(new Date()).toISOString()}]`, ...arguments),
  warn: (...arguments) => console.warn(`[${(new Date()).toISOString()}]`, ...arguments),
  error: (...arguments) => console.error(`[${(new Date()).toISOString()}]`, ...arguments),
}

// NOTE: Once this Pod is delegated as an owner of a kubernetes resource it enables those
// resources to be garbage collected if this pod dies.
const OWNER_REFERENCE = {
  'apiVersion': 'v1',
  'blockOwnerDeletion': true,
  'controller': true,
  'kind': 'Pod',
  'name': process.env.THIS_POD_NAME,
  'uid': process.env.THIS_POD_UID,
}

/**
 * Get a `Client` in `THIS_POD_NAMESPACE` used to make requests to the Kubernetes API.
 */
async function getClient(namespace = process.env.THIS_POD_NAMESPACE) {
  let cache;

  async function makeClient() {
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

  return !cache ? await makeClient() : cache;
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
   * Create a node that is healthy and ready to accept pods.
   *
   * NOTE: The GCP VM cold startup time is around 25 - 45 seconds. Afterwards, it takes time to
   * download and start the docker image. Learn more:
   * https://medium.com/google-cloud/understanding-and-profiling-gce-cold-boot-time-32c209fe86ab
   * https://cloud.google.com/blog/products/gcp/three-steps-to-compute-engine-startup-time-bliss-google-cloud-performance-atlas?m=1
   *
   * @param {int} statusRetries Number of status checks before giving up on node creation.
   * @param {number} statusLoop Length in milliseconds between node status checks.
   * @throws Error if node is unavailable for work, after querying status `statusRetries` times.
   * @returns {string} Returns the name of the node created.
   */
  static async _buildNode({
    statusRetries = 90,
    statusLoop = 2000
  } = {}) {
    // TODO: Protect against the rare case that this name already exists.
    const name = `${process.env.WORKER_NODE_PREFIX}-${uuidv4()}`;
    logger.log(`Pod._buildNode: Creating node named ${name}.`);

    let info = await (await getClient(null)).nodes.post({
      'body': {
        'apiVersion': 'v1',
        'kind': 'Node',
        'metadata': {
          'name': name,
          'labels': {
            'run': process.env.WORKER_NODE_PREFIX,
            'cloud.google.com/gke-nodepool': process.env.WORKER_NODE_POOL,
          },
          'ownerReferences': [OWNER_REFERENCE],
        },
      }
    });
    logger.log(`Pod._buildNode: Node ${name} manifest sent.`);

    try {
      await retry(async () => {
        /**
         * Check if `Node` is "healthy and ready to accept pods"; otherwise, throw an error. Learn
         * more: https://kubernetes.io/docs/concepts/architecture/nodes/#manual-node-administration
         */
        info = await (await getClient(null)).nodes(name).get();
        if ('conditions' in info.body.status) {
          const conditions = info.body.status.conditions.filter(c => c.type == 'Ready');
          if (conditions.length == 0 || conditions[0].status != 'True') {
            throw `Pod._buildNode Error: Not running, recieved:\n${JSON.stringify(conditions)}`;
          }
        } else {
          throw `Pod._buildNode Error: Unable to learn of node condition:\n${JSON.stringify(info)}`;
        }
      }, {
        retries: statusRetries,
        delay: statusLoop,
      });
    } catch (error) {
      try {
        logger.warn(`Pod._buildNode: Unable to start, deleting node ${name}.`);
        await (await getClient(null)).nodes(name).delete();
      } catch (error) {
        logger.error(`Pod._buildNode: Failed to delete node ${name} due to error: ${error}`);
      }
      return null;
    }

    logger.log(`Pod._buildNode: Node ${name} is healthy and ready to accept pods.`);
    return name;
  }

  /**
   * Create a `Pod` running an image, ready to recieve requests.
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
    statusRetries = 90,
    statusLoop = 2000,
    timeToLive = Infinity,
  } = {}) {
    const nodeName = await Pod._buildNode();
    if (nodeName == null) {
      return null;
    }
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
          'ownerReferences': [OWNER_REFERENCE],
        },
        'spec': {
          'restartPolicy': 'Never',
          'nodeName': nodeName,
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
    logger.log(`Pod.build: Pod ${name} sent.`);

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
      Pod.destroy(name, nodeName);
      return null;
    }

    logger.log(`Pod.build: Pod ${name} is ready to recieve traffic at ${info.body.status.podIP}`);
    return new Pod(name, nodeName, info.body.status.podIP, podPort, timeToLive);
  }
}

class PodPool {
  /**
   * This manages the scaling and reservation of pods for work.
   *
   * @param {number} scaleLoop This is the maximum time between recomputing the number of pods. This
   *   ensures that pods are scaled down quickly if pods are no longer in-use.
   * @param {number} reservePodLoop This is the time between checking for pod availability. It is
   *   important to schedule the work for the user as fast as possible.
   * @param {number} scalingWindow This is the time we look into the past to compute the number of
   *   resources needed for the next time period.
   */
  constructor(
    scaleLoop = parseInt(process.env.AUTOSCALE_LOOP, 10),
    reservePodLoop = parseInt(process.env.AVAILABILITY_LOOP, 10),
    scalingWindow = parseInt(process.env.AUTOSCALE_WINDOW, 10),
  ) {
    this.pods = [];
    this.numPodsBuilding = 0;
    this.reservePodLoop = reservePodLoop;
    this.unmetNeed = 0;
    this.reservationLog = new EventLog(scalingWindow);
    this.loop(scaleLoop);
  }

  /**
   * Start a loop to scale `this`.
   *
   * @param {number} delay The number of milliseconds between loops.
   */
  loop(delay) {
    this.loopTimeout = setTimeout(() => {
      this.scale();
      this.loop(delay);
    }, delay);
  }

  logNumReservedPods() {
    this.reservationLog.addEvent(this.pods.filter(p => p.isReserved()).length);
    this.scale();
  }

  /**
   * Reserve a pod for work.
   *
   * @returns {Pod} A reserved pod available to complete work.
   */
  async reservePod() {
    this.unmetNeed += 1;
    this.scale();

    async function _reservePod() {
      while (true) {
        const available = this.pods.filter(p => !p.isReserved());
        logger.log(`PodPool.reservePod: Number of unreserved pods ${available.length}.`);
        // NOTE: Sort most recently used first, allowing unused pods to stay unused.
        const pod = available.sort((a, b) => b.freeSince - a.freeSince)[0];
        // NOTE: Reserve preemptively before `await` during which `javascript` could run another
        // thread that'll reserve it.
        pod.reserve();
        if (await pod.isReady()) {
          this.unmetNeed -= 1;
          this.logNumReservedPods();
          return pod;
        }
        pod.release();
        await sleep(this.reservePodLoop);
      }
    }

    const pod = await _reservePod();
    logger.log(`PodPool.reservePod: Reserved ${pod.name} for work.`);
    return pod;
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
  }

  /**
   * Recompute the number of pods needed and scale.
   */
  async scale() {
    await this.clean();
    const shortTermNumPodsDesired = PodPool.getNumShortTermPods(this.unmetNeed, this.pods.length);
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
    let toDestory = this.pods.filter(p => !p.isReserved());

    // Due to GCP pricing, it does not make sense to destroy pods prior to
    // `MINIMUM_POD_TIME_TO_LIVE`.
    toDestory = toDestory.filter(p => Date.now() - p.createdAt >= MINIMUM_POD_TIME_TO_LIVE);

    // Sort by least recently used
    toDestory = toDestory.sort((a, b) => a.freeSince - b.freeSince);

    // Select pods to destroy
    let numPodsToDestory = Math.min(this.pods.length - numPods, toDestory.length);
    logger.log(`PodPool.downscale: Destorying ${numPodsToDestory} pods.`);
    toDestory = toDestory.slice(numPodsToDestory);

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
      const pod = await Pod.build();
      this.numPodsBuilding -= 1;
      if (pod != null) {
        this.pods.push(pod);
      }
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
    const jobsCompletedDuringPodBuildTime = Math.floor(averagePodBuildTime / averageJobTime);
    if (numJobsOutstanding / jobsCompletedDuringPodBuildTime > numPods) {
      // Get the number of pods to build to finish the outstanding jobs as quickly as possible.
      const jobsUnaccountedFor = numJobsOutstanding - numPods * jobsCompletedDuringPodBuildTime;

      const numPodsToBuild = Math.floor(jobsUnaccountedFor / minJobsPerPod);
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
    if (reservationLog.events.length == 0) {
      return minPods;
    }
    if (reservationLog.events.length == 1) {
      return Math.max(minPods, Math.ceil(reservationLog.events[0] * (1 + extraPods)));
    }

    const mapTimeCounter = new Map();
    for (let i = 0; i < reservationLog.events.length - 1; i++) {
      const numReserved = reservationLog.events[i];
      const timeReserved = reservationLog.timestamps[i + 1] - reservationLog.timestamps[i];
      if (mapTimeCounter.has(numReserved)) {
        mapTimeCounter.set(numReserved, mapTimeCounter.get(numReserved) + timeReserved);
      } else {
        mapTimeCounter.set(numReserved, timeReserved);
      }
    }

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

if (require.main === module) {
  const pool = new PodPool();
}

// TODO: Consider remove `/api/*` so it is not redudant with the subdomain `api`.

APP.all('/api/*', async (request, response, next) => {
  logger.log(`/api/*: Got request.`);
  let pod;
  try {
    pod = await pool.reservePod();
  } catch (error) {
    next(`/api/*: Error: ${error}`);
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
      pool.releasePod(pod);
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
    exit();
    next(`${prefix}Error: ${error}`);
  }
});

if (require.main === module) {
  APP.listen(8000, '0.0.0.0', () => logger.log(`Listening on port ${8000}!`));
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
