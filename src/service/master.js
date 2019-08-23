/**
 * TODO: Ensure that there are multiple copies of master running, in case of version upgrades.
 * TODO: Switch from `npm` to `yarn` for dependancy management.
 * TODO: Write tests for `master.js`.
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
let PODS = []; // Add and removing pods is handled by `Pod` class.

const logger = {
  log: (...arguments) => console.log(`[${(new Date()).toISOString()}]`, ...arguments),
  warn: (...arguments) => console.warn(`[${(new Date()).toISOString()}]`, ...arguments),
  error: (...arguments) => console.error(`[${(new Date()).toISOString()}]`, ...arguments),
}

async function getClient() {
  /**
   * Get a `Client` in `THIS_POD_NAMESPACE` used to make requests to the Kubernetes API.
   */
  let cache;

  async function makeClient() {
    const client = new Client({
      config: config.getInCluster(),
    });
    await client.loadSpec();
    return client.api.v1.namespaces(process.env.THIS_POD_NAMESPACE);
  }

  return !cache ? await makeClient() : cache;
}

class Pod {

  constructor(name, ip, port) {
    /**
     * `Pod` in `PODS` represents a worker.
     *
     * @param {string} name Name of the pod created.
     * @param {string} ip IP address of this pod.
     * @param {number} port An exposed port on the pod that is accepting http requests.
     */
    this.name = name;
    this.freeSince = Date.now();
    this.ip = ip;
    this.port = port;

    PODS.push(this);
  }

  // Learn more about pod status: https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/

  static get STATUS_RUNNING() {
    return 'Running';
  }

  static async isReady(host, port, timeout = 1000) {
    /**
     * Send a HTTP request to work, ensuring it's ready for more requests.
     *
     * @param {string} host Host to make HTTP request to.
     * @param {number} port Port on host to query.
     * @param {number} timeout Timeout for `http.request`.
     */
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
    return Pod.isReady(this.ip, this.port);
  }

  async isAvailable() {
    /**
     * Get if `this` is available for work.
     */
    // If `this` is reserved then this does not make a request for readiness due to the synchronous
    // implementation of the worker pods.
    return !this.isReserved() && (await this.isReady());
  }

  isReserved() {
    return this.freeSince === undefined;
  }

  reserve() {
    /**
     * Reserve `this` for a job.
     *
     * TODO: Have a leak prevention mechanism that cleans up pods reserved for more than an hour.
     * This can be implemented with `isReady` due to synchronous nature of the workers; However,
     * we'd need to timeout / abort mechanism. A pod that `isReady` is no longer occupied with work,
     * hence it is not reserved.
     */
    if (this.isReserved()) {
      throw `Pod.reserve Error: Pod ${this.name} is reserved, it cannot be reserved again.`
    }

    logger.log(`Reserving pod ${this.name}.`);
    this.freeSince = undefined;
    return this;
  }

  release() {
    /**
     * Release `this` Pod from the job.
     */
    logger.log(`Releasing pod ${this.name}.`);
    this.freeSince = Date.now();
    return this;
  }

  async isDead() {
    /**
     * Return `true` if the `this` is no longer suitable for work.
     */
    // If `this` is reserved then this does not make a request for readiness due to the synchronous
    // implementation of the worker pods.
    // TODO: Retry a couple times before preemptively killing a pod.
    return !this.isReserved() && !(await this.isReady());
  }

  async destroy() {
    /**
     * Destroy `this` and remove from `PODS`.
     */
    if (this.isReserved()) {
      throw `Pod.destroy Error: Pod ${this.name} is reserved, it cannot be destroyed.`
    }

    logger.log(`Pod.destroy: Deleting Pod ${this.name}.`);
    PODS = PODS.filter(pod => pod !== this);
    try {
      await (await getClient()).pods(this.name).delete();
      logger.log(`Pod.destroy: Deleted Pod ${this.name}.`);
    } catch (error) {
      logger.log(`Pod.destroy: Failed to delete Pod ${this.name} due to error: ${error}`);
      // TODO: Handle this scenario, try avoiding leaking a pod.
    }
  }

  static async build({
    statusRetries = 90,
    statusLoop = 2000
  } = {}) {
    /**
     * Create a `Pod` running an image, ready to recieve requests.
     *
     * NOTE: The GCP VM cold startup time is around 25 - 45 seconds. Afterwards, it takes time to
     * download and start the docker image. Learn more:
     * https://medium.com/google-cloud/understanding-and-profiling-gce-cold-boot-time-32c209fe86ab
     * https://cloud.google.com/blog/products/gcp/three-steps-to-compute-engine-startup-time-bliss-google-cloud-performance-atlas?m=1
     *
     * @param {int} statusRetries Number of status checks before giving up on `Pod` creation.
     * @param {number} statusLoop Length in milliseconds between pod status checks.
     * @throws Error if Pod is unavailable for work, after querying status `statusRetries` times.
     * @returns {Pod} Returns a `Pod` in the `PODS` pool.
     */
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
    const manifest = {
      'body': {
        'apiVersion': 'v1',
        'kind': 'Pod',
        'metadata': {
          'name': name,
          'labels': { // Default setting created by GKE tooling
            'run': process.env.WORKER_POD_PREFIX,
          },
          'ownerReferences': [{ // Ensure this pod is garbage collected if this pod dies.
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
          }],
          'terminationGracePeriodSeconds': 30, // Default setting created by GKE tooling
          'securityContext': {} // Default setting created by GKE tooling
        }
      }
    }
    let info = await (await getClient()).pods.post(manifest);
    logger.log(`Pod.build: Pod ${name} created.`);

    try {
      await retry(async () => { // While Pod is not running or not ready, keep retrying.
        /**
         * Check if `Pod` is ready; otherwise, throw an error.
         */
        info = await (await getClient()).pods(name).get();
        if (info.body.status.phase == Pod.STATUS_RUNNING) {
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
      try {
        logger.warn(`Pod.build Warning: Unable to start, deleting Pod ${name}.`);
        await (await getClient()).pods(name).delete();
      } catch (error) {
        logger.error(`Pod.build Error: Failed to delete Pod ${name} due to error: ${error}`);
      }
      throw error;
    }

    logger.log(`Pod.build: Pod ${name} is ready to recieve traffic at ${info.body.status.podIP}`);
    return new Pod(name, info.body.status.podIP, podPort);
  }
}

function sleep(milliseconds) {
  /**
   * Sleep for a number of milliseconds.
   *
   * @param {number} milliseconds Number of milliseconds to sleep.
   * @returns {Promise}
   */
  return new Promise(resolve => setTimeout(resolve, milliseconds));
}

async function retry(toTry, {
  retries = 3,
  delay = 100,
} = {}) {
  /**
   * Retries a function mulitple times.
   *
   * @param {Function} toTry Function to retry.
   * @param {number} retries Number of times to retry the function.
   * @param {number} delay Initial delay in milliseconds.
   * @returns {any}
   */
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

async function async_filter(iterator, func) {
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
  let promises = iterator.map(element => func(element));
  promises = await Promise.all(promises);
  return iterator.filter((_, index) => promises[index]);
}


(async function autoscaleJob() {
  /**
   * Job to ensure there is at least
   * `min(INITIAL_WORKER_PODS, ceil(numNotAvailable * (1 + EXTRA_WORKER_PODS)))`
   * pods available or doing work.
   */
  while (true) {
    // Clean up dead pods
    const deadPods = await async_filter(PODS, pod => pod.isDead());
    logger.log(`autoscaleJob: There are ${deadPods.length} dead pod(s).`);
    await Promise.all(deadPods.map(pod => pod.destroy()));

    // Compute the `POD` statistics
    const available = await async_filter(PODS, pod => pod.isAvailable());
    const numNotAvailable = PODS.length - available.length;
    const numDesired = Math.max(parseInt(process.env.INITIAL_WORKER_PODS, 10),
      Math.ceil(numNotAvailable * (1 + parseFloat(process.env.EXTRA_WORKER_PODS))));

    logger.log(`autoscaleJob: There are ${numNotAvailable} unavailable and ` +
      `${available.length} available pod(s).`);
    logger.log(`autoscaleJob: This desires ${numDesired} pod(s).`);

    // Autoscale to `numDesired`
    if (PODS.length > numDesired) {
      let toDestory = PODS.filter((pod) => !pod.isReserved());

      // Get pods that exceeded `AUTOSCALE_DOWNSCALE_DELAY`
      const AUTOSCALE_DOWNSCALE_DELAY = parseInt(process.env.AUTOSCALE_DOWNSCALE_DELAY, 10);
      toDestory = toDestory.filter((pod) => Date.now() - pod.freeSince > AUTOSCALE_DOWNSCALE_DELAY);

      // Sort by least recently used
      toDestory = toDestory.sort((a, b) => a.freeSince - b.freeSince);

      // Select pods to destroy
      let numPodsToDestory = Math.min(PODS.length - numDesired, toDestory.length);
      logger.log(`autoscaleJob: Destorying ${numPodsToDestory} pods.`);
      toDestory = toDestory.slice(numPodsToDestory);

      // Destroy pods
      await Promise.all(toDestory.map((pod) => pod.destroy()));
    } else if (PODS.length < numDesired) {
      // Create additional pods
      const numPodsToCreate = numDesired - PODS.length;
      logger.log(`autoscaleJob: Creating ${numPodsToCreate} pods.`);
      try {
        await Promise.all(Array.from({
          length: numPodsToCreate
        }, () => Pod.build()));
      } catch (error) {
        logger.error(`autoscaleJob: Unable to create pods due to error: ${error}`);
      }
    }

    await sleep(parseInt(process.env.AUTOSCALE_LOOP, 10));
  }
})();

// TODO: Function level comments should be outside of the function.

async function getPodForWork() {
  /**
   * Get an available pod and reserve it, useful for completing a job.
   *
   * @return {Pod} Available and reserved pod.
   */
  // TODO: Consider logging and monitoring the average time it takes for a pod to respond to a
  // job request.
  // NOTE: In this case, there are not enough machines to serve all the concurrent requests.
  let available = PODS.filter(pod => !pod.isReserved());
  if (available.length == 0) {
    Pod.build();
  }

  async function _getPodForWork() {
    while (true) {
      let available = PODS.filter(pod => !pod.isReserved());
      // Sort most recently used first, allowing unused pods to stay unused, triggering down scaling.
      available = available.sort((a, b) => b.freeSince - a.freeSince);
      logger.log(`_getPodForWork: Number of unreserved pods ${available.length}.`);
      for (let pod of available) {
        // Reserve preemptively before `await` during which `javascript` could run another thread
        // that'll reserve it.
        pod.reserve();
        if (await pod.isReady()) { // Get first ready pod.
          return pod;
        }
        pod.release();
      }

      await sleep(parseInt(process.env.AVAILABILITY_LOOP, 10));
    }
  }

  const pod = await _getPodForWork();
  logger.log(`getPodForWork: Reserved ${pod.name} for work.`);
  return pod;
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
    pod = await getPodForWork();
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
      pod.release(); // Release pod for more work.
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

APP.listen(8000, '0.0.0.0', () => logger.log(`Listening on port ${8000}!`));
