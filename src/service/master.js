/**
 * TODO: Ensure that there are multiple copies of master running, in case of version upgrades.
 * TODO: Switch from `npm` to `yarn` for dependancy management.
 * TODO: Write tests for `master.js`.
 * TODO: Consider using HTTPS protocall between LoadBalancer and the container so that the API
 * Key is not sniffable?
 * TODO: Consider creating a namespace in kubernetes, it's proper practice.
 * TODO: Put a cache in front of the API frontend, ensuring it does not fail under a DDOS
 * attack.
 * TODO: Check that `worker.py` PyTorch uses MKL, log this.
 * TODO: Rewrite these dependancies with conda, and without pyenv or pip.
 */
const express = require('express');
const Client = require('kubernetes-client').Client
const config = require('kubernetes-client').config
const uuidv4 = require('uuid/v4');
const bodyParser = require('body-parser');
const http = require('http');

const APP = express();
let PODS = []; // Add and removing pods is handled by `Pod` class.

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

  static isReady(host, port, timeout = 1000) {
    /**
     * Send a HTTP request to work, ensuring it's ready for more requests.
     *
     * @param {string} host Host to make HTTP request to.
     * @param {number} port Port on host to query.
     * @param {number} timeout Timeout for `http.request`.
     */
    return new Promise((resolve, _) => {
      console.log(`Pod.isReady: Requesting readiness from ${host}`);
      const request = http
        .request({
          host: host,
          port: port,
          path: '/healthy',
          method: 'GET',
          headers: {
            'Connection': 'keep-alive',
          },
          timeout: timeout
        }, (response) => {
          resolve(response.statusCode == 200);
        })
        .on('error', (error) => {
          console.error(`Pod.isReady Error: ${error.message}`);
          resolve(false);
        }).on('timeout', () => request.abort()).end();
    });
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

    console.log(`Reserving pod ${this.name}.`);
    this.freeSince = undefined;
    return this;
  }

  release() {
    /**
     * Release `this` Pod from the job.
     */
    console.log(`Releasing pod ${this.name}.`);
    this.freeSince = Date.now();
    return this;
  }

  async isDead() {
    /**
     * Return `true` if the `this` is no longer suitable for work.
     */
    // If `this` is reserved then this does not make a request for readiness due to the synchronous
    // implementation of the worker pods.
    return !this.isReserved() && !(await this.isReady());
  }

  async destroy() {
    /**
     * Destroy `this` and remove from `PODS`.
     */
    if (this.isReserved()) {
      throw `Pod.destroy Error: Pod ${this.name} is reserved, it cannot be destroyed.`
    }

    console.log(`Pod.destroy: Deleting Pod ${this.name}.`);
    PODS = PODS.filter(pod => pod !== this);
    try {
      await (await getClient()).pods(this.name).delete();
      console.log(`Pod.destroy: Deleted Pod ${this.name}.`);
    } catch (error) {
      console.log(`Pod.destroy: Failed to delete Pod ${this.name} due to error: ${error}`);
      // TODO: Handle this scenario, try avoiding leaking a pod.
    }
  }

  static async build({
    statusRetries = 15,
    statusLoop = 2000,
    reserved = false
  } = {}) {
    /**
     * Create a `Pod` running an image, ready to recieve requests.
     *
     * @param {int} statusRetries Number of status checks before giving up on `Pod` creation.
     * @param {number} statusLoop Length in milliseconds between pod status checks.
     * @param {bool} reserved Reserve the pod on creation, preventing race to reserve the new Pod.
     * @throws Error if Pod is unavailable for work, after querying status `statusRetries` times.
     * @returns {Pod} Returns a `Pod` in the `PODS` pool.
     */
    const name = `${process.env.WORKER_POD_PREFIX}-${uuidv4()}`;
    console.log(`Pod.build: Creating pod named ${name}.`);

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
            'env': apiKeys,
            'resources': {
              'requests': {
                'memory': '3Gi',
                'cpu': '4000m'
              },
              'limits': {
                'memory': '3Gi',
                'cpu': '4000m'
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
    console.log(`Pod.build: Pod ${name} created.`);

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
        exponentialBackoff: false
      });
    } catch (error) {
      try {
        console.warn(`Pod.build Warning: Unable to start, deleting Pod ${name}.`);
        await (await getClient()).pods(name).delete();
      } catch (error) {
        console.error(`Pod.build Error: Failed to delete Pod ${name} due to error: ${error}`);
      }
      throw error;
    }

    console.log(`Pod.build: Pod ${name} is ready to recieve traffic at ${info.body.status.podIP}`);
    const pod = new Pod(name, info.body.status.podIP, podPort);
    return reserved ? pod.reserve() : pod;
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
  exponentialBackoff = true
} = {}) {
  /**
   * Retries a function mulitple times.
   *
   * @param {Function} toTry Function to retry.
   * @param {number} retries Number of times to retry the function.
   * @param {number} delay Initial delay in milliseconds.
   * @param {bool} exponentialBackoff Increase the delay exponentially every retry.
   * @returns {any}
   */
  for (let i = 0; i < retries; i++) {
    try {
      console.log(`retry: Attempt #${i}.`);
      const result = await toTry();
      return result;
    } catch (error) {
      console.warn(`retry: Caught this error: ${error}`);
      if (i == retries - 1) {
        console.error(`retry: Reached maximum retries ${retries}, throwing error.`);
        throw error;
      } else {
        if (exponentialBackoff) {
          delay *= 2;
        }
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
    console.log(`autoscaleJob: There are ${deadPods.length} dead pod(s).`);
    await Promise.all(deadPods.map(pod => pod.destroy()));

    // Compute the `POD` statistics
    const available = await async_filter(PODS, pod => pod.isAvailable());
    const numNotAvailable = PODS.length - available.length;
    const numDesired = Math.max(parseInt(process.env.INITIAL_WORKER_PODS, 10),
      Math.ceil(numNotAvailable * (1 + parseFloat(process.env.EXTRA_WORKER_PODS))));

    console.log(`autoscaleJob: There are ${numNotAvailable} unavailable and ` +
      `${available.length} available pod(s).`);
    console.log(`autoscaleJob: This desires ${numDesired} pod(s).`);

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
      console.log(`autoscaleJob: Destorying ${numPodsToDestory} pods.`);
      toDestory = toDestory.slice(numPodsToDestory);

      // Destroy pods
      await Promise.all(toDestory.map((pod) => pod.destroy()));
    } else if (PODS.length < numDesired) {
      // Create additional pods
      const numPodsToCreate = numDesired - PODS.length;
      console.log(`autoscaleJob: Creating ${numPodsToCreate} pods.`);
      try {
        await Promise.all(Array.from({
          length: numPodsToCreate
        }, () => Pod.build()));
      } catch (error) {
        console.error(`autoscaleJob: Unable to create pods due to error: ${error}`);
      }
    }

    await sleep(parseInt(process.env.AUTOSCALE_LOOP, 10));
  }
})();

async function getPodForWork() {
  /**
   * Get an available pod and reserve it, useful for completing a job.
   *
   * @return {Pod} Available and reserved pod.
   */

  async function _getPodForWork() {
    let available = PODS.filter(pod => !pod.isReserved());
    // Sort most recently used first, allowing unused pods to stay unused, triggering down scaling.
    available = available.sort((a, b) => b.freeSince - a.freeSince);
    console.log(`_getPodForWork: Number of unreserved pods ${available.length}.`);
    for (let pod of available) {
      // Reserve preemptively before `await` during which `javascript` could run another thread
      // that'll reserve it.
      pod.reserve();
      if (await pod.isReady()) { // Get first ready pod.
        return pod;
      } else {
        pod.release();
      }
    }
    // TODO: Consider if `Pod.build` is running too long to revist getting other `available` pods.
    console.log(`_getPodForWork: Created an extra Pod to fufill demand.`);
    return await Pod.build({
      reserved: true
    });
  }

  const pod = await _getPodForWork();
  console.log(`getPodForWork: Reserved ${pod.name} for work.`);
  return pod;
}

APP.use(bodyParser.raw()); // Learn more: https://github.com/request/request/issues/2391

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

APP.all('/api/*', async (request, response, next) => {
  console.log(`/api/*: Got request.`);
  let pod;
  try {
    pod = await getPodForWork();
  } catch (error) {
    next(`/api/*: Error: ${error}`);
    return;
  }

  try {
    const options = {
      host: pod.ip,
      port: pod.port,
      path: request.url,
      method: request.method,
      headers: request.headers,
    }
    const prefix = `/api/* [${pod.name}]: `;
    console.log(`${prefix}Sending request with headers:\n` +
      JSON.stringify(options.headers));
    const destination = http.request(options, (stream) => {
        // Write stream back.
        response.writeHead(stream.statusCode, stream.headers);
        stream
          .on('data', (chunk) => response.write(chunk))
          .on('close', () => response.end());
      })
      .on('error', (error) => {
        pod.release();
        next(`${prefix}\`http.request\` emitted error event: ${error}`);
      })
      .on('abort', () => {
        pod.release();
        console.log(`${prefix}Stream emitted 'abort' event.`);
      });

    function clean(message) {
      if (message) {
        console.log(message);
      }
      pod.release(); // Release pod for more work.
      destination.abort(); // Clean up proxy stream
    }

    // Clean up after responding
    response
      .on('close', () => clean(`${prefix}Response emitted 'close' event.`))
      .on('finish', () => clean(`${prefix}Response emitted 'finish' event.`))
      .on('error', (error) => clean(`${prefix}Response emitted 'error' (${error}) event.`));

    // Send data to `http.request`
    request
      .on('data', chunk => destination.write(chunk))
      .on('end', () => destination.end())
      .on('error', () => clean(`${prefix}Request emitted 'error' (${error}) event.`));

  } catch (error) { // Catch and clean up after any other error
    pod.release();
    next(`${prefix}Error: ${error}`);
  }
});

APP.listen(8000, '0.0.0.0',
  () => console.log(`Listening on port ${8000}!`));
