/**
 * This file contains code that produces the configuration for running the
 * WellSaidLabs Text-to-Speech service on a GKE cluster using Google Cloud
 * Run.
 *
 * We run the same image twice so that we can scale them separately.
 * One is dedicated to handling input validation requests, which are fast
 * to return and can handle multiple requests per pod. The other handles
 * streaming audio to the client, which is slow and means each pod can
 * only handle a single request at once.
 *
 * This file can be converted into the desired configuration like so:
 *
 *    jsonnet tts.jsonnet \
 *      -y \
 *      --tla-str env=staging \
 *      --tla-str model=v3 \
 *      --tla-str version=1 \
 *      --tla-str image=gcr.io/voice/tts@sha256:... \
 *      --tla-str imageEntrypoint=src.service.worker:app \
 *      --tla-str provideApiKeyAuthForLegacyContainerSupport=false \
 *      --tla-code stream="{minScale:0,maxScale:32,concurrency:1,paths:['/api/text_to_speech/stream']}" \
 *      --tla-code validate="{minScale:0,maxScale:32,concurrency:4,paths:['/api/text_to_speech/input_validated']}" \
 *      --tla-code traffic="[{tag:'1',percent:100}]" \
 *      --ext-code config={}
 *
 * Alternatively, you can store the configurations in a json file and
 * the following command
 *
 *    jsonnet tts.jsonnet \
 *      -y \
 *      --ext-code-file config=<path_to_config>
 *
 * The model parameter is a unique identifier for the model contained
 * in the image.
 *
 * The version parameter is a unique identifier for for the revision that's
 * being released. It must a valid domain, meaning it should be only lowercase
 * alphanumeric characters and dashes. So things like "v3" or "001" are
 * acceptable.
 *
 * The image parameter is the docker image to run.
 *
 * The imageEntrypoint parameter (optional) is passed to the docker container
 * args. Currently, this is for legacy image support (images prior to v9
 * that required a different entry).
 *
 * The provideApiKeyAuthForLegacyContainerSupport parameter (optional) is
 * intended to support our existing docker images that require api key
 * authentication. These values will be injected as environment variables to
 * the tts image. Kong will inject these api keys prior to proxying the
 * upstream request and will only exist for backwards compatibility. On
 * that note, consumer-facing credentials will be handled via Kong so make
 * sure not to get provideApiKeyAuthForLegacyContainerSupport confused with
 * consumer-facing api keys.
 *
 * The stream.minScale|validate.minScale option determines the min number of
 * cloud run containers to run at all time. For our staging environment and
 * low demand model versions we will want to scale to 0. Note that changes
 * to this value will require a bump in the `version` parameter.
 *
 * The stream.maxScale|validate.maxScale option places a ceiling on the number
 * of cloud run containers that can be scaled to. Note that changes
 * to this value will require a bump in the `version` parameter.
 *
 * The stream.concurrency|validate.concurrency option defines the number
 * of concurrent requests that this service can handle.
 *
 * The stream.paths|validate.paths option defines the http endpoints that
 * the service will respond to.
 *
 * The output of the command can be redirected to a file, or piped to
 * kubectl directly like so:
 *
 *    jsonnet tts.jsonnet ... | kubectl apply -f -
 *
 */
local common = import 'svc.libsonnet';

local config = std.extVar("config");

function(
  env=config.env,
  model=config.model,
  version=config.version,
  image=config.image,
  imageEntrypoint=config.imageEntrypoint,
  provideApiKeyAuthForLegacyContainerSupport=config.provideApiKeyAuthForLegacyContainerSupport,
  stream=config.stream,
  validate=config.validate,
  traffic=config.traffic
)
  local ns = {
    apiVersion: 'v1',
    kind: 'Namespace',
    metadata: {
      name: model,
    },
  };

  local hostname = if env == 'staging' then
    'staging.tts.wellsaidlabs.com'
    else if env == 'prod' then
    'tts.wellsaidlabs.com';

  local legacyContainerApiKey = if provideApiKeyAuthForLegacyContainerSupport == 'true' then
    // NOTE: this value is arbitrary and is only intended to support the auth
    // interface of our existing TTS docker images. It cannot be used for
    // authenticating external users and is only applicable to internal cluster
    // communications.
    'pKfRepQY-ln4pCOnCxZOoHNXArHbxLwj';

  // list of headers derived from namespace and traffic revisions
  // ex: v8, v8.1, v8.2 that map to individual cloud run revision tags
  local acceptVersionHeaders = [ns.metadata.name] +
    [ns.metadata.name + '.' + t.tag for t in traffic];

  local validateSvc = common.Service({
    name: 'validate',
    namespace: ns.metadata.name,
    image: image,
    entrypoint: imageEntrypoint,
    version: version,
    scale: {
      min: validate.minScale,
      max: validate.maxScale
    },
    resources: {
      requests: {
        cpu: '1',
        memory: '1G',
      },
    },
    concurrency: validate.concurrency,
    timeout: 10,
    restartTimeout: 10,
    legacyContainerApiKey: legacyContainerApiKey,
    traffic: [
      {
        tag: 'revision-' + t.tag,
        percent: t.percent,
        revisionName: 'validate-' + t.tag
      } for t in traffic
    ]
  });

  local streamSvc = common.Service({
    name: 'stream',
    namespace: ns.metadata.name,
    image: image,
    entrypoint: imageEntrypoint,
    version: version,
    scale: {
      min: stream.minScale,
      max: stream.maxScale
    },
    resources: {
      requests: {
        cpu: '6',
        memory: '4G',
      },
      limits: {
        cpu: '7',
        memory: '5G',
      },
    },
    concurrency: stream.concurrency,
    timeout: 300, // 5 minutes
    restartTimeout: 300,  // 5 minutes
    legacyContainerApiKey: legacyContainerApiKey,
    traffic: [
      {
        tag: 'revision-' + t.tag,
        percent: t.percent,
        revisionName: 'stream-' + t.tag
      } for t in traffic
    ]
  });

  local validateRoute = common.Route({
    hostname: hostname,
    namespace: ns.metadata.name,
    serviceName: validateSvc.metadata.name,
    servicePaths: validate.paths,
    legacyContainerApiKey: legacyContainerApiKey,
    acceptVersionHeaders: acceptVersionHeaders,
  });

  local streamRoute = common.Route({
    hostname: hostname,
    namespace: ns.metadata.name,
    serviceName: streamSvc.metadata.name,
    servicePaths: stream.paths,
    legacyContainerApiKey: legacyContainerApiKey,
    acceptVersionHeaders: acceptVersionHeaders,
  });

  [
    ns,
    validateSvc,
    streamSvc
  ] + validateRoute + streamRoute
