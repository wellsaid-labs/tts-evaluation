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
 *      --tla-str model=v3 \
 *      --tla-str version=v0-1-1 \
 *      --tla-str image=gcr.io/voice/tts@sha256:...
 *      --ext-code-file legacyImageApiKeys=apikeys.json
 *
 * The model parameter is a unique identifier for the model contained
 * in the image.
 *
 * The version parameter is a unique identifier for for the revision that's
 * being released. It must a valid domain, meaning it should be only lowercase
 * alphanumeric characters and dashes. So things like "v3" or "001" are
 * acceptbale.
 *
 * The image parameter is the docker image to run.
 *
 * The includeImageApiKeys parameter (optional) is intended to support our
 * existing docker images that require api key authentication. These values
 * are stored in a Secret and will be injected as environment variables to
 * the tts image. Kong will inject these api keys prior to proxying the
 * upstream request and will only exist for backwards compatibility. On
 * that note, consumer-facing credentials will be handled via Kong so make
 * sure not to get includeImageApiKeys confused with consumer-facing api keys.
 *
 * The output of the command can be redirected to a file, or piped to
 * kubectl directly like so:
 *
 *    jsonnet tts.jsonnet ... | kubectl apply -f -
 *
 */
local common = import 'svc.libsonnet';

function(model, version, image, includeImageApiKeys='false')

  local ns = {
    apiVersion: 'v1',
    kind: 'Namespace',
    metadata: {
      name: model,
    },
  };

  local apiKey = if includeImageApiKeys == 'true' then
    local apiKeys = import 'apikeys.json';
    std.objectValues(apiKeys)[0];

  local apiKeySecret = if includeImageApiKeys == 'true' then
    local apiKeys = import 'apikeys.json';
    {
      apiVersion: 'v1',
      kind: 'Secret',
      metadata: {
        name: 'api-keys',
        namespace: ns.metadata.name,
      },
      data: { [k]: std.base64(apiKeys[k]) for k in std.objectFields(apiKeys) },
    };

  local apiKeySecretName = if apiKeySecret != null then apiKeySecret.metadata.name;

  local validateSvc = common.Service({
    name: 'validate',
    namespace: ns.metadata.name,
    image: image,
    version: version,
    scale: { min: 1, max: 30 },
    concurrency: 4,
    timeout: 10,
    restartTimeout: 10,
    apiKeySecretName: apiKeySecretName,
  });

  local streamSvc = common.Service({
    name: 'stream',
    namespace: ns.metadata.name,
    image: image,
    version: version,
    scale: { min: 1, max: 30 },
    concurrency: 1,
    timeout: 3600,  // 1hr
    restartTimeout: 600,  // 10 minutes
    apiKeySecretName: apiKeySecretName,
  });

  local validateRoute = common.Route({
    namespace: ns.metadata.name,
    serviceName: validateSvc.metadata.name,
    servicePaths: ['/api/text_to_speech/input_validated'],
    apiKey: apiKey,
  });

  local streamRoute = common.Route({
    namespace: ns.metadata.name,
    serviceName: streamSvc.metadata.name,
    servicePaths: ['/api/text_to_speech/stream'],
    apiKey: apiKey,
  });

  // Note: prune simply removes the apiKeySecret entry if null
  std.prune([
    ns,
    apiKeySecret,
    validateSvc,
    streamSvc
  ] + validateRoute + streamRoute)
