# Runtime Service Configuration

This directory contains code for deploying the TTS service. The service is
deployed to a [GKE Cluster](https://cloud.google.com/kubernetes-engine), and
orchestrated via [Cloud Run](https://cloud.google.com/run).

## Prerequisites

You'll need the following installed:

- [gcloud](https://cloud.google.com/sdk/docs/quickstart)
- [jsonnet](https://github.com/google/jsonnet#packages)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)

You'll also need access to the appropriate Google Cloud Project. Prior to
running the deploy script ensure that your gcloud/kubectl context is correct.

```bash
CLUSTER_NAME="" # Example: "staging"
PROJECT_ID="voice-service-2-313121"
gcloud config set project $PROJECT_ID
gcloud config set container/cluster $CLUSTER_NAME
gcloud container clusters get-credentials $CLUSTER_NAME --region=us-central1
# Sanity check
kubectl config current-context
```

## High Level Architecture

The TTS service is a single Python application that:

- Streams audio from text that's provided via a request.
- Validates text that's to be converted to audio.

Two versions of the TTS service are deployed per model:

- One for streaming audio.
- One for validating input.

This decision was made because the resource needs and concurrency of these
endpoints are different. Deploying them separately allows us to adjust their
scaling factors to reduce cost and increase service resiliency.

Each TTS service instance isn't exposed to the public internet. Rather they can
only be queried from the cluster, using cluster-local DNS. The DNS name for a
service is of the following format:

```bash
# input validation
validate.${model}.svc.cluster.local

# audio streaming
stream.${model}.svc.cluster.local
```

For instance, if we were deploying a model whose identifier is `v3` we'd use
hostnames `validate.v3.svc.cluster.local` and `stream.v3.svc.cluster.local`,
respectively.

Public access will handled by via a
[Kong gateway service](../gateway/README.md). Before proceeding, ensure that the
gateway has been deployed to the cluster. The TTS service deployment has
kong-related configurations in order to properly register and route requests
through Kong to our internal Cloud Run instances.

## Building an Image

The TTS software is packaged via [Docker](https://docker.com). If you already
know the image you want to deploy you can skip this step. Otherwise you'll need
to build and push an image. These steps can be found in
[docs/BUILD.md](/docs/BUILD.md).

## Deploying

For deployment purposes, we will consider a new version of our service to be a
'model' release, with minor adjustments to that model release being 'version'
releases. To deploy a new model, follow these steps:

1. Connect to the target cluster:

   ```bash
   CLUSTER_NAME="" # Example: "staging"
   gcloud container clusters get-credentials $CLUSTER_NAME --region us-central1
   ```

1. Create the configuration for the new model release, ex:
   `deployments/staging/v9.config.json`. These configurations map directly to
   arguments found in `tts.jsonnet`.

   ```json
   {
     "env": "staging",
     "model": "v9",
     "version": "1",
     "image": "gcr.io/voice-service-2-313121/speech-api-worker@sha256:c5de71f13aff22b9171f23f9921796f93e1765fefa9ba5ca6426836696996a75",
     "imageEntrypoint": "run.deploy.worker:app",
     "provideApiKeyAuthForLegacyContainerSupport": "true",
     "stream": {
       "minScale": 0,
       "maxScale": 32,
       "concurrency": 1,
       "paths": ["/api/text_to_speech/stream"]
     },
     "validate": {
       "minScale": 0,
       "maxScale": 32,
       "concurrency": 4,
       "paths": ["/api/text_to_speech/input_validated"]
     },
     "traffic": [
       {
         "tag": "1",
         "percent": 100
       }
     ]
   }
   ```

   A few notes about the parameters:

   - `env` refers to the cluster environment (currently `staging` or `prod`) and
     is mainly used for hostname configuration.

   - `model` is a unique identifier for the model being deployed. It'll
     determine the on-cluster DNS name for the service, which will be
     `[stream|validate].$model.svc.cluster.local`. It can only contain lowercase
     alphanumeric characters and dashes.

   - `version` is a unique identifier for the revision being released. It can
     only include lowercase alphanumeric characters and dashes, so we choose to
     use a simple monotonic integer. So if it's the first time it's being
     released us `1`, then `2` and so on.

   - `image` is the fully qualified image to deploy. You should use an
     [image digest](https://cloud.google.com/architecture/using-container-images)
     instead of a tag, since they're immutable. If you know the tag you'd like
     to release you can use the command below to determine the digest:

     ```bash
     docker inspect \
       gcr.io/voice-service-2-313121/speech-api-worker:latest \
       --format="{{index .RepoDigests 0}}"
     ```

     The result should look something like this:

     ```bash
     gcr.io/voice-service-2-313121/speech-api-worker@sha256:3af2c7a3a88806e0ff5e5c0659ab6a97c42eba7f6e5d61e33dbc9244163e17d3
     ```

   - `imageEntrypoint` is a string value that references the python service
     entry point. Currently, this is for legacy image support (images prior to
     v9 that required a different entry).

   - `provideApiKeyAuthForLegacyContainerSupport` is a boolean flag that
     determines whether or not to inject api keys into the proxied upstream
     request. This is only for backwards compatibility, enabling support of our
     existing tts worker images.

   - `stream.minScale|validate.minScale` is an integer value determining how
     many container instances should remain idle, see the
     [cloud run configuration](https://cloud.google.com/run/docs/configuring/min-instances)
     for more details. Considering our validation service can handle multiple
     concurrent short-lived requests it would make sense to have a smaller min
     scale value than the stream service. For our staging environment and low
     demand model versions we will want to scale to 0.

   - `stream.maxScale|validate.maxScale` is an integer value that puts a ceiling
     on the number of container instances we can scale to. This may be useful
     for preventing over-scaling in response to a spike in requests and/or
     managing costs.

   - `stream.concurrency|validate.concurrency` option defines the number of
     concurrent requests that this service can handle.

   - `stream.paths|validate.paths` option defines the http endpoints that the
     service will respond to.

1. Deploy the configured model release. If you would like to debug the release
   manifests prior to deployment, run the `jsonnet` command below

   ```bash
   cd ops/run
   ENV="" # Example: staging
   MODEL="" # Example: v9
   ./deploy.sh deployments/$ENV/$MODEL.config.json
   ```

   Optionally, if you need to debug or view the generated manifests:

   ```bash
   # (optional) dump the release manifests
   jsonnet -y tts.jsonnet \
     --ext-code-file "config=./deployments/$ENV/$MODEL.config.json"
   ```

1. After running the command you can see the status of what was deployed via
   this command:

   ```bash
   kubectl get --namespace $MODEL service.serving.knative.dev
   ```

   If something didn't work, you can investigate by first listing all resources:

   ```bash
   kubectl get --namespace $MODEL all
   ```

   Then depending on the output use commands like `kubectl describe` and
   `kubectl logs` to further debug things.

## Accessing the model

Once the model is deployed, it should be accessible via the Kong gateway
interface. You may need to follow the guide for
[generating a Kong Consumer api key](../gateway/README.md) in the gateway docs.

```bash
# Assumes staging environment
curl https://staging.tts.wellsaidlabs.com/api/text_to_speech/stream \
  -H "X-Api-Key: $API_KEY" \
  -H "Accept-Version: $MODEL" \
  -H "Content-Type: application/json" \
  -X POST --data '{"speaker_id":"4","text":"Lorem ipsum","consumerId":"id","consumerSource":"source"}' \
  -o sample.mp3
```

You may also access specific model revisions via the `Accept-Version` header.
For example, given the following deployment configuration:

```json
{
  "model": "v9",
  "version": "2",
  "traffic": [
    {
      "tag": "1",
      "percent": 100
    },
    {
      "tag": "2",
      "percent": 0
    }
  ]
  ...
}
```

The `Accept-Version` header will behave like:

```bash
Accept-Version: v9 # obeys the traffic policy, in this case routes to revision 1
Accept-Version: v9.1 # manually specify revision 1, ignoring traffic policy
Accept-Version: v9.2 # manually specify revision 2, ignoring traffic policy
```

## Management

### Viewing a list of Cloud Run revisions

The following command will list all Cloud Run revisions for a given model.

```bash
gcloud run revisions list \
  --platform=kubernetes \
  --namespace=$MODEL
```

### Pinning `latest` release

Kong will route any request with the `Accept-Version: latest` header to a
"pinned" model release. This is used for directing traffic from one major model
release to another (ex: v8 -> v9). For minor, incremental releases it is
recommended to use the Cloud Run revisioning feature (ie update the
configuration/version tag for that model and re-deploy).

```bash
# Fetching the currently pinned version
kubectl get kongclusterplugin.configuration.konghq.com \
  latest-pinned-service -n kong -o jsonpath='{.config.latest_version}'
# Updating the latest pinned version
jsonnet ./ops/gateway/kong/plugins/latest-pinned-service.jsonnet \
  -y \
  --tla-str latestVersion=$LATEST_VERSION \
  | kubectl apply -f -
```

Refer to
[Deploying the `latest-version-transformation` plugin](../gateway/README.md) for
additional details.

### Updating an existing release

**_Warning: avoid using the Cloud Console UI for deploying new Cloud Run
revisions. The UI currently fails to include all of our annotations and
configurations. You may, however, use the UI for managing revision traffic._**

This scenario is likely in the event we need to update a model image or change
the scaling configuration of an existing deployment. It is recommended to use
the `--dry-run=server` flag as a sanity check to see which resources will be
updated by this command. Note that changes to the cloud run service will require
a bump in the `version` parameter. For example, if this model was released with
`version=1` and then we want to update that release to modify the scaling
parameters, you would run this command with `version=2`.

1. Modify the relevant release configuration. This will involve bumping the
   `version` number.

1. Re-run the deploy script:

   ```bash
   # (optional) kubectl dry run
   jsonnet -y tts.jsonnet \
     --ext-code-file "config=./ops/run/deployments/$ENV/$MODEL.config.json" \
     | kubectl apply --dry-run=server -f -
   # deploy
   ./deploy.sh deployments/$ENV/$MODEL.config.json
   ```

### Deleting a release

Since all resources related to a model release are namespace'd we can simply
delete the namespace.

```bash
kubectl delete namespace $MODEL
```
