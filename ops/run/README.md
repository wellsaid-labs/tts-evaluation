# Runtime Service Configuration

This directory contains code for deploying the TTS service. The service is
deployed to a [GKE Cluster](https://cloud.google.com/kubernetes-engine),
and orchestrated via [Cloud Run](https://cloud.google.com/run).

## Prerequisites

You'll need the following installed:

- [gcloud](https://cloud.google.com/sdk/docs/quickstart)
- [jsonnet](https://github.com/google/jsonnet#packages)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)

You'll also need access to the appropriate Google Cloud Project.

## High Level Architecture

The TTS service is a single Python application that:

- Streams audio from text that's provided via a request.
- Validates text that's to be converted to audio.

Two versions of the TTS service are deployed per model:

- One for streaming audio.
- One for validating input.

This decision was made because the resource needs and concurrency of these
endpoints are different. Deploying them separately allows us to adjust
their scaling factors to reduce cost and increase service resiliency.

Each TTS service instance isn't exposed to the public internet. Rather they
can only be queried from the cluster, using cluster-local DNS. The
DNS name for a service is of the following format:

```bash
# input validation
validate.${model}.svc.cluster.local

# audio streaming
stream.${model}.svc.cluster.local
```

For instance, if we were deploying a model whose identifier is `v3` we'd use
hostnames `validate.v3.svc.cluster.local` and `stream.v3.svc.cluster.local`,
respectively.

Public access will handled by a [Kong](https://konghq.com/) proxy instance.
That's not in place yet, so it's not documented.

## Building an Image

The TTS software is packaged via [Docker](https://docker.com). If
you already know the image you want to deploy you can skip this step.
Otherwise you'll need to build and push an image.

TODO: Document how to build and push the image(s).

## Deploying

To deploy a new version of a service, follow these steps:

1. Connect to the target cluster:

    ```bash
    gcloud container clusters get-credentials $cluster --region us-central1
    ```

1. Populate the api keys that should be used to access the endpoint.
   The API keys must be present in a file named `apikeys.json` that lives
   in the same directory as this README. WellSaidLabs should eventually store
   this file in something like [Google Secret Manager](https://cloud.google.com/secret-manager)
   and add the command for downloading it here.

   The contents of that file should look something like this:

    ```json
    {
        "SAMS_SPEECH_API_KEY": "XXX",
        "NEIL_SPEECH_API_KEY": "YYY"
    }
    ```

    Again, if you're updating an existing environment make sure you first download the
    existing API keys first. Otherwise you'll overwrite the ones that are there.

1. Deploy the version you'd like to release:

    ```bash
    jsonnet \
        -y tts.jsonnet \
        --tla-str env=$env \
        --tla-str model=$model \
        --tla-str version=$version \
        --tla-str image=$image \
        --tla-str includeImageApiKeys=false \
        --tla-str minScale=0 \
        --tla-str maxScale=32 \
        | kubectl apply -f -
    ```

    A few notes about the parameters:

    - `$env` refers to the cluster environment (currently `staging` or `prod`)
      and is mainly used for hostname configuration.

    - `$model` is a unique identifier for the model being deployed. It'll
      determine the on-cluster DNS name for the service, which will be
      `[stream|validate].$model.svc.cluster.local`. It can only contain
      lowercase alphanumeric characters and dashes.

    - `$version` is a unique identifier for the revision being released.
      It can only include lowercase alphanumeric characters and dashes,
      so we choose to use a simple monotonic integer. So if it's the first
      time it's being released us `1`, then `2` and so on.

    - `$image` is the fully qualified image to deploy. You should use an
      [image digest](https://cloud.google.com/architecture/using-container-images)
      instead of a tag, since they're immutable. If you know the tag you'd
      like to release you can use the command below to determine the
      digest:

      ```bash
      docker inspect \
        gcr.io/voice-service-2-313121/speech-api-worker:latest \
        --format="{{index .RepoDigests 0}}"
      ```

      The result should look something like this:

      ```bash
      gcr.io/voice-service-2-313121/speech-api-worker@sha256:3af2c7a3a88806e0ff5e5c0659ab6a97c42eba7f6e5d61e33dbc9244163e17d3
      ```

    - `$imageEntrypoint` is a string value that references the python service entry point.
      Currently, this is for legacy image support (images prior to v9 that required a different entry).

    - `$includeImageApiKeys` is a boolean flag that determines whether or not to
      inject api keys into the proxied upstream request. This is only for backwards
      compatibility, enabled support of our existing tts worker images.

    - `$minScale` is an integer value determining how many container instances
      should remain idle, see the [cloud run configuration](https://cloud.google.com/run/docs/configuring/min-instances) for more details. For our staging environment and low demand model versions we will want to scale to 0. Note that changes to this value will require a bump in the `version` parameter.

    - `$maxScale` is an integer value that puts a ceiling on the number of container
      instances we can scale to. This may be useful for preventing over-scaling in
      response to a spike in requests and/or managing costs.

1. After running the command you can see the status of what was deployed via
   this command:

    ```bash
    kubectl get --namespace $model service.serving.knative.dev
    ```

    If something didn't work, you can investigate by first listing all
    resources:

    ```bash
    kubectl get --namespace $model all
    ```

    Then depending on the output use commands like `kubectl describe`
    and `kubectl logs` to further debug things.

## Management

### Pinning `latest` release

```bash
jsonnet ./ops/gateway/kong/plugins/latest-pinned-service.jsonnet \
  -y \
  --tla-str latestVersion=$LATEST_VERSION \
  | kubectl apply -f -
```

Refer to [Deploying the `latest-version-transformation` plugin](../gateway/README.md)
for additional details.

### Updating an existing release

This scenario is likely in the event we need to update a model image or change
the scaling configuration of an existing deployment. It is recommended to use
the `--dry-run=server` flag as a sanity check to see which resources will be
updated by this command. Note that changes to the cloud run service will require
a bump in the `version` parameter. For example, if this model was released with
`version=1` and then we want to update that release to modify the scaling
parameters, you would run this command with `version=2`.

```bash
jsonnet \
  -y tts.jsonnet \
  --tla-str env=$env \
  --tla-str model=$model \
  --tla-str version=$version \
  --tla-str image=$image \
  --tla-str includeImageApiKeys=false \
  --tla-str minScale=0 \
  --tla-str maxScale=32 \
  | kubectl apply --dry-run=server -f -
```

### Deleting a release

Since all resources related to a model release are namespaced we can simply
delete the namespace.

```bash
kubectl delete namespace $model
```
