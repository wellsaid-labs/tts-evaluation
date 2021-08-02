# Build

This document contains the steps required to build our TTS worker image. This
image can be ran and tested locally, or deployed and ran in our cloud
environment. The source code for this image is located in the `run/deploy`
directory.

## Building the Docker image

```bash
PROJECT_ID="voice-service-2-313121"
CHECKPOINTS="" # Example: v9
TTS_PACKAGE_PATH=$(python -m run.deploy.package_tts $CHECKPOINTS)
docker build -f run/deploy/Dockerfile \
    --build-arg TTS_PACKAGE_PATH=${TTS_PACKAGE_PATH} \
    -t gcr.io/${PROJECT_ID}/speech-api-worker:v9.00 .
```

Check the worker image size:

```bash
docker images gcr.io/${PROJECT_ID}/speech-api-worker:v9.00
```

## Running the Docker image locally

This is useful for testing a build prior to deployment. However, be aware of the
resources available to docker and how that may limit the performance of this
image. For reference, see the resource requests in
[tts.jsonnet](/ops/run/tts.jsonnet).

```bash
docker run --rm -p 8000:8000 -e "YOUR_SPEECH_API_KEY=123" \
  gcr.io/${PROJECT_ID}/speech-api-worker:v9.00
```

## Pushing the Docker image

Prior to pushing the docker image, ensure the proper GKE context is set:

```bash
gcloud config set project $PROJECT_ID
gcloud config set container/cluster $CLUSTER_NAME
gcloud container clusters get-credentials $CLUSTER_NAME --region=us-central1
# Sanity check
kubectl config current-context
```

Note that our Docker image repository is available to the entire Google Cloud
Project, so the `$CLUSTER_NAME` is somewhat irrelevant with respect to pushing
the local image to the remote repository.

Push the local image to our remote repository:

```bash
docker push gcr.io/${PROJECT_ID}/speech-api-worker:v9.00
```

Viewing a list of images in the remote repository:

```bash
gcloud container images list --repository=gcr.io/${PROJECT_ID}
```

Viewing a list of image tags:

```bash
gcloud container images list-tags gcr.io/${PROJECT_ID}/speech-api-worker
```

## Next Steps

Once the Docker image has been pushed to the remote registry, we can reference
that image in our cloud deployments. To deploy this image in our cloud
environment please follow the official deployment documentation in
[ops/run/README.md](/ops/run/README.md).
