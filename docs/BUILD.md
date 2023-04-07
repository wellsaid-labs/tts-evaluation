# Build

This document contains the steps required to build our TTS worker image. This
image can be ran and tested locally, or deployed and ran in our cloud
environment. The source code for this image is located in the `run/deploy`
directory.

## Prerequisite

You may need to update some files, in particular:

- You may need to add any new speakers to `run/deploy/worker.py` if needed.
- You may need to update `run/deploy/Dockerfile` with any new dependencies.

## Building the Docker image

_Naming Convention_: voiceModel.deployment.version # Example v9.viacom.00
_Hint_: See precedence for image tagging by running

```bash
gcloud container images list-tags gcr.io/${PROJECT_ID}/speech-api-worker
```

Set image variables and build Docker image:

```bash
PROJECT_ID="voice-service-2-313121"
CHECKPOINTS="" # Example: "v9" (see list of Checkpoints in [run/_tts.py](/run/_tts.py)])
TTS_PACKAGE_PATH=$(python -m run.deploy.package_tts $CHECKPOINTS)
IMAGE_TAG="" # Example: v9.00

docker build -f run/deploy/Dockerfile \
    --build-arg TTS_PACKAGE_PATH=${TTS_PACKAGE_PATH} \
    -t gcr.io/${PROJECT_ID}/speech-api-worker:${IMAGE_TAG} .
```

Check the worker image size:

```bash
docker images gcr.io/${PROJECT_ID}/speech-api-worker:${IMAGE_TAG}
```

## Running the Docker image locally

This is useful for testing a build prior to deployment. However, be aware of the
resources available to docker and how that may limit the performance of this
image. For reference, see the resource requests in
[tts.jsonnet](/ops/run/tts.jsonnet).

```bash
docker run --rm -p 8000:8000 \
  gcr.io/${PROJECT_ID}/speech-api-worker:${IMAGE_TAG}
```

```bash
curl http://localhost:8000/api/text_to_speech/stream \
  -H "Content-Type: application/json" \
  -X POST --data '{"speaker_id":"4","text":"Lorem ipsum"}' \
  -o sample.mp3
```

If this fails and you need to rebuild your Docker image, first remove the tag and delete the image:

```bash
docker rmi gcr.io/${PROJECT_ID}/speech-api-worker:${IMAGE_TAG}
```

Then fix, and rebuild from the top ^^

## Pushing the Docker image

Prior to pushing the docker image, ensure the proper GKE context is set:

```bash
CLUSTER_NAME="" # Example: "staging"
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
docker push gcr.io/${PROJECT_ID}/speech-api-worker:${IMAGE_TAG}
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
