# Kubernetes Deployment

## Synopsis

These steps go over deploying the service at `run/deploy/` to GKE. This service
creates a scalable endpoint to run our TTS model.

## Update Container

These deployment steps are loosely based on these guides below:

- https://cloud.google.com/kubernetes-engine/docs/tutorials/hello-app

Refer to the above guides in case there are missing details in the below steps.

1. Assuming GKE is installed. Log into your cluster via:

   ```bash
   gcloud container clusters get-credentials yourclustername --zone=yourclusterzone
   ```

1. Build the container image:

   ```bash
   PROJECT_ID="voice-service-2-313121"
   CHECKPOINTS="" # Example: v9
   TTS_PACKAGE_PATH=$(python -m run.deploy.package_tts $CHECKPOINTS)
   docker build -f run/deploy/docker/master/Dockerfile -t gcr.io/${PROJECT_ID}/speech-api:v8.31 .
   docker build -f run/deploy/docker/worker/Dockerfile \
        --build-arg TTS_PACKAGE_PATH=${TTS_PACKAGE_PATH} \
        -t gcr.io/${PROJECT_ID}/speech-api-worker:v9.00 .
   ```

1. Check the worker image size:

   ```bash
   docker images gcr.io/${PROJECT_ID}/speech-api-worker:v9.00
   ```

   The image size should be around 750mb.

1. Push the build:

   ```bash
   docker push gcr.io/${PROJECT_ID}/speech-api:v8.31
   docker push gcr.io/${PROJECT_ID}/speech-api-worker:v9.00
   ```

1. Test the build:

   ```bash
   docker run --rm -p 8000:8000 -e "YOUR_SPEECH_API_KEY=123" \
      gcr.io/${PROJECT_ID}/speech-api-worker:v9.00
   ```

   Or:

   ```bash
   docker run --rm -p 8000:8000 -e "AUTOSCALE_LOOP=5000 YOUR_SPEECH_API_KEY=123" \
      gcr.io/${PROJECT_ID}/speech-api:v8.31
   ```

1. Update the Kubernetes deployment manifest (e.g. `run/deploy/deployment.yaml`)
   with the updated images.

1. Update the Kubernetes deployment with:

   ```bash
   kubectl apply -f run/deploy/deployment.yaml
   ```

### Update Container from GCP Machine

Similar to the above, except:

- Docker will need to be installed like so:
  https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-18-04
- For authentication reasons, the build should be pushed with the `gcloud` tool:

  ```bash
  sudo gcloud docker -- push gcr.io/${PROJECT_ID}/speech-api-worker:v9.00
  ```

  Learn more here:
  https://cloud.google.com/container-registry/docs/advanced-authentication
