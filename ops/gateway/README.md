# Kong Gateway

This directory contains the configuration and documentation around managing our Kong gateway;
a service which routes traffic to our internal Cloud Run containers.

TODO: reference/resources for quick start using kong
TODO: embed architecture diagram (probably in ../README.md)
## Prerequisites

This document assumes the following dependencies have been installed and [cluster setup](../ClusterSetup.md) has been completed. Additionally, you may need [authorize docker](https://cloud.google.com/container-registry/docs/advanced-authentication) in order to push images to the cloud registry.

- [gcloud](https://cloud.google.com/sdk/docs/quickstart)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- [helm@v3](https://helm.sh/docs/intro/install/)

## Kong Gateway Configuration

Our current Kong setup leverages the kong [Kubernetes Ingress Controller](https://docs.konghq.com/kubernetes-ingress-controller/).
The controller listens for changes to our kubernetes resources and updates the Kong service
accordingly. This allows us to run Kong in a db-less manor, meaning all of the configurations
are defined using kubernetes manifests and custom resource definitions.

### Configuring the base Kong docker image

The base [Kong image](https://hub.docker.com/_/kong) provides several bundled plugins by default. In order to add custom or third-party plugins to our deployment, we need to rebuild the image to include the plugin _and_ add that plugin to our [kong configuration](./kong/kong.yaml). See [kong plugin distribution](https://docs.konghq.com/gateway-oss/1.0.x/plugin-development/distribution/) for more details.

1. Setup env variables for image tagging

    ```bash
    export PROJECT_ID=voice-service-2-313121
    export ENV=$ENV # example: staging
    export KONG_IMAGE_TAG="gcr.io/$PROJECT_ID/kong:wellsaid-$ENV"
    ```

1. Build and tag the docker image locally

   ```bash
   docker build -t $KONG_IMAGE_TAG ./ops/gateway/kong
   ```

1. Push the image to our cloud registry

   ```bash
   docker push $KONG_IMAGE_TAG
   ```

   And confirm the tagged image exists:

   ```bash
   docker image ls gcr.io/$PROJECT_ID/kong
   ```

1. Now that the image exists in our remote registry, we need to update our deployment configuration to properly reference this image. Update the corresponding `kong.$ENV.yaml` like so:

    ```yaml
    image:
      repository: gcr.io/voice-service-2-313121/kong
      tag: wellsaid-$ENV
    ```

### Configuring the Kong service

### Configuring the Kubernetes Ingress Controller


## Kong Gateway Deployment

TODO: sanity check: `kubectl config current-context`

Useful commands:

```bash
# List deployments managed by helm
helm list
```

### Deploying the Kong Gateway

The following will deploy the configured Kong gateway along with the Kong
Ingress Controller.

```bash
# We deploy the kong gateway into the kong namespace
kubectl create namespace kong
# Install helm chart, referencing our configurations (order of file paths is important!)
helm install gateway kong/kong \
  --version 2.1.0 \
  -f ./ops/gateway/kong/kong.base.yaml \
  -f ./ops/gateway/kong/kong.$ENV.yaml
```

Let's confirm the proxy is live on our cluster.

```bash
# Display the kong proxy service
kubectl get service gateway-kong-proxy
# Using, the External IP from above command, make a request to the proxy
curl -XGET http://$KONG_IP
```

At this point we should get back a 404 response with the following message:
`{"message":"no Route matched with those values"}`. Kong is deployed, but we
have yet to configure any routes/services.

### Deploying the `google-logging` plugin

Logging is currently handled via the [google-logging](https://github.com/SmartParkingTechnology/kong-google-logging-plugin) Kong plugin.

```bash
cd ops/gateway/kong/plugins/
```

1. Setup a service account to allow this plugin to write logs directly to Google Cloud Logging.

    1. Create the Service Account

        ```bash
        gcloud iam service-accounts create kong-google-logging \
          --description="Service account used for writing detailed logs directly from our Kong proxy" \
          --display-name="kong-google-logging"
        ```

    1. Grant the Service Account an IAM Role that will allow writing to Google Cloud Logging.

        ```bash
        gcloud projects add-iam-policy-binding voice-service-2-313121 \
          --member="serviceAccount:kong-google-logging@voice-service-2-313121.iam.gserviceaccount.com" \
          --role="roles/logging.logWriter"
        ```

    1. Generate the Service Account Key that the `kong-google-logging` plugin will use to
       authenticate with.

        ```bash
        gcloud iam service-accounts keys create ./kong-google-logging-service-account.secrets.json \
        --iam-account=kong-google-logging@voice-service-2-313121.iam.gserviceaccount.com
        ```

1. Deploy the `google-logging` plugin. Note that the configuration for this plugin includes the
   service account credentials which are stored in a `Secret`.

   ```bash
   jsonnet google-logging.jsonnet \
      -y \
      --tla-str location=us-central1 \
      --tla-str cluster=staging \
      | kubectl apply -f -
   ```

1. Confirm that this plugin has been successfully picked up by the kong ingress controller.

  ```bash
  # Fetch a kong gateway pod
  kubectl get pods -n kong
  # Monitor logs on the `ingress-controller` container. These logs will tell us if the resources
  # deployed in the previous step were synced properly by kong
  kubectl logs POD_NAME -n kong -c ingress-controller --tail=20 --follow
  ```

  Finally, once the plugin is synced by kong we can start reading logs!

  ```bash
  gcloud logging read "labels.source=kong-google-logging" --limit=1
  ```

## Kong Gateway Management

### Updating our Kong Gateway configuration

```bash
helm upgrade gateway kong/kong \
  --version 2.1.0 \
  -f ./ops/gateway/kong/kong.base.yaml \
  -f ./ops/gateway/kong/kong.$ENV.yaml
```

### Logging and Metrics

TODO

### Debugging Kong Ingress Controller

The Kong Ingress Controller is in charge of responding to changes in kubernetes
resources and updating kong with the derived configurations (services/routes/plugins). At some point it may be helpful to see what the Kong proxy configuration looks
like.

```bash
# Grab kong gateway pod name
kubectl get pods -n kong
# Grab shell (note the container!)
kubectl exec -it POD_NAME -n kong -c ingress-controller -- /bin/sh
# Run the ingress controller with dump-config flag. This will output config files
# into the /tmp directory. Wait for the 'syncing configuration` output.
/kong-ingress-controller --dump-config=enabled
# output the configuration
cat /tmp/controller.../last_good.json
```
