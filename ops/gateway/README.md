# Kong Gateway

This directory contains the configuration and documentation around managing our
Kong gateway; a service which routes traffic to our internal TTS services.

Visit the
[Getting Started with Kong Gateway](https://docs.konghq.com/getting-started-guide/2.4.x/overview/)
guide for a better understanding of Kong-related concepts.

## Prerequisites

This document assumes the following dependencies have been installed and
[cluster setup](../ClusterSetup.md) has been completed. Additionally, you may
need
[authorize docker](https://cloud.google.com/container-registry/docs/advanced-authentication)
in order to push images to the cloud registry.

- [gcloud](https://cloud.google.com/sdk/docs/quickstart)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- [helm@v3](https://helm.sh/docs/intro/install/)

## Kong Gateway Configuration

Our current Kong setup leverages the kong
[Kubernetes Ingress Controller](https://docs.konghq.com/kubernetes-ingress-controller/).
The controller listens for changes to our kubernetes resources and updates the
Kong service accordingly. This allows us to run Kong without a database, meaning
all of the configurations are defined by Kubernetes manifests in this codebase.

### Configuring the base Kong docker image

The base [Kong image](https://hub.docker.com/_/kong) provides several bundled
plugins by default. In order to add custom or third-party plugins to our
deployment, we need to rebuild the image to include the plugin _and_ add those
plugins to our [kong configuration](./kong/kong.yaml). See
[kong plugin distribution](https://docs.konghq.com/gateway-oss/1.0.x/plugin-development/distribution/)
for more details.

1. Ensure the custom plugins exist locally (via git submodules)

   ```bash
   git submodule update --init --recursive
   ```

1. Setup env variables for image tagging.

   ```bash
   ENV=$ENV # ex: staging
   PROJECT_ID=voice-service-2-313121
   KONG_IMAGE_TAG=$KONG_IMAGE_TAG # ex: v1
   KONG_IMAGE="gcr.io/$PROJECT_ID/kong:$KONG_IMAGE_TAG"
   ```

   ```bash
   # List existing tags for the kong image, useful for incrementing the
   # `$KONG_IMAGE_TAG` off a previous tag.
   gcloud container images list-tags gcr.io/$PROJECT_ID/kong
   ```

1. Build and tag the docker image locally

   ```bash
   docker build -t $KONG_IMAGE ./ops/gateway/kong
   ```

1. Push the image to our cloud registry

   ```bash
   docker push $KONG_IMAGE
   ```

   And confirm the tagged image exists:

   ```bash
   docker image ls gcr.io/$PROJECT_ID/kong
   ```

1. Now that the image exists in our remote registry, we need to update our
   deployment configuration to properly reference this image. Update the
   corresponding `kong.$ENV.yaml` like so:

   ```bash
   # Grab the image digest
   docker inspect \
        $KONG_IMAGE \
        --format="{{index .RepoDigests 0}}"
   ```

   ```yaml
   image:
     repository: gcr.io/voice-service-2-313121/kong@sha256
     tag: <IMAGE_DIGEST>
   ```

### Configuring the Kong deployment

The configuration for our Kong deployment is located in the `kong.base.yaml` and
`kong.$ENV.yaml` files, where the `kong.$ENV.yaml` file provides environment
specific configuration overrides to the base helm chart config. The first update
you will want to make is to configure the static IP used by the proxy. If you
have not setup the static IP yet, see
[Reserving a static IP](../ClusterSetup.md).

```yaml
proxy:
  loadBalancerIP: <STATIC_IP_GOES_HERE>
```

### Configuring the Kubernetes Ingress Controller

## Kong Gateway Deployment

Useful commands:

```bash
# List deployments managed by helm
helm list
```

### Deploying the Kong Gateway

The following will deploy the configured Kong gateway along with the Kong
Ingress Controller.

```bash
# Add repository so we can reference the kong/kong helm chart
helm repo add kong https://charts.konghq.com
# Namespace required for helm install, see `namespace` argument in
# `./kong/kong.base.yaml`
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
kubectl get service gateway-kong-proxy -n kong
# Using the External IP from above command, make a request to the proxy
# Note that this should be the static ip reserved during the cluster setup
curl -XGET http://$KONG_IP
```

At this point we should get back a 404 response with the following message:
`{"message":"no Route matched with those values"}`. Kong is deployed, but we
have yet to configure any routes/services.

### Deploying the `file-log` plugin

We leverage the [file-log](https://docs.konghq.com/hub/kong-inc/file-log/)
plugin in order to write request information to stdout which is picked up by
Stackdriver.

```bash
kubectl apply -f ./ops/gateway/kong/plugins/file-log.yaml
```

### Deploying the `latest-version-transformation` plugin

This plugin allows us to "pin" all traffic containing the
`Accept-Version: latest` header to a specific model release. Note that
`$LATEST_VERSION` refers to a model name used during the tts deployment (see
[../run/README.md](../run/README.md)).

```bash
jsonnet ./ops/gateway/kong/plugins/latest-pinned-service.jsonnet \
  -y \
  --tla-str latestVersion=$LATEST_VERSION \
  | kubectl apply -f -
```

Additionally, once deployed, you can fetch the latest release using the
following command

```bash
kubectl get kongclusterplugin.configuration.konghq.com \
  latest-pinned-service -n kong -o jsonpath='{.config.latest_version}'
```

### Deploying the `cert-manager` and setting up TLS

See [TLS Certificate issuance for HTTPS](../tls/README.md).

### Deploying a fallback/catch-all route

Note that this Ingress resource is also responsible for requesting a TLS
Certificate via the `cert-manager`, see the `fallback-route` Ingress resource
annotations!

```bash
jsonnet ./ops/gateway/kong/plugins/fallback-route.jsonnet \
  -y \
  --tla-str env=$ENV \
  | kubectl apply -f -
```

### Deploying the `key-auth` plugin

The [`key-auth`](https://docs.konghq.com/hub/kong-inc/key-auth/) plugin allows
us to secure our API using API key authorization. It is possible to restrict
access on a route, service, or consumer basis.

```bash
kubectl apply -f ./ops/gateway/kong/plugins/key-auth.yaml
```

Once applied, any requests to the gateway will now fail with a 401 Unauthorized
response.

## Kong Gateway Management

### Updating our Kong Gateway configuration

At some point we may want to update our kong configurations (proxy configs,
scaling, resource requirements, etc..). Similar to installing kong, we will also
use `helm` to "upgrade" the release.

```bash
helm upgrade gateway kong/kong \
  --version 2.1.0 \
  -f ./ops/gateway/kong/kong.base.yaml \
  -f ./ops/gateway/kong/kong.$ENV.yaml
```

It may be helpful to see the configured values for a previous release. Omit the
`--all` flag if you just want to see our user-defined configurations for the
current release.

```bash
helm get values gateway --all
```

In the event that we need to rollback a release (for example, due to a bad
configuration) we can easily do that via helm.

```bash
# Find the current revision number for the `gateway` release
helm list
# Rollback to a previous version
helm rollback gateway <REVISION_NUMBER>
```

### Kong Consumers

Once the `key-auth` plugin is enabled, a `KongConsumer` is required to make
authenticated requests to the API. For reference, see
[Provisioning a consumer](https://docs.konghq.com/kubernetes-ingress-controller/1.3.x/guides/using-consumer-credential-resource/#provision-a-consumer).

```bash
# We will be namespacing all of the kong consumers
kubectl create namespace kong-consumers
# Export the username. For example, we use the username `studio` as the consumer
# of our Studio product, and `api` as the developer-facing api gateway. For local
# consumer credentials, you might us `johndoe-local`.
KONG_CONSUMER_USERNAME=$KONG_CONSUMER_USERNAME
# Create the consumer!
jsonnet ./ops/gateway/kong/auth/consumer.jsonnet \
  -y \
  --tla-str username=$KONG_CONSUMER_USERNAME \
  --tla-str secretKey="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" \
  | kubectl apply -f -
# Once the above resources are created, use the following command can be used
# to display the auto generated secret (api key) for the new consumer.
kubectl get secrets/$KONG_CONSUMER_USERNAME-consumer-apikey \
  --template={{.data.key}} \
  -n kong-consumers \
  | base64 -D
```

At this point you should be able to test that the new consumer credentials are
working

```bash
# Should respond with 404 Not Found
curl -I http://$KONG_IP
```

Listing existing KongConsumers

```bash
kubectl get kongconsumers.configuration.konghq.com -n kong-consumers
```

Delete an existing consumer

```bash
# Delete the KongConsumer
kubectl delete kongconsumers.configuration.konghq.com/$KONG_CONSUMER_USERNAME-consumer -n kong-consumers
# Delete the secret
kubectl delete secret/$KONG_CONSUMER_USERNAME-consumer-apikey -n kong-consumers
```

### Logging and Metrics

Once the `file-log` plugin has been configured, the Kong proxy will write
detailed logs to stdout. These logs are
[automatically picked up by Stackdriver](https://cloud.google.com/stackdriver/docs/solutions/gke/managing-logs#what_logs).
View [instructions](../metrics/README.md) for enabling log-based metrics.

### Debugging Kong Ingress Controller

The Kong Ingress Controller is in charge of responding to changes in kubernetes
resources and updating kong with the derived configurations
(services/routes/plugins). At some point it may be helpful to see what the Kong
proxy configuration looks like.

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

### Debugging Kong Custom Resource Definitions (CRDs)

```bash
# list kong crds
kubectl get crd | grep kong
# list globally applied plugins
kubectl get kongclusterplugins.configuration.konghq.com
# list scoped plugins
kubectl get kongplugins.configuration.konghq.com --all-namespaces
# list consumers
kubectl get kongconsumers.configuration.konghq.com -n kong-consumers
```