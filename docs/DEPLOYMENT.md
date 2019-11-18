# Kubernetes Deployment

## Synopsis

These steps go over deploying the service at `src/service/` to GKE. This service creates a scalable
endpoint to run our TTS model.

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
   PROJECT_ID="voice-service-255602"
   docker build -f docker/master/Dockerfile -t gcr.io/${PROJECT_ID}/speech-api:v3.02 .
   docker build -f docker/worker/Dockerfile -t gcr.io/${PROJECT_ID}/speech-api-worker:v3.02 .
   ```

1. Push the build:

   ```bash
   docker push gcr.io/${PROJECT_ID}/speech-api:v3.02
   docker push gcr.io/${PROJECT_ID}/speech-api-worker:v3.02
   ```

1. Test the build:

   ```bash
   docker run --rm -p 8000:8000 -e "YOUR_SPEECH_API_KEY=123" \
      gcr.io/${PROJECT_ID}/speech-api-worker:v3.02
   ```

   Or:

   ```bash
   docker run --rm -p 8000:8000 -e "AUTOSCALE_LOOP=5000 YOUR_SPEECH_API_KEY=123" \
      gcr.io/${PROJECT_ID}/speech-api:v3.02
   ```

1. Update the Kubernetes deployment manifest (e.g. `src/service/deployment.yaml`) with the updated
   images.
1. Update the Kubernetes deployment with:

   ```bash
   kubectl apply -f src/service/deployment.yaml
   ```

### Update Container from GCP Machine

Similar to the above, except:

- Docker will need to be installed like so:
  https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-18-04
- For authentication reasons, the build should be pushed with the `gcloud` tool:

  ```bash
  sudo gcloud docker -- push gcr.io/${PROJECT_ID}/speech-api-worker:v3.02
  ```

  Learn more here: https://cloud.google.com/container-registry/docs/advanced-authentication

## Staging Namespace

These steps will allow you to setup a staging `namespace` to a test Kubernetes setup. Also
these deployment steps are loosely based on the below "New Cluster" guide.

1. Create a `staging` namespace like so:

   ```bash
   kubectl create namespace staging
   ```

1. Permanently save the namespace for all subsequent `kubectl` commands in that context:

   ```bash
   kubectl config set-context --current --namespace=staging
   ```

1. Follow the instructions in "New Cluster" to add secrets for the new namespace.
1. Follow the instructions in "New Cluster" to add permissions. For `Role` and `RoleBinding`
   resources, you'll need to update the `namespace` manifest configurations. For
   `ClusterRoleBinding` resources, you'll need to add to the `subjects` configuration, like so:

   ```bash
    subjects:
    ...
    - kind: ServiceAccount
      name: default
      namespace: staging
    ...
   ```

1. Update `public/script.js` to remove any absolute paths to voice.wellsaidlabs.com or
   voice2.wellsaidlabs.com in favor of relative paths.

1. Update the deployment manifest `namespace` configuration in `src/service/deployment.yaml` to
   `staging`. Then apply the deployment manifest, creating a `Deployment`.

   ```bash
   kubectl apply -f src/service/deployment.yaml
   ```

1. Expose the staging environment like so:

   ```bash
   kubectl expose deployment speech-api --type=LoadBalancer
   ```

1. Get the external IP address via:

   ```bash
   kubectl get services
   ```

   If the external IP address is shown as `<pending>`, wait for a minute and enter the same command
   again. Finally you can construct the URL like so: `http://<EXTERNAL-IP>:<PORT>/`.

1. Delete the namespace after your done using it, like so:

   ```bash
   kubectl delete namespaces staging

   # Reset the context
   kubectl config set-context --current --namespace=default
   ```

   You may need to delete any remaining nodes separately because they are no tied to the namespace.

## New Cluster

These deployment steps are loosely based on these guides below:

- https://cloud.google.com/kubernetes-engine/docs/tutorials/hello-app
- https://cloud.google.com/kubernetes-engine/docs/tutorials/http-balancer

Refer to the above guides in case there are missing details in the below steps.

1. A cluster consists of a pool of Compute Engine VM instances running Kubernetes, the open source
   cluster orchestration system that powers GKE. Create a cluster by following these steps:
   1. Navigate to GCPs `Create a Kubernetes cluster` page.
   1. Pick the `Standard cluster` template on the left hand side.
   1. Create a `Node pool` for worker (i.e. 'worker-pool') and master nodes (i.e. 'master-pool').
      The worker and master likely do not need default 100GB disk size. The worker likely has an
      optimized architecture that it runs most quickly on. The worker pool will need autoscaling
      enabled with 1 node minimum and a maximum of 100 nodes.
   1. Pick the required computer resources for this deployment.
1. Assuming GKE is installed on your system. Log into your cluster via:

   ```bash
   gcloud container clusters get-credentials yourclustername --zone=yourclusterzone
   ```

1. Set up permissions via RBAC:

   1. To run the next step, your account will need to be a `cluster-admin`:

      ```bash
      kubectl create clusterrolebinding your-name-cluster-admin-binding \
          --clusterrole=cluster-admin \
          --user=your@email.com
      ```

   2. Give permission to the service to create `Pods` via:

      ```bash
      kubectl apply -f src/service/rbac.yaml
      ```

1. Set up API Keys. Our API Keys are stored on Kubernetes as secrets. Create them via:

   ```bash
   kubectl create secret generic speech-api-key \
     --from-literal=matt_hocking_api_key='' \
     --from-literal=michael_petrochuk_api_key='' \
     --from-literal=website_backend_api_key=''
   ```

1. Deploy the web application to Kubernetes.

   1. Apply the deployment manifest, creating a `Deployment`.

      ```bash
      kubectl apply -f src/service/deployment.yaml
      ```

   This command assumes that the required images were built and pushed similar to
   `Update Container`.

   1. Check that the deployment is working by inspecting logs like so:

      ```bash
      kubectl get pods
      kubectl logs -f some_pod_name
      ```

1. Create a `Service` resource to make the web deployment reachable within your container cluster.

   ```bash
   kubectl expose deployment speech-api --target-port=8000 --type=NodePort
   ```

1. Create an `Ingress`, a resource that encapsulates a collection of rules and configurations for
   routing external HTTP(S) traffic to internal services. On GKE, Ingress is implemented using
   Cloud Load Balancing. When you create an Ingress in your cluster, GKE creates an HTTP(S) load
   balancer and configures it to route traffic to your application.

   1. Reserve a static address
      [here](https://console.cloud.google.com/networking/addresses/add?project=mythical-runner-203817).
      Ensure that the static IP address is of type "Global (to be used with Global forwarding
      rules)".
   1. Change `kubernetes.io/ingress.global-static-ip-name` in `src/service/ingress.yaml` to
      your static IP address name.
   1. Apply the ingress manifest, creating a `Ingress`.

      ```bash
      kubectl apply -f src/service/ingress.yaml
      ```

      Wait 30 minutes. On the GKE frontend, it'll display a status for ingress creation.

   1. The speech API tends to run jobs that can take upwards of 15 minutes or more. By default,
      the GKE load balancer has a timeout for requests that is much smaller. To adjust that you
      need to:
      1. On a web browser, navigate to the ingress service in the GKE frontend landing on a
         page called "Ingress Details".
      1. Within the details, there is a section titled "Ingress". Within the section, click the
         link to the right of "Backend services".
      1. On the top nav bar select "edit", edit the timeout to 3600 (1 hour).
      1. Wait up to 10 minutes to the changes to take effect.
   1. It is important to set up a secure HTTPS endpoint, here are the steps to do so (this can
      be done multiple times for additional domains):
      1. On a web browser, navigate to the ingress service in the GKE frontend landing on a
         page called "Ingress Details".
      1. Within the details, there is a section titled "Ingress". Within the section, click the
         link to the right of "Load balancer".
      1. On the top nav bar select "edit", then on the left panel click "Frontend configuration".
      1. On the right side click "+ Add Frontend IP and port".
      1. Configure the "New Frontend IP and port" with 443 for port, HTTPS for the protocol,
         some HTTPS certificate, and some IP address.
      1. Wait up to 5 minutes to the changes to take effect.
