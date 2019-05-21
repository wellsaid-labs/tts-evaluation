# Kubernetes Deployment

## Synopsis

These steps go over deploying the service at `src/service/` to GKE. This service creates a scalable
endpoint to run our TTS model.

## Update Container

These deployment steps are loosely based on these guides below:
- https://cloud.google.com/kubernetes-engine/docs/tutorials/hello-app

Refer to the above guides in case there are missing details in the below steps.

1. Build the container image:
   ```bash
   export PROJECT_ID="$(gcloud config get-value project -q)"
   docker build -f docker/master/Dockerfile -t gcr.io/${PROJECT_ID}/speech-api:v1.17 .
   docker build -f docker/worker/Dockerfile -t gcr.io/${PROJECT_ID}/speech-api-worker:v1.17 .
   ```
1. Push the build:
   ```bash
   docker push gcr.io/${PROJECT_ID}/speech-api:v1.17
   docker push gcr.io/${PROJECT_ID}/speech-api-worker:v1.17
   ```
1. Test the build:
   ```bash
   docker run --rm -p 8000:8000 gcr.io/${PROJECT_ID}/speech-api-worker:v1.17
   ```
1. Update the Kubernetes deployment manifest (e.g. `src/service/deployment.yaml`) with the updated
   images.
1. Update the Kubernetes deployment with:
   ```bash
   kubectl apply -f src/service/deployment.yaml
   ```

## Update Container from GCP Machine

Similar to the above, except:

- Docker will need to be installed like so:
  https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-18-04
- For authentication reasons, the build should be pushed with the `gcloud` tool:
  ```bash
  sudo gcloud docker -- push gcr.io/${PROJECT_ID}/speech-api-worker:v1.04
  ```
  Learn more here: https://cloud.google.com/container-registry/docs/advanced-authentication

## New Cluster

These deployment steps are loosely based on these guides below:
- https://cloud.google.com/kubernetes-engine/docs/tutorials/hello-app
- https://cloud.google.com/kubernetes-engine/docs/tutorials/http-balancer

Refer to the above guides in case there are missing details in the below steps.

1. A cluster consists of a pool of Compute Engine VM instances running Kubernetes, the open source
   cluster orchestration system that powers GKE. Create a cluster by following these steps:
    1. Navigate to GCPs `Create a Kubernetes cluster` page.
    1. Pick the `Highly available` template on the left hand side. Among other things, this template
       ensures a regional deployment with a 3:00am `Maintenance window`.
    1. Name the cluster at the top of the form. The name should describe properties of the cluster
       like compute resources, region, usage, etc.
    1. Use the latest Skylake CPUs. This option is found under the `Customize` option in the
       `Node pools` section.
    1. Pick the required computer resources for this deployment.
    1. Expand the `Advanced options` section. Google has some suggestions marked by an exclamation
      mark, so follow them.
1. Assuming GKE is installed on your system. Log into your cluster via:
   ```bash
   gcloud container clusters get-credentials yourclustername --zone=yourclusterzone
   ```
1. Set up permissions via RBAC:
    1. To run the next step, your account will need to be a `cluster-admin`:
       ```bash
       kubectl create clusterrolebinding yourname-cluster-admin-binding \
           --clusterrole=cluster-admin \
           --user=youremail
       ```
    2. Give permission to the service to create `Pods` via:
       ```bash
       kubectl apply -f src/service/rbac.yaml
       ```
1. Set up API Keys. Our API Keys are stored on Kubernetes as secrets. Create them via:
   ```bash
   kubectl create secret generic speech-api-key \
     --from-literal=matt_hocking_api_key='blahblah' \
     --from-literal=michael_petrochuk_api_key='blahblah' \
     --from-literal=web_backend_api_key='blahblah'
   ```
   The API Keys must be 32 characters each.
1. Deploy the web application to Kubernetes.
    1. Apply the deployment manifest, creating a `Deployment`.
       ```bash
       kubectl apply -f src/service/deployment.yaml
       ```
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
    1. It is important to set up a secure HTTPS endpoint, here are the steps to do so:
        1. On a web browser, navigate to the ingress service in the GKE frontend landing on a
           page called "Ingress Details".
        1. Within the details, there is a section titled "Ingress". Within the section, click the
           link to the right of "Load balancer".
        1. On the top nav bar select "edit", then on the left panel click "Frontend configuration".
        1. On the right side click "+ Add Frontend IP and port".
        1. Configure the "New Frontend IP and port" with 443 for port, HTTPS for the protocol,
           some HTTPS certificate, and some IP address.
        1. Wait up to 5 minutes to the changes to take effect.