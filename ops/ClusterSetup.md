# GKE Cluster Setup

This document details how to setup a new GKE cluster that's intended to act as
an execution runtime for the TTS service.

1. First enable the required services. You'll only need to do this once per
   Google Cloud Project.

   ```bash
   gcloud services enable \
       container.googleapis.com \
       containerregistry.googleapis.com \
       cloudbuild.googleapis.com
   ```

2. Next, create a GKE cluster:

   ```bash
   CLUSTER_NAME=<name> # ex: "staging"
   CLUSTER_REGION=<region> # ex: "us-central1"
   # The person at WellSaid Labs who is responsible for the cluster. This should
   # be the portion of the users email before the `@` sign. For instance, if your
   # email is `sams@wellsaidlabs.com`, then this should be `sams`.
   OWNER=<owner>
   IS_NON_PROD=<is_not_prod_env?> # Whether or not to label this as a non-production resource (Vanta)

   gcloud beta container clusters create "$CLUSTER_NAME" \
       --region "$CLUSTER_REGION" \
       --no-enable-basic-auth \
       --release-channel "regular" \
       --machine-type "n1-highcpu-8" \
       --image-type "COS_CONTAINERD" \
       --disk-type "pd-ssd" \
       --disk-size "100" \
       --metadata disable-legacy-endpoints=true \
       --scopes "https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" \
       --num-nodes "1" \
       --enable-stackdriver-kubernetes \
       --enable-ip-alias \
       --no-enable-intra-node-visibility \
       --default-max-pods-per-node "110" \
       --enable-autoscaling \
       --min-nodes "0" \
       --max-nodes "9" \
       --enable-dataplane-v2 \
       --no-enable-master-authorized-networks \
       --addons HorizontalPodAutoscaling,HttpLoadBalancing,CloudRun,GcePersistentDiskCsiDriver \
       --enable-autoupgrade \
       --enable-autorepair \
       --max-surge-upgrade 6 \
       --max-unavailable-upgrade 0 \
       --maintenance-window-start "2021-05-08T07:00:00Z" \
       --maintenance-window-end "2021-05-08T11:00:00Z" \
       --maintenance-window-recurrence "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR,SA,SU" \
       --enable-shielded-nodes \
       --cloud-run-config=load-balancer-type=INTERNAL \
       --labels vanta-owner=$OWNER,vanta-non-prod=$IS_NON_PROD
   ```

   Note you can modify cluster labels after creation, ex:

   ```bash
   # Mark cluster as non-production environment
   gcloud container clusters update $CLUSTER_NAME --region $CLUSTER_REGION --update-labels vanta-non-prod=$IS_NON_PROD
   # Update the Vanta owner
   gcloud container clusters update $CLUSTER_NAME --region $CLUSTER_REGION --update-labels vanta-owner=$OWNER
   ```

3. Then read the [instructions for deploying the TTS service](./run/README.md).

## Reserving a static IP

See
[Reserving a static external IP address](https://cloud.google.com/compute/docs/ip-addresses/reserve-static-external-ip-address).
The static IP will be referenced in the configuration/deployment of our
[Kong Gateway](./gateway/README.md). At this point it appears that Kong does
[not support global static ips](https://docs.konghq.com/kubernetes-ingress-controller/1.3.x/deployment/gke/#requirements).

```bash
gcloud compute addresses create gateway-$ENV --region=$CLUSTER_REGION
# View newly reserved ip address
gcloud compute addresses list
```

## GPU enabled node pool

In order for our TTS Service to take advantage of GPU resources, a GPU node pool must be deployed.
For further information, see [Running GPUs in GKE Standard clusters](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus#gpu_pool).
Note that GPU availability is region dependent.

1. Create a GPU enabled node pool within the GKE cluster. The following creates a node-pool
   named "gpu-worker-pool" with n1-highmem-2 machines and nvidia-tesla-t4 gpus.

   ```bash
   PROJECT_ID=voice-service-2-313121
   CLUSTER_NAME=$CLUSTER_NAME # ex: "staging"
   CLUSTER_REGION=us-central1 # ex: "us-central1"
   gcloud beta container node-pools create "gpu-worker-pool" \
      --project "$PROJECT_ID" \
      --cluster "$CLUSTER_NAME" \
      --region "$CLUSTER_REGION" \
      --node-version "1.22.16-gke.2000" \
      --machine-type "n1-highmem-2" \
      --accelerator "type=nvidia-tesla-t4,count=1" \
      --image-type "COS_CONTAINERD" \
      --disk-type "pd-standard" \
      --disk-size "100" \
      --metadata disable-legacy-endpoints=true \
      --scopes "https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" \
      --num-nodes "1" \
      --enable-autoscaling \
      --min-nodes "1" \
      --max-nodes "128" \
      --enable-autoupgrade \
      --enable-autorepair \
      --max-surge-upgrade 1 \
      --max-unavailable-upgrade 0 \
      --max-pods-per-node "110"
   ```

2. [Install NVIDIA device drivers](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus#installing_drivers)

   ```bash
   kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
   ```

3. At this point, a pod [requesting gpu resources](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus#pods_gpus)
   will be provisioned within the newly created GPU node pool. See the [instructions for deploying
   the TTS service](./run/README.md) for next steps.
