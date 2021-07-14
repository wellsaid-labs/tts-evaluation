# GKE Cluster Setup

This document details how to setup a new GKE cluster that's intended to act as
an execution runtime for the TTS service.

1. First enable the required services. You'll only need to do this once per
   Google Cloud Project.

   ```
   gcloud services enable \
       container.googleapis.com \
       containerregistry.googleapis.com \
       cloudbuild.googleapis.com \
   ```

2. Next, create a GKE cluster:

   ```
   export name="dev"
   gcloud beta container clusters create "$name" \
       --region "us-central1" \
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
       --cloud-run-config=load-balancer-type=INTERNAL
   ```

3. Then read the [instructions for deploying the TTS service](./run/README.md).

## Reserving a static IP

See
[Reserving a static external IP address](https://cloud.google.com/compute/docs/ip-addresses/reserve-static-external-ip-address).
The static IP will be referenced in the configuration/deployment of our
[Kong Gateway](./gateway/README.md). At this point it appears that Kong does
[not support global static ips](https://docs.konghq.com/kubernetes-ingress-controller/1.3.x/deployment/gke/#requirements).

```bash
~ export ENV=$ENV
~ gcloud compute addresses create gateway-$ENV --region=us-central1
~ # View newly reserved ip address
~ gcloud compute addresses list
```
