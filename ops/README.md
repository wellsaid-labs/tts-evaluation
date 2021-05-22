# TTS Operations

This directory contains code and documentation for the infrastructure required to run
the TTS service.

The service is deployed via Google Cloud Run, using Cloud Run for Anthos. The setup utilizes
a GKE cluster as the execution runtime for the Cloud Run services.

## Cluster Setup

To setup a new cluster, follow [this guide](./ClusterSetup.md).

## Deploying the Services

To deploy instances of the service, follow [this guide](./run/README.md).
