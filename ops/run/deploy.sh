#!/bin/bash
#
# Deployment script for our tts cloud run services.
#
#   Usage: ./deploy.sh <path_to_deployment_config>
#
# This script generates the kubernetes manifest files (via jsonnet), applies
# those resources (via kubectl), and then patches the underlying services
# with kong-related annotations. This is currently necessary due to the
# fact that we cannot annotate the underlying knative services, see:
# https://github.com/knative/serving/issues/5549

set -eo pipefail

# Assert `kubectl` command exists
if ! command -v kubectl &> /dev/null
then
  echo "kubectl could not be found"
  exit 1
fi

# Assert `jsonnet` command exists
if ! command -v jsonnet &> /dev/null
then
  echo "jsonnet could not be found"
  exit 1
fi

# Assert argument was passed
if [ -z "$1" ]
then
  echo "Usage: ./deploy <path_to_deployment_config>"
  exit 1
fi

# Assert env argument file exists
if [ ! -f $1 ]; then
  echo "File not found: $1"
  exit 1
fi

# Pull $model argument from input configuration file
MODEL=$(jq -r '.model' $1)
VERSION=$(jq -r '.version' $1)

if [ -z "${MODEL}" ] || [ $MODEL == "null" ]
then
  echo "Missing 'model' field in deployment configuration"
  exit 1
fi
if [ -z "${VERSION}" ] || [ $VERSION == "null" ]
then
  echo "Missing 'version' field in deployment configuration"
  exit 1
fi

# Quick sanity check, avoid deploying cloud run service with same version.
# The deployment will fail, but we can provide a more straightforward error
# message here.
if [[ "$SKIP_VERSION_CHECK" != "true" ]]; then
    if kubectl get deployment stream-$VERSION-deployment -n $MODEL &> /dev/null; then
      echo "Deployment version '${VERSION}' already exists for model '${MODEL}'!"
      echo "This is not an issue if updating traffic configurations for existing revisions."
      echo "For changes to scaling or image configs, try bumping the version number and running again."
      read -p "Continue? (Y/N): " confirm && [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]] || exit 1
    fi
fi

# Generate manifests
MANIFESTS=$(jsonnet ./tts.jsonnet \
  -y \
  --ext-code-file config=$1)

# Apply manifests
echo "${MANIFESTS}" | kubectl apply -f -

# Check for non-zero exit from kubectl apply and bail early
if [ $? -gt 0 ]; then
  exit 1
fi

# Wait for service to exist before attempting patch
until kubectl get service stream -n $MODEL &> /dev/null
do
  echo "Waiting for existence of stream service in namespace ${MODEL}..."
  sleep 2
done
# Wait for service to exist before attempting patch
until kubectl get service validate -n $MODEL &> /dev/null
do
  echo "Waiting for existence of validate service in namespace ${MODEL}..."
  sleep 2
done

# Patch knative-created services with kong configuration references
kubectl patch service stream \
  -n $MODEL \
  -p '{"metadata":{"annotations":{"konghq.com/override":"route-stream-configuration"}}}'
kubectl patch service validate \
  -n $MODEL \
  -p '{"metadata":{"annotations":{"konghq.com/override":"route-validate-configuration"}}}'
