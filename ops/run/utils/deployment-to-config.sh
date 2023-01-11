#!/bin/bash
#
# This is a small utility that will attempt to generate a deployment configuration from
# an existing deployment. This was originally used to recover deployment configs that
# were not comitted committed to the repository. Note that there are quite a few
# assumptions in the output of this script, use with caution!
#
#   Usage: ./deployment-to-config $MODEL_NAME
#   Usage: ./deployment-to-config $MODE_NAME > config.json
#

# Assert argument was passed
if [ -z "$1" ]; then
  echo "Usage: ./helmme <model_name> <service=stream|validate>"
  exit 1
fi

# Assert env argument file exists
if [[ "$2" && -f $2 ]]; then
  echo "File already exists: $2"
  exit 1
fi

MODEL=$1
FILE=$2
CLUSTER_NAME=$(gcloud config get container/cluster)

# Usage: countRevision <model> <service>
function countRevisions {
  kubectl get revision.serving.knative.dev -n $MODEL -l 'serving.knative.dev/service'=$SERVICE,'serving.knative.dev/routingState'=active --no-headers 2> /dev/null | wc -l
}

# Get `stream` service information
#
#
SERVICE="stream"
# Sanity check, if more than one _active_ revision we need to manually parse traffic structure
ACTIVE_REVISION_COUNT=$(countRevisions $MODEL $SERVICE)
if (( $ACTIVE_REVISION_COUNT > 1 )) ; then
  echo "The $SERVICE service has more than 1 active revision, manual generation of the configuration file is required"
  echo $ACTIVE_REVISION_COUNT
  exit 1
fi
# LATEST_STREAM_REVISION format = {SERVICE}-{revisionNumber}
LATEST_STREAM_REVISION_NAME=$(kubectl get service.serving.knative.dev/$SERVICE -n $MODEL -o yaml 2> /dev/null | awk '/revisionName:/ {print $2;exit;}')
LATEST_STREAM_REVISION=$(kubectl get revision.serving.knative.dev/$LATEST_STREAM_REVISION_NAME -n $MODEL -o yaml 2> /dev/null)
LATEST_STREAM_REVISION_IMAGE=$(echo "$LATEST_STREAM_REVISION" | awk '/image:/ {print $2; exit;}')
LATEST_STREAM_REVISION_SCALE_MIN=$(echo "$LATEST_STREAM_REVISION" | awk '/minScale:/ {print $2; exit;}')
LATEST_STREAM_REVISION_SCALE_MAX=$(echo "$LATEST_STREAM_REVISION" | awk '/maxScale:/ {print $2; exit;}')
LATEST_STREAM_REVISION_CONCURRENCY=$(echo "$LATEST_STREAM_REVISION" | awk '/containerConcurrency:/ {print $2; exit;}')
# Sanity check, ensure imageEntrypoint matches current deployment configuration (manual intervention required otherwise)
if ! grep -q "run.deploy.worker:app" <<< "$LATEST_STREAM_REVISION"; then
  echo "Image entrypoint differs from expected, manual generation of the configuration file is required"
  exit 1
fi

# Get `validate` service information
#
#
SERVICE="validate"
# Sanity check, if more than one _active_ revision we need to manually parse traffic structure
ACTIVE_REVISION_COUNT=$(countRevisions $MODEL $SERVICE)
if (( $ACTIVE_REVISION_COUNT > 1 )) ; then
  echo "The $SERVICE has more than 1 active revision, manual generation of the configuration file is required"
  exit 1
fi
# LATEST_VALIDATE_REVISION format = {SERVICE}-{revisionNumber}
LATEST_VALIDATE_REVISION_NAME=$(kubectl get service.serving.knative.dev/$SERVICE -n $MODEL -o yaml 2> /dev/null | awk '/revisionName:/ {print $2;exit;}')
LATEST_VALIDATE_REVISION=$(kubectl get revision.serving.knative.dev/$LATEST_VALIDATE_REVISION_NAME -n $MODEL -o yaml 2> /dev/null)
LATEST_VALIDATE_REVISION_IMAGE=$(echo "$LATEST_VALIDATE_REVISION" | awk '/image:/ {print $2; exit;}')
LATEST_VALIDATE_REVISION_SCALE_MIN=$(echo "$LATEST_VALIDATE_REVISION" | awk '/minScale:/ {print $2; exit;}')
LATEST_VALIDATE_REVISION_SCALE_MAX=$(echo "$LATEST_VALIDATE_REVISION" | awk '/maxScale:/ {print $2; exit;}')
LATEST_VALIDATE_REVISION_CONCURRENCY=$(echo "$LATEST_VALIDATE_REVISION" | awk '/containerConcurrency:/ {print $2; exit;}')
if ! grep -q "run.deploy.worker:app" <<< "$LATEST_VALIDATE_REVISION"; then
  echo "Image entrypoint differs from expected, manual generation of the configuration file is required"
  exit 1
fi

# Sanity check that stream and validate images are identical (current feature of our deployments)
if [ "$LATEST_STREAM_REVISION_IMAGE" != "$LATEST_VALIDATE_REVISION_IMAGE" ]; then
  echo "Images for stream and validate service differ"
  exit 1
fi

VERSION=$(echo $LATEST_STREAM_REVISION_NAME | grep -oE '[0-9]+')
PROVIDE_LEGACY_API_KEY_AUTH=$(grep -q "LEGACY_SPEECH_API_KEY" <<< "$LATEST_STREAM_REVISION" && echo "true" || echo "false")

JSON=$(cat << EOF
{
  "env": "$CLUSTER_NAME",
  "model": "$MODEL",
  "version": "$VERSION",
  "image": "$LATEST_STREAM_REVISION_IMAGE",
  "imageEntrypoint": "run.deploy.worker:app",
  "provideApiKeyAuthForLegacyContainerSupport": "$PROVIDE_LEGACY_API_KEY_AUTH",
  "stream": {
    "minScale": $LATEST_STREAM_REVISION_SCALE_MIN,
    "maxScale": $LATEST_STREAM_REVISION_SCALE_MAX,
    "concurrency": $LATEST_STREAM_REVISION_CONCURRENCY,
    "paths": [
      "/api/text_to_speech/stream"
    ]
  },
  "validate": {
    "minScale": $LATEST_VALIDATE_REVISION_SCALE_MIN,
    "maxScale": $LATEST_VALIDATE_REVISION_SCALE_MAX,
    "concurrency": $LATEST_VALIDATE_REVISION_CONCURRENCY,
    "paths": [
      "/api/text_to_speech/input_validated"
    ]
  },
  "traffic": [
    {
      "tag": "$VERSION",
      "percent": 100
    }
  ]
}
EOF
)

echo $JSON | jq .
