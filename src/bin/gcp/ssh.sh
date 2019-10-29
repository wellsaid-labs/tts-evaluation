#!/bin/sh
#
# SSH into Google Compute Engine without requiring the `--zone` flag.
#
# Example:
# . src/bin/gcp/ssh.sh instance-name
ZONE=`gcloud compute instances list | grep "^$1\s" | awk '{ print $2 }'`
gcloud compute ssh --zone=$ZONE "$@"
