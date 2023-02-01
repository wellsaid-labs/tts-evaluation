#!/bin/bash
#
# This script deploys all of the configurations for a particular environment.
# It should be used carefully because of it's breadth.
#
# Deploys are executed sequentially. The script stops if a deploy fails.
#

set -euo pipefail

if [[ -z "$1" ]]; then
    echo "Error: must specify path to configuration files"
    echo "Usage: deploy_all.sh PATH"
    exit 1
fi

for conf in "$1"*.json; do
    echo "Updating $conf..."
    echo

    dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
    "$dir"/deploy.sh "$conf"
done

echo
echo "OK"
