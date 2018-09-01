""" This executable file runs parameterized lsyncd (https://github.com/axkibe/lsyncd).

NOTE: This script uses ``sudo``, be prepared to type in your password.

Example:

    python3 -m src.bin.lsyncd --source ~/Code/WellSaid-Labs-Text-To-Speech/ \
                              --destination /home/michaelp/WellSaid-Labs-Text-To-Speech \
                              --user michaelp --instance tensorboard
"""
from pathlib import Path

import argparse
import json
import logging
import os
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_ip(instance_name):  # pragma: no cover
    """ Get the IP address of an instance

    Args:
        instance_name (str): Name of GCP instance

    Returns:
        (str): IP address of GCP instance
    """
    instances = json.loads(
        subprocess.check_output('gcloud compute instances list --format json',
                                shell=True).decode("utf-8"))
    instances = [i for i in instances if i['name'] == instance_name]
    assert len(instances) == 1
    return instances[0]['networkInterfaces'][0]['accessConfigs'][0]['natIP']


def main(source,
         destination,
         instance,
         user,
         template='src/bin/lsyncd.conf.lua',
         tmp='/tmp/lsyncd.conf.lua'):  # pragma: no cover
    """ Starts a lsyncd session.

    Args:
        source (str): Path on local machine to sync.
        destination (str): Path on remote machine to sync.
        instance (str): Name of remote GCP instance.
        user (str): Username on remote machine.
        template (str): Template configuration for lsyncd.
        tmp (str): Tmp filename to save configuration.
    """
    config = Path(template).read_text().strip()
    config = config.replace('{source}', source)
    config = config.replace('{user}', user)
    config = config.replace('{destination}', destination)
    config = config.replace('{ip}', get_ip(instance))
    config = config.replace('{home}', os.environ['HOME'])

    with open(tmp, 'w+') as file_:
        file_.write(config)

    os.execvp('sudo', ['sudo', 'lsyncd', tmp])


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--source', type=str, required=True, help='Path on local machine to sync')
    parser.add_argument(
        '-d', '--destination', type=str, required=True, help='Path on remote machine to sync')
    parser.add_argument(
        '-i', '--instance', type=str, required=True, help='Name of remote GCP instance')
    parser.add_argument('-u', '--user', type=str, required=True, help='Username on remote machine')
    args = parser.parse_args()
    main(**vars(args))
