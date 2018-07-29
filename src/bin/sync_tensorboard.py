"""
This script syncs files from multiple servers to one server using rsync periodically.

NOTE:
    Remember to run ``gcloud compute config-ssh`` before this becomes possible and ensure
    each instance has cloud access scope set to: ``Allow full access to all Cloud APIs``.

Example:

    python3 src/bin/sync_tensorboard.py --destination ~/Tacotron-2/sync/ \
                                        --source ~/Tacotron-2/experiments/signal_model
"""
import argparse
import json
import logging
import os
import sched
import subprocess
import time

from src.utils import ROOT_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INSTANCE_RUNNING = 'RUNNING'


def get_available_instances():
    """ Get a list of preemtible instances to keep alive.
    """
    instances = json.loads(
        subprocess.check_output('gcloud compute instances list --format json',
                                shell=True).decode("utf-8"))
    filtered_instances = []
    for instance in sorted(instances, key=lambda i: i['name']):
        response = ''

        if 'guestAccelerators' in instance:
            # EXAMPLE:
            # https://www.googleapis.com/compute/v1/projects/mythical-runner-203817/zones/us-west1-b/acceleratorTypes/nvidia-tesla-p100
            gpu = instance['guestAccelerators'][0]['acceleratorType'].split('/')[-1].upper()
            num_gpu = instance['guestAccelerators'][0]['acceleratorCount']
        else:
            num_gpu = 0
            gpu = 'GPU'

        while response not in ['Y', 'n']:
            response = input('Sync "%s" %dx%s instance? Y/n\n' % (instance['name'], num_gpu, gpu))
            if response == 'Y':
                filtered_instances.append(instance)
    logger.info('Syncing instances: %s', [i['name'] for i in filtered_instances])
    print('-' * 100)
    return filtered_instances


def sync(instances, source, destination, scheduler, repeat_every=5):
    """ ``rsync`` from ``cli_args.server`` to ``cli_args.path`` on local server.

    Args:
        instances (list of dict): Instances with the ``name``, ``zone`` and ``status`` defined.
        source (str): Directory to sync from.
        destination (str): Root directory to sync multiple servers too.
        scheduler (sched.scheduler): Scheduler to rerun this function.
        repeat_every (int): Repeat this call every ``repeat_every`` seconds.
    """
    for instance in instances:
        # EXAMPLE:
        # https://www.googleapis.com/compute/v1/projects/mythical-runner-203817/zones/us-west1-b"
        zone = instance['zone'].split('/')[-1]
        project = instance['zone'].split('/')[-3]
        name = instance['name']

        logger.info('Checking instance "%s" status in zone %s', name, zone)
        output = json.loads(
            subprocess.check_output(
                'gcloud compute instances describe %s --zone=%s --format=json' % (name, zone),
                shell=True).decode("utf-8"))
        status = output['status']
        logger.info('Status of the instance is: %s', status)

        if status == INSTANCE_RUNNING:
            server = '.'.join([name, zone, project])
            server_destination = os.path.abspath(
                os.path.expanduser(os.path.join(ROOT_PATH, destination, name)))

            if not os.path.isdir(server_destination):
                logger.info('Making directory %s', os.path.abspath(server_destination))
                os.makedirs(server_destination)

            # NOTE: Updates must be inplace due to this:
            # https://github.com/tensorflow/tensorboard/issues/349
            # NOTE: ``--rsh="ssh -o ConnectTimeout=1"`` in case a server is not responsive.
            # NOTE: Exclude ``*.pt`` or pytorch files, typically, large checkpoint files.
            command = [
                'rsync', '--archive', '--verbose', '--rsh', 'ssh -o ConnectTimeout=1', '--exclude',
                '*.pt', '--human-readable', '--compress', '--inplace',
                '%s:%s' % (server, source),
                '%s' % (server_destination)
            ]
            process = subprocess.Popen(command)
            logger.info('Running: %s', process.args)
            process.wait()
        print('-' * 100)
    scheduler.enter(
        delay=repeat_every,
        priority=1,
        action=sync,
        argument=(instances, source, destination, scheduler, repeat_every))


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--source', type=str, required=True, help='Path on remote server to sync')
    parser.add_argument(
        '-d', '--destination', type=str, required=True, help='Path on local server to sync')
    args = parser.parse_args()
    instances = get_available_instances()

    scheduler = sched.scheduler(time.time, time.sleep)
    sync(instances, args.source, args.destination, scheduler)
    scheduler.run()
