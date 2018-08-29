""" This executable periodically rsyncs multiple GCP instances to a local GCP instance.

NOTE:
    Remember to run ``gcloud compute config-ssh`` before running this script and ensure
    each instance has cloud access scope set to ``Allow full access to all Cloud APIs`` on
    https://console.cloud.google.com/.

Example:

    python3 src/bin/periodic_rsync.py \
      --destination ~/WellSaid-Labs-Text-To-Speech/sync/ \
      --source ~/WellSaid-Labs-Text-To-Speech/experiments/signal_model
"""
import argparse
import json
import logging
import os
import sched
import subprocess
import time

from src.utils import ROOT_PATH

logging.basicConfig(
    format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

INSTANCE_RUNNING = 'RUNNING'


def get_instances(all_=False):
    """ Get a list of instances sync.

    Args:
        all_ (bool): If ``True`` sync all instances; otherwise, prompt the user to pick.

    Returns:
        (list of dict): List of instances to sync with the instance details.
    """
    instances = json.loads(
        subprocess.check_output('gcloud compute instances list --format json',
                                shell=True).decode("utf-8"))

    names = [i['name'] for i in instances]
    assert len(set(names)) == len(instances), 'All instances must have a unique name'

    this_instance = subprocess.check_output('hostname').decode("utf-8").strip()
    instances = [instance for instance in instances if instance['name'] != this_instance]
    if all_:
        filtered_instances = instances
    else:
        filtered_instances = []
        for instance in sorted(instances, key=lambda i: i['name']):
            response = ''
            num_gpu = 0
            gpu = 'GPU'

            if 'guestAccelerators' in instance:
                # Example ``instance['guestAccelerators'][0]['acceleratorType']`` value:
                # https://www.googleapis.com/compute/v1/projects/mythical-runner-203817/zones/us-west1-b/acceleratorTypes/nvidia-tesla-p100
                gpu = instance['guestAccelerators'][0]['acceleratorType'].split('/')[-1].upper()
                num_gpu = instance['guestAccelerators'][0]['acceleratorCount']

            while response not in ['Y', 'n']:
                response = input(
                    'Sync "%s" %dx%s instance? (Y/n) ' % (instance['name'], num_gpu, gpu))
                if response == 'Y':
                    filtered_instances.append(instance)

    logger.info('Syncing instances: %s', [i['name'] for i in filtered_instances])
    print('-' * 100)
    return filtered_instances


def main(instances, source, destination, scheduler, repeat_every=5):
    """ ``rsync`` ``instances`` ``source`` to ``destination`` on local instance.

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
        action=main,
        argument=(instances, source, destination, scheduler, repeat_every))


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--source', type=str, required=True, help='Path on remote server to sync')
    parser.add_argument(
        '-d', '--destination', type=str, required=True, help='Path on local server to sync')
    parser.add_argument(
        '-a', '--all', action='store_true', default=False, help='Sync all instances.')
    args = parser.parse_args()
    instances = get_instances(all_=args.all)

    scheduler = sched.scheduler(time.time, time.sleep)
    main(instances, args.source, args.destination, scheduler)
    scheduler.run()
