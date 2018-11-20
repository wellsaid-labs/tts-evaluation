"""
This script runs a loop to restart preetible servers.

NOTE: Within the example, we add ``shutdown now`` incase the ``python3`` process dies; therefore,
queuing it up to be rebooted by ``keep_alive.py``.

TODO: Create a GCP ulities package merging functionality in ``keep_alive`` and ``periodic_rsync``.

Example:

    python3 -m src.bin.gcp.keep_alive --command="screen -dm bash -c \
        'source ~/.bashrc;
        source ~/.bash_profile;
        cd WellSaid-Labs-Text-To-Speech/; \
        ulimit -n 65536; \
        python3 -m src.bin.train.signal_model -c; \
        sudo shutdown;'"
"""
import argparse
import json
import logging
import sched
import subprocess
import time

logging.basicConfig(
    format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

INSTANCE_RUNNING = 'RUNNING'
INSTANCE_STOPPED = 'TERMINATED'


def get_available_instances():  # pragma: no cover
    """ Get a list of preemtible instances to keep alive.
    """
    instances = json.loads(
        subprocess.check_output('gcloud compute instances list --format json', shell=True))
    instances = [i for i in instances if i['scheduling']['preemptible']]
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
            response = input('Keep alive instance "%s" (%dx%s, %s)? (Y/n) ' %
                             (instance['name'], num_gpu, gpu, instance['status']))
            if response == 'Y':
                filtered_instances.append(instance)
    logger.info('Keeping alive instances: %s', [i['name'] for i in filtered_instances])
    print('-' * 100)
    return filtered_instances


def is_halted(name, zone, command='find . -type f -not -name \'.*\' -cmin -10 2>/dev/null'):
    """ Check if instance halted by executing a command

    Args:
        name (str): Instance name.
        zone (str): Instance zone.
        command (str): Command returns some output if the instance is running by default checks if
            any none-private files have been updated in the last 5 minutes.

    Returns
        (bool)
    """
    logger.info('Checking if instance halted execution with command: %s', command)
    try:
        output = subprocess.check_output(
            'gcloud compute ssh %s --zone=%s --command="%s"' % (name, zone, command), shell=True)
        output = output.decode('utf-8')
    except subprocess.CalledProcessError as e:
        output = e.output.decode('utf-8')
    output = output.strip()
    logger.info('Command output:\n%s', output)
    if len(output) == 0:
        return True
    else:
        return False


def keep_alive(instances, command, scheduler, repeat_every=60 * 5, retry_timeout=60,
               retry=3):  # pragma: no cover
    """ Restart GCP instances every ``repeat_every`` seconds with ``command``.

    Args:
        instances (list of dict): Instances with the ``name``, ``zone`` and ``status`` defined.
        command (str): Command to run at the start of the instance.
        scheduler (sched.scheduler): Scheduler to rerun this function.
        repeat_every (int): Repeat this call every ``repeat_every`` seconds.
        retry_timeout (int): Timeout between retries.
        retry (int): Number of retries incase failure.
    """
    for instance in instances:
        name = instance['name']
        zone = instance['zone'].split('/')[-1]

        logger.info('Checking instance "%s" status in zone %s', name, zone)
        output = json.loads(
            subprocess.check_output(
                'gcloud compute instances describe %s --zone=%s --format=json' % (name, zone),
                shell=True))
        status = output['status']
        logger.info('Status of the instance is: %s', status)

        if status == INSTANCE_STOPPED:
            for i in range(retry):
                if i > 0:
                    logger.info('Retrying again in %d', repeat_every)
                    time.sleep(retry_timeout)

                try:
                    logger.info('Restarting instance "%s" in zone %s', name, zone)
                    output = subprocess.check_output(
                        'gcloud compute instances start %s --zone=%s' % (name, zone), shell=True)
                    logger.info('Restarting instance output:\n%s', output.decode('utf-8'))

                    logger.info('Running command on instance: %s', command)
                    output = subprocess.check_output(
                        'gcloud compute ssh %s --zone=%s --command="%s"' % (name, zone, command),
                        shell=True)
                    logger.info('Command output:\n%s', output.decode('utf-8'))
                    break

                except Exception as e:
                    logger.warning('Exception: %s', e)
        elif status == INSTANCE_RUNNING and is_halted(name, zone):
            logger.info('Stopping instance "%s" in zone %s', name, zone)
            output = subprocess.check_output(
                'gcloud compute instances stop %s --zone=%s' % (name, zone), shell=True)
            logger.info('Stoppping instance output:\n%s', output.decode('utf-8'))

        print('-' * 100)

    scheduler.enter(
        delay=repeat_every,
        priority=1,
        action=keep_alive,
        argument=(instances, command, scheduler, repeat_every))


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--command', type=str, default=None, required=True, help='GCP project name.')
    args = parser.parse_args()
    instances = get_available_instances()
    scheduler = sched.scheduler(time.time, time.sleep)
    keep_alive(instances, args.command, scheduler)
    scheduler.run()