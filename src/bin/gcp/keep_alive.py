"""
This script runs a loop to restart preetible servers or shutdown halted experiments.

Example:

    $ python -m src.bin.gcp.keep_alive --project_name 'your_comet_ml_project_name' \
             --instance your_gcp_instance_name \
             --command="screen -dm bash -c \
                  'cd /opt/wellsaid-labs/Text-to-Speech;
                  . venv/bin/activate
                  PYTHONPATH=. python src/bin/train/spectrogram_model/__main__.py --checkpoint;'"
"""
import argparse
import json
import logging
import os
import sched
import subprocess
import time

from comet_ml import API as CometAPI
from dotenv import load_dotenv

from src.utils import seconds_to_string

logging.basicConfig(
    format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

INSTANCE_RUNNING = 'RUNNING'
INSTANCE_STOPPED = 'TERMINATED'

load_dotenv()

COMET_ML_WORKSPACE = os.getenv('COMET_ML_WORKSPACE')
COMET_ML_API_KEY = os.getenv('COMET_ML_API_KEY')
COMET_ML_REST_API_KEY = os.getenv('COMET_ML_REST_API_KEY')


def get_comet_ml_api():
    """ Get an instance of `CometAPI`.

    NOTE: Data received from `CometAPI` can be cached. To reset the cache, create a new instance.
    Learn more: https://www.comet.ml/docs/python-sdk/Comet-REST-API/#utility-functions
    """
    return CometAPI(
        api_key=os.getenv('COMET_ML_API_KEY'), rest_api_key=os.getenv('COMET_ML_REST_API_KEY'))


def get_available_instances(names=None):
    """ Get a list of preemtible instances to keep alive.

    Args:
        names (None or list of str): Names of instances to keep alive.

    Returns
        (list of dict): List of instances to keep alive.
    """
    instances = json.loads(
        subprocess.check_output('gcloud compute instances list --format json', shell=True))
    instances = [i for i in instances if i['scheduling']['preemptible']]
    if names is not None:
        return [i for i in instances if i['name'] in set(names)]

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

    assert len(names) == len(
        filtered_instances), 'Some of the instances you selected were not found.'

    return filtered_instances


def get_running_experiments(instances, comet_ml_project_name):
    """ Get a list of experiments running at `instances`.

    Args:
        instances (list of dict): Instances with the ``name``, ``zone`` and ``status`` defined.
        comet_ml_project_name (str)

    Returns:
        (list): List of the most recently created experiments for each instance in `instances`.
    """
    logger.info('Getting comet experiment metadata.')
    comet_ml_api = get_comet_ml_api()
    experiments = comet_ml_api.get(COMET_ML_WORKSPACE, comet_ml_project_name)

    experiments = sorted(experiments, key=lambda e: e.data['start_server_timestamp'], reverse=True)
    get_hostname = lambda e: comet_ml_api.get_experiment_system_details(e.key)['hostname']
    return [next(e for e in experiments if get_hostname(e) == i['name']) for i in instances]


def keep_alive(comet_ml_project_name,
               instances,
               experiments,
               command,
               scheduler,
               repeat_every=60 * 5,
               retry_timeout=60,
               retry=3,
               max_halt_time=1000 * 60 * 5):
    """ Restart GCP instances every ``repeat_every`` seconds with ``command``.

    Args:
        comet_ml_project_name (str)
        instances (list of dict): Instances with the ``name``, ``zone`` and ``status`` defined.
        experiments (list): List of the most recently created experiments for each instance in
            `instances`.
        command (str): Command to run at the start of the instance.
        scheduler (sched.scheduler): Scheduler to rerun this function.
        repeat_every (int, optional): Repeat this call every ``repeat_every`` seconds.
        retry_timeout (int, optional): Timeout between retries in seconds. This function requires
            `retry_timeout` is longer than it takes for an experiment to start.
        retry (int, optional): Number of retries incase failure.
        max_halt_time (int, optional): The maximum time an experiment can be halted before being
            restarted in milliseconds.
    """
    for instance, experiment in zip(instances, experiments):
        try:
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
                        logger.info('Retrying again in %d', retry_timeout)
                        time.sleep(retry_timeout)

                    try:
                        logger.info('Restarting instance "%s" in zone %s', name, zone)
                        output = subprocess.check_output(
                            'gcloud compute instances start %s --zone=%s' % (name, zone),
                            shell=True)
                        logger.info('Restarting instance output:\n%s', output.decode('utf-8'))

                        logger.info('Running command on instance: %s', command)
                        output = subprocess.check_output(
                            'gcloud compute ssh %s --zone=%s --command="%s"' %
                            (name, zone, command),
                            shell=True)
                        logger.info('Command output:\n%s', output.decode('utf-8'))
                        break

                    except Exception:
                        logger.exception('Fatal error caught while restarting instance.')
            elif status == INSTANCE_RUNNING:
                # NOTE: Checks if an experiment has been halted for longer than `max_halt_time`.
                logger.info('Checking on experiment %s/%s/%s', COMET_ML_WORKSPACE,
                            comet_ml_project_name, experiment.key)
                updated_experiment = get_comet_ml_api().get(COMET_ML_WORKSPACE,
                                                            comet_ml_project_name, experiment.key)
                # NOTE: This API returns a `list` if the `experimet.key` is partial or invalid.
                assert not isinstance(updated_experiment,
                                      list), 'The experiment key is no longer valid.'
                elapsed = time.time() * 1000 - updated_experiment.data['end_server_timestamp']
                logger.info('The instance was heard from %s ago.',
                            seconds_to_string(elapsed / 1000))
                if elapsed > max_halt_time:
                    logger.info(
                        'Stopping instance "%s" in zone %s, it has not been heard from for %s.',
                        name, zone, seconds_to_string(max_halt_time / 1000))
                    output = subprocess.check_output(
                        'gcloud compute instances stop %s --zone=%s' % (name, zone), shell=True)
                    logger.info('Stoppping instance output:\n%s', output.decode('utf-8'))

        except Exception:
            logger.exception('Fatal error caught, trying again in %ds.', repeat_every)

        print('-' * 100)

    scheduler.enter(
        delay=repeat_every,
        priority=1,
        action=keep_alive,
        argument=(args.project_name, instances, experiments, command, scheduler, repeat_every))


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--project_name', type=str, required=True, help='The comet.ml project name.')
    parser.add_argument(
        '--command', type=str, required=True, help='Command to run on GCP instance on its restart.')
    parser.add_argument(
        '--instance', default=None, action='append', help='A GCP instance to keep alive.')
    args = parser.parse_args()

    instances = get_available_instances(args.instance)
    experiments = get_running_experiments(instances, args.project_name)

    logger.info('Keeping alive instances: %s', [i['name'] for i in instances])
    logger.info('-' * 100)
    scheduler = sched.scheduler(time.time, time.sleep)
    keep_alive(args.project_name, instances, experiments, args.command, scheduler)
    scheduler.run()
