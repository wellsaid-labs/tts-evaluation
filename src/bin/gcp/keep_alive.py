"""
This script runs a loop to restart preetible servers or shutdown halted experiments.

Example:

    $ python -m src.bin.gcp.keep_alive \
        --project_name your_comet_ml_project_name \
        --instance your_gcp_instance_name \
        --instance your_other_gcp_instance_name \
        --command="screen -dmL bash -c \
                    'sudo chmod -R a+rwx /opt/;
                    cd /opt/wellsaid-labs/Text-to-Speech;
                    . venv/bin/activate;
                    PYTHONPATH=. python src/bin/train/spectrogram_model/__main__.py --checkpoint;'"
"""
import argparse
import json
import logging
import sched
import subprocess
import time

from comet_ml.config import get_config
from comet_ml.papi import API as CometAPI
from retry import retry

from src.environment import set_basic_logging_config
from src.utils import seconds_to_string

set_basic_logging_config()
logger = logging.getLogger(__name__)

INSTANCE_RUNNING = 'RUNNING'
INSTANCE_STOPPED = 'TERMINATED'
COMET_WORKSPACE = get_config()['comet.workspace']


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
            # Example `instance['guestAccelerators'][0]['acceleratorType']` value:
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


@retry(delay=15, tries=3)
def get_running_experiments(instances, comet_ml_project_name):
    """ Get a list of experiments running at `instances`.

    Args:
        instances (list of dict): Instances with the `name`, `zone` and `status` defined.
        comet_ml_project_name (str)

    Returns:
        (list of comet.APIExperiment): List of the most recently created experiments for each
            instance in `instances`.
    """
    logger.info('Getting comet experiment metadata.')
    comet_ml_api = CometAPI()
    experiments = comet_ml_api.get(COMET_WORKSPACE, comet_ml_project_name)
    experiments = sorted(
        experiments, key=lambda e: e.to_json()['start_server_timestamp'], reverse=True)
    get_hostname = lambda e: e.get_system_details()['hostname']
    return [next(e for e in experiments if get_hostname(e) == i['name']) for i in instances]


@retry(delay=15, tries=3)
def start_instance(name, zone):
    """ This starts a GCP instance.

    Args:
        name (str): The GCP instance name.
        zone (str): The GCP instance zone.

    Returns:
        (str): The output of the command.
    """
    logger.info('Starting instance "%s" in zone "%s".', name, zone)
    output = subprocess.check_output(
        'gcloud compute instances start %s --zone=%s' % (name, zone), shell=True)
    logger.info('Starting instance output:\n%s', output.decode('utf-8'))
    return output.decode('utf-8')


@retry(delay=15, tries=3)
def run_command_on_instance(name, zone, command):
    """ This runs a command on a GCP instance.

    Args:
        name (str): The GCP instance name.
        zone (str): The GCP instance zone.
        command (str): A bash command to run.

    Returns:
        (str): The output of the command.
    """
    logger.info('On instance "%s" in zone "%s" running command: %s', name, zone, command)
    output = subprocess.check_output(
        'gcloud compute ssh %s --zone=%s --command="%s"' % (name, zone, command), shell=True)
    logger.info('Command output:\n%s', output.decode('utf-8'))
    return output.decode('utf-8')


@retry(delay=15, tries=3)
def stop_instance(name, zone):
    """ This stops a GCP instance.

    Args:
        name (str): The GCP instance name.
        zone (str): The GCP instance zone.

    Returns:
        (str): The output of the command.
    """
    logger.info('Stopping instance "%s" in zone "%s".', name, zone)
    output = subprocess.check_output(
        'gcloud compute instances stop %s --zone=%s' % (name, zone), shell=True)
    logger.info('Stoppping instance output:\n%s', output.decode('utf-8'))
    return output.decode('utf-8')


@retry(delay=15, tries=3)
def get_instance_status(name, zone):
    """ This returns the GCP instance status.

    Args:
        name (str): The GCP instance name.
        zone (str): The GCP instance zone.

    Returns:
        (str): The GCP instance status, learn more here:
               https://cloud.google.com/compute/docs/instances/instance-life-cycle
    """
    logger.info('Checking status of instance "%s" in zone "%s".', name, zone)
    output = json.loads(
        subprocess.check_output(
            'gcloud compute instances describe %s --zone=%s --format=json' % (name, zone),
            shell=True))
    status = output['status']
    logger.info('Status of the instance is: %s', status)
    return status


@retry(delay=15, tries=3)
def get_instance_last_started_time(name, id, zone):
    """ This returns the last time this GCP instance was started.

    Args:
        name (str): The GCP instance name.
        name (str): The GCP instance id.
        zone (str): The GCP instance zone.

    Returns:
        (int): The Unix timestamp in seconds this instance was started.
    """
    logger.info('Checking start time of instance "%s" with id "%s" in zone "%s".', name, id, zone)
    output = json.loads(
        subprocess.check_output(
            ('gcloud logging read \'resource.type="gce_instance" AND ' +
             'resource.labels.instance_id="%s" AND logName:activity_log AND ' +
             'jsonPayload.event_subtype:start\' --format json --limit 1') % (id,),
            shell=True))
    assert len(output) == 1, 'The proceeding code expects one output.'
    last_start_time = int(output[0]['jsonPayload']['event_timestamp_us']) / 1000 / 1000
    logger.info('The instance was started %s ago.',
                seconds_to_string(time.time() - last_start_time))
    return last_start_time


@retry(delay=15, tries=3)
def get_experiment_last_message_time(comet_ml_project_name, experiment):
    """ This returns the last time this experiment recieved a message.

    Args:
        comet_ml_project_name (str): The comet ml project name.
        experiment (comet.APIExperiment): The comet ml experiment.

    Returns:
        (int): The Unix timestamp in seconds this experiment recieved a message.
    """
    logger.info('Checking on the Comet experiment\'s %s last message time.', experiment.url)
    updated_experiment = CometAPI().get(COMET_WORKSPACE, comet_ml_project_name, experiment.id)
    # NOTE: This API returns not a `list` if the `experimet.id` is partial or invalid.
    assert not isinstance(updated_experiment, list), 'The experiment key is no longer valid.'
    last_message_time = max(
        [m['timestampCurrent'] for m in updated_experiment.get_metrics_summary()]) / 1000
    logger.info('The Comet experiment recieved a message %s ago.',
                seconds_to_string(time.time() - last_message_time))
    return last_message_time


def keep_alive(comet_ml_project_name,
               instances,
               experiments,
               command,
               scheduler,
               repeat_every=60,
               max_halt_time=60 * 45):
    """ This ensures that `experiments` on `instances` keep running.

    NOTE: Our pipeline may take up to 20m to start; therefore, we set the max halt time
    to be double that at ~45 minutes.

    Args:
        comet_ml_project_name (str)
        instances (list of dict): Instances with the `name`, `zone` and `status` defined.
        experiments (list): List of the most recently created experiments for each instance in
            `instances`.
        command (str): Command to run at the start of the instance.
        scheduler (sched.scheduler): Scheduler to rerun this function.
        repeat_every (int, optional): Repeat this call every `repeat_every` seconds.
        max_halt_time (int, optional): The maximum time an experiment can be halted before being
            restarted in seconds.
    """
    for instance, experiment in zip(instances, experiments):
        try:
            name = instance['name']
            zone = instance['zone'].split('/')[-1]
            id = instance['id']

            status = get_instance_status(name, zone)
            if status == INSTANCE_STOPPED:
                start_instance(name, zone)
                run_command_on_instance(name, zone, command)
            elif status == INSTANCE_RUNNING:
                last_start_time = get_instance_last_started_time(name, id, zone)
                last_message_time = get_experiment_last_message_time(comet_ml_project_name,
                                                                     experiment)
                halt_time = time.time() - max(last_start_time, last_message_time)
                logger.info(
                    'The instance "%s" has been halted for %s of the maximum allowed %s. '
                    'If this halts for more than `max_halt_time`, then the experiment has '
                    'stopped or is stuck and needs to restarted.', name,
                    seconds_to_string(halt_time), seconds_to_string(max_halt_time))
                if halt_time > max_halt_time:
                    logger.info('Stopping instance %s because it has been halted longer than %s',
                                name, seconds_to_string(max_halt_time))
                    stop_instance(name, zone)

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
    print('-' * 100)
    scheduler = sched.scheduler(time.time, time.sleep)
    keep_alive(args.project_name, instances, experiments, args.command, scheduler)
    scheduler.run()
