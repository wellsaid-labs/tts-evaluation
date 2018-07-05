"""
This script syncs files from multiple servers to one server using rsync periodically.

NOTE:
    Remember to run ``gcloud compute config-ssh`` before this becomes possible and ensure
    each instance has cloud access scope set to: ``Allow full access to all Cloud APIs``.
"""
import argparse
import logging
import os
import sched
import subprocess
import time

from src.utils import ROOT_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sync(cli_args, scheduler, destination_root='sync/', repeat_every=5):
    for server in cli_args.server:
        source = server + '.' + cli_args.project
        name = server.split('.')[0]
        destination = os.path.abspath(
            os.path.expanduser(os.path.join(ROOT_PATH, destination_root, name)))

        if not os.path.isdir(destination):
            logger.info('Making directory %s', os.path.abspath(destination))
            os.makedirs(destination)

        # NOTE: Updates must be inplace due to this:
        # https://github.com/tensorflow/tensorboard/issues/349
        # NOTE: ``--rsh="ssh -o ConnectTimeout=1"`` in case a server is not responsive.
        # NOTE: Exclude ``*.pt`` or pytorch files, typically, large checkpoint files.
        command = [
            'rsync', '--archive', '--verbose', '--rsh', 'ssh -o ConnectTimeout=1', '--exclude',
            '*.pt', '--human-readable', '--compress', '--inplace',
            '%s:%s' % (source, cli_args.path),
            '%s' % (destination)
        ]
        process = subprocess.Popen(command)
        logger.info('Running: %s', process.args)
        process.wait()
        print('-' * 100)
    scheduler.enter(
        delay=repeat_every,
        priority=1,
        action=sync,
        argument=(cli_args, scheduler, destination_root, repeat_every))


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',
        '--server',
        type=str,
        nargs='+',
        required=True,
        help='GCP server name formatted as such: ``{server name}.{zone}``')
    parser.add_argument(
        '--project', type=str, default='mythical-runner-203817', help='GCP project name.')
    parser.add_argument(
        '-p', '--path', type=str, required=True, help='Path on server to sync to ``~/sync/**``')
    cli_args = parser.parse_args()

    scheduler = sched.scheduler(time.time, time.sleep)
    sync(cli_args, scheduler)
    scheduler.run()
