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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sync(cli_args, scheduler, destination_root='~/sync/', repeat_every=60):
    for server in cli_args.server:
        source = server + '.' + cli_args.project
        destination = os.path.expanduser(os.path.join(destination_root, server))
        if not os.path.isdir(destination):
            logger.info('Making directory %s', os.path.abspath(destination))
            os.makedirs(destination)
        sync = '%s:%s %s' % (source, cli_args.path, destination)
        command = ' '.join(
            ['rsync', '--archive', '--verbose', '--rsh=ssh', "--exclude='*.pt'", sync])
        logger.info('\tRunning:\n%s', command)
        os.system(command)
    print('-' * 100)
    scheduler.enter(
        delay=repeat_every,
        priority=1,
        action=sync,
        argument=(cli_args, scheduler, destination, repeat_every))


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
