""" This executable file runs parameterized lsyncd (https://github.com/axkibe/lsyncd).

NOTE: This script uses ``sudo``, be prepared to type in your password.
NOTE: `rsync` must be installed on the remote machine for this to work.

Example:

    $ python -m src.bin.gcp.lsyncd --source $(pwd) \
                                   --destination /opt/wellsaid-labs/Text-to-Speech \
                                   --user your_gcp_user_name \
                                   --instance your_gcp_instance_name
"""
from pathlib import Path

import argparse
import logging
import os

from src.environment import ROOT_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(source,
         destination,
         public_dns,
         identity_file,
         user,
         template=ROOT_PATH / 'src' / 'bin' / 'cloud' / 'lsyncd.conf.lua',
         tmp='/tmp/lsyncd.conf.lua'):
    """ Starts a lsyncd session.

    Args:
        source (str): Path on local machine to sync.
        destination (str): Path on remote machine to sync.
        public_dns (str): Name of remote GCP instance.
        identity_file (str): File from which the identity (private key) for authentication is read.
        user (str): Username on remote machine.
        template (Path): Template configuration for lsyncd.
        tmp (str): Tmp filename to save configuration.
    """
    config = template.read_text().strip()
    config = config.replace('{source}', source)
    config = config.replace('{user}', user)
    config = config.replace('{destination}', destination)
    config = config.replace('{public_dns}', public_dns)
    config = config.replace('{identity_file}', identity_file)
    tmp = Path(tmp)
    tmp.write_text(config)
    os.execvp('sudo', ['sudo', 'lsyncd', tmp])


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Path on local machine to sync')
    parser.add_argument(
        '--destination', type=str, required=True, help='Path on remote machine to sync')
    parser.add_argument(
        '--public_dns', type=str, required=True, help='The public DNS of the instance')
    parser.add_argument(
        '--identity_file',
        type=str,
        required=True,
        help='File from which the identity (private key) for authentication is read.')
    parser.add_argument('--user', type=str, required=True, help='Username on remote machine')
    args = parser.parse_args()
    main(**vars(args))
