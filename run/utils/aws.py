"""Utilities for working with AWS.

TODO:
- Add a script for creating a batch of instances.
- Add a script for imaging a machine and printing the progress.
- Add a script for transfering a spot request across availability zones.
- Add a script for updating a machine's start up script for signal model training.
"""
import base64
import json
import logging
import math
import pathlib
import typing
import urllib.request
from enum import Enum

import boto3
import tabulate
import typer

import lib

logger = logging.getLogger(__name__)
app = typer.Typer(context_settings=dict(max_content_width=math.inf))
client = boto3.client("ec2")


@app.command()
def image_id(name: str):
    """Get the image id of the image named NAME."""
    filters = [{"Name": "name", "Values": [name]}]
    images = client.describe_images(Owners=["amazon"], Filters=filters)["Images"]
    assert len(images) == 1, "Found multiple images."
    typer.echo(images[0]["ImageId"])


class _OperatingSystem(str, Enum):
    LINUX = "Linux"
    WINDOWS = "Windows"


# Learn more: https://aws.amazon.com/ec2/spot/instance-advisor/
_SPOT_ADVISOR_JSON = "https://spot-bid-advisor.s3.amazonaws.com/spot-advisor-data.json"


@app.command()
def interruptions(
    machine_type: str,
    operating_system: _OperatingSystem = _OperatingSystem.LINUX,
    source: str = _SPOT_ADVISOR_JSON,
):
    """Get the "frequency of interruption" of a spot instance with MACHINE-TYPE and
    OPERATING-SYSTEM."""
    with urllib.request.urlopen(source) as url:
        data = json.loads(url.read().decode())
    filtered = []
    for region, rates in data["spot_advisor"].items():
        if machine_type in rates[operating_system]:
            filtered.append((data["ranges"][rates[operating_system][machine_type]["r"]], region))
    filtered = sorted(filtered, key=lambda k: k[0]["index"])
    filtered = [(rate["label"], region) for rate, region in filtered]
    typer.echo(tabulate.tabulate(filtered, headers=["Frequency of interruption", "Region"]))


def _import_key_pair(ssh_key_path: pathlib.Path):
    """ Import public SSH key to AWS, if doesn't already exist. """
    public_ssh_key_path = ssh_key_path.parent / (ssh_key_path.name + ".pub")
    assert public_ssh_key_path.exists(), f"Expected public key at {public_ssh_key_path}"
    public_ssh_key = public_ssh_key_path.read_bytes()
    response = client.import_key_pair(KeyName=ssh_key_path.name, PublicKeyMaterial=public_ssh_key)
    logger.info("Imported key-pair: %s", response["KeyName"])


def _create_security_group(security_group_name: str, security_group_description: str):
    """ Create an AWS security group that authorizes SSH connections. """
    response = client.create_security_group(
        GroupName=security_group_name, Description=security_group_description
    )
    logger.info("Created security group: %s", response["GroupId"])
    # Learn more:
    # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/ec2.html#EC2.Client.authorize_security_group_ingress
    response = client.authorize_security_group_ingress(
        GroupId=response["GroupId"], IpProtocol="tcp", CidrIp="0.0.0.0/0", FromPort=22, ToPort=22
    )


class _SpotInstanceState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting-down"
    TERMINATED = "terminated"
    STOPPING = "stopping"
    STOPPED = "stopped"


def _find_instance(name_tag: str, name: str) -> None:
    """ Find instance named `name`. """
    response = client.describe_instances(Filters=[{"Key": f"tag:{name_tag}", "Value": name}])
    assert len(response["Reservations"]) == 1
    response = response["Reservations"][0]
    if len(response["Instances"]) == 0:
        return None
    assert len(response["Instances"]) == 1, "Found multiple instances"
    return response["Instances"][0]


def _wait_until_instance_is_running(name_tag: str, name: str):
    logger.info("Waiting until instance is created...")
    response = None
    while response is None or response["State"]["Name"] != _SpotInstanceState.RUNNING:
        response = _find_instance(name_tag, name)


"""
Learn more about startup scripts in EC2:
https://aws.amazon.com/premiumsupport/knowledge-center/execute-user-data-ec2/
"""

_USER_DATA = """Content-Type: multipart/mixed; boundary=\"//\"
MIME-Version: 1.0

--//
Content-Type: text/cloud-config; charset=\"us-ascii\"
MIME-Version: 1.0
Content-Transfer-Encoding: 7bit
Content-Disposition: attachment; filename=\"cloud-config.txt\"

#cloud-config
cloud_final_modules:
- [scripts-user, always]

--//
Content-Type: text/x-shellscript; charset=\"us-ascii\"
MIME-Version: 1.0
Content-Transfer-Encoding: 7bit
Content-Disposition: attachment; filename=\"userdata.txt\"

#!/bin/bash
{startup_script}
--//"""


class _SpotInstanceType(str, Enum):
    ONE_TIME = "one-time"
    PERSISTANT = "persistent"


class _SpotInstanceInterruptionBehavior(str, Enum):
    HIBERNATE = "hibernate"
    STOP = "stop"
    TERMINATE = "terminate"


_SECURITY_GROUP_NAME = typer.Option("only-ssh", "Name for security group that allows for SSH.")
_SECURITY_GROUP_DESCRIPTION = typer.Option(
    "Only allow SSH connections", "Description for security group that allows for SSH."
)
_INTERRUPTION_BEHAVIOR = typer.Option(
    _SpotInstanceInterruptionBehavior.STOP, "The behavior when a Spot Instance is interrupted."
)
_TYPE = typer.Option(_SpotInstanceType.PERSISTANT, "The Spot Instance request type.")
_DISK_SIZE = typer.Option(512, "The size of the disk, in GiB.")
_BASE_STARTUP_SCRIPT_PATH = lib.environment.ROOT_PATH / "run" / "utils" / "aws_spot_start_up.sh"
_STARTUP_SCRIPT = typer.Option(
    "",
    "A bash script to run on the Spot Instances startup. "
    "The output of the startup script will be saved on the VM here:"
    " `/var/log/cloud-init-output.log`",
)
_NAME_TAG = typer.Option("Name", "The name of the tag used to identify the instance name.")


@app.command()
def spot_instance(
    name: str = typer.Option(...),
    image_id: str = typer.Option(...),
    machine_type: str = typer.Option(...),
    ssh_key_path: pathlib.Path = typer.Option(..., exists=True, dir_okay=False),
    disk_size: int = _DISK_SIZE,
    startup_script: str = _STARTUP_SCRIPT,
    security_group_name: str = _SECURITY_GROUP_NAME,
    security_group_description: str = _SECURITY_GROUP_DESCRIPTION,
    type: _SpotInstanceType = _TYPE,
    interruption_behavior: _SpotInstanceInterruptionBehavior = _INTERRUPTION_BEHAVIOR,
    base_startup_script_path: pathlib.Path = _BASE_STARTUP_SCRIPT_PATH,
    aws_credentials_path: pathlib.Path = pathlib.Path("~/.aws/credentials").expanduser(),
    availability_zone: typing.Optional[str] = typer.Option(None),
    name_tag: str = _NAME_TAG,
):
    """Create a Spot Instance named NAME with the corresponding IMAGE-ID, MACHINE-TYPE, DISK-SIZE
    and STARTUP-SCRIPT."""
    filters = [{"Name": f"tag:{name_tag}", "Values": [name]}]
    requests = client.describe_spot_instance_requests(Filters=filters)["SpotInstanceRequests"]
    assert len(requests) == 0, f"Spot request named '{name}' already exists."
    assert _find_instance(name_tag, name) is None, f"Instance named '{name}' already exists."

    # NOTE: Enable SSH
    _import_key_pair(ssh_key_path)
    _create_security_group(security_group_name, security_group_description)

    base_startup_script = (base_startup_script_path).read_text()
    base_startup_script = base_startup_script.format(
        aws_credentials=aws_credentials_path.read_text(),
        vm_region=client.meta.region_name,
        vm_name=name,
        more_bash_script=startup_script,
    )
    launch_specification = {
        "SecurityGroups": [security_group_name],
        "BlockDeviceMappings": [
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "DeleteOnTermination": True,
                    "VolumeSize": disk_size,
                },
            }
        ],
        "ImageId": image_id,
        "InstanceType": machine_type,
        "KeyName": ssh_key_path.name,
        "UserData": base64.b64encode(base_startup_script.encode("ascii")),
    }
    if availability_zone is not None:
        launch_specification["Placement"] = {"AvailabilityZone": availability_zone}
    response = client.request_spot_instances(
        Type=type,
        InstanceInterruptionBehavior=interruption_behavior,
        LaunchSpecification=launch_specification,
    )
    request_id = response["SpotInstanceRequests"][0]["SpotInstanceRequestId"]
    logger.info("Created Spot Instance request: %s", request_id)
    client.create_tags(Resources=[request_id], Tags=[{"Key": name_tag, "Value": name}])

    _wait_until_instance_is_running(name_tag, name)


def spot_request_id(name: str = typer.Option(...), name_tag: str = _NAME_TAG):
    """Print the spot request id of the spot request NAME."""
    requests = client.describe_spot_instance_requests(
        Filters=[{"Name": f"tag:{name_tag}", "Values": [name]}]
    )["SpotInstanceRequests"]
    assert len(requests) <= 1, "Found multiple requests."
    assert len(requests) > 0, "Found no requests."
    typer.echo(requests[0]["SpotInstanceRequestId"])


def instance_id(name: str = typer.Option(...), name_tag: str = _NAME_TAG):
    """Print the instance id of the instance NAME."""
    instance = _find_instance(name_tag, name)
    assert instance is not None, "Unable to find instance."
    typer.echo(instance["InstanceId"])


def public_dns(name: str = typer.Option(...), name_tag: str = _NAME_TAG):
    """Print the public DNS address of the instance NAME."""
    instance = _find_instance(name_tag, name)
    assert instance is not None, "Unable to find instance."
    typer.echo(instance["PublicDnsName"])


@app.command()
def most_recent(name_tag: str = _NAME_TAG):
    """Print the name of the most recent instance created named NAME."""
    requests = client.describe_spot_instance_requests()["SpotInstanceRequests"]
    request = sorted(requests, key=lambda r: r["CreateTime"], reverse=True)[0]
    tags = {t["Key"]: t["Value"] for t in request["Tags"]}
    typer.echo(tags[name_tag])


if __name__ == "__main__":
    app()
