"""Utilities for working with Google Cloud Platform (GCP).

TODO:
- Add a script for creating a batch of instances.
- Add a script for imaging a machine and printing the progress.
- Add a script for updating a machine's start up script for signal model training.
"""
import logging
import math
import pathlib
import subprocess
import time
import typing
from enum import Enum

import google.auth
import google.auth.credentials
import googleapiclient.discovery
import googleapiclient.errors
import typer

import lib

logger = logging.getLogger(__name__)
app = typer.Typer(context_settings=dict(max_content_width=math.inf))
credentials, project = google.auth.default()
compute = googleapiclient.discovery.build("compute", "v1", credentials=credentials)


class _OperationStatus(str, Enum):
    DONE = "DONE"
    RUNNING = "RUNNING"
    PENDING = "PENDING"


def _wait_for_operation(
    operation: str, poll_interval: int = 1, is_global: bool = True, **kwargs
) -> typing.Dict:
    """ Wait for an operation to finish, and return the finished operation. """
    while True:
        client = compute.globalOperations() if is_global else compute.zoneOperations()
        result = client.get(project=project, operation=operation, **kwargs).execute()
        if result["status"] == _OperationStatus.DONE.value:
            if "error" in result:
                raise Exception(result["error"])
            return result
        time.sleep(poll_interval)


@app.command()
def make_instance(
    name: str = typer.Option(...),
    zone: str = typer.Option(...),
    machine_type: str = typer.Option(...),
    gpu_type: str = typer.Option(...),
    gpu_count: int = typer.Option(...),
    disk_size: int = typer.Option(...),
    disk_type: str = typer.Option(...),
    image_project: str = typer.Option(...),
    image: str = typer.Option(...),
    metadata: typing.List[str] = typer.Option([]),
    metadata_from_file: typing.List[str] = typer.Option([]),
):
    """ Create a managed and preemptible instance named NAME in ZONE. """
    lib.environment.set_basic_logging_config()

    image_ = compute.images().getFromFamily(project=image_project, family=image).execute()
    logger.info("Found image: %s", image_["selfLink"])

    # NOTE: There is some predefined and special metadata, like startup-script:
    # https://cloud.google.com/compute/docs/startupscript
    splits = [m.split("=", maxsplit=1) for m in metadata]
    metadata_ = [{"key": k, "value": v} for k, v in splits]
    splits = [m.split("=", maxsplit=1) for m in metadata_from_file]
    metadata_ += [{"key": k, "value": pathlib.Path(v).read_text()} for k, v in splits]
    body = {
        "name": name,
        "properties": {
            "machineType": machine_type,
            "metadata": {"kind": "compute#metadata", "items": metadata_},
            "guestAccelerators": [{"acceleratorCount": gpu_count, "acceleratorType": gpu_type}],
            "disks": [
                {
                    "kind": "compute#attachedDisk",
                    "type": "PERSISTENT",
                    "boot": True,
                    "autoDelete": True,
                    "deviceName": name,
                    "initializeParams": {
                        "sourceImage": image_["selfLink"],
                        "diskType": disk_type,
                        "diskSizeGb": str(disk_size),
                    },
                }
            ],
            "networkInterfaces": [
                {
                    "kind": "compute#networkInterface",
                    "network": f"projects/{project}/global/networks/default",
                    "accessConfigs": [
                        {
                            "kind": "compute#accessConfig",
                            "name": "External NAT",
                            "type": "ONE_TO_ONE_NAT",
                            "networkTier": "PREMIUM",
                        }
                    ],
                }
            ],
            "scheduling": {"preemptible": True},
            "serviceAccounts": [
                {
                    "email": "default",
                    "scopes": ["https://www.googleapis.com/auth/cloud-platform"],
                }
            ],
        },
    }
    template_op = compute.instanceTemplates().insert(project=project, body=body).execute()
    template_op = _wait_for_operation(template_op["name"])
    logger.info("Created instance template: %s", template_op["targetLink"])

    body = {
        "name": name,
        "baseInstanceName": name,
        "instanceTemplate": template_op["targetLink"],
        "targetSize": 1,
        "statefulPolicy": {
            "preservedState": {"disks": {name: {"autoDelete": "ON_PERMANENT_INSTANCE_DELETION"}}}
        },
    }
    client = compute.instanceGroupManagers()
    manager_op = client.insert(project=project, zone=zone, body=body).execute()
    manager_op = _wait_for_operation(manager_op["name"], zone=zone, is_global=False)
    logger.info("Created instance group manager: %s", manager_op["targetLink"])


@app.command()
def watch_instance(
    name: str = typer.Option(...),
    zone: str = typer.Option(...),
    poll_interval: int = typer.Option(5),
):
    """ Print the status of instance named NAME in ZONE. """
    lib.environment.set_basic_logging_config()
    client = compute.instanceGroupManagers()
    while True:
        list_op = client.listManagedInstances(project=project, zone=zone, instanceGroupManager=name)
        instance = list_op.execute()["managedInstances"][0]
        if "instanceStatus" in instance:
            logger.info("The status of the instance is '%s'.", instance["instanceStatus"])
        else:
            message = "Instance group manager is '%s' '%s'."
            logger.info(message, instance["currentAction"], instance["instance"].split("/")[-1])
            if len(instance["lastAttempt"]) > 0:
                message = instance["lastAttempt"]["errors"]["errors"][0]["message"]
                logger.warning("The last attempt failed because... '%s'", message)
        time.sleep(poll_interval)


@app.command()
def delete_instance(name: str = typer.Option(...), zone: str = typer.Option(...)):
    """ Delete the instance named NAME in ZONE. """
    lib.environment.set_basic_logging_config()

    try:
        client = compute.instanceGroupManagers()
        manager_op = client.delete(project=project, zone=zone, instanceGroupManager=name).execute()
        manager_op = _wait_for_operation(manager_op["name"], zone=zone, is_global=False)
        logger.info("Deleted instance group manager: %s", manager_op["targetLink"])
    except googleapiclient.errors.HttpError as error:
        logger.warning(error._get_reason())

    try:
        client = compute.instanceGroups()
        group_op = client.delete(project=project, zone=zone, instanceGroup=name).execute()
        group_op = _wait_for_operation(group_op["name"], zone=zone, is_global=False)
        logger.info("Deleted instance group: %s", group_op["targetLink"])
    except googleapiclient.errors.HttpError as error:
        logger.warning(error._get_reason())

    try:
        client = compute.instanceTemplates()
        template_op = client.delete(project=project, instanceTemplate=name).execute()
        template_op = _wait_for_operation(template_op["name"])
        logger.info("Deleted instance template: %s", template_op["targetLink"])
    except googleapiclient.errors.HttpError as error:
        logger.warning(error._get_reason())


@app.command()
def most_recent(filter: str = ""):
    """Print the name of the most recent instance created containing the string FILTER."""
    # NOTE: This wasn't implemented with Google's Python SDK because:
    # - The client must query all zones, and preferably in parallel.
    # - The client must deal with pagination.
    # - The client must deal with parsing the json.
    # - It isn't typed.
    # It's much easier to run the below command...
    command = (
        "gcloud compute instances list --sort-by=creationTimestamp "
        '--format="value(name,creationTimestamp)"'
    )
    lines = subprocess.check_output(command, shell=True).decode().strip().split("\n")
    machines = [l.split()[0].strip() for l in lines]
    machines = [m for m in machines if filter in m]
    if len(machines) == 0:
        logger.error("No instance found.")
    else:
        typer.echo(machines[-1])


@app.command()
def zone(name: str = typer.Option(...)):
    """Print the zone of the instance NAME."""
    command = 'gcloud compute instances list --format="value(name,zone)"'
    output = subprocess.check_output(command, shell=True).decode().strip().split("\n")
    lines = [(r.split("\t")[0], r.split("\t")[1]) for r in output]
    assert len(set(r[0] for r in lines)) == len(lines), "Instance name is not unique."
    zones = [r[1] for r in lines if r[0] == name]
    assert len(zones) == 1, "Instance not found."
    typer.echo(zones[0])


@app.command()
def user(name: str = typer.Option(...), zone: str = typer.Option(...)):
    """Print the default user of the instance NAME at ZONE."""
    command = f'gcloud compute ssh {name} --zone {zone} --command="echo $USER"'
    typer.echo(subprocess.check_output(command, shell=True).decode().strip())


@app.command()
def ip(zone: str = typer.Option(...), name: str = typer.Option(...)):
    """Print the ip address of the instance NAME at ZONE."""
    request = compute.instances().get(project=project, zone=zone, instance=name)
    response = request.execute()
    assert len(response["networkInterfaces"]) == 1
    assert len(response["networkInterfaces"][0]["accessConfigs"]) == 1
    typer.echo(response["networkInterfaces"][0]["accessConfigs"][0]["natIP"])


if __name__ == "__main__":
    app()
