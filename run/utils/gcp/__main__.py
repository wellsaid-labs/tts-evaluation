"""Utilities for working with Google Cloud Platform (GCP).

TODO:
- Add a script for creating a batch of instances.
- Add a script for updating a machine's start up script for signal model training, like so:
https://cloud.google.com/compute/docs/instance-groups/rolling-out-updates-to-managed-instance-groups
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

from lib.environment import set_basic_logging_config
from lib.utils import mazel_tov

logger = logging.getLogger(__name__)
app = typer.Typer(context_settings=dict(max_content_width=math.inf))
persistent_app = typer.Typer()
app.add_typer(persistent_app, name="persistent")
preemptible_app = typer.Typer()
app.add_typer(preemptible_app, name="preemptible")
credentials, project = google.auth.default()
# NOTE: The documentation for `compute` is difficult to find, so here is a link:
# https://googleapis.github.io/google-api-python-client/docs/dyn/compute_v1.html
compute = googleapiclient.discovery.build("compute", "v1", credentials=credentials)


class _OperationStatus(str, Enum):
    DONE = "DONE"
    RUNNING = "RUNNING"
    PENDING = "PENDING"


def _wait_for_operation(
    operation: str, poll_interval: int = 1, is_global: bool = True, **kwargs
) -> typing.Dict:
    """Wait for an operation to finish, and return the finished operation."""
    while True:
        client = compute.globalOperations() if is_global else compute.zoneOperations()
        result = client.get(project=project, operation=operation, **kwargs).execute()
        if result["status"] == _OperationStatus.DONE.value:
            if "error" in result:
                raise Exception(result["error"])
            return result
        time.sleep(poll_interval)


def _has_made_preemptible_instance(name: str, zone: str, poll_interval: int = 1):
    """Check if instance group manager successfully created an instance on it's first attempt."""
    client = compute.instanceGroupManagers()
    while True:
        list_op = client.listManagedInstances(project=project, zone=zone, instanceGroupManager=name)
        instance = list_op.execute()["managedInstances"][0]
        # NOTE: Learn more about the instance life cycle, here:
        # https://cloud.google.com/compute/docs/instances/instance-life-cycle
        if "instanceStatus" in instance and instance["instanceStatus"] == "RUNNING":
            return True
        if "lastAttempt" in instance and len(instance["lastAttempt"]) > 0:
            return False
        time.sleep(poll_interval)


def _get_zones() -> typing.List[str]:
    """Get a list of available zones."""
    zones_op = compute.zones().list(project=project).execute()
    return [z["name"] for z in zones_op["items"]]


def _make_preemptible_instance(
    name: str, zone: typing.Optional[str], template_op: typing.Dict, health_check: str
):
    """Create a preemptible instance."""
    try:
        body = {
            "checkIntervalSec": 10,
            "healthyThreshold": 2,
            "logConfig": {"enable": False},
            "name": health_check,
            "tcpHealthCheck": {"port": 22, "proxyHeader": "NONE", "request": "", "response": ""},
            "timeoutSec": 5,
            "type": "TCP",
            "unhealthyThreshold": 3,
        }
        health_check_op = compute.healthChecks().insert(project=project, body=body).execute()
        health_check_op = _wait_for_operation(health_check_op["name"])
        logger.info("Created health check: %s", health_check_op["targetLink"])
    except googleapiclient.errors.HttpError as error:
        logger.warning(error._get_reason())

    for zone in _get_zones() if zone is None else [zone]:
        logger.info(f"Attempting zone: '{zone}'")
        body = {
            "name": name,
            "baseInstanceName": name,
            "instanceTemplate": template_op["targetLink"],
            "targetSize": 1,
            "statefulPolicy": {
                "preservedState": {
                    "disks": {name: {"autoDelete": "ON_PERMANENT_INSTANCE_DELETION"}}
                }
            },
            "autoHealingPolicies": [
                {
                    "initialDelaySec": 300.0,
                    "healthCheck": f"projects/{project}/global/healthChecks/{health_check}",
                }
            ],
        }

        try:
            client = compute.instanceGroupManagers()
            manager_op = client.insert(project=project, zone=zone, body=body).execute()
            manager_op = _wait_for_operation(manager_op["name"], zone=zone, is_global=False)
            logger.info("Created instance group manager: %s", manager_op["targetLink"])
            if _has_made_preemptible_instance(name, zone):
                logger.info(f"Success! {mazel_tov()}")
                break
            else:
                _delete_instance_group(name, zone)
        except Exception as e:
            logger.error(f"Failed to create instance on '{zone}':\n{str(e)}")


def _make_and_watch_persistent_instance(
    name: str, zone: typing.Optional[str], template_op: typing.Dict
):
    """Create and then watch a presistent instance."""
    for zone in _get_zones() if zone is None else [zone]:
        logger.info(f"Attempting zone: '{zone}'")
        client = compute.instances()
        # NOTE: An instance template is created and used to create a single instance to simplify the
        # code.
        link = template_op["targetLink"]
        instance_op = client.insert(
            body={"name": name}, project=project, zone=zone, sourceInstanceTemplate=link
        )
        try:
            instance_op = instance_op.execute()
            instance_op = _wait_for_operation(instance_op["name"], zone=zone, is_global=False)
            logger.info(f"{mazel_tov()} Created instance: {instance_op['targetLink']}")
            break
        except Exception as e:
            logger.error(f"Failed to create instance on '{zone}':\n{str(e)}")

    if zone is not None:
        watch_persistent_instance(name, zone)


def _make_instance(
    name: str,
    zone: typing.Optional[str],
    machine_type: str,
    gpu_type: str,
    gpu_count: int,
    disk_size: int,
    disk_type: str,
    image_project: str,
    image_family: typing.Optional[str],
    image: typing.Optional[str],
    metadata: typing.List[str],
    metadata_from_file: typing.List[str],
    health_check: typing.Optional[str],
    preemptible: bool,
):
    """Create a managed and preemptible instance named NAME in ZONE.

    Args:
        ...
        zone: This can be a zone like "us-east1-c". If a zone is not provided, this will try to
            many different zones until one succeeds.
        ...
    """
    set_basic_logging_config()

    images = compute.images()
    if image_family is not None and image is None:
        image_ = images.getFromFamily(project=image_project, family=image_family).execute()
    elif image_family is None and image is not None:
        image_ = images.get(project=image_project, image=image).execute()
    else:
        # TODO: The error message should reflect that `image` and `image_family` are not `None`.
        typer.echo("Unable to find image.")
        raise typer.Exit(code=1)
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
            "scheduling": {
                "preemptible": preemptible,
                "automaticRestart": not preemptible,
                "onHostMaintenance": "TERMINATE",
            },
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

    if preemptible:
        assert health_check is not None
        _make_preemptible_instance(name, zone, template_op, health_check)
    else:
        _make_and_watch_persistent_instance(name, zone, template_op)


@persistent_app.command("make-instance")
def make_persistent_instance(
    name: str = typer.Option(...),
    zone: str = typer.Option(None, help="Select a specific zone for training."),
    machine_type: str = typer.Option(...),
    gpu_type: str = typer.Option(...),
    gpu_count: int = typer.Option(...),
    disk_size: int = typer.Option(...),
    disk_type: str = typer.Option(...),
    image_project: str = typer.Option(project),
    image_family: typing.Optional[str] = typer.Option(None),
    image: typing.Optional[str] = typer.Option(None),
    metadata: typing.List[str] = typer.Option([]),
    metadata_from_file: typing.List[str] = typer.Option([]),
    health_check: str = typer.Option(None),
):
    return _make_instance(**locals(), preemptible=False)


@preemptible_app.command("make-instance")
def make_preemptible_instance(
    name: str = typer.Option(...),
    zone: str = typer.Option(None, help="Select a specific zone for training."),
    machine_type: str = typer.Option(...),
    gpu_type: str = typer.Option(...),
    gpu_count: int = typer.Option(...),
    disk_size: int = typer.Option(...),
    disk_type: str = typer.Option(...),
    image_project: str = typer.Option(project),
    image_family: typing.Optional[str] = typer.Option(None),
    image: typing.Optional[str] = typer.Option(None),
    metadata: typing.List[str] = typer.Option([]),
    metadata_from_file: typing.List[str] = typer.Option([]),
    health_check: str = typer.Option("check-ssh"),
):
    return _make_instance(**locals(), preemptible=True)


@persistent_app.command("watch-instance")
def watch_persistent_instance(
    name: str = typer.Option(...),
    zone: str = typer.Option(...),
    poll_interval: int = 5,
):
    """Print the status of instance named NAME in ZONE."""
    set_basic_logging_config()
    client = compute.instances()
    while True:
        instance = client.get(project=project, zone=zone, instance=name).execute()
        logger.info("The status of the instance is '%s'.", instance["status"])
        time.sleep(poll_interval)


@preemptible_app.command("watch-instance")
def watch_preemptible_instance(
    name: str = typer.Option(...),
    zone: str = typer.Option(...),
    poll_interval: int = 5,
):
    """Print the status of instance named NAME in ZONE."""
    set_basic_logging_config()
    client = compute.instanceGroupManagers()
    while True:
        list_op = client.listManagedInstances(project=project, zone=zone, instanceGroupManager=name)
        instance = list_op.execute()["managedInstances"][0]
        if "instanceStatus" in instance:
            logger.info("The status of the instance is '%s'.", instance["instanceStatus"])
        else:
            message = "Instance group manager is '%s' '%s'."
            logger.info(message, instance["currentAction"], instance["instance"].split("/")[-1])
            if "lastAttempt" in instance and len(instance["lastAttempt"]) > 0:
                message = instance["lastAttempt"]["errors"]["errors"][0]["message"]
                logger.warning("The last attempt failed because... '%s'", message)
        time.sleep(poll_interval)


def _delete_instance_template(name: str):
    try:
        client = compute.instanceTemplates()
        template_op = client.delete(project=project, instanceTemplate=name).execute()
        template_op = _wait_for_operation(template_op["name"])
        logger.info("Deleted instance template: %s", template_op["targetLink"])
    except googleapiclient.errors.HttpError as error:
        logger.warning(error._get_reason())


@persistent_app.command("delete-instance")
def delete_persistent_instance(name: str = typer.Option(...), zone: str = typer.Option(...)):
    """Delete the instance named NAME in ZONE."""
    set_basic_logging_config()

    try:
        client = compute.instances()
        instance_op = client.delete(project=project, zone=zone, instance=name).execute()
        instance_op = _wait_for_operation(instance_op["name"], zone=zone, is_global=False)
        logger.info("Deleted instance: %s", instance_op["targetLink"])
    except googleapiclient.errors.HttpError as error:
        logger.warning(error._get_reason())

    _delete_instance_template(name)


def _delete_instance_group(name: str, zone: str):
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


@preemptible_app.command("delete-instance")
def delete_preemptible_instance(name: str = typer.Option(...), zone: str = typer.Option(...)):
    """Delete the instance named NAME in ZONE."""
    set_basic_logging_config()
    _delete_instance_group(name, zone)
    _delete_instance_template(name)


"""
NOTE: This `most_recent` wasn't implemented with Google's Python SDK because:
- The client must query all zones, and preferably in parallel.
- The client must deal with pagination.
- The client must deal with parsing the json.
- It isn't typed.
It's much easier to run the below command...
"""


@persistent_app.command("most-recent")
def most_recent_persistent(
    filter: str = "",
    name: typing.Optional[str] = typer.Option(None, help="Filter by the name of a instance."),
):
    """Print the name of the most recent instance created containing the string FILTER."""
    command = (
        "gcloud compute instances list --sort-by=creationTimestamp "
        '--format="value(name,creationTimestamp)"'
    )
    lines = subprocess.check_output(command, shell=True).decode().strip().split("\n")
    machines = [[s.strip() for s in l.split()] for l in lines]
    if name is not None:
        machines = [s for s in machines if len(s) == 2 and s[0] == name]
    machines = [s[0] for s in machines if filter in s[0]]
    if len(machines) == 0:
        logger.error("No instance found.")
    else:
        typer.echo(machines[-1])


@preemptible_app.command("most-recent")
def most_recent_preemptible(
    filter: str = "",
    name: typing.Optional[str] = typer.Option(None, help="Filter by the name of a instance group."),
):
    """Print the name of the most recent instance created containing the string FILTER."""
    command = (
        "gcloud compute instances list --sort-by=creationTimestamp "
        '--format="value(name,metadata.items.created-by,creationTimestamp)"'
    )
    lines = subprocess.check_output(command, shell=True).decode().strip().split("\n")
    machines = [[s.strip() for s in l.split()] for l in lines]
    if name is not None:
        machines = [s for s in machines if len(s) == 3 and s[1].split("/")[-1] == name]
    machines = [s[0] for s in machines if filter in s[0]]
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


def image_and_delete(
    image_family: str, image_name: str, name: str, vm_name: str, zone: str, preemptible: bool
):
    """Image the underlying managed instance, and delete it afterwards.

    NOTE: This will remove the instance from it's managed group, and it cannot be put back.
    """
    set_basic_logging_config()
    try:
        compute.images().get(project=project, image=image_name).execute()
        logger.error("The image '%s' already exists.", image_name)
        return
    except googleapiclient.errors.HttpError:
        pass

    api = "gcloud compute"
    suffix = f"--zone={zone}"
    commands = [
        f"{api} instances stop {vm_name} {suffix}",
        f"{api} images create {image_name} --family={image_family} "
        f"--source-disk={vm_name} --source-disk-zone={zone} --storage-location=us",
    ]
    cmd = f"{api} instance-groups managed abandon-instances {name} --instances={vm_name} {suffix}"
    if preemptible:
        commands.insert(0, cmd)
    for command in commands:
        typer.echo(subprocess.check_output(command, shell=True).decode().strip())

    typer.confirm("Are you sure you want to delete the instance?", abort=True)
    command = f"gcloud compute instances delete {vm_name} --zone={zone}"
    typer.echo(subprocess.check_output(command, shell=True).decode().strip())
    (delete_preemptible_instance if preemptible else delete_persistent_instance)(name, zone)


@persistent_app.command("image-and-delete")
def persistent_image_and_delete(
    image_family: str = typer.Option(...),
    image_name: str = typer.Option(...),
    name: str = typer.Option(...),
    vm_name: str = typer.Option(...),
    zone: str = typer.Option(...),
):
    image_and_delete(**locals(), preemptible=False)


@preemptible_app.command("image-and-delete")
def preemptible_image_and_delete(
    image_family: str = typer.Option(...),
    image_name: str = typer.Option(...),
    name: str = typer.Option(...),
    vm_name: str = typer.Option(...),
    zone: str = typer.Option(...),
):
    image_and_delete(**locals(), preemptible=True)


if __name__ == "__main__":
    app()
