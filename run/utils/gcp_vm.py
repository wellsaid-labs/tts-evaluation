"""Get details about Google Cloud virtual machine instances."""
import math
import subprocess

import google.auth
import googleapiclient.discovery
import typer

app = typer.Typer(context_settings=dict(max_content_width=math.inf))
credentials, project = google.auth.default()
compute = googleapiclient.discovery.build("compute", "v1", credentials=credentials)


@app.command()
def most_recent():
    """Print the name of the most recent instance created."""
    # NOTE: This wasn't implemented with Google's Python SDK because:
    # - The client must query all zones, and preferably in parallel.
    # - The client must deal with pagination.
    # - The client must deal with parsing the json.
    # - It isn't typed.
    # It's much easier to run the below command...
    command = (
        "gcloud compute instances list --limit=1 --sort-by=creationTimestamp "
        '--format="value(name,creationTimestamp)"'
    )
    typer.echo(subprocess.check_output(command, shell=True).decode().strip().split()[0])


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
