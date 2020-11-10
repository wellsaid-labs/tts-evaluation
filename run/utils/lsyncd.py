import logging
import math
import os
import pathlib
import tempfile

import typer

import lib

app = typer.Typer(context_settings=dict(max_content_width=math.inf))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(
    source: pathlib.Path = typer.Argument(..., help="A path on the local machine."),
    destination: pathlib.Path = typer.Argument(..., help="A path on the remote machine."),
    public_dns: str = typer.Option(..., help="The remote machine public DNS"),
    user: str = typer.Option(..., help="A remote machine username."),
    identity_file: pathlib.Path = typer.Option(
        ...,
        help="The path to private key used for authentication "
        "(e.g. ~/.ssh/michaelp_amazon_web_services)",
    ),
    template: pathlib.Path = typer.Option(
        lib.environment.ROOT_PATH / "run" / "utils" / "lsyncd.conf.lua",
        help="The path to a lsyncd config and template.",
    ),
):
    """Sync SOURCE to USER@PUBLIC_DNS:DESTINATION via [lsyncd](https://github.com/axkibe/lsyncd).

    This script requires `sudo`, and for `rsync` to be installed on the remote machine.
    """
    config = template.read_text().strip()
    config = config.replace("{source}", str(source))
    config = config.replace("{user}", user)
    config = config.replace("{destination}", str(destination))
    config = config.replace("{public_dns}", public_dns)
    config = config.replace("{identity_file}", str(identity_file))
    with tempfile.NamedTemporaryFile() as file_:
        path = pathlib.Path(file_.name)
        path.write_text(config)
        os.execvp("lsyncd", ["lsyncd", str(path)])


if __name__ == "__main__":  # pragma: no cover
    typer.run(main)
