import logging
from pathlib import Path

ROOT_PATH: Path = Path(__file__).parents[1].resolve()  # Repository root path


def setup_logger():
    """Add a configured logger to the script being run"""

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %("
        "message)s",
        datefmt="%Y-%m-%d - %H:%M:%S",
        level=logging.INFO,
    )

    return logging.getLogger()


logger = setup_logger()
