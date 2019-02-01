import sys
import logging
import datetime as dt

from pathlib import Path


def get_logger(name="Processor", tag="exp", log_dir="../log"):
    path = Path(log_dir)
    path.mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    filename = path / (
        f"{tag}_" + dt.datetime.now().strftime("%Y%m%d%H%M%S") + ".log")
    fh = logging.FileHandler(filename)
    sh = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s")
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger
