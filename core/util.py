import sys
import time
import logging
import datetime as dt

from pathlib import Path
from contextlib import contextmanager


@contextmanager
def timer(name, logger=None):
    t0 = time.time()
    yield
    msg = f'[{name}] done in {time.time() - t0:.0f} s'
    if logger:
        logger.info(msg)
    else:
        print(msg)


def get_logger(name="Main", exp_name="exp"):
    log_path = Path("log/")
    log_path.mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    filename = log_path / (
        f"{exp_name}_" + dt.datetime.now().strftime("%Y%m%d%H%M%S") + ".log")
    fh = logging.FileHandler(filename)
    sh = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s")
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger
