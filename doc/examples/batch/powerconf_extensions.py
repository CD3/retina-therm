import logging
from pathlib import Path

import numpy
import yaml
from powerconf.units import Q_
from powerconf.utils import get_id

logging.basicConfig(filename="powerconf_extensions.log", level=logging.INFO)


def get_scale(filename):
    filepath = Path(filename)
    logging.info(f"getting scale from {filepath}")
    if not filepath.exists():
        logging.info(f"file does not exist")
        return None
    config = yaml.safe_load(filepath.read_text())
    return config["scale"]


def get_peak_temperature(filename):
    data = numpy.loadtxt(filename)
    v = max(data[:, 1])

    return Q_(v, "degC")
