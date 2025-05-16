from powerconf.utils import get_id
import yaml
import numpy
from pathlib import Path
from powerconf.units import Q_


def get_scale(filename):
    filepath = Path(filename)
    if not filepath.exists():
        return None
    config = yaml.safe_load(filepath.read_text())
    return config['scale']

def get_peak_temperature(filename):
    data = numpy.loadtxt(filename)
    v = max(data[:,1])

    return Q_(v,'degC')
