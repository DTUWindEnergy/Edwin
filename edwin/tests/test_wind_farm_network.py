from edwin.tests import npt
from edwin.wind_farm_network import WindFarmNetwork

method = 'test'
wfn = WindFarmNetwork(method)


def test_wind_farm_network():
    assert wfn.method == 'test'
