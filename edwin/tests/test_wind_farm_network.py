from edwin.wind_farm_network import WindFarmNetwork
from edwin.method import MetaHeuristicMethod

method = MetaHeuristicMethod()
geometry = {'turbine_coordinates': {'x': [1, 1, 1, 2, 2, 2, 3, 3, 3],
                                    'y': [1, 2, 3, 2, 3, 4, 3, 4, 5], },
            'sub_station_coordinates': {'x': [2.5],
                                        'y': [3.5], }}
financial = {}
electrical = {}
constraints = {'crossing': False,
               'tree': False,
               'thermal capacity': False,
               'number of main feeders': False}
settings = {}
wfn = WindFarmNetwork(method=method, geometry=geometry, financial=financial,
                      electrical=electrical, constraints=constraints)


def test_wind_farm_network():
    result_dict = wfn.design(settings)
    assert result_dict == {'hello from': 'MetaHeuristicMethod'}
