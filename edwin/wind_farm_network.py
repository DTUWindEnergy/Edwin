from abc import ABC, abstractmethod


class WindFarmNetwork():
    def __init__(self, method, geometry, financial, electrical, contraints):
        self.method = method

    def design(self, settings):
        self.settings = settings
        return self.method._design(**settings)

class Method(ABC):
    def __init__(self, **kwargs):
        return
    
    @abstractmethod
    def _design(**settings):
        '''

        Parameters
        ----------
        **settings : dict
            Configuration of algorithm.

        Returns
        -------
        dictionary of connections.

        '''

class HeuristicMethod(Method):
    def __init__(self, **kwargs):
        Method.__init__(self)

    def _design(self, **settings):
        return {'hello from': 'HeuristicMethod'}
    
class MetaHeuristicMethod(Method):
    def __init__(self, **kwargs):
        Method.__init__(self)

    def _design(self, **settings):
        return {'hello from': 'MetaHeuristicMethod'}
    
class GlobalMethod(Method):
    def __init__(self, **kwargs):
        Method.__init__(self)

    def _design(self, **settings):
        return {'hello from': 'GlobalMethod'}

def main():
    if __name__ == '__main__':
        hm = MetaHeuristicMethod()
        geometry = {'turbine_coordinates': {'x': [1],
                                            'y': [1],},
                    'sub_station_coordinates': {'x': [2],
                                                'y': [2],}}
        financial = {}
        electrical = {}
        contraints = {'crossing': False,
                       'tree': False,
                       'thermal capacity': False,
                       'number of main feeders': False}
        settings = {}
        wfn = WindFarmNetwork(method=hm, geometry=geometry, financial=financial,
                              electrical=electrical, contraints=contraints)
        result_dict = wfn.design(settings)
        print(result_dict)
main()