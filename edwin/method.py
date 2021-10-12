from abc import ABC, abstractmethod


class Method(ABC):
    def __init__(self, **kwargs):
        return

    @abstractmethod
    def _design(geometry, financial, electrical, contraints, **settings):
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

    def _design(self, geometry, financial, electrical, contraints, **settings):
        return {'hello from': 'HeuristicMethod'}


class MetaHeuristicMethod(Method):
    def __init__(self, **kwargs):
        Method.__init__(self)

    def _design(self, geometry, financial, electrical, contraints, **settings):
        return {'hello from': 'MetaHeuristicMethod'}


class GlobalMethod(Method):
    def __init__(self, **kwargs):
        Method.__init__(self)

    def _design(self, geometry, financial, electrical, contraints, **settings):
        return {'hello from': 'GlobalMethod'}
