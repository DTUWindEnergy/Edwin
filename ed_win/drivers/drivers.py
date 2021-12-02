from abc import ABC, abstractmethod
from ed_win.drivers.tsh.collection_system import collection_system
from ed_win.drivers.planarize.planarize import detecting_and_fixing_crossings
from ed_win.drivers.utils import half_edges


class Driver(ABC):
    def __init__(self, **kwargs):
        '''
        '''

    def run(self, x=None, y=None, T=None):
        '''
        x : array-like
            concatenated array of sub-station and turbine x-coordinates
        y : array-like
            concatenated array of sub-station and turbine y-coordinates
        T : array-like
            solution tree

        '''
        T, cables_cost = self._run(x=x, y=y, T=T)
        return T, cables_cost

    @abstractmethod
    def _run():
        '''

        '''


class Planarize(Driver):
    def __init__(self, **kwargs):
        self.supports_constraints = False
        self.supports_primary = False
        self.supports_secondary = True

        Driver.__init__(self, **kwargs)

    def _run(self, x, y, T, **kwargs):
        n_wt_oss, edges_tot, half = half_edges(x, y)
        nodes = int(len(x))
        T, cost, number_crossings = detecting_and_fixing_crossings(x, y, edges_tot, T, self.wfn.cables, nodes)
        return T, cost


class NCC(Driver):
    def __init__(self, **kwargs):
        self.supports_constraints = False
        self.supports_primary = False
        self.supports_secondary = True

        Driver.__init__(self, **kwargs)

    def _run(self, T, **kwargs):
        print('I will refine T')
        cost = self.wfn.cost
        return T, cost


class TwoStepHeuristicDriver(Driver):
    def __init__(self, option=3, Inters_const=True, max_it=20000, **kwargs):
        self.supports_constraints = False
        self.supports_primary = True
        self.supports_secondary = False

        self.option = option
        self.Inters_const = Inters_const
        self.max_it = max_it
        Driver.__init__(self, **kwargs)

    def _run(self, x, y, **kwargs):
        T, cables_cost = collection_system(x,
                                           y,
                                           self.option,
                                           self.Inters_const,
                                           self.max_it,
                                           self.wfn.cables)
        return T, cables_cost


class GlobalDriver(Driver):
    def __init__(self, option=3, Inters_const=True, max_it=20000, **kwargs):
        self.supports_constraints = True
        self.supports_primary = True
        self.supports_secondary = True

        self.option = option
        self.Inters_const = Inters_const
        self.max_it = max_it
        Driver.__init__(self, **kwargs)

    def _run(self, x, y, T, **kwargs):
        T, cables_cost = collection_system(x,
                                           y,
                                           self.option,
                                           self.Inters_const,
                                           self.max_it,
                                           self.wfn.cables)
        return T, cables_cost


class GeneticAlgorithmDriver(Driver):
    def __init__(self, option=3, Inters_const=True, max_it=20000, **kwargs):
        self.supports_constraints = True
        self.supports_primary = True
        self.supports_secondary = True

        self.option = option
        self.Inters_const = Inters_const
        self.max_it = max_it
        Driver.__init__(self, **kwargs)

    def _run(self, x, y, T, **kwargs):
        T, cables_cost = collection_system(x,
                                           y,
                                           self.option,
                                           self.Inters_const,
                                           self.max_it,
                                           self.wfn.cables)
        return T, cables_cost
