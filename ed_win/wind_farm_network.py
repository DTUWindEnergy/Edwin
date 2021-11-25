from abc import ABC, abstractmethod
from ed_win.collection_system import collection_system
from ed_win.c_mst_cables import plot_network
# from ed_win.repair import repair
import pandas as pd
import numpy as np


class Driver(ABC):
    def __init__(self, **kwargs):
        '''
        '''
        
    def run(self, x, y, T):
        T, cables_cost = self._run(x=x, y=y, T=T)
        return T, cables_cost


    @abstractmethod
    def _run():
        '''

        '''
class Repair(Driver):
    def __init__(self, **kwargs):
        self.supports_constraints = False
        self.supports_primary = False
        self.supports_secondary = True
        
        Driver.__init__(self, **kwargs)
    
    def _run(self, T, **kwargs):
        print('I will repair T')
        cost = np.nan
        return T, cost

class Refine(Driver):
    def __init__(self, **kwargs):
        self.supports_constraints = False
        self.supports_primary = False
        self.supports_secondary = True
        
        Driver.__init__(self, **kwargs)
    
    def _run(self, T, **kwargs):
        print('I will refine T')
        cost = np.nan
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

    def _run(self, x, y, **kwargs):
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

    def _run(self, x, y, **kwargs):
        T, cables_cost = collection_system(x,
                                           y,
                                           self.option,
                                           self.Inters_const,
                                           self.max_it,
                                           self.wfn.cables)
        return T, cables_cost

class WindFarmNetwork():
    def __init__(self, turbine_positions, substation_positions, drivers=[TwoStepHeuristicDriver()], cables=[], T=None, sequence=None):
        """WindFarmNetwork object

        Parameters
        ----------
        initial_layout : array-like
            The shape of the array is (i, j), where i is 2 and j is the number of  turbines + 1.
            i=1 is x and i=2 is y. j=0 is the coordinates of the offshore sub station and j=1: are the turbine coordinates.
        drivers : list
            List of Driver objects
        cables : array-like
            The shape of the array is (n, m), where n is the number of available cables and m is 3.
            m=1 is cross-section, m=2 is the allowed number of connected WTs and m=3 is the price/km of the cable
        """
        self.turbine_positions = turbine_positions
        self.substation_positions = substation_positions
        self.drivers = drivers
        self.cables = cables
        self.state = None
        self.T = T
        self.columns = ['from_node', 'to_node', 'cable_length', 'cable_type', 'cable_cost']
        if isinstance(sequence, type(None)):
            sequence = range(len(drivers))
        self.sequence = sequence
        self.setup()

    def setup(self):
        for driver in self.drivers:
            setattr(driver, 'wfn', self)

    def design(self, x=None, y=None, **kwargs):
        """designs or optimizes the electrical wind farm network

        Parameters
        ----------
        x : array-like
            concatenated list of sub station and turbine x-coordinates
        y : array-like
            concatenated list of sub station and turbine y-coordinates

        Returns
        -------
        cost : float
            The cost of the electrical network
        state : DataFrame
            The current network tree with the columns f{self.columns}
        """
        if isinstance(x, type(None)):
            x = self.turbine_positions[0]
        if isinstance(y, type(None)):
            y = self.turbine_positions[1]
        x = np.concatenate((self.substation_positions[0], x), axis=0)
        y = np.concatenate((self.substation_positions[1], y), axis=0)
        
        edges = self.get_edges(x, y)
        self.edges = edges
        
        self.x = x
        self.y = y
        T = self.T
        for n, driver_no in enumerate(self.sequence):
            driver = self.drivers[driver_no]
            if n == 0 and not driver.supports_primary:
                raise Exception(driver + ' cannot be the first driver in a sequence')
            if n > 0 and not driver.supports_secondary:
                raise Exception(driver + ' cannot be used as a secondary driver')                
            T, cost = driver.run(x=x, y=y, T=T)
        
        # T, cost = self.driver.run(x, y)
        tree_table = pd.DataFrame(T, columns=self.columns)
        tree_table = tree_table.astype({'from_node': int,
                              'to_node': int,
                              'cable_type': int})
        self.T = T
        self.cost = cost
        self.tree_table = tree_table
        return cost, tree_table

    def get_edges(self, x, y):
        return 'edges'
    

    def plot(self):
        if self.T is None:
            self.design()
        plot_network(self.x, self.y, self.cables, self.T)


class Constraints(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, {'crossing': False,
                             'tree': False,
                             'thermal_capacity': False,
                             'number_of_main_feeders': False})
        self.update(kwargs)


def main():
    if __name__ == '__main__':
        turbine_positions = np.asarray([[2000., 4000., 6000.,
                                          8000., 498.65600569, 2498.65600569, 4498.65600569,
                                          6498.65600569, 8498.65600569, 997.31201137, 2997.31201137,
                                          4997.31201137, 11336.25662483, 8997.31201137, 1495.96801706,
                                          3495.96801706, 5495.96801706, 10011.39514341, 11426.89538545,
                                          1994.62402275, 3994.62402275, 5994.62402275, 7994.62402275,
                                          10588.90471566],
                                       [0., 0., 0.,
                                          0., 2000., 2000., 2000.,
                                          2000., 2000., 4000., 4000.,
                                          4000., 6877.42528387, 4000., 6000.,
                                          6000., 6000., 3179.76530545, 5953.63051694,
                                          8000., 8000., 8000., 8000.,
                                          4734.32972738]])
        substation_positions = np.asarray([[0], [0]])
        settings = {'option': 3,
                    'Inters_const': True,
                    'max_it': 20000,
                    'repair': True}
        cables = np.array([[500, 3, 100000], [800, 5, 150000], [1000, 10, 250000]])
        wfn = WindFarmNetwork(turbine_positions=turbine_positions,
                              substation_positions=substation_positions,
                              drivers=[TwoStepHeuristicDriver(**settings), Refine(), Repair()],
                              sequence = [0, 2, 1],
                              cables=cables)
        cost, state = wfn.design()
        wfn.plot()


main()
