from abc import ABC, abstractmethod
from ed_win.collection_system import collection_system
from ed_win.c_mst_cables import plot_network
import pandas as pd
import numpy as np


class Driver(ABC):
    @abstractmethod
    def run():
        '''

        '''


class HeuristicDriver(Driver):
    def __init__(self, option=3, Inters_const=True, max_it=20000):
        self.option = option
        self.Inters_const = Inters_const
        self.max_it = max_it
        Driver.__init__(self)

    def run(self, x, y):
        T, cables_cost = collection_system(x,
                                           y,
                                           self.option,
                                           self.Inters_const,
                                           self.max_it,
                                           self.wfn.cables)
        return T, cables_cost


class WindFarmNetwork():
    def __init__(self, initial_layout, driver=HeuristicDriver(), cables=[]):
        self.initial_layout = initial_layout
        self.driver = driver
        self.cables = cables
        self.state = None
        self.T = None
        self.columns = ['from_node', 'to_node', 'cable_length', 'cable_type', 'cable_cost']
        self.setup()

    def setup(self):
        setattr(self.driver, 'wfn', self)

    def design(self, x=None, y=None, **kwargs):
        if isinstance(x, type(None)):
            x = self.initial_layout['x']
        if isinstance(y, type(None)):
            y = self.initial_layout['y']
        self.x = x
        self.y = y
        T, cost = self.driver.run(x, y)
        state = pd.DataFrame(T, columns=self.columns)
        state = state.astype({'from_node': int,
                              'to_node': int,
                              'cable_type': int})
        self.T = T
        self.cost = cost
        self.state = state
        return cost, state

    def plot(self):
        if self.state is None:
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
        initial_layout = dict(x=np.array([0., 2000., 4000., 6000.,
                                          8000., 498.65600569, 2498.65600569, 4498.65600569,
                                          6498.65600569, 8498.65600569, 997.31201137, 2997.31201137,
                                          4997.31201137, 11336.25662483, 8997.31201137, 1495.96801706,
                                          3495.96801706, 5495.96801706, 10011.39514341, 11426.89538545,
                                          1994.62402275, 3994.62402275, 5994.62402275, 7994.62402275,
                                          10588.90471566]),
                              y=np.array([0., 0., 0., 0.,
                                          0., 2000., 2000., 2000.,
                                          2000., 2000., 4000., 4000.,
                                          4000., 6877.42528387, 4000., 6000.,
                                          6000., 6000., 3179.76530545, 5953.63051694,
                                          8000., 8000., 8000., 8000.,
                                          4734.32972738]))
        settings = {'option': 3,
                    'Inters_const': True,
                    'max_it': 20000}
        cables = np.array([[500, 3, 100000], [800, 5, 150000], [1000, 10, 250000]])
        wfn = WindFarmNetwork(initial_layout=initial_layout,
                              driver=HeuristicDriver(**settings),
                              cables=cables)
        cost, state = wfn.design()
        wfn.plot()


main()
