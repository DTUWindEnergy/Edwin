from ed_win.plot import plot_network
from ed_win.drivers.drivers import TwoStepHeuristicDriver, NCC, Planarize
import pandas as pd
import numpy as np


class WindFarmNetwork():
    def __init__(self, turbine_positions, substation_positions, drivers=[TwoStepHeuristicDriver()], cables=[], T=None, sequence=None):
        """WindFarmNetwork object

        Parameters
        ----------
        turbine_positions : array-like
            Two dimentional array with turbine x- and y-coordinates
        substation_positions : array-like
            Two dimentional array with sub station x- and y-coordinates
        drivers : list
            List of Driver objects
        cables : array-like
            The shape of the array is (n, m), where n is the number of available cables and m is 3.
            m=1 is cross-section, m=2 is the allowed number of connected WTs and m=3 is the price/km of the cable
        """
        if not isinstance(drivers, list):
            drivers = [drivers]
        self.turbine_positions = turbine_positions
        self.substation_positions = substation_positions
        self.drivers = drivers
        self.cables = cables
        self.state = None
        self.T = T
        self.columns = ['from_node', 'to_node', 'cable_length', 'cable_type', 'nwt_connected', 'cable_cost']
        if isinstance(sequence, type(None)):
            sequence = range(len(drivers))
        self.sequence = sequence
        self.setup()

    def setup(self):
        for driver in self.drivers:
            setattr(driver, 'wfn', self)

    def design(self, turbine_positions=None, T=None, **kwargs):
        """designs or optimizes the electrical wind farm network

        Parameters
        ----------
        turbine_positions : array-like
            Two dimentional array with turbine x- and y-coordinates

        T : array
            The current network tree with the columns f{self.columns}

        Returns
        -------
        cost : float
            The cost of the electrical network
        T : array
            The current network tree with the columns f{self.columns}
        """
        if not isinstance(turbine_positions, type(None)):
            self.turbine_positions = turbine_positions
        if isinstance(T, type(None)):
            T = self.T

        x, y = self.positions_to_xy()

        for n, driver_no in enumerate(self.sequence):
            driver = self.drivers[driver_no]
            if n == 0 and not driver.supports_primary:
                raise Exception(f'{driver} cannot be the first driver in a sequence')
            elif n > 0 and not driver.supports_secondary:
                raise Exception(f'{driver} cannot be used as a secondary driver')
            T, cost = driver.run(x=x, y=y, T=T)
            self.T = T
            self.cost = cost

        return cost, T

    def positions_to_xy(self):
        return [np.concatenate((self.substation_positions[i], self.turbine_positions[i]), axis=0) for i in [0, 1]]

    def tree_as_table(self):
        tree_table = pd.DataFrame(self.T, columns=self.columns)
        tree_table = tree_table.astype({'from_node': int,
                                        'to_node': int,
                                        'cable_type': int,
                                        'nwt_connected': int})
        return tree_table

    def get_edges(self, x, y):
        return 'edges'

    def plot(self):
        x, y = self.positions_to_xy()
        plot_network(x, y, self.cables, self.T)


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
                    'crossing_constraint': True,
                    'max_it': 20000,
                    'repair': True}
        cables = np.array([[500, 3, 100000], [800, 5, 150000], [1000, 10, 250000]])
        wfn = WindFarmNetwork(turbine_positions=turbine_positions,
                              substation_positions=substation_positions,
                              drivers=[TwoStepHeuristicDriver(**settings), Planarize()],
                              # sequence=[0, 2, 1],
                              cables=cables)
        cost, state = wfn.design()
        wfn.plot()
        print(wfn.tree_as_table())
        print(cost)


main()
