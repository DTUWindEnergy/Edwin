from abc import ABC, abstractmethod
from edwin.collection_system import collection_system
from edwin.c_mst_cables import plot_network
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

    def run(self):
        T, cables_cost = collection_system(self.wfn.x,
                                           self.wfn.y,
                                           self.option,
                                           self.Inters_const,
                                           self.max_it,
                                           self.wfn.cables)
        return T, cables_cost


class WindFarmNetwork():
    def __init__(self, layout, driver=HeuristicDriver(), cables=[]):
        self.layout = layout
        for k, v in layout.items():
            setattr(self, k, v)
        self.driver = driver
        self.cables = cables
        self.state = None
        self.T = None
        self.columns = ['from_node', 'to_node', 'cable_length', 'cable_type', 'cable_cost']
        self.setup()

    def setup(self):
        setattr(self.driver, 'wfn', self)

    def design(self):
        T, cost = self.driver.run()
        state = pd.DataFrame(T, columns=self.columns)
        state = state.astype({'from_node': int,
                              'to_node': int,
                              'cable_type': int})
        self.T = T
        self.cost = cost
        self.state = state
        return cost, state

    def plot(self):
        if self.state is not None:
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
        # layout = {'x': [473095,471790,471394,470998,470602,470207,469811,472523,469415,472132,471742,471351,470960,470569,470179,469788,472866,472480,472094,471708,471322,470937,470551,473594,473213,472833,472452,472071,471691,471310,470929],
        #           'y': [5992345,5991544,5991899,5992252,5992607,5992960,5993315,5991874,5993668,5992236,5992598,5992960,5993322,5993684,5994047,5994409,5992565,5992935,5993306,5993675,5994045,5994416,5994786,5992885,5993264,5993643,5994021,5994400,5994779,5995156,5995535]}
        layout = dict(x=np.array([0., 2000., 4000., 6000.,
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
        # layout = {'x': [387100, 383400, 383400, 383900, 383200, 383200, 383200, 383200, 383200, 383200, 383200, 383200, 383300, 384200, 384200, 384100, 384000, 383800, 383700, 383600, 383500, 383400, 383600, 384600, 385400, 386000, 386100, 386200, 386300, 386500, 386600, 386700, 386800, 386900, 387000, 387100, 387200, 383900, 387400, 387500, 387600, 387800, 387900, 388000, 387600, 386800, 385900, 385000, 384100, 384500, 384800, 385000, 385100, 385200, 385400, 385500, 385700, 385800, 385900, 385900, 385500, 385500, 386000, 386200, 386200, 384500, 386200, 386700, 386700, 386700, 384300, 384400, 384500, 384600, 384300, 384700, 384700, 384700, 385500, 384300, 384300],
        #           'y': [6109500, 6103800, 6104700, 6105500, 6106700, 6107800, 6108600, 6109500, 6110500, 6111500, 6112400, 6113400, 6114000, 6114200, 6115100, 6115900, 6116700, 6118400, 6119200, 6120000, 6120800, 6121800, 6122400, 6122000, 6121700, 6121000, 6120000, 6119100, 6118100, 6117200, 6116200, 6115300, 6114300, 6113400, 6112400, 6111500, 6110700, 6117600, 6108900, 6108100, 6107400, 6106300, 6105200, 6104400, 6103600, 6103600, 6103500, 6103400, 6103400, 6104400, 6120400, 6119500, 6118400, 6117400, 6116500, 6115500, 6114600, 6113500, 6112500, 6111500, 6105400, 6104200, 6110400, 6109400, 6108400, 6121300, 6107500, 6106400, 6105300, 6104400, 6113300, 6112500, 6111600, 6110800, 6110100, 6109200, 6108400, 6107600, 6106500, 6106600, 6105000]}

        settings = {'option': 3,
                    'Inters_const': True,
                    'max_it': 20000}
        cables = np.array([[500, 3, 100000], [800, 5, 150000], [1000, 10, 250000]])
        wfn = WindFarmNetwork(layout=layout,
                              driver=HeuristicDriver(**settings),
                              cables=cables)
        cost, state = wfn.design()
        wfn.plot()


main()
