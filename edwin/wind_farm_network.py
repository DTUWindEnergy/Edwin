import matplotlib.pyplot as plt


class WindFarmNetwork():
    def __init__(self, method, geometry, financial, electrical, constraints):
        self.method = method
        self.geometry = geometry
        self.financial = financial
        self.electrical = electrical
        self.constraints = constraints

    def design(self, settings):
        self.settings = settings
        return self.method._design(self.geometry, self.financial, self.electrical,
                                   self.constraints, **settings)

    def plot(self):
        x = self.geometry['turbine_coordinates']['x']
        y = self.geometry['turbine_coordinates']['y']
        xss = self.geometry['sub_station_coordinates']['x']
        yss = self.geometry['sub_station_coordinates']['y']
        plt.plot(x, y, '.')
        plt.plot(xss, yss, 'or', label='Sub station')
        plt.legend()


def main():
    if __name__ == '__main__':
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
        result_dict = wfn.design(settings)
        print(result_dict)
        wfn.plot()


main()
