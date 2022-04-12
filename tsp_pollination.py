import numpy as np



class pollenSwarm:
    def __init__(self, size, field):
        self.agents = np.array([])
        self.speed = 0.5
        self.size = size
        self.routes = []
        self.map = field

    def spawn_agents(self):
        self.agents = np.zeros((self.size, 2))



        x = np.full(self.size, 1.0) + np.random.normal(0, 1, self.size)
        y = np.full(self.size, 1.0) + np.random.normal(0, 1, self.size)

        self.agents = np.stack((x, y), axis=1)