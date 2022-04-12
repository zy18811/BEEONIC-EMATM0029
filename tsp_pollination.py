import numpy as np
from field_setup import Field
from swarm_tsp import GeneticAlgorithm, read_flowers
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

class pollenSwarm:
    def __init__(self, size, field):
        field.found_flowers = field.unfound_flowers  # hack for when 1st step isnt done

        gen = GeneticAlgorithm(cities=read_flowers(field.found_flowers), iterations=1, population_size=100,
                               elites_num=20, mutation_rate=0.008, greedy_seed=1,
                               roulette_selection=True, plot_progress=False)
        gen.run()
        best_loop = gen.best_chromosome()
        self.loop = best_loop
        self.reverse_loop = self.loop[::-1]
        self.agents = np.zeros((size, 2))
        self.speed = 0.5
        self.size = size
        self.n_found = len(field.found_flowers)
        self.start_ixs = np.random.choice(range(self.n_found), size=size, replace=False)
        self.routes = [list(np.roll(best_loop, -ix)) for ix in self.start_ixs]
        self.directions = np.random.randint(2, size=size)  # 0 = clockwise, 1 = anticlockwise
        self.visited = [[] for _ in range(self.size)]

        for i in range(self.size):
            if self.directions[i] == 1:
                self.routes[i] = list(np.roll(self.routes[i], -1))
            self.routes[i].append(self.routes[i][0])

        self.map = field
        self.steps = np.zeros(self.size, dtype=int)
        self.pause_counts = np.zeros(self.size)
        self.pause = np.full(self.size, False, dtype=bool)

    def spawn_agents(self):
        x = np.full(self.size, 1.0) + np.random.normal(0, 1, self.size)
        y = np.full(self.size, 1.0) + np.random.normal(0, 1, self.size)
        self.agents = np.stack((x, y), axis=1)

    def iterate(self):
        for i, (x_agent, y_agent) in enumerate(self.agents):

            if len(self.visited[i]) == len(self.map.found_flowers):
                x_dest, y_dest = 0, 0
                v = np.array([x_dest - x_agent, y_dest - y_agent])
                v /= np.linalg.norm(v)
                self.agents[i] += v
            else:
                if not self.pause[i]:

                    route = self.routes[i]
                    target = route[self.steps[i]]
                    x_dest, y_dest = target.x, target.y

                    v = np.array([x_dest - x_agent, y_dest - y_agent])

                    if np.linalg.norm(v) <= self.speed:
                        self.visited[i].append((x_dest, y_dest))
                        self.agents[i] = x_dest, y_dest
                        self.next_flower(i)

                        self.steps[i] += 1
                        self.pause[i] = True
                    else:
                        v /= np.linalg.norm(v)
                        self.agents[i] += v
                else:
                    self.pause_counts[i] += 1
                    if self.pause_counts[i] >= 5:
                        self.pause[i] = False
                        self.pause_counts[i] = 0

    def next_flower(self, i):
        route = self.routes[i]

        current_flower = route[self.steps[i]]

        #current_flower = np.array([current_flower.x, current_flower.y])

        next_flower = route[self.steps[i]+1]
        next_flower = np.array([next_flower.x, next_flower.y])

        if np.random.uniform() < 0.9:
            pass
        else:
            # roulette select by distance thats not current or next
            visited = np.array(self.visited[i])
            flowers = np.array(self.map.found_flowers)
            flowers = np.delete(flowers, np.where((flowers == next_flower).all(1))[0], axis=0)
            unvisited = np.delete(flowers, np.where((flowers == visited[:, None]).all(-1))[1], axis=0)

            dists = [np.sqrt((current_flower.x-a)**2+(current_flower.y-b)**2) for a, b in unvisited]

            max_minus_dists = np.max(dists) - dists
            dist_w = max_minus_dists/np.sum(max_minus_dists)

            choice_ix = np.random.choice(range(len(dists)), p=dist_w)
            #choice_ix = np.argmin(dists)
            choice = unvisited[choice_ix]

            np_route = np.array([[f.x, f.y] for f in route])
            root_ix = np.where((np_route == choice).all(1))[0][0]
            self.routes[i] = np.roll(route, -root_ix+self.steps[i]+1)


if __name__ == '__main__':
    field = Field(50, 50, 50, 100)
    field.found_flowers = field.unfound_flowers

    pollinators = pollenSwarm(1, field)
    pollinators.spawn_agents()
    while True:
        pollinators.iterate()





