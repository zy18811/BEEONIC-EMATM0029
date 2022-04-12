import numpy as np
from field_setup import Field
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def waypoint_grid(x_lim, y_lim):
    n = 6
    p = 0.95

    x_grid_max = p*x_lim
    y_grid_max = p*y_lim

    x_grid_min = x_lim-x_grid_max
    y_grid_min = y_lim-y_grid_max

    X, Y = np.meshgrid(np.linspace(x_grid_min, x_grid_max, n), np.linspace(y_grid_min, y_grid_max, n))
    X[1::2, :] = X[1::2, ::-1]
    return np.hstack([X.ravel().reshape(-1, 1), Y.ravel().reshape(-1, 1)])








class scoutSwarm:
    def __init__(self, size, field):
        self.agents = np.array([])
        self.speed = 0.5
        self.size = size

        self.map = field

        self.agent_modes = []
        self.agent_destinations = []
        self.agent_spiral_count = []
        self.agent_spiral_origin = np.empty((self.size, 2))
        self.waypoints = waypoint_grid(self.map.xlim, self.map.ylim)
        self.waypoint_counter = 0
        self.flocked = np.full(self.size, False)

    def spawn_agents(self):
        self.agents = np.zeros((self.size, 2))
        self.agent_destinations = [self.map.unchecked[ix] for ix in np.random.choice(range(len(self.map.unchecked)),
                                                                                     self.size, replace=False)]

        self.agent_modes = ['waypoint'] * self.size
        self.agent_spiral_count = np.zeros(self.size)

        x = np.full(self.size, 1.0) + np.random.normal(0, 1, self.size)
        y = np.full(self.size, 1.0) + np.random.normal(0, 1, self.size)

        self.agents = np.stack((x, y), axis=1)


    def check_found(self):
        for i, agent in enumerate(self.agents):
            unfound = self.map.unfound_flowers
            dists = np.linalg.norm(unfound-agent, axis=1)
            found_ixs = np.where(dists <= 1.5)[0]

            if found_ixs.size != 0:
                if self.agent_modes[i] != 'spiral':
                    self.agent_modes[i] = 'spiral'
                    self.agent_spiral_origin[i] = agent

            if self.map.found_flowers is None:
                self.map.found_flowers = unfound[found_ixs]
            else:
                self.map.found_flowers = np.concatenate((self.map.found_flowers, unfound[found_ixs]))
            self.map.unfound_flowers = np.delete(unfound, found_ixs, 0)

    def iterate(self):
        for agent in self.agents:
            try:
                self.map.unchecked.remove(tuple(np.rint(agent)))
            except ValueError:
                pass

        flock(self)

        for i, (x_agent, y_agent) in enumerate(self.agents):
            if self.flocked[i]:
                self.flocked[i] = False
            else:
                mode = self.agent_modes[i]

                if mode == 'waypoint':
                    if self.waypoint_counter >= 36:
                        dists = np.linalg.norm(self.map.unchecked - np.array([x_agent, y_agent]), axis=1)
                        self.waypoints = np.append(self.waypoints, [self.map.unchecked[np.argmax(dists)]], axis=0)

                    x_dest, y_dest = self.waypoints[self.waypoint_counter]
                    v = np.array([x_dest-x_agent, y_dest-y_agent])

                    if np.linalg.norm(v) <= 3:
                        #self.agents[i] = x_dest, y_dest
                        self.waypoint_counter += 1
                    #else:
                    v/= np.linalg.norm(v)
                    self.agents[i] += v

                if mode == 'random_destination':
                    x_dest, y_dest = self.agent_destinations[i]

                    if x_dest == x_agent and y_dest == y_agent:
                        self.agent_destinations[i] = self.map.unchecked[np.random.choice(range(len(self.map.unchecked)))]
                        x_dest, y_dest = self.agent_destinations[i]

                    v = np.array([x_dest-x_agent, y_dest-y_agent])
                    if np.linalg.norm(v) <= self.speed:
                        self.agents[i] = self.agent_destinations[i]
                    else:
                        v /= np.linalg.norm(v)
                        self.agents[i] += v

                elif mode == 'spiral':
                    x_o, y_o = self.agent_spiral_origin[i]
                    x_spiral, y_spiral = spiral(x_o, y_o, x_agent, y_agent, self.agent_spiral_count[i], self.speed)
                    if self.agent_spiral_count[i] == 200 or \
                            x_spiral < 0 or x_spiral > self.map.xlim or y_spiral < 0 or y_spiral > self.map.ylim:
                        self.agent_spiral_count[i] = 0
                        self.agent_modes[i] = 'waypoint'
                    else:
                        self.agents[i] = np.array([x_spiral, y_spiral])
                        self.agent_spiral_count[i] += 1

        self.check_found()


def flock(swarm):
    agent_dists = cdist(swarm.agents, swarm.agents)

    for i, agent in enumerate(swarm.agents):
        m = np.zeros(swarm.size, dtype=bool)
        m[i] = True

        neighbour_dists = np.ma.array(agent_dists[i, :], mask=m)
        closest_dist = np.min(neighbour_dists)
        closest_agent = swarm.agents[np.argmin(neighbour_dists)]

        v = (closest_agent-agent)/np.linalg.norm(closest_agent-agent)
        if closest_dist <= 5:
            swarm.agents[i] -= v
            swarm.flocked[i] = True
        elif closest_dist > 10:
            if swarm.agent_modes[i] != 'spiral':
                swarm.agents[i] += v
                swarm.flocked[i] = True


def archspiral(x, y, s, r):
    theta = np.sqrt(x**2+y**2)/r
    f = s/np.sqrt(1 + theta**2)
    dx = f*(1*x - theta*y)
    dy = f*(theta*x + 1*y)
    if theta == 0: dx = s

    dx, dy = (np.array([dx, dy])/np.linalg.norm([dx, dy]))*s
    x += dx
    y += dy

    return x, y


def spiral(x_o, y_o, x, y, s_count, s=1.0, r=0.00001):
    if s_count == 0:
        return x_o, y_o
    else:
        x -= x_o
        y -= y_o
    x_spiral, y_spiral = archspiral(x, y, s, r)
    return x_spiral + x_o, y_spiral + y_o


if __name__ == '__main__':
    f = Field(100, 100, 20, 100)
    s = scoutSwarm(3, f)
    s.spawn_agents()
    print(s.agents)
    for _ in range(10):
        s.iterate()
        print(s.agents)






