import numpy as np
from field_setup import Field
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


def avoidance(agents, map):
    size = len(agents)
    # Compute vectors between agents and wall planes
    diffh = np.array([map.planeh - agents[n][1] for n in range(size)])
    diffv = np.array([map.planev - agents[n][0] for n in range(size)])

    # split agent positions into x and y arrays
    agentsx = agents.T[0]
    agentsy = agents.T[1]

    # Check intersection of agents with walls
    low = agentsx[:, np.newaxis] >= map.limh.T[0]
    up = agentsx[:, np.newaxis] <= map.limh.T[1]
    intmat = up * low

    # Compute force based vector and multiply by intersection matrix
    Fy = np.exp(-2 * abs(diffh) + 5)
    Fy = Fy * diffh * intmat

    low = agentsy[:, np.newaxis] >= map.limv.T[0]
    up = agentsy[:, np.newaxis] <= map.limv.T[1]
    intmat = up * low

    Fx = np.exp(-2 * abs(diffv) + 5)
    Fx = Fx * diffv * intmat

    # Sum the forces between every wall into one force.
    Fx = np.sum(Fx, axis=1)
    Fy = np.sum(Fy, axis=1)
    # Combine x and y force vectors
    F = np.array([[Fx[n], Fy[n]] for n in range(size)])
    return F


def spiral(x, y, speed):
    r = 0.1
    s = speed

    theta = np.sqrt(x**2 + y**2) / r
    f = s / np.sqrt(1 + theta**2)
    dx = f * (x - theta * y)
    dy = f * (theta * x + y)
    if theta == 0: dx = 1

    dx, dy = np.array([dx, dy]) / np.linalg.norm([dx, dy])
    return x + dx, y + dy


def random_choice(swarm):
    pass



def random_walk(swarm, param):
    alpha = 0.01
    beta = 50

    noise = param * np.random.randint(-beta, beta, (swarm.size))
    swarm.headings += noise

    # Calculate new heading vector
    gx = 1 * np.cos(swarm.headings)
    gy = 1 * np.sin(swarm.headings)
    G = -np.array([[gx[n], gy[n]] for n in range(0, swarm.size)])

    # Agent avoidance
    R = 20
    r = 2
    A = 1
    a = 20

    a = np.zeros((swarm.size, 2))

    # mag = cdist(swarm.agents, swarm.agents)

    # # Compute vectors between agents
    # diff = swarm.agents[:,:,np.newaxis]-swarm.agents.T[np.newaxis,:,:]

    # R = 5; r = 5
    # repel = R*r*np.exp(-mag/r)[:,np.newaxis,:]*diff/(swarm.size-1)
    # repel = np.sum(repel, axis = 0).T

    B = np.zeros((swarm.size, 2))
    # B = beacon(swarm)
    A = avoidance(swarm.agents, swarm.map)
    a += A + G + B

    vecx = a.T[0]
    vecy = a.T[1]

    angles = np.arctan2(vecy, vecx)
    Wx = swarm.speed * np.cos(angles)
    Wy = swarm.speed * np.sin(angles)

    W = -np.stack((Wx, Wy), axis=1)
    swarm.agents += W


class scoutSwarm:
    def __init__(self, size, field):
        self.agents = []
        self.speed = 0.5
        self.size = size

        self.param = 3
        self.map = field

        self.headings = []

        self.agent_modes = []
        self.agent_destinations = []
        self.agent_spiral_count = []

    def spawn_agents(self):
        self.dead = np.zeros(self.size)
        self.agents = np.zeros((self.size, 2))
        self.headings = 0.0314 * np.random.randint(-100, 100, self.size)
        self.agent_destinations = np.array([[np.random.choice(range(self.map.xlim)),
                                             np.random.choice(range(self.map.ylim))] for _ in range(self.size)])

        self.agent_modes = ['random_destination'] * self.size
        self.agent_spiral_count = np.zeros(self.size)

        x = np.full(self.size, 1.0)
        y = np.full(self.size, 1.0)

        self.agents = np.stack((x, y), axis=1)
        self.shadows = np.zeros((4, self.size, 2))

    def check_found(self):
        for i, agent in enumerate(self.agents):
            unfound = self.map.unfound_flowers
            dists = np.linalg.norm(unfound-agent, axis=1)
            found_ixs = np.where(dists <= 1)[0]

            if found_ixs.size == 0:
                pass
                #self.agent_modes[i] = 'spiral'

            if self.map.found_flowers is None:
                self.map.found_flowers = unfound[found_ixs]
            else:
                self.map.found_flowers = np.concatenate((self.map.found_flowers, unfound[found_ixs]))
            self.map.unfound_flowers = np.delete(unfound, found_ixs, 0)

    def iterate(self):
        for i, (x_agent, y_agent) in enumerate(self.agents):
            mode = self.agent_modes[i]
            if mode == 'random_destination':
                x_dest, y_dest = self.agent_destinations[i]


                if x_dest == x_agent and y_dest == y_agent:
                    self.agent_destinations[i] = [np.random.choice(range(self.map.xlim)),
                                                  np.random.choice(range(self.map.ylim))]
                    x_dest, y_dest = self.agent_destinations[i]

                v = np.array([x_dest-x_agent, y_dest-y_agent])
                if np.linalg.norm(v) <= self.speed:
                    self.agents[i] = self.agent_destinations[i]
                else:
                    v /= np.linalg.norm(v)
                    self.agents[i] += v

            elif mode == 'spiral':
                x_spiral, y_spiral = spiral(x_agent, y_agent, self.speed)
                if self.agent_spiral_count[i] == 9 or \
                        x_spiral < 0 or x_spiral > self.map.xlim or y_spiral < 0 or y_spiral > self.map.ylim:
                    self.agent_modes[i] = 'random_destination'
                else:
                    self.agents[i] += np.array([x_spiral, y_spiral])
                    self.agent_spiral_count[i] += 1

        self.check_found()


if __name__ == '__main__':
    f = Field(100, 100, 3, 100)
    s = scoutSwarm(3, f)
    s.spawn_agents()
    print(s.agents)
    s.iterate()
    print(s.agents)




