import numpy as np
import matplotlib.pyplot as plt


def gen_flowers(x_lim, y_lim, n_clusts, n_flowers, seed=42):
    rng = np.random.default_rng(seed)
    centers = np.array([[rng.choice(range(x_lim)), rng.choice(range(y_lim))] for _ in range(n_clusts)])
    flower_std = 2

    def random_sum(n, m):
        r = rng.choice(range(1, m), n - 1, replace=False)
        r = np.diff(np.sort(np.append(r, [0, m])))
        return r

    flowers_per_clust = random_sum(n_clusts, n_flowers)

    def flower(x_bar, y_bar, x_std, y_std):
        def coord(bar, std, lim):
            c = rng.normal(bar, std)
            if c >= lim:
                return coord(bar, std, lim)
            else:
                return c

        flower_x = coord(x_bar, x_std, x_lim)
        flower_y = coord(y_bar, y_std, y_lim)

        return flower_x, flower_y

    flowers = np.empty((n_flowers, 2), dtype=np.int64)
    for i, (x, y) in enumerate(centers):
        n_flowers_clust = flowers_per_clust[i]
        flowers_done = np.sum(flowers_per_clust[:i])
        flowers[flowers_done:flowers_done + n_flowers_clust] = np.array([flower(x, y, flower_std, flower_std)
                                                                        for _ in range(n_flowers_clust)])

    return flowers


def gen_field(x_lim, y_lim, n_clusts, n_flowers, seed=42):
    field = np.zeros((x_lim, y_lim))
    flowers = gen_flowers(x_lim, y_lim, n_clusts, n_flowers, seed)
    field[flowers] = 1
    return field, flowers


class Field:
    def __init__(self, x_lim, y_lim, n_clusts, n_flowers, seed=42):
        self.field, self.unfound_flowers = gen_field(x_lim, y_lim, n_clusts, n_flowers, seed)
        self.found_flowers = None
        self.xlim = x_lim
        self.ylim = y_lim

        self.obsticles = []
        box = make_box(x_lim, y_lim, [x_lim/2, y_lim/2])
        [self.obsticles.append(box.walls[x]) for x in range(0, len(box.walls))]

        self.walls = np.zeros((2 * len(self.obsticles), 2))
        self.wallh = np.zeros((2 * len(self.obsticles), 2))
        self.wallv = np.zeros((2 * len(self.obsticles), 2))
        self.planeh = np.zeros(len(self.obsticles))
        self.planev = np.zeros(len(self.obsticles))
        self.limh = np.zeros((len(self.obsticles), 2))
        self.limv = np.zeros((len(self.obsticles), 2))

        for n in range(0, len(self.obsticles)):
            # if wall is vertical
            if self.obsticles[n].start[0] == self.obsticles[n].end[0]:
                self.wallv[2 * n] = np.array([self.obsticles[n].start[0], self.obsticles[n].start[1]])
                self.wallv[2 * n + 1] = np.array([self.obsticles[n].end[0], self.obsticles[n].end[1]])

                self.planev[n] = self.wallv[2 * n][0]
                self.limv[n] = np.array([np.min([self.obsticles[n].start[1], self.obsticles[n].end[1]]) - 0.5,
                                         np.max([self.obsticles[n].start[1], self.obsticles[n].end[1]]) + 0.5])

            # if wall is horizontal
            if self.obsticles[n].start[1] == self.obsticles[n].end[1]:
                self.wallh[2 * n] = np.array([self.obsticles[n].start[0], self.obsticles[n].start[1]])
                self.wallh[2 * n + 1] = np.array([self.obsticles[n].end[0], self.obsticles[n].end[1]])

                self.planeh[n] = self.wallh[2 * n][1]
                self.limh[n] = np.array([np.min([self.obsticles[n].start[0], self.obsticles[n].end[0]]) - 0.5,
                                         np.max([self.obsticles[n].start[0], self.obsticles[n].end[0]]) + 0.5])

            self.walls[2 * n] = np.array([self.obsticles[n].start[0], self.obsticles[n].start[1]])
            self.walls[2 * n + 1] = np.array([self.obsticles[n].end[0], self.obsticles[n].end[1]])


class make_wall(object):

    def __init__(self):

        self.start = np.array([0, 0])
        self.end = np.array([0, 0])
        self.width = 1
        self.hitbox = []


class make_box(object):

    def __init__(self, h, w, origin):

        self.height = h
        self.width = w
        self.walls = []

        self.walls.append(make_wall())
        self.walls[0].start = [origin[0]-(0.5*w), origin[1]+(0.5*h)]; self.walls[0].end = [origin[0]+(0.5*w), origin[1]+(0.5*h)]
        self.walls.append(make_wall())
        self.walls[1].start = [origin[0]-(0.5*w), origin[1]-(0.5*h)]; self.walls[1].end = [origin[0]+(0.5*w), origin[1]-(0.5*h)]
        self.walls.append(make_wall())
        self.walls[2].start = [origin[0]-(0.5*w), origin[1]+(0.5*h)]; self.walls[2].end = [origin[0]-(0.5*w), origin[1]-(0.5*h)]
        self.walls.append(make_wall())
        self.walls[3].start = [origin[0]+(0.5*w), origin[1]+(0.5*h)]; self.walls[3].end = [origin[0]+(0.5*w), origin[1]-(0.5*h)]


if __name__ == '__main__':
    f = Field(1000, 1000, 20, 100)
    plt.scatter(f.flowers[:, 0], f.flowers[:, 1], color='r')
    plt.show()