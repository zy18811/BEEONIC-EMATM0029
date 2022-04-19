"""
https://github.com/rameziophobia/Travelling_Salesman_Optimization
"""

import random
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from field_setup import Field
from tqdm import tqdm
import imageio


class Flower:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, flower):
        return math.hypot(self.x - flower.x, self.y - flower.y)

    def __repr__(self):
        return f"({self.x}, {self.y})"


def path_cost(route):
    return sum([flower.distance(route[index - 1]) for index, flower in enumerate(route)])


def read_flowers(flower_list):
    flowers = []
    for (flower_x, flower_y) in flower_list:
        flowers.append(Flower(flower_x, flower_y))
    return flowers


class Particle:
    def __init__(self, route, cost=None):
        self.route = route
        self.pbest = route
        self.current_cost = cost if cost else self.path_cost()
        self.pbest_cost = cost if cost else self.path_cost()
        self.velocity = []

    def clear_velocity(self):
        self.velocity.clear()

    def update_costs_and_pbest(self):
        self.current_cost = self.path_cost()
        if self.current_cost < self.pbest_cost:
            self.pbest = self.route
            self.pbest_cost = self.current_cost

    def path_cost(self):
        return path_cost(self.route)


class PSO:

    def __init__(self, iterations, population_size, gbest_probability=1.0, pbest_probability=1.0, cities=None):
        self.cities = cities
        self.gbest = None
        self.gcost_iter = []
        self.iterations = iterations
        self.population_size = population_size
        self.particles = []
        self.gbest_probability = gbest_probability
        self.pbest_probability = pbest_probability

        solutions = self.initial_population()
        self.particles = [Particle(route=solution) for solution in solutions]

    def random_route(self):
        return random.sample(self.cities, len(self.cities))

    def initial_population(self):
        random_population = [self.random_route() for _ in range(self.population_size - 1)]
        greedy_population = [self.greedy_route(0)]
        return [*random_population, *greedy_population]
        # return [*random_population]

    def greedy_route(self, start_index):
        unvisited = self.cities[:]
        del unvisited[start_index]
        route = [self.cities[start_index]]
        while len(unvisited):
            index, nearest_city = min(enumerate(unvisited), key=lambda item: item[1].distance(route[-1]))
            route.append(nearest_city)
            del unvisited[index]
        return route

    def run(self):
        self.gbest = min(self.particles, key=lambda p: p.pbest_cost)
        print(f"initial cost is {self.gbest.pbest_cost}")
        plt.ion()
        plt.draw()
        for t in range(self.iterations):
            self.gbest = min(self.particles, key=lambda p: p.pbest_cost)
            if t % 20 == 0:
                plt.figure(0)
                plt.plot(pso.gcost_iter, 'g')
                plt.ylabel('Distance')
                plt.xlabel('Generation')
                fig = plt.figure(0)
                fig.suptitle('pso iter')
                x_list, y_list = [], []
                for city in self.gbest.pbest:
                    x_list.append(city.x)
                    y_list.append(city.y)
                x_list.append(pso.gbest.pbest[0].x)
                y_list.append(pso.gbest.pbest[0].y)
                fig = plt.figure(1)
                fig.clear()
                fig.suptitle(f'pso TSP iter {t}')

                plt.plot(x_list, y_list, 'ro')
                plt.plot(x_list, y_list, 'g')
                plt.draw()
                plt.pause(.001)
            self.gcost_iter.append(self.gbest.pbest_cost)

            for particle in self.particles:
                particle.clear_velocity()
                temp_velocity = []
                gbest = self.gbest.pbest[:]
                new_route = particle.route[:]

                for i in range(len(self.cities)):
                    if new_route[i] != particle.pbest[i]:
                        swap = (i, particle.pbest.index(new_route[i]), self.pbest_probability)
                        temp_velocity.append(swap)
                        new_route[swap[0]], new_route[swap[1]] = \
                            new_route[swap[1]], new_route[swap[0]]

                for i in range(len(self.cities)):
                    if new_route[i] != gbest[i]:
                        swap = (i, gbest.index(new_route[i]), self.gbest_probability)
                        temp_velocity.append(swap)
                        gbest[swap[0]], gbest[swap[1]] = gbest[swap[1]], gbest[swap[0]]

                particle.velocity = temp_velocity

                for swap in temp_velocity:
                    if random.random() <= swap[2]:
                        new_route[swap[0]], new_route[swap[1]] = \
                            new_route[swap[1]], new_route[swap[0]]

                particle.route = new_route
                particle.update_costs_and_pbest()


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def path_cost(self):
        if self.distance == 0:
            distance = 0
            for index, city in enumerate(self.route):
                distance += city.distance(self.route[(index + 1) % len(self.route)])
            self.distance = distance
        return self.distance

    def path_fitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.path_cost())
        return self.fitness


class GeneticAlgorithm:
    def __init__(self, iterations, population_size, cities, elites_num, mutation_rate,
                 greedy_seed=0, roulette_selection=True, plot_progress=True):
        self.plot_progress = plot_progress
        self.roulette_selection = roulette_selection
        self.progress = []
        self.mutation_rate = mutation_rate
        self.cities = cities
        self.elites_num = elites_num
        self.iterations = iterations
        self.population_size = population_size
        self.greedy_seed = greedy_seed

        self.population = self.initial_population()
        self.average_path_cost = 1
        self.ranked_population = None

    def best_chromosome(self):
        return self.ranked_population[0][0]

    def best_distance(self):
        return 1 / self.ranked_population[0][1]

    def random_route(self):
        return random.sample(self.cities, len(self.cities))

    def initial_population(self):
        p1 = [self.random_route() for _ in range(self.population_size - self.greedy_seed)]
        greedy_population = [greedy_route(start_index % len(self.cities), self.cities)
                             for start_index in range(self.greedy_seed)]
        return [*p1, *greedy_population]

    def rank_population(self):
        fitness = [(chromosome, Fitness(chromosome).path_fitness()) for chromosome in self.population]
        self.ranked_population = sorted(fitness, key=lambda f: f[1], reverse=True)

    def selection(self):
        selections = [self.ranked_population[i][0] for i in range(self.elites_num)]
        if self.roulette_selection:
            df = pd.DataFrame(np.array(self.ranked_population), columns=["index", "fitness"])
            self.average_path_cost = sum(1 / df.fitness) / len(df.fitness)
            df['cum_sum'] = df.fitness.cumsum()
            df['cum_perc'] = 100 * df.cum_sum / df.fitness.sum()

            for _ in range(0, self.population_size - self.elites_num):
                pick = 100 * random.random()
                for i in range(0, len(self.ranked_population)):
                    if pick <= df.iat[i, 3]:
                        selections.append(self.ranked_population[i][0])
                        break
        else:
            for _ in range(0, self.population_size - self.elites_num):
                pick = random.randint(0, self.population_size - 1)
                selections.append(self.ranked_population[pick][0])
        self.population = selections

    @staticmethod
    def produce_child(parent1, parent2):
        gene_1 = random.randint(0, len(parent1))
        gene_2 = random.randint(0, len(parent1))
        gene_1, gene_2 = min(gene_1, gene_2), max(gene_1, gene_2)
        child = [parent1[i] for i in range(gene_1, gene_2)]
        child.extend([gene for gene in parent2 if gene not in child])
        return child

    def generate_population(self):
        length = len(self.population) - self.elites_num
        children = self.population[:self.elites_num]
        for i in range(0, length):
            child = self.produce_child(self.population[i],
                                       self.population[(i + random.randint(1, self.elites_num)) % length])
            children.append(child)
        return children

    def mutate(self, individual):
        for index, city in enumerate(individual):
            if random.random() < max(0, self.mutation_rate):
                sample_size = min(min(max(3, self.population_size // 5), 100), len(individual))
                random_sample = random.sample(range(len(individual)), sample_size)
                sorted_sample = sorted(random_sample,
                                       key=lambda c_i: individual[c_i].distance(individual[index - 1]))
                random_close_index = random.choice(sorted_sample[:max(sample_size // 3, 2)])
                individual[index], individual[random_close_index] = \
                    individual[random_close_index], individual[index]
        return individual

    def next_generation(self):
        self.rank_population()
        self.selection()
        self.population = self.generate_population()
        self.population[self.elites_num:] = [self.mutate(chromosome)
                                             for chromosome in self.population[self.elites_num:]]

    def run(self):
        if self.plot_progress:
            plt.ion()
        for ind in tqdm(range(0, self.iterations)):
            self.next_generation()
            self.progress.append(self.best_distance())
            if self.plot_progress and ind % 10 == 0:
                self.plot(ind)
            elif not self.plot_progress and ind % 10 == 0:
                pass
                #print(self.best_distance())

    def plot(self, ind):
        #print(self.best_distance())
        #fig = plt.figure(0)
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), dpi=70, facecolor='w', edgecolor='k')
        ax2.plot(self.progress, 'g')
        ax2.set_ylabel('Distance')
        ax2.set_xlabel('Generation')

        x_list, y_list = [], []
        for city in self.best_chromosome():
            x_list.append(city.x)
            y_list.append(city.y)
        x_list.append(self.best_chromosome()[0].x)
        y_list.append(self.best_chromosome()[0].y)

        #fig.clear()
        flower_marker = "$\u273F$"
        ax1.plot(x_list, y_list, marker=flower_marker, markersize=10, color='g', linewidth=0,)
        ax1.plot(x_list, y_list, 'k--')
        ax1.legend([Line2D([], [], color='k', ls='--'),
                    Line2D([], [], marker=flower_marker, markersize=10, color='g', linewidth=0, )],
                   ['Route', 'Target'], loc='upper left', bbox_to_anchor=(0.3, 1.06), ncol=2)

        if self.plot_progress:
            #plt.draw()
            plt.savefig(f'{ind}.png')
            plt.pause(0.05)
        #plt.show()


def greedy_route(start_index, cities):
    unvisited = cities[:]
    del unvisited[start_index]
    route = [cities[start_index]]
    while len(unvisited):
        index, nearest_city = min(enumerate(unvisited), key=lambda item: item[1].distance(route[-1]))
        route.append(nearest_city)
        del unvisited[index]
    return route


if __name__ == '__main__':
    """
    field = Field(50, 50, 50, 100)
    flowers = read_flowers(field.unfound_flowers)

    # pso = PSO(iterations=1200, population_size=300, pbest_probability=0.9, gbest_probability=0.02, cities=flowers)
    # pso.run()
    gen = GeneticAlgorithm(cities=flowers, iterations=1000, population_size=100,
                           elites_num=20, mutation_rate=0.008, greedy_seed=1,
                           roulette_selection=True, plot_progress=True)
    gen.run()
    print(len(gen.best_chromosome()))
    """
    with imageio.get_writer('gen_algo.gif', mode='I') as writer:
        for filename in [f'{i}.png' for i in range(0,1000,10)]:
            image = imageio.imread(filename)
            writer.append_data(image)
