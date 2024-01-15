from model import *
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy


POPULATION_SIZE = 300
P_CROSSOVER = 0.9
P_MUTATION = 0.9
MAX_GENERATIONS = 100

random.seed(21)
np.random.seed(21)


def create_individual():
    abs_weight = 0.2
    abs_bias = 10
    weights_1 = 2 * abs_weight * np.random.rand(20, 20) - abs_weight
    biases_1 = 2 * abs_bias * np.random.rand(20) - abs_bias
    weights_2 = 2 * abs_weight * np.random.rand(20, 20) - abs_weight
    biases_2 = 2 * abs_bias * np.random.rand(20) - abs_bias
    weights_3 = 2 * abs_weight * np.random.rand(20, 10) - abs_weight
    biases_3 = 2 * abs_bias * np.random.rand(10) - abs_bias
    weights = np.array([weights_1, biases_1, weights_2, biases_2, weights_3, biases_3], dtype=object)
    return weights


population = []
for i in range(POPULATION_SIZE):
    population.append(create_individual())

max_fit = []
avg_fit = []
min_fit = []

generations_count = 0

with tqdm(total=MAX_GENERATIONS) as pbar:
    while generations_count < MAX_GENERATIONS:
        offsprings = []
        population = sorted(population, key=lambda x: - run_model(x))
        elite = 2
        for i in range(elite):
            offsprings.append(population[i])

        for i in range(elite, POPULATION_SIZE):
            i1 = i2 = i3 = 0
            while i1 == i2 or i2 == i3 or i3 == i1:
                i1 = random.randint(0, POPULATION_SIZE - 1)
                i2 = random.randint(0, POPULATION_SIZE - 1)
                i3 = random.randint(0, POPULATION_SIZE - 1)
            offsprings.append(deepcopy(max(population[i1], population[i2], population[i3], key=lambda x: run_model(x))))

        for i in range(elite, POPULATION_SIZE // 2 + elite // 2):
            if random.random() < P_CROSSOVER:
                parent1 = offsprings[i]
                parent2 = offsprings[i + POPULATION_SIZE // 2 - elite // 2]
                for arr in range(len(parent1)):
                    if random.random() < 0.1:
                        if arr % 2:
                            for bias in range(len(parent1[arr])):
                                if random.random() < 0.05:
                                    alpha = 1
                                    a = parent1[arr][bias] - alpha * (parent2[arr][bias] - parent1[arr][bias])
                                    b = parent2[arr][bias] + alpha * (parent2[arr][bias] - parent1[arr][bias])
                                    parent1[arr][bias] = random.uniform(a, b)
                                    parent2[arr][bias] = random.uniform(a, b)
                        else:
                            for arr2 in range(len(parent1[arr])):
                                for gene in range(len(parent1[arr][arr2])):
                                    if random.random() < 0.1:
                                        alpha = 0.5
                                        a = parent1[arr][arr2][gene] - alpha * \
                                            (parent2[arr][arr2][gene] - parent1[arr][arr2][gene])
                                        b = parent2[arr][arr2][gene] + alpha * \
                                            (parent2[arr][arr2][gene] - parent1[arr][arr2][gene])
                                        parent1[arr][arr2][gene] = random.uniform(a, b)
                                        parent2[arr][arr2][gene] = random.uniform(a, b)

        for i in range(elite, POPULATION_SIZE):
            if random.random() < P_MUTATION:
                for arr in range(len(offsprings[i])):
                    if random.random() < 0.1:
                        if arr % 2:
                            for bias in range(len(offsprings[i][arr])):
                                if random.random() < 0.1:
                                    betta = 5
                                    offsprings[i][arr][bias] = random.uniform(offsprings[i][arr][bias] - betta,
                                                                              offsprings[i][arr][bias] + betta)
                        else:
                            for arr2 in range(len(offsprings[i][arr])):
                                for gene in range(len(offsprings[i][arr][arr2])):
                                    if random.random() < 0.5:
                                        betta = 0.5
                                        offsprings[i][arr][arr2][gene] = \
                                            random.uniform(offsprings[i][arr][arr2][gene] - betta,
                                                           offsprings[i][arr][arr2][gene] + betta)

        fitnesses = []
        for offspring in offsprings:
            fitnesses.append(run_model(offspring))
        max_fit.append(max(fitnesses))
        min_fit.append(min(fitnesses))
        avg_fit.append(sum(fitnesses) / len(fitnesses))
        population = offsprings
        pbar.update(1)
        generations_count += 1

# for offspring in population:
#     for i in range(len(offspring[5])):
#         print(round(offspring[5][i], 4), end=' ')
#     print()

# max_fitness = run_model(population[0])
# for i in range(POPULATION_SIZE):
#     if max_fitness == run_model(population[i]):
#         model.set_weights(population[i])
#         y = model(x).numpy().tolist()[0]
#         print(y)

plt.plot(max_fit, color='green')
plt.plot(avg_fit, color='blue')
plt.plot(min_fit, color='red')

# for i in range(POPULATION_SIZE):
#     plt.scatter(i + 1, run_model(population[i]), color='purple', s=5)

plt.show()
