import random
import matplotlib.pyplot as plt
import itertools
import statistics
import time

number_of_points = 10
side_of_field = 20
random.seed(21)


def length_of_path(permutation):
    result = 0
    for i in range(len(permutation)):
        result += ((permutation[i][0] - permutation[i - 1][0]) ** 2 +
                   (permutation[i][1] - permutation[i - 1][1]) ** 2) ** 0.5
    return result


start_time = time.time()

points = []
for i in range(number_of_points):
    points.append([random.randint(- side_of_field // 2, side_of_field // 2),
                   random.randint(- side_of_field // 2, side_of_field // 2)])

permutations = list(itertools.permutations(points))
path_lengths = []
for permutation in permutations:
    path_lengths.append(length_of_path(permutation))

with open('dataset.txt', 'w') as f:
    f.write(f'{number_of_points}\n')
    for point in points:
        f.write(f'{point[0]} {point[1]}\n')
        plt.scatter(point[0], point[1], color='red', s=5)
    f.write(f'{min(path_lengths)}\n')
    f.write(f'{max(path_lengths)}\n')
    f.write(f'{statistics.median(path_lengths)}\n')

end_time = time.time()
elapsed_time = end_time - start_time
print('Amount of permutations: ', len(permutations))
print('Elapsed time: ', round(elapsed_time, 3), 'seconds')

# plt.show()
