import numpy as np

flower = np.array([[1,1], [2,3], [7,7]])
agents = np.rint(np.array([[1,1],[2.2,3.1],[4.1,5.6]])).astype(int)

print(np.concatenate((flower,agents)))