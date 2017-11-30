#test1.py

import numpy as np

index_set = []

for i in range(0, 100):
    batch_mask = np.random.choice(10000, 100)
    index_set.extend(batch_mask)

index_set = set(index_set)

print(sorted(index_set))
print(len(index_set))