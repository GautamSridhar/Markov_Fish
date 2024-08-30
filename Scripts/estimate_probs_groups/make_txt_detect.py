import numpy as np

kg_range = np.arange(7)
l_range = np.arange(3)

all_indices = []
for kg in kg_range:
    for k in l_range:
        all_indices.append([kg,k])

all_indices = np.array(np.vstack(all_indices),dtype=int)
np.savetxt('iteration_indices_l_detect.txt',all_indices,fmt='%i')
print(len(all_indices))
