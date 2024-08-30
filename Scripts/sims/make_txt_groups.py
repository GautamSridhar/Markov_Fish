import numpy as np

group_range = np.arange(7)
sim_range = np.arange(100)

all_indices = []
for g in group_range:
    for k in sim_range:
        all_indices.append([g,k])

all_indices = np.array(np.vstack(all_indices),dtype=int)
np.savetxt('iteration_indices_groups.txt',all_indices,fmt='%i')
print(len(all_indices))
