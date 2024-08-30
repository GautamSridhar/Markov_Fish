import numpy as np

state_range = np.arange(4)
sim_range = np.arange(100)

all_indices = []
for ms in state_range:
    for k in sim_range:
        all_indices.append([ms,k])

all_indices = np.array(np.vstack(all_indices),dtype=int)
np.savetxt('iteration_indices_strats.txt',all_indices,fmt='%i')
print(len(all_indices))
