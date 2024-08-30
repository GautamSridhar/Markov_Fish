import numpy as np

ms_range = np.arange(4)
l_range = np.arange(4)

all_indices = []
for ms in ms_range:
    for k in l_range:
        all_indices.append([ms,k])

all_indices = np.array(np.vstack(all_indices),dtype=int)
np.savetxt('iteration_indices_l_hunt.txt',all_indices,fmt='%i')
print(len(all_indices))
