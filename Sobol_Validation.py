from scipy.stats import qmc
import numpy as np
import matplotlib.pyplot as plt


random_seed = 42
np.random.seed(random_seed)
m_lst = [m + 2 for m in range(13)]
discrepancy_list = []
number_of_samples_list = []

for m in m_lst:
    sobol_sequence = qmc.Sobol(d=1, seed=42)
    sequence = sobol_sequence.random_base2(m=m)
    discrepancy_list.append(qmc.discrepancy(sequence))
    number_of_samples_list.append(2**m)


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.discrepancy.html
plt.plot(m_lst, discrepancy_list, label='1D Sobol Sample')
plt.ylabel('Sobol Distribution Discrepancy [-]', fontsize=16)
plt.xlabel(r'Base 2 logarithm of number of samples [-]', fontsize=16)


random_seed = 42
np.random.seed(random_seed)
m_lst = [m + 2 for m in range(13)]
discrepancy_list = []
number_of_samples_list = []

for m in m_lst:
    sobol_sequence = qmc.Sobol(d=4, seed=42)
    sequence = sobol_sequence.random_base2(m=m)
    discrepancy_list.append(qmc.discrepancy(sequence))
    number_of_samples_list.append(2**m)

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.discrepancy.html
plt.plot(m_lst, discrepancy_list, label='4D Sobol Sample')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend()
plt.tight_layout()
plt.grid()
plt.savefig('./Figures/SobolDiscrepancy.pdf')