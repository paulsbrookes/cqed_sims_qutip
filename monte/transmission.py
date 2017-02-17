from qutip import *
import matplotlib.pyplot as plt
import numpy as np

folder_path = '/homes/pbrookes/PycharmProjects/cqed_sims_qutip/monte/results/test/2017-02-16--21-24-48'
expectations_path = folder_path + '/expectations'
expectations = qload(expectations_path)
abs_transmissions = np.absolute(expectations[0])
plt.plot(abs_transmissions)
plt.show()