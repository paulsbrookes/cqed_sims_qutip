from qutip import *
import matplotlib.pyplot as plt
import numpy as np

def mean_photons(probabilities):
    photons = np.zeros(probabilities.shape[1])
    for i in range(photons.shape[0]):
        for j in range(probabilities.shape[0]):
            photons[i] = photons[i] + j * probabilities[j, i]
    return photons

folder_path = '/homes/pbrookes/PycharmProjects/cqed_sims_qutip/monte/results/decoherence/2017-02-18--18-51-50'
expectations_path = folder_path + '/expectations'
expectations = qload(expectations_path)
times_path = folder_path + '/times'
times = qload(times_path)
times = times / (2 * np.pi * 1000)
probabilities = np.array([expectations[x] for x in range(0, len(expectations))]);
print np.max(probabilities)
n_snaps = probabilities.shape[1]
c_levels = probabilities.shape[0]
levels_array = np.linspace(0, c_levels - 1, c_levels)
probabilities_array = probabilities[:, n_snaps - 1]

plt.subplot(3,1,1)
plt.bar(levels_array, probabilities_array)
plt.xlabel('Cavity level');
plt.ylabel('Probability');

plt.subplot(3, 1, 2)
plt.pcolor(probabilities)
plt.xlabel('Snapshot');
plt.ylabel('Photons');

plt.subplot(3, 1, 3)
photons = mean_photons(probabilities)
plt.plot(times, photons)
plt.xlabel('Time');
plt.ylabel('Photons');


plt.show()