import numpy as np
from qutip import *
from pylab import *
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import yaml


class Parameters:
    def __init__(self, wc, wq, eps, g, chi, kappa, gamma, t_levels, c_levels):
        self.wc = wc
        self.wq = wq
        self.eps = eps
        self.g = g
        self.chi = chi
        self.gamma = gamma
        self.kappa = kappa
        self.t_levels = t_levels
        self.c_levels = c_levels


class Results:
    def __init__(self, wd_points, transmissions, abs_transmissions, params):
        self.params = params
        self.wd_points = wd_points
        self.transmissions = transmissions
        self.abs_transmissions = abs_transmissions

class CurvatureInfo:
    def __init__(self, wd_points, transmissions):
        self.wd_points = wd_points
        self.transmissions = transmissions

def derivative(x, y, n_derivative = 1):
    
    derivatives = np.zeros(y.size - 1)
    positions = np.zeros(x.size - 1)
    for index in np.arange(y.size - 1):
        grad = (y[index + 1] - y[index]) / (x[index + 1] - x[index])
        position = np.mean([x[index], x[index + 1]])
        derivatives[index] = grad
        positions[index] = position

    if n_derivative > 1:
        positions, derivatives = derivative(positions, derivatives, n_derivative - 1)

    return positions, derivatives

def curvature_vector(wd_points, transmissions):

    is_ordered = all([wd_points[i] <= wd_points[i + 1] for i in xrange(len(wd_points) - 1)])
    assert is_ordered, "Vector of wd_points is not ordered."
    assert len(wd_points) == len(transmissions), "Vectors of wd_points and transmissions are not of equal length."

    metric_vector = []
    for index in range(len(wd_points) - 2):
        metric = curvature(wd_points[index:index + 3], transmissions[index:index + 3])
        metric_vector.append(metric)
    return metric_vector

def curvature(wd_triplet, transmissions_triplet):

    wd_delta_0 = wd_triplet[1] - wd_triplet[0]
    wd_delta_1 = wd_triplet[2] - wd_triplet[1]
    transmissions_delta_0 = transmissions_triplet[1] - transmissions_triplet[0]
    transmissions_delta_1 = transmissions_triplet[2] - transmissions_triplet[1]
    metric = 2 * (wd_delta_1 * transmissions_delta_1 - wd_delta_0 * transmissions_delta_0) / (wd_delta_0 + wd_delta_1)
    abs_normalised_metric = np.absolute(metric / transmissions_triplet[1])
    return abs_normalised_metric

def hamiltonian(params, wd):
    a = tensor(destroy(params.c_levels), qeye(params.t_levels))
    sm = tensor(qeye(params.c_levels), destroy(params.t_levels))
    H = - (params.wc - wd) * a.dag() * a - (params.wq - wd) * sm.dag() * sm \
        + params.chi * sm.dag() * sm * (sm.dag() * sm - 1) + params.g * (a.dag() * sm + a * sm.dag()) \
        + params.eps * (a + a.dag())
    return H

def transmission_calc_array(params, wd_points):

    transmissions = parallel_map(transmission_calc, wd_points, (params,), num_cpus = 10)
    transmissions = np.array(transmissions)

#    transmissions = [transmission_calc(wd, params) for wd in wd_points]
#    transmissions = np.array(transmissions)

    return transmissions

def transmission_calc(wd, params):

    a = tensor(destroy(params.c_levels), qeye(params.t_levels))
    sm = tensor(qeye(params.c_levels), destroy(params.t_levels))
    c_ops = []
    c_ops.append(np.sqrt(params.kappa) * a)
    c_ops.append(np.sqrt(params.gamma) * sm)
    H = hamiltonian(params, wd)
    rho_ss = steadystate(H, c_ops)
    transmission = expect(a, rho_ss)

    return transmission

def new_points(wd_points, transmissions, threshold):

    metric_vector = curvature_vector(wd_points, transmissions)
    indices = np.array([index for index, metric in enumerate(metric_vector) if metric > threshold]) + 1
    new_wd_points = generate_points(wd_points, indices)

    return new_wd_points

def generate_points(wd_points, indices):
    n_points = 6
    new_wd_points = np.array([])
    for index in indices:
        multi_section = np.linspace(wd_points[index - 1], wd_points[index + 1], n_points)
        new_wd_points = np.concatenate((new_wd_points, multi_section))
    unique_set = set(new_wd_points) - set(wd_points)
    new_wd_points_unique = np.array(list(unique_set))
    return new_wd_points_unique

def sweep(eps, wd_lower, wd_upper, params, fidelity):
    params.eps = eps
    wd_points = np.linspace(wd_lower, wd_upper, 10)
    transmissions = transmission_calc_array(params, wd_points)
    abs_transmissions = np.absolute(transmissions)
    new_wd_points = new_points(wd_points, abs_transmissions, fidelity)

    while (len(new_wd_points) > 0):
        new_transmissions = transmission_calc_array(params, new_wd_points)
        new_abs_transmissions = np.absolute(new_transmissions)
        wd_points = np.concatenate([wd_points, new_wd_points])
        transmissions = concatenate([transmissions, new_transmissions])
        abs_transmissions = concatenate([abs_transmissions, new_abs_transmissions])
        sort_indices = np.argsort(wd_points)
        wd_points = wd_points[sort_indices]
        transmissions = transmissions[sort_indices]
        abs_transmissions = abs_transmissions[sort_indices]
        new_wd_points = new_points(wd_points, abs_transmissions, fidelity)

    results = Results(wd_points, transmissions, abs_transmissions, params)
    return results

def multi_sweep(eps_array, wd_lower, wd_upper, params, fidelity):
    multi_results_dict = dict()

    #multi_results_list = parallel_map(sweep, eps_array, (wd_lower, wd_upper, params, fidelity), num_cpus = 2)
    #for index, eps in enumerate(eps_array):
    #    multi_results_dict[eps] = multi_results_list[index]

    for eps in eps_array:
        multi_results_dict[eps] = sweep(eps, wd_lower, wd_upper, params, fidelity)

    return multi_results_dict


if __name__ == '__main__':
    #wc, wq, eps, g, chi, kappa, gamma, t_levels, c_levels
    params = Parameters(10.3641, 9.4914, 0.0001, 0.389, -0.097, 0.00146, 0.000833, 2, 10)
    eps = 0.0001
    fidelity = 0.5
    wd_lower = 10.4
    wd_upper = 10.55
    eps_array = np.linspace(0.0001, 0.0002, 2)
    multi_results = multi_sweep(eps_array, wd_lower, wd_upper, params, fidelity)
    results = multi_results[0.0002]
    plt.plot(results.wd_points, results.abs_transmissions)
    plt.show()
