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
    def __init__(self, wd_points, transmissions, threshold = 0.05):
        self.threshold = threshold
        self.wd_points = wd_points
        self.new_wd_points_unique = None
        self.transmissions = transmissions
        self.n_points = transmissions.size
        self.curvature_positions, self.curvatures = derivative(wd_points, transmissions, 2)
        self.mean_curvatures = moving_average(np.absolute(self.curvatures), 2)
        #self.midpoint_curvatures = \
        #    np.concatenate((self.curvatures[0], self.mean_curvatures, self.curvatures[self.n_points - 3]))
        self.midpoint_curvatures = \
            np.concatenate((np.array([self.curvatures[0]]), self.mean_curvatures))
        self.midpoint_curvatures = \
            np.concatenate((self.midpoint_curvatures, np.array([self.curvatures[self.n_points - 3]])))
        self.midpoint_transmissions = moving_average(self.transmissions, 2)
        self.midpoint_curvatures_normed = self.midpoint_curvatures / self.midpoint_transmissions
        self.midpoints = moving_average(self.wd_points, 2)
        self.intervals = np.diff(self.wd_points)
        self.num_of_sections_required = \
            np.ceil(self.intervals * np.sqrt(self.midpoint_curvatures_normed / threshold))

    def new_points(self):
        new_wd_points = np.array([])
        for index in np.arange(self.n_points - 1):
            multi_section = \
                np.linspace(self.wd_points[index], self.wd_points[index + 1], self.num_of_sections_required[index] + 1)
            new_wd_points = np.concatenate((new_wd_points, multi_section))
        unique_set = set(new_wd_points) - set(self.wd_points)
        self.new_wd_points_unique = np.array(list(unique_set))
        return self.new_wd_points_unique


def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    averages = np.convolve(interval, window, 'same')
    return averages[window_size - 1 : averages.size]

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

def sweep(eps, wd_lower, wd_upper, params, threshold):
    params.eps = eps
    wd_points = np.linspace(wd_lower, wd_upper, 10)
    transmissions = transmission_calc_array(params, wd_points)
    abs_transmissions = np.absolute(transmissions)
    curvature_info = CurvatureInfo(wd_points, abs_transmissions, threshold)
    new_wd_points = curvature_info.new_points()

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
        curvature_info = CurvatureInfo(wd_points, abs_transmissions, threshold)
        new_wd_points = curvature_info.new_points()

    results = Results(wd_points, transmissions, abs_transmissions, params)
    return results

def multi_sweep(eps_array, wd_lower, wd_upper, params, threshold):
    multi_results_dict = dict()

    #multi_results_list = parallel_map(sweep, eps_array, (wd_lower, wd_upper, params, fidelity), num_cpus = 2)
    #for index, eps in enumerate(eps_array):
    #    multi_results_dict[eps] = multi_results_list[index]

    for eps in eps_array:
        multi_results_dict[eps] = sweep(eps, wd_lower, wd_upper, params, threshold)

    return multi_results_dict


if __name__ == '__main__':
    #wc, wq, eps, g, chi, kappa, gamma, t_levels, c_levels
    params = Parameters(10.3641, 9.4914, 0.0001, 0.389, -0.097, 0.00146, 0.000833, 2, 10)
    eps = 0.0001
    threshold = 0.001
    wd_lower = 10.4
    wd_upper = 10.55
    eps_array = np.linspace(0.0001, 0.0002, 2)
    multi_results = multi_sweep(eps_array, wd_lower, wd_upper, params, threshold)
    results = multi_results[0.0002]
    plt.scatter(results.wd_points, results.abs_transmissions)
    plt.show()
