import numpy as np
from qutip import *
from pylab import *
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import yaml
from scipy.interpolate import interp1d

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
    def __init__(self, params, wd_points, transmissions):
        self.params = params
        self.wd_points = wd_points
        self.transmissions = transmissions
        self.abs_transmissions = np.absolute(self.transmissions)

    def concatenate(self, results):
        combined_params = np.concatenate([self.params, results.params])
        combined_wd_points = np.concatenate([self.wd_points, results.wd_points])
        combined_transmissions = np.concatenate([self.transmissions, results.transmissions])
        sort_indices = np.argsort(combined_wd_points)
        combined_params = combined_params[sort_indices]
        combined_wd_points = combined_wd_points[sort_indices]
        combined_transmissions = combined_transmissions[sort_indices]
        combined_results = Results(combined_params, combined_wd_points, combined_transmissions)
        return combined_results


class Queue:
    def __init__(self, params = None, wd_points = None):
        self.params = params
        self.wd_points = wd_points

    def curvature_generate(self, results, threshold = 0.05):
        curvature_info = CurvatureInfo(results, threshold)
        self.wd_points = curvature_info.new_points()
        self.params = hilbert_interpolation(self.wd_points, results)

class CurvatureInfo:
    def __init__(self, results, threshold = 0.05):
        self.threshold = threshold
        self.wd_points = results.wd_points
        self.new_wd_points_unique = None
        self.abs_transmissions = results.abs_transmissions
        self.n_points = self.abs_transmissions.size

    def new_points(self):
        self.curvature_positions, self.curvatures = derivative(self.wd_points, self.abs_transmissions, 2)
        self.mean_curvatures = moving_average(np.absolute(self.curvatures), 2)
        self.midpoint_curvatures = \
            np.concatenate((np.array([self.curvatures[0]]), self.mean_curvatures))
        self.midpoint_curvatures = \
            np.concatenate((self.midpoint_curvatures, np.array([self.curvatures[self.n_points - 3]])))
        self.midpoint_transmissions = moving_average(self.abs_transmissions, 2)
        self.midpoint_curvatures_normed = self.midpoint_curvatures / self.midpoint_transmissions
        self.midpoints = moving_average(self.wd_points, 2)
        self.intervals = np.diff(self.wd_points)
        self.num_of_sections_required = \
            np.ceil(self.intervals * np.sqrt(self.midpoint_curvatures_normed / threshold))
        new_wd_points = np.array([])
        for index in np.arange(self.n_points - 1):
            multi_section = \
                np.linspace(self.wd_points[index], self.wd_points[index + 1], self.num_of_sections_required[index] + 1)
            new_wd_points = np.concatenate((new_wd_points, multi_section))
        unique_set = set(new_wd_points) - set(self.wd_points)
        self.new_wd_points_unique = np.array(list(unique_set))
        return self.new_wd_points_unique

def hilbert_interpolation(new_wd_points, results):
    c_levels_array = np.array([params.c_levels for params in results.params])
    t_levels_array = np.array([params.t_levels for params in results.params])
    wd_points = results.wd_points
    c_interp = interp1d(wd_points, c_levels_array)
    t_interp = interp1d(wd_points, t_levels_array)
    base_params = results.params[0]
    params_list = []
    for wd in new_wd_points:
        new_params = base_params
        new_params.c_levels = int(round(c_interp(wd)))
        new_params.t_levels = int(round(t_interp(wd)))
        params_list.append(new_params)
    return params_list

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

def transmission_calc_array(queue):
    args = []
    for index, value in enumerate(queue.wd_points):
        args.append([value, queue.params[index]])
    transmissions = parallel_map(transmission_calc, args, num_cpus = 10)
    transmissions = np.array(transmissions)
    results = Results(queue.params, queue.wd_points, transmissions)

    return results

def transmission_calc(args):
    wd = args[0]
    params = args[1]
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
    params_list = [params for wd in wd_points]
    queue = Queue(params_list, wd_points)
    results = transmission_calc_array(queue)
    new_queue = Queue()
    new_queue.curvature_generate(results, threshold)

    while (len(new_queue.wd_points) > 0):
        new_results = transmission_calc_array(new_queue)
        results = results.concatenate(new_results)
        new_queue.curvature_generate(results, threshold)

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
    threshold = 0.01
    wd_lower = 10.4
    wd_upper = 10.55
    eps_array = np.linspace(0.0001, 0.0002, 2)
    multi_results = multi_sweep(eps_array, wd_lower, wd_upper, params, threshold)
    results = multi_results[0.0002]
    plt.scatter(results.wd_points, results.abs_transmissions)
    plt.show()
