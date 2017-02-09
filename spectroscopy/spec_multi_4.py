import numpy as np
from qutip import *
from pylab import *
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import yaml
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

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
    def __init__(self, params, wd_points, transmissions, edge_occupations_c, edge_occupations_t):
        self.params = params
        self.wd_points = wd_points
        self.transmissions = transmissions
        self.edge_occupations_c = edge_occupations_c
        self.edge_occupations_t = edge_occupations_t
        self.abs_transmissions = np.absolute(self.transmissions)

    def concatenate(self, results):
        combined_params = np.concatenate([self.params, results.params])
        combined_wd_points = np.concatenate([self.wd_points, results.wd_points])
        combined_transmissions = np.concatenate([self.transmissions, results.transmissions])
        combined_edge_occupations_c = np.concatenate([self.edge_occupations_c, results.edge_occupations_c])
        combined_edge_occupations_t = np.concatenate([self.edge_occupations_t, results.edge_occupations_t])
        sort_indices = np.argsort(combined_wd_points)
        combined_params = combined_params[sort_indices]
        combined_wd_points = combined_wd_points[sort_indices]
        combined_transmissions = combined_transmissions[sort_indices]
        combined_edge_occupations_c = combined_edge_occupations_c[sort_indices]
        combined_edge_occupations_t = combined_edge_occupations_t[sort_indices]
        combined_results = Results(combined_params, combined_wd_points,
                                   combined_transmissions, combined_edge_occupations_c, combined_edge_occupations_t)
        return combined_results




class Queue:
    def __init__(self, params = [], wd_points = np.array([])):
        self.params = params
        self.wd_points = wd_points
        self.size = self.wd_points.size

    def curvature_generate(self, results, threshold = 0.05):
        curvature_info = CurvatureInfo(results, threshold)
        self.wd_points = curvature_info.new_points()
        self.params = hilbert_interpolation(self.wd_points, results)
        self.size = self.wd_points.size

    def hilbert_generate(self, results, threshold_c, threshold_t):
        self.wd_points = []
        self.params = []
        for index, params in enumerate(results.params):
            threshold_c_weighted = threshold_c / params.c_levels
            threshold_t_weighted = threshold_t / params.t_levels
            overload_c = (results.edge_occupations_c[index] > threshold_c_weighted)
            overload_t = (results.edge_occupations_t[index] > threshold_t_weighted)
            if overload_c or overload_t:
                self.wd_points.append(results.wd_points[index])
                params_copy = results.params[index]
                if overload_c:
                    params_copy.c_levels = \
                        size_correction(results.edge_occupations_c[index], params_copy.c_levels, threshold_c_weighted / 2)
                if overload_t:
                    params_copy.t_levels = \
                        size_correction(results.edge_occupations_t[index], params_copy.t_levels, threshold_t_weighted / 2)
                self.params.append(params_copy)
        self.wd_points = np.array(self.wd_points)
        self.size = self.wd_points.size


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


def size_correction(edge_occupation, size, threshold):
    beta = fsolve(zero_func, 0.01, args=(edge_occupation, size, size))
    new_size = 1 + np.log((1 - np.exp(-beta)) / threshold) / beta
    new_size = int(np.ceil(new_size))
    return new_size

def exponential_occupation(n, beta, size):
    factor = np.exp(-beta)
    f = np.power(factor, n) * (1 - factor) / (1 - np.power(factor, size))
    return f

def zero_func(beta, p, level, size):
    f = exponential_occupation(level - 1, beta, size)
    f = f - p
    return f

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
    steady_states = parallel_map(transmission_calc, args, num_cpus = 10)
    transmissions = np.array([steady_state[0] for steady_state in steady_states])
    edge_occupations_c = np.array([steady_state[1] for steady_state in steady_states])
    edge_occupations_c = np.absolute(edge_occupations_c)
    edge_occupations_t = np.array([steady_state[2] for steady_state in steady_states])
    edge_occupations_t = np.absolute(edge_occupations_t)
    results = Results(queue.params, queue.wd_points, transmissions, edge_occupations_c, edge_occupations_t)
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
    rho_c_ss = rho_ss.ptrace(0)
    rho_t_ss = rho_ss.ptrace(1)
    c_occupations = rho_c_ss.diag()
    t_occupations = rho_t_ss.diag()
    edge_occupation_c = c_occupations[params.c_levels - 1]
    edge_occupation_t = t_occupations[params.t_levels - 1]
    transmission = expect(a, rho_ss)

    return np.array([transmission, edge_occupation_c, edge_occupation_t])

def sweep(eps, wd_lower, wd_upper, params, threshold):
    params.eps = eps
    wd_points = np.linspace(wd_lower, wd_upper, 10)
    params_list = [params for wd in wd_points]
    queue = Queue(params_list, wd_points)
    results = transmission_calc_array(queue)
    new_queue = Queue()
    new_queue.curvature_generate(results, threshold)

    while (new_queue.size > 0):
        new_results = transmission_calc_array(new_queue)
        results = results.concatenate(new_results)

        new_queue.hilbert_generate(results, 0.01, 0.01)
        while (new_queue.size > 0):
            new_results = transmission_calc_array(new_queue)
            results = results.concatenate(new_results)
            new_queue.hilbert_generate(results, 0.01, 0.01)

        new_queue.curvature_generate(results, threshold)

    return results

def multi_sweep(eps_array, wd_lower, wd_upper, params, threshold):
    multi_results_dict = dict()

    for eps in eps_array:
        multi_results_dict[eps] = sweep(eps, wd_lower, wd_upper, params, threshold)

    return multi_results_dict


if __name__ == '__main__':
    #wc, wq, eps, g, chi, kappa, gamma, t_levels, c_levels
    params = Parameters(10.3641, 9.4914, 0.0001, 0.389, -0.097, 0.00146, 0.000833, 2, 10)
    eps = 0.0001
    threshold = 1.
    wd_lower = 10.4
    wd_upper = 10.55
    eps_array = np.linspace(0.0001, 0.0002, 2)
    multi_results = multi_sweep(eps_array, wd_lower, wd_upper, params, threshold)
    results = multi_results[0.0001]
    plt.scatter(results.wd_points, results.abs_transmissions)
    plt.show()
