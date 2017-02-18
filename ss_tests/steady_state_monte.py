import numpy as np
from qutip import *
from pylab import *
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from datetime import datetime
import os

# runfile('C:/Users/User/Documents/Python Scripts/untitled3.py',
#                 wdir='C:/Users/User/Documents/Python Scripts',
#                 args="5450 5500")

class system_parameters:
    def __init__(self, fc, fq, g, chi, eps, fd, kappa, gamma):
        self.fc = fc
        self.fq = fq
        self.eps = eps
        self.g = g
        self.chi = chi
        self.eps = eps
        self.fd = fd
        self.gamma = gamma
        self.kappa = kappa

class hilbert_dimensions:
    def __init__(self, c_levels, t_levels):
        self.c_levels = c_levels
        self.t_levels = t_levels

class simulation_options:
    def __init__(self, end_time, n_snaps, m_ops, n_traj = 50):
        self.end_time = end_time
        self.n_snaps = n_snaps
        self.m_ops = m_ops
        self.n_traj = n_traj

def hamiltonian(sys_params, hilbert_dims):
    a = tensor(destroy(hilbert_dims.c_levels), qeye(hilbert_dims.t_levels))
    sm = tensor(qeye(hilbert_dims.c_levels), destroy(hilbert_dims.t_levels))
    H = (sys_params.fc - sys_params.fd) * a.dag() * a + (sys_params.fq - sys_params.fd) * sm.dag() * sm \
        + sys_params.chi * sm.dag() * sm * (sm.dag() * sm - 1) + sys_params.g * (a.dag() * sm + a * sm.dag()) \
        + sys_params.eps * (a + a.dag())
    return H

def collapse_operators(sys_params, hilbert_dims):
    a = tensor(destroy(hilbert_dims.c_levels), qeye(hilbert_dims.t_levels))
    sm = tensor(qeye(hilbert_dims.c_levels), destroy(hilbert_dims.t_levels))
    c_ops = []
    c_ops.append(np.sqrt(sys_params.kappa) * a)
    c_ops.append(np.sqrt(sys_params.gamma) * sm)
    return c_ops

def solution(sys_params, hilbert_dims, initial_state, sim_options):
    H = hamiltonian(sys_params, hilbert_dims)
    c_ops = collapse_operators(sys_params, hilbert_dims)
    snapshot_times = linspace(0, sim_options.end_time, sim_options.n_snaps)
    output = mcsolve(H, initial_state, snapshot_times, c_ops, sim_options.m_ops, ntraj=sim_options.n_traj)
    return output

if __name__ == '__main__':
    sys_params = system_parameters(10.4267, 9.39128, 0.3096, -0.097, 0.004, 10.50662, 0.00146, 0.000833)
    c_levels = 20
    t_levels = 5
    hilbert_dims = hilbert_dimensions(c_levels, t_levels)
    m_ops = []
    snapshots = 100
    trajectories = 1000
    endtime = 20000
    sim_options = simulation_options(endtime, snapshots, m_ops, trajectories)
    initial_state = tensor(basis(hilbert_dims.c_levels, 0), basis(hilbert_dims.t_levels, 0))
    results = solution(sys_params, hilbert_dims, initial_state, sim_options)
    cavity_zero_vector = Qobj(np.zeros(c_levels))
    cavity_zero_matrix = cavity_zero_vector * cavity_zero_vector.dag()
    transmon_zero_vector = Qobj(np.zeros(t_levels))
    transmon_zero_matrix = transmon_zero_vector * transmon_zero_vector.dag()
    rho_ss = tensor(cavity_zero_matrix, transmon_zero_matrix)
    for i in range(trajectories):
        state = results.states[i, snapshots - 1]
        rho_state = state * state.dag()
        trace_rho_state = rho_state.tr()
        trace_rho_before = rho_ss.tr()
        rho_ss = rho_ss + rho_state
        trace_rho_after = rho_ss.tr()
        trace_rho_difference = trace_rho_after - trace_rho_before
        print i
    rho_ss = rho_ss / trajectories
    trace = rho_ss.tr()
    rho_c = rho_ss.ptrace(0)
    indices = np.arange(c_levels)
    qsave(rho_ss, 'rho_ss_monte')
    plt.bar(indices, rho_c.diag())
    plt.show()