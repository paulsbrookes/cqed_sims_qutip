import numpy as np
import yaml
from qutip import *
from pylab import *
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import yaml
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from qutip.ui.progressbar import TextProgressBar

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

    def copy(self):
        params = Parameters(self.wc, self.wq, self.eps, self.g, self.chi, self.kappa, self.gamma, self.t_levels, self.c_levels)
        return params

def hamiltonian(params, wd):
    a = tensor(destroy(params.c_levels), qeye(params.t_levels))
    sm = tensor(qeye(params.c_levels), destroy(params.t_levels))
    H =  (params.wc - wd) * a.dag() * a + (params.wq - wd) * sm.dag() * sm \
        + params.chi * sm.dag() * sm * (sm.dag() * sm - 1) + params.g * (a.dag() * sm + a * sm.dag()) \
        + params.eps * (a + a.dag())
    return H

if __name__ == '__main__':
    t_levels = 5
    c_levels = 20
    rho_ss = qload('rho_ss_monte')
    #rho_ss = tensor(coherent_dm(c_levels, 0), coherent_dm(t_levels, 0))
    rho_c_ss = rho_ss.ptrace(0)
    occupations = rho_c_ss.diag()
    indices = np.arange(occupations.size)
    params = Parameters(10.4267, 9.39128, 0.004, 0.3096, -0.097, 0.00146, 0.000833, t_levels, c_levels)
    wd = 10.50662
    H = hamiltonian(params, wd)
    steps = 5000
    snapshot_times = np.linspace(0, 100000, steps)
    a = tensor(destroy(params.c_levels), qeye(params.t_levels))
    sm = tensor(qeye(params.c_levels), destroy(params.t_levels))
    c_ops = []
    c_ops.append(np.sqrt(params.kappa) * a)
    c_ops.append(np.sqrt(params.gamma) * sm)
    m_ops = [tensor(fock_dm(c_levels, x), qeye(t_levels)) for x in range(c_levels)]
    results = mesolve(H, rho_ss, snapshot_times, c_ops, m_ops)
    occupations = [values[steps - 1] for values in results.expect]
    plt.bar(indices, occupations)
    plt.show()