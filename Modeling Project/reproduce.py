# Imports
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt  # root-finding algorithm
# @title Figure Settings
import logging
logging.getLogger('matplotlib.font_manager').disabled = True

import ipywidgets as widgets  # interactive display

#%config InlineBackend.figure_format = 'retina'
plt.style.use("https://raw.githubusercontent.com/NeuromatchAcademy/course-content/main/nma.mplstyle")


# @title Plotting Functions

def plot_FI_inverse(x, a, theta):
  f, ax = plt.subplots()
  ax.plot(x, F_inv(x, a=a, theta=theta))
  ax.set(xlabel="$x$", ylabel="$F^{-1}(x)$")


def plot_FI_EI(x, FI_exc, FI_inh):
  plt.figure()
  plt.plot(x, FI_exc, 'b', label='E population')
  plt.plot(x, FI_inh, 'r', label='I population')
  plt.legend(loc='lower right')
  plt.xlabel('x (a.u.)')
  plt.ylabel('F(x)')
  plt.show()


def my_test_plot(t, rE1, rI1, rE2, rI2):

  plt.figure()
  ax1 = plt.subplot(211)
  ax1.plot(pars['range_t'], rE1, 'b', label='E population')
  ax1.plot(pars['range_t'], rI1, 'r', label='I population')
  ax1.set_ylabel('Activity')
  ax1.legend(loc='best')

  ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)
  ax2.plot(pars['range_t'], rE2, 'b', label='E population')
  ax2.plot(pars['range_t'], rI2, 'r', label='I population')
  ax2.set_xlabel('t (ms)')
  ax2.set_ylabel('Activity')
  ax2.legend(loc='best')

  plt.tight_layout()
  plt.show()


def plot_nullclines(Exc_null_rE, Exc_null_rI, Inh_null_rE, Inh_null_rI):

  plt.figure()
  plt.plot(Exc_null_rE, Exc_null_rI, 'b', label='E nullcline')
  plt.plot(Inh_null_rE, Inh_null_rI, 'r', label='I nullcline')
  plt.xlabel(r'$r_E$')
  plt.ylabel(r'$r_I$')
  plt.legend(loc='best')
  plt.show()


def my_plot_nullcline(pars):
  Exc_null_rE = np.linspace(-0.01, 0.96, 100)
  Exc_null_rI = get_E_nullcline(Exc_null_rE, **pars)
  Inh_null_rI = np.linspace(-.01, 0.8, 100)
  Inh_null_rE = get_I_nullcline(Inh_null_rI, **pars)

  plt.plot(Exc_null_rE, Exc_null_rI, 'b', label='E nullcline')
  plt.plot(Inh_null_rE, Inh_null_rI, 'r', label='I nullcline')
  plt.xlabel(r'$r_E$')
  plt.ylabel(r'$r_I$')
  plt.legend(loc='best')


def my_plot_vector(pars, my_n_skip=2, myscale=5):
  EI_grid = np.linspace(0., 1., 20)
  rE, rI = np.meshgrid(EI_grid, EI_grid)
  drEdt, drIdt = EIderivs(rE, rI, **pars)

  n_skip = my_n_skip

  plt.quiver(rE[::n_skip, ::n_skip], rI[::n_skip, ::n_skip],
             drEdt[::n_skip, ::n_skip], drIdt[::n_skip, ::n_skip],
             angles='xy', scale_units='xy', scale=myscale, facecolor='c')

  plt.xlabel(r'$r_E$')
  plt.ylabel(r'$r_I$')


def my_plot_trajectory(pars, mycolor, x_init, mylabel):
  pars = pars.copy()
  pars['rE_init'], pars['rI_init'] = x_init[0], x_init[1]
  rE_tj, rI_tj = simulate_wc(**pars)

  plt.plot(rE_tj, rI_tj, color=mycolor, label=mylabel)
  plt.plot(x_init[0], x_init[1], 'o', color=mycolor, ms=8)
  plt.xlabel(r'$r_E$')
  plt.ylabel(r'$r_I$')


def my_plot_trajectories(pars, dx, n, mylabel):
  """
  Solve for I along the E_grid from dE/dt = 0.

  Expects:
  pars    : Parameter dictionary
  dx      : increment of initial values
  n       : n*n trjectories
  mylabel : label for legend

  Returns:
    figure of trajectory
  """
  pars = pars.copy()
  for ie in range(n):
    for ii in range(n):
      pars['rE_init'], pars['rI_init'] = dx * ie, dx * ii
      rE_tj, rI_tj = simulate_wc(**pars)
      if (ie == n-1) & (ii == n-1):
          plt.plot(rE_tj, rI_tj, 'gray', alpha=0.8, label=mylabel)
      else:
          plt.plot(rE_tj, rI_tj, 'gray', alpha=0.8)

  plt.xlabel(r'$r_E$')
  plt.ylabel(r'$r_I$')


def plot_complete_analysis(pars):
  plt.figure(figsize=(7.7, 6.))

  # plot example trajectories
  my_plot_trajectories(pars, 0.2, 6,
                       'Sample trajectories \nfor different init. conditions')
  my_plot_trajectory(pars, 'orange', [0.6, 0.8],
                     'Sample trajectory for \nlow activity')
  my_plot_trajectory(pars, 'm', [0.6, 0.6],
                     'Sample trajectory for \nhigh activity')

  # plot nullclines
  my_plot_nullcline(pars)

  # plot vector field
  EI_grid = np.linspace(0., 1., 20)
  rE, rI = np.meshgrid(EI_grid, EI_grid)
  drEdt, drIdt = EIderivs(rE, rI, **pars)
  n_skip = 2
  plt.quiver(rE[::n_skip, ::n_skip], rI[::n_skip, ::n_skip],
             drEdt[::n_skip, ::n_skip], drIdt[::n_skip, ::n_skip],
             angles='xy', scale_units='xy', scale=5., facecolor='c')

  plt.legend(loc=[1.02, 0.57], handlelength=1)
  plt.show()


def plot_fp(x_fp, position=(0.02, 0.1), rotation=0):
  plt.plot(x_fp[0], x_fp[1], 'ko', ms=8)
  plt.text(x_fp[0] + position[0], x_fp[1] + position[1],
           f'Fixed Point1=\n({x_fp[0]:.3f}, {x_fp[1]:.3f})',
           horizontalalignment='center', verticalalignment='bottom',
           rotation=rotation)


# @title Helper Functions

def default_pars(**kwargs):
  pars = {}

  # Excitatory parameters
  pars['tau_E'] = 1.     # Timescale of the E population [ms]
  pars['a_E'] = 1.2      # Gain of the E population
  pars['theta_E'] = 2.8  # Threshold of the E population

  # Inhibitory parameters
  pars['tau_I'] = 2.0    # Timescale of the I population [ms]
  pars['a_I'] = 1.0      # Gain of the I population
  pars['theta_I'] = 4.0  # Threshold of the I population

  # Connection strength
  pars['wEE'] = 9.   # E to E
  pars['wEI'] = 4.   # I to E
  pars['wIE'] = 13.  # E to I
  pars['wII'] = 11.  # I to I

  # External input
  pars['I_ext_E'] = 0.
  pars['I_ext_I'] = 0.

  # simulation parameters
  pars['T'] = 50.        # Total duration of simulation [ms]
  pars['dt'] = .1        # Simulation time step [ms]
  pars['rE_init'] = 0.2  # Initial value of E
  pars['rI_init'] = 0.2  # Initial value of I

  # External parameters if any
  for k in kwargs:
      pars[k] = kwargs[k]

  # Vector of discretized time points [ms]
  pars['range_t'] = np.arange(0, pars['T'], pars['dt'])

  return pars


def F(x, a, theta):
  """
  Population activation function, F-I curve

  Args:
    x     : the population input
    a     : the gain of the function
    theta : the threshold of the function

  Returns:
    f     : the population activation response f(x) for input x
  """

  # add the expression of f = F(x)
  f = (1 + np.exp(-a * (x - theta)))**-1 - (1 + np.exp(a * theta))**-1

  return f


def dF(x, a, theta):
  """
  Derivative of the population activation function.

  Args:
    x     : the population input
    a     : the gain of the function
    theta : the threshold of the function

  Returns:
    dFdx  :  Derivative of the population activation function.
  """

  dFdx = a * np.exp(-a * (x - theta)) * (1 + np.exp(-a * (x - theta)))**-2

  return dFdx

def simulate_wc(tau_E, a_E, theta_E, tau_I, a_I, theta_I,
                wEE, wEI, wIE, wII, I_ext_E, I_ext_I,
                rE_init, rI_init, dt, range_t, **other_pars):
  """
  Simulate the Wilson-Cowan equations

  Args:
    Parameters of the Wilson-Cowan model

  Returns:
    rE, rI (arrays) : Activity of excitatory and inhibitory populations
  """
  # Initialize activity arrays
  Lt = range_t.size
  rE = np.append(rE_init, np.zeros(Lt - 1))
  rI = np.append(rI_init, np.zeros(Lt - 1))
  I_ext_E = I_ext_E * np.ones(Lt)
  I_ext_I = I_ext_I * np.ones(Lt)

  # Simulate the Wilson-Cowan equations
  for k in range(Lt - 1):

    # Calculate the derivative of the E population
    drE = dt / tau_E * (-rE[k] + F(wEE * rE[k] - wEI * rI[k] + I_ext_E[k],
                                   a_E, theta_E))

    # Calculate the derivative of the I population
    drI = dt / tau_I * (-rI[k] + F(wIE * rE[k] - wII * rI[k] + I_ext_I[k],
                                   a_I, theta_I))

    # Update using Euler's method
    rE[k + 1] = rE[k] + drE
    rI[k + 1] = rI[k] + drI

  return rE, rI


pars = default_pars()

# Simulate first trajectory
rE1, rI1 = simulate_wc(**default_pars(rE_init=.32, rI_init=.15))

# Simulate second trajectory
rE2, rI2 = simulate_wc(**default_pars(rE_init=.33, rI_init=.15))

# Visualize
with plt.xkcd():
  my_test_plot(pars['range_t'], rE1, rI1, rE2, rI2)     


# ------------------------------------------------------------------
# Bifurcation sweep over wEE  (Li et al. Fig 9 style)
# ------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

pars0 = default_pars(T=400, dt=0.1)
pars0.update({
    'wEI': 12.,
    'wIE': 10.,
    'wII': 0.0,
    'I_ext_E': 1.6,   # ← NEW: tonic drive
    'I_ext_I': 0.0,
})


# 2.  Sweep the control parameter
wEE_vals = np.linspace(6, 16, 50)      # Li’s range (~8–15), broaden a bit
amp_E   = []                           # peak-to-peak amplitude of E
is_lc   = []                           # 1 = limit cycle, 0 = fixed point

for w in wEE_vals:
    pars = pars0.copy()
    pars['wEE'] = w

    # Simulate
    rE, rI = simulate_wc(**pars)

    # Examine the last 40 % of samples (post-transient)
    n_tail = int(0.4 * len(rE))
    rE_tail = rE[-n_tail:]

    # Peak-to-peak amplitude
    ptp = rE_tail.max() - rE_tail.min()
    amp_E.append(ptp)

    # Classify: oscillatory if amplitude above small threshold
    is_lc.append(1 if ptp > 1e-2 else 0)

# 3.  Plot the bifurcation diagram
plt.figure(figsize=(6,4))
plt.plot(wEE_vals, amp_E, 'k.-')
plt.xlabel(r'$w_{EE}$ (E$\!\to$E strength)')
plt.ylabel('E-population\noscillation amplitude')
plt.title('Hopf-like bifurcation (reproduction of Li et al. Fig 9)')
plt.axhline(0, color='gray', lw=0.5)
plt.show()



# =============================================================
# Li et al. 2022 – two-parameter sweep with on-grid diagnostics
# =============================================================

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import welch

# ---------- Li baseline parameters (from Table 1) ------------
def li_pars():
    return default_pars(
        tau_E=20., tau_I=10.,
        a_E=1., a_I=1.,
        theta_E=5., theta_I=20.,
        wEI=26., wIE=20.,
        I_ext_E=2., I_ext_I=7.,
        T=1000, dt=0.1
    )

# ---------- parameter grid -----------------------------------
grid_n = 50                     # 50×50 = 2 500 sims (~45 s on laptop)
wEE_vec = np.linspace(5, 40, grid_n)
wII_vec = np.linspace(-5, 5, grid_n)

amp_mat  = np.zeros((grid_n, grid_n))
freq_mat = np.zeros_like(amp_mat)

# pick a few indices to print diagnostics
probe_idx = [(0,  0),              # low WEE, low WII
             (grid_n//2, grid_n//2),  # mid grid
             (grid_n-1, grid_n//2)]   # high WEE, mid WII

print("i  j   WEE   WII   amp   freq(Hz)")
print("-----------------------------------")

for i, wII in enumerate(wII_vec):
    for j, wEE in enumerate(wEE_vec):
        p = li_pars()
        p['wEE'], p['wII'] = wEE, wII
        rE, _ = simulate_wc(**p)

        tail = rE[len(rE)//2:]       # ignore first half
        sig  = tail - tail.mean()
        amp  = sig.max() - sig.min()
        amp_mat[i, j] = amp

        if amp > 0.05:               # limit cycle detected
            fs  = 1000 / p['dt']     # ms → Hz
            f, P = welch(sig, fs, nperseg=2048)
            peak = np.argmax(P[1:])+1
            freq_mat[i, j] = f[peak]
        else:
            freq_mat[i, j] = 0.

        # print diagnostics for selected points
        if (i, j) in probe_idx:
            print(f"{i:2d} {j:2d}  {wEE:5.1f} {wII:5.1f} "
                  f"{amp:6.3f}  {freq_mat[i,j]:6.1f}")

# ---------- panel A: amplitude + Hopf curve ------------------
plt.figure(figsize=(6,5))
plt.imshow(amp_mat, origin='lower',
           extent=[wEE_vec[0], wEE_vec[-1], wII_vec[0], wII_vec[-1]],
           aspect='auto', cmap='Greys')
plt.contour(wEE_vec, wII_vec, amp_mat,
            levels=[0.05], colors='blue', linewidths=2)
plt.xlabel(r'$W_{EE}$'); plt.ylabel(r'$W_{II}$')
plt.title('Hopf boundary (blue) over amplitude background')
plt.show()

# ---------- panel B: frequency heat-map ----------------------
plt.figure(figsize=(6,5))
plt.imshow(freq_mat, origin='lower',
           extent=[wEE_vec[0], wEE_vec[-1], wII_vec[0], wII_vec[-1]],
           aspect='auto', cmap='jet', vmin=0, vmax=50)
plt.colorbar(label='Dominant freq (Hz)')
plt.xlabel(r'$W_{EE}$'); plt.ylabel(r'$W_{II}$')
plt.title('Oscillation frequency map')
plt.show()

