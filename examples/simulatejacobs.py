#!/usr/bin/env python3
"""Example script for running simulations using reactGPU.

   Run this as a program or Jupyter notebook to simulate a uniformly
   conserved pattern-forming model modified from:
      Jacobs, B., Molenaar, J., and Deinum, E. E. (2019).
      doi: 10.1371/journal.pone.0213188
"""

# %%
import sys
import numpy as np, matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation

from tqdm import tqdm
from IPython.display import display, clear_output

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from reactgpu import reactor


def plot_fields(fields, fig=None, axes=None, fontsize=8):
    """Show the evolving fields during program execution"""
    if fig is not None:
        for field, ax, im, cbar in zip(fields, *axes):
            im.set_data(field)
            im.set_norm(Normalize(field.min(), vmax=field.max()))

        return fig, axes

    fig, axes = plt.subplots(1, len(fields), sharey=True)

    ims, cbars = [], []
    for i, (ax, label) in enumerate(zip(axes, 'uvwp')):
        im = ax.imshow(fields[i])
        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=2e-2, location='top')
        cbar.set_label(f'${label}$', fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)

        ims += [im]
        cbars += [cbar]

        ax.set_xlabel('$x$', fontsize=fontsize)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].set_ylabel('$y$', fontsize=fontsize)
    return fig, (axes, ims, cbars)

def animate(trajectory):
    """Update figure during simulation to show fields evolving."""
    steps = np.array(list(trajectory.keys()))
    fig, (axes, ims, cbars) = plot_fields(trajectory[steps[-1]])

    def update(frame):
        for i, (im,data) in enumerate(zip(ims,trajectory[steps[frame]])):
            im.set_data(data)

    anim = FuncAnimation(fig, update, frames=len(trajectory))
    return anim


# %%
# Parameters tuned to give patterns (droplets).
Du, Dv, Dw = 1, 10, 1000
D = (Du, Dv, Dw)
k, c, d, psi = 0.07, 1, 1, 10
params = (k, c, d, psi)

# Initialise inside the pattern-forming region of phase space.

# nullcline = homogeneously stable states
nullcline_v = lambda u: (u*(d + c*(1 + psi)*u)*(1 + u**2))/((d + c*u)*(k + u**2))
nullcline_w = lambda u: c*u*psi / (d + c*u)
u0 = 0.35
v0 = nullcline_v(u0)
w0 = nullcline_w(u0)

# Discretisation resolution in space and time.
dx, dy, dt = 1, 1, 1e-4 # space/time increments
Nx, Ny = 256, 256       # lattice dimensions

# Initial condition: step off nullcline with small perturbation
eps = 1e-2
u = eps * np.random.random((Nx, Ny))
v = eps * np.random.random((Nx, Ny))
w = eps * np.random.random((Nx, Ny))
u += u0 - np.average(u)
v += v0 - np.average(v)
w += w0 - np.average(w)

# Initialise kernel on GPU.
kernel = reactor.JacobsModel((u, v, w), dt, dx, dy, *D, *params)

# %%
# Run simulation showing periodic updates.

tfinal = 1e4
tdump = tfinal / 100
plot = plot_fields

ndumps = int(np.round(tfinal / tdump))
error = np.abs(ndumps - tfinal / tdump)
assert np.isclose(error, 0)
nsteps_per_dump = int(np.round(tdump / dt))
error = np.abs(nsteps_per_dump - tdump / dt)
assert np.isclose(error, 0)

step_iterator = range(ndumps)
if ndumps > 1:
    step_iterator = tqdm(step_iterator)
    fig, axes = plot(kernel.state)

trajectory = {}
for i in step_iterator:
    for out in [sys.stdout, sys.stderr]: out.flush()
    trajectory[kernel.step] = kernel.state # store (u,v,w) fields
    kernel.run(nsteps_per_dump)

    if ndumps > 1:
        # Show latest field in output (may only work in Jupyter).
        # clear_output(wait=True)
        plot(kernel.state, fig, axes)
        display(fig)

trajectory[kernel.step] = kernel.state

if ndumps > 1: plt.close()
fig, axes = plot(kernel.state) # show final state

# Analysis/plotting code would go here.
# Retrieve final fields via `u, v, w = kernel.state`, or retrieve
# historical data from `trajectory` which is a list containing
# each periodic dump for time-series analysis.