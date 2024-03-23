#!/usr/bin/env python3

import numpy as np
from scipy.optimize import brentq, minimize

from pde import FieldCollection, PDEBase
import numba as nb

def find_roots(residual, x0, x1, N=1000):
    """Finds the roots of a residual function solving f(x) = 0 in the interval [x0, x1].

    Args:
        residual: the function f(x)
        x0: lower boundary
        x1: upper boundary
        N: number of points to sample to determine number of roots
    Returns:
        Container of roots [x_1, cdots, x_n].
    """

    # Check for sign changes
    trial_x = np.linspace(x0, x1, N)
    trial_residual = residual(trial_x)
    sign_changes = np.where(np.diff(np.sign(trial_residual)))[0]

    # Find root in each interval with a sign change
    roots = []
    for index in sign_changes:
        root = brentq(residual, trial_x[index], trial_x[index + 1])
        roots.append(root)

    return np.array(roots)

n = 2
ksaddle = (n - 1) / (n + 1)**2
kcusp = (n - 1)**2 / (n**2 + 6*n + 1)

class CellPolarisationPDE(PDEBase):
    def __init__(self, D=[1,10], k=0.08, bc="auto_periodic_neumann"):
        super().__init__()
        self.bc = bc
        self.Dp = D[0]
        self.Ds = D[1]
        self.k = k

    def chemical_flux(self, p, s):
        return (self.k + (1-self.k) * p**n / (1 + p**n)) * s - p

    def grad_chemical_flux(self, p, s):
        L = np.empty((2, 2))
        L[0,0] = - (self.k + p**n) / (1 + p**n)
        L[0,1] = ( p + p**(2*n + 1) + p**n * ((self.k - 1) * n * s + 2*p) ) / (p * (1 + p**n)**2)
        L[1] = -L[0]
        return L

    def reactive_eigenvalue(self, p, s):
        return - ((self.k + 1) * p + \
                  (self.k + 3) * p**(n+1) + 2*p**(2*n+1) + \
                  (self.k - 1) * n * p**n * s) / (p * (1 + p**n)**2)

    def density_one_form(self, p, s):
        return np.array((1., 1.))

    def density_tangent(self, p, s):
        g_p = p * (1 + p**n) * (self.k + p**n)
        g_s = p + (self.k - 1) * n * s * p**n + 2*p**(n+1) + p**(2*n+1)
        G = (self.k + 1) * p + (self.k - 1) * n * s * p**n + (self.k + 3) * p**(n+1) + 2*p**(2*n+1)
        return g_p/G, g_s/G

    def nullcline_s(self, p):
        return p * (1 + p**n) / (self.k + p**n)

    def nullcline_intersections(self, v, C):
        """Finds the points along the nullcline solving $v \cdot (s,p)^\top = C$ where $v$
        is a vector (representing e.g. the normal to a plane) and $C$ is some constant.
        """

        residual = lambda p: (v[0] * p + v[1] * self.nullcline_s(p)) - C
        return find_roots(residual, 0, 2.5)

    def nullcline_tangent(self, p):
        g_p = 1
        g_s = (self.k + (3*self.k - 1)*p**2 + p**4) / (self.k + p**2)**2
        G = g_p + g_s
        return g_p/G, g_s/G

    def nullcline_tangent_derivative(self, p):
        g_p = 1
        g_s = (self.k + (3*self.k - 1)*p**2 + p**4) / (self.k + p**2)**2
        G = g_p + g_s
        dGdp = 2*p * (3*self.k**2 + p**2 - self.k * (3 + p**2)) / (self.k + p**2)**3
        dgp = -dGdp * g_p / G**2
        dgs = dGdp * (1 - g_s / G) / G 
        return dgp, dgs

    def instability_parameter(self, p):
        g = self.nullcline_tangent(p)
        return self.Dp * g[0] + self.Ds * g[1]

    @property
    def most_unstable_point(self):
        if self.k < kcusp:
            raise RuntimeError('cannot define a most unstable point following cusp catastrophe (k < {:.4f}) as everywhere is unstable in reentrant region!'.format(self.k))
        p = minimize(self.instability_parameter, x0=1).x[0]
        return p, self.nullcline_s(p)

    @property
    def saddle_point(self, N=1000):
        dsdp = lambda p: self.nullcline_tangent(p)[1]
        d2sdp2 = lambda p: self.nullcline_tangent_derivative(p)[1]
        roots = find_roots(d2sdp2, 0.01, 2.5)
        assert len(roots) == 1
        p = roots[0]
        return p, self.nullcline_s(p)

    @property
    def steady_state_nullcline_points(self):
        """Finds the points along the nullcline reached in the steady-state.

        For k > ksaddle there will be only one, but for k < ksaddle there will be either
        one (if the steady-state is homogeneous) or three (if it phase separates).
        The additional two points are the binodal points.

        Returns:
            Sorted container of points.
        """

        p0, s0 = self.saddle_point
        v = np.array([self.Dp, self.Ds])
        C0 = v.dot(self.nullcline_tangent(p0))
        print(v, C0)

        return self.nullcline_intersections(v, C0)
        # residual = lambda p: self.Ds * nullcline_s + self.Dp
        # f_array = residual(m_array)

    def evolution_rate(self, state, t=0):
        p, s = state

        R = self.chemical_flux(p, s)
        p_t = R + self.Dp * p.laplace(bc=self.bc)
        s_t = -R + self.Ds * s.laplace(bc=self.bc)

        return FieldCollection([p_t, s_t])

    def _make_pde_rhs_numba(self, state):
        """numba-compiled implementation of the PDE"""
        # gradient_squared = state.grid.make_operator("gradient_squared", bc=self.bc)
        laplace = state.grid.make_operator("laplace", bc=self.bc)
        Dp, Ds = self.Dp, self.Ds
        k = self.k

        @nb.jit
        def pde_rhs(data, t):
            p, s = data[0], data[1]
            rate = np.empty_like(data)
            rate[0] = (k + (1-k) * p**n / (1 + p**n)) * s - p
            rate[1] = -rate[0]
            rate[0] += Dp * laplace(p)
            rate[1] += Ds * laplace(s)
            return rate

        return pde_rhs
