"""
====================================================================
 Antibody Binding Kinetics on a Fiber-Optic Biosensor Surface
 Complete FEM solution + ALL plots for Google Colab
====================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import interact, IntSlider, Dropdown
from IPython.display import display
import matplotlib

# Set backend for Colab
matplotlib.use('module://matplotlib_inline.backend_inline')

# ============================================================
# PART A - PHYSICAL MODEL, FEM ASSEMBLY, SOLVER
# ============================================================

# Physical parameters
D = 1.0e-10        # diffusivity, m^2/s
kf = 1.0e5         # forward binding rate, 1/(M*s)
kr = 1.0e1         # reverse binding rate, 1/s
cb_sat = 1.66e-9   # saturation concentration, mol/m^2
c_bulk = 4.48e-5   # bulk concentration, mol/m^3
Hdom = 5.0e-5      # domain length, m
c0_IC = 0.0        # initial concentration, mol/m^3
cb0_IC = 0.0       # initial bound concentration, mol/m^2

def build_mesh(n_nodes, H):
    """Build 1D mesh with linear elements"""
    z = np.linspace(0.0, H, n_nodes)
    elements = np.array([[i, i + 1] for i in range(n_nodes - 1)])
    return z, elements

def shape_functions(xi):
    """Linear shape functions on [-1,1]"""
    N = np.array([0.5*(1.0-xi), 0.5*(1.0+xi)])
    dN_dxi = np.array([-0.5, 0.5])
    return N, dN_dxi

# Gauss quadrature points and weights
GAUSS_PTS = np.array([-1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)])
GAUSS_WTS = np.array([1.0, 1.0])

def element_matrices(z_nodes):
    """Compute element mass and stiffness matrices"""
    z1, z2 = z_nodes
    Je = (z2 - z1) / 2.0
    Me = np.zeros((2, 2))
    Ke = np.zeros((2, 2))
    for xi, w in zip(GAUSS_PTS, GAUSS_WTS):
        N, dN_dxi = shape_functions(xi)
        dN_dz = dN_dxi / Je
        Me += w * np.outer(N, N) * Je
        Ke += w * np.outer(dN_dz, dN_dz) * Je
    return Me, Ke

def assemble_global_matrices(z, elements):
    """Assemble global mass and stiffness matrices"""
    n_nodes = len(z)
    M = np.zeros((n_nodes, n_nodes))
    K = np.zeros((n_nodes, n_nodes))
    for elem in elements:
        i, j = elem
        Me, Ke = element_matrices(z[[i, j]])
        idx = [i, j]
        for a in range(2):
            for b in range(2):
                M[idx[a], idx[b]] += Me[a, b]
                K[idx[a], idx[b]] += Ke[a, b]
    return M, K

def binding_flux(c0, cb):
    """Nonlinear binding flux at surface"""
    return kf * c0 * (cb_sat - cb) - kr * cb

def dflux_dc0(c0, cb):
    """Derivative of flux w.r.t. c0"""
    return kf * (cb_sat - cb)

def dflux_dcb(c0, cb):
    """Derivative of flux w.r.t. cb"""
    return -kf * c0 - kr

def residual(U, U_old, dt, M, K, n_nodes):
    """Backward Euler residual"""
    c = U[:n_nodes]
    cb = U[-1]
    flux = binding_flux(c[0], cb)

    R = np.zeros(n_nodes + 1)
    Mc_t = (M @ (c - U_old[:n_nodes])) / dt
    DKc = D * (K @ c)

    R[:n_nodes-1] = (Mc_t + DKc)[:n_nodes-1]
    R[0] += flux
    R[n_nodes-1] = c[n_nodes-1] - c_bulk
    R[-1] = (cb - U_old[-1]) / dt - flux
    return R

def jacobian(U, dt, M, K, n_nodes):
    """Analytic Jacobian"""
    c = U[:n_nodes]
    cb = U[-1]
    ndof = n_nodes + 1
    J = np.zeros((ndof, ndof))

    J[:n_nodes-1, :n_nodes] = (M[:n_nodes-1, :] / dt) + D * K[:n_nodes-1, :]
    J[0, 0] += dflux_dc0(c[0], cb)
    J[0, -1] += dflux_dcb(c[0], cb)
    J[n_nodes-1, :] = 0.0
    J[n_nodes-1, n_nodes-1] = 1.0
    J[-1, 0] = -dflux_dc0(c[0], cb)
    J[-1, -1] = 1.0/dt - dflux_dcb(c[0], cb)
    return J

def newton_step(U_old, dt, M, K, n_nodes, tol=1e-14, max_iter=30):
    """Newton-Raphson solver for one time step"""
    U = U_old.copy()
    for it in range(max_iter):
        R = residual(U, U_old, dt, M, K, n_nodes)
        if np.linalg.norm(R) < tol:
            return U, it
        J = jacobian(U, dt, M, K, n_nodes)
        dU = np.linalg.solve(J, -R)
        U = U + dU
        if np.linalg.norm(dU) < tol:
            return U, it
    raise RuntimeError("Newton failed to converge")

def solve(n_nodes=21, dt=0.5, t_final=100.0):
    """Solve the coupled PDE/ODE system"""
    z, elements = build_mesh(n_nodes, Hdom)
    M, K = assemble_global_matrices(z, elements)

    n_steps = int(round(t_final / dt))
    U = np.zeros(n_nodes + 1)
    U[:n_nodes] = c0_IC
    U[-1] = cb0_IC

    t_hist = [0.0]
    U_hist = [U.copy()]
    newton_iters = []

    for step in range(n_steps):
        U_old = U.copy()
        U, iters = newton_step(U_old, dt, M, K, n_nodes)
        newton_iters.append(iters)
        t_hist.append((step + 1) * dt)
        U_hist.append(U.copy())

    t_hist = np.array(t_hist)
    U_hist = np.array(U_hist)
    c_hist = U_hist[:, :n_nodes]
    cb_hist = U_hist[:, -1]
    theta_hist = cb_hist / cb_sat
    rate_hist = kf * c_hist[:, 0] * (cb_sat - cb_hist) - kr * cb_hist

    return {
        "z": z, "t": t_hist, "c": c_hist, "cb": cb_hist,
        "theta": theta_hist, "rate": rate_hist,
        "newton_iters": np.array(newton_iters),
    }

# ============================================================
# RUN SOLVER
# ============================================================
print("="*60)
print("SOLVING BIOSENSOR MODEL WITH FEM + BACKWARD EULER + NEWTON")
print("="*60)

sol = solve(n_nodes=21, dt=0.5, t_final=100.0)
t, z = sol["t"], sol["z"]
c, cb, theta, rate = sol["c"], sol["cb"], sol["theta"], sol["rate"]

# Convert units for plotting
z_um = z * 1e6
c_nM = c * 1e6

# Print results
print("\n{:>6} {:>14} {:>14} {:>12} {:>14}".format(
    "t", "c(0,t)", "cb(t)", "theta", "rate"))
for tt in [0, 2, 4, 6, 8, 10, 20, 40, 60, 80, 100]:
    idx = np.argmin(np.abs(t - tt))
    print(f"{t[idx]:6.0f} {c[idx,0]:14.3e} {cb[idx]:14.3e} "
          f"{theta[idx]:12.3e} {rate[idx]:14.3e}")

print(f"\nAverage Newton iterations/step: {sol['newton_iters'].mean():.2f}")
print(f"Final: c(0,100)={c[-1,0]:.4e} mol/m³")
print(f"Final: cb(100)={cb[-1]:.4e} mol/m²")
print(f"Final: theta={theta[-1]:.4f}")

# ============================================================
# PART B - STATIC PLOTS
# ============================================================

print("\n" + "="*60)
print("GENERATING STATIC PLOTS")
print("="*60)

# --- PLOT 1: Concentration profiles at selected times ---
plt.figure(figsize=(8, 5))
for pt in [0, 2, 10, 20, 40, 100]:
    idx = np.argmin(np.abs(t - pt))
    plt.plot(z_um, c_nM[idx, :], label=f"t={t[idx]:.0f} s", lw=2)
plt.xlabel("z (µm)", fontsize=12)
plt.ylabel("c(z,t) (nM)", fontsize=12)
plt.title("Concentration Profiles at Selected Times", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- PLOT 2: 3D surface plot ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
Z, T = np.meshgrid(z_um, t)
surf = ax.plot_surface(Z, T, c_nM, cmap='viridis', alpha=0.8)
ax.set_xlabel("z (µm)", fontsize=12)
ax.set_ylabel("t (s)", fontsize=12)
ax.set_zlabel("c(z,t) (nM)", fontsize=12)
ax.set_title("FEM Solution: c(z,t) Surface", fontsize=14)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="c (nM)")
plt.tight_layout()
plt.show()

# --- PLOT 3: 2x2 panel ---
fig, axs = plt.subplots(2, 2, figsize=(12, 9))

# c(0,t)
axs[0, 0].plot(t, c_nM[:, 0], 'b-', lw=2)
axs[0, 0].set_xlabel("t (s)", fontsize=11)
axs[0, 0].set_ylabel("c(0,t) (nM)", fontsize=11)
axs[0, 0].set_title("Surface Concentration", fontsize=12)
axs[0, 0].grid(True, alpha=0.3)

# cb(t)
axs[0, 1].plot(t, cb, 'r-', lw=2)
axs[0, 1].set_xlabel("t (s)", fontsize=11)
axs[0, 1].set_ylabel("cb(t) (mol/m²)", fontsize=11)
axs[0, 1].set_title("Bound Concentration", fontsize=12)
axs[0, 1].grid(True, alpha=0.3)

# theta(t)
axs[1, 0].plot(t, theta * 100, 'g-', lw=2)
axs[1, 0].set_xlabel("t (s)", fontsize=11)
axs[1, 0].set_ylabel("θ (%)", fontsize=11)
axs[1, 0].set_title("Surface Coverage", fontsize=12)
axs[1, 0].grid(True, alpha=0.3)

# Rate
axs[1, 1].plot(t, rate, 'm-', lw=2)
axs[1, 1].set_xlabel("t (s)", fontsize=11)
axs[1, 1].set_ylabel("Rate (mol/m²·s)", fontsize=11)
axs[1, 1].set_title("Net Binding Rate", fontsize=12)
axs[1, 1].grid(True, alpha=0.3)

plt.suptitle("Biosensor Response: Key Quantities vs Time", fontsize=14)
plt.tight_layout()
plt.show()

# ============================================================
# PART C - INTERACTIVE PLOT
# ============================================================

print("\n" + "="*60)
print("INTERACTIVE PLOT (Use dropdown to switch views)")
print("="*60)

# Create interactive figure
fig2, ax2 = plt.subplots(figsize=(11, 7))
fig2.suptitle("Interactive Biosensor Visualization", fontsize=14)

# Initialize all possible lines
line_conc, = ax2.plot(z_um, c_nM[0], 'b-', lw=2.5, label='c(z,t)')
line_c0, = ax2.plot([], [], 'r-', lw=2.5, label='c(0,t)')
line_cb, = ax2.plot([], [], 'g-', lw=2.5, label='cb(t)')
line_theta, = ax2.plot([], [], 'm-', lw=2.5, label='θ(t)×100')
line_rate, = ax2.plot([], [], 'orange', lw=2.5, label='Rate(t)')

# Marker for current time
marker, = ax2.plot([], [], 'ko', markersize=10, label='Current time')

# Time text
time_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes,
                     fontsize=12, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# Legend
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)

# Initial setup - show concentration profile
line_conc.set_visible(True)
line_c0.set_visible(False)
line_cb.set_visible(False)
line_theta.set_visible(False)
line_rate.set_visible(False)

ax2.set_xlabel("z (µm)", fontsize=12)
ax2.set_ylabel("c(z,t) (nM)", fontsize=12)
ax2.set_title("Concentration Profile", fontsize=12)
ax2.set_xlim(z_um.min(), z_um.max())
ax2.set_ylim(0, c_nM.max() * 1.1)

def update_interactive(plot_choice, time_idx):
    """Update interactive plot"""
    idx = int(time_idx)
    idx = max(0, min(idx, len(t) - 1))

    # Hide all lines
    line_conc.set_visible(False)
    line_c0.set_visible(False)
    line_cb.set_visible(False)
    line_theta.set_visible(False)
    line_rate.set_visible(False)

    # Show selected plot
    if plot_choice == 'Concentration Profile (c vs z)':
        line_conc.set_visible(True)
        line_conc.set_ydata(c_nM[idx])
        ax2.set_xlabel("z (µm)", fontsize=12)
        ax2.set_ylabel("c(z,t) (nM)", fontsize=12)
        ax2.set_title(f"Concentration Profile at t = {t[idx]:.1f} s", fontsize=12)
        ax2.set_xlim(z_um.min(), z_um.max())
        ax2.set_ylim(0, c_nM.max() * 1.1)
        marker.set_data([z_um[0]], [c_nM[idx, 0]])

    elif plot_choice == 'Surface Concentration c(0,t)':
        line_c0.set_visible(True)
        line_c0.set_data(t, c_nM[:, 0])
        ax2.set_xlabel("Time (s)", fontsize=12)
        ax2.set_ylabel("c(0,t) (nM)", fontsize=12)
        ax2.set_title("Surface Concentration vs Time", fontsize=12)
        ax2.set_xlim(0, t.max())
        ax2.set_ylim(0, c_nM[:, 0].max() * 1.1)
        marker.set_data([t[idx]], [c_nM[idx, 0]])

    elif plot_choice == 'Bound Concentration cb(t)':
        line_cb.set_visible(True)
        line_cb.set_data(t, cb)
        ax2.set_xlabel("Time (s)", fontsize=12)
        ax2.set_ylabel("cb(t) (mol/m²)", fontsize=12)
        ax2.set_title("Bound Concentration vs Time", fontsize=12)
        ax2.set_xlim(0, t.max())
        ax2.set_ylim(0, cb.max() * 1.1)
        marker.set_data([t[idx]], [cb[idx]])

    elif plot_choice == 'Surface Coverage θ(t)':
        line_theta.set_visible(True)
        line_theta.set_data(t, theta * 100)
        ax2.set_xlabel("Time (s)", fontsize=12)
        ax2.set_ylabel("θ (%)", fontsize=12)
        ax2.set_title("Surface Coverage vs Time", fontsize=12)
        ax2.set_xlim(0, t.max())
        ax2.set_ylim(0, theta.max() * 110)
        marker.set_data([t[idx]], [theta[idx] * 100])

    elif plot_choice == 'Binding Rate':
        line_rate.set_visible(True)
        line_rate.set_data(t, rate)
        ax2.set_xlabel("Time (s)", fontsize=12)
        ax2.set_ylabel("Rate (mol/m²·s)", fontsize=12)
        ax2.set_title("Net Binding Rate vs Time", fontsize=12)
        ax2.set_xlim(0, t.max())
        y_max = max(abs(rate.min()), abs(rate.max()))
        ax2.set_ylim(-y_max * 1.1, y_max * 1.1)
        marker.set_data([t[idx]], [rate[idx]])

    time_text.set_text(f"t = {t[idx]:.1f} s")
    ax2.legend(loc='best', fontsize=10)
    fig2.canvas.draw()

# Create widgets
dropdown = Dropdown(
    options=['Concentration Profile (c vs z)',
             'Surface Concentration c(0,t)',
             'Bound Concentration cb(t)',
             'Surface Coverage θ(t)',
             'Binding Rate'],
    value='Concentration Profile (c vs z)',
    description='Plot:',
    style={'description_width': 'initial'}
)

slider = IntSlider(
    min=0, max=len(t)-1, step=1, value=0,
    description='Time step:',
    continuous_update=False,
    style={'description_width': 'initial'},
    layout={'width': '500px'}
)

# Connect and display
interact(update_interactive, plot_choice=dropdown, time_idx=slider)
plt.show()

print("\n" + "="*60)
print("✅ COMPLETE! All plots displayed successfully.")
print("="*60)
print("\nSummary of plots shown:")
print("1. Concentration profiles at selected times")
print("2. 3D surface plot of c(z,t)")
print("3. 2x2 panel with c(0,t), cb(t), θ(t), and rate(t)")
print("4. Interactive plot with dropdown menu and time slider")