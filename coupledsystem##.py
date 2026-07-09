import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Parameters (Tables 2.1 and 2.2)
# ==========================================================

D = 1.0e-10          # m^2/s
kf = 1.0e5           # m^3/(mol*s)
kr = 1.0e-2          # 1/s

cbsat = 2.66e-8      # mol/m^2
cbulk = 4.48e-5      # mol/m^3

h = 5.0e-5           # m

# ==========================================================
# Grid
# ==========================================================

n = 21

z = np.linspace(0.0, h, n)

dz = z[1] - z[0]

print(f"dz = {dz:.2e} m = {dz*1e6:.2f} μm")

# ==========================================================
# Time step (FTCS stability)
# alpha <= 0.5
# ==========================================================

dt = 0.8 * dz**2 / (2.0 * D)

alpha = D * dt / dz**2

print(f"dt = {dt:.5f} s")
print(f"alpha = {alpha:.4f}")

t_final = 500.0

nsteps = int(t_final / dt)

print(f"Number of time steps = {nsteps}")

# ==========================================================
# Initial conditions
# ==========================================================

c = np.zeros(n)

# Dirichlet BC at z=h
c[-1] = cbulk

cb = 0.0

# ==========================================================
# Storage
# ==========================================================

save_every = max(1, nsteps // 8)

profiles = [c.copy()]
profile_times = [0.0]

time_history = [0.0]
cb_history = [cb]

# ==========================================================
# Time loop
# ==========================================================

for step in range(1, nsteps + 1):

    c_old = c.copy()
    cb_old = cb

    # ------------------------------------------------------
    # Interior FTCS update
    # ------------------------------------------------------

    for i in range(1, n - 1):
        c[i] = (
            c_old[i]
            + alpha
            * (c_old[i + 1]
               - 2.0 * c_old[i]
               + c_old[i - 1])
        )

    # ------------------------------------------------------
    # Right boundary (Dirichlet)
    # ------------------------------------------------------

    c[-1] = cbulk

    # ------------------------------------------------------
    # Left boundary (Robin BC)
    #
    # D(dc/dz) =
    # kf*c0*(cbsat-cb) - kr*cb
    #
    # dc/dz ≈ (c1-c0)/dz
    # ------------------------------------------------------

    denominator = (
        1.0
        + (dz / D) * kf * (cbsat - cb_old)
    )

    numerator = (
        c[1]
        + (dz / D) * kr * cb_old
    )

    c[0] = numerator / denominator

    # ------------------------------------------------------
    # ODE update
    # dcb/dt =
    # kf*c0*(cbsat-cb) - kr*cb
    # ------------------------------------------------------

    dcbdt = (
        kf * c[0] * (cbsat - cb_old)
        - kr * cb_old
    )

    cb = cb_old + dt * dcbdt

    # enforce physical bounds
    cb = max(0.0, min(cb, cbsat))

    # ------------------------------------------------------
    # Save results
    # ------------------------------------------------------

    t = step * dt

    time_history.append(t)
    cb_history.append(cb)

    if step % save_every == 0:
        profiles.append(c.copy())
        profile_times.append(t)

# ==========================================================
# Results
# ==========================================================

theta = cb / cbsat

print("\n" + "=" * 50)
print("FINAL RESULTS")
print("=" * 50)

print(f"Final time            = {t:.2f} s")
print(f"Surface concentration = {c[0]:.4e} mol/m^3")
print(f"Bulk concentration    = {cbulk:.4e} mol/m^3")
print(f"c(0)/cbulk           = {c[0]/cbulk:.4f}")

print()

print(f"Bound concentration   = {cb:.4e} mol/m^2")
print(f"Fractional coverage   = {theta:.4f}")

# ==========================================================
# Plot concentration profiles
# ==========================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

for profile, t_plot in zip(profiles, profile_times):
    plt.plot(
        z * 1e6,
        profile,
        lw=2,
        label=f"{t_plot:.0f} s"
    )

plt.xlabel("z (μm)")
plt.ylabel("c(z,t) (mol/m³)")
plt.title("Analyte Concentration in Bulk")
plt.grid(True)
plt.legend()

# ==========================================================
# Plot binding kinetics
# ==========================================================

plt.subplot(1, 2, 2)

time_history = np.array(time_history)
cb_history = np.array(cb_history)

plt.plot(
    time_history,
    cb_history,
    lw=2,
    label=r"$c_b(t)$"
)

plt.plot(
    time_history,
    cb_history / cbsat,
    "--",
    lw=2,
    label=r"$\theta=c_b/c_{b,sat}$"
)

plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.title("Antibody Binding Kinetics")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()