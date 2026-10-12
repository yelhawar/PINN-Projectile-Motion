"""
Physics-Informed Neural Network (PINN) vs. classical ODE fitting
for 2D projectile motion under quadratic air drag.

Physics research:
"A Comparative Analysis of Physics-Informed Neural Networks and
Traditional Numerical Methods for Modelling Projectile Motion under
Air Resistance."
"""

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim

# ---------------- config ----------------

g = 9.81          # gravity (m/s^2)
m = 0.057         # tennis ball mass (kg)
D = 0.067         # diameter (m)
A = np.pi * (D / 2) ** 2
rho = 1.225       # air density (kg/m^3)

Cd_true = 0.5
c_true = 0.5 * Cd_true * rho * A   # F_d = c * v^2

v0 = 30.0
angle_deg = 45.0

vx0 = v0 * np.cos(np.radians(angle_deg))
vy0 = v0 * np.sin(np.radians(angle_deg))
state0 = [0.0, 0.0, vx0, vy0]

DEBUG = False

# ---------------- dynamics ----------------

def f_dyn(t, s, c):
    # state: [x, y, vx, vy]
    x, y, vx, vy = s
    v = (vx ** 2 + vy ** 2) ** 0.5
    ax = -(c / m) * v * vx
    ay = -g - (c / m) * v * vy
    return [vx, vy, ax, ay]


# ---------------- generate "true" trajectory ----------------

T_MAX = 5.0
t_all = np.linspace(0, T_MAX, 600)

print("Simulating high-res trajectory...")
sol_true = solve_ivp(
    f_dyn, (0, T_MAX), state0,
    args=(c_true,),
    t_eval=t_all,
    rtol=1e-9
)

x_full = sol_true.y[0]
y_full = sol_true.y[1]

# keep only while above ground
mask = y_full >= 0
t_full = t_all[mask]
x_full = x_full[mask]
y_full = y_full[mask]

if DEBUG:
    print("len(t_full) =", len(t_full))

# ---------------- noisy measurements ----------------

np.random.seed(42)
n_samples = 60

t_data = np.linspace(0, t_full[-1], n_samples)

sol_samp = solve_ivp(
    f_dyn, (0, t_full[-1]), state0,
    args=(c_true,),
    t_eval=t_data,
    rtol=1e-9
)

x_clean = sol_samp.y[0]
y_clean = sol_samp.y[1]

sigma_x = 0.05  # m
sigma_y = 0.08  # m

X_data = x_clean + np.random.normal(0, sigma_x, size=x_clean.shape)
Y_data = y_clean + np.random.normal(0, sigma_y, size=y_clean.shape)

# ---------------- ODE least squares fit ----------------

def estimate_v0(t_arr, x_arr, y_arr, n=3):
    # rough linear fit on the first n points
    vx_est = np.polyfit(t_arr[:n], x_arr[:n], 1)[0]
    vy_est = np.polyfit(t_arr[:n], y_arr[:n], 1)[0]
    return vx_est, vy_est


def residual_logc(logc, t_obs, x_obs, y_obs, vx_guess, vy_guess):
    c = np.exp(logc[0])  # keep positive
    sol = solve_ivp(
        f_dyn,
        (t_obs[0], t_obs[-1]),
        [0.0, 0.0, vx_guess, vy_guess],
        args=(c,),
        t_eval=t_obs,
        rtol=1e-8
    )
    x_pred = sol.y[0]
    y_pred = sol.y[1]
    res = np.concatenate([x_pred - x_obs, y_pred - y_obs])
    return res


print("Estimating initial velocities from first few data points...")
vx_est, vy_est = estimate_v0(t_data, X_data, Y_data)
if DEBUG:
    print("vx_est, vy_est =", vx_est, vy_est)

logc0 = np.log(c_true)
res_ls = least_squares(
    residual_logc,
    x0=[logc0],
    args=(t_data, X_data, Y_data, vx_est, vy_est),
    verbose=0
)

c_fit_ode = float(np.exp(res_ls.x[0]))
Cd_fit_ode = 2 * c_fit_ode / (rho * A)

sol_ode_fit = solve_ivp(
    f_dyn,
    (t_data[0], t_data[-1]),
    [0.0, 0.0, vx_est, vy_est],
    args=(c_fit_ode,),
    t_eval=t_data,
    rtol=1e-8
)
x_ode_fit = sol_ode_fit.y[0]
y_ode_fit = sol_ode_fit.y[1]

sol_ode_true = solve_ivp(
    f_dyn,
    (t_data[0], t_data[-1]),
    state0,
    args=(c_true,),
    t_eval=t_data,
    rtol=1e-9
)
x_ode_true = sol_ode_true.y[0]
y_ode_true = sol_ode_true.y[1]

print("== ODE fit summary ==")
print("c_true      =", c_true)
print("c_fit_ode   =", c_fit_ode)
print("Cd_true     =", Cd_true)
print("Cd_fit_ode  =", Cd_fit_ode)
print()

# ---------------- PINN ----------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

T_t = torch.tensor(t_data, dtype=torch.float32, device=device).view(-1, 1)
T_t.requires_grad_(True)

X_t = torch.tensor(X_data, dtype=torch.float32, device=device).view(-1, 1)
Y_t = torch.tensor(Y_data, dtype=torch.float32, device=device).view(-1, 1)

t0 = torch.tensor([[0.0]], dtype=torch.float32, device=device, requires_grad=True)
x0_tgt = torch.tensor([[0.0]], dtype=torch.float32, device=device)
y0_tgt = torch.tensor([[0.0]], dtype=torch.float32, device=device)
vx0_tgt = torch.tensor([[vx0]], dtype=torch.float32, device=device)
vy0_tgt = torch.tensor([[vy0]], dtype=torch.float32, device=device)


class PINNNet(nn.Module):
    def __init__(self, hidden=(128, 128, 128)):
        super().__init__()
        layers = []
        in_dim = 1
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, 2))  # outputs x, y
        self.net = nn.Sequential(*layers)

        # trainable drag constant (raw)
        self.c_raw = nn.Parameter(torch.tensor(-7.0, dtype=torch.float32, device=device))

    def forward(self, t):
        return self.net(t)

    def c_phys(self):
        # softplus keeps the drag coefficient strictly positive
        return torch.nn.functional.softplus(self.c_raw)


torch.manual_seed(0)
model = PINNNet().to(device)

w_data = 1.0
w_phys = 1.0
w_ic = 10.0

adam_lr = 2e-3
adam_epochs = 3000

opt = optim.Adam(model.parameters(), lr=adam_lr)

history = {"total": [], "data": [], "phys": [], "ic": [], "c": []}


def compute_losses(verbose=False):
    pred = model(T_t)
    x_pred = pred[:, 0:1]
    y_pred = pred[:, 1:2]

    # data mismatch
    L_data = ((x_pred - X_t) ** 2).mean() + ((y_pred - Y_t) ** 2).mean()

    # first derivatives wrt time
    x_t = torch.autograd.grad(
        x_pred, T_t,
        grad_outputs=torch.ones_like(x_pred),
        create_graph=True
    )[0]
    y_t = torch.autograd.grad(
        y_pred, T_t,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True
    )[0]

    # second derivatives
    x_tt = torch.autograd.grad(
        x_t, T_t,
        grad_outputs=torch.ones_like(x_t),
        create_graph=True
    )[0]
    y_tt = torch.autograd.grad(
        y_t, T_t,
        grad_outputs=torch.ones_like(y_t),
        create_graph=True
    )[0]

    v = torch.sqrt(x_t ** 2 + y_t ** 2 + 1e-12)
    c_hat = model.c_phys()

    rx = x_tt + (c_hat / m) * v * x_t
    ry = y_tt + g + (c_hat / m) * v * y_t

    L_phys = (rx ** 2).mean() + (ry ** 2).mean()

    # initial conditions at t=0
    p0 = model(t0)
    x0_pred = p0[:, 0:1]
    y0_pred = p0[:, 1:2]

    vx0_pred = torch.autograd.grad(
        x0_pred, t0,
        grad_outputs=torch.ones_like(x0_pred),
        create_graph=True
    )[0]
    vy0_pred = torch.autograd.grad(
        y0_pred, t0,
        grad_outputs=torch.ones_like(y0_pred),
        create_graph=True
    )[0]

    L_ic = (
        (x0_pred - x0_tgt) ** 2
        + (y0_pred - y0_tgt) ** 2
        + (vx0_pred - vx0_tgt) ** 2
        + (vy0_pred - vy0_tgt) ** 2
    ).mean()

    L_total = w_data * L_data + w_phys * L_phys + w_ic * L_ic

    if verbose and DEBUG:
        print("L_total, data, phys, ic =", L_total.item(), L_data.item(), L_phys.item(), L_ic.item())

    return (
        L_total,
        L_data.detach().cpu().item(),
        L_phys.detach().cpu().item(),
        L_ic.detach().cpu().item(),
        c_hat.detach().cpu().item(),
    )


print("Training PINN (Adam)...")
start_time = time.time()
for ep in range(1, adam_epochs + 1):
    opt.zero_grad()
    L_total, Ld, Lp, Lic, c_val = compute_losses()
    L_total.backward()
    opt.step()

    if ep % 50 == 0 or ep == 1:
        history["total"].append(L_total.item())
        history["data"].append(Ld)
        history["phys"].append(Lp)
        history["ic"].append(Lic)
        history["c"].append(c_val)
        print(
            f"epoch {ep:4d} | "
            f"total={L_total.item():.3e}, data={Ld:.3e}, phys={Lp:.3e}, ic={Lic:.3e}, c_hat={c_val:.3e}"
        )

print("Adam done in", round(time.time() - start_time, 2), "s")
print("Running L-BFGS (polish step)...")

opt_lbfgs = optim.LBFGS(model.parameters(), max_iter=200, lr=0.8, history_size=50)


def closure():
    opt_lbfgs.zero_grad()
    L_total, _, _, _, _ = compute_losses()
    L_total.backward()
    return L_total


opt_lbfgs.step(closure)

model.eval()
with torch.no_grad():
    pred_T = model(T_t)
    x_pinn = pred_T[:, 0].cpu().numpy()
    y_pinn = pred_T[:, 1].cpu().numpy()

c_pinn = float(model.c_phys().detach().cpu().numpy())
Cd_pinn = 2 * c_pinn / (rho * A)

print("\n== PINN result ==")
print("c_phys (PINN) =", c_pinn)
print("Cd (PINN)     =", Cd_pinn)
print()

# ODE using c from PINN
sol_ode_pinnc = solve_ivp(
    f_dyn,
    (t_data[0], t_data[-1]),
    state0,
    args=(c_pinn,),
    t_eval=t_data,
    rtol=1e-9
)
x_ode_pinnc = sol_ode_pinnc.y[0]
y_ode_pinnc = sol_ode_pinnc.y[1]

# ---------------- metrics + comparison table ----------------

def metric_fun(x_pred, y_pred, x_obs, y_obs):
    rmse_x = np.sqrt(mean_squared_error(x_obs, x_pred))
    rmse_y = np.sqrt(mean_squared_error(y_obs, y_pred))
    rmse_total = np.sqrt(np.mean((x_obs - x_pred) ** 2 + (y_obs - y_pred) ** 2))

    r2_x = r2_score(x_obs, x_pred)
    r2_y = r2_score(y_obs, y_pred)
    r2_mean = 0.5 * (r2_x + r2_y)

    return dict(
        RMSE_x=rmse_x,
        RMSE_y=rmse_y,
        RMSE_total=rmse_total,
        R2_x=r2_x,
        R2_y=r2_y,
        R2_mean=r2_mean,
    )


m_true = metric_fun(x_ode_true, y_ode_true, X_data, Y_data)
m_odefit = metric_fun(x_ode_fit, y_ode_fit, X_data, Y_data)
m_odepinnc = metric_fun(x_ode_pinnc, y_ode_pinnc, X_data, Y_data)
m_pinn = metric_fun(x_pinn, y_pinn, X_data, Y_data)

rows = [
    {"Model": "ODE true c", "c_phys": c_true, "Cd": Cd_true, **m_true},
    {"Model": "ODE fitted c", "c_phys": c_fit_ode, "Cd": Cd_fit_ode, **m_odefit},
    {"Model": "ODE PINN c", "c_phys": c_pinn, "Cd": Cd_pinn, **m_odepinnc},
    {"Model": "PINN direct", "c_phys": c_pinn, "Cd": Cd_pinn, **m_pinn},
]

df = pd.DataFrame(rows)

num_cols = ["c_phys", "Cd", "RMSE_x", "RMSE_y", "RMSE_total", "R2_x", "R2_y", "R2_mean"]
avg = df[num_cols].mean()
std = df[num_cols].std()

avg_row = {"Model": "avg ± sd"}
for col in num_cols:
    avg_row[col] = f"{avg[col]:.4g} ± {std[col]:.4g}"

df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

pd.set_option("display.float_format", "{:0.6f}".format)
print("=== comparison vs noisy data ===")
print(df.to_string(index=False))

df.to_csv("model_comparison_table.csv", index=False)

# ---------------- plots ----------------

plt.figure(figsize=(10, 6))
plt.scatter(X_data, Y_data, s=18, alpha=0.6, label="noisy data")
plt.plot(x_ode_true, y_ode_true, "k--", lw=2, label=f"ODE true (Cd={Cd_true:.2f})")
plt.plot(x_ode_fit, y_ode_fit, "g-.", lw=2, label=f"ODE fit (Cd={Cd_fit_ode:.3f})")
plt.plot(x_ode_pinnc, y_ode_pinnc, "c:", lw=2, label=f"ODE + PINN c (Cd={Cd_pinn:.3f})")
plt.plot(x_pinn, y_pinn, "r-", lw=2, label="PINN")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Projectile with drag: data vs models")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("trajectory_comparison.png", dpi=300)

plt.figure(figsize=(9, 5))
plt.plot(t_data, Y_data - y_pinn, label="data - PINN")
plt.plot(t_data, Y_data - y_ode_fit, label="data - ODE fit")
plt.plot(t_data, Y_data - y_ode_true, label="data - ODE true")
plt.axhline(0, color="k", ls="--", lw=0.8)
plt.xlabel("time (s)")
plt.ylabel("vertical error (m)")
plt.title("Vertical error vs time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("vertical_errors.png", dpi=300)

if history["total"]:
    plt.figure(figsize=(8, 4))
    iters = np.arange(1, len(history["total"]) + 1)
    plt.plot(iters, history["total"], label="total")
    plt.plot(iters, history["data"], label="data")
    plt.plot(iters, history["phys"], label="phys")
    plt.yscale("log")
    plt.xlabel("checkpoint (every 50 epochs)")
    plt.ylabel("loss")
    plt.title("PINN training history")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_loss.png", dpi=300)

print("\nSaved files:")
print("  - trajectory_comparison.png")
print("  - vertical_errors.png")
print("  - training_loss.png (if history not empty)")
print("  - model_comparison_table.csv")
