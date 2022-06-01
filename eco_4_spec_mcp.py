import casadi as ca
import numpy as np
from infrastructure import *
import siconos.numerics as sn
from scipy.interpolate import interp1d

Mx = discrete_patches(100, 30)
light_levels = [1]
time_lengths = [1]

tot_times = len(time_lengths)

tot_points = Mx.x.size
tot_cont_p = tot_points

inte = np.ones(tot_points).reshape(1, tot_points)

h = 20 / 365
a = 0.4
m0 = 10 ** (-3)
k = 0.1
masses = np.array([0.1, 11])
r = 1
eps0 = 0.05  # 0.05
R_max = 1  # Varied between 5 and 100
z_mld = 0
sigma = 5
Cmax = h * masses ** (-0.25)
epsi = eps0 * ((1 - a) * np.log(masses / m0) - np.log(eps0))
rz = R_max * np.exp(-((Mx.x - z_mld))**2 / (sigma ** 2))
upright_wc = np.exp(-k * Mx.x)  # .reshape((-1, 1))
c_z = Cmax[0]
c_ff = Cmax[1]

f_c = 0.15 / 365
bg_M = 0.1 / 365

beta_0 = 10 ** (-4)

Vi = 1
beta_i = 330 / 365 * Vi * masses ** (-0.25)
beta_z = (masses[0] ** (-0.25) * upright_wc ** 0)
beta_ff = (2 * beta_i[1] * upright_wc / (1 + upright_wc) + beta_0)
lam = ca.MX.sym('lam', 2)
sigma_z = ca.MX.sym('sigma_z', tot_cont_p)
sigma_ff = ca.MX.sym('sigma_ff', tot_cont_p)
state = np.ones(2)
res_level = rz


z_pp_enc = (inte @ (Mx.M @ (beta_z * sigma_z * res_level)))
z_satiation = (1 / (z_pp_enc + c_z))  # c_z*ff_z_enc

ff_z_enc = (inte @ (Mx.M @ (beta_ff * sigma_z * sigma_ff)))
ff_satiation = (1 / (state[0] * ff_z_enc + c_ff))  # c_ff*ff_z_enc

df_z = (c_z ** 2 * beta_z * res_level * z_satiation ** 2
            - 1/epsi[0] * state[1] * c_ff * sigma_ff * beta_ff * ff_satiation +
    lam[0] * np.ones(tot_points))

df_ff = (state[0] * c_ff ** 2 * beta_ff * sigma_z * ff_satiation ** 2
             + lam[1] * np.ones(tot_points))

p1 = inte @ (Mx.M @ sigma_z) - 1
p2 = inte @ (Mx.M @ sigma_ff) - 1

probability = ca.vertcat(p1, p2)

normal_cone = (ca.vertcat(-df_z, -df_ff))

f = ca.vertcat(*[probability, normal_cone])

sigmas = ca.vertcat(sigma_z, sigma_ff)
xo = ca.vertcat(*[lam, sigmas])

mcp_function_ca = ca.Function('fun', [xo], [f])
mcp_Nablafunction_ca = ca.Function('fun', [xo], [ca.jacobian(f, xo)])


def mcp_function(n, z, F):
    F[:] = np.array(*[mcp_function_ca(z)]).flatten()
    pass

def mcp_Nablafunction(n, z, nabla_F):
    nabla_F[:] = mcp_Nablafunction_ca(z)
    pass

def mcp_solver():
    tot_times = 1
    tot_points = Mx.x.shape[0]
    tot = 2*tot_points*tot_times + tot_times*2
    mcp = sn.MCP(2, tot -2, mcp_function, mcp_Nablafunction)

    z = np.ones(tot, dtype=float)
    w = np.ones(tot, dtype=float)
    SO = sn.SolverOptions(sn.SICONOS_MCP_NEWTON_FB_FBLSA)
    info = sn.mcp_newton_FB_FBLSA(mcp, z, w, SO)
    print("z = ", z)
    print("w = ", w)
    print(info)

mcp_solver()

