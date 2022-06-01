import casadi as ca
import numpy as np
from infrastructure import *
import matplotlib.pyplot as plt
import siconos.numerics as sn

class discrete_patches:
    def __init__(self, depth, total_points):
        self.x = np.linspace(0, depth, total_points)

        self.M = depth/total_points * np.identity(total_points)

Mx = spectral_method(5, 100, segments = 1) #spectral_method(4, 20) #spectral_method(5,50) # discrete_patches(1,20)#

tot_points = Mx.x.size
inte = np.ones(tot_points).reshape(1,tot_points)
s = ca.MX.sym('s', 2)
state = np.ones(2)
s = state

car_cap = 5
par = {'res_renew': 1, 'eff': 0.1, 'c_handle': 1, 'c_enc_freq': 1, 'c_met_loss': 0.001, 'p_handle': 0.01, 'p_enc_freq': 0.1, 'p_met_loss': 0.15, 'competition': 0, 'q': 3}
k = 0.08

gridx, gridy = np.meshgrid(Mx.x, Mx.x)
ker = lambda x, y: np.exp(-(x - y) ** 2 / (4 * k)) + np.exp(-(-y - x) ** 2 / (4 * k)) + np.exp(-(2 * Mx.x[-1] - x - y) ** 2 / (4 * k))
out = (4 * k * np.pi) ** (-1 / 2) * ker(gridx, gridy)
for j in range(tot_points):
    out[j,:] = out[j,:]/(inte @ Mx.M @ out[j,:])
#out_inv = np.linalg.inv(np.transpose(out))

res_conc = np.exp(-par['q']*Mx.x) #np.exp(-Mx.x**2)#np.exp(-Mx.x)+0.001
res_conc = 1/(inte @ (Mx.M @ res_conc))*res_conc

beta = np.exp(-(par['q']*Mx.x)**2) #+0.001 2*np.exp(-Mx.x)/(1+np.exp(-Mx.x))#np.exp(-Mx.x**2)#+0.001
beta = 0.5*1 / (inte @ (Mx.M @ beta)) * beta + 10**(-4)

lam = ca.MX.sym('lam', 2)
sigma_o = ca.MX.sym('sigma', Mx.x.shape[0])
sigma_o_p = ca.MX.sym('sigma_p', Mx.x.shape[0])

sigma = np.transpose(out) @ sigma_o
sigma_p = np.transpose(out) @ sigma_o_p
#for j in range(tot_points):
#    sigma += sigma_o[j]*out[j,:]
#    sigma_p += sigma_o_p[j]*out[j,:]
trans_out = np.transpose(out)

cons_dyn = inte @ (Mx.M @ ( sigma / par['c_enc_freq'] * (
            1 - s[0] * (sigma) ** 2 / (par['c_enc_freq'] * res_conc * car_cap)))) - inte @ (
                       Mx.M @ (s[1] * sigma * beta * sigma_p)) / (
                       par['p_enc_freq'] + par['p_handle'] * inte @ (Mx.M @ (s[0] * sigma * beta * sigma_p)))
pred_dyn = par['eff'] * inte @ (Mx.M @ (s[0] * sigma * beta * sigma_p)) / (
            par['p_enc_freq'] + par['p_handle'] * inte @ (Mx.M @ (s[0] * sigma * beta * sigma_p))) - par[
               'p_met_loss'] - par['competition'] * inte @ (Mx.M @ (sigma_p ** 2 * beta))

df1 = (1 / par['c_enc_freq'] * (1 - s[0] * sigma / (res_conc * par['c_enc_freq'] * car_cap)) - s[
    1] * sigma_p * beta / (
                  par['p_enc_freq'] + inte @ (Mx.M @ (par['p_handle'] * s[0] * sigma * beta * sigma_p)))) + ca.log(lam[
          0]**2)
df2 = (par['eff'] * s[0] * par['p_enc_freq'] * sigma * beta / (
            inte @ (Mx.M @ (par['p_handle'] * s[0] * sigma * beta * sigma_p)) + par['p_enc_freq']) ** 2 - par['competition'] * sigma_p * beta) + ca.log(lam[
          1]**2)

df1_sic = trans_out @ (1 / par['c_enc_freq'] * (1 - s[0] * sigma / (res_conc * par['c_enc_freq'] * car_cap)) - s[
    1] * sigma_p * beta / (
                  par['p_enc_freq'] + inte @ (Mx.M @ (par['p_handle'] * s[0] * sigma * beta * sigma_p)))) + ca.log(lam[0]**2)
df2_sic = trans_out @ (par['eff'] * s[0] * par['p_enc_freq'] * sigma * beta / (
            inte @ (Mx.M @ (par['p_handle'] * s[0] * sigma * beta * sigma_p)) + par['p_enc_freq']) ** 2 - par['competition'] * sigma_p * beta) + ca.log(lam[
          1]**2)

sdot = ca.vertcat(s[0]*cons_dyn, s[1]*pred_dyn)

g1 = inte @ sigma - 1
g2 = inte @ sigma_p - 1
g3 = ca.log(inte @ Mx.M @ (df1**2 + df2**2))
g3_sic = ca.vertcat(-df1_sic, -df2_sic)

ipg = ca.vertcat(*[g1, g2])
z = ca.vertcat(*[sigma_o, sigma_o_p, lam])

s_opts = {'ipopt': {'print_level': 5, 'linear_solver': 'ma57'}}

prob = {'x': ca.vertcat(z), 'f': g3, 'g': ipg}
lbx = ca.vertcat(*[np.zeros(tot_points*2), 2*[0]])
solver = ca.nlpsol('solver', 'ipopt', prob, s_opts)

lbg = np.zeros(ipg.size()[0])
ubg = ca.vertcat(*[np.zeros(ipg.size()[0])])
init = np.ones(ca.vertcat(z).size()[0])/np.max(Mx.x)

sol = solver(lbx =lbx, lbg=lbg, ubg=ubg, x0=init)
t_x = np.array(sol['x']).flatten()

plt.plot(Mx.x, np.transpose(out)@((t_x[0:tot_points])))

func = ca.vertcat(*[g3_sic, g1, g2])
x = ca.vertcat(sigma_o, sigma_o_p, lam)
mcp_function_ca = ca.Function('fun', [x], [func])

plt.show()

mcp_Nablafunction_ca = ca.Function('fun', [x], [ca.jacobian(func, x)])


def mcp_function(n, z, F):
    F[:] = np.array(*[mcp_function_ca(z)]).flatten()
    pass


def mcp_Nablafunction(n, z, nabla_F):
    nabla_F[:] = mcp_Nablafunction_ca(z)
    pass

warmstart_info = np.zeros(2*t_x.size)

warmstart_info[0:t_x.size] = np.copy(t_x).flatten()
warmstart_info[t_x.size:] = np.array(mcp_function_ca(t_x)).flatten()

def mcp_solver():
    tot_points = x.size()[0]
    ncp = sn.NCP(tot_points, mcp_function,
                 mcp_Nablafunction)
    z = np.copy(warmstart_info[0:tot_points])
    w = np.copy(warmstart_info[tot_points:])
    print(np.dot(z, w))

    SO = sn.SolverOptions(
        sn.SICONOS_NCP_NEWTON_FB_FBLSA)  # sn.SolverOptions(sn.SICONOS_MCP_NEWTON_FB_FBLSA) #sn.SolverOptions(sn.SICONOS_NCP_NEWTON_FB_FBLSA)
    SO.dparam[sn.SICONOS_DPARAM_TOL] = 10 ** (-4)
    SO.iparam[sn.SICONOS_IPARAM_MAX_ITER] = 200
    #SO.iparam[sn.SICONOS_IPARAM_LSA_NONMONOTONE_LS] = 0.5
    info = sn.ncp_newton_FBLSA(ncp, z, w, SO)
    print(info, "Newton status", np.dot(z, w))
    print(np.dot(z, w))

    return np.concatenate([z, w])

mcp_sol = mcp_solver()
#print(out_inv @ a[50:100])
plt.plot(Mx.x, np.transpose(out) @ mcp_sol[0:tot_points])
print(mcp_sol[tot_points*2+2:tot_points*3+2])
#plt.plot(Mx.x, mcp_sol[tot_points*2+2:tot_points*3+2])


plt.show()