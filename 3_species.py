import casadi as ca
from infrastructure import *

Mx = discrete_patches(200, 30) #spectral_method(50, 50) #discrete_patches(100, 15)#spectral_method(50, 50)  # simple_method(50, 80)#spectral_method(50, 10, segments = 1) #spectral_method(30, 30, segments = 1)

light_levels = [0, 0.2, 1, 0.2] #[1, 0]
time_lengths = [0.4, 0.1, 0.4, 0.1] #[0.5, 0.5]


tot_times = len(time_lengths)
tot_points = Mx.x.size

print(tot_times, len(light_levels))


inte = np.ones(tot_points).reshape(1, tot_points)

h = 20 / 365
a = 0.4
m0 = 10 ** (-3)
gamma = 0.6
k = 0.05
masses = np.array([11, 4000, 4000])
f_c = 0.15 / 365
r = 1 / 365
r_b = 0.1 / 365
eps0 = 0.05
R_max = 10  # Varied between 5 and 100
z_mld = 30
sigma = 10
Cmax = h * masses ** (-0.25)
epsi = eps0 * ((1 - a) * np.log(masses / m0) - np.log(eps0))
rz = R_max * np.exp(-((Mx.x - z_mld)) ** 2 / (sigma ** 2))
Bmax = 3
upright_wc = np.exp(-k * Mx.x)  # .reshape((-1, 1))
c_ff = Cmax[0]
c_lp = Cmax[1]
c_ld = Cmax[2]
bg_M = 0.1 / 365

beta_ld_b = []
beta_ff = []
beta_lp = []
beta_ld = []
lam = []
sigma_ff = []
sigma_lp = []
sigma_ld = []
sigma_ld_b = []
state = []
res_level = []
beta_0 = 10 ** (-4)
for j in range(tot_times):
    Vi = light_levels[j]
    beta_i = 330/365* Vi * masses**(-0.25)
    beta_ld_b.append(330/365* masses[-1]**(-0.25)+beta_0)
    beta_ff.append(2*beta_i[0] * upright_wc/(1+upright_wc) + beta_0)
    beta_lp.append(2*beta_i[1] * upright_wc/(1+upright_wc) + beta_0)
    beta_ld.append(2*beta_i[2] * upright_wc/(1+upright_wc) + beta_0)

    lam_j = ca.MX.sym('lam' + str(j), 3)
    sigma_ff_j = ca.MX.sym('sigma_ff' + str(j), tot_points)
    sigma_lp_j = ca.MX.sym('sigma_lp' + str(j), tot_points)
    sigma_ld_j = ca.MX.sym('sigma_ld' + str(j), tot_points)
    sigma_ld_b_j = ca.MX.sym('sigma_ld_b' + str(j), 1)
    state_j = ca.MX.sym('state' + str(j), 4)
    res_level_j = ca.MX.sym('res_level' + str(j), tot_points)

    lam.append(lam_j)
    sigma_ff.append(sigma_ff_j)
    sigma_lp.append(sigma_lp_j)
    sigma_ld.append(sigma_ld_j)
    sigma_ld_b.append(sigma_ld_b_j)
    state.append(state_j)
    res_level.append(res_level_j)

ff_z_enc = []
ff_satiation = []
lp_ff_enc = []
lp_satiation = []
ld_ff_enc = []
ld_bc_enc = []
ld_satiation = []
bent_dyn = []


lp_loss = []
ld_loss = []
ld_loss_b = []
com_lp = 5
com_ld = 5

for j in range(tot_times):
    ff_z_enc.append(inte @ (Mx.M @ (beta_ff[j] * sigma_ff[j] * res_level[j])))
    ff_satiation.append(1 / (ff_z_enc[j] + c_ff))  # c_ff*ff_z_enc

    lp_ff_enc.append(inte @ (Mx.M @ (beta_lp[j] * sigma_ff[j] * sigma_lp[j])))
    lp_satiation.append(1 / (state[j][0] * lp_ff_enc[j] + c_lp))  # c_lp*lp_ff_enc

    ld_ff_enc.append(inte @ (Mx.M @ (beta_ld[j] * sigma_ff[j] * sigma_ld[j])))
    ld_bc_enc.append(beta_ld_b[j] * 0.5 * sigma_ld_b[j])

    ld_satiation.append(1 / (state[j][0] * ld_ff_enc[j] + state[j][3] * ld_bc_enc[j] + c_ld))

    lp_loss.append(com_lp * inte @ (Mx.M @ ( beta_lp[j] * sigma_lp[j]*sigma_lp[j])))
    ld_loss.append(com_ld * inte @ (Mx.M @ ( beta_ld[j] * sigma_ld[j]*sigma_ld[j])))
    ld_loss_b.append(com_ld*beta_ld_b[j] * sigma_ld_b[j]**2)

# Bentic dynamics

ff_dyn = []
lp_dyn = []
ld_dyn = []
res_dyn = []
bent_dyn = []

diff_eq_ff = 0
diff_eq_lp = 0
diff_eq_ld = 0
diff_eq_bc = 0
diff_eq_res = 0

for j in range(tot_times):
    res_dyn.append(
        r * (rz - res_level[j]) - state[j][0] * c_ff * beta_ff[j] * res_level[j] * sigma_ff[j] * ff_satiation[j])
    bent_dyn.append(r_b * (Bmax - state[j][3]) - c_ld * state[j][2] * state[j][2] * ld_bc_enc[j] * ld_satiation[j])
    ff_dyn.append(state[j][0] * (
                epsi[0] * (c_ff * ff_z_enc[j] * ff_satiation[j] - f_c) - state[j][1] * c_lp * lp_ff_enc[j] *
                lp_satiation[j] - state[j][2] * ld_ff_enc[j] * ld_satiation[j] - bg_M))
    lp_dyn.append(
        state[j][1] * (epsi[1] * (c_lp * state[j][0] * lp_ff_enc[j] * lp_satiation[j] - f_c) - lp_loss[j] - bg_M))
    ld_dyn.append(state[j][2] * (epsi[2] * (
                c_ld * state[j][0] * ld_ff_enc[j] * ld_satiation[j] + state[j][3] * ld_bc_enc[j] * c_ld * ld_satiation[
            j] - f_c) - ld_loss[j] - ld_loss_b[j] - bg_M))

    diff_eq_res += time_lengths[j] * res_dyn[j]

f = 0 #inte @ (Mx.M @ (diff_eq_res * diff_eq_res))

diff_eqs = []
# res_level_dyn = []
for j in range(tot_times):
    diff_eqs.append((state[j][0] - state[j - 1][0] - time_lengths[j - 1] * ff_dyn[j-1])**2)
    diff_eqs.append((state[j][1] - state[j - 1][1] - time_lengths[j - 1] * lp_dyn[j-1])**2)
    diff_eqs.append((state[j][2] - state[j - 1][2] - time_lengths[j - 1] * ld_dyn[j-1])**2)
    diff_eqs.append((state[j][3] - state[j - 1][3] - time_lengths[j - 1] * bent_dyn[j-1])**2)
    diff_eqs.append((res_level[j] - res_level[j - 1] - time_lengths[j - 1] * res_dyn[j-1]))

df_ff = []
df_lp = []  #
df_ld_up = []
df_ld_b = []
for j in range(tot_times):
    df_ff.append((epsi[0]) * c_ff ** 2 * beta_ff[j] * sigma_ff[j] * ff_satiation[j] ** 2
                 - state[j][1] * c_lp * sigma_lp[j] * beta_lp[j] * lp_satiation[j]
                 - state[j][2] * c_ld * sigma_ld[j] * beta_ld[j] * ld_satiation[j] + lam[j][0])
    df_lp.append((epsi[1]) * state[j][0] * sigma_lp[j] * c_lp ** 2 * beta_lp[j] * lp_satiation[j] ** 2 - com_lp*sigma_lp[j]*beta_lp[j] + lam[j][
        1])
    df_ld_up.append(epsi[2] * state[j][0] * sigma_ld[j] * c_ld ** 2 * beta_ld[j] * ld_satiation[j] ** 2 - com_ld*sigma_ld[j]*beta_ld[j] + lam[j][2])
    df_ld_b.append(epsi[2] * beta_ld_b[j] * 0.5 * state[j][3] * c_ld ** 2 * ld_satiation[j] ** 2 - com_ld*beta_ld_b[j]*sigma_ld_b[j] + lam[j][2])

p1 = inte @ Mx.M @ sigma_ff[0] - 1
p2 = inte @ Mx.M @ sigma_lp[0] - 1
p3 = inte @ Mx.M @ sigma_ld[0] + sigma_ld_b[0] - 1

prob_pres = []
for j in range(1, tot_times):  # Preservation of probability
    prob_pres.append(inte @ Mx.M @ (sigma_ff[j] - 1))
    prob_pres.append(inte @ Mx.M @ (sigma_lp[j] - 1))
    prob_pres.append(inte @ Mx.M @ (sigma_ld[j]) + sigma_ld_b[j] - 1)

probability = ca.vertcat(*[*[p1, p2, p3], *prob_pres])
# Zeros contributed: 4*total_times
complementarity = []
normal_cone = []
for j in range(tot_times):  # Complementarity at every poitn, we need to be in the normal cone as well
    complementarity.append(inte @ (Mx.M @ (df_ff[j] * sigma_ff[j])) +
                           inte @ (Mx.M @ (df_lp[j] * sigma_lp[j])) + inte @ (Mx.M @ (df_ld_up[j] * sigma_ld[j]))
                           + sigma_ld_b[j] * df_ld_b[j])
    normal_cone.append(ca.vertcat(-df_ff[j], -df_lp[j], -df_ld_up[j], -df_ld_b[j]))

# Lower zeros contributed: 2*total_times + 4*tot_points*tot_times
# Unbounded above: 4*tot_points*tot_times + tot_times

#
g = ca.vertcat(*[diff_eq_ff, diff_eq_lp, diff_eq_ld, diff_eq_bc, *diff_eqs, probability, *complementarity, *normal_cone])
print(g.size())
lbg = np.zeros(g.size()[0])
ubg = ca.vertcat(*[*np.zeros(g.size()[0] - (3 * tot_points * tot_times + tot_times)),
                   [ca.inf] * (3 * tot_points * tot_times + tot_times)])

sigmas = ca.vertcat(*[*sigma_ff, *sigma_lp, *sigma_ld, *sigma_ld_b])  # sigma_bar
x = ca.vertcat(*[sigmas, *res_level, *state, *lam])
print(g.size(), x.size(), 4*tot_points*tot_times, 4*tot_times, 4*tot_times)

lbx = ca.vertcat(*[np.zeros(4 * tot_points * tot_times + 5 * tot_times), 3 * tot_times * [-ca.inf]])
print("Lower bounds")
s_opts = {'ipopt': {'print_level': 5, 'linear_solver': 'ma57'}}  # 'hessian_approximation':'limited-memory',
init = np.ones(x.size()[0]) / Mx.x[-1]
init[-7 * tot_times:-3 * tot_times] = 10  # np.array([0.0265501, 0, 5.6027, 0.940087])
print("Init set")
prob = {'x': x, 'f': f, 'g': g}
print("Problem formulated")
solver = ca.nlpsol('solver', 'ipopt', prob, s_opts)
print("Solving")
sol = solver(lbx=lbx, lbg=lbg, ubg=ubg, x0=init)
print(sol['x'])

print(sol['x'][-(7 * tot_times):-3 * tot_times].size())
print(sol['x'][-7 * tot_times:-3 * tot_times])
print(sol['x'][-(7 + tot_points) * tot_times:-7 * tot_times])
print(sol['x'][0:tot_points * 4 + 1])
print(sol['x'].size())
