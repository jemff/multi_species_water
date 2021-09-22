import casadi as ca
import numpy as np
from infrastructure import *

Mx = spectral_method(50, 60) #simple_method(50, 80)#spectral_method(50, 10, segments = 1) #spectral_method(30, 30, segments = 1)
light_levels = [0, 1]#[0, 20, 40, 20]
time_lengths = [0.5, 0.5] #[0.4, 0.1, 0.4, 0.1] #sum to 1

tot_times = len(time_lengths)

tot_points = Mx.x.size
tot_cont_p = tot_points

inte = np.ones(tot_points).reshape(1,tot_points)

h = 20/365
a = 0.4
m0 = 10**(-3)
gamma = 0.6
k = 0.09
masses = np.array([0.1, 11, 5000, 5000])
f_c = 0.15/365
r = 1
r_b = 1/365
eps0 = 0.01
R_max = 30 #Varied between 5 and 100
z_mld = 10
sigma = 3
Cmax = h * masses **(-0.25)
epsi = eps0*((1-a)*np.log(masses/m0) - np.log(eps0))
rz = R_max*np.exp(-((Mx.x-z_mld))**2/(sigma**2))
Bmax = 30
upright_wc = np.exp(-k * Mx.x)#.reshape((-1, 1))
c_z = Cmax[0]
c_ff = Cmax[1]
c_lp = Cmax[2]
c_ld = Cmax[3]
bg_M = 0.1/365

beta_ld_b = []
beta_z = []
beta_ff = []
beta_lp = []
beta_ld = []
lam = []
sigma_z = []
sigma_ff = []
sigma_lp = []
sigma_ld = []
sigma_ld_b = []
state = []
res_level = []

for j in range(tot_times):
    Vi = light_levels[j]
    beta_i = Vi/masses
    beta_ld_b.append(1/masses[-1])
    beta_z.append(max(light_levels)/masses[0] * upright_wc**0)
    beta_ff.append(beta_i[1] * upright_wc)
    beta_lp.append(beta_i[2] * upright_wc)
    beta_ld.append(beta_i[3] * upright_wc)

    lam_j = ca.MX.sym('lam'+str(j), 4)
    sigma_z_j = ca.MX.sym('sigma_z'+str(j), tot_cont_p)
    sigma_ff_j = ca.MX.sym('sigma_ff'+str(j), tot_cont_p)
    sigma_lp_j = ca.MX.sym('sigma_lp'+str(j), tot_cont_p)
    sigma_ld_j = ca.MX.sym('sigma_ld'+str(j), tot_cont_p)
    sigma_ld_b_j = ca.MX.sym('sigma_ld_b'+str(j), 1)
    state_j = ca.MX.sym('state'+str(j), 5)
    res_level_j = ca.MX.sym('res_level'+str(j), tot_cont_p)

    lam.append(lam_j)
    sigma_z.append(sigma_z_j)
    sigma_ff.append(sigma_ff_j)
    sigma_lp.append(sigma_lp_j)
    sigma_ld.append(sigma_ld_j)
    sigma_ld_b.append(sigma_ld_b_j)
    state.append(state_j)
    res_level.append(res_level_j)
#Resource dynamics
#D_2 = (Mx.D @ Mx.D)
#D_2[0] = np.copy(Mx.D[0])
#D_2[-1] = np.copy(Mx.D[-1])
#almost_ID = np.identity(tot_points)
#almost_ID[0,0] = 0
#almost_ID[-1,-1] = 0
lp_loss = 0
ld_loss = 0
z_pp_enc = []
z_satiation = []
ff_z_enc = []
ff_satiation = []
lp_ff_enc = []
lp_satiation = []
ld_ff_enc = []
ld_bc_enc = []
ld_satiation = []
bent_dyn = []

for j in range(tot_times):
    z_pp_enc.append(inte @ (Mx.M @ (beta_z[j]*sigma_z[j]*res_level[j])))
    z_satiation.append(1/(z_pp_enc[j]+c_z)) #c_z*ff_z_enc

    ff_z_enc.append(state[j][0]*inte @ (Mx.M @ ((beta_ff[j])*sigma_z[j] * sigma_ff[j])))
    ff_satiation.append(1/(state[j][0]*ff_z_enc[j]+c_ff)) # c_ff*ff_z_enc

    lp_ff_enc.append(inte @ (Mx.M @ ((beta_lp[j]) * sigma_ff[j] * sigma_lp[j])))
    lp_satiation.append(1/(state[j][1]*lp_ff_enc[j]+c_lp))  #c_lp*lp_ff_enc

    ld_ff_enc.append(inte @ (Mx.M @ ((beta_ld[j]) * sigma_ff[j] * sigma_lp[j])))
    ld_bc_enc.append(beta_ld_b[j]*0.5*sigma_ld_b[j])

    ld_satiation.append(1/(state[j][1]*ld_ff_enc[j]+state[j][-1]*ld_bc_enc[j]+c_ld))

#Bentic dynamics

z_dyn = []
ff_dyn = []
lp_dyn = []
ld_dyn = []
res_dyn = []
bent_dyn = []

diff_eq_z = 0
diff_eq_ff = 0
diff_eq_lp = 0
diff_eq_ld = 0
diff_eq_bc = 0
diff_eq_res = 0

for j in range(tot_times):
    res_dyn.append(r * (rz - res_level[j]) - state[j][0] * c_z * beta_z[j] * res_level[j] * sigma_z[j] * z_satiation[j])
    bent_dyn.append(r_b * (Bmax - state[j][-1]) - c_ld*state[j][3]*state[j][4]*ld_bc_enc[j] * ld_satiation[j])
    z_dyn.append(state[j][0]*(epsi[0]*(c_z*z_pp_enc[j]*z_satiation[j] - f_c) - state[j][1]*c_ff*ff_z_enc[j]*ff_satiation[j] - bg_M))
    ff_dyn.append(state[j][1]*(epsi[1]*(c_ff*state[j][0]*ff_z_enc[j]*ff_satiation[j] - f_c)- state[j][2] * c_lp * lp_ff_enc[j] * lp_satiation[j] - state[j][3] * ld_ff_enc[j] * ld_satiation[j] - bg_M))
    lp_dyn.append(state[j][2]*(epsi[2]*(c_lp*state[j][1]*lp_ff_enc[j]*lp_satiation[j] - f_c) - lp_loss - bg_M))
    ld_dyn.append(state[j][3]*(epsi[3]*(c_ld*state[j][1]*ld_ff_enc[j]*ld_satiation[j] + state[j][4]*ld_bc_enc[j]*c_ld*ld_satiation[j] - f_c)- ld_loss - bg_M))

    diff_eq_res += time_lengths[j]*res_dyn[j]
    diff_eq_bc += time_lengths[j]*bent_dyn[j]
    diff_eq_z += time_lengths[j]*z_dyn[j]
    diff_eq_ff += time_lengths[j]*ff_dyn[j]
    diff_eq_lp += time_lengths[j]*lp_dyn[j]
    diff_eq_ld += time_lengths[j]*ld_dyn[j]
#    print(j)

#print(diff_eq_res.size())
#print(Mx.M @ diff_eq_res)
f = inte @ (Mx.M @ (diff_eq_res*diff_eq_res))


df_z = []
df_ff = []
df_lp = [] #
df_ld_up = []
df_ld_b = []
for j in range(tot_times):
    df_z.append(c_z**2*beta_z[j]*res_level[j]*z_satiation[j]**2 - 1/(epsi[0])*state[j][1]*c_ff*sigma_ff[j]*beta_ff[j]*ff_satiation[j] + lam[j][0]) #epsi[0]*
    df_ff.append(state[j][0]*c_ff**2*beta_ff[j]*sigma_z[j]*ff_satiation[j]**2
                 - 1/(epsi[1])*state[j][2]*c_lp*sigma_lp[j]*beta_lp[j]*lp_satiation[j]
                 - 1/(epsi[1])*state[j][3]*c_ld*sigma_ld[j]*beta_ld[j]*ld_satiation[j] + lam[j][1])
    df_lp.append(state[j][1]*sigma_lp[j]*c_lp**2*beta_lp[j]*lp_satiation[j]**2 + lam[j][2]) #-loss derivative
    df_ld_up.append(state[j][1]*sigma_ld[j]*c_ld**2*beta_ld[j]*ld_satiation[j]**2 + lam[j][3])
    df_ld_b.append(beta_ld_b[j]*0.5*state[j][-1]*c_ld**2*ld_satiation[j]**2 + lam[j][3])

diff_eqs = []
#res_level_dyn = []
for j in range(tot_times):
    #diff_eqs.append([0])
    #diff_eqs.append([0])
    #diff_eqs.append([0])
    #diff_eqs.append([0])
    #diff_eqs.append([0])
    #diff_eqs.append([0]*tot_points)

    diff_eqs.append((state[j][0] - state[j-1][0] - time_lengths[j-1]*z_dyn[j-1]))
    diff_eqs.append((state[j][1] - state[j-1][1] - time_lengths[j-1]*ff_dyn[j-1]))
    diff_eqs.append((state[j][2] - state[j-1][2] - time_lengths[j-1]*lp_dyn[j-1]))
    diff_eqs.append((state[j][3] - state[j-1][3] - time_lengths[j-1]*ld_dyn[j-1]))
    diff_eqs.append((state[j][4] - state[j-1][4] - time_lengths[j-1]*bent_dyn[j-1]))
    diff_eqs.append((res_level[j] - res_level[j-1] - time_lengths[j-1]*res_dyn[j-1]))

#Zeros contributed 5*total times


p1 = inte @ Mx.M @ sigma_z[0] - 1
p2 = inte @ Mx.M @ sigma_ff[0] - 1
p3 = inte @ Mx.M @ sigma_lp[0] - 1
p4 = inte @ Mx.M @ sigma_ld[0] + sigma_ld_b[0] - 1

prob_pres = []
for j in range(1, tot_times): #Preservation of probability
    prob_pres.append(inte @ Mx.M @ (sigma_z[j] - sigma_z[j-1]))
    prob_pres.append(inte @ Mx.M @ (sigma_ff[j] - sigma_ff[j - 1]))
    prob_pres.append(inte @ Mx.M @ (sigma_lp[j] - sigma_lp[j - 1]))
    prob_pres.append(inte @ Mx.M @ (sigma_ld[j] - sigma_ld[j - 1]) + sigma_ld_b[j] - sigma_ld_b[j-1])

probability = ca.vertcat(*[*[p1,p2,p3,p4], *prob_pres])
#Zeros contributed: 4*total_times
complementarity = []
normal_cone = []
for j in range(tot_times): #Complementarity at every poitn, we need to be in the normal cone as well
    complementarity.append(inte @ Mx.M @ (df_z[j]*sigma_z[j]) + inte @ Mx.M @ (df_ff[j]*sigma_ff[j]) +
                           inte @ Mx.M @ (df_lp[j]*sigma_lp[j]) + inte @ Mx.M @ (df_ld_up[j]*sigma_ld[j])
                           + sigma_ld_b[j]*df_ld_b[j])
    normal_cone.append(ca.vertcat(-df_z[j], -df_ff[j], -df_lp[j], -df_ld_up[j], -df_ld_b[j]))

#Lower zeros contributed: 2*total_times + 4*tot_points*tot_times
#Unbounded above: 4*tot_points*tot_times + tot_times


g = ca.vertcat(*[diff_eq_z, diff_eq_ff, diff_eq_lp, diff_eq_ld, diff_eq_bc, *diff_eqs, probability, *complementarity, *normal_cone])
lbg = np.zeros(2*tot_times+5*tot_points*tot_times+4*tot_times + 5*tot_times + 5)
ubg = ca.vertcat(*[*np.zeros(5*tot_times+tot_times*tot_points + 5*tot_times + 5), [ca.inf]*(4*tot_points*tot_times + tot_times)])


sigmas = ca.vertcat(*[*sigma_z, *sigma_ff, *sigma_lp, *sigma_ld, *sigma_ld_b]) #sigma_bar
x = ca.vertcat(*[sigmas, *res_level, *state, *lam])
lbx = ca.vertcat(*[np.zeros(5*tot_points*tot_times + 6*tot_times), 4*tot_times*[-ca.inf]])



s_opts = {'ipopt': {'print_level' : 5, 'linear_solver':'ma57'}} #'hessian_approximation':'limited-memory',
init = np.ones(x.size()[0])/Mx.x[-1]
init[-9*tot_times:-4*tot_times] = 1

prob = {'x': x, 'f': f, 'g': g}

solver = ca.nlpsol('solver', 'ipopt', prob, s_opts)

sol = solver(lbx = lbx, lbg = lbg, ubg = ubg, x0 = init)
print(sol['x'])

print(sol['x'][-(9*tot_times):-4*tot_times].size())
print(sol['x'][-9*tot_times:-4*tot_times])
print(sol['x'][-(9+tot_points)*tot_times:-9*tot_times])
print(sol['x'][0:tot_points*4+1])
print(sol['x'].size())
