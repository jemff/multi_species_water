import casadi as ca
import numpy as np
from infrastructure import *

Mx = simple_method(40, 5)
light_levels = [0, 80]
time_lengths = [12, 12]


tot_times = len(time_lengths)

tot_points = Mx.x.size
tot_cont_p = tot_points

inte = np.ones(tot_points).reshape(1,tot_points)

h = 20
a = 0.4
m0 = 10**(-3)
gamma = 0.6
k = 0.03
masses = np.array([0.1, 11, 5000, 5000])
f_c = 0.15
r = 1
r_b = 1
eps0 = 0.01
R_max = 20 #Varied between 5 and 100
z_mld = 30
sigma = 10
Cmax = h * masses **(-0.25)
epsi = eps0*((1-a)*np.log(masses/m0) - np.log(eps0))
rz = 50*np.exp(-(Mx.x-z_mld)/sigma**2)
Bmax = 10
upright_wc = np.exp(-k * Mx.x)#.reshape((-1, 1))
c_z = Cmax[0]
c_ff = Cmax[1]
c_lp = Cmax[2]
c_ld = Cmax[3]
bg_M = 0.1

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
    beta_ld_b.append(Vi/masses[-1])
    beta_z.append(beta_i[0] * upright_wc)
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

    lam += [lam_j]
    sigma_z += [sigma_z_j]
    sigma_ff += [sigma_ff_j]
    sigma_lp += [sigma_lp_j]
    sigma_ld += [sigma_ld_j]
    sigma_ld_b += [sigma_ld_b_j]
    state += [state_j]
    res_level += [res_level_j]
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
#print(inte.shape, aa.size(), "AA size", Mx.M @ aa.T)
for j in range(tot_times):
    z_pp_enc.append(inte @ (Mx.M @ ((beta_z[j])*sigma_z[j]*res_level[j])))
    z_satiation.append(1/(z_pp_enc[j]+c_z)) #c_z*ff_z_enc

    ff_z_enc.append(inte @ (Mx.M @ ((beta_ff[j])*sigma_z[j] * sigma_ff[j])))
    ff_satiation.append(1/(state[j][0]*ff_z_enc[j]+c_ff)) # c_ff*ff_z_enc

    lp_ff_enc.append(inte @ (Mx.M @ ((beta_lp[j]) * sigma_ff[j] * sigma_lp[j])))
    lp_satiation.append(1/(state[j][1]*lp_ff_enc[j]+c_lp))  #c_lp*lp_ff_enc

    ld_ff_enc.append(inte @ (Mx.M @ ((beta_ld[j]) * sigma_ff[j] * sigma_lp[j])))
    ld_bc_enc.append(beta_ld_b[j]*0.5*state[j][-1]*sigma_ld_b[j])

    ld_satiation.append(1/(ld_ff_enc[j]+ld_bc_enc[j]+c_ld))

#Bentic dynamics

z_dyn = []
ff_dyn = []
lp_dyn = []
ld_dyn = []
res_dyn_one = []
bent_dyn = []
r * (rz - res_level[j]) - state[j][0] * beta_z[j] * res_level[j] * sigma_z[j] * z_satiation[j]
#print(r * (rz - res_level[j]))
for j in range(tot_times):
    res_dyn_one.append(r * (rz - res_level[j]) - state[j][0] * beta_z[j] * res_level[j] * sigma_z[j] * z_satiation[j])
    bent_dyn.append(r_b * (state[j][-1] - Bmax) - ld_bc_enc[j] * ld_satiation[j])
    z_dyn.append(state[j][0]*(epsi[0]*(c_z*z_pp_enc[j]*z_satiation[j] - f_c) - state[j][1]*ff_z_enc[j]*ff_satiation[j] - bg_M))
    ff_dyn.append(state[j][1]*(epsi[1]*(c_ff*state[j][0]*ff_z_enc[j]*ff_satiation[j] - f_c)- state[j][2] * lp_ff_enc[j] * lp_satiation[j] - state[j][3] * ld_ff_enc[j] * ld_satiation[j] - bg_M))
    lp_dyn.append(state[j][2]*(epsi[2]*(c_lp*state[j][1]*lp_ff_enc[j]*lp_satiation[j] - f_c) - lp_loss - bg_M))
    ld_dyn.append(state[j][3]*(epsi[3]*(c_ld*state[j][1]*ld_ff_enc[j]*ld_satiation[j] + ld_bc_enc[j]*ld_satiation[j] - f_c)- ld_loss - bg_M))

df_z = []
df_ff = []
df_lp = [] #
df_ld_up = []
df_ld_b = []
for j in range(tot_times):
    df_z.append(epsi[0]*beta_z[j]*res_level[j]*z_satiation[j]**2 - state[1]*sigma_ff[j]*beta_ff[j]*ff_satiation[j] + lam[j][0])
    df_ff.append(epsi[1]*state[j][0]*beta_ff[j]*sigma_z[j]*ff_satiation[j]**2
                 - state[j][2]*sigma_lp[j]*beta_lp[j]*lp_satiation[j]
                 - state[j][3]*sigma_ld[j]*beta_ld[j]*ld_satiation[j] + lam[j][1])
    df_lp.append(epsi[2]*state[j][1]*sigma_lp[j]*beta_lp[j]*lp_satiation[j]**2 + lam[j][2]) #-loss derivative
    df_ld_up.append(epsi[3]*state[j][1]*sigma_ld[j]*beta_ld[j]*ld_satiation[j]**2 + lam[j][3])
    df_ld_b.append(epsi[3]*beta_ld_b[j]*0.5*state[j][-1]*ld_satiation[j]**2 + lam[j][3])

#tot_bent_dyn = 0
#tot_z_dyn = 0
#tot_ff_dyn = 0
#tot_lp_dyn = 0
#tot_ld_dyn = 0
#for j in range(tot_times): #Total dynamics, should equal zero
#    tot_bent_dyn = tot_bent_dyn + bent_dyn[j]
#    tot_z_dyn = tot_z_dyn + z_dyn[j]
#    tot_ff_dyn = tot_ff_dyn + ff_dyn[j]
#    tot_lp_dyn = tot_lp_dyn + lp_dyn[j]
#    tot_ld_dyn = tot_ld_dyn + ld_dyn[j]
#dynamics = ca.vertcat(tot_bent_dyn, tot_z_dyn, tot_ff_dyn, tot_lp_dyn, tot_ld_dyn) # Total dynamics, should equal zero
diff_eqs = []
res_level_dyn = []
f = 0
for j in range(tot_times):
    diff_eqs.append(state[j][0] - state[j-1][0] + time_lengths[j-1]*z_dyn[j-1])
    diff_eqs.append(state[j][1] - state[j-1][1] + time_lengths[j-1]*ff_dyn[j-1])
    diff_eqs.append(state[j][2] - state[j-1][2] + time_lengths[j-1]*lp_dyn[j-1])
    diff_eqs.append(state[j][3] - state[j-1][3] + time_lengths[j-1]*ld_dyn[j-1])
    diff_eqs.append(state[j][4] - state[j-1][4] + time_lengths[j-1]*z_dyn[j-1])
    res_level_dyn.append(res_level[j] - res_level[j-1] + time_lengths[j-1]*res_dyn_one[j-1])

    f = f + inte @ (Mx.M @ ((res_level[j] - res_level[j-1] + time_lengths[j-1]*res_dyn_one[j-1])**2))
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
    complementarity.append(inte @ Mx.M @ (df_z[j]*sigma_ff[j]) + inte @ Mx.M @ (df_ff[j]*sigma_ff[j]) +
                           inte @ Mx.M @ (df_lp[j]*sigma_lp[j]) + inte @ Mx.M @ (df_ld_up[j]*sigma_ld[j])
                           + sigma_ld_b[j]*df_ld_b[j])  #
    normal_cone.append(ca.vertcat(-df_z[j], -df_ff[j], -df_lp[j], -df_ld_up[j], -df_ld_b[j]))

#Lower zeros contributed: 2*total_times + 4*tot_points*tot_times
#Unbounded above: 4*tot_points*tot_times + tot_times


g = ca.vertcat(*[*diff_eqs, probability, *complementarity, *normal_cone])
lbg = np.zeros(2*tot_times+4*tot_points*tot_times+4*tot_times + 5*tot_times)
ubg = ca.vertcat(*[np.zeros(tot_times+4*tot_times + 5*tot_times), [ca.inf]*(4*tot_points*tot_times + tot_times)])


sigmas = ca.vertcat(*[*sigma_z, *sigma_ff, *sigma_lp, *sigma_ld, *sigma_ld_b]) #sigma_bar
x = ca.vertcat(*[sigmas, *res_level, *state, *lam])
lbx = ca.vertcat(*[np.zeros(4*tot_points*tot_times+tot_points*tot_times + 4*tot_times+4), 4*tot_times*[-ca.inf]])



s_opts = {'ipopt': {'print_level' : 5, 'linear_solver':'ma57'}} #'hessian_approximation':'limited-memory',
init = np.ones(x.size()[0])*100#/100

prob = {'x': x, 'f': f, 'g': g}

solver = ca.nlpsol('solver', 'ipopt', prob, s_opts)

sol = solver(lbx = lbx, lbg = lbg, ubg = ubg, x0 = init)

