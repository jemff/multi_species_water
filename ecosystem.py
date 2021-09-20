import casadi as ca
import numpy as np
from infrastructure import *

Mx = simple_method(40, 40)

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
print(epsi)
rz = 50*np.exp(-(Mx.x-z_mld)/sigma**2)
Bmax = 30
V0 = 40


tot_points = Mx.x.size
inte = np.ones(tot_points).reshape(1,tot_points)

tot_cont_p = tot_points
tot_times = 1
Vi = V0
beta_i = Vi/masses

upright_wc = np.exp(-k*Mx.x).reshape((-1,1))
beta_ld_b = Vi/masses[-1]
beta_z = beta_i[0] * upright_wc
beta_ff = beta_i[1] * upright_wc
beta_lp = beta_i[2] * upright_wc
beta_ld = beta_i[3] * upright_wc
c_z = Cmax[0]
c_ff = Cmax[1]
c_lp = Cmax[2]
c_ld = Cmax[3]
bg_M = 0.1

lam = ca.MX.sym('lam', 4)

sigma_z = ca.MX.sym('sigma_z', tot_cont_p)
sigma_ff = ca.MX.sym('sigma_ff', tot_cont_p)
sigma_lp = ca.MX.sym('sigma_lp', tot_cont_p)
sigma_ld = ca.MX.sym('sigma_ld', tot_cont_p)
sigma_ld_b = ca.MX.sym('sigma_ld_b', 1)
state = ca.MX.sym('state', 5)
res_level = ca.MX.sym('res_level', tot_cont_p) #Resource dynamics
#D_2 = (Mx.D @ Mx.D)
#D_2[0] = np.copy(Mx.D[0])
#D_2[-1] = np.copy(Mx.D[-1])
#almost_ID = np.identity(tot_points)
#almost_ID[0,0] = 0
#almost_ID[-1,-1] = 0

z_pp_enc = inte @ (Mx.M @ (beta_z*sigma_z*res_level))
z_satiation = 1/(z_pp_enc+c_z) #c_z*ff_z_enc

ff_z_enc = inte @ (Mx.M @ (beta_ff*sigma_z * sigma_ff))
ff_satiation = 1/(state[0]*ff_z_enc+c_ff) # c_ff*ff_z_enc

lp_ff_enc =  inte @ (Mx.M @ (beta_lp*sigma_ff * sigma_lp))
lp_satiation = 1/(state[1]*lp_ff_enc+c_lp)  #c_lp*lp_ff_enc

ld_ff_enc =  inte @ (Mx.M @ (beta_ld*sigma_ff * sigma_lp))
ld_bc_enc = beta_ld_b*0.5*state[-1]*sigma_ld_b

ld_satiation = 1/(ld_ff_enc+ld_bc_enc+c_ld)

lp_loss = 0
ld_loss = 0


res_dyn_one = r*(rz - res_level) - state[0]*res_level*sigma_z*z_satiation

#pde_form = D_2 @ res_level + almost_ID @ res_dyn_one
res_dyn_l2 = inte @ (Mx.M @ (res_dyn_one * res_dyn_one)) / (Mx.x[-2] ** 2)

bent_dyn = r_b*(state[-1]-Bmax)-ld_bc_enc*ld_satiation
#Bentic dynamics

z_dyn =  state[0]*(epsi[0]*(c_z*z_pp_enc*z_satiation - f_c) - state[1]*c_ff*ff_z_enc*ff_satiation - bg_M)
ff_dyn = state[1]*(epsi[1]*(c_ff*state[0]*ff_z_enc*ff_satiation - f_c) - state[2] * c_ff*lp_ff_enc * lp_satiation
                   - state[3] * c_ld * ld_ff_enc * ld_satiation - bg_M)
lp_dyn = state[2]*(epsi[2]*(c_lp*state[1]*lp_ff_enc*lp_satiation - f_c) - lp_loss - bg_M)
ld_dyn = state[3]*(epsi[3]*(c_ld*state[1]*ld_ff_enc*ld_satiation + ld_bc_enc*ld_satiation - f_c)- ld_loss - bg_M)


df_z = epsi[0]*c_z**2*beta_z*res_level*z_satiation**2 - state[1]*c_ff*sigma_ff*beta_ff*ff_satiation + lam[0]
df_ff = epsi[1]*c_ff**2*state[0]*beta_ff*sigma_z*ff_satiation**2 - c_lp*state[2]*sigma_lp*beta_lp*lp_satiation - c_ld*state[3]*sigma_ld*beta_ld*ld_satiation + lam[1]
df_lp = epsi[2]*c_lp**2*state[1]*sigma_lp*beta_lp*lp_satiation**2 + lam[2] #-loss derivative
df_ld_up = epsi[3]*c_ld**2*state[1]*sigma_ld*beta_ld*ld_satiation**2 + lam[3]
df_ld_b = epsi[3]*c_ld**2*beta_ld_b*0.5*state[-1]*ld_satiation**2 + lam[3]

g0 = ca.vertcat(bent_dyn, z_dyn, ff_dyn, lp_dyn, ld_dyn)
g1 = inte @ Mx.M @ (df_z*sigma_ff) + inte @ Mx.M @ (df_ff*sigma_ff) + inte @ Mx.M @ (df_lp*sigma_lp) + inte @ Mx.M @ (df_ld_up*sigma_ld) + sigma_ld_b*df_ld_b  #
g2 = inte @ Mx.M @ sigma_z - 1
g3 = inte @ Mx.M @ sigma_ff - 1
g4 = inte @ Mx.M @ sigma_lp - 1
g5 = inte @ Mx.M @ sigma_ld + sigma_ld_b - 1
g6 = ca.vertcat(-df_z, -df_ff, -df_lp, -df_ld_up, -df_ld_b)

g = ca.vertcat(*[g0, g1, g2, g3, g4, g5, g6])

f = res_dyn_l2 #ca.sin(res_dyn)**2 #ca.cosh(res_dyn)-1 #ca.cosh(res_dyn)-1 # ca.exp(res_dyn)-1 #ca.sqrt(res_dyn) #ca.cosh(res_dyn) - 1 #- 1 cosh most stable, then exp, then identity

sigmas = ca.vertcat(sigma_z, sigma_ff, sigma_lp, sigma_ld, sigma_ld_b) #sigma_bar
x = ca.vertcat(*[sigmas, res_level, state, lam])

lbg = np.zeros(7 + 4*tot_points+4)
ubg = ca.vertcat(*[np.zeros(10), [ca.inf]*4*tot_points, [ca.inf]])
s_opts = {'ipopt': {'print_level' : 5, 'linear_solver':'ma57'}} #'hessian_approximation':'limited-memory',
init = np.ones(x.size()[0])*100#/100

prob = {'x': x, 'f': f, 'g': g}
lbx = ca.vertcat(*[np.zeros(x.size()[0] - 4), 4*[-ca.inf]])
solver = ca.nlpsol('solver', 'ipopt', prob, s_opts)

sol = solver(lbx = lbx, lbg = lbg, ubg = ubg, x0 = init)

print(sol['x'][-10], sol['x'][-9:-4], sol['x'][-4:])
