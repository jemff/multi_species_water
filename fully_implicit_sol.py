def differentation_matrix(N):
    D = np.zeros((N,N))
    D_ana = lambda t, v : \
    1/2*(-1)**(t-v)* \
             1/np.sin((t-v)/N*np.pi)* \
             np.cos((t-v)/N*np.pi)

    for i in range(N):
        for k in range(N):
            if k != i:
                D[i,k] = D_ana(i,k)
                #print(i,k)
    return D

def fin_diff_mat(N):
    D = np.zeros((N,N))
    D[0,0] = -3
    D[-1,-1] = 3
    D[0,2] = -1
    D[-1,-3] = 1
    D = D - np.diag(np.ones(N-1), -1)
    D = D + np.diag(np.ones(N-1), 1)
    D[0,1] += 3
    D[-1,-2] -= 3

    return D


def fin_diff_mat_periodic(N):
    D = np.zeros((N, N))
    D = D - np.diag(np.ones(N - 1), -1)
    D = D + np.diag(np.ones(N - 1), 1)
    D[-1,0] = 1
    D[0,-1]=-1
    h = 1/(N)*(2*np.pi)
    return 1/(2*h)*D

from four_spec_sim import *
import pickle as pkl
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
fitness = 'gm'
#print(fin_diff_mat(5))

tot_points = 5
Mx = spectral_method(100, tot_points)
import numpy as np
import matplotlib.pyplot as plt
fidelity = 20 #must be even
inte = np.ones(tot_points).reshape(1, tot_points)
x_vals = np.linspace(0, 1, fidelity + 1)[:-1]
D = differentation_matrix(fidelity)
#print(D @ np.cos(x_vals))
print(np.max(D @ np.cos(np.pi*2*x_vals)))
plt.plot(x_vals, D @ np.cos(np.pi*2*x_vals))
plt.show()
h = 20 / 365
a = 0.4
m0 = 10 ** (-3)
gamma = 0.6
k = 0.06
masses = np.array([0.05, 11, 5000, 4000])
f_c = 0  # 0.15 / 365
r = 1 / 365
r_b = 1 / 365
eps0 = 0.05
#comp = 0
Cmax = h * masses ** (0.75)
metabolism = 0.2 * h *masses**(0.75)
metabolism[-1] = 0.5*metabolism[-1]

print(metabolism)
epsi = eps0 * ((1 - a) * np.log(masses / m0) - np.log(eps0))
# epsi = 0.7*epsi/epsi
rz = 1 / (8.8) * np.exp(-((Mx.x)) ** 2 / (5 ** 2)) + 10 ** (-5)
T_dist = np.exp(-((Mx.x - Mx.x[-1])) ** 2 / (10 ** 2))
mass_T = inte @ Mx.M @ T_dist
bent_dist = T_dist / mass_T
upright_wc = np.exp(-k * Mx.x)  # .reshape((-1, 1))
c_z = Cmax[0]
c_ff = Cmax[1]
c_lp = Cmax[2]
c_ld = Cmax[3]
bg_M = 0.1 / 365
beta_0 = 10 ** (-5)
A = np.cos(2*np.pi*np.linspace(0,1,5*fidelity))*1/np.sqrt(2)
A[A < -1 / 2] = -1 / 2
A[A > 1 / 2] = 1 / 2
A = A+1/2
smoothed_A = savgol_filter(A, window_length = 5, polyorder = 3)
light_levels = smoothed_A[0:-1:5]


#light_levels = (np.sin(2*np.pi*x_vals) + 1)/2#* 1 / np.sqrt(3)
plt.plot(x_vals, light_levels)
plt.show()
#light_levels[light_levels < -1 / 2] = -1 / 2
#light_levels[light_levels > 1 / 2] = 1 / 2

#light_levels = light_levels + 1 / 2  # + 0.0001
#plt.plot(np.linspace(0, 1, fidelity), light_levels)

#plt.plot(np.linspace(0, 1, fidelity), D @ light_levels)
#plt.plot(np.linspace(0, 1, fidelity), light_levels)
#plt.show()

def outputs(R_max = 5, B_max = 1, warmstart_info = None):
    Rmax = R_max #ca.MX.sym('Rmax') # #
    Bmax = B_max

    lam_l = []
    sigma_z_l = []
    sigma_ff_l = []
    sigma_lp_l = []
    sigma_ld_l = []
    state_l = []
    prob_l = []
    beta_ld_b = []
    beta_z = []
    beta_ff = []
    beta_lp = []
    beta_ld = []
    complementarity = []
    normal_cone = []

    df_z = []
    df_ff = []
    df_lp = []
    df_ld = []
    ff_z_enc = []
    ff_satiation = []

    lp_ff_enc = []
    lp_satiation = []
    ld_ff_enc = []
    ld_bc_enc = []
    ld_satiation = []

    dyn_0 = []
    dyn_1 = []
    dyn_2 = []
    dyn_3 = []
    dyn_4 = []
    dyn_5 = []
    s0_vec = []
    s1_vec = []
    s2_vec = []
    s3_vec = []
    s4_vec = []
    s5_vec = []


    g_z = []
    m_z = []
    dg_z = []
    dm_z = []

    g_ff = []
    m_ff = []
    dg_ff = []
    dm_ff = []
    for j in range(fidelity):
        Vi = light_levels[j]
        beta_i = 330 / 365  * masses ** (0.75) * gamma
        beta_ld_b.append(0.5 * 330 / 365 * (upright_wc ** 0 )* gamma *masses[-1] ** (0.75))
        beta_z.append( 330 / 365 * 0.6 * upright_wc ** 0 * masses[0] ** (0.75))
        beta_ff.append(2 * beta_i[1] * (Vi*upright_wc / (1 + upright_wc) + beta_0))
        beta_lp.append(2 * beta_i[2] * (Vi*upright_wc / (1 + upright_wc) + beta_0))
        beta_ld.append(beta_i[3] * (Vi*upright_wc / (1 + upright_wc) + beta_0))

        lam_l.append(ca.MX.sym('lam_'+str(j), 4))
        sigma_z_l.append(ca.MX.sym('sigma_z_l[j]_'+str(j), tot_points))
        sigma_ff_l.append(ca.MX.sym('sigma_ff_l[j]_'+str(j), tot_points))
        sigma_lp_l.append(ca.MX.sym('sigma_lp_l[j]_'+str(j), tot_points))
        sigma_ld_l.append(ca.MX.sym('sigma_ld_l[j]_'+str(j), tot_points))
        state_l.append(ca.MX.sym('state_'+str(j), 6))

        ff_z_enc.append(beta_ff[j] * sigma_z_l[j] * sigma_ff_l[j])
        ff_satiation.append(1 / (state_l[j][0] * ff_z_enc[j] + c_ff))

        lp_ff_enc.append(inte @ (Mx.M @ (beta_lp[j] * sigma_ff_l[j] * sigma_lp_l[j])))
        lp_satiation.append(1 / (state_l[j][1] * lp_ff_enc[j] + c_lp))

        ld_ff_enc.append(beta_ld[j] * sigma_ld_l[j] * sigma_ff_l[j])
        ld_bc_enc.append(beta_ld_b[j] * sigma_ld_l[j] * bent_dist)

        ld_satiation.append((1 / (state_l[j][1] * ld_ff_enc[j] + state_l[j][4] * ld_bc_enc[j] + c_ld)))
        if fitness == 'gm':
            df_z.append(epsi[0] * state_l[j][5] * c_z**2*beta_z[j] * rz/(c_z+state_l[j][5] * sigma_z_l[j] * beta_z[j] * rz)**2 - state_l[j][1] * (
                        c_ff * sigma_ff_l[j] * beta_ff[j] * ff_satiation[j]) + lam_l[j][0])
            df_ff.append((epsi[1] * state_l[j][0] * c_ff ** 2 * beta_ff[j] * sigma_z_l[j] * ff_satiation[j] ** 2 \
                     - state_l[j][2] * c_lp * sigma_lp_l[j] * beta_lp[j] * lp_satiation[j] \
                     - state_l[j][3] * c_ld * sigma_ld_l[j] * beta_ld[j] *ld_satiation[j]) +
                lam_l[j][1])  # ca.log(lam[1]**2)  #lam[1]
            df_lp.append((epsi[2] * state_l[j][1] * sigma_ff_l[j] * c_lp ** 2 * beta_lp[j] * lp_satiation[j] ** 2) +
                lam_l[j][2])
            df_ld.append(c_ld ** 2 * epsi[3] * (state_l[j][1] * sigma_ff_l[j] * beta_ld[j] * (ld_satiation[j] ** 2) + state_l[j][4] * beta_ld_b[
                    j] * bent_dist * (ld_satiation[j] ** 2)) + lam_l[j][3])
        else:
            g_z.append(inte @ Mx.M @ (epsi[0] * state_l[j][5] * sigma_z_l[j]*c_z*beta_z[j] * rz/(c_z+state_l[j][5] * sigma_z_l[j] * beta_z[j] * rz)))
            m_z.append(state_l[j][1] * inte @ Mx.M @  (c_ff * sigma_z_l[j] * sigma_ff_l[j] * beta_ff[j] * ff_satiation[j]) + metabolism[0])
            dg_z.append(epsi[0] * state_l[j][5] * c_z**2*beta_z[j] * rz/(c_z+state_l[j][5] * sigma_z_l[j] * beta_z[j] * rz)**2)
            dm_z.append(state_l[j][1]*c_ff*sigma_ff_l[j] * beta_ff[j] * ff_satiation[j])

            df_z.append(dg_z[j]/g_z[j] - dm_z[j]/m_z[j] + lam_l[j][0])

            g_ff.append(inte @ (Mx.M @ (sigma_ff_l[j] * epsi[1]*state_l[j][0]*c_ff*beta_ff[j]*sigma_z_l[j]*ff_satiation[j])))
            m_ff.append(inte @ Mx.M @ (sigma_ff_l[j] * ( state_l[j][2]*c_lp*sigma_lp_l[j]*beta_lp[j]*lp_satiation[j]\
                    + state_l[j][3]*c_ld*sigma_ld_l[j]*beta_ld[j]*ld_satiation[j] )) + epsi[1]*bg_M + metabolism[1])
            dg_ff.append(epsi[1]*state_l[j][0]*c_ff**2*beta_ff[j]*sigma_z_l[j]*ff_satiation[j]**2)
            dm_ff.append(state_l[j][2]*c_lp*sigma_lp_l[j]*beta_lp[j]*lp_satiation[j] + state_l[j][3]*c_ld*sigma_ld_l[j]*beta_ld[j]*ld_satiation[j])

            df_ff.append(dg_ff[j]/g_ff[j] - dm_ff[j]/m_ff[j] + lam_l[j][1])

            df_lp.append(epsi[2] * state_l[j][1] * sigma_ff_l[j] * c_lp ** 2 * beta_lp[j] * lp_satiation[j] ** 2/(epsi[2]*bg_M + metabolism[2]) + lam_l[j][2] ) #lam[2]# ca.log(lam[2]**2)

            df_ld.append(c_ld**2 * epsi[3] * (state_l[j][1] * sigma_ff_l[j] * beta_ld[j] + state_l[j][3]*beta_ld_b[j] * sigma_ld_l[j] * bent_dist) * ld_satiation[j] ** 2 /(epsi[3]*bg_M + metabolism[3]) + lam_l[j][3]) # lam[3] #ca.log(lam[3]**2)

        prob_l.append(inte @ Mx.M @ sigma_z_l[j] - 1)
        prob_l.append(inte @ Mx.M @ sigma_ff_l[j] - 1)
        prob_l.append(inte @ Mx.M @ sigma_lp_l[j] - 1)
        prob_l.append(inte @ Mx.M @ sigma_ld_l[j] - 1)

        complementarity.append(inte @ ( (df_z[j] * sigma_z_l[j])))
        complementarity.append(inte @ ( (df_ff[j] * sigma_ff_l[j])))
        complementarity.append(inte @ ((df_lp[j] * sigma_lp_l[j])))
        complementarity.append(inte @ ((df_ld[j] * sigma_ld_l[j])))
        normal_cone.append(-df_z[j])
        normal_cone.append(-df_ff[j])
        normal_cone.append(-df_lp[j])
        normal_cone.append(-df_ld[j])

        #The dynamics are defined below

        s0_vec.append(state_l[j][0])
        s1_vec.append(state_l[j][1])
        s2_vec.append(state_l[j][2])
        s3_vec.append(state_l[j][3])
        s4_vec.append(state_l[j][4])
        s5_vec.append(state_l[j][5])

        dyn_0.append(state_l[j][0] * (inte @ Mx.M @ (epsi[0] * state_l[j][5] * sigma_z_l[j]*c_z*beta_z[j] * rz/(c_z+state_l[j][5] * sigma_z_l[j] * beta_z[j] * rz))
                    - state_l[j][1] * (inte @ (Mx.M @ (c_ff * ff_z_enc[j] * ff_satiation[j]))) - metabolism[0]))
        dyn_1.append(state_l[j][1] * (epsi[1] * (c_ff * state_l[j][0] * inte @ (Mx.M @ (ff_z_enc[j] * ff_satiation[j])) - f_c) -
                state_l[j][2] * (c_lp * lp_ff_enc[j] * lp_satiation[j]) - (inte @ (Mx.M @ (state_l[j][3] * ld_ff_enc[j] * ld_satiation[j]))) - bg_M - metabolism[1]))

        dyn_2.append(state_l[j][2] * (epsi[2] * (c_lp * state_l[j][1] * lp_ff_enc[j] * lp_satiation[j] - f_c) - bg_M -
                    metabolism[2] ))#- state_l[j][2] * inte @ (Mx.M @ (comp * beta_lp[j] * sigma_lp_l[j]))))
        dyn_3.append(state_l[j][3] * (epsi[3] * inte @ (Mx.M @ ((c_ld * state_l[j][1] * ld_ff_enc[j] * ld_satiation[j] + state_l[j][4] * ld_bc_enc[j] * c_ld * ld_satiation[j])) - f_c) - bg_M - metabolism[3]))
                                    #  - comp * state_l[j][3] * inte @ (Mx.M @ ((beta_ld_b[j] * bent_dist) * sigma_ld_l[j] ** 2))))
        dyn_4.append(r_b * (Bmax - state_l[j][4]) - state_l[j][4]*state_l[j][3] * inte @ ( Mx.M @ (c_ld * ld_bc_enc[j] * ld_satiation[j])))
        dyn_5.append(r * (Rmax - state_l[j][5]) - state_l[j][5]* state_l[j][0] * inte @ Mx.M @ (sigma_z_l[j]*c_z*beta_z[j] * rz/(c_z+state_l[j][5] * sigma_z_l[j] * beta_z[j] * rz)))


    v_c = lambda x: ca.vertcat(*x)
    i1 = np.ones(fidelity).reshape(1,fidelity)

    f = i1 @ (D @ v_c(s0_vec) - v_c(dyn_0))**2 + i1 @ (D @ v_c(s1_vec) - v_c(dyn_1))**2 + i1 @ (D @ v_c(s2_vec) - v_c(dyn_2))**2 + i1 @ (D @ v_c(s3_vec) - v_c(dyn_3))**2 + i1 @ (D @ v_c(s4_vec) - v_c(dyn_4))**2 + i1 @ (D @ v_c(s5_vec) - v_c(dyn_5))**2

    #ca.log(i1 @ ((D @ v_c(s0_vec) - v_c(dyn_0))**2) + 1) + ca.log(i1 @ (D @ v_c(s1_vec) - v_c(dyn_1))**2 + 1) + ca.log(i1 @ (D @ v_c(s2_vec) - v_c(dyn_2))**2 + 1) \
        #+ ca.log(i1 @ (D @ v_c(s3_vec) - v_c(dyn_3))**2) \
        #+ ca.log( i1 @ (D @ v_c(s4_vec) - v_c(dyn_4))**2 + 1) + ca.log(i1 @ (D @ v_c(s5_vec) - v_c(dyn_5))**2 + 1)
    x = ca.vertcat(*[*lam_l, *sigma_z_l, *sigma_ff_l, *sigma_lp_l, *sigma_ld_l, *state_l])

    g = ca.vertcat(*[*normal_cone, *prob_l, *complementarity])
    lbg = np.zeros(g.size()[0])
    ubg = ca.vertcat(*[[ca.inf] * (4*fidelity* tot_points), *np.zeros(4*fidelity), [0.00]*4*fidelity])
    lbx = ca.vertcat(fidelity*4*[-ca.inf], np.zeros(x.size()[0]-fidelity*4-fidelity*6), np.ones(fidelity*6)*10**(-7))
    ubx = ca.vertcat(*[[ca.inf]*(x.size()[0]-6*fidelity), np.array(fidelity*[Rmax, Rmax, Rmax, max(Rmax,Bmax), Bmax, Rmax])])
    prob = {'x': x, 'f': f, 'g': g}

    #s_opts = {'ipopt':{'print_level': 5, 'linear_solver': 'ma57', 'fixed_variable_treatment': "make_constraint"}}  #

    if warmstart_info is None:
        x_init = np.ones(x.size()[0])/Mx.x[-1]#np.concatenate([np.zeros(4*fidelity), np.ones(4*fidelity*tot_points)/Mx.x[-1], np.ones(6*fidelity)])
        s_opts = {'ipopt': {'print_level': 5, 'linear_solver': 'ma57', 'max_iter':5000}}#, 'ma57_automatic_scaling':'yes'}}# 'hessian_approximation': 'limited-memory'}}  #
    else:
        s_opts = {'ipopt': {'print_level': 3, 'linear_solver': 'ma57',
                               'acceptable_iter': 15, #'hessian_approximation': 'limited-memory',
                               'warm_start_init_point': 'yes', 'warm_start_bound_push': 1e-9,
                               'warm_start_bound_frac': 1e-9, 'warm_start_slack_bound_frac': 1e-9,
                              'warm_start_slack_bound_push': 1e-9, 'warm_start_mult_bound_push': 1e-9}}



    solver = ca.nlpsol('solver', 'ipopt', prob, s_opts)
    #sol = solver(lbx=lbx, lbg=lbg, ubg=ubg, x0=init, p = Rmax)
    if warmstart_info is None:
        sol = solver(lbx=lbx, ubx = ubx, lbg=lbg, ubg=ubg) #

    else:
       sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, x0=warmstart_info['x0'], lam_g0=warmstart_info['lam_g0'],
                     lam_x0=warmstart_info['lam_x0'])

    ret_dict = {'x0': np.array(sol['x']).flatten(), 'lam_g0': np.array(sol['lam_g']).flatten(),
                'lam_x0': np.array(sol['lam_x']).flatten()}
    return ret_dict #ret_dict['x0']




def vary_res(min = 5, max = 50, fineness = 45, B_max = 5, warmstart_info = None):
    res_vals = np.linspace(min, max, fineness)

    #print(res_vals)
    #print(F(res_vals))
    val = []
    if warmstart_info is None:
        val.append(outputs(R_max = min, B_max = B_max))
    else:
        val.append(outputs(R_max = min, B_max = B_max, warmstart_info=warmstart_info))

    for k in range(1, fineness):
        val.append(outputs(R_max = res_vals[k], B_max = B_max, warmstart_info=val[k-1]))
        if np.min(val[k]['x0'][-6*fidelity:]) < 10**(-6):
            print("I got here")
            val[k] = outputs(R_max = res_vals[k], B_max = B_max)

    out = {'fidelity': fidelity, 'pts': tot_points, 'fineness':fineness, 'min': min, 'max': max, 'bmax':B_max, 'data':val}
    with open('data/' + '75_rv_'+str(fidelity)+'_'+str(tot_points) +'_'+ str(fineness)+'_b_'+str(B_max)+'.pkl', 'wb') as f:
        pkl.dump(out, f, pkl.HIGHEST_PROTOCOL)
    return val[0]
def vary_bmax(min = 1, max = 5, fineness = 10):
    val = []
    res_vals = np.linspace(min, max, fineness)[::-1]

    val.append(outputs(B_max = max))
    print(val[0]['x0'][-6*fidelity:-6*(fidelity-1)])
    for k in range(1, fineness):
        val.append(outputs(B_max = res_vals[k], warmstart_info=val[k-1]))

    if np.min(val[k]['x0'][-6 * fidelity:]) < 10 ** (-6):
        print("I got here")
        val[k] = outputs(R_max=res_vals[k], warmstart_info=val[k-2])

    val = val[::-1]
    with open('data/' + 'varying_b_'+str(tot_points) +'_fineness_'+ str(fineness)+'.pkl', 'wb') as f:
        pkl.dump(val, f, pkl.HIGHEST_PROTOCOL)

#vary_res(min = 5, max = 10, fineness = 5, B_max = 2)
b_n = 20
b_var = np.linspace(0.001, 3, b_n)
warmstart_inf = vary_res(B_max=b_var[0], fineness=45)
for j in range(1, b_n):
    warmstart_inf = vary_res(B_max=b_var[j], warmstart_info=warmstart_inf, fineness=45)