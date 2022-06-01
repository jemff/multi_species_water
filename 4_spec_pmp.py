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

from four_spec_sim import *
import pickle as pkl
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
fitness = 'gm'
#print(fin_diff_mat(5))

tot_points = 20
Mx = spectral_method(100, tot_points)
import numpy as np
import matplotlib.pyplot as plt
fidelity = 30 #must be even
inte = np.ones(tot_points).reshape(1, tot_points)
x_vals = np.linspace(0, 1, fidelity + 1)[:-1]
D = differentation_matrix(fidelity)
D_x = fin_diff_mat(tot_points)
#print(D @ np.cos(x_vals))
plt.plot(x_vals, D @ np.cos(np.pi*2*x_vals))
plt.show()
h = 20 / 365
a = 0.4
m0 = 10 ** (-3)
gamma = 0.6
k = 0.06
masses = np.array([0.05, 11])
f_c = 0  # 0.15 / 365
r = 1 / 365
r_b = 1 / 365
eps0 = 0.05
#comp = 0
Cmax = h * masses ** (-0.25)
metabolism = 0.2 * h *masses**(-0.25)
metabolism[-1] = 0.5*metabolism[-1]

print(metabolism)
epsi = eps0 * ((1 - a) * np.log(masses / m0) - np.log(eps0))
# epsi = 0.7*epsi/epsi
rz = 1 / (8.8) * np.exp(-((Mx.x)) ** 2 / (10 ** 2)) + 10 ** (-5)
upright_wc = np.exp(-k * Mx.x)  # .reshape((-1, 1))
c_z = Cmax[0]
c_ff = Cmax[1]
bg_M = 0.1 / 365
beta_0 = 10 ** (-5)
A = np.cos(2*np.pi*np.linspace(0,1,5*fidelity))*1/np.sqrt(2)
A[A < -1 / 2] = -1 / 2
A[A > 1 / 2] = 1 / 2
A = A+1/2
smoothed_A = savgol_filter(A, window_length = 5, polyorder = 3)
light_levels = smoothed_A[0:-1:5]


def outputs(R_max = 5, warmstart_info = None):
    Rmax = R_max

    p_z_l = []
    p_ff_l = []
    sigma_z_l = []
    sigma_ff_l = []
    dz_sigma_z_l = []
    dz_sigma_ff_l = []
    dz_vel_z_l = []
    dz_vel_ff_l = []

    vel_z_l = []
    vel_ff_l = []
    dHsig_z = []
    dHsig_ff = []
    dHp_z = []
    dHp_ff = []
    dHv_z = []
    dHv_ff = []

    state_l = []
    prob_l = []
    beta_z = []
    beta_ff = []
    complementarity = []
    normal_cone = []

    df_z = []
    df_ff = []
    ff_z_enc = []
    ff_satiation = []


    dyn_0 = []
    dyn_1 = []
    dyn_5 = []

    s0_vec = []
    s1_vec = []
    s5_vec = []



    for j in range(fidelity):
        Vi = light_levels[j]
        beta_i = 330 / 365 * Vi * masses ** (-0.25) * gamma
        beta_z.append( 330 / 365 * 0.6 * upright_wc ** 0 * masses[0] ** (-0.25))
        beta_ff.append(2 * beta_i[1] * (upright_wc / (1 + upright_wc) + beta_0))

        p_z_l.append(ca.MX.sym('p_z_'+str(j), tot_points))
        p_ff_l.append(ca.MX.sym('p_ff_'+str(j), tot_points))
        sigma_z_l.append(ca.MX.sym('sigma_z_l_'+str(j), tot_points))
        sigma_ff_l.append(ca.MX.sym('sigma_ff_l_'+str(j), tot_points))
        vel_z_l.append(ca.MX.sym('vel_z_l_'+str(j), tot_points))
        vel_ff_l.append(ca.MX.sym('vel_ff_l_'+str(j), tot_points))

        state_l.append(ca.MX.sym('state_'+str(j), 3))

        ff_z_enc.append(beta_ff[j] * sigma_z_l[j] * sigma_ff_l[j])
        ff_satiation.append(1 / (state_l[j][0] * ff_z_enc[j] + c_ff))

        dz_sigma_z_l.append(Mx.D @ sigma_z_l[j])
        dz_sigma_ff_l.append(Mx.D @ sigma_ff_l[j])
        dz_vel_z_l.append(Mx.D @ vel_z_l[j])
        dz_vel_ff_l.append(Mx.D @ vel_ff_l[j])

        dHsig_z.append(epsi[0] * state_l[j][-1] * c_z**2*beta_z[j] * rz/(c_z+state_l[j][-1] * sigma_z_l[j] * beta_z[j] * rz)**2 - state_l[j][1] * (
                    c_ff * sigma_ff_l[j] * beta_ff[j] * ff_satiation[j]) - vel_z_l[j]**2 + p_z_l[j]*dz_vel_z_l[j])
        dHsig_ff.append((epsi[1] * state_l[j][0] * c_ff ** 2 * beta_ff[j] * sigma_z_l[j] * ff_satiation[j] ** 2) - vel_ff_l[j]**2 + p_ff_l[j]*dz_vel_ff_l[j])
        dHp_z.append(dz_vel_z_l[j]*sigma_z_l[j] + dz_sigma_z_l[j]*vel_z_l[j])
        dHp_ff.append(dz_vel_ff_l[j]*sigma_ff_l[j] + dz_sigma_ff_l[j]*vel_ff_l[j])
        dHv_z.append(vel_z_l[j]*sigma_z_l[j]+dz_sigma_z_l[j]*p_z_l[j])
        dHv_ff.append(vel_ff_l[j]*sigma_ff_l[j]+dz_sigma_ff_l[j]*p_ff_l[j])


        prob_l.append(inte @ Mx.M @ sigma_z_l[j] - 1)
        prob_l.append(inte @ Mx.M @ sigma_ff_l[j] - 1)

        complementarity.append(inte @ ( (dHv_z[j] * sigma_z_l[j])))
        complementarity.append(inte @ ( (dHv_ff[j] * sigma_ff_l[j])))
        normal_cone.append(-dHv_z[j])
        normal_cone.append(-dHv_ff[j])


        s0_vec.append(state_l[j][0])
        s1_vec.append(state_l[j][1])
        s5_vec.append(state_l[j][-1])

        dyn_0.append(state_l[j][0] * (inte @ Mx.M @ (epsi[0] * state_l[j][-1] * sigma_z_l[j]*c_z*beta_z[j] * rz/(c_z+state_l[j][-1] * sigma_z_l[j] * beta_z[j] * rz))
                    - state_l[j][1] * (inte @ (Mx.M @ (c_ff * ff_z_enc[j] * ff_satiation[j]))) - metabolism[0]))
        dyn_1.append(state_l[j][1] * (epsi[1] * (c_ff * state_l[j][0] * inte @ (Mx.M @ (ff_z_enc[j] * ff_satiation[j])) - f_c) - bg_M - metabolism[1]))
        dyn_5.append(r * (Rmax - state_l[j][-1]) - state_l[j][-1]* state_l[j][0] * inte @ Mx.M @ (sigma_z_l[j]*c_z*beta_z[j] * rz/(c_z+state_l[j][-1] * sigma_z_l[j] * beta_z[j] * rz)))


    v_c = lambda x: ca.vertcat(*x)
    i1 = np.ones(fidelity).reshape(1,fidelity)
    #Total time dynamics
    t_p_z = ca.hcat(p_z_l).T
    t_p_ff = ca.hcat(p_ff_l).T
    t_sigma_z = ca.hcat(sigma_z_l).T
    t_sigma_ff = ca.hcat(sigma_ff_l).T
    #Hamiltonian total dynamics
    t_dHsigma_z = ca.hcat(dHsig_z).T
    t_dHp_z = ca.hcat(dHp_z).T

    t_dHsigma_ff = ca.hcat(dHsig_ff).T
    t_dHp_ff = ca.hcat(dHp_ff).T

    print(t_p_z.size(), D.shape)
    canonical_p_z = (D @ t_p_z + t_dHsigma_z)**2
    canonical_sigma_z = (D @ t_sigma_z - t_dHp_z)**2
    canonical_p_ff = (D @ t_p_ff + t_dHsigma_ff)**2
    canonical_sigma_ff = (D @ t_sigma_ff - t_dHp_ff)**2

    bc_mat = np.identity(tot_points)
    bc_mat[-1, -1] = 0
    bc_mat[0,0] = 0
    ones = np.ones(tot_points)

    print(canonical_p_z.size(), (canonical_p_z @ ones).size())
    can_eqs = i1 @ (canonical_p_z @ ones) + i1 @ (canonical_sigma_z @ ones) + i1 @ (canonical_p_ff @ ones) + i1 @ (canonical_sigma_ff @ ones)
    trans_z = (D @ t_sigma_z + (ca.hcat(dz_vel_z_l).T*ca.hcat(sigma_z_l).T + ca.hcat(dz_sigma_z_l).T*ca.hcat(vel_z_l).T) @ bc_mat)**2
    trans_ff = (D @ t_sigma_ff + (ca.hcat(dz_vel_ff_l).T*ca.hcat(sigma_ff_l).T + ca.hcat(dz_sigma_ff_l).T*ca.hcat(vel_ff_l).T) @ bc_mat)**2
    trans_eqs = i1 @ ( trans_z @ ones) + i1 @ (trans_ff @ ones)

    pop_dyn_eqs = i1 @ (D @ v_c(s0_vec) - v_c(dyn_0))**2 + i1 @ (D @ v_c(s1_vec) - v_c(dyn_1))**2 + i1 @ (D @ v_c(s5_vec) - v_c(dyn_5))**2
    f =  pop_dyn_eqs + can_eqs + trans_eqs

    x = ca.vertcat(*[*vel_z_l, *vel_ff_l, *p_z_l, *p_ff_l, *sigma_z_l, *sigma_ff_l, *state_l])

    g = ca.vertcat(*[*normal_cone, *prob_l, *complementarity])
    lbg = np.zeros(g.size()[0])
    ubg = ca.vertcat(*[[ca.inf] * (2*fidelity* tot_points), *np.zeros(2*fidelity), [0.00]*2*fidelity])
    lbx = ca.vertcat(fidelity*4*tot_points*[-ca.inf], np.zeros(tot_points*fidelity*2), np.ones(fidelity*3)*10**(-7))
    ubx = ca.vertcat(*[[ca.inf]*(x.size()[0]-3*fidelity), np.repeat(Rmax, 3*fidelity)])
    prob = {'x': x, 'f': f, 'g': g}

    #s_opts = {'ipopt':{'print_level': 5, 'linear_solver': 'ma57', 'fixed_variable_treatment': "make_constraint"}}  #

    if warmstart_info is None:
        x_init = np.ones(x.size()[0])/Mx.x[-1]#np.concatenate([np.zeros(4*fidelity), np.ones(4*fidelity*tot_points)/Mx.x[-1], np.ones(6*fidelity)])
        s_opts = {'ipopt': {'print_level': 5, 'linear_solver': 'ma57', 'max_iter':5000}}  #
    else:
        s_opts = {'ipopt': {'print_level': 3, 'linear_solver': 'ma57',
                               'acceptable_iter': 15, #'hessian_approximation': 'limited-memory',
                               'warm_start_init_point': 'yes', 'warm_start_bound_push': 1e-9,
                               'warm_start_bound_frac': 1e-9, 'warm_start_slack_bound_frac': 1e-9,
                              'warm_start_slack_bound_push': 1e-9, 'warm_start_mult_bound_push': 1e-9}}



    solver = ca.nlpsol('solver', 'ipopt', prob, s_opts)
    #sol = solver(lbx=lbx, lbg=lbg, ubg=ubg, x0=init, p = Rmax)
    if warmstart_info is None:
        sol = solver(lbx=lbx, ubx = ubx, lbg=lbg, ubg=ubg, x0 = x_init) #

    else:
       sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, x0=warmstart_info['x0'], lam_g0=warmstart_info['lam_g0'],
                     lam_x0=warmstart_info['lam_x0'])

    ret_dict = {'x0': np.array(sol['x']).flatten(), 'lam_g0': np.array(sol['lam_g']).flatten(),
                'lam_x0': np.array(sol['lam_x']).flatten()}
    return ret_dict #ret_dict['x0']




def vary_res(min = 5, max = 50, fineness = 45):
    res_vals = np.linspace(min, max, fineness)

    val = []
    val.append(outputs(R_max = min))
    for k in range(1, fineness):
        val.append(outputs(R_max = res_vals[k], warmstart_info=val[k-1]))
        if np.min(val[k]['x0'][-6*fidelity:]) < 10**(-6):
            print("I got here")
            val[k] = outputs(R_max = res_vals[k])

    out = {'fidelity': fidelity, 'pts': tot_points, 'fineness':fineness, 'min': min, 'max': max, 'bmax':B_max, 'data':val}
    with open('data/' + 'rv_'+str(fidelity)+'_'+str(tot_points) +'_'+ str(fineness)+'_b_'+str(B_max)+'.pkl', 'wb') as f:
        pkl.dump(out, f, pkl.HIGHEST_PROTOCOL)


vary_res(min = 5, max = 5, fineness = 1)
