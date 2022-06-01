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


def fin_diff_mat_periodic(N):
    D = np.zeros((N, N))
    D = D - np.diag(np.ones(N - 1), -1)
    D = D + np.diag(np.ones(N - 1), 1)
    D[-1,0] = 1
    D[0,-1]=-1
    h = 1/(N)*2*np.pi
    return 1/(2*h)*D

from four_spec_sim import *
import pickle as pkl
from scipy.signal import savgol_filter
import numpy as np

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
Cmax = h * masses ** (0.75)
metabolism = 0.2 * h *masses**(0.75)
print(metabolism)
epsi = eps0 * ((1 - a) * np.log(masses / m0) - np.log(eps0))


def output(tot_points = 20, fidelity = 20, R_max = 5, warmstart_info = None):
    Mx = simple_method(100, tot_points)
    #must be even
    inte = np.ones(tot_points).reshape(1, tot_points)
    D = fin_diff_mat_periodic(fidelity)
    rz = 1 / (8.8) * np.exp(-((Mx.x)) ** 2 / (10 ** 2)) #+ 10 ** (-5)
    upright_wc = np.exp(-k * Mx.x)  # .reshape((-1, 1))
    c_z = Cmax[0]
    c_ff = Cmax[1]
    bg_M = 0.1 / 365
    beta_0 = 10 ** (-5)
    A = np.cos(2*np.pi*np.linspace(0,1,5*fidelity+5)[0:-5])*1/np.sqrt(2)
    A[A < -1 / 2] = -1 / 2
    A[A > 1 / 2] = 1 / 2
    A = A+1/2
    smoothed_A = savgol_filter(A, window_length = 5, polyorder = 3)
    light_levels = smoothed_A[0::5]
    gamma0 = 0.02
    gamma1 = 0.01
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
    dJsig_z = []
    dJsig_ff = []
    dJv_z = []
    dJv_ff = []

    state_l = []
    prob_l = []
    beta_z = []
    beta_ff = []
    complementarity = []

    ff_z_enc = []
    ff_satiation = []


    dyn_0 = []
    dyn_1 = []
    dyn_5 = []

    s0_vec = []
    s1_vec = []
    s5_vec = []
    compsum = 0


    for j in range(fidelity):
        Vi = light_levels[j]
        beta_i = 330 / 365 * Vi * masses ** (0.75) * gamma
        beta_z.append( 330 / 365 * 0.6 * upright_wc ** 0 * masses[0] ** (0.75))
        beta_ff.append(2 * beta_i[1] * (upright_wc / (1 + upright_wc)) + beta_0*330/365*gamma*11**(0.75))

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

        dJsig_z.append(-epsi[0] * state_l[j][-1] * c_z**2*beta_z[j] * rz/(c_z+state_l[j][-1] * sigma_z_l[j] * beta_z[j] * rz)**2 + state_l[j][1] * (
                    c_ff * sigma_ff_l[j] * beta_ff[j] * ff_satiation[j]) + gamma0/2*vel_z_l[j]**2)
        dJsig_ff.append(-(epsi[1] * state_l[j][0] * c_ff ** 2 * beta_ff[j] * sigma_z_l[j] * ff_satiation[j] ** 2) + gamma1/2*vel_ff_l[j]**2)
        dJv_z.append(gamma0*vel_z_l[j]*sigma_z_l[j])
        dJv_ff.append(gamma1*vel_ff_l[j]*sigma_ff_l[j])

        prob_l.append(inte @ Mx.M @ sigma_z_l[j] - 1)
        prob_l.append(inte @ Mx.M @ sigma_ff_l[j] - 1)

    #        if j is 0:
        #else:
        #    prob_l.append([0])
        #    prob_l.append([0])


        s0_vec.append(state_l[j][0])
        s1_vec.append(state_l[j][1])
        s5_vec.append(state_l[j][-1])

        dyn_0.append(state_l[j][0] * (inte @ Mx.M @ (epsi[0] * state_l[j][-1] * sigma_z_l[j]*c_z*beta_z[j] * rz/(c_z+state_l[j][-1] * sigma_z_l[j] * beta_z[j] * rz))
                    - state_l[j][1] * (inte @ (Mx.M @ (c_ff * ff_z_enc[j] * ff_satiation[j]))) - metabolism[0] - gamma0/2*inte @ Mx.M @ (sigma_z_l[j]*vel_z_l[j]**2)))
        dyn_1.append(state_l[j][1] * (epsi[1] * (c_ff * state_l[j][0] * inte @ (Mx.M @ (ff_z_enc[j] * ff_satiation[j])) - f_c) - bg_M - metabolism[1] - gamma1/2*inte @ Mx.M @ (sigma_ff_l[j]*vel_ff_l[j]**2)))
        dyn_5.append(r * (Rmax - state_l[j][-1]) - state_l[j][-1]* state_l[j][0] * inte @ Mx.M @ (sigma_z_l[j]*c_z*beta_z[j] * rz/(c_z+state_l[j][-1] * sigma_z_l[j] * beta_z[j] * rz)))


    v_c = lambda x: ca.vertcat(*x)
    i1 = np.ones(fidelity).reshape(1,fidelity)
    #Total time dynamics
    t_p_z = ca.hcat(p_z_l).T
    t_p_ff = ca.hcat(p_ff_l).T
    t_sigma_z = ca.hcat(sigma_z_l).T
    t_sigma_ff = ca.hcat(sigma_ff_l).T
    #Lagrangian total dynamics
    t_dJsigma_z = ca.hcat(dJsig_z).T
    t_dJsigma_ff = ca.hcat(dJsig_ff).T
    t_dJv_z = ca.hcat(dJv_z).T
    t_dJv_ff = ca.hcat(dJv_ff).T


    bc_mat = np.identity(tot_points)
    bc_mat[-1, -1] = 0
    bc_mat[0,0] = 0
    ones = np.ones(tot_points)

    J_z_p = i1 @(((t_p_z*(ca.hcat(dz_vel_z_l).T) - t_dJsigma_z))**2 @ ones)/(fidelity*tot_points)
    J_ff_p = i1 @ (((t_p_ff*(ca.hcat(dz_vel_ff_l).T) - t_dJsigma_ff))**2 @ ones)/(fidelity*tot_points)
    J_z_v =  i1 @ ((t_dJv_z - t_sigma_z*t_p_z)**2 @ ones)/(fidelity*tot_points)
    J_ff_v = i1 @ ((t_dJv_ff - t_sigma_ff*t_p_ff)**2 @ ones)/(fidelity*tot_points)

    #    can_eqs = i1 @ (canonical_p_z @ ones) + i1 @ (canonical_sigma_z @ ones) + i1 @ (canonical_p_ff @ ones) + i1 @ (canonical_sigma_ff @ ones)
    trans_z = (D @ t_sigma_z + ((bc_mat @  Mx.D) @ ( ca.hcat(sigma_z_l) * ca.hcat(vel_z_l))).T)**2/(fidelity*tot_points)
    trans_ff = (D @ t_sigma_ff + ((bc_mat @  Mx.D) @ ( ca.hcat(sigma_ff_l) * ca.hcat(vel_ff_l))).T)**2/(fidelity*tot_points)
    trans_eqs = i1 @ ( trans_z @ ones) + i1 @ (trans_ff @ ones)

    pop_dyn_eqs = i1 @ (D @ v_c(s0_vec) - v_c(dyn_0))**2 + i1 @ (D @ v_c(s1_vec) - v_c(dyn_1))**2 + i1 @ (D @ v_c(s5_vec) - v_c(dyn_5))**2
    f = pop_dyn_eqs + trans_eqs + J_z_p + J_ff_p + J_z_v + J_ff_v

    x = ca.vertcat(*[*vel_z_l, *vel_ff_l, *p_z_l, *p_ff_l, *sigma_z_l, *sigma_ff_l, *state_l])

    g = ca.vertcat(*[*prob_l])#, ca.reshape(J_z_p, (-1,1)), ca.reshape(J_ff_p, (-1,1)), ca.reshape(J_ff_p, -1,1), ca.reshape(J_z_v, (-1,1)), ca.reshape(J_ff_v, (-1,1))])#, ca.reshape(trans_z,  (-1,1)), ca.reshape(trans_ff, (-1,1))])
    probs = 2*fidelity
    lbg = np.concatenate([np.zeros(probs)])#, np.repeat(-10**(-6), g.size()[0] - probs)])
    #upper_zeros = 2*fidelity+2*fidelity*tot_points
    ubg = np.concatenate([np.zeros(probs)])#, np.repeat(10**(-6), g.size()[0] - probs)])
    #np.zeros(g.size()[0])#ca.vertcat(*[*np.zeros(upper_zeros)])#, (g.size()[0]-(upper_zeros))*[ca.inf]])
    lbx = ca.vertcat(fidelity*4*tot_points*[-ca.inf], np.zeros(tot_points*fidelity*2), np.ones(fidelity*3)*10**(-7))
    ubx = ca.vertcat(*[[ca.inf]*(x.size()[0]-3*fidelity), np.repeat(Rmax, 3*fidelity)])
    prob = {'x': x, 'f': f, 'g': g}

    #s_opts = {'ipopt':{'print_level': 5, 'linear_solver': 'ma57', 'fixed_variable_treatment': "make_constraint"}}  #
    if x.size()[0]<40000:
        linsol = 'ma57'
    else:
        linsol = 'ma86'

    if warmstart_info is None:
        x_init = np.ones(x.size()[0])/Mx.x[-1]#np.concatenate([np.zeros(4*fidelity), np.ones(4*fidelity*tot_points)/Mx.x[-1], np.ones(6*fidelity)])
        s_opts = {'ipopt': {'print_level': 5, 'linear_solver': 'ma57', 'max_iter':10000}}
                            #'tol':10**(-5)}}#, 'hessian_approximation': 'limited-memory'}}  #
    else:
        s_opts = {'ipopt': {'print_level': 3, 'linear_solver': linsol, 'max_iter':35000, 'tol':10**(-5),
                               'hessian_approximation': 'limited-memory',
                               'warm_start_init_point': 'yes', 'warm_start_bound_push': 1e-3,
                               'warm_start_bound_frac': 1e-3, 'warm_start_slack_bound_frac': 1e-3,
                              'warm_start_slack_bound_push': 1e-3, 'warm_start_mult_bound_push': 1e-3}}



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





#def vary_res(min = 5, max = 50, fineness = 45):
#    res_vals = np.linspace(min, max, fineness)

#    with open('data/' + 'pmp2_'+str(fidelity)+'_'+str(tot_points) +'_'+ str(fineness)+'.pkl', 'wb') as f:
#        pkl.dump(out, f, pkl.HIGHEST_PROTOCOL)

#print([range(1,1)])
#vary_res(min = 5, max = 5, fineness = 1)

from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline

from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline

def increase_resolution(ir = 20, fr = 50, jumpsize = 5):
    x_vals = np.linspace(0,100,ir)
    t_vals = np.linspace(0, 1, ir + 1)[:-1]

        #np.linspace(0, 1, ir + 1)[:-1]
    results = []
    results.append(output(tot_points = ir, fidelity = ir))
    decision_vars = []
    state_vars = []

    mult_dec_var = []
    mult_stat_var = []
    rs = lambda x,y: x.reshape((y, y))
    for k in range(6):
        decision_vars.append(RectBivariateSpline(x_vals, t_vals, rs(results[0]['x0'][k*ir**2 : (k+1)*ir**2], ir)))
        mult_dec_var.append(RectBivariateSpline(x_vals, t_vals, rs(results[0]['lam_x0'][k*ir**2 : (k+1)*ir**2], ir)))
    offset = 6*ir**2

    for k in range(3):
        state_vars.append(interp1d(t_vals, results[0]['x0'][offset+k*ir:offset+(k+1)*ir], fill_value="extrapolate"))
        mult_stat_var.append(interp1d(t_vals, results[0]['lam_x0'][offset+k*ir:offset+(k+1)*ir], fill_value="extrapolate"))

    mult_ineq = []
    #offset = 2*ir
    #for k in range(5):
    #    mult_ineq.append(RectBivariateSpline(x_vals, t_vals, rs(results[0]['lam_g0'][offset + k*ir**2 : offset + (k+1)*ir**2], ir)))

    for k in range(2):
        mult_ineq.append(interp1d(t_vals, results[0]['lam_g0'][k*ir:(k+1)*ir], fill_value="extrapolate"))

    counter = 0
    x0_j = []
    lam_x0_j = []
    lam_g0_j = []
    for k in range(6):
        x0_j.append(decision_vars[k](x_vals, t_vals, ).flatten())
        lam_x0_j.append(mult_dec_var[k](x_vals, t_vals, ).flatten())

    for k in range(3):
        x0_j.append(state_vars[k](t_vals))
        lam_x0_j.append(mult_stat_var[k](t_vals))

    for k in range(2):
        lam_g0_j.append(mult_ineq[k](t_vals))
    #for k in range(5):
    #    lam_g0_j.append(mult_ineq[k](x_vals, t_vals).flatten())

    print("Testing info:", "\n", np.linalg.norm(np.concatenate([*x0_j])-results[0]['x0']),  "\n",  np.linalg.norm(np.concatenate([*lam_x0_j])-results[0]['lam_x0']),  "\n", np.linalg.norm(results[0]['lam_g0'] - np.concatenate([*lam_g0_j])))

    for j in range(ir+1, fr, jumpsize):
        counter+=1
        print("Counter: ", counter)
        x_vals = np.linspace(0, 100, j)
        t_vals = np.linspace(0, 1, j + 1)[:-1]
        x0_j = []
        lam_x0_j = []
        lam_g0_j = []
        for k in range(6):
            x0_j.append(decision_vars[k](x_vals, t_vals,).flatten())
            lam_x0_j.append(mult_dec_var[k](x_vals, t_vals,).flatten())

        for k in range(3):
            x0_j.append(state_vars[k](t_vals))
            lam_x0_j.append(mult_stat_var[k](t_vals))

        for k in range(2):
            lam_g0_j.append(mult_ineq[k](t_vals))
        #for k in range(5):
        #    lam_g0_j.append(mult_ineq[k](x_vals, t_vals).flatten())

        warmstart_inf = {'x0': np.concatenate([*x0_j]), 'lam_x0': np.concatenate([*lam_x0_j]), 'lam_g0': np.concatenate([*lam_g0_j])}
        results.append(output(tot_points = j, fidelity = j, warmstart_info=warmstart_inf))

        decision_vars = []
        state_vars = []

        mult_dec_var = []
        mult_stat_var = []
        for k in range(6):
            decision_vars.append(RectBivariateSpline(x_vals, t_vals,  rs(results[counter]['x0'][k * j ** 2: (k + 1) * j ** 2], j)))
            mult_dec_var.append(RectBivariateSpline(x_vals, t_vals,  rs(results[counter]['lam_x0'][k * j ** 2: (k + 1) * j ** 2], j)))
        offset = 6 * j ** 2

        for k in range(3):
            state_vars.append(interp1d(t_vals, results[counter]['x0'][offset + k * j:offset + (k + 1) * j], fill_value="extrapolate"))
            mult_stat_var.append(interp1d(t_vals, results[counter]['lam_x0'][offset + k * j:offset + (k + 1) * j], fill_value="extrapolate"))

        mult_ineq = []
        #offset = 2 * j
        #for k in range(5):
        #    mult_ineq.append(RectBivariateSpline(x_vals, t_vals, rs(results[counter]['lam_g0'][offset+ k * j ** 2: offset + (k + 1) * j ** 2], j)))

        for k in range(2):
            mult_ineq.append(interp1d(t_vals, results[counter]['lam_g0'][k * j:(k + 1) * j], fill_value="extrapolate"))
    with open('data/' + 'pmp2_'+str(fr)+'_'+str(fr) + '.pkl', 'wb') as f:
        pkl.dump(results, f, pkl.HIGHEST_PROTOCOL)

increase_resolution(ir = 10, fr = 85, jumpsize = 15)