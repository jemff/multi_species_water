


from four_spec_sim import *
import pickle as pkl
from scipy.signal import savgol_filter
import numpy as np
h = 17 / (365*24)
a = 0.4
m0 = 10 ** (-3)
gamma = 1
k = 0.05
masses = np.array([0.05, 15, 5000, 4000])

f_c = 0  # 0.15 / (365*24)
r = 1 / (365*24)
r_b = 1 / (365*24)
eps0 = 0.05
#comp = 0
Cmax = h * masses ** (0.75)#*50
metabolism = 0.1 * h *masses**(0.75)
print(metabolism)
epsi = eps0 * ((1 - a) * np.log(masses / m0) - np.log(eps0))

diffusivity_ld = 1 #10**(-3) #0.3*10**(-4)
diffusivity_lp = 1 #10**(-3) #0.3*10**(-4)
diffusivity_ff = 1 #10**(-3) #0.3*10**(-4)
diffusivity_z = 1 #10**(-3) #0.3*10**(-4)

depth = 100



def output(tot_points = 20, fidelity = 20, Rmax = 5, Bmax = 0.1, warmstart_info = None, warmstart_opts = 1e-6, scalar = 'scalar2', hessian_approximation = True):
#    Mx = simple_method(100, tot_points)
    Mx = simple_method(depth, tot_points)
    #must be even
    inte = np.ones(tot_points).reshape(1, tot_points)
    D = fin_diff_mat_periodic(fidelity)
    M_per = M_per_calc(fidelity)
    rz = 1/(1+np.exp(0.2*(Mx.x - 20)))
    rz = rz/(inte @ Mx.M @ rz)

    T_dist = np.exp(-((Mx.x - Mx.x[-1])) ** 2 / (10 ** 2))
    mass_T = inte @ Mx.M @ T_dist
    bent_dist = T_dist / mass_T

    upright_wc = np.exp(-k * Mx.x)  # .reshape((-1, 1))
    c_z = Cmax[0]
    c_ff = Cmax[1]
    c_lp = Cmax[2]
    c_ld = Cmax[3]
    bg_M = 0 #0.1 / (365*24)
    beta_0 = 5*10 ** (-2)
    A = np.cos(2*np.pi*np.linspace(0,1,5*fidelity+5)[0:-5])*1/np.sqrt(2)
    A[A < -1 / 2] = -1 / 2
    A[A > 1 / 2] = 1 / 2
    A = A+1/2
    smoothed_A = savgol_filter(A, window_length = 5, polyorder = 3)
    light_levels = smoothed_A[0::5]

    gamma0 = 1e-9
    gamma1 = 1e-8
    gamma2 = 1e-7
    gamma3 = 1e-7


    p_z_l = []
    p_ff_l = []
    p_lp_l = []
    p_ld_l = []

    sigma_z_l = []
    sigma_ff_l = []
    sigma_lp_l = []
    sigma_ld_l = []

    dz_sigma_z_l = []
    dz_sigma_ff_l = []
    dz_sigma_lp_l = []
    dz_sigma_ld_l = []

    dz_vel_z_l = []
    dz_vel_ff_l = []
    dz_vel_lp_l = []
    dz_vel_ld_l = []

    vel_z_l = []
    vel_ff_l = []
    vel_lp_l = []
    vel_ld_l = []


    dJsig_z = []
    dJsig_ff = []
    dJsig_lp = []
    dJsig_ld = []

    dJv_z = []
    dJv_ff = []
    dJv_lp = []
    dJv_ld = []

    state_l = []
    prob_l = []
    beta_z = []
    beta_ff = []
    beta_ld = []
    beta_ld_b = []
    beta_lp = []

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


    for j in range(fidelity):
        Vi = light_levels[j]
        beta_i = 330 / (365*24) * masses ** (0.75) * gamma
        beta_z.append( 330 / (365*24) * gamma * upright_wc ** 0 * masses[0] ** (0.75))
        beta_ff.append(2 * beta_i[1] * (Vi*upright_wc / (1 + upright_wc) + beta_0)) #+ *330/365*gamma*11**(0.75))

        beta_ld_b.append(0.5 * 330 / (365*24) * (upright_wc ** 0 )* gamma *masses[-1] ** (0.75))
        beta_lp.append(2 * beta_i[2] * (Vi*upright_wc / (1 + upright_wc) + beta_0))
        beta_ld.append(beta_i[3] * (Vi*upright_wc / (1 + upright_wc) + beta_0))


        p_z_l.append(ca.MX.sym('p_z_'+str(j), tot_points))
        p_ff_l.append(ca.MX.sym('p_ff_'+str(j), tot_points))
        p_lp_l.append(ca.MX.sym('p_lp_'+str(j), tot_points))
        p_ld_l.append(ca.MX.sym('p_ld_'+str(j), tot_points))

        sigma_z_l.append(ca.MX.sym('sigma_z_l_'+str(j), tot_points))
        sigma_ff_l.append(ca.MX.sym('sigma_ff_l_'+str(j), tot_points))
        sigma_lp_l.append(ca.MX.sym('sigma_z_l_'+str(j), tot_points))
        sigma_ld_l.append(ca.MX.sym('sigma_ff_l_'+str(j), tot_points))


        vel_z_l.append(ca.MX.sym('vel__l_'+str(j), tot_points))
        vel_ff_l.append(ca.MX.sym('vel_ff_l_'+str(j), tot_points))
        vel_lp_l.append(ca.MX.sym('vel__l_'+str(j), tot_points))
        vel_ld_l.append(ca.MX.sym('vel_ff_l_'+str(j), tot_points))

        state_l.append(ca.MX.sym('state_'+str(j), 6))

        ff_z_enc.append(beta_ff[j] * sigma_z_l[j] * sigma_ff_l[j])
        ff_satiation.append(1 / (state_l[j][0] * ff_z_enc[j] + c_ff))

        lp_ff_enc.append( ( (beta_lp[j] * sigma_ff_l[j] * sigma_lp_l[j])))
        lp_satiation.append(1 / (state_l[j][1] * lp_ff_enc[j] + c_lp))

        ld_ff_enc.append(beta_ld[j] * sigma_ld_l[j] * sigma_ff_l[j])
        ld_bc_enc.append(beta_ld_b[j] * sigma_ld_l[j] * bent_dist)

        ld_satiation.append((1 / (state_l[j][1] * ld_ff_enc[j] + state_l[j][4] * ld_bc_enc[j] + c_ld)))

        dz_sigma_z_l.append(Mx.D @ sigma_z_l[j])
        dz_sigma_ff_l.append(Mx.D @ sigma_ff_l[j])
        dz_sigma_lp_l.append(Mx.D @ sigma_lp_l[j])
        dz_sigma_ld_l.append(Mx.D @ sigma_ld_l[j])

        dz_vel_z_l.append(Mx.D @ vel_z_l[j])
        dz_vel_ff_l.append(Mx.D @ vel_ff_l[j])
        dz_vel_lp_l.append(Mx.D @ vel_lp_l[j])
        dz_vel_ld_l.append(Mx.D @ vel_ld_l[j])


        dJsig_z.append(-epsi[0] * state_l[j][-1] * c_z**2*beta_z[j] * rz/(c_z+state_l[j][-1] * sigma_z_l[j] * beta_z[j] * rz)**2 + state_l[j][1] * (
                    c_ff * sigma_ff_l[j] * beta_ff[j] * ff_satiation[j]) + gamma0/2*vel_z_l[j]**2)
        dJsig_ff.append(-(epsi[1] * state_l[j][0] * c_ff ** 2 * beta_ff[j] * sigma_z_l[j] * ff_satiation[j] ** 2) + state_l[j][2] * sigma_lp_l[j] * c_lp* beta_lp[j] * lp_satiation[j] ** 2 + state_l[j][3] * sigma_ld_l[j] * beta_ld[j] * (ld_satiation[j] ** 2) + gamma1/2*vel_ff_l[j]**2)
        dJsig_lp.append(-(epsi[2] * state_l[j][1] * sigma_ff_l[j] * c_lp ** 2 * beta_lp[j] * lp_satiation[j] ** 2) + gamma2/2*vel_lp_l[j]**2)
        dJsig_ld.append(-c_ld ** 2 * epsi[3] * (
                    state_l[j][1] * sigma_ff_l[j] * beta_ld[j] * (ld_satiation[j] ** 2) + state_l[j][4] * beta_ld_b[j] * bent_dist * (ld_satiation[j] ** 2)) + gamma3/2*vel_ld_l[j]**2)

        dJv_z.append(gamma0*vel_z_l[j]*sigma_z_l[j])
        dJv_ff.append(gamma1*vel_ff_l[j]*sigma_ff_l[j])
        dJv_lp.append(gamma2*vel_lp_l[j]*sigma_lp_l[j])
        dJv_ld.append(gamma3*vel_ld_l[j]*sigma_ld_l[j])

        prob_l.append(inte @ Mx.M @ sigma_z_l[j] - 1)
        prob_l.append(inte @ Mx.M @ sigma_ff_l[j] - 1)
        prob_l.append(inte @ Mx.M @ sigma_lp_l[j] - 1)
        prob_l.append(inte @ Mx.M @ sigma_ld_l[j] - 1)


        s0_vec.append(state_l[j][0])
        s1_vec.append(state_l[j][1])
        s2_vec.append(state_l[j][2])
        s3_vec.append(state_l[j][3])
        s4_vec.append(state_l[j][4])
        s5_vec.append(state_l[j][5])

        dyn_0.append(state_l[j][0] * (inte @ Mx.M @ (epsi[0] * state_l[j][-1] * sigma_z_l[j]*c_z*beta_z[j] * rz/(c_z+state_l[j][-1] * sigma_z_l[j] * beta_z[j] * rz))
                    - state_l[j][1] * (inte @ (Mx.M @ (c_ff * ff_z_enc[j] * ff_satiation[j]))) - metabolism[0] - gamma0/2*inte @ Mx.M @ (sigma_z_l[j]*vel_z_l[j]**2)))
        dyn_1.append(state_l[j][1] * (epsi[1] * (c_ff * state_l[j][0] * inte @ (Mx.M @ (ff_z_enc[j] * ff_satiation[j])) - f_c) - bg_M - metabolism[1] - gamma1/2*inte @ Mx.M @ (sigma_ff_l[j]*vel_ff_l[j]**2)))

        dyn_2.append(state_l[j][2] * (epsi[2] * (inte @ (Mx.M @ (c_lp * state_l[j][1] * lp_ff_enc[j] * lp_satiation[j] ))- f_c) - bg_M -
                    metabolism[2] - gamma2/2*inte @ Mx.M @ (sigma_lp_l[j]*vel_lp_l[j]**2)))
        dyn_3.append(state_l[j][3] * (epsi[3] * inte @ (Mx.M @ ((c_ld * state_l[j][1] * ld_ff_enc[j] * ld_satiation[j] + state_l[j][4] * ld_bc_enc[j] * c_ld * ld_satiation[j])) - f_c) - bg_M - metabolism[3] - gamma3/2*inte @ Mx.M @ (sigma_ld_l[j]*vel_ld_l[j]**2)))
                                    #  - comp * state_l[j][3] * inte @ (Mx.M @ ((beta_ld_b[j] * bent_dist) * sigma_ld_l[j] ** 2))))
        dyn_4.append(r_b * (Bmax - state_l[j][4]) - state_l[j][4]*state_l[j][3] * inte @ ( Mx.M @ (c_ld * ld_bc_enc[j] * ld_satiation[j])))
        dyn_5.append(r * (Rmax - state_l[j][-1]) - state_l[j][-1]* state_l[j][0] * inte @ Mx.M @ (sigma_z_l[j]*c_z*beta_z[j] * rz/(c_z+state_l[j][-1] * sigma_z_l[j] * beta_z[j] * rz)))


    v_c = lambda x: ca.vertcat(*x)
    i1 = np.ones(fidelity).reshape(1,fidelity)
    #Total time dynamics
    t_p_z = ca.hcat(p_z_l).T
    t_p_ff = ca.hcat(p_ff_l).T
    t_p_lp = ca.hcat(p_lp_l).T
    t_p_ld = ca.hcat(p_ld_l).T


    t_sigma_z = ca.hcat(sigma_z_l).T
    t_sigma_ff = ca.hcat(sigma_ff_l).T
    t_sigma_lp = ca.hcat(sigma_lp_l).T
    t_sigma_ld = ca.hcat(sigma_ld_l).T


    #Lagrangian total dynamics
    t_dJsigma_z = ca.hcat(dJsig_z).T
    t_dJsigma_ff = ca.hcat(dJsig_ff).T
    t_dJsigma_lp = ca.hcat(dJsig_lp).T
    t_dJsigma_ld = ca.hcat(dJsig_ld).T

    t_dJv_z = ca.hcat(dJv_z).T
    t_dJv_ff = ca.hcat(dJv_ff).T
    t_dJv_lp = ca.hcat(dJv_lp).T
    t_dJv_ld = ca.hcat(dJv_ld).T

    bc_mat = np.identity(tot_points)
    bc_mat[-1, -1] = 0
    bc_mat[0,0] = 0
    ones = np.ones(tot_points)

    J_z_p = i1 @(((t_p_z*(ca.hcat(dz_vel_z_l).T) - t_dJsigma_z))**2 @ ones)/(fidelity*tot_points)
    J_ff_p = i1 @ (((t_p_ff*(ca.hcat(dz_vel_ff_l).T) - t_dJsigma_ff))**2 @ ones)/(fidelity*tot_points)
    J_lp_p = i1 @(((t_p_lp*(ca.hcat(dz_vel_lp_l).T) - t_dJsigma_lp))**2 @ ones)/(fidelity*tot_points)
    J_ld_p = i1 @ (((t_p_ld*(ca.hcat(dz_vel_ld_l).T) - t_dJsigma_ld))**2 @ ones)/(fidelity*tot_points)



    J_z_v =  i1 @ ((t_dJv_z - ca.hcat(dz_sigma_z_l).T*t_p_z)**2 @ ones)/(fidelity*tot_points)
    J_ff_v = i1 @ ((t_dJv_ff - ca.hcat(dz_sigma_ff_l).T*t_p_ff)**2 @ ones)/(fidelity*tot_points)
    J_lp_v =  i1 @ ((t_dJv_lp - ca.hcat(dz_sigma_lp_l).T*t_p_lp)**2 @ ones)/(fidelity*tot_points)
    J_ld_v = i1 @ ((t_dJv_ld - ca.hcat(dz_sigma_ld_l).T*t_p_ld)**2 @ ones)/(fidelity*tot_points)


    D_trans, D_diff_z = transport_matrix(depth, tot_points, diffusivity = diffusivity_z)
    D_trans, D_diff_ff = transport_matrix(depth, tot_points, diffusivity = diffusivity_ff)
    D_trans, D_diff_lp = transport_matrix(depth, tot_points, diffusivity = diffusivity_lp)
    D_trans, D_diff_ld = transport_matrix(depth, tot_points, diffusivity = diffusivity_lp)


    #    can_eqs = i1 @ (canonical_p_z @ ones) + i1 @ (canonical_sigma_z @ ones) + i1 @ (canonical_p_ff @ ones) + i1 @ (canonical_sigma_ff @ ones)
    trans_z = (D @ t_sigma_z + (D_trans @ ( ca.hcat(sigma_z_l) * ca.hcat(vel_z_l))).T + (D_diff_z @  ca.hcat(sigma_z_l)).T)**2/(fidelity*tot_points)
    trans_ff = (D @ t_sigma_ff + (D_trans @ ( ca.hcat(sigma_ff_l) * ca.hcat(vel_ff_l))).T + (D_diff_ff @  ca.hcat(sigma_ff_l)).T)**2/(fidelity*tot_points)
    trans_lp = (D @ t_sigma_lp + (D_trans @ ( ca.hcat(sigma_lp_l) * ca.hcat(vel_lp_l))).T + (D_diff_lp @  ca.hcat(sigma_lp_l)).T)**2/(fidelity*tot_points)
    trans_ld = (D @ t_sigma_ld + (D_trans @ ( ca.hcat(sigma_ld_l) * ca.hcat(vel_ld_l))).T + (D_diff_lp @  ca.hcat(sigma_ld_l)).T)**2/(fidelity*tot_points)

    trans_z_t = i1 @ ( trans_z @ ones)
    trans_ff_t = i1 @ (trans_ff @ ones)
    trans_lp_t = i1 @ ( trans_lp @ ones)
    trans_ld_t = i1 @ (trans_ld @ ones)

    trans_eqs = trans_z_t + trans_ff_t + trans_lp_t + trans_ld_t

    force_periodic = i1 @ (M_per @ v_c(dyn_0))**2 + i1 @ (M_per @ v_c(dyn_1))**2 + i1 @ (M_per @ v_c(dyn_2))**2 + i1 @ (M_per @ v_c(dyn_3))**2 + i1 @ (M_per @ v_c(dyn_4))**2 + i1 @ (M_per @ v_c(dyn_5))**2

    pop_dyn_eqs = i1 @ (D @ v_c(s0_vec) - v_c(dyn_0))**2 + i1 @ (D @ v_c(s1_vec) - v_c(dyn_1))**2 + \
                  i1 @ (D @ v_c(s2_vec) - v_c(dyn_2))**2 + i1 @ (D @ v_c(s3_vec) - v_c(dyn_3))**2 + \
                  i1 @ (D @ v_c(s4_vec) - v_c(dyn_4))**2 + i1 @ (D @ v_c(s5_vec) - v_c(dyn_5))**2

    f = pop_dyn_eqs

    x = ca.vertcat(*[*vel_z_l, *vel_ff_l, *vel_lp_l, *vel_ld_l, *p_z_l, *p_ff_l, *p_lp_l, *p_ld_l, *sigma_z_l, *sigma_ff_l, *sigma_lp_l, *sigma_ld_l, *state_l])

    g = ca.vertcat(*[*prob_l, trans_z_t, trans_ff_t, trans_lp_t, trans_ld_t,  force_periodic,  J_z_v , J_ff_v, J_lp_v , J_ld_v, J_lp_p, J_ld_p, J_z_p, J_ff_p])#, ca.reshape(J_z_p, (-1,1)), ca.reshape(J_ff_p, (-1,1)), ca.reshape(J_ff_p, -1,1), ca.reshape(J_z_v, (-1,1)), ca.reshape(J_ff_v, (-1,1))])#, ca.reshape(trans_z,  (-1,1)), ca.reshape(trans_ff, (-1,1))])

#    f = pop_dyn_eqs + J_z_p + J_ff_p + J_z_v + J_ff_v + J_lp_p + J_ld_p + J_lp_v + J_ld_v

#    x = ca.vertcat(*[*vel_z_l, *vel_ff_l, *vel_lp_l, *vel_ld_l, *p_z_l, *p_ff_l, *p_lp_l, *p_ld_l, *sigma_z_l, *sigma_ff_l, *sigma_lp_l, *sigma_ld_l, *state_l])

#    g = ca.vertcat(*[*prob_l, trans_z_t, trans_ff_t, trans_lp_t, trans_ld_t, force_periodic])#, ca.reshape(J_z_p, (-1,1)), ca.reshape(J_ff_p, (-1,1)), ca.reshape(J_ff_p, -1,1), ca.reshape(J_z_v, (-1,1)), ca.reshape(J_ff_v, (-1,1))])#, ca.reshape(trans_z,  (-1,1)), ca.reshape(trans_ff, (-1,1))])
    probs = 4*fidelity
#    lbg = np.concatenate([np.zeros(probs+5)])#, np.repeat(-10**(-6), g.size()[0] - probs)])
    #upper_zeros = 2*fidelity+2*fidelity*tot_points
#    ubg = np.concatenate([np.zeros(probs), 1e-7*np.ones(5)])#, np.repeat(10**(-6), g.size()[0] - probs)])
    #np.zeros(g.size()[0])#ca.vertcat(*[*np.zeros(upper_zeros)])#, (g.size()[0]-(upper_zeros))*[ca.inf]])
#    lbx = ca.vertcat(fidelity*8*tot_points*[-ca.inf], np.zeros(tot_points*fidelity*4), np.ones(fidelity*6)*10**(-7))
#    ubx = ca.vertcat(*[[ca.inf]*(x.size()[0]-6*fidelity), np.repeat(Rmax, 6*fidelity)])
#    prob = {'x': x, 'f': f, 'g': g}

    lbg = np.concatenate([np.zeros(probs+13)])#, np.repeat(-10**(-6), g.size()[0] - probs)])
    #upper_zeros = 2*fidelity+2*fidelity*tot_points
    ubg = np.concatenate([np.zeros(probs), 1e-7*np.ones(13)])#, np.repeat(10**(-6), g.size()[0] - probs)])
    #np.zeros(g.size()[0])#ca.vertcat(*[*np.zeros(upper_zeros)])#, (g.size()[0]-(upper_zeros))*[ca.inf]])
    lbx = ca.vertcat(fidelity*8*tot_points*[-ca.inf], np.zeros(tot_points*fidelity*4), np.ones(fidelity*6)*10**(-5))
    ubx = ca.vertcat(*[[ca.inf]*(x.size()[0]-6*fidelity), np.repeat(Rmax, 6*fidelity)])
    prob = {'x': x, 'f': f, 'g': g}

    #s_opts = {'ipopt':{'print_level': 5, 'linear_solver': 'ma57', 'fixed_variable_treatment': "make_constraint"}}  #
    if x.size()[0]<19000:
        linsol = 'ma57'
    else:
        linsol = 'ma57'

    if hessian_approximation is True:
        hess = 'limited-memory'
    else:
        hess = 'exact'
    if warmstart_info is None:
        x_init = 0*np.ones(x.size()[0])/Mx.x[-1]#np.concatenate([np.zeros(4*fidelity), np.ones(4*fidelity*tot_points)/Mx.x[-1], np.ones(6*fidelity)])
        s_opts = {'ipopt': {'print_level': 3, 'linear_solver': 'ma57', 'max_iter':10000}}
                            #'tol':10**(-5)}}#, 'hessian_approximation': 'limited-memory'}}  #
    else:
        s_opts = {'ipopt': {'print_level': 3, 'linear_solver': linsol, 'max_iter':50000, 'tol': 1e-6,
                               'hessian_approximation': hess,
                               'warm_start_init_point': 'yes', 'warm_start_bound_push': warmstart_opts,
                               'warm_start_bound_frac': warmstart_opts, 'warm_start_slack_bound_frac': warmstart_opts,
                              'warm_start_slack_bound_push': warmstart_opts, 'warm_start_mult_bound_push': warmstart_opts, 'limited_memory_max_history': 10, 'limited_memory_initialization':scalar}} #Remark that -3 works well, esp. with -5 above.

                            #limited memory max history improves hessian approximation, hence improves dual efasibility, the limited memory initiizaltion choice of scalar2 was based on trial-and-error, but seems tow rok best

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





from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline

from scipy.interpolate import interp1d
from scipy.interpolate import splrep
from scipy.interpolate import UnivariateSpline

def increase_resolution(ir = 20, fr = 50, jumpsize = 5, Bmax = 0.1, Rmax = 5, warmstart_info = None, warmstart_opts=1e-3):
    x_vals = np.linspace(0,depth,ir)
    t_vals = np.linspace(0, 24, ir + 1)[:-1]

    rs = lambda x,y: x.reshape((y, y))
    counter = 0
    results = []

    for j in range(ir, fr+1, jumpsize):
        print("Current step: ", counter)

        if counter is 0:
            results.append(output(tot_points=ir, fidelity=ir, Bmax = Bmax, Rmax=Rmax, warmstart_info=warmstart_info, warmstart_opts=warmstart_opts, hessian_approximation=False))
            carryover_trans = np.array([results[counter]['lam_g0'][-13:]]).squeeze()

        if counter > 0:
            x_vals = np.linspace(0, depth, j)
            t_vals = np.linspace(0, 24, j + 1)[:-1]
            x0_j = []
            lam_x0_j = []
            lam_g0_j = []
            for k in range(12):
                x0_j.append(decision_vars[k](x_vals, t_vals,).flatten())
                lam_x0_j.append(mult_dec_var[k](x_vals, t_vals,).flatten())

            x0_j_state = np.zeros(6 * j)
            lam_x0_j_state = np.zeros(6 * j)
            for k in range(6):
                #print(state_vars[k])
                x0_j_state[k::6] = (state_vars[k](t_vals))
                lam_x0_j_state[k::6] = (mult_stat_var[k](t_vals))

            for k in range(4):
                lam_g0_j.append(mult_ineq[k](t_vals))
            lam_g0_j.append(carryover_trans)
            warmstart_inf = {'x0': np.concatenate([*x0_j, x0_j_state]), 'lam_x0': np.concatenate([*lam_x0_j, lam_x0_j_state]), 'lam_g0': np.concatenate([*lam_g0_j])}
            results.append(output(tot_points = j, fidelity = j, warmstart_info=warmstart_inf, Bmax=Bmax, Rmax=Rmax))
            carryover_trans = np.array([results[counter]['lam_g0'][-13:]]).squeeze()


        decision_vars = []
        state_vars = []

        mult_dec_var = []
        mult_stat_var = []
        for k in range(12):
            decision_vars.append(interp2d(x_vals, t_vals,  rs(results[counter]['x0'][k * j ** 2: (k + 1) * j ** 2], j), kind = 'linear'))
            mult_dec_var.append(interp2d(x_vals, t_vals,  rs(results[counter]['lam_x0'][k * j ** 2: (k + 1) * j ** 2], j), kind = 'linear'))

        offset = 12 * j ** 2

        state_var_cop = np.copy(results[counter]['x0'][offset:])
        mult_stat_var_cop = np.copy(results[counter]['lam_x0'][offset:])
        s_l = []
        lam_s_l = []
        for k in range(6):
            s_l.append(state_var_cop[k::6])
            lam_s_l.append(mult_stat_var_cop[k::6])
            state_vars.append(interp1d(t_vals, s_l[k], fill_value="extrapolate"))
            mult_stat_var.append(interp1d(t_vals, lam_s_l[k], fill_value="extrapolate"))

        mult_ineq = []
        for k in range(4):
            mult_ineq.append(interp1d(t_vals, results[counter]['lam_g0'][k * j:(k + 1) * j], fill_value="extrapolate"))


        counter+=1

    with open('data/' + 'pdeco4_'+str(fr)+'_'+str(jumpsize) + "_B"+ str(Bmax) + "_R" + str(Rmax) + '.pkl', 'wb') as f:
        pkl.dump(results, f, pkl.HIGHEST_PROTOCOL)

    return results


def vary_resources(start_r = 5.0, stop_r = 50.0, start_b = 0.01, stop_b = 2.0, steps_r = 23, steps_b = 10, ir = 10, fr = 50, jumpsize = 10):
    print("Starting resource variation")
    resources = np.linspace(start_r, stop_r, steps_r)
    benthos = np.linspace(start_b, stop_b, steps_b)
    grid_variation = []
    for j in range(0,steps_b):
        for k in range(0, steps_r):
            if j is 0 and k is 0:
                grid_variation.append(
                    increase_resolution(ir=ir, fr=fr, jumpsize=jumpsize, Bmax=benthos[0], Rmax=resources[0]))
            else:
                if j>0 and k == 0:
                    warmstarter = -steps_r*j + 1
                else:
                    warmstarter = - 1
                grid_variation.append(
                    increase_resolution(ir=ir, fr=fr, jumpsize=jumpsize, warmstart_info=grid_variation[warmstarter][0],  Rmax=resources[k], Bmax=benthos[j], warmstart_opts = 1e-6))
                #grid_variation.append(
                #    increase_resolution(ir=ir, fr=fr, jumpsize=jumpsize, Bmax=benthos[j], Rmax=resources[k])[-1])

            print(print(100*((j*steps_b+(k+1))/(steps_r*steps_b)), " % complete"))
    with open('data/' + 'pdeco4_res_' + str(jumpsize) + '.pkl', 'wb') as f:
        pkl.dump(grid_variation, f, pkl.HIGHEST_PROTOCOL)

#    with open('data/' + 'pdeco4_res_'+str(jumpsize) + '.pkl', 'wb') as f:
#    pkl.dump(grid_variation, f, pkl.HIGHEST_PROTOCOL)

vary_resources(ir = 5, fr = 50, jumpsize=5)