


from four_spec_sim import *
import pickle as pkl
from scipy.signal import savgol_filter
import numpy as np
h = 20 / (365*24*60)
a = 0.4
m0 = 10 ** (-3)
gamma = 1
k = 0.05
masses = np.array([0.05, 20])

f_c = 0  # 0.15 / (365*24)
r = 1 / (365*24)
r_b = 1 / (365*24)
eps0 = 0.05
#comp = 0
Cmax = h * masses ** (0.75)#*50
metabolism = 0.1 * h *masses**(0.75)
print(metabolism)
epsi = eps0 * ((1 - a) * np.log(masses / m0) - np.log(eps0))

diffusivity_ff = 1/60 #10**(-3) #0.3*10**(-4)
diffusivity_z = 1/60 #10**(-3) #0.3*10**(-4)
gamma0 = 1e-4
gamma1 = 1e-3

depth = 100
def output(tot_points = 20, fidelity = 20, Rmax = 5, Bmax = 0.1, warmstart_info = None, warmstart_opts = 1e-3, scalar = 'scalar2', hessian_approximation = True):
    Mx = discrete_patches(depth, tot_points)
    #must be even
    D_trans, D_diff_z = transport_matrix(depth, tot_points, diffusivity = diffusivity_z)
    D_trans, D_diff_ff = transport_matrix(depth, tot_points, diffusivity = diffusivity_ff)

    inte = np.ones(tot_points).reshape(1, tot_points)
    D = fin_diff_mat_periodic(fidelity, length=24*60)
    D_hjb = D_trans #D_HJB(depth = depth, total_points=tot_points)

    M_per = M_per_calc(fidelity, length = 24*60)
    rz = 1/(1+np.exp(0.2*(Mx.x - 20)))
    rz = rz/(inte @ Mx.M @ rz)
    bg_M = 0
    upright_wc = np.exp(-k * Mx.x)  # .reshape((-1, 1))
    c_z = Cmax[0]
    c_ff = Cmax[1]
    beta_0 = 5*10 ** (-2)
    A = np.cos(2*np.pi*np.linspace(0,1,5*fidelity+5)[0:-5])#*1/np.sqrt(2)
    A[A < -1 / 2] = -1 / 2
    A[A > 1 / 2] = 1 / 2
    A = A+1/2
    smoothed_A = savgol_filter(A, window_length = 5, polyorder = 3)
    light_levels = smoothed_A[0::5]


    p_z_l = []
    p_ff_l = []

    sigma_z_l = []
    sigma_ff_l = []

    dz_sigma_z_l = []
    dz_sigma_ff_l = []


    vel_z_l = []
    vel_ff_l = []


    Jsig_z = []
    Jsig_ff = []


    state_l = []
    prob_l = []
    beta_z = []
    beta_ff = []

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
        beta_i = 330 / (365*24*60) * masses ** (0.75) * gamma
        beta_z.append( 330 / (365*24*60) * gamma * upright_wc ** 0 * masses[0] ** (0.75))
        beta_ff.append(2 * beta_i[1] * (Vi*upright_wc / (1 + upright_wc) + beta_0)) #+ *330/365*gamma*11**(0.75))



        p_z_l.append(ca.MX.sym('p_z_'+str(j), tot_points))
        p_ff_l.append(ca.MX.sym('p_ff_'+str(j), tot_points))

        sigma_z_l.append(ca.MX.sym('sigma_z_l_'+str(j), tot_points))
        sigma_ff_l.append(ca.MX.sym('sigma_ff_l_'+str(j), tot_points))


        vel_z_l.append(D_hjb @ p_z_l[j])
        vel_ff_l.append(1/gamma1*D_hjb @ p_ff_l[j])

        state_l.append(ca.MX.sym('state_'+str(j), 3))

        ff_z_enc.append(beta_ff[j] * sigma_z_l[j] * sigma_ff_l[j])
        ff_satiation.append(1 / (state_l[j][0] * ff_z_enc[j] + c_ff))


        dz_sigma_z_l.append(Mx.D @ sigma_z_l[j])
        dz_sigma_ff_l.append(Mx.D @ sigma_ff_l[j])


        Jsig_z.append(epsi[0] * state_l[j][-1] * c_z*beta_z[j] * rz/(c_z+state_l[j][-1] * beta_z[j] * rz) - state_l[j][1] * (
                    c_ff * sigma_ff_l[j] * beta_ff[j] * ff_satiation[j]) - 1/(gamma0*2)*vel_z_l[j]**2)
        Jsig_ff.append((epsi[1] * state_l[j][0] * c_ff * beta_ff[j] * sigma_z_l[j]/(c_ff + beta_ff[j]*state_l[j][0]*sigma_z_l[j])) - 1/(gamma1*2)*vel_ff_l[j]**2)



        prob_l.append(inte @ Mx.M @ sigma_z_l[j] - 1)
        prob_l.append(inte @ Mx.M @ sigma_ff_l[j] - 1)


        s0_vec.append(state_l[j][0])
        s1_vec.append(state_l[j][1])
        s5_vec.append(state_l[j][2])

        dyn_0.append(state_l[j][0] * (inte @ Mx.M @ (epsi[0] * state_l[j][-1] * sigma_z_l[j]*c_z*beta_z[j] * rz/(c_z+state_l[j][-1] * sigma_z_l[j] * beta_z[j] * rz))
                    - state_l[j][1] * (inte @ (Mx.M @ (c_ff * ff_z_enc[j] * ff_satiation[j]))) - metabolism[0] - 1/(gamma0*2)*inte @ Mx.M @ (sigma_z_l[j]*vel_z_l[j]**2)))
        dyn_1.append(state_l[j][1] * (epsi[1] * (c_ff * state_l[j][0] * inte @ (Mx.M @ (ff_z_enc[j] * ff_satiation[j])) - f_c) - bg_M - metabolism[1] - 1/(gamma1*2)*inte @ Mx.M @ (sigma_ff_l[j]*vel_ff_l[j]**2)))

        dyn_5.append(r * (Rmax - state_l[j][-1]) - state_l[j][-1]* state_l[j][0] * inte @ Mx.M @ (sigma_z_l[j]*c_z*beta_z[j] * rz/(c_z+state_l[j][-1] * sigma_z_l[j] * beta_z[j] * rz)))


    v_c = lambda x: ca.vertcat(*x)
    i1 = np.ones(fidelity).reshape(1,fidelity)
    #Total time dynamics
    t_p_z = ca.hcat(p_z_l).T
    t_p_ff = ca.hcat(p_ff_l).T


    t_sigma_z = ca.hcat(sigma_z_l).T
    t_sigma_ff = ca.hcat(sigma_ff_l).T


    #Lagrangian total dynamics
    t_Jsigma_z = ca.hcat(Jsig_z).T
    t_Jsigma_ff = ca.hcat(Jsig_ff).T

    bc_mat = np.identity(tot_points)
    bc_mat[-1, -1] = 0
    bc_mat[0,0] = 0
    ones = np.ones(tot_points)



    J_z_p = i1 @ (M_per @ (t_Jsigma_z + D @ t_p_z + 1/gamma0*(D_hjb @ t_p_z.T).T * ca.hcat(vel_z_l).T - (D_diff_z @ t_p_z.T).T) ** 2 @ ones) /(tot_points-1) #((fidelity - 1) * (tot_points - 1)) #Repurposed to HJB

    J_ff_p = i1 @ (M_per @ (t_Jsigma_ff + D @ t_p_ff + 1/gamma1*(D_hjb @ t_p_ff.T).T * ca.hcat(vel_ff_l).T - (D_diff_ff @ t_p_ff.T).T )**2 @ ones)/(tot_points-1) #/((fidelity-1)*(tot_points-1)) #Repurposed to HJB

    #    can_eqs = i1 @ (canonical_p_z @ ones) + i1 @ (canonical_sigma_z @ ones) + i1 @ (canonical_p_ff @ ones) + i1 @ (canonical_sigma_ff @ ones)
    trans_z = (D @ t_sigma_z + 1/gamma0*(D_trans @ ( ca.hcat(sigma_z_l) * ca.hcat(vel_z_l))).T + (D_diff_z @  ca.hcat(sigma_z_l)).T)**2#/((fidelity-1)*(tot_points-1))
    trans_ff = (D @ t_sigma_ff +1/gamma1* (D_trans @ ( ca.hcat(sigma_ff_l) * ca.hcat(vel_ff_l))).T + (D_diff_ff @  ca.hcat(sigma_ff_l)).T)**2#/((fidelity-1)*(tot_points-1))

    trans_z_t = i1 @ (M_per @ trans_z @ ones)/(tot_points-1)
    trans_ff_t = i1 @ (M_per @ trans_ff @ ones)/(tot_points-1)

    force_periodic = i1 @ (M_per @ v_c(dyn_0))**2 + i1 @ (M_per @ v_c(dyn_1))**2 + i1 @ (M_per @ v_c(dyn_5))**2

    pop_dyn_eqs = i1 @ (D @ v_c(s0_vec) - v_c(dyn_0))**2 + i1 @ (D @ v_c(s1_vec) - v_c(dyn_1))**2 + \
                  + i1 @ (D @ v_c(s5_vec) - v_c(dyn_5))**2


    f = pop_dyn_eqs

    x = ca.vertcat(*[*p_z_l, *p_ff_l, *sigma_z_l, *sigma_ff_l, *state_l])

    g = ca.vertcat(*[*prob_l, trans_z_t, trans_ff_t, force_periodic,   J_z_p, J_ff_p])#, ca.reshape(J_z_p, (-1,1)), ca.reshape(J_ff_p, (-1,1)), ca.reshape(J_ff_p, -1,1), ca.reshape(J_z_v, (-1,1)), ca.reshape(J_ff_v, (-1,1))])#, ca.reshape(trans_z,  (-1,1)), ca.reshape(trans_ff, (-1,1))])
    probs = 2*fidelity
    lbg = np.concatenate([np.zeros(probs+5)])#, np.repeat(-10**(-6), g.size()[0] - probs)])
    #upper_zeros = 2*fidelity+2*fidelity*tot_points
    ubg = np.concatenate([np.zeros(probs), 1e-8*np.ones(2), 1e-7*np.ones(3)])#, np.repeat(10**(-6), g.size()[0] - probs)])
    #np.zeros(g.size()[0])#ca.vertcat(*[*np.zeros(upper_zeros)])#, (g.size()[0]-(upper_zeros))*[ca.inf]])
    lbx = ca.vertcat(fidelity*2*tot_points*[-ca.inf], np.zeros(tot_points*fidelity*2), np.ones(fidelity*3)*10**(-5))
    ubx = ca.vertcat(*[[ca.inf]*(x.size()[0]-3*fidelity), np.repeat(Rmax, 3*fidelity)])
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
        x_init = np.ones(x.size()[0])/Mx.x[-1]#np.concatenate([np.zeros(4*fidelity), np.ones(4*fidelity*tot_points)/Mx.x[-1], np.ones(6*fidelity)])
        s_opts = {'ipopt': {'print_level': 3, 'linear_solver': 'ma57', 'max_iter':10000}}
                            #'tol':10**(-5)}}#, 'hessian_approximation': 'limited-memory'}}  #
    else:
        s_opts = {'ipopt': {'print_level': 3, 'linear_solver': linsol, 'max_iter':50000, 'tol': 1e-6,
                               'hessian_approximation': hess,
                               'warm_start_init_point': 'yes', 'warm_start_bound_push': warmstart_opts,
                               'warm_start_bound_frac': warmstart_opts, 'warm_start_slack_bound_frac': warmstart_opts,
                              'warm_start_slack_bound_push': warmstart_opts, 'warm_start_mult_bound_push': warmstart_opts, 'limited_memory_max_history': 20, 'limited_memory_initialization':scalar}} #Remark that -3 works well, esp. with -5 above.

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

def increase_resolution(ir_t = 20, fr_t = 50, ir_s = 20, fr_s = 50, jumpsize_t = 5, jumpsize_s = 5, save = True, Rmax = 5, warmstart_info = None, warmstart_opts = 10**(-6)):
    x_vals = np.linspace(0,depth,ir_s)
    t_vals = np.linspace(0, 24, ir_t + 1)[:-1]

        #np.linspace(0, 1, ir + 1)[:-1]
    results = []
    results.append(output(tot_points = ir_s, fidelity = ir_t, Rmax=Rmax, warmstart_info = warmstart_info, warmstart_opts=warmstart_opts, hessian_approximation=False))
    decision_vars = []
    state_vars = []

    mult_dec_var = []
    mult_stat_var = []
    def rs(x, y, z = None):
        if z is None:
            return x.reshape((y,y))
        else:
            return x.reshape((y,z))

    for k in range(4):
        decision_vars.append(interp2d(t_vals, x_vals, rs(results[0]['x0'][k*ir_s*ir_t : (k+1)*ir_s*ir_t], ir_s, ir_t), kind = 'linear'))
        mult_dec_var.append(interp2d(t_vals, x_vals, rs(results[0]['lam_x0'][k*ir_s*ir_t : (k+1)*ir_s*ir_t], ir_s, ir_t), kind = 'linear'))
    offset = 4*ir_s*ir_t

    state_var_cop = np.copy(results[0]['x0'][offset:])
    mult_stat_var_cop = np.copy(results[0]['lam_x0'][offset:])
    s0 = state_var_cop[0::3]
    s1 = state_var_cop[1::3]
    s2 = state_var_cop[2::3]
    s_l = [s0, s1, s2]
    lam_s0 = mult_stat_var_cop[0::3]
    lam_s1 = mult_stat_var_cop[1::3]
    lam_s2 = mult_stat_var_cop[2::3]

    lam_s_l = [lam_s0, lam_s1, lam_s2]
    for k in range(3):
        state_vars.append(interp1d(t_vals, s_l[k], fill_value="extrapolate"))
        mult_stat_var.append(interp1d(t_vals, lam_s_l[k], fill_value="extrapolate"))

    mult_ineq = []
    for k in range(2):
        mult_ineq.append(interp1d(t_vals, results[0]['lam_g0'][k*ir_t:(k+1)*ir_t], fill_value="extrapolate"))
    #mult_ineq.append(results[0]['lam_g0'][-5:])
    #offset = 2*ir_t
    #for k in range(5):
    #    mult_ineq.append(interp2d(x_vals, t_vals, rs(results[0]['lam_g0'][offset + k*ir**2 : offset + (k+1)*ir**2], ir)))

    counter = 0
    x0_j = []
    lam_x0_j = []
    lam_g0_j = []
    for k in range(4):
        x0_j.append(decision_vars[k](t_vals, x_vals).flatten())
        lam_x0_j.append(mult_dec_var[k](t_vals, x_vals).flatten())


    x0_j_state = np.zeros(3 * ir_t)
    lam_x0_j_state = np.zeros(3 * ir_t)
    for k in range(3):
        x0_j_state[k::3] = (state_vars[k](t_vals))
        lam_x0_j_state[k::3] = (mult_stat_var[k](t_vals))


    for k in range(2):
        lam_g0_j.append(mult_ineq[k](t_vals))
    lam_g0_j.append(np.array([results[0]['lam_g0'][-5:]]).squeeze())
    #for k in range(5):
    #    lam_g0_j.append(mult_ineq[2+k](x_vals, t_vals).flatten())
    print(np.concatenate([*lam_g0_j]).shape)
    print("Testing info:", "\n", np.linalg.norm(np.concatenate([*x0_j, x0_j_state])-results[0]['x0']),  "\n",  np.linalg.norm(np.concatenate([*lam_x0_j, lam_x0_j_state])-results[0]['lam_x0']),  "\n", np.linalg.norm(results[0]['lam_g0'] - np.concatenate([*lam_g0_j])))
    j_t_l = np.array(range(ir_t, fr_t+1, jumpsize_t))
    j_s_l = np.array(range(ir_s, fr_s+1, jumpsize_s))
    print(j_t_l, j_s_l)

    j_l = []
    for k in range(max(len(j_s_l), len(j_t_l))):
        j_l.append([j_s_l[min(k, len(j_s_l)-1)], j_t_l[min(k, len(j_t_l)-1)]])

    for j in range(1, len(j_l)):
        j_t = j_l[j][1]
        j_s = j_l[j][0]

        counter+=1
        print("Counter: ", counter)
        x_vals = np.linspace(0, depth, j_s)
        t_vals = np.linspace(0, 24, j_t + 1)[:-1]
        x0_j = []
        lam_x0_j = []
        lam_g0_j = []
        for k in range(4):
            x0_j.append(decision_vars[k](t_vals, x_vals).flatten())
            lam_x0_j.append(mult_dec_var[k](t_vals, x_vals).flatten())

        x0_j_state = np.zeros(3 * j_t)
        lam_x0_j_state = np.zeros(3 * j_t)
        for k in range(3):
            x0_j_state[k::3] = (state_vars[k](t_vals))
            lam_x0_j_state[k::3] = (mult_stat_var[k](t_vals))

        for k in range(2):
            lam_g0_j.append(mult_ineq[k](t_vals))
        lam_g0_j.append(np.array([results[j-1]['lam_g0'][-5:]]).squeeze())

        warmstart_inf = {'x0': np.concatenate([*x0_j, x0_j_state]), 'lam_x0': np.concatenate([*lam_x0_j, lam_x0_j_state]), 'lam_g0': np.concatenate([*lam_g0_j])}
        results.append(output(tot_points = j_s, fidelity = j_t, warmstart_info=warmstart_inf, Rmax=Rmax))

        decision_vars = []
        state_vars = []

        mult_dec_var = []
        mult_stat_var = []
        for k in range(4):
            decision_vars.append(interp2d(t_vals, x_vals,  rs(results[counter]['x0'][k * j_s*j_t: (k + 1) * j_s*j_t], j_s, j_t), kind = 'linear'))
            mult_dec_var.append(interp2d(t_vals, x_vals,  rs(results[counter]['lam_x0'][k * j_s*j_t: (k + 1) * j_s*j_t], j_s, j_t), kind = 'linear'))
        offset = 4 * j_s*j_t

        state_var_cop = np.copy(results[counter]['x0'][offset:])
        mult_stat_var_cop = np.copy(results[counter]['lam_x0'][offset:])
        s0 = state_var_cop[0::3]
        s1 = state_var_cop[1::3]
        s2 = state_var_cop[2::3]
        s_l = [s0, s1, s2]
        lam_s0 = mult_stat_var_cop[0::3]
        lam_s1 = mult_stat_var_cop[1::3]
        lam_s2 = mult_stat_var_cop[2::3]

        lam_s_l = [lam_s0, lam_s1, lam_s2]
        for k in range(3):
            state_vars.append(interp1d(t_vals, s_l[k], fill_value="extrapolate"))
            mult_stat_var.append(interp1d(t_vals, lam_s_l[k], fill_value="extrapolate"))

        mult_ineq = []
        for k in range(2):
            mult_ineq.append(interp1d(t_vals, results[counter]['lam_g0'][k * j_t:(k + 1) * j_t], fill_value="extrapolate"))
        #mult_ineq.append(results[counter]['lam_g0'][-7:])
    if save is True:
        with open('data/' + 'pdeco2_'+str(fr_t*fr_s)+'_' + str(Rmax) + '.pkl', 'wb') as f:
            pkl.dump(results, f, pkl.HIGHEST_PROTOCOL)

    return results


def vary_resources(start = 5, stop = 50, steps = 45, ir_t = 10, fr_t = 70, ir_s = 10, fr_s = 70, jumpsize = 1):
    grid_variation = []
    jump_scale = int(ir_t/ir_s)
    print("Starting resource variation")
    resources = np.linspace(start, max(stop,start), steps)
    grid_variation.append(increase_resolution(ir_t=ir_t, ir_s=ir_s, fr_t=fr_t, fr_s=fr_s, jumpsize_s=jumpsize, jumpsize_t=jump_scale*jumpsize, Rmax=resources[0]))
    print(resources)
    for i in range(1,steps):
        grid_variation.append(
            increase_resolution(ir_t=ir_t, ir_s=ir_s, fr_t=fr_t, fr_s=fr_s, jumpsize_s = jumpsize, jumpsize_t = jump_scale*jumpsize, Rmax=resources[i], warmstart_info=grid_variation[-1][0], warmstart_opts = 1e-3))
        print((i+1)/steps, " Completion ratio")
    results_and_params = {'gamma0': gamma0, 'gamma1': gamma1, 'simulations':grid_variation}
    with open('data/' + 'pdeco2_res_'+str(fr_s*fr_t)+'_'+str(stop) + '.pkl', 'wb') as f:
        pkl.dump(results_and_params, f, pkl.HIGHEST_PROTOCOL)

#    with open('data/' + 'pdeco4_res_'+str(jumpsize) + '.pkl', 'wb') as f:
#    pkl.dump(grid_variation, f, pkl.HIGHEST_PROTOCOL)

vary_resources(steps = 46, ir_s = 4, ir_t= 12, fr_s=34, fr_t=102, jumpsize=1)