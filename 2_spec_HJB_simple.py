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





def M_per_calc(N, length = 24):
    M_per = 2 / 3 * np.identity(N) + 1 / 6 * np.diag(np.ones(N - 1), -1) + 1 / 6 * np.diag(
        np.ones(N - 1), 1)
    M_per[0,-1] = 1/6
    M_per[-1,0] = 1/6
    h = N/length
    M_per = M_per / h

    return M_per

from four_spec_sim import *
import pickle as pkl
from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt
h = 17 / (24*365)
a = 0.4
m0 = 10 ** (-3)
gamma = 1 # 0.6
k = 0.05 #5# 0.05 #0.1 #0.06
masses = np.array([0.02, 10])
f_c = 0  # 0.15 / 36
r = 1 / (24*365)
r_b = 1 /(24*365)
eps0 = 0.05
#comp = 0
Cmax = h * masses ** (0.75)
Cmax[0] = Cmax[0]
Cmax[1] = Cmax[1]
metabolism = 0.1 * h *masses**(0.75) #* Cmax #h *masses**(0.75)
print(metabolism)
epsi = eps0 * ((1 - a) * np.log(masses / m0) - np.log(eps0)) #0.7*np.ones(2)
gamma0 = 1e-9 #7 #-8
gamma1 = 1e-8 #6 #-7
diffusivity_ff = 1 #10**(-3) #0.3*10**(-4)
diffusivity_z = 1 #10**(-3) #0.3*10**(-4)
print(Cmax)

#diffusivity = 10**(-3)
depth = 100

def output(tot_points = 20, fidelity = 20, R_max = 5, warmstart_info = None, warmstart_opts = 1e-6, tol = 1e-6):
    Mx = simple_method(depth, tot_points)

    M_per = M_per_calc(fidelity)
    #must be even
    inte = np.ones(tot_points).reshape(1, tot_points)
    D = fin_diff_mat_periodic(fidelity) #fin_diff_mat_periodic(fidelity)
    D_hjb = D_HJB(depth = depth, total_points=tot_points)
    rz = 1/(1+np.exp(0.2*(Mx.x - 20)))
    rz = rz/(inte @ Mx.M @ rz)
    #1 / (8.8) * np.exp(-((Mx.x)) ** 2 / ((0.1*depth) ** 2))
    #np.exp(-0.1*Mx.x)/10 #1 / (8.8) * np.exp(-((Mx.x)) ** 2 / (10 ** 2)) #+ 10 ** (-4)
    upright_wc = np.exp(-k * Mx.x)  # .reshape((-1, 1))
    c_z = Cmax[0]
    c_ff = Cmax[1]
    bg_M = 0 #0.1 / (36*24)
    beta_0 = 5*10**(-2) #10 ** (-5)
    A = np.cos(2*np.pi*np.linspace(0,1,5*fidelity+5)[0:-5])*1/np.sqrt(2)
    A[A < -1 / 2] = -1 / 2
    A[A > 1 / 2] = 1 / 2
    A = A+1/2
    smoothed_A = savgol_filter(A, window_length = 5, polyorder = 3)
    light_levels = smoothed_A[0::5]
    freq = str(int(24 * 60 / (fidelity))) + "Min"
    #light_levels = solar_input_calculator(freq=freq)[0:-1]
    #print(len((light_levels)))
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
    Jsig_z = []
    Jsig_ff = []
    dJv_z = []
    dJv_ff = []

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
        beta_i = 330 / ((24*365)) * Vi * masses ** (0.75) * gamma
        beta_z.append(330 / ((24*365)) * gamma * upright_wc ** 0 * masses[0] ** (0.75))
        beta_ff.append(2 * beta_i[1] * (upright_wc / (1 + upright_wc)) + beta_0*330/((24*365))*masses[1]**(0.75))

        p_z_l.append(ca.MX.sym('p_z_'+str(j), tot_points))
        p_ff_l.append(ca.MX.sym('p_ff_'+str(j), tot_points))
        sigma_z_l.append(ca.MX.sym('sigma_z_l_'+str(j), tot_points))
        sigma_ff_l.append(ca.MX.sym('sigma_ff_l_'+str(j), tot_points))
        vel_z_l.append(ca.MX.sym('vel_z_l_'+str(j), tot_points))
        vel_ff_l.append(ca.MX.sym('vel_ff_l_'+str(j), tot_points))

        state_l.append(ca.MX.sym('state_'+str(j), 3))

        ff_z_enc.append(beta_ff[j] * sigma_z_l[j] * sigma_ff_l[j])
        ff_satiation.append((state_l[j][0] * ff_z_enc[j] + c_ff))

        dz_sigma_z_l.append(Mx.D @ sigma_z_l[j])
        dz_sigma_ff_l.append(Mx.D @ sigma_ff_l[j])
        dz_vel_z_l.append(Mx.D @ vel_z_l[j])
        dz_vel_ff_l.append(Mx.D @ vel_ff_l[j])

        Jsig_z.append(epsi[0] * state_l[j][-1] * c_z*beta_z[j] * rz/(c_z+state_l[j][-1] * beta_z[j] * rz) - state_l[j][1] *
                    c_ff * beta_ff[j] * sigma_ff_l[j]/(ff_satiation[j]) - (gamma0/2)*vel_z_l[j]**2)
        Jsig_ff.append((epsi[1] * state_l[j][0] * c_ff * beta_ff[j] * sigma_z_l[j]/(state_l[j][0] * beta_ff[j] * sigma_z_l[j] + c_ff)) - (gamma1/2)*vel_ff_l[j]**2)
        dJv_z.append(-gamma0*vel_z_l[j])
        dJv_ff.append(-gamma1*vel_ff_l[j])

        prob_l.append(inte @ Mx.M @ sigma_z_l[j] - 1)
        prob_l.append(inte @ Mx.M @ sigma_ff_l[j] - 1)

        s0_vec.append(state_l[j][0])
        s1_vec.append(state_l[j][1])
        s5_vec.append(state_l[j][-1])

        dyn_0.append(state_l[j][0] * (inte @ Mx.M @ (epsi[0] * state_l[j][-1] * sigma_z_l[j]*c_z*beta_z[j] * rz/(c_z+state_l[j][-1] * sigma_z_l[j] * beta_z[j] * rz))
                    - state_l[j][1] * (inte @ (Mx.M @ (c_ff * ff_z_enc[j] / ff_satiation[j]))) - metabolism[0] - gamma0/2*inte @ Mx.M @ (sigma_z_l[j]*vel_z_l[j]**2)))
        dyn_1.append(state_l[j][1] * (epsi[1] * (c_ff * state_l[j][0] * inte @ (Mx.M @ (ff_z_enc[j] / ff_satiation[j])) - f_c) - bg_M - metabolism[1] - gamma1/2*inte @ Mx.M @ (sigma_ff_l[j]*vel_ff_l[j]**2)))
        dyn_5.append(r * (Rmax - state_l[j][-1]) - state_l[j][-1]* state_l[j][0] * inte @ Mx.M @ (sigma_z_l[j]*c_z*beta_z[j] * rz/(c_z+state_l[j][-1] * sigma_z_l[j] * beta_z[j] * rz)))


    v_c = lambda x: ca.vertcat(*x)
    i1 = np.ones(fidelity).reshape(1,fidelity)
    #Total time dynamics
    t_p_z = ca.hcat(p_z_l).T
    t_p_ff = ca.hcat(p_ff_l).T
    t_sigma_z = ca.hcat(sigma_z_l).T
    t_sigma_ff = ca.hcat(sigma_ff_l).T
    t_sigma_z_dz = ca.hcat(dz_sigma_z_l).T
    t_sigma_ff_dz = ca.hcat(dz_sigma_ff_l).T
    #Lagrangian total dynamics
    t_Jsigma_z = ca.hcat(Jsig_z).T
    t_Jsigma_ff = ca.hcat(Jsig_ff).T
    t_dJv_z = ca.hcat(dJv_z).T
    t_dJv_ff = ca.hcat(dJv_ff).T


    bc_mat = np.identity(tot_points)
    bc_mat[-1, -1] = 0
    bc_mat[0,0] = 0
    ones = np.ones(tot_points)
    i1 = np.ones(fidelity).reshape(1,fidelity)


    D_trans, D_diff_z = transport_matrix(depth, tot_points, diffusivity = diffusivity_z)
    D_trans, D_diff_ff = transport_matrix(depth, tot_points, diffusivity = diffusivity_ff)

    J_z_p = i1 @(((t_Jsigma_z + D @ t_p_z + (D_hjb @ t_p_z.T).T * ca.hcat(vel_z_l).T - (D_diff_z @ t_p_z.T).T ))**2 @ ones)/((fidelity-1)*(tot_points-1)) #Repurposed to HJB
    J_ff_p = i1 @ ((((t_Jsigma_ff + D @ t_p_ff + (D_hjb @ t_p_ff.T).T * ca.hcat(vel_ff_l).T - (D_diff_ff @ t_p_ff.T).T )))**2 @ ones)/((fidelity-1)*(tot_points-1)) #Repurposed to HJB


    J_z_v =  i1 @ ((t_dJv_z + (D_hjb @ t_p_z.T).T)**2 @ ones)/((fidelity-1)*(tot_points-1))
    J_ff_v = i1 @ ((t_dJv_ff + (D_hjb @ t_p_ff.T).T)**2 @ ones)/((fidelity-1)*(tot_points-1))

    #D_trans = bc_mat @ np.copy(Mx.D)
    trans_z = (D @ t_sigma_z + ((D_trans @ (ca.hcat(sigma_z_l) * ca.hcat(vel_z_l))).T + (D_diff_z @ ca.hcat(sigma_z_l)).T))**2/((fidelity-1)*(tot_points-1))
    trans_ff = (D @ t_sigma_ff + ((D_trans @ (ca.hcat(sigma_ff_l) * ca.hcat(vel_ff_l))).T + (D_diff_ff @ ca.hcat(sigma_ff_l)).T))**2/((fidelity-1)*(tot_points-1))
    #print((i1 @ M_per @ trans_ff).size())
    trans_eqs = i1 @ Mx.M @ ( (i1 @ M_per @ trans_z).T ) + i1 @ Mx.M @ ( (i1 @ M_per @ trans_ff).T)# i1 @ ( trans_z @ ones) + i1 @ (trans_ff @ ones) #i1 @ Mx.M @ ( (i1 @ M_per @ trans_z).T ) + i1 @ Mx.M @ ( (i1 @ M_per @ trans_ff).T)

    pop_dyn_eqs = i1 @ (M_per @ v_c(dyn_0))**2 + i1 @ (M_per @ v_c(dyn_1))**2 + i1 @ (M_per @ v_c(dyn_5))**2 + i1 @ (D @ v_c(s0_vec) - v_c(dyn_0))**2 + i1 @ (D @ v_c(s1_vec) - v_c(dyn_1))**2 + i1 @ (D @ v_c(s5_vec) - v_c(dyn_5))**2
    f = pop_dyn_eqs #J_z_p + J_ff_p

    x = ca.vertcat(*[*vel_z_l, *vel_ff_l, *p_z_l, *p_ff_l, *sigma_z_l, *sigma_ff_l, *state_l])

    g = ca.vertcat(*[*prob_l, trans_eqs, J_z_p, J_ff_p, J_z_v, J_ff_v])#, ca.reshape(J_z_p, (-1,1)), ca.reshape(J_ff_p, (-1,1)), ca.reshape(J_ff_p, -1,1), ca.reshape(J_z_v, (-1,1)), ca.reshape(J_ff_v, (-1,1))])#, ca.reshape(trans_z,  (-1,1)), ca.reshape(trans_ff, (-1,1))])
    probs = 2*fidelity
    lbg = np.concatenate([np.zeros(probs+5)])#, np.repeat(-10**(-6), g.size()[0] - probs)])
    #upper_zeros = 2*fidelity+2*fidelity*tot_points
    ubg = np.concatenate([np.zeros(probs), 10**(-7)*np.ones(5)])#, np.repeat(10**(-6), g.size()[0] - probs)])
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
        x_init = 0*np.ones(x.size()[0])/Mx.x[-1]#np.concatenate([np.zeros(4*fidelity), np.ones(4*fidelity*tot_points)/Mx.x[-1], np.ones(6*fidelity)])
        s_opts = {'ipopt': {'print_level': 3, 'linear_solver': 'ma57', 'max_iter':10000}}
                            #'tol':10**(-5)}}#, 'hessian_approximation': 'limited-memory'}}  #
    else:
        s_opts = {'ipopt': {'print_level': 3, 'linear_solver': linsol, 'max_iter':35000, 'tol':tol, #-6 appears to be the lowest possible value for reasonable speed.
                               'hessian_approximation': 'limited-memory',
                               'warm_start_init_point': 'yes', 'warm_start_bound_push': warmstart_opts,
                               'warm_start_bound_frac': warmstart_opts, 'warm_start_slack_bound_frac': warmstart_opts,
                              'warm_start_slack_bound_push': warmstart_opts, 'warm_start_mult_bound_push': warmstart_opts, 'limited_memory_max_history': 10, 'limited_memory_initialization':'scalar2'}} #Remark that -3 works well, esp. with -5 above.

#Scalar2 works best with differential formulation of population dynamics
#Scalar4 works best with integral formulation
#The reaosn? Not known.
    #Scalar1 is the default anyway
    #The standard tolerance is e-8, but we have lowered it for speed reasons.
#Scalar 4 is a combination of scalar1 and ascalar2


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

from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp2d

from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline

def increase_resolution(ir_t = 20, fr_t = 50, ir_s = 20, fr_s = 50, jumpsize_t = 5, jumpsize_s = 5, save = True, R_max = 5):
    x_vals = np.linspace(0,depth,ir_s)
    t_vals = np.linspace(0, 24, ir_t + 1)[:-1]

        #np.linspace(0, 1, ir + 1)[:-1]
    results = []
    results.append(output(tot_points = ir_s, fidelity = ir_t, R_max=R_max))
    decision_vars = []
    state_vars = []

    mult_dec_var = []
    mult_stat_var = []
    def rs(x, y, z = None):
        if z is None:
            return x.reshape((y,y))
        else:
            return x.reshape((y,z))

    for k in range(6):
        decision_vars.append(interp2d(x_vals, t_vals, rs(results[0]['x0'][k*ir_s*ir_t : (k+1)*ir_s*ir_t], ir_s, ir_t), kind = 'linear'))
        mult_dec_var.append(interp2d(x_vals, t_vals, rs(results[0]['lam_x0'][k*ir_s*ir_t : (k+1)*ir_s*ir_t], ir_s, ir_t), kind = 'linear'))
    offset = 6*ir_s*ir_t

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
    mult_ineq.append(results[0]['lam_g0'][-5:])
    #offset = 2*ir_t
    #for k in range(5):
    #    mult_ineq.append(interp2d(x_vals, t_vals, rs(results[0]['lam_g0'][offset + k*ir**2 : offset + (k+1)*ir**2], ir)))

    counter = 0
    x0_j = []
    lam_x0_j = []
    lam_g0_j = []
    for k in range(6):
        x0_j.append(decision_vars[k](x_vals, t_vals).flatten())
        lam_x0_j.append(mult_dec_var[k](x_vals, t_vals).flatten())


    x0_j_state = np.zeros(3 * ir_t)
    lam_x0_j_state = np.zeros(3 * ir_t)
    for k in range(3):
        x0_j_state[k::3] = (state_vars[k](t_vals))
        lam_x0_j_state[k::3] = (mult_stat_var[k](t_vals))


    for k in range(2):
        lam_g0_j.append(mult_ineq[k](t_vals))
    lam_g0_j.append(mult_ineq[-1])
    #for k in range(5):
    #    lam_g0_j.append(mult_ineq[2+k](x_vals, t_vals).flatten())

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
        for k in range(6):
            x0_j.append(decision_vars[k](x_vals, t_vals,).flatten())
            lam_x0_j.append(mult_dec_var[k](x_vals, t_vals,).flatten())

        x0_j_state = np.zeros(3 * j_t)
        lam_x0_j_state = np.zeros(3 * j_t)
        for k in range(3):
            x0_j_state[k::3] = (state_vars[k](t_vals))
            lam_x0_j_state[k::3] = (mult_stat_var[k](t_vals))

        for k in range(2):
            lam_g0_j.append(mult_ineq[k](t_vals))
        lam_g0_j.append(mult_ineq[-1])
        #for k in range(5):
        #    lam_g0_j.append(mult_ineq[2+k](x_vals, t_vals).flatten())

        warmstart_inf = {'x0': np.concatenate([*x0_j, x0_j_state]), 'lam_x0': np.concatenate([*lam_x0_j, lam_x0_j_state]), 'lam_g0': np.concatenate([*lam_g0_j])}
        results.append(output(tot_points = j_s, fidelity = j_t, warmstart_info=warmstart_inf, R_max=R_max))

        decision_vars = []
        state_vars = []

        mult_dec_var = []
        mult_stat_var = []
        for k in range(6):
            decision_vars.append(interp2d(x_vals, t_vals,  rs(results[counter]['x0'][k * j_s*j_t: (k + 1) * j_s*j_t], j_s, j_t), kind = 'linear'))
            mult_dec_var.append(interp2d(x_vals, t_vals,  rs(results[counter]['lam_x0'][k * j_s*j_t: (k + 1) * j_s*j_t], j_s, j_t), kind = 'linear'))
        offset = 6 * j_s*j_t

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
        mult_ineq.append(results[counter]['lam_g0'][-5:])
    if save is True:
        with open('data/' + 'pmp2_'+str(fr_t*fr_s)+'_' + str(R_max) + '.pkl', 'wb') as f:
            pkl.dump(results, f, pkl.HIGHEST_PROTOCOL)

    return results[-1]
#increase_resolution(ir = 10, fr = 71, jumpsize = 15)

def vary_resources(start = 5, stop = 50, steps = 45, ir_t = 10, fr_t = 70, ir_s = 10, fr_s = 70):
    res_var_list = []
    print("Starting resource variation")
    resources = np.linspace(start, stop, steps)
    res_var_list.append(increase_resolution(ir_t=ir_t, ir_s=ir_s, fr_t=fr_t, fr_s=fr_s, jumpsize_s=10, jumpsize_t=10, R_max=resources[0]))
    print(resources)
    for i in range(1,steps):
        res_var_list.append(
            output(fidelity=fr_t, tot_points=fr_s, warmstart_info=res_var_list[-1],  R_max=resources[i], warmstart_opts=1e-3))
       # res_var_list.append(
       #     increase_resolution(ir_t=ir_t, ir_s=ir_s, fr_t=fr_t, fr_s=fr_s, jumpsize_s=10, jumpsize_t=10,  R_max=resources[i]))

        print((i+1)/steps, " Completion ratio")
    results_and_params = {'gamma0': gamma0, 'gamma1': gamma1, 'simulations':res_var_list}
    with open('data/' + 'pdeco2_res_'+str(fr_s*fr_t)+'_'+str(stop) + '.pkl', 'wb') as f:
        pkl.dump(results_and_params, f, pkl.HIGHEST_PROTOCOL)


vary_resources(steps = 46, ir_t=10, ir_s = 10, fr_s=50, fr_t=50, start=5, stop = 50)