


from four_spec_sim import *
import pickle as pkl
from scipy.signal import savgol_filter
import numpy as np
t_s = 60

h = 20 / (365*24*t_s)
a = 0.4
m0 = 10 ** (-3)
gamma = 1
k = 0.05
masses = np.array([0.05, 20])

f_c = 0  # 0.15 / (365*24)
r = 1 / (365*24*t_s)
r_b = 1 / (365*24*t_s)
eps0 = 0.05
#comp = 0
Cmax = h * masses ** (0.75)#*50
metabolism = 0.1 * h *masses**(0.75)
print(metabolism)
epsi = eps0 * ((1 - a) * np.log(masses / m0) - np.log(eps0))

diffusivity_ff = 1/t_s #0.25 #0.3 #10**(-3) #0.3*10**(-4)
diffusivity_z = 1/t_s #0.25 #0.3 #10**(-3) #0.3*10**(-4)

depth = 100
gamma0 = 1e-4 #1e-9 in seconds #1e-4  # 1e-9*(310**2) REMARK 1e-4 and 1e-3 work remarkably well!!! Now shifting to realistic units...
gamma1 = 1e-3 #1e-8

#We can also use wwarmstart?opts 1e-6 if we change the gammes to realistic values
def output(tot_points = 5, fidelity = 20, Rmax = 5, Bmax = 0.1, warmstart_info = None, warmstart_opts = 1e-6, scalar = 'scalar2', hessian_approximation = True, tol = 1e-8): #scalar4 works best based on testing. Why? Noone knows.
    Mx = simple_method(depth, tot_points, central = True)
    #must be even
    inte = np.ones(tot_points).reshape(1, tot_points)
    D = fin_diff_mat_periodic(fidelity,length = 24*60, central = True)
    #D = spectral_periodic(fidelity, length=24*t_s)

    M_per = M_per_calc(fidelity, length = 24*t_s)
    rz = 1/(1+np.exp(0.2*(Mx.x - 20)))
    rz = rz/(inte @ Mx.M @ rz)


    upright_wc = np.exp(-k * Mx.x)  # .reshape((-1, 1))
    c_z = Cmax[0]
    c_ff = Cmax[1]
    bg_M = 0 #0.1 / (365*24)
    beta_0 = 5*10**(-2) #5*10 ** (-3)
    A = np.cos(2*np.pi*np.linspace(0,1,5*fidelity+5)[0:-5])*1/np.sqrt(2)
    A[A < -1 / 2] = -1 / 2
    A[A > 1 / 2] = 1 / 2
    A = A+1/2
    smoothed_A = savgol_filter(A, window_length = 5, polyorder = 3)
    light_levels = smoothed_A[0::5]
    #light_levels = (np.hstack([np.tanh(5 * np.linspace(-1, 1, int(np.ceil(fidelity / 2))))[::-1], np.tanh(5 * np.linspace(-1, 1, int(np.floor(fidelity/ 2))))])+1)/2
    #print(gamma0, gamma1)
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

    growth_c = []
    death_c =[]
    dyn_0 = []
    dyn_1 = []
    dyn_5 = []

    s0_vec = []
    s1_vec = []
    s5_vec = []

    one_col = np.ones((tot_points,1))
    for j in range(fidelity):
        Vi = light_levels[j]
        beta_i = 330 / (365*24*t_s) * masses ** (0.75)
        #print(beta_i)
        beta_z.append( 330 / (365*24*t_s) * one_col * masses[0] ** (0.75))
        beta_ff.append(2 * beta_i[1] * (Vi*upright_wc / (1 + upright_wc) ) + beta_i[1] *beta_0) #+ *330/365*gamma*11**(0.75))



        p_z_l.append(ca.MX.sym('p_z_'+str(j), tot_points))
        p_ff_l.append(ca.MX.sym('p_ff_'+str(j), tot_points))

        sigma_z_l.append(ca.MX.sym('sigma_z_l_'+str(j), tot_points))
        sigma_ff_l.append(ca.MX.sym('sigma_ff_l_'+str(j), tot_points))


        vel_z_l.append(ca.MX.sym('vel_l_'+str(j), tot_points))
        vel_ff_l.append(ca.MX.sym('vel_ff_l_'+str(j), tot_points))

        state_l.append(ca.MX.sym('state_'+str(j), 3))

        ff_z_enc.append(beta_ff[j] * sigma_z_l[j] * sigma_ff_l[j])
        ff_satiation.append(1 / (state_l[j][0] * ff_z_enc[j] + c_ff))


        dz_sigma_z_l.append(Mx.D @ sigma_z_l[j])
        dz_sigma_ff_l.append(Mx.D @ sigma_ff_l[j])

        dz_vel_z_l.append(Mx.D @ vel_z_l[j])
        dz_vel_ff_l.append(Mx.D @ vel_ff_l[j])


        Jsig_z.append(epsi[0] * state_l[j][-1] * c_z*beta_z[j] * rz/(c_z+state_l[j][-1] * beta_z[j] * rz) - state_l[j][1] * (
                    c_ff * sigma_ff_l[j] * beta_ff[j] * ff_satiation[j]) - gamma0/2*vel_z_l[j]**2)

        Jsig_ff.append((epsi[1] * state_l[j][0] * c_ff * beta_ff[j] * sigma_z_l[j]/(c_ff + beta_ff[j]*state_l[j][0]*sigma_z_l[j])) - gamma1/2*vel_ff_l[j]**2)

        dJv_z.append(-gamma0*vel_z_l[j])
        dJv_ff.append(-gamma1*vel_ff_l[j])


        prob_l.append(inte @ Mx.M @ sigma_z_l[j] - 1)
        prob_l.append(inte @ Mx.M @ sigma_ff_l[j] - 1)


        s0_vec.append(state_l[j][0])
        s1_vec.append(state_l[j][1])
        s5_vec.append(state_l[j][2])
        growth_c.append((inte @ Mx.M @ (epsi[0] * state_l[j][-1] * sigma_z_l[j]*c_z*beta_z[j] * rz/(c_z+state_l[j][-1] * sigma_z_l[j] * beta_z[j] * rz))))
        death_c.append((state_l[j][1] * (inte @ (Mx.M @ (c_ff * ff_z_enc[j] * ff_satiation[j]))) - metabolism[0] - gamma0/2*inte @ Mx.M @ (sigma_z_l[j]*vel_z_l[j]**2)))

        dyn_0.append(state_l[j][0] * (inte @ Mx.M @ (epsi[0] * state_l[j][-1] * sigma_z_l[j]*c_z*beta_z[j] * rz/(c_z+state_l[j][-1] * sigma_z_l[j] * beta_z[j] * rz))
                    - state_l[j][1] * (inte @ (Mx.M @ (c_ff * ff_z_enc[j] * ff_satiation[j]))) - metabolism[0] - gamma0/2*inte @ Mx.M @ (sigma_z_l[j]*vel_z_l[j]**2)))
        dyn_1.append(state_l[j][1] * (epsi[1] * (c_ff * state_l[j][0] * inte @ (Mx.M @ (ff_z_enc[j] * ff_satiation[j])) - f_c) - bg_M - metabolism[1] - gamma1/2*inte @ Mx.M @ (sigma_ff_l[j]*vel_ff_l[j]**2)))

        dyn_5.append(r * (Rmax - state_l[j][-1]) - state_l[j][-1]* state_l[j][0] * inte @ Mx.M @ (sigma_z_l[j]*c_z*beta_z[j] * rz/(c_z+state_l[j][-1] * sigma_z_l[j] * beta_z[j] * rz)))


    v_c = lambda x: ca.vertcat(*x)
    i1 = np.ones(tot_points).reshape(tot_points, 1)
    #Total time dynamics
    t_p_z = ca.hcat(p_z_l).T
    t_p_ff = ca.hcat(p_ff_l).T


    t_sigma_z = ca.hcat(sigma_z_l).T
    t_sigma_ff = ca.hcat(sigma_ff_l).T


    #Lagrangian total dynamics
    t_Jsigma_z = ca.hcat(Jsig_z).T
    t_Jsigma_ff = ca.hcat(Jsig_ff).T

    t_dJv_z = ca.hcat(dJv_z).T
    t_dJv_ff = ca.hcat(dJv_ff).T

    bc_mat = np.identity(tot_points)
    bc_mat[-1, -1] = 0
    bc_mat[0,0] = 0
    ones = np.ones((1, fidelity))

    D_trans, D_diff_z = transport_matrix(depth, tot_points, diffusivity = diffusivity_z, central = True)
    D_trans, D_diff_ff = transport_matrix(depth, tot_points, diffusivity = diffusivity_ff, central = True)

    D_hjb = D_trans#.T# D_HJB(depth = depth, total_points=tot_points) #This, or -D_trans.T, unclear which. D_trans should also work.
    J_z_p = ones @ M_per @ (i1.T @ Mx.M @ ((t_Jsigma_z + D @ t_p_z + (D_hjb @ t_p_z.T).T * ca.hcat(vel_z_l).T - (D_diff_z @ t_p_z.T).T )**2).T).T  #@ i1/(tot_points-1) #((fidelity - 1) * (tot_points - 1)) #Repurposed to HJB
    J_z_v =  ones @ M_per @ (i1.T @ Mx.M @ ((t_dJv_z + (D_hjb @ t_p_z.T).T)**2 ).T).T #@ i1/(tot_points-1)#/((fidelity-1)*(tot_points-1))
    #Mx.M @
    #Mx.M @
    J_ff_p =  ones @ M_per @ (i1.T @ Mx.M @ ((t_Jsigma_ff + D @ t_p_ff + (D_hjb @ t_p_ff.T).T * ca.hcat(vel_ff_l).T - (D_diff_ff @ t_p_ff.T).T )**2).T).T  #@ i1/(tot_points-1) #/((fidelity-1)*(tot_points-1)) #Repurposed to HJB
    J_ff_v = ones @ M_per @ (i1.T @ Mx.M @ ((t_dJv_ff + (D_hjb @ t_p_ff.T).T)**2 ).T).T #@ i1/(tot_points-1) #((fidelity-1)*(tot_points-1))
#The best results were actually achieved by having the dumbest possible integrals here, i.e. removing Mx.M and dividing by the total number of points, and a tolerance of 10^-8. Having precise integrals probably introduces too tight a coupling for the numerical scheme.
    # Remark the smart integarls should also be removed from the transport equation, but then it should be normalized so the factor of 100 is still included.
    trans_z = (D @ t_sigma_z + (D_trans @ ( ca.hcat(sigma_z_l) * ca.hcat(vel_z_l))).T + (D_diff_z @  ca.hcat(sigma_z_l)).T)**2#/((fidelity-1)*(tot_points-1))
    trans_ff = (D @ t_sigma_ff + (D_trans @ ( ca.hcat(sigma_ff_l) * ca.hcat(vel_ff_l))).T + (D_diff_ff @  ca.hcat(sigma_ff_l)).T)**2#/((fidelity-1)*(tot_points-1))
    #print(( ( Mx.M @ (ones @ (M_per @ trans_z)).T).T @ i1).size())
    trans_z_t = i1.T @ Mx.M @ (ones @ (M_per @ trans_z)).T #@ ones/(tot_points-1)
    trans_ff_t = i1.T @ Mx.M @ (ones @ (M_per @ trans_ff)).T# @ ones/(tot_points-1)

    fp_z = ones @ (M_per @ v_c(dyn_0))**2
    fp_ff = ones @ (M_per @ v_c(dyn_0))**2
    fp_r = ones @ (M_per @ v_c(dyn_5))**2

    D_ex = D # fin_diff_mat_periodic(fidelity,length = 24*60, central = True)
    #spectral_periodic(fidelity)
    p_eq_z = ones  @ (D_ex @ v_c(s0_vec) - v_c(dyn_0))**2
    p_eq_ff = ones  @ (D_ex @ v_c(s1_vec) - v_c(dyn_1))**2
    p_eq_r = ones  @ (D_ex @ v_c(s5_vec) - v_c(dyn_5))**2
    pop_dyn_eqs = p_eq_z + p_eq_ff + p_eq_r
    force_periodic= fp_r+fp_z+fp_ff

    f = pop_dyn_eqs #pop_dyn_eqs # ((ones @ M_per @ (v_c(death_c)))/(ones @ M_per @ v_c(growth_c))-1)**2 #fp_z+fp_ff+fp_r #pop_dyn_eqs #fp_z+fp_ff+fp_r #0 #pop_dyn_eqs#

    x = ca.vertcat(*[*vel_z_l, *vel_ff_l, *p_z_l, *p_ff_l, *sigma_z_l, *sigma_ff_l, *state_l])

    g = ca.vertcat(*[*prob_l, trans_z_t, trans_ff_t, pop_dyn_eqs, force_periodic, 0, 0, J_z_v , J_ff_v, J_z_p, J_ff_p])#, ca.reshape(J_z_p, (-1,1)), ca.reshape(J_ff_p, (-1,1)), ca.reshape(J_ff_p, -1,1), ca.reshape(J_z_v, (-1,1)), ca.reshape(J_ff_v, (-1,1))])#, ca.reshape(trans_z,  (-1,1)), ca.reshape(trans_ff, (-1,1))])
    probs = 2*fidelity
    lbg = np.concatenate([np.zeros(probs+10)])#, np.repeat(-10**(-6), g.size()[0] - probs)])
    #upper_zeros = 2*fidelity+2*fidelity*tot_points
    #ubg = np.concatenate([np.zeros(probs), 1e-8*np.ones(6), 1e-7*np.ones(4)])#, np.repeat(10**(-6), g.size()[0] - probs)])
    ubg = np.concatenate([np.zeros(probs), 1e-8*np.ones(2), 1e-8*np.ones(4), 1e-6*np.ones(4)])#, np.repeat(10**(-6), g.size()[0] - probs)])

    #np.zeros(g.size()[0])#ca.vertcat(*[*np.zeros(upper_zeros)])#, (g.size()[0]-(upper_zeros))*[ca.inf]])
    lbx = ca.vertcat(fidelity*4*tot_points*[-ca.inf], np.zeros(tot_points*fidelity*2), np.ones(fidelity*3)*10**(-5))
    ubx = ca.vertcat(*[[ca.inf]*(x.size()[0]-3*fidelity), np.repeat(Rmax, 3*fidelity)])
    prob = {'x': x, 'f': f, 'g': g}

    #s_opts = {'ipopt':{'print_level': 5, 'linear_solver': 'ma57', 'fixed_variable_treatment': "make_constraint"}}  #
    if x.size()[0]<15000:
        linsol = 'ma57'
    else:
        linsol = 'ma57' #'ma86'
    if hessian_approximation is True:
        hess = 'limited-memory'
    else:
        hess = 'exact'
    if warmstart_info is None:
        x_init = np.ones(x.size()[0])/Mx.x[-1]#np.concatenate([np.zeros(4*fidelity), np.ones(4*fidelity*tot_points)/Mx.x[-1], np.ones(6*fidelity)])
        s_opts = {'ipopt': {'print_level': 3, 'linear_solver': 'ma57', 'max_iter':1000}}
                            #'tol':10**(-5)}}#, 'hessian_approximation': 'limited-memory'}}  #
    else:
        s_opts = {'ipopt': {'print_level': 3, 'linear_solver': linsol, 'max_iter':3000, 'tol': tol,
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
from scipy.ndimage import gaussian_filter
def increase_resolution(ir_t = 20, fr_t = 50, ir_s = 20, fr_s = 50, jumpsize_t = 5, jumpsize_s = 5, save = True, Rmax = 5, warmstart_info = None, warmstart_opts = 1e-3, lockstep = True):
    x_vals = np.linspace(0,depth,ir_s)
    t_vals = np.linspace(0, 24*t_s, ir_t + 1)[:-1]
    def rs(x, y, z = None):
        if z is None:
            return x.reshape((y,y))
        else:
            return x.reshape((y,z))

    counter = 0
    j_t_l = np.array(range(ir_t, fr_t+1, jumpsize_t))
    j_s_l = np.array(range(ir_s, fr_s+1, jumpsize_s))

    j_l = []
    if lockstep is True: #This allows us to refine time then space
        for k in range(max(len(j_s_l), len(j_t_l))):
            j_l.append([j_s_l[min(k, len(j_s_l)-1)], j_t_l[min(k, len(j_t_l)-1)]])
    else:
        tick = True
        j_l.append([j_s_l[0], j_t_l[0]])
        #print("I get in here")
        if fr_t>fr_s:
            for k in range(1, 2*max(len(j_s_l), len(j_t_l))):
                if tick is True:
                    j_l.append([j_s_l[int((k-1)/2)], j_l[k-1][1]])
                    tick = False
                else:
                    j_l.append([j_l[k-1][0], j_t_l[int((k)/2)]])
                    tick = True
        else:
            print("fr_s>fr_t")
            for k in range(1, 2*max(len(j_s_l), len(j_t_l))):
                if tick is False:
                    j_l.append([j_s_l[int((k)/2)], j_l[k-1][1]])
                    tick = True
                else:
                    j_l.append([j_l[k-1][0], j_t_l[int((k-1)/2)]])
                    tick = False

        #np.linspace(0, 1, ir + 1)[:-1]
    print(j_l)
    results = []
    for j in range(0, len(j_l)):
        print("Current step: ", counter)
        j_t = j_l[j][1]
        j_s = j_l[j][0]
        if counter is 0:
            if warmstart_info is None:
                w_s_inf = None
            else:
                w_s_inf=warmstart_info[counter]
            results.append(output(tot_points = ir_s, fidelity = ir_t, Rmax=Rmax, warmstart_info =w_s_inf, warmstart_opts=warmstart_opts, hessian_approximation=False, tol = 1e-8))
            carryover_trans = np.array([results[counter]['lam_g0'][-10:]]).squeeze()

        if counter > 0:
            x_vals = np.linspace(0, depth, j_s)
            t_vals = np.linspace(0, 24 * t_s, j_t + 1)[:-1]

            x0_j = []
            lam_x0_j = []
            lam_g0_j = []

            for k in range(6):
                x0_j.append((decision_vars[k](x_vals, t_vals)).flatten())
                lam_x0_j.append((mult_dec_var[k](x_vals, t_vals)).flatten())
                if k is 0:
                    print(t_s*np.max(x0_j[k]))

            x0_j_state = np.zeros(3 * j_t)
            lam_x0_j_state = np.zeros(3 * j_t)
            for k in range(3):
                x0_j_state[k::3] = (state_vars[k](t_vals))
                lam_x0_j_state[k::3] = (mult_stat_var[k](t_vals))

            for k in range(2):
                lam_g0_j.append(mult_ineq[k](t_vals))
            lam_g0_j.append(carryover_trans)
            warmstart_inf = {'x0': np.concatenate([*x0_j, x0_j_state]), 'lam_x0': np.concatenate([*lam_x0_j, lam_x0_j_state]), 'lam_g0': np.concatenate([*lam_g0_j])}
            #if warmstart_info is None:
            w_s_inf = warmstart_inf
            #else:
            #    w_s_inf = {'x0': (0.9*warmstart_inf['x0']+0.1*warmstart_info[counter]['x0']), 'lam_x0':  (0.9*warmstart_inf['lam_x0']+0.1*warmstart_info[counter]['lam_x0']), 'lam_g0':  (0.9*warmstart_inf['lam_g0']+0.1*warmstart_info[counter]['lam_g0'])}


            results.append(output(tot_points = j_s, fidelity = j_t, warmstart_info=w_s_inf, Rmax=Rmax))
            carryover_trans = np.array([results[counter]['lam_g0'][-10:]]).squeeze()
        decision_vars = []
        state_vars = []

        mult_dec_var = []
        mult_stat_var = []
        for k in range(6):
            decision_vars.append(interp2d(x_vals, t_vals,  rs(results[counter]['x0'][k * j_s*j_t: (k + 1) * j_s*j_t], j_t, j_s), kind = 'linear'))
            mult_dec_var.append(interp2d(x_vals, t_vals, rs(results[counter]['lam_x0'][k * j_s*j_t: (k + 1) * j_s*j_t], j_t, j_s), kind = 'linear'))
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
        #if counter is 1:
        #    print("Testing info:", "\n", np.linalg.norm(np.concatenate([*x0_j, x0_j_state])-results[0]['x0']),  "\n",  np.linalg.norm(np.concatenate([*lam_x0_j, lam_x0_j_state])-results[0]['lam_x0']),  "\n", np.linalg.norm(results[0]['lam_g0'] - np.concatenate([*lam_g0_j])))
        counter+=1


    if save is True:
        with open('data/' + 'pdeco2_'+str(fr_t*fr_s)+'_' + str(np.round(Rmax)) + '.pkl', 'wb') as f:
            pkl.dump(results, f, pkl.HIGHEST_PROTOCOL)

    return results


def vary_resources_grid_all(start = 5, stop = 50, steps = 45, ir_t = 10, fr_t = 70, ir_s = 10, fr_s = 70, jumpsize = 1):
    grid_variation = []
    if ir_t > ir_s:
        jump_scale_t = int(ir_t/ir_s)
        jump_scale_s = 1
    else:
        jump_scale_s = int(ir_s / ir_t)
        jump_scale_t = 1
    print("Starting resource variation")
    print(jump_scale_s, jump_scale_s)
    resources = np.linspace(start, max(stop,start), steps)
    grid_variation.append(increase_resolution(ir_t=ir_t, ir_s=ir_s, fr_t=fr_t, fr_s=fr_s, jumpsize_s=jumpsize*jump_scale_s, jumpsize_t=jump_scale_t*jumpsize, Rmax=resources[0], lockstep=True)) #Changing to false stablizies, but doubles the number of iterations. Hence True is ideal
    print(resources)
    for i in range(1,steps):
        grid_variation.append(
            increase_resolution(ir_t=ir_t, ir_s=ir_s, fr_t=fr_t, fr_s=fr_s, jumpsize_s = jumpsize*jump_scale_s, jumpsize_t = jump_scale_t*jumpsize,
                                Rmax=resources[i], warmstart_info=grid_variation[-1], warmstart_opts = 1e-3, lockstep=True))
        #res_var_list.append(
        #    increase_resolution(ir_t=ir_t, ir_s=ir_s, fr_t=fr_t, fr_s=fr_s, jumpsize_s=10, jumpsize_t=10,  Rmax=resources[i]))

        print((i+1)/steps, " Completion ratio")
    results_and_params = {'depth': depth, 'gamma0': gamma0, 'gamma1': gamma1, 'simulations':grid_variation}
    with open('data/' + 'pdeco2_res_'+str(fr_s*fr_t)+'_'+str(stop) + '.pkl', 'wb') as f:
        pkl.dump(results_and_params, f, pkl.HIGHEST_PROTOCOL)

#    with open('data/' + 'pdeco4_res_'+str(jumpsize) + '.pkl', 'wb') as f:
#    pkl.dump(grid_variation, f, pkl.HIGHEST_PROTOCOL)

def vary_resources(start = 5, stop = 50, steps = 45, ir_t = 10, fr_t = 70, ir_s = 10, fr_s = 70, jumpsize = 1):
    grid_variation = []
    if ir_t > ir_s:
        jump_scale_t = int(ir_t/ir_s)
        jump_scale_s = 1
    else:
        jump_scale_s = int(ir_s / ir_t)
        jump_scale_t = 1
    print("Starting resource variation")
    print(jump_scale_s, jump_scale_s)
    resources = np.linspace(start, max(stop,start), steps)
    grid_variation.append(increase_resolution(ir_t=ir_t, ir_s=ir_s, fr_t=fr_t, fr_s=fr_s, jumpsize_s=jumpsize*jump_scale_s, jumpsize_t=jump_scale_t*jumpsize, Rmax=resources[0], lockstep=True)[-1])
    print(resources)
    for i in range(1,steps):
        grid_variation.append(
            output(fidelity=fr_t, tot_points=fr_s, warmstart_info=grid_variation[-1],  Rmax=resources[i], warmstart_opts=1e-1))
       # res_var_list.append(
       #     increase_resolution(ir_t=ir_t, ir_s=ir_s, fr_t=fr_t, fr_s=fr_s, jumpsize_s=10, jumpsize_t=10,  R_max=resources[i]))

        print((i+1)/steps, " Completion ratio")
    results_and_params = {'gamma0': gamma0, 'gamma1': gamma1, 'simulations':grid_variation}
    with open('data/' + 'dvm_var'+str(fr_s*fr_t)+'_'+str(stop) + '.pkl', 'wb') as f:
        pkl.dump(results_and_params, f, pkl.HIGHEST_PROTOCOL)

vary_resources_grid_all(steps = 15, ir_s = 4, ir_t = 12, fr_s=30, fr_t=int(3*30), jumpsize=1, start = 5, stop = 30)

#REMEMBER! THE BEST RESULT WAS CREATED WITH A JUMPSIZE OF 1 AND 46 STEPS and central is True
#WORS ALSO WITH JUMPSIZE 2 AND 23 STEPS (simple method)
#NOW TESTING: JS 1 23 steps

