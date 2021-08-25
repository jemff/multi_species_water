#class ecosystem:
#    def __init__:
import casadi as ca
import numpy as np

def threed_predator_prey_dyn(resources = None, beta_f = None, res_conc_f = None, minimal_pops = 10**(-5), fixed_point = False, pops = np.array([1,1]), Mx = None, warmstart_info = None, warmstart_out = False, par = None, car_cap = 1):

    tot_points = Mx.x.size
    inte = np.ones(tot_points).reshape(1,tot_points)

    tot_cont_p = tot_points-1
    tot_times = 3

    lam = ca.MX.sym('lam', (tot_times, 5))

    sigma_z = ca.MX.sym('sigma_z', (tot_times, tot_cont_p))
    sigma_ff = ca.MX.sym('sigma_ff', (tot_times, tot_cont_p))
    sigma_lp = ca.MX.sym('sigma_ff', (tot_times, tot_cont_p))
    sigma_ld = ca.MX.sym('sigma_ff', (tot_times, tot_cont_p))


    state = ca.MX.sym('state', 5)
    res_level = ca.MX.sym('res_level', tot_cont_p) #Resource dynamics
    res_dyn_one = par['res_renew']*(res_conc*car_cap - res_level) - state[0]*res_level*sigma_z/(par['c_handle']*inte @ (Mx.M @ (res_level*sigma_z)) + par['c_enc_freq'])

    D_2 = (Mx.D @ Mx.D)
    D_2[0] = np.copy(Mx.D[0])
    D_2[-1] = np.copy(Mx.D[-1])
    almost_ID = np.identity(tot_points)
    almost_ID[0,0] = 0
    almost_ID[-1,-1] = 0
    pde_form = D_2 @ res_level + almost_ID @ res_dyn_one
    res_dyn = inte @ (Mx.M @ (pde_form * pde_form)) / (Mx.x[-2] ** 2)

    who_eats_who = np.zeros((4,4)) #z, FF, lp, ld
    who_eats_who[1,0] = 1 #FF Eat z
    who_eats_who[2,1] = 1 #LP eat FF
    who_eats_who[3,1] = 1 #LD eat FF


    benthos = ca.MX.sym('benthos', 1)
    #Bentic dynamics
    bent_dyn = lam_b*(state[4]-R_b)-0.5*state[-1]*sigma_ld[-1]*ld_satiation(state, sigma_ff, sigma_ld)
    z_dyn =  state[0]*res_level*sigma_z/(par['c_handle']*inte @ (Mx.M @ (res_level*sigma_z)) + par['c_enc_freq']) - state[1]*ff_satiation
    ff_dyn = state[1]*(state[0]*ff_satiation - state[2] * lp_satiation - state[3] * ld_satiation)
    lp_dyn = state[2]*(state[1]*lp_satiation - lp_loss)
    ld_dyn = state[3]*()



    for k in range(self.parameters.who_eats_who.shape[0]):
        interaction_term = self.parameters.who_eats_who[k] * temp_pops * self.parameters.clearance_rate[k]

        lin_g_others = np.dot((x_temp * self.parameters.layered_attack[:, k, :].T @ self.spectral.M @ x_temp[k]),
                              interaction_term)
        foraging_term = self.water.res_counts * self.parameters.forager_or_not[k] \
                        * self.parameters.clearance_rate[k] * self.parameters.layered_foraging[:, k] * \
                        x_temp[k]

        predation_ext_food[k] = np.sum(np.dot(self.spectral.M, foraging_term)) + lin_g_others
        predator_hunger[k] = self.parameters.clearance_rate[k] * np.dot(self.spectral.M,
                                                                        self.parameters.layered_attack[:, k, i] *
                                                                        x_temp[k]) * self.parameters.who_eats_who[k, i]

    actual_growth = self.parameters.efficiency * (lin_growth(x) + foraging_term_self(x)) \
                    / (1 + self.parameters.handling_times[i] * (lin_growth(x) + foraging_term_self(
        x))) - self.movement_cost * time_step * cum_diff.T @ self.spectral.M @ cum_diff

    pred_loss = ca.dot((predator_hunger @ (x.T @ self.heat_kernels[i]).T / (
                1 + self.parameters.handling_times.reshape((self.populations.shape[0], 1)) * (
                    self.populations[i] * predator_hunger @ (x.T @ self.heat_kernels[i]).T) + predation_ext_food)),
                       self.populations)  # ca.dot((x.T @ predator_hunger.T).T, self.populations)   This snippet is for the linear case, known good

    lin_growth = ca.Function('lin_growth', [x], [
        ca.dot(interaction_term, (x_temp * self.parameters.layered_attack[:, i, :].T) @ self.spectral.M @ x)])

    #Functions to implement:
    #   1) Total growth pr. animal.
    #   2)
    #
    cons_dyn = par['eff']*state[0]*inte @ (Mx.M @ (sigma*res_level))/(par['c_handle']*inte @ (Mx.M @ (sigma*res_level)) + par['c_enc_freq']) - inte @ (Mx.M @ (state_ss[0]*state_ss[1]*sigma*beta*sigma_p))/(par['p_enc_freq']+par['p_handle']*inte @ (Mx.M @ (state_ss[0]*sigma*beta*sigma_p))) - par['c_met_loss']*state_ss[0]
    pred_dyn = par['eff']*inte @ (Mx.M @ (state_ss[0]*state_ss[1]*sigma*beta*sigma_p))/(par['p_enc_freq']+par['p_handle']*inte @ (Mx.M @ (state_ss[0]*sigma*beta*sigma_p))) - par['competition']*inte @ (Mx.M @ (sigma_p**2*beta)) - par['p_met_loss']*state_ss[1]


    df1 = res_level * par['c_enc_freq']/(par['c_handle']*inte @ (Mx.M @ (sigma*res_level) + par['c_enc_freq']))**2-1/par['eff']*state_ss[1]*sigma_p*beta/(par['p_enc_freq']+par['p_handle']*inte @ (Mx.M @ (state_ss[0]*sigma*beta*sigma_p))) - lam[0]*np.ones(tot_points)
    df2 = state_ss[0]*par['p_enc_freq']*sigma*beta/(par['p_handle']*inte @ (Mx.M @ (state_ss[0]*sigma*beta*sigma_p))+par['p_enc_freq'])**2 - 1/par['eff']*par['competition']*sigma_p*beta - lam[1]*np.ones(tot_points)
    #df1 = res_level * par['c_enc_freq']/(par['c_handle']*sigma*res_level + par['c_enc_freq'])**2-1/par['eff']*state_ss[1]*sigma_p*beta/(par['p_enc_freq']+par['p_handle']*state_ss[0]*sigma*beta*sigma_p) - lam[0]*np.ones(tot_points)
    #df2 = state_ss[0]*par['p_enc_freq']*sigma*beta/(par['p_handle']*state_ss[0]*sigma*beta*sigma_p+par['p_enc_freq'])**2 - 1/par['eff']*par['competition']*sigma_p*beta - lam[1]*np.ones(tot_points)

    #g0 = ca.vertcat(cons_dyn, pred_dyn)
    g0 = ca.vertcat(cons_dyn, pred_dyn)
    g1 = inte @ Mx.M @ (df1*sigma) + inte @ Mx.M @ (df2*sigma_p)  #
    g2 = inte @ Mx.M @ sigma_p - 1
    g3 = inte @ Mx.M @ sigma - 1
    g4 = ca.vertcat(-df1, -df2)

    g = ca.vertcat(g0, g1, g2, g3, g4)

    #print(g0.size())
    f = res_dyn #ca.sin(res_dyn)**2 #ca.cosh(res_dyn)-1 #ca.cosh(res_dyn)-1 # ca.exp(res_dyn)-1 #ca.sqrt(res_dyn) #ca.cosh(res_dyn) - 1 #- 1 cosh most stable, then exp, then identity

    sigmas = ca.vertcat(sigma, sigma_p) #sigma_bar
    x = ca.vertcat(*[sigmas, res_level, state, lam])

    lbg = np.zeros(vars + 2*tot_points)
    ubg = ca.vertcat(*[np.zeros(vars), [ca.inf]*2*tot_points])
    if warmstart_info is None:
        s_opts = {'ipopt': {'print_level' : 3, 'linear_solver':'ma57',  'acceptable_iter': 15}} #'hessian_approximation':'limited-memory',
        init = np.ones(x.size()[0]) / np.max(Mx.x)
        init[-4:] = 1

    else:
        s_opts = {'ipopt': {'print_level': 3, 'linear_solver': 'ma57',
                            'acceptable_iter': 15, 'hessian_approximation':'limited-memory',
                            'warm_start_init_point': 'yes', 'warm_start_bound_push': 1e-9,
                            'warm_start_bound_frac': 1e-9, 'warm_start_slack_bound_frac': 1e-9,
                            'warm_start_slack_bound_push': 1e-9, 'warm_start_mult_bound_push': 1e-9}}

    prob = {'x': x, 'f': f, 'g': g}
    lbx = ca.vertcat(*[np.zeros(x.size()[0] - 2), -ca.inf, -ca.inf])
    solver = ca.nlpsol('solver', 'ipopt', prob, s_opts)

    if warmstart_info is None:
        sol = solver(lbx = lbx, lbg = lbg, ubg = ubg, x0 = init)

    else:
        sol = solver(lbx = lbx, lbg = lbg, ubg = ubg, x0 = warmstart_info['x0'], lam_g0 = warmstart_info['lam_g0'], lam_x0 = warmstart_info['lam_x0'] )


    if warmstart_out is False:
        return np.array(sol['x']).flatten()

    else:
        ret_dict = {'x0': np.array(sol['x']).flatten(), 'lam_g0': np.array(sol['lam_g']).flatten(), 'lam_x0': np.array(sol['lam_x']).flatten()}
        return ret_dict
