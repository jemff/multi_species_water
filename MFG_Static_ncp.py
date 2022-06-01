import casadi as ca
import numpy as np
from infrastructure import *
import siconos.numerics as sn

def mcp_function(n, z, F):
    Mx = discrete_patches(1,60)
    tot_points = Mx.x.size
    inte = np.ones(tot_points).reshape(1,tot_points)

    par = {'res_renew': 1, 'eff': 0.1, 'c_handle': 1, 'c_enc_freq': 1, 'c_met_loss': 0.001, 'p_handle': 0.01, 'p_enc_freq': 0.1, 'p_met_loss': 0.15, 'competition': 0.1, 'q': 3}
    car_cap = 3

    res_conc = np.exp(-par['q']*Mx.x)
    res_conc = 1/(inte @ (Mx.M @ res_conc))*res_conc + 0.0001

    beta = np.exp(-(par['q']*Mx.x)**2) #+0.001 2*np.exp(-Mx.x)/(1+np.exp(-Mx.x))#np.exp(-Mx.x**2)#+0.001
    beta = 0.5*1 / (inte @ (Mx.M @ beta)) * beta +0.0001

    lam = z[0:2]
    state = z[2:4]
    sigma = z[4:tot_points+4] #ca.MX.sym('sigma', Mx.x.shape[0])
    sigma_p = z[tot_points+4:] #ca.MX.sym('sigma_p', Mx.x.shape[0])

    #    mu1 = ca.MX.sym('mu1', Mx.x.shape[0])
    #    mu2 = ca.MX.sym('mu2', Mx.x.shape[0])

    cons_dyn = inte @ (Mx.M @ (sigma/par['c_enc_freq']*(1-state[0]*sigma**2/(par['c_enc_freq']*res_conc*car_cap)))) - inte @ (Mx.M @ (state[1]*sigma*beta*sigma_p))/(par['p_enc_freq']+par['p_handle']*inte @ (Mx.M @ (state[0]*sigma*beta*sigma_p)))
    pred_dyn = par['eff']*inte @ (Mx.M @ (state[0]*sigma*beta*sigma_p))/(par['p_enc_freq']+par['p_handle']*inte @ (Mx.M @ (state[0]*sigma*beta*sigma_p))) - par['p_met_loss'] - par['competition']*inte @ (Mx.M @ (sigma_p**2*beta))

    df1 = 1/par['c_enc_freq']*(1-2*state[0]*sigma/(res_conc*par['c_enc_freq']*car_cap)) - state[1]*sigma_p*beta/(par['p_enc_freq']+inte @ (Mx.M @ (par['p_handle']*state[0]*sigma*beta*sigma_p))) - np.log(lam[0]**2)*np.ones(tot_points)
    df2 = par['eff']*state[0]*par['p_enc_freq']**2*sigma*beta/(inte @ (Mx.M @ (par['p_handle']* state[0]*sigma*beta*sigma_p))+par['p_enc_freq'])**2 - np.log(lam[1]**2)*np.ones(tot_points) - par['competition']*sigma_p*beta
    g1 = inte @ Mx.M @ sigma_p - 1
    g2 = inte @ Mx.M @ sigma - 1


    F[:] = np.concatenate([g1, g2, cons_dyn, pred_dyn, -df1, -df2])

    pass

def mcp_Nablafunction(n, z, F_nabla):
    Mx = discrete_patches(1,60)
    tot_points = Mx.x.size
    inte = np.ones(tot_points).reshape(1,tot_points)

    par = {'res_renew': 1, 'eff': 0.1, 'c_handle': 1, 'c_enc_freq': 1, 'c_met_loss': 0.001, 'p_handle': 0.01,
           'p_enc_freq': 0.1, 'p_met_loss': 0.15, 'competition': 0.1, 'q': 3}

    res_conc = np.exp(-par['q'] * Mx.x)  # np.exp(-Mx.x**2)#np.exp(-Mx.x)+0.001
    res_conc = 1 / (inte @ (Mx.M @ res_conc)) * res_conc + 0.0001
    beta = np.exp(-(par['q'] * Mx.x) ** 2)  # +0.001 2*np.exp(-Mx.x)/(1+np.exp(-Mx.x))#np.exp(-Mx.x**2)#+0.001
    beta = 0.5 * 1 / (inte @ (Mx.M @ beta)) * beta + 0.0001
    lam = ca.MX.sym('lam', 2)
    car_cap = 3

    sigma = ca.MX.sym('sigma', Mx.x.shape[0])
    sigma_p = ca.MX.sym('sigma_p', Mx.x.shape[0])

    state = ca.MX.sym('state', 2)

    cons_dyn = inte @ (Mx.M @ (sigma / par['c_enc_freq'] * (
                1 - state[0] * sigma ** 2 / (par['c_enc_freq'] * res_conc * car_cap)))) - inte @ (
                           Mx.M @ (state[1] * sigma * beta * sigma_p)) / (
                           par['p_enc_freq'] + par['p_handle'] * inte @ (Mx.M @ (state[0] * sigma * beta * sigma_p)))
    pred_dyn = par['eff'] * inte @ (Mx.M @ (state[0] * sigma * beta * sigma_p)) / (
                par['p_enc_freq'] + par['p_handle'] * inte @ (Mx.M @ (state[0] * sigma * beta * sigma_p))) - par[
                   'p_met_loss'] - par['competition'] * inte @ (Mx.M @ (sigma_p ** 2 * beta))

    df1 = 1 / par['c_enc_freq'] * (1 - 2*state[0] * sigma / (res_conc * par['c_enc_freq'] * car_cap)) - state[
        1] * sigma_p * beta / (
                      par['p_enc_freq'] + inte @ (Mx.M @ (par['p_handle'] * state[0] * sigma * beta * sigma_p))) - \
          ca.log(lam[0]**2) * np.ones(tot_points)
    df2 = par['eff'] * state[0] * par['p_enc_freq']**2 * sigma * beta / (
                inte @ (Mx.M @ (par['p_handle'] * state[0] * sigma * beta * sigma_p)) + par['p_enc_freq']) ** 2 - \
          ca.log(lam[1]**2) * np.ones(tot_points) - par['competition'] * sigma_p * beta

    # g0 = ca.vertcat(cons_dyn, pred_dyn)
    g0 = ca.vertcat(cons_dyn, pred_dyn)
    g2 = inte @ Mx.M @ sigma_p - 1
    g3 = inte @ Mx.M @ sigma - 1
    g4 = ca.vertcat(-df1, -df2)
    f = ca.vertcat(g2, g3, g0, g4)
    zo = ca.vertcat(lam, state, sigma,sigma_p)

    fun = ca.Function('fun', [zo], [ca.jacobian(f,zo)])

    F_nabla[:] = fun(z)

    pass

def test_new():
    mcp = sn.NCP(124, mcp_function, mcp_Nablafunction)

    z = np.ones(124,dtype=float)
    w = np.ones(124,dtype=float)
    SO = sn.SolverOptions(sn.SICONOS_NCP_NEWTON_FB_FBLSA)
    info = sn.ncp_newton_FBLSA(mcp, z, w, SO)
    print("z = ", z)
    print("w = ", w)
    print(z[2:4])
    print(info)

test_new()
