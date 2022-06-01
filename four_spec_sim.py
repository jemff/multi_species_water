import casadi as ca
import numpy as np
import siconos.numerics as sn
from infrastructure import *

class spec_simulator:
    def __init__(self, Mx = None, light_levels = [1], time_lengths = [1], smoothing = False, fitness_proxy = "gm"):
        self.Mx = Mx
        self.light_levels = light_levels
        self.time_lengths = time_lengths
        self.tot_times = len(time_lengths)
        self.fitness_proxy = fitness_proxy
        self.tot_points = Mx.x.size
        self.smoothing = smoothing
        self.inte = np.ones(self.tot_points).reshape(1, self.tot_points)

        h = 20 / 365
        a = 0.4
        m0 = 10 ** (-3)
        gamma = 0.6
        k = 0.05
        masses = np.array([0.1, 11, 4000, 4000])
        self.f_c = 0 #0.15 / 365
        self.r = 1 / 365
        self.r_b = 1 / 365
        eps0 = 0.05
        self.comp = 0
        self.Rmax = 5
        Cmax = h * masses ** (0.75)
        self.metabolism = 0.2*Cmax#*masses
        self.metabolism[-1] = 0.5*self.metabolism[-1]
        self.epsi = eps0 * ((1 - a) * np.log(masses / m0) - np.log(eps0))
        #self.epsi = 0.7*self.epsi/self.epsi
        self.eps_assim = 0.7
        self.rz = 1/(8.8)*np.exp(-((Mx.x)) ** 2 / (10 ** 2)) + 10**(-4)
        self.Bmax = 1
        T_dist = np.exp(-((Mx.x-Mx.x[-1])) ** 2 / (10 ** 2))
        mass_T = self.inte @ self.Mx.M @ T_dist
        self.bent_dist = T_dist/mass_T
        upright_wc = np.exp(-k * Mx.x)  # .reshape((-1, 1))
        self.c_z = Cmax[0]
        self.c_ff = Cmax[1]
        self.c_lp = Cmax[2]
        self.c_ld = Cmax[3]
        self.bg_M = 0.1 / 365
        self.beta_0 = 10 ** (-5)
        tau = 5
        smoothers = 0.5 * (tau + tau * np.log10(masses / masses[0])) ** 2
        gaussians = []
        if self.smoothing is True:
            for j in range(4):
                smoothness = smoothers[j]
                gridx, gridy = np.meshgrid(Mx.x, Mx.x)
                ker = lambda x, y: np.exp(-(x - y) ** 2 / (4 * smoothness)) + np.exp(-(-y - x) ** 2 / (4 * smoothness)) + np.exp(
                    -(2 * Mx.x[-1] - x - y) ** 2 / (4 * smoothness))  # Calculate gaussians at every point

                out = (4 * smoothness * np.pi) ** (-1 / 2) * ker(gridx, gridy)
                for j in range(self.tot_points):
                    out[j, :] = out[j, :] / (self.inte @ Mx.M @ out[j, :])
                out_t = (np.transpose(out))  # Invert the Gaussians for use in the NCP
                gaussians.append(out_t)
        else:
            gaussians = 4 * [np.identity(self.tot_points)]
        self.gaussians = gaussians
        beta_ld_b = []
        beta_z = []
        beta_ff = []
        beta_lp = []
        beta_ld = []

        for j in range(self.tot_times):
            Vi = light_levels[j]
            beta_i = 330 / 365 * Vi * masses ** (0.75) * gamma
            beta_ld_b.append(gamma*0.5*330 / 365 *masses[-1]**(0.75) * (upright_wc ** 0) + self.beta_0)
            beta_z.append(2* upright_wc ** 0 *masses[0] ** (0.75))
            beta_ff.append(2*beta_i[1] * (upright_wc / (1 + upright_wc) + self.beta_0))
            beta_lp.append(2*beta_i[2] * (upright_wc / (1 + upright_wc)  + self.beta_0))
            beta_ld.append(beta_i[3] * (upright_wc / (1 + upright_wc)  + self.beta_0))

        #print(beta_ld_b, "BETA_LD_B")
        self.beta_z = beta_z
        self.beta_ld_b = beta_ld_b
        self.beta_ff = beta_ff
        self.beta_lp = beta_lp
        self.beta_ld = beta_ld

    def nash_eq_calculator(self, j=0, state=None, warmstart_info = None, warmstart_out = True, ipopt = True):
        lam = ca.MX.sym('lam', 4)
        sigma_z_o = ca.MX.sym('sigma_z_o', self.tot_points)
        sigma_ff_o = ca.MX.sym('sigma_ff_o', self.tot_points)
        sigma_lp_o = ca.MX.sym('sigma_lp_o', self.tot_points)
        sigma_ld_o = ca.MX.sym('sigma_ld_o', self.tot_points)

        sigma_z = self.gaussians[0] @ sigma_z_o
        sigma_ff = self.gaussians[1] @ sigma_ff_o
        sigma_lp = self.gaussians[2] @ sigma_lp_o
        sigma_ld = self.gaussians[3] @ sigma_ld_o


        ff_z_enc =  (self.beta_ff[j] * sigma_z * sigma_ff)
        ff_satiation = (1 / (state[0] * ff_z_enc + self.c_ff))  # c_ff*ff_z_enc

        lp_ff_enc = (self.inte @ (self.Mx.M @ (self.beta_lp[j] * sigma_ff * sigma_lp)))
        lp_satiation = (1 / (state[1] * lp_ff_enc + self.c_lp))  # c_lp*lp_ff_enc

        ld_ff_enc = self.beta_ld[j] * sigma_ld * sigma_ff
        ld_bc_enc = self.beta_ld_b[j] * sigma_ld * self.bent_dist

        ld_satiation = (1 / (state[1] * ld_ff_enc + state[4] * ld_bc_enc + self.c_ld))

        if self.fitness_proxy == "gm":
            df_z = self.epsi[0]*self.c_z**2*state[5]*self.beta_z[j]*self.rz/(self.c_z + state[5]*self.beta_z[j]*sigma_z *self.rz)**2 - state[1]*( self.c_ff*sigma_ff*self.beta_ff[j]*ff_satiation) + ca.log(lam[0]**2)
            df_ff = (self.epsi[1]*state[0]*self.c_ff**2*self.beta_ff[j]*sigma_z*ff_satiation**2 \
                    - state[2]*self.c_lp*sigma_lp*self.beta_lp[j]*lp_satiation\
                    - state[3]*self.c_ld*sigma_ld*self.beta_ld[j]*ld_satiation) + ca.log(lam[1]**2)  # ca.log(lam[1]**2)  #lam[1]
            df_lp = (self.epsi[2] * state[1] * sigma_ff * self.c_lp ** 2 * self.beta_lp[j] * lp_satiation ** 2) + ca.log(lam[2]**2) - self.comp*state[2]*self.beta_lp[j]*sigma_lp  #- 0.5*self.beta_lp[j]*sigma_lp #lam[2]# ca.log(lam[2]**2)
            df_ld_up = self.c_ld**2 * self.epsi[3] * (state[1] * sigma_ff * self.beta_ld[j]* (ld_satiation ** 2) + state[4]*self.beta_ld_b[j] * self.bent_dist* (ld_satiation ** 2))  \
                       + ca.log(lam[3]**2) - self.comp*state[3]*(sigma_ld*self.bent_dist*self.beta_ld_b[j])  #- 0.5*self.beta_ld[j]*sigma_ld # lam[3] #ca.log(lam[3]**2)

        elif self.fitness_proxy == "gilliam":
            g_z = self.inte @ self.Mx.M @ (self.c_z*self.beta_z[j])
            m_z = state[1]*self.inte @ self.Mx.M @ (sigma_z *self.c_ff*sigma_ff*self.beta_ff[j]*ff_satiation) + self.inte @ self.Mx.M @ (self.c_z*self.beta_z[j]*state[0]*sigma_z**2/self.rz) + self.epsi[0]*self.bg_M
            dg_z = 0 #(self.c_z*self.beta_z[j])
            dm_z = state[1]*self.c_ff*sigma_ff*self.beta_ff[j]*ff_satiation + state[0]*(self.c_z*self.beta_z[j]*sigma_z/self.rz)

            df_z = dg_z/g_z - dm_z/m_z + ca.log(lam[0]**2)

            g_ff = self.inte @ (self.Mx.M @ (sigma_ff * self.epsi[1]*state[0]*self.c_ff*self.beta_ff[j]*sigma_z*ff_satiation))
            m_ff = self.inte @ self.Mx.M @ (sigma_ff * ( state[2]*self.c_lp*sigma_lp*self.beta_lp[j]*lp_satiation\
                    + state[3]*self.c_ld*sigma_ld*self.beta_ld[j]*ld_satiation )) + self.epsi[1]*self.bg_M
            dg_ff = self.epsi[1]*state[0]*self.c_ff**2*self.beta_ff[j]*sigma_z*ff_satiation**2
            dm_ff = state[2]*self.c_lp*sigma_lp*self.beta_lp[j]*lp_satiation\
                + state[3]*self.c_ld*sigma_ld*self.beta_ld[j]*ld_satiation

            df_ff = dg_ff/g_ff - dm_ff/m_ff + ca.log(lam[1]**2)

            df_lp = self.epsi[2] * state[1] * sigma_ff * self.c_lp ** 2 * self.beta_lp[j] * lp_satiation ** 2/(self.epsi[2]*self.bg_M) + ca.log(lam[2]**2) #lam[2]# ca.log(lam[2]**2)

            df_ld_up = self.c_ld**2 * self.epsi[3] * (state[1] * sigma_ff * self.beta_ld[j] + state[3]*self.beta_ld_b[j] * sigma_ld * self.bent_dist) * ld_satiation ** 2 /(self.epsi[3]*self.bg_M) + ca.log(lam[3]**2) # lam[3] #ca.log(lam[3]**2)

        else:
            print("Invalid fitness proxy")
        if self.smoothing is False:
            p1 = self.inte @ self.Mx.M @ sigma_z - 1
            p2 = self.inte @ self.Mx.M @ sigma_ff - 1
            p3 = self.inte @ self.Mx.M @ sigma_lp - 1
            p4 = self.inte @ self.Mx.M @ sigma_ld - 1
        else:
            p1 = self.inte @ sigma_z_o - 1
            p2 = self.inte @ sigma_ff_o - 1
            p3 = self.inte @ sigma_lp_o - 1
            p4 = self.inte @ sigma_ld_o - 1
        complementarity = (self.inte @ self.Mx.M  @(df_z * sigma_z)+ self.inte @ self.Mx.M @( (df_ff * sigma_ff)) + self.inte @ self.Mx.M @ ( (df_lp * sigma_lp)) + self.inte @ self.Mx.M @ ((df_ld_up * sigma_ld)))
        normal_cone = ca.vertcat(-df_z, -df_ff, -df_lp, -df_ld_up)
        sigmas = ca.vertcat(*[sigma_z_o, sigma_ff_o, sigma_lp_o, sigma_ld_o])  # sigma_bar
        nc_sic = ca.vertcat(-self.gaussians[0] @ df_z, -self.gaussians[1] @ df_ff, -self.gaussians[2] @ df_lp, -self.gaussians[3] @ df_ld_up)
        g = ca.vertcat(*[p1, p2, p3, p4, complementarity, normal_cone])


        lbg = np.zeros(g.size()[0])
        ubg = ca.vertcat(*[*np.zeros(4), [0.00], [ca.inf] * (4 * self.tot_points)])

        x = ca.vertcat(*[lam, sigmas])
        lbx = ca.vertcat(np.zeros(x.size()[0]))
        func = ca.vertcat(*[p1, p2, p3, p4, nc_sic])
        mcp_function_ca = ca.Function('fun', [x], [func])
        mcp_Nablafunction_ca = ca.Function('fun', [x], [ca.jacobian(func, x)])

        def mcp_function(n, z, F):
            F[:] = np.array(*[mcp_function_ca(z)]).flatten()
            pass

        def mcp_Nablafunction(n, z, nabla_F):
            nabla_F[:] = mcp_Nablafunction_ca(z)
            pass

        if ipopt is True:
            if warmstart_info is None:
                s_opts = {'ipopt': {'print_level' : 5, 'linear_solver':'ma57'}} #
                init = np.ones(x.size()[0]) / np.max(self.Mx.x)
                init[0:3] = 10

            else:
                s_opts = {'ipopt': {'print_level': 3, 'linear_solver': 'ma57',
                                    'acceptable_iter': 5,# 'hessian_approximation':'limited-memory',
                                    'warm_start_init_point': 'yes', 'warm_start_bound_push': 1e-9,
                                    'warm_start_bound_frac': 1e-9, 'warm_start_slack_bound_frac': 1e-9,
                                    'warm_start_slack_bound_push': 1e-9, 'warm_start_mult_bound_push': 1e-9}}


              # np.array([0.0265501, 0, 5.6027, 0.940087])
            #f = 0 #ca.dot(normal_cone+sigmas - ca.sqrt(normal_cone**2 + sigmas**2),normal_cone+sigmas - ca.sqrt(normal_cone**2 + sigmas**2) )
            f = 0
            prob = {'x': x, 'f': f, 'g': g}
            solver = ca.nlpsol('solver', 'ipopt', prob, s_opts)


            if warmstart_info is None:
                sol = solver(lbx = lbx, lbg = lbg, ubg = ubg, x0 = init)

            else:
                sol = solver(lbx = lbx, lbg = lbg, ubg = ubg, x0 = warmstart_info['x0'], lam_g0 = warmstart_info['lam_g0'], lam_x0 = warmstart_info['lam_x0'] )

            if warmstart_out is False:
                return np.array(sol['x']).flatten()

            else:
                ret_dict = {'x0': np.array(sol['x']).flatten(), 'lam_g0': np.array(sol['lam_g']).flatten(),
                            'lam_x0': np.array(sol['lam_x']).flatten(), 'f_val':mcp_function_ca(sol['x'])}
                return ret_dict
        else:
            def mcp_solver():
                tot_points = x.size()[0]
                ncp = sn.NCP(tot_points, mcp_function, mcp_Nablafunction) #sn.MCP(4, tot_points-4, mcp_function, mcp_Nablafunction) #sn.NCP(tot_points, mcp_function, mcp_Nablafunction)
                uniform_dist = np.ones(tot_points)/self.Mx.x[-1]
                z = np.copy(warmstart_info[0:tot_points]*0.9+0.1*uniform_dist)
                w = np.array(mcp_function_ca(z)).flatten()#np.copy(warmstart_info[tot_points:])
                print(np.dot(z,w))
                SO = sn.SolverOptions(sn.SICONOS_NCP_NEWTON_FB_FBLSA) #sn.SolverOptions(sn.SICONOS_MCP_NEWTON_FB_FBLSA) #sn.SolverOptions(sn.SICONOS_NCP_NEWTON_FB_FBLSA)
                SO.dparam[sn.SICONOS_DPARAM_TOL] = 10 ** (-4)
                SO.iparam[sn.SICONOS_IPARAM_MAX_ITER] = 50
                SO.iparam[sn.SICONOS_IPARAM_LSA_NONMONOTONE_LS] = 0.5
                info = sn.ncp_newton_FBLSA(ncp, z, w, SO)
                print(info, "Newton status", np.dot(z,w))
                if info != 0 and self.smoothing is False:
                    s_opts = {'ipopt': {'print_level': 3, 'linear_solver': 'ma57'}}  #
                    f = 0  # ca.dot(normal_cone+sigmas - ca.sqrt(normal_cone**2 + sigmas**2),normal_cone+sigmas - ca.sqrt(normal_cone**2 + sigmas**2) )
                    prob_t = {'x': x, 'f': f, 'g': g}
                    solver_t = ca.nlpsol('solver', 'ipopt', prob_t, s_opts)
                    sol_t = solver_t(lbx=lbx, lbg=lbg, ubg=ubg, x0=z)
                    z = np.array(sol_t['x']).flatten()
                    w = np.array(*[mcp_function_ca(z)]).flatten()

                #if info !=0 and self.smoothing is True:
                #Implement replicator equation here.

                return np.concatenate([z,w])

            return mcp_solver()

    def simulator(self, in_state=None, sigma_z_o=None, sigma_ff_o=None, sigma_lp_o=None, sigma_ld_o=None, j=0, theta = 0):
        step = self.time_lengths[j]
        #print(step)
        sigma_z = self.gaussians[0] @ sigma_z_o
        sigma_ff = self.gaussians[1] @ sigma_ff_o
        sigma_lp = self.gaussians[2] @ sigma_lp_o
        sigma_ld = self.gaussians[3] @ sigma_ld_o
        new_state = ca.MX.sym('n_s', 6)
        state = theta*np.array(in_state).squeeze() + (1-theta) * new_state


        ff_z_enc =  (self.beta_ff[j] * sigma_z * sigma_ff)
        ff_satiation = (1 / (state[0] * ff_z_enc + self.c_ff))  # c_ff*ff_z_enc

        lp_ff_enc = (self.inte @ (self.Mx.M @ (self.beta_lp[j] * sigma_ff * sigma_lp)))
        lp_satiation = (1 / (state[1] * lp_ff_enc + self.c_lp))  # c_lp*lp_ff_enc

        ld_ff_enc = (self.beta_ld[j] * sigma_ff * sigma_ld)#)))
        ld_bc_enc = (self.beta_ld_b[j] * sigma_ld * self.bent_dist)#)

        ld_satiation = (1 / (state[1] * ld_ff_enc + state[4] * ld_bc_enc + self.c_ld))

        o1 = -new_state[0] + in_state[0] + step*(state[0]*(self.inte @ (self.Mx.M @ (state[5]*self.beta_z[j]*sigma_z *self.rz/(self.c_z + state[5]*self.beta_z[j]*sigma_z *self.rz)) )
                                                  - state[1]* ( self.inte @ ( self.Mx.M @ (self.c_ff*ff_z_enc*ff_satiation))) - self.metabolism[0]))
        o2 = -new_state[1] + in_state[1] + step*(state[1] * (
                    self.epsi[1] * (self.c_ff *state[0]* self.inte @ ( self.Mx.M @ (ff_z_enc * ff_satiation)) - self.f_c) - state[2] * (self.c_lp * lp_ff_enc *
                    lp_satiation) - ( self.inte @ ( self.Mx.M @ (state[3] * ld_ff_enc * ld_satiation))) - self.bg_M  - self.metabolism[1]))

        o3 = -new_state[2] + in_state[2] + step*state[2] * (self.epsi[2] * (self.c_lp * state[1] * lp_ff_enc * lp_satiation - self.f_c) - self.bg_M - self.metabolism[2] - state[2]*self.inte @ (self.Mx.M @ (self.comp*self.beta_lp[j]*sigma_lp)))
        o4 = -new_state[3] + in_state[3] + step*state[3] * (self.epsi[3] *  self.inte @ ( self.Mx.M @ (
                     ( self.c_ld * state[1] * ld_ff_enc * ld_satiation + state[4] * ld_bc_enc * self.c_ld * ld_satiation)) - self.f_c)  - self.bg_M - self.metabolism[3] - self.comp* state[3]* self.inte @ ( self.Mx.M @ ((self.beta_ld_b[j]*self.bent_dist)*sigma_ld**2 )))

        o5 = -new_state[4] + in_state[4] + step*(self.r_b * (self.Bmax - state[4]) - state[3] * state[4] * self.inte @ ( self.Mx.M @ (self.c_ld * (ld_bc_enc * ld_satiation))))
        o6 = -new_state[5] + in_state[5] + step*(self.r * (self.Rmax - state[5]) - state[5] * state[1] * self.inte @ (self.Mx.M @ (self.beta_z[j] *self.rz*sigma_z)))
        #print("o5", o5.size(), "o4", o4.size(), "o3", o3.size(), "o2", o2.size(), "o1", o1.size())
        o_tot = ca.vertcat(*[o1,o2,o3,o4,o5,o6])
        mcp_function_ca = ca.Function('fun', [new_state], [o_tot])
        mcp_Nablafunction_ca = ca.Function('fun', [new_state], [ca.jacobian(o_tot, new_state)])

        def mcp_function(n, z, F):
            F[:] = np.array(*[mcp_function_ca(z)]).flatten()
            pass

        def mcp_Nablafunction(n, z, nabla_F):
            nabla_F[:] = mcp_Nablafunction_ca(z)
            pass

        def mcp_solver():
            tot_points = 6
            ncp = sn.NCP(tot_points, mcp_function, mcp_Nablafunction) #sn.MCP(4, tot_points-4, mcp_function, mcp_Nablafunction) #sn.NCP(tot_points, mcp_function, mcp_Nablafunction)
            z = np.copy(in_state)
            w = np.array(mcp_function_ca(in_state)).flatten()#np.copy(warmstart_info[tot_points:])
            print(np.dot(z,w))
            SO = sn.SolverOptions(sn.SICONOS_NCP_NEWTON_FB_FBLSA) #sn.SolverOptions(sn.SICONOS_MCP_NEWTON_FB_FBLSA) #sn.SolverOptions(sn.SICONOS_NCP_NEWTON_FB_FBLSA)
            SO.dparam[sn.SICONOS_DPARAM_TOL] = 10 ** (-6)
            SO.iparam[sn.SICONOS_IPARAM_MAX_ITER] = 500
            SO.iparam[sn.SICONOS_IPARAM_LSA_NONMONOTONE_LS] = 0.5
            info = sn.ncp_newton_FBLSA(ncp, z, w, SO)
            print(info, "Newton status of implicit newton", np.dot(z,w))
            #if info !=0 and self.smoothing is True:
            #Implement replicator equation here.

            return z

        results = mcp_solver()


        return results

