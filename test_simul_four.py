from four_spec_sim import *
import matplotlib.pyplot as plt
Mx = simple_method(100, 50) #spectral_method(100, 60) #discrete_patches(150, 60)  #spectral_method(100, 20, segments=5) #simple_method(200, 100) # spectral_method(200, 20, segments=5) # spectral_method(50, 50, segments=1)  #simple_method(50, 50) #simple_method(100, 100) #spectral_method(50, 50, segments=1) #simple_method(50, 50) #spectral_method(300, 50, segments=1) #simple_method(300,50) #spectral_method(300, 5, segments=10) #
import numpy as np
import matplotlib.pyplot as plt

#fidelity_dusk = 4 #currently 11
fidelity = 30
 #np.concatenate([np.linspace(0,1,fidelity_dusk + 2), np.linspace(0,1,fidelity_dusk + 2)[::-1][1:-1]], axis = 0) #np.repeat(0,3), np.repeat(1,3)
time_lengths = np.repeat(1/fidelity, fidelity)#np.concatenate([np.repeat(0.4, 1), np.repeat(0.1/fidelity_dusk, fidelity_dusk), np.repeat(0.4, 1), np.repeat(0.1/(fidelity_dusk), fidelity_dusk)]) #np.repeat(0.1,4), np.repeat(0.1,4)
from scipy.signal import savgol_filter


A = np.cos(2*np.pi*np.linspace(0,1,5*fidelity))*1/np.sqrt(2)
A[A < -1 / 2] = -1 / 2
A[A > 1 / 2] = 1 / 2
A = A+1/2
smoothed_A = savgol_filter(A, window_length = 5, polyorder = 3)
light_levels = smoothed_A[0:-1:5]

light_levels = light_levels + 1/2 #+ 0.0001
plt.plot(np.linspace(0, 1, fidelity), light_levels)
plt.show()
times = len(time_lengths)
print(light_levels[0], time_lengths)
print(light_levels.shape, time_lengths.shape)

time_lengths = time_lengths/(np.sum(time_lengths))
# simple_method(50, 80)#spectral_method(50, 10, segments = 1) #spectral_method(30, 30, segments = 1)
simul_test = spec_simulator(Mx=Mx, time_lengths=time_lengths, light_levels=light_levels, smoothing=False, fitness_proxy="gm")
print(simul_test.epsi)
days = 1000
state = np.array([0.1, 0.001, 1.02251753e-03, 0.00205281743, 0.001, 1])


output = simul_test.nash_eq_calculator(0, state, warmstart_out=True)


sigma_r_z = output['x0'][4:simul_test.tot_points+4]
sigma_r_ff = output['x0'][4+simul_test.tot_points:2*simul_test.tot_points+4]
sigma_r_lp = output['x0'][2*simul_test.tot_points+4:3*simul_test.tot_points+4]
sigma_r_ld = output['x0'][3*simul_test.tot_points+4:4*simul_test.tot_points+4]
sigma_r_ld_b = output['x0'][-1]

plt.plot(Mx.x, simul_test.gaussians[0] @ sigma_r_z, label = 'zoo')
plt.plot(Mx.x, simul_test.gaussians[1] @ sigma_r_ff, label = 'ff')
plt.plot(Mx.x, simul_test.gaussians[2] @ sigma_r_lp, label = 'lp')
plt.plot(Mx.x, simul_test.gaussians[3] @ sigma_r_ld, label = 'ld')
plt.legend(loc='upper right')
plt.plot(Mx.x, simul_test.bent_dist)
plt.show()
#plt.plot(Mx.x, sigma_r_ld_b)


warm_inf = np.concatenate([output['x0'],np.array(output['f_val']).flatten()])
historical_strategies = np.zeros((times, 4, Mx.x.size))
historial_lambdas = np.zeros((times, 4))
#1.10417127e+01 2.40170198e-02 8.36082998e-01 9.06444574e-01
 #5.48334968e-03
#warm_inf[0:4] = np.abs(warm_inf[0:4])
print(simul_test.inte @ Mx.M @ sigma_r_z, simul_test.inte @ Mx.M @ sigma_r_ld + sigma_r_ld_b, simul_test.inte @ Mx.M @ sigma_r_lp, simul_test.inte @ Mx.M @ sigma_r_ff)
ipopt = True #More robust with ipopt, but slower
if ipopt is False:
    output = warm_inf
simulate = True
if simulate is True:
    state_log = np.zeros((len(time_lengths), 6))
    new_error = 1
    old_error = 1
    for j in range(len(time_lengths)*days):
        state = simul_test.simulator(in_state=state, sigma_z_o = sigma_r_z, sigma_ff_o = sigma_r_ff, sigma_lp_o = sigma_r_lp, sigma_ld_o = sigma_r_ld, j=j%times)
        if j<= times or ipopt is True:
            output = simul_test.nash_eq_calculator(j % times, state=state, warmstart_info=output, ipopt=ipopt)
        else:
            output = simul_test.nash_eq_calculator(j % times, state=state, warmstart_info=np.concatenate([historial_lambdas[j%times].flatten(), historical_strategies[j%times].flatten()]), ipopt=ipopt)

        if ipopt is True:
            sigma_r_z = output['x0'][4:simul_test.tot_points + 4]
            sigma_r_ff = output['x0'][4 + simul_test.tot_points:2 * simul_test.tot_points + 4]
            sigma_r_lp = output['x0'][2 * simul_test.tot_points + 4:3 * simul_test.tot_points + 4]
            sigma_r_ld = output['x0'][3 * simul_test.tot_points + 4:4 * simul_test.tot_points + 4]

        else:
            sigma_r_z = output[4:simul_test.tot_points + 4]
            sigma_r_ff = output[4 + simul_test.tot_points:2 * simul_test.tot_points + 4]
            sigma_r_lp = output[2 * simul_test.tot_points + 4:3 * simul_test.tot_points + 4]
            sigma_r_ld = output[3 * simul_test.tot_points + 4:4 * simul_test.tot_points + 4]

            historical_strategies[j%times,0,:] = np.copy(sigma_r_z)
            historical_strategies[j % times, 1, :] = np.copy(sigma_r_ff)
            historical_strategies[j % times, 2, :] = np.copy(sigma_r_lp)
            historical_strategies[j % times, 3, :] = np.copy(sigma_r_ld)
            historial_lambdas[j%times] = np.copy(output[0:4])
        plotting = False
        if plotting is True:
            plt.plot(Mx.x,  sigma_r_z, label='zoo')
            plt.plot(Mx.x, sigma_r_ff, label='ff')
            plt.plot(Mx.x, sigma_r_lp, label='lp')
            plt.plot(Mx.x,  sigma_r_ld, label='ld')
            plt.legend(loc='upper right')
            plt.plot(Mx.x, simul_test.bent_dist)
            plt.show()




        #print(sigma_r_ld, simul_test.bent_dist)

        #print(sigma_r_ld*simul_test.bent_dist, "Product")
        new_error = np.max(np.abs(state_log[j%times] - state))
        min_error = min(new_error,old_error)
        relative_error = np.max(np.abs(state_log[j%times] - state)/state)
        print("Error change: ", old_error - new_error,"Minimal error: ", min_error, "Relative error: ", relative_error, '\n', "Population levels: ", state, "Light: ", light_levels[j%times], "Step: ", j)

        state_log[j%times] = np.copy(state)
        if min_error<10**(-6) or old_error - new_error<10**(-7) or relative_error < 10**(-4):
            simulate = False

        old_error = np.copy(new_error)


