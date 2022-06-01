from three_spec import *
import matplotlib.pyplot as plt
Mx = discrete_patches(300, 10)
import numpy as np

light_levels = np.concatenate([np.repeat(0,3), np.linspace(0,1,13), np.repeat(1,3), np.linspace(0,1,13)[::-1][1:-1]], axis = 0) #np.repeat(0,3), np.repeat(1,3)
time_lengths = np.concatenate([np.repeat(0.1, 4), np.repeat(1/(24*4), 11), np.repeat(0.1, 4), np.repeat(1/(24*4), 11)]) #np.repeat(0.1,4), np.repeat(0.1,4)
times = len(time_lengths)
print(time_lengths)
print(light_levels.shape, time_lengths.shape)
time_lengths = time_lengths/(np.sum(time_lengths))
# simple_method(50, 80)#spectral_method(50, 10, segments = 1) #spectral_method(30, 30, segments = 1)
simul_test = spec_simulator(Mx=Mx, time_lengths=time_lengths, light_levels=light_levels)
#print(simul_test.beta_ld_b, "Beta_ld_b")
days = 1
res_level = simul_test.rz #5*np.exp(-(Mx.x-10)**2/3**2)
state = np.array([1000, 1, 1, 1, 0.1])
output = simul_test.nash_eq_calculator(0, res_level, state, warmstart_out=True)
print(output['x0'].shape[0])
sigma_r_ff = output['x0'][3:simul_test.tot_points+3]
sigma_r_lp = output['x0'][3+simul_test.tot_points:2*simul_test.tot_points+3]
sigma_r_ld = output['x0'][2*simul_test.tot_points+3:3*simul_test.tot_points+3]
sigma_r_ld_b = output['x0'][-1]
warm_inf = np.concatenate([output['x0'],np.array(output['f_val']).flatten()])
output = warm_inf
print(simul_test.inte @ Mx.M @ sigma_r_ld + sigma_r_ld_b, simul_test.inte @ Mx.M @ sigma_r_lp, simul_test.inte @ Mx.M @ sigma_r_ff)
print(state, sigma_r_ld_b)
for j in range(len(time_lengths)*days):
    res_level, state = simul_test.simulator(res_level=res_level, state=state, sigma_ff = sigma_r_ff, sigma_lp = sigma_r_lp, sigma_ld = sigma_r_ld, sigma_ld_b = sigma_r_ld_b, j=j%times)
    output = simul_test.nash_eq_calculator(j % times, res_level=res_level, state=state, warmstart_info=output, ipopt=False)
    sigma_r_ff = output[3:simul_test.tot_points + 3]
    sigma_r_lp = output[3 + simul_test.tot_points:2 * simul_test.tot_points + 3]
    sigma_r_ld = output[2 * simul_test.tot_points + 3:3 * simul_test.tot_points + 3]
    sigma_r_ld_b = output[3 * simul_test.tot_points + 3]
    print(state, light_levels[j%times], j, sigma_r_ld_b)

print(res_level)