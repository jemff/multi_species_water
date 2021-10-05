from three_spec_dyn import *
import matplotlib.pyplot as plt
Mx = spectral_method(50, 50)


light_levels = np.concatenate([np.linspace(0,1,13), np.linspace(0,1,13)[::-1][1:-1]], axis = 0)
time_lengths = np.array([0.4, *np.repeat(1/(24*4), 11), 0.4, *np.repeat(1/(24*4), 11)])
times = len(time_lengths)
time_lengths = time_lengths/(np.sum(time_lengths))
# simple_method(50, 80)#spectral_method(50, 10, segments = 1) #spectral_method(30, 30, segments = 1)
simul_test = spec_simulator(Mx=Mx, time_lengths=time_lengths, light_levels=light_levels)
#print(simul_test.beta_ld_b, "Beta_ld_b")
days = 1000
res_level = 5*np.exp(-(Mx.x-10)**2/3**2)
state = np.array([1, 1, 1, 1])
output = simul_test.nash_eq_calculator(0, res_level, state, warmstart_out=True)
sigma_r_ff = output['x0'][0:simul_test.tot_points]
sigma_r_lp = output['x0'][simul_test.tot_points:2*simul_test.tot_points]
sigma_r_ld = output['x0'][2*simul_test.tot_points:3*simul_test.tot_points]
sigma_r_ld_b = output['x0'][3*simul_test.tot_points:3*simul_test.tot_points+1]
for j in range(len(time_lengths)*days):
    res_level, state = simul_test.simulator(res_level=res_level, state=state, sigma_ff = sigma_r_ff, sigma_lp = sigma_r_lp, sigma_ld = sigma_r_ld, sigma_ld_b = sigma_r_ld_b, j=j%times)
    output = simul_test.nash_eq_calculator(j % times, res_level=res_level, state=state, warmstart_out=True, warmstart_info=output)
    sigma_r_ff = output['x0'][0:simul_test.tot_points]
    sigma_r_lp = output['x0'][simul_test.tot_points:2 * simul_test.tot_points]
    sigma_r_ld = output['x0'][2 * simul_test.tot_points:3 * simul_test.tot_points]
    sigma_r_ld_b = output['x0'][3 * simul_test.tot_points:3 * simul_test.tot_points + 1]
    print(state, light_levels[j%times], j, sigma_r_ld_b)