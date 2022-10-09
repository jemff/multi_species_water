import numpy as np
class spectral_method:
    def __init__(self, depth, layers, segments=1):

        self.n = layers
        self.x = self.JacobiGL(0, 0, layers - 1)

        D_calc = lambda n: np.matmul(np.transpose(self.vandermonde_dx()),
                                     np.linalg.inv(np.transpose(self.vandermonde_calculator()))) * (depth / 2)

        self.D = D_calc(layers)
        M_calc = lambda n: np.dot(np.linalg.inv(self.vandermonde_calculator()),
                                  np.linalg.inv(np.transpose(self.vandermonde_calculator()))) * (depth / 2)

        self.M = M_calc(layers)
        self.x = ((self.x + 1) / 2) * depth
        self.segments = segments

        if segments > 1:
            M_T = np.zeros((layers * segments, layers * segments))
            D_T = np.zeros((layers * segments, layers * segments))
            x_T = np.zeros(layers * segments)
            s_x = depth / segments
            x_n = np.copy(self.x) / segments

            for k in range(segments):
                M_T[k * layers:(k + 1) * layers, k * layers:(k + 1) * layers] = self.M / segments
                D_T[k * layers:(k + 1) * layers, k * layers:(k + 1) * layers] = self.D / segments
                x_T[k * layers:(k + 1) * layers] = x_n + k * s_x

            self.D = D_T
            self.M = M_T
            self.x = x_T


    def JacobiGL(self, a, b, n):
        alpha = a + 1
        beta = b + 1
        N = n - 2
        if N == 0:
            x = np.array([(alpha - beta) / (alpha + beta + 2)])
            w = 2
        else:
            h1 = 2 * np.arange(0, N + 1) + alpha + beta
            J1 = np.diag(-1 / 2 * (alpha ** 2 - beta ** 2) / (h1 + 2) / h1)
            J2 = np.diag(2 / (h1[0:N] + 2) * np.sqrt(np.arange(1, N + 1) * (np.arange(1, N + 1) + alpha + beta) *
                                                     (np.arange(1, N + 1) + alpha) * (np.arange(1, N + 1) + beta) * (
                                                             1 / (h1[0:N] + 1)) * (1 / (h1[0:N] + 3))), 1)
            J = J1 + J2
            J = J + J.T
            x, w = np.linalg.eig(J)

        return np.array([-1, *np.sort(x), 1])

    def JacobiP(self, x, alpha, beta, n):
        P_n = np.zeros((n, x.shape[0]))
        P_n[0] = 1
        P_n[1] = 0.5 * (alpha - beta + (alpha + beta + 2) * x)
        for i in range(1, n - 1):
            an1n = 2 * (i + alpha) * (i + beta) / ((2 * i + alpha + beta + 1) * (2 * i + alpha + beta))
            ann = (alpha ** 2 - beta ** 2) / ((2 * i + alpha + beta + 2) * (2 * i + alpha + beta))
            anp1n = 2 * (i + 1) * (i + alpha + beta + 1) / ((2 * i + alpha + beta + 2) * (2 * i + alpha + beta + 1))

            P_n[i + 1] = ((ann + x) * P_n[i] - an1n * P_n[i - 1]) / anp1n

        return P_n

    def JacobiP_n(self, x, alpha, beta, n):
        P_n = self.JacobiP(x, alpha, beta, n)
        if alpha == 1 and beta == 1:
            gamma = lambda alpha, beta, m: 2 ** (3) * (m + 1) / (m + 2) * 1 / ((2 * m + alpha + beta + 1))
        elif alpha == 0 and beta == 0:
            gamma = lambda alpha, beta, m: 2 / ((2 * m + alpha + beta + 1))
        for i in range(n):
            d = np.sqrt(gamma(alpha, beta, i))
            P_n[i] = P_n[i] / d

        return P_n

    def GradJacobi_n(self, x, alpha, beta, n):
        P_diff = np.zeros((n, x.shape[0]))
        JacobiPnorma = self.JacobiP_n(x, alpha + 1, beta + 1, n)
        for i in range(1, n):
            P_diff[i] = JacobiPnorma[i - 1] * np.sqrt(i * (i + alpha + beta + 1))
        return P_diff

    def vandermonde_calculator(self):
        n = self.n
        x = self.x
        return (self.JacobiP_n(x, 0, 0, n))

    def vandermonde_dx(self):
        n = self.n
        x = self.x
        return (self.GradJacobi_n(x, 0, 0, n))

    def expander(self, old_spectral, transform_vec):
        new_spectral = self
        length = old_spectral.x[-1]
        coeffs = np.linalg.inv(old_spectral.JacobiP_n(2 * old_spectral.x / length - 1, 0, 0, old_spectral.n).T) @ transform_vec
        transformer = new_spectral.JacobiP_n(2 * new_spectral.x / new_spectral.x[-1] - 1, 0, 0, old_spectral.n).T
        return transformer @ coeffs

    def interpolater(self, new_points, transform_vec):
        length = self.x[-1]
        coeffs = np.linalg.inv(self.JacobiP_n(2 * self.x / length - 1, 0, 0, self.n).T) @ transform_vec
        transformer = self.JacobiP_n(2 * new_points / new_points[-1] - 1, 0, 0, self.n).T

        return transformer @ coeffs




class simple_method:
    def __init__(self, depth, total_points, central = True):
        tot_points = total_points

        self.x = np.linspace(0, depth, tot_points)
        if central is True:
            self.M =  2/3*np.identity(tot_points) + 1/6*np.diag(np.ones(tot_points - 1), -1) + 1/6*np.diag(np.ones(tot_points - 1), 1)
            self.M[0,0] = 1/3
            self.M[-1,-1] = 1/3
        else:
            self.M = 0.5 * (np.identity(tot_points) + np.diag(np.ones(tot_points - 1), -1))

        h = (tot_points-1)/depth

        self.M = self.M/h
        self.D = h/(2)*self.fin_diff_mat(tot_points)

    def fin_diff_mat(self, N):
        D = np.zeros((N, N))
        D[0, 0] = -3
        D[-1, -1] = 3
        D[0, 2] = -1
        D[-1, -3] = 1
        D = D - np.diag(np.ones(N - 1), -1)
        D = D + np.diag(np.ones(N - 1), 1)
        D[0, 1] += 3
        D[-1, -2] -= 3

        return D

def M_per_calc(N, length = 24):
    M_per = 2 / 3 * np.identity(N) + 1 / 6 * np.diag(np.ones(N - 1), -1) + 1 / 6 * np.diag(
        np.ones(N - 1), 1)
    M_per[0,-1] = 1/6
    M_per[-1,0] = 1/6
    h = N/length
    M_per = M_per / h

    return M_per

def fin_diff_mat_periodic(N, length =24, central = False):
    h = length/(N) #*2*np.pi

    if central is True:
        D = np.zeros((N, N))
        D = D - np.diag(np.ones(N - 1), -1)
        D = D + np.diag(np.ones(N - 1), 1)
        D[-1,0] = 1
        D[0,-1] = -1

        D = 1/(2*h)*D
    if central is False:
        D = np.identity(N)
        D = -D + np.diag(np.ones(N - 1), 1)
        D[-1,0] = 1
        D = D/h

    return D

def spectral_periodic(N, length = 2*np.pi):
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
    D = D*2*np.pi/length
    return D
def transport_matrix(depth = 200, total_points = 100, diffusivity = 0, central = False):
    #D = np.identity(total_points)
    #D = D - np.diag(np.ones(total_points - 1), -1)
    #D = D/h
    #D[0,:] = 0
    #D[-1,:] = 0
    h = depth/(total_points - 1)
    if central:
        N = total_points
        D = np.zeros((N, N))
        D[0, 0] = -3
        D[-1, -1] = 3
        D[0, 2] = -1
        D[-1, -3] = 1
        D = D - np.diag(np.ones(N - 1), -1)
        D = D + np.diag(np.ones(N - 1), 1)
        D[0, 1] += 3
        D[-1, -2] -= 3
        D = 1/(2*h) * D
        bc_mat = np.identity(total_points)
        bc_mat[-1, -1] = 0
        bc_mat[0,0] = 0

        D = bc_mat @ D
    else:
        D = np.identity(total_points)
        D = D - np.diag(np.ones(total_points - 1), -1)
        D = D/h
        D[0,:] = 0
        D[-1,:] = 0

    D2 = -2*np.identity(total_points) + np.diag(np.ones(total_points - 1), -1) + np.diag(np.ones(total_points - 1), 1)
    D2[0,1] = 2
    D2[-1,-2] = 2
    D2 = D2/h**2
    D2 = diffusivity*D2

    return D, -D2

class discrete_patches:
    def __init__(self, depth, total_points):
        self.x = np.linspace(0, depth, total_points)

        h = (total_points-1)/depth
        self.M = 1/h * np.identity(total_points)
        self.M[-1,-1] = 1/2*self.M[-1,-1]
        self.M[0,0] = 1/2*self.M[0,0]


        self.D = h/(2)*self.fin_diff_mat(total_points)

    def fin_diff_mat(self, N):
        D = np.zeros((N, N))
        D[0, 0] = -3
        D[-1, -1] = 3
        D[0, 2] = -1
        D[-1, -3] = 1
        D = D - np.diag(np.ones(N - 1), -1)
        D = D + np.diag(np.ones(N - 1), 1)
        D[0, 1] += 3
        D[-1, -2] -= 3

        return D


def heat_kernel(spectral, t = 1, k = 1):
    gridx, gridy = np.meshgrid(spectral.x, spectral.x)
    ker = lambda x, y: np.exp(-(x - y) ** 2 / (4 * k * t)) + np.exp(-(-y - x) ** 2 / (4 * k * t)) + np.exp(-(2*spectral.x[-1] - x - y) ** 2 / (4 * k * t))
    out = (4 * t * k * np.pi) ** (-1 / 2) * ker(gridx, gridy)
    normalizations = np.sum(spectral.M @ out, axis = 0)
    normalizations = np.diag(1/normalizations)
    return normalizations @ spectral.M @ out


import pandas as pd
import pvlib
from pvlib import clearsky

def solar_input_calculator(latitude = 55.571831046, longitude = 12.822830042, tz = 'Europe/Vatican', name = 'Oresund', start_date = '2014-03-22', end_date = '2014-03-23', freq = '15Min', normalized = True):
    altitude = 0
    times = pd.date_range(start=start_date, end=end_date, freq=freq, tz=tz)

    solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)

    apparent_elevation = solpos['apparent_elevation']

    aod700 = 0.1

    precipitable_water = 1

    pressure = pvlib.atmosphere.alt2pres(altitude)

    dni_extra = pvlib.irradiance.get_extra_radiation(times)

    # an input is a Series, so solis is a DataFrame

    solis = clearsky.simplified_solis(apparent_elevation, aod700, precipitable_water,
                                      pressure, dni_extra)

    if normalized is True:
        return solis.dhi.values/np.max(solis.dhi.values)
    else:
        return solis.dhi.values

from scipy.interpolate import interp2d

def D_HJB(depth = 100, total_points = 10):
    h = depth/(total_points - 1)
    D = -np.identity(total_points)
    D = D + np.diag(np.ones(total_points - 1), +1)
    D = D / h
    D[0, :] = 0
    D[-1, :] = 0

    return D

def euler_maruyama(z0=0, vel_field=None, drift=1, h=1e-2):
    drift_term = drift * np.random.normal(scale=np.sqrt(h), size=int(1 / h) - 1)
    timesteps = np.linspace(0, 24, vel_field.shape[1])

    sim_times = np.linspace(0, 24, int(1 / h) - 1)
    x_vals = np.linspace(0, 100, vel_field.shape[0])
    velocity = interp2d(x_vals, timesteps, vel_field, kind='linear')
    locations = np.zeros(int(1 / h))
    locations[0] = z0
    for k in range(1, int(1 / h)):
        locations[k] = locations[k - 1] + velocity(locations[k - 1], sim_times[k - 1]) * h + drift_term[k - 1]
        if locations[k] < 0:
            locations[k] = -locations[k]
        if locations[k] > 100:
            locations[k] = 100 - np.abs(100 - locations[k])
        # print(velocity(locations[k-1], sim_times[k-1]))
        # print(sim_times[k-1], k)
    return locations
