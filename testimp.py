
from casadi import *
import matplotlib.pyplot as plt
plt

x = MX.sym('x')
p = MX.sym('p')

nlp = {'x': x,
       'f': (x-p)**2,
       'g': x,
       'p': p}

solver = nlpsol('S', 'ipopt', nlp, {"ipopt.fixed_variable_treatment":"make_constraint"})

sol = solver(x0=1,p=p,lbg=-inf,ubg=0)
F = Function('F',[p],[sol['f']])
J = jacobian(sol['f'], p)

JF = Function('JF',[p],[J])


ps = np.linspace(-2, 2, 100)


plt.plot(ps, F(ps),label='F')
#plt.plot(ps, JF(ps),label='JF')
plt.legend()
plt.show()