# Copyright (C) 2005, 2018 by INRIA
#!/usr/bin/env python

import numpy as np

import siconos.numerics as sn


def vi_function_1D(n, x, F):
    F[0] = 1.0 + x[0]
    pass


def vi_nabla_function_1D(n, x, nabla_F):
    nabla_F[0] = 1.0
    pass


def vi_function_2D(n, z, F):
    M = np.array([[2., 1.],
                  [1., 2.]])

    q = np.array([-5., -6.])
    F[:] = np.dot(M, z) + q
    pass


def vi_nabla_function_2D(n, z, nabla_F):
    M = np.array([[2., 1.],
                  [1., 2.]])
    nabla_F[:] = M
    pass


def vi_function_3D(n, z, F):
    M = np.array(((0., -1., 2.),
                  (2., 0., -2.),
                  (-1., 1., 0.)))

    q = np.array((-3., 6., -1))
    F[:] = np.dot(M, z) + q
    pass


def vi_nabla_function_3D(n, z, nabla_F):
    M = np.array(((0., -1., 2.),
                  (2., 0., -2.),
                  (-1., 1., 0.)))
    nabla_F[:] = M
    pass


# solution
xsol_1D = np.array([-1.0])
Fsol_1D = np.array([0.0])

xsol_2D = np.array([1.0, 1.0])
Fsol_2D = np.array([-2.0, -3.0])

xsol_3D = np.array((-1.0, -1.0, 1.0))
Fsol_3D = np.array((0., 2.0, -1.0))
# problem
#vi=N.MCP(1,1,vi_function,vi_Nablafunction)

xtol = 1e-8


def test_new():
    return sn.VI(1)


def test_vi_1D():
    vi = sn.VI(1, vi_function_1D)
    vi.set_compute_nabla_F(vi_nabla_function_1D)
    x = np.array([0.])
    F = np.array([0.])

    SO = sn.SolverOptions(sn.SICONOS_VI_BOX_QI)
    lb = np.array((-1.0,))
    ub = np.array((1.0,))
    vi.set_box_constraints(lb, ub)
    info = sn.variationalInequality_box_newton_QiLSA(vi, x, F, SO)
    print(info)
    print("x = ", x)
    print("F = ", F)
    assert (np.linalg.norm(x - xsol_1D) <= xtol)
    assert not info


def test_vi_2D():
    vi = sn.VI(2, vi_function_2D)
    vi.set_compute_nabla_F(vi_nabla_function_2D)
    x = np.array((0., 0.))
    F = np.array((0., 0.))
    sn.SICONOS_VI
    SO = sn.SolverOptions(sn.SICONOS_VI_BOX_QI)
    lb = np.array((-1.0, -1.0))
    ub = np.array((1.0, 1.0))
    vi.set_box_constraints(lb, ub)
    info = sn.variationalInequality_box_newton_QiLSA(vi, x, F, SO)
    print(info)
    print('number of iteration {:} ; precision {:}'.format(
        SO.iparam[sn.SICONOS_IPARAM_ITER_DONE],
        SO.dparam[sn.SICONOS_DPARAM_RESIDU]))
    print("x = ", x)
    print("F = ", F)
    assert (np.linalg.norm(x - xsol_2D) <= xtol)
    assert not info


def test_vi_3D():
    vi = sn.VI(3, vi_function_3D)
    x = np.zeros((3,))
    F = np.zeros((3,))

    SO = sn.SolverOptions(sn.SICONOS_VI_BOX_QI)
    vi.set_compute_nabla_F(vi_nabla_function_3D)
    lb = np.array((-1.0, -1.0, -1.0))
    ub = np.array((1.0, 1.0, 1.0))
    vi.set_box_constraints(lb, ub)
    info = sn.variationalInequality_box_newton_QiLSA(vi, x, F, SO)
    print(info)
    print('number of iteration {:} ; precision {:}'.format(
        SO.iparam[sn.SICONOS_IPARAM_ITER_DONE],
        SO.dparam[sn.SICONOS_DPARAM_RESIDU]))
    print("x = ", x)
    print("F = ", F)
    assert (np.linalg.norm(x - xsol_3D) <= xtol)
    assert not info
    assert(np.abs(SO.dparam[sn.SICONOS_DPARAM_RESIDU]) < 1e-10)

test_vi_1D()
test_vi_2D()
test_vi_3D()