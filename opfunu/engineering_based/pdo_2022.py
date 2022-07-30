#!/usr/bin/env python
# Created by "Thieu" at 15:00, 29/07/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

# Source: Prairie Dog Optimization Algorithm (PDO-2022)

import numpy as np


WBD_lb = [0.1, 0.1, 0.1, 0.1]
WBD_ub = [2., 10., 10., 2.]

PVD_lb = [0., 0., 10., 10.]
PVD_ub = [99., 99., 200., 200.]

CSD_lb = [0.05, 0.25, 2.0]
CSD_ub = [2.0, 1.3, 15.0]

SRD_lb = [2.6, 0.7, 17., 7.3, 7.3, 2.9, 5.0]
SRD_ub = [3.6, 0.8, 28., 8.3, 8.3, 3.9, 5.5]

TBTD_lb = [0., 0.]
TBTD_ub = [1., 1.]

GTD_lb = [12, 12, 12, 12]
GTD_ub = [60, 60, 60, 60]

CBD_lb = (0.01 * np.ones(5)).tolist()
CBD_ub = (100 * np.ones(5)).tolist()

IBD_lb = [10., 10., 0.9, 0.9]
IBD_ub = [50., 80., 5., 5.]

TCD_lb = [2., 0.2]
TCD_ub = [14., 0.8]

PLD_lb = [0.05, 0.05, 0.05, 0.05]
PLD_ub = [500., 500., 120., 500.]

CBHD_lb = np.zeros(4).tolist()
CBHD_ub = [100., 100., 100., 5.]

RCB_lb = [0., 28., 5.]
RCB_ub = [9.99, 40., 10.]


def welded_beam_design(x):
    """
    WBD is subjected to 4 design constraints: shear, beam blending stress, bar buckling load beam, and deflection
    variables: h=x1, l=x2, t=x3, b=x4
    l: length, h: height, t: thickness, b: weld thickness of the bar

    https://sci-hub.se/10.1016/s0166-3615(99)00046-9

    Parameters
    ----------
    x :

    Returns
    -------

    """
    xichma_max = 30000
    P = 6000
    L = 14
    delta_max = 0.25
    E = 30*10**6
    theta_max=13600
    G = 12*10**6

    fx = x[0] ** 2 * x[1] * 1.10471 + 0.04811 * x[2] * x[3] * (14.0 + x[1])

    Pc_X = 4.013*E*np.sqrt(x[2]**2 * x[3]**6 / 36) / L**2 * (1. - x[2]*np.sqrt(E/(4*G)) / (2*L))
    J = 2*(np.sqrt(2)*x[0]*x[1]* (x[1]**2/4 + (x[0] + x[2]/2)**2))
    M = P*(L + x[1]/2)
    R = np.sqrt(x[1]**2/4 + (x[0]+x[2])**2/4)
    t2 = M*R/J
    t1 = P/(np.sqrt(2)*x[0]*x[1])
    t_X = np.sqrt(t1**2 + 2*t1*t2*x[1]/(2*R) + t2**2)
    xichma_X = 6*P*L / (x[3]*x[2]**2)
    delta_X = 4*P*L**3 / (E * x[2]**3 * x[3])

    g1 = t_X - theta_max
    g2 = xichma_X - xichma_max
    g3 = x[0] - x[3]
    g4 = 0.10471 * x[0]**2 + 0.04811*x[2]*x[3]*(14.0 + x[1]) - 5.0
    g5 = 0.125 - x[0]
    g6 = delta_X - delta_max
    g7 = P - Pc_X
    gx = [g1, g2, g3, g4, g5, g6, g7]

    return fx, gx


def pressure_vessel_design(x):
    """
    Variables: the inner radius (R=x3), the thickness of the head (Th=x2),
        the length of the cylindrical section of the vessel (L=x4), and the thickness of the shell (Ts=x1)

    https://sci-hub.se/10.1115/1.2912596

    Parameters
    ----------
    x :

    Returns
    -------

    """
    fx = 0.6224*x[2]*x[0]*x[3] + 1.7781*x[2]**2*x[1] + 3.1611*x[0]**2*x[3] + 19.8621*x[2]*x[0]**2
    g1 = -x[0] + 0.0193*x[2]
    g2 = -x[2] + 0.00954*x[2]
    g3 = -np.pi*x[1]**2*x[3] - 4./3*np.pi*x[2]**3 + 750*1728
    g4 = -240 + x[3]
    gx = [g1, g2, g3, g4]

    return fx, gx


def compression_srping_design(x):
    """
    CSD aims to minimize the weight of a tension/compression spring given the values of 3 parameters:
        the wire diameter (d=x1), number of active coils (P=x3), and mean coil diameter (D=x2).

    https://sci-hub.se/10.1016/s0166-3615(99)00046-9

    Parameters
    ----------
    x :

    Returns
    -------

    """
    fx = (x[2] + 2)*x[1]*x[0]**2
    g1 = 1 - x[1]**3*x[2]/(71785*x[0]**4)
    g2 = (4*x[1]**2 - x[0]*x[1]) / (12566* (x[2]*x[0]**3 - x[0]**4)) + 1./(5108*x[0]**2) - 1
    g3 = 1 - 140.45*x[0] / (x[1]**2 * x[2])
    g4 = (x[0] + x[1]) / 1.5 - 1
    gx = [g1, g2, g3, g4]

    return fx, gx


def speed_reducer_design(x):
    """
    Depicts a gearbox that sits between the propeller and engine of an aeroplane
    [x1, x2, x3, x4, x5, x6, x7] = [b, m, z, l1, l2, d1, d2]

    Parameters
    ----------
    x :

    Returns
    -------

    """
    fx = 0.7854*x[0]*x[1]**2*(3.3333*x[2]**2 + 14.9334*x[2] - 43.0934) - 1.508*x[0]*(x[5]**2 + x[6]**2) +\
        7.4777*(x[5]**3 + x[6]**3) + 0.7854*(x[3]*x[5]**2 + x[4]*x[6]**2)

    g1 = 27./(x[0]*x[1]**2*x[2]) - 1
    g2 = 397.5 / (x[0]*x[1]**2*x[2]**2) - 1
    g3 = 1.93*x[3]**2/(x[1]*x[5]**4*x[2]) - 1
    g4 = 1.93*x[4]**2/(x[1]*x[6]**4*x[2]) - 1
    g5 = np.sqrt((745*x[3]/(x[1]*x[2]))**2 + 16*10**6) / (110 * x[5]**3) - 1
    g6 = np.sqrt((745*x[4]/(x[1]*x[2]))**2 + 157.5*10**6) / (85*x[6]**3) - 1
    g7 = x[1]*x[2] / 40 - 1
    g8 = 5*x[1] / x[0] - 1
    g9 = x[0] / (12. * x[1]) - 1
    g10 = (1.5*x[5] + 1.9) / x[3] - 1
    g11 = (1.1*x[6] + 1.9) / x[4] - 1
    gx = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11]

    return fx, gx


def three_bar_truss_design(x):
    """
    Minimize three-bar structure weight subject to supporting a total load P acting vertically downwards

    Parameters
    ----------
    x :

    Returns
    -------

    """
    L = 100
    P = 2
    xichma = 2

    fx = (2*np.sqrt(2)*x[0] + x[1]) * L
    g1 = (np.sqrt(2)*x[0] + x[1]) / (np.sqrt(2)*x[0]**2 + 2*x[0]*x[1])*P - xichma
    g2 = x[1] * P / (np.sqrt(x[0]**2 + 2*x[0]*x[1])) - xichma
    g3 = P / (np.sqrt(2)*x[1] + x[0]) - xichma
    gx = [g1, g2, g3]

    return fx, gx


def gear_train_design(x):
    """
    Unconstrained discrete design optimization problem
        [x1, x2, x3, x4] = [n_A, n_B, n_C, n_D]

    Parameters
    ----------
    x :

    Returns
    -------

    """
    x = np.asarray(x, int)
    fx = (1. / 6.931 - x[2]*x[1] / (x[0] * x[3]))**2
    return fx, 0


def cantilevel_beam_design(x):
    """
    Minimize a cantilevel beam's weight.

    Parameters
    ----------
    x :

    Returns
    -------

    """
    fx = 0.0624 * np.sum(x)
    gx = 61./x[0]**3 + 37./x[1]**3 + 19./x[2]**3 + 7./x[3]**3 + 1./x[4]**3 - 1

    return fx, gx


def i_beam_design(x):
    """
    Minimizes the vertical deflection of a beam
        [x1, x2, x3, x4] = [b, h, t_w, t_f]
    Parameters
    ----------
    x :

    Returns
    -------

    """
    fx = 500. / ( (x[2]*(x[1]-2*x[3])**3)/12 + (x[0]*x[3]**3/6) + 2*x[0]*x[3]*(x[1] - x[3])**2 )
    g1 = 2*x[0]*x[2] + x[2]*(x[1] - 2*x[3]) - 300
    g2 = (18*x[1] * 10**4) / (x[2]*(x[1] - 2*x[3])**3 + 2*x[0]*x[2]*(4*x[3]**2 + 3*x[1]*(x[1] - 2*x[3]))) +\
        15*x[0] * 10**3 / ((x[1] - 2*x[3]) * x[2]**2 +2*x[2]*x[0]**3) - 56
    gx = [g1, g2]

    return fx, gx


def tubular_column_design(x):
    """
    [x1, x2] = [d, t]

    https://apmonitor.com/me575/index.php/Main/TubularColumn

    Parameters
    ----------
    x :

    Returns
    -------

    """
    xichma_y = 450
    E = 0.65*10**6
    P = 2300
    pro = 0.002
    L = 300

    fx = 9.8*x[0]*x[1] + 2*x[0]
    g1 = P / (np.pi*x[0]*x[1]*xichma_y) - 1
    g2 = (8*P*L**2) / (np.pi**3 * E *x[0]*x[1] * (x[0]**2 + x[1]**2)) - 1
    g3 = 2. / x[0] - 1
    g4 = x[0] / 14 - 1
    g5 = 0.2 / x[1] - 1
    g6 = x[1] / 8 - 1
    gx = [g1, g2, g3, g4, g5, g6]

    return fx, gx


def piston_lever_design(x):
    """
    [x1, x2, x3, x4] = [H, B, D, X]

    Parameters
    ----------
    x :

    Returns
    -------

    """
    L = 240
    M_max = 1.8*10**6
    P = 1500
    Q = 10000
    theta = np.pi/4
    L1 = np.sqrt((x[3] - x[1])**2 + x[0]**2)
    L2 = np.sqrt((x[3]*np.sin(theta) + x[0])**2 + (x[1] - x[3]*np.cos(theta))**2)
    R = np.abs(-x[3]*(x[3]*np.sin(theta) + x[0]) + x[0]*(x[1] - x[3]*np.cos(theta))) / np.sqrt((x[3] - x[1])**2 + x[0]**2)
    F = np.pi*P*x[2]**2 / 4

    fx = 0.25*np.pi*x[2]**2 * (L2 - L1)
    g1 = Q*L*np.cos(theta) - R*F
    g2 = Q*(L - x[3]) - M_max
    g3 = 1.2*(L2 - L1) - L1
    g4 = x[2]/2 - x[1]
    gx = [g1, g2, g3, g4]

    return fx, gx


def corrugated_bulkhead_design(x):
    """
    [x1, x2, x3, x4] = [width, depth, length, thickness]
    Parameters
    ----------
    x :

    Returns
    -------

    """
    fx = 5.885*x[3]*(x[0] + x[2]) / (x[0] + np.sqrt(np.abs(x[2]**2 - x[1]**2)))
    g1 = -x[3]*x[2]*(0.4*x[0] + x[2]/6) + 8.94*(x[0] + np.sqrt(np.abs(x[2]**2 - x[1]**2)))
    g2 = -x[3]*x[1]**2*(0.2*x[0] + x[2]/12) + 2.2*(8.94*(x[0] + np.sqrt(np.abs(x[2]**2 - x[1]**2))))**(4./3)
    g3 = -x[3] + 0.0156 * x[0] + 0.15
    g4 = -x[3] + 0.0156 * x[2] + 0.15
    g5 = -x[3] + 1.05
    g6 = -x[2] + x[1]
    gx = [g1, g2, g3, g4, g5, g6]

    return fx, gx


def reinforced_concrete_beam_design(x):
    """

    Parameters
    ----------
    x :

    Returns
    -------

    """
    x1_list = [6.0, 6.16, 6.32, 6.6, 7.0, 7.11, 7.2, 7.8, 7.9, 8.0, 8.4]
    x1, x2, x3 = x1_list[int(x[0])], x[1], x[2]

    fx = 2.9*x1 + 0.6*x2*x3
    g1 = x2/x3 -4
    g2 = 180 + 7.375*x1**2/x3 - x1*x2
    gx = [g1, g2]

    return fx, gx


WBD = welded_beam_design
PVD = pressure_vessel_design
CSD = compression_srping_design
SRD = speed_reducer_design
TBTD = three_bar_truss_design
GTD = gear_train_design
CBD = cantilevel_beam_design
IBD = i_beam_design
TCD = tubular_column_design
PLD = piston_lever_design
CBHD = corrugated_bulkhead_design
RCB = reinforced_concrete_beam_design
