#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 08:04, 21/09/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
# -------------------------------------------------------------------------------------------------------%

from numpy import zeros, array, log, abs, exp, sqrt, pi, round, sin, cos, arccos, remainder, arcsin, int, arctan, imag, log10
from scipy.optimize import fminbound
from opfunu.cec.cec2020 import constant


# Industrial Chemical Processes
def p1(x):
    # Heat Exchanger Network Design (case 1)
    out = constant.benchmark_function(1)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]
    fx = 35 * x[0] ** 0.6 + 35 * x[1] ** 0.6
    gx = 0
    hx = zeros(h)
    hx[0] = 200 * x[0] * x[3] - x[2]
    hx[1] = 200 * x[1] * x[5] - x[4]
    hx[2] = x[2] - 1000 * (x[6] - 100)
    hx[3] = x[4] - 10000 * (300 - x[6])
    hx[4] = x[2] - 10000 * (600 - x[7])
    hx[5] = x[4] - 10000 * (900 - x[8])
    hx[6] = x[3] * log(abs(x[7] - 100) + 1e-8) - x[3] * log((600 - x[6]) + 1e-8) - x[7] + x[6] + 500
    hx[7] = x[5] * log(abs(x[8] - x[6]) + 1e-8) - x[5] * log(600) - x[8] + x[6] + 600
    return fx, gx, hx


def p2(x):
    # Heat Exchanger Network Design (case 1)
    out = constant.benchmark_function(2)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]
    fx = (x[0] / (120 * x[3])) ** 0.6 + (x[1] / (80 * x[4])) ** 0.6 + (x[2] / (40 * x[5])) * 0.6
    gx = 0
    hx = zeros(h)
    hx[0] = x[0] - 1e4 * (x[6] - 100)
    hx[1] = x[1] - 1e4 * (x[7] - x[6])
    hx[2] = x[2] - 1e4 * (500 - x[7])
    hx[3] = x[0] - 1e4 * (300 - x[8])
    hx[4] = x[1] - 1e4 * (400 - x[9])
    hx[5] = x[2] - 1e-4 * (600 - x[10])
    hx[6] = x[3] * log(abs(x[8] - 100) + 1e-8) - x[3] * log(300 - x[6] + 1e-8) - x[8] - x[6] + 400
    hx[7] = x[4] * log(abs(x[9] - x[6]) + 1e-8) - x[4] * log(abs(400 - x[7]) + 1e-8) - x[9] + x[6] - x[7] + 400
    hx[8] = x[5] * log(abs(x[10] - x[7]) + 1e-8) - x[5] * log(100) - x[10] + x[7] + 100
    return fx, gx, hx


def p3(x):
    # Optimal Operation of Alkylation Unit
    out = constant.benchmark_function(3)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]
    fx = -1.715 * x[0] - 0.035 * x[0] * x[5] - 4.0565 * x[2] - 10.0 * x[1] + 0.063 * x[2] * x[4]
    hx = 0
    gx = zeros(g)
    gx[0] = 0.0059553571 * x[5] ** 2 * x[0] + 0.88392857 * x[2] - 0.1175625 * x[5] * x[0] - x[0]
    gx[1] = 1.1088 * x[0] + 0.1303533 * x[0] * x[5] - 0.0066033 * x[0] * x[5] ** 2 - x[2]
    gx[2] = 6.66173269 * x[5] ** 2 + 172.39878 * x[4] - 56.596669 * x[3] - 191.20592 * x[5] - 10000
    gx[3] = 1.08702 * x[5] + 0.32175 * x[3] - 0.03762 * x[5] ** 2 - x[4] + 56.85075
    gx[4] = 0.006198 * x[6] * x[3] * x[2] + 2462.3121 * x[1] - 25.125634 * x[1] * x[3] - x[2] * x[3]
    gx[5] = 161.18996 * x[2] * x[3] + 5000.0 * x[1] * x[3] - 489510.0 * x[1] - x[2] * x[3] * x[6]
    gx[6] = 0.33 * x[6] - x[4] + 44.333333
    gx[7] = 0.022556 * x[4] - 0.007595 * x[6] - 1.0
    gx[8] = 0.00061 * x[2] - 0.0005 * x[0] - 1.0
    gx[9] = 0.819672 * x[0] - x[2] + 0.819672
    gx[10] = 24500.0 * x[1] - 250.0 * x[1] * x[3] - x[2] * x[3]
    gx[11] = 1020.4082 * x[3] * x[1] + 1.2244898 * x[2] * x[3] - 100000 * x[1]
    gx[12] = 6.25 * x[0] * x[5] + 6.25 * x[0] - 7.625 * x[2] - 100000
    gx[13] = 1.22 * x[2] - x[5] * x[0] - x[0] + 1.0
    return fx, gx, hx


def p4(x):
    # Reactor Network Design (RND)
    out = constant.benchmark_function(4)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]
    hx = zeros(h)
    k1 = 0.09755988
    k2 = 0.99 * k1
    k3 = 0.0391908
    k4 = 0.9 * k3
    fx = -x[3]
    hx[0] = x[0] + k1 * x[1] * x[4] - 1
    hx[1] = x[1] - x[0] + k2 * x[1] * x[5]
    hx[2] = x[2] + x[0] + k3 * x[2] * x[4] - 1
    hx[3] = x[3] - x[2] + x[1] - x[0] + k4 * x[3] * x[5]
    gx = x[4] ** 0.5 + x[5] ** 0.5 - 4
    return fx, gx, hx


def p5(x):
    # Haverly's Pooling Problem
    out = constant.benchmark_function(5)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]
    fx = -(9 * x[0] + 15 * x[1] - 6 * x[2] - 16 * x[3] - 10 * (x[4] + x[5]))
    gx = zeros(g)
    hx = zeros(h)
    gx[0] = x[8] * x[6] + 2 * x[4] - 2.5 * x[0]
    gx[1] = x[8] * x[7] + 2 * x[5] - 1.5 * x[1]
    hx[0] = x[6] + x[7] - x[2] - x[3]
    hx[1] = x[0] - x[6] - x[4]
    hx[2] = x[1] - x[7] - x[5]
    hx[3] = x[8] * x[6] + x[8] * x[7] - 3 * x[2] - x[3]
    return fx, gx, hx


def p6(x):
    #  Blending-Pooling-Separation problem
    out = constant.benchmark_function(6)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]

    fx = 0.9979 + 0.00432 * x[4] + 0.01517 * x[12]
    gx = 0
    hx = zeros(h)
    hx[0] = x[0] + x[1] + x[2] + x[3] - 300
    hx[1] = x[5] - x[6] - x[7]
    hx[2] = x[8] - x[9] - x[10] - x[11]
    hx[3] = x[13] - x[14] - x[15] - x[16]
    hx[4] = x[17] - x[18] - x[19]
    hx[5] = x[4] * x[20] - x[5] * x[21] - x[8] * x[22]
    hx[6] = x[4] * x[23] - x[5] * x[24] - x[8] * x[25]
    hx[7] = x[4] * x[26] - x[5] * x[27] - x[8] * x[28]
    hx[8] = x[12] * x[29] - x[13] * x[30] - x[17] * x[31]
    hx[9] = x[12] * x[32] - x[13] * x[33] - x[17] * x[34]
    hx[10] = x[12] * x[35] - x[13] * x[36] - x[17] * x[37]
    hx[11] = 1 / 3 * x[0] + x[14] * x[30] - x[4] * x[20]
    hx[12] = 1 / 3 * x[0] + x[14] * x[33] - x[4] * x[23]
    hx[13] = 1 / 3 * x[0] + x[14] * x[36] - x[4] * x[26]
    hx[14] = 1 / 3 * x[1] + x[9] * x[22] - x[12] * x[29]
    hx[15] = 1 / 3 * x[1] + x[9] * x[25] - x[12] * x[32]
    hx[16] = 1 / 3 * x[1] + x[9] * x[28] - x[12] * x[35]
    hx[17] = 1 / 3 * x[2] + x[6] * x[21] + x[10] * x[22] + x[15] * x[30] + x[18] * x[31] - 30
    hx[18] = 1 / 3 * x[2] + x[6] * x[24] + x[10] * x[25] + x[15] * x[33] + x[18] * x[34] - 50
    hx[19] = 1 / 3 * x[2] + x[6] * x[27] + x[10] * x[28] + x[15] * x[36] + x[18] * x[37] - 30
    hx[20] = x[20] + x[23] + x[26] - 1
    hx[21] = x[21] + x[24] + x[27] - 1
    hx[22] = x[22] + x[25] + x[28] - 1
    hx[23] = x[29] + x[32] + x[35] - 1
    hx[24] = x[30] + x[33] + x[36] - 1
    hx[25] = x[31] + x[34] + x[37] - 1
    hx[26] = x[24]
    hx[27] = x[27]
    hx[28] = x[22]
    hx[29] = x[36]
    hx[30] = x[31]
    hx[31] = x[34]
    return fx, gx, hx


def p7(x):
    #  Propane, Isobutane, n-Butane Nonsharp Separation
    out = constant.benchmark_function(7)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]
    c = array([[0.23947, 0.75835], [-0.0139904, -0.0661588], [0.0093514, 0.0338147],
               [0.0077308, 0.0373349], [-0.0005719, 0.0016371], [0.0042656, 0.0288996]])
    fx = c[0, 0] + (c[1, 0] + c[2, 0] * x[23] + c[3, 0] * x[27] + c[4, 0] * x[32] + c[5, 0] * x[33]) * x[4] \
         + c[0, 1] + (c[1, 1] + c[2, 1] * x[25] + c[3, 1] * x[30] + c[4, 1] * x[37] + c[5, 1] * x[38]) * x[12]
    gx = 0
    hx = zeros(h)
    hx[0] = x[0] + x[1] + x[2] + x[3] - 300
    hx[1] = x[5] - x[6] - x[7]
    hx[2] = x[8] - x[9] - x[10] - x[11]
    hx[3] = x[13] - x[14] - x[15] - x[16]
    hx[4] = x[17] - x[18] - x[19]
    hx[5] = x[5] * x[20] - x[23] * x[24]
    hx[6] = x[13] * x[21] - x[25] * x[26]
    hx[7] = x[8] * x[22] - x[27] * x[28]
    hx[8] = x[17] * x[29] - x[30] * x[31]
    hx[9] = x[24] - x[4] * x[32]
    hx[10] = x[28] - x[4] * x[33]
    hx[11] = x[34] - x[4] * x[35]
    hx[12] = x[36] - x[12] * x[37]
    hx[13] = x[26] - x[12] * x[38]
    hx[14] = x[31] - x[12] * x[39]
    hx[15] = x[24] - x[5] * x[20] - x[8] * x[40]
    hx[16] = x[28] - x[5] * x[41] - x[8] * x[22]
    hx[17] = x[34] - x[5] * x[42] - x[8] * x[43]
    hx[18] = x[36] - x[13] * x[44] - x[17] * x[45]
    hx[19] = x[26] - x[13] * x[21] - x[17] * x[46]
    hx[20] = x[31] - x[13] * x[47] - x[17] * x[29]
    hx[21] = 1 / 3 * x[0] + x[14] * x[44] - x[24]
    hx[22] = 1 / 3 * x[0] + x[14] * x[21] - x[28]
    hx[23] = 1 / 3 * x[0] + x[14] * x[47] - x[34]
    hx[24] = 1 / 3 * x[1] + x[9] * x[40] - x[36]
    hx[25] = 1 / 3 * x[1] + x[9] * x[22] - x[26]
    hx[26] = 1 / 3 * x[1] + x[9] * x[43] - x[31]
    hx[27] = x[32] + x[33] + x[35] - 1
    hx[28] = x[20] + x[41] + x[42] - 1
    hx[29] = x[40] + x[22] + x[43] - 1
    hx[30] = x[37] + x[38] + x[39] - 1
    hx[31] = x[44] + x[21] + x[47] - 1
    hx[32] = x[45] + x[46] + x[29] - 1
    hx[33] = x[42]
    hx[34] = x[45]
    hx[35] = 1 / 3 * x[2] + x[6] * x[20] + x[10] * x[40] + x[15] * x[44] + x[18] * x[45] - 30
    hx[36] = 1 / 3 * x[2] + x[6] * x[41] + x[10] * x[22] + x[15] * x[21] + x[18] * x[46] - 50
    hx[37] = 1 / 3 * x[2] + x[6] * x[42] + x[10] * x[43] + x[15] * x[47] + x[18] * x[29] - 30

    return fx, gx, hx


def p8(x):
    #  Process synthesis problem
    out = constant.benchmark_function(8)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]
    x[1] = round(x[1])
    fx = 2 * x[0] + x[1]
    hx = 0
    gx = zeros(g)
    gx[0] = 1.25 - x[0] ** 2 - x[1]
    gx[1] = x[0] + x[1] - 1.6
    return fx, gx, hx


def p9(x):
    #  Process synthesis and design problem
    out = constant.benchmark_function(9)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]
    x[2] = round(x[2])
    fx = -x[2] + 2 * x[0] + x[1]
    hx = x[0] - 2 * exp(-x[1])
    gx = -x[0] + x[1] + x[2]
    return fx, gx, hx


def p10(x):
    #  Process flow sheeting problem
    out = constant.benchmark_function(10)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]
    gx = zeros(g)
    x[2] = round(x[2])
    fx = -0.7 * x[2] + 5 * (x[0] - 0.5) ** 2 + 0.8
    gx[0] = -exp(x[0] - 0.2) - x[1]
    gx[1] = x[1] + 1.1 * x[2] + 1
    gx[2] = x[0] - x[2] - 0.2
    hx = 0
    return fx, gx, hx


def p11(x):
    #  Two-reactor Problem
    out = constant.benchmark_function(11)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]

    x1 = x[0]
    x2 = x[1]
    v1 = x[2]
    v2 = x[3]
    y1 = round(x[4])
    y2 = round(x[5])
    x_ = x[6]
    z1 = 0.9 * (1 - exp(-0.5 * v1)) * x1
    z2 = 0.8 * (1 - exp(-0.4 * v2)) * x2

    fx = 7.5 * y1 + 5.5 * y2 + 7 * v1 + 6 * v2 + 5 * x_
    hx = zeros(h)
    gx = zeros(g)

    hx[0] = y1 + y2 - 1
    hx[1] = z1 + z2 - 10
    hx[2] = x1 + x2 - x_
    hx[3] = z1 * y1 + z2 * y2 - 10
    gx[0] = v1 - 10 * y1
    gx[1] = v2 - 10 * y2
    gx[2] = x1 - 20 * y1
    gx[3] = x2 - 20 * y2
    return fx, gx, hx


def p12(x):
    #    Process synthesis problem
    out = constant.benchmark_function(12)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    y1 = round(x[3])
    y2 = round(x[4])
    y3 = round(x[5])
    y4 = round(x[6])
    fx = (y1 - 1) ** 2 + (y2 - 1) ** 2 + (y3 - 1) ** 2 - log(y4 + 1) + (x1 - 1) ** 22 + (x2 - 2) ** 2 + (x3 - 3) ** 2
    gx = zeros(g)
    gx[0] = x1 + x2 + x3 + y1 + y2 + y3 - 5
    gx[1] = y3 ** 2 + x1 ** 2 + x2 ** 2 + x3 ** 2 - 5.5
    gx[2] = x1 + y1 - 1.2
    gx[3] = x2 + y2 - 1.8
    gx[4] = x3 + y3 - 2.5
    gx[5] = x1 + y4 - 1.2
    gx[6] = y2 ** 2 + x2 ** 2 - 1.64
    gx[7] = y3 ** 2 + x3 ** 2 - 4.25
    gx[8] = y2 ** 2 + x3 ** 2 - 4.64
    hx = 0
    return fx, gx, hx


def p13(x):
    #  Process design Problem
    out = constant.benchmark_function(13)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    y1 = round(x[3])
    y2 = round(x[4])
    fx = -5.357854 * x1 ** 2 - 0.835689 * y1 * x3 - 37.29329 * y1 + 40792.141
    a = [85.334407, 0.0056858, 0.0006262, 0.0022053, 80.51249, 0.0071317, 0.0029955, 0.0021813, 9.300961, 0.0047026, 0.0012547, 0.0019085]
    gx = zeros(g)
    gx[0] = a[0] + a[1] * y2 * x3 + a[2] * y1 * x2 - a[3] * y1 * y1 * x3 - 92
    gx[1] = a[4] + a[5] * y2 * x3 + a[6] * y1 * x2 + a[7] * x1 ** 2 - 90 - 20
    gx[2] = a[8] + a[9] * y1 * x2 + a[10] * y1 * x1 + a[11] * x1 * x2 - 20 - 5
    hx = 0
    return fx, gx, hx


def p14(x):
    # Multi-product batch plant
    out = constant.benchmark_function(14)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]
    gx = zeros(g)
    # constant
    S = array([[2, 3, 4], [4, 6, 3]])
    t = array([[8, 20, 8], [16, 4, 4]])
    H = 6000
    alp = 250
    beta = 0.6
    Q1 = 40000
    Q2 = 20000
    ## Decision Variable
    N1 = round(x[0])
    N2 = round(x[1])
    N3 = round(x[2])
    V1 = x[3]
    V2 = x[4]
    V3 = x[5]
    TL1 = x[6]
    TL2 = x[7]
    B1 = x[8]
    B2 = x[9]
    ## Objective function
    fx = alp * (N1 * V1 ** beta + N2 * V2 ** beta + N3 * V3 ** beta)
    ## constraints
    gx[0] = Q1 * TL1 / B1 + Q2 * TL2 / B2 - H
    gx[1] = S[0, 0] * B1 + S[1, 0] * B2 - V1
    gx[2] = S[0, 1] * B1 + S[1, 1] * B2 - V2
    gx[3] = S[0, 2] * B1 + S[1, 2] * B2 - V3
    gx[4] = t[0, 0] - N1 * TL1
    gx[5] = t[0, 1] - N2 * TL1
    gx[6] = t[0, 2] - N3 * TL1
    gx[7] = t[1, 0] - N1 * TL2
    gx[8] = t[1, 1] - N2 * TL2
    gx[9] = t[1, 2] - N3 * TL2
    hx = 0
    return fx, gx, hx


def p15(x):
    ## Weight Minimization of a Speed Reducer
    out = constant.benchmark_function(15)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]
    hx = 0
    gx = zeros(g)
    fx = 0.7854 * x[0] * x[1] ** 2 * (3.3333 * x[2] ** 2 + 14.9334 * x[2] - 43.0934) - 1.508 * x[0] * (x[5] ** 2 + x[6] ** 2) \
         + 7.477 * (x[5] ** 3 + x[6] ** 3) + 0.7854 * (x[3] * x[5] ** 2 + x[4] * x[6] ** 2)

    gx[0] = -x[0] * x[1] ** 2 * x[2] + 27
    gx[1] = -x[0] * x[1] ** 2 * x[2] ** 2 + 397.5
    gx[2] = -x[1] * x[5] ** 4 * x[2] * x[3] ** (-3) + 1.93
    gx[3] = -x[1] * x[6] ** 4 * x[2] / x[4] ** 3 + 1.93
    gx[4] = 10 * x[5] ** (-3) * sqrt(16.91 * 10 ** 6 + (745 * x[3] / (x[1] * x[2])) ** 2) - 1100
    gx[5] = 10 * x[6] ** (-3) * sqrt(157.5 * 10 ** 6 + (745 * x[4] / (x[1] * x[2])) ** 2) - 850
    gx[6] = x[1] * x[2] - 40
    gx[7] = -x[0] / x[1] + 5
    gx[8] = x[0] / x[1] - 12
    gx[9] = 1.5 * x[5] - x[3] + 1.9
    gx[10] = 1.1 * x[6] - x[4] + 1.9

    return fx, gx, hx


def p16(x):
    ## Optimal Design of Industrial refrigeration System
    out = constant.benchmark_function(16)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]
    hx = 0
    gx = zeros(g)
    fx = 63098.88 * x[1] * x[3] * x[11] + 5441.5 * x[1] ** 2 * x[11] + 115055.5 * x[1] ** 1.664 * x[5] + 6172.27 * x[1] ** 2 * x[5] + 63098.88 * x[0] * x[2] * \
         x[10] + 5441.5 * x[0] ** 2 * x[10] + 115055.5 * x[0] ** 1.664 * x[4] + 6172.27 * x[0] ** 2 * x[4] + 140.53 * x[0] * x[10] + 281.29 * x[2] * x[10] \
         + 70.26 * x[0] ** 2 + 281.29 * x[0] * x[2] + 281.29 * x[2] ** 2 + 14437 * x[7] ** 1.8812 * x[11] ** 0.3424 * x[9] * x[13] ** (-1) * x[0] ** 2 * \
         x[6] * x[8] ** (-1) + 20470.2 * x[6] ** (2.893) * x[10] ** 0.316 * x[0] ** 2
    gx[0] = 1.524 * x[6] ** (-1) - 1
    gx[1] = 1.524 * x[7] ** (-1) - 1
    gx[2] = 0.07789 * x[0] - 2 * x[6] ** (-1) * x[8] - 1
    gx[3] = 7.05305 * x[8] ** (-1) * x[0] ** 2 * x[9] * x[7] ** (-1) * x[1] ** (-1) * x[13] ** (-1) - 1
    gx[4] = 0.0833 / x[12] * x[13] - 1
    gx[5] = 0.04771 * x[9] * x[7] ** 1.8812 * x[11] ** 0.3424 - 1
    gx[6] = 0.0488 * x[8] * x[6] ** 1.893 * x[10] ** 0.316 - 1
    gx[7] = 0.0099 * x[0] / x[2] - 1
    gx[8] = 0.0193 * x[1] / x[3] - 1
    gx[9] = 0.0298 * x[0] / x[4] - 1
    gx[10] = 47.136 * x[1] ** 0.333 / x[9] * x[11] - 1.333 * x[7] * x[12] ** 2.1195 + 62.08 * x[12] ** 2.1195 * x[7] ** 0.2 / (x[11] * x[9]) - 1
    gx[11] = 0.056 * x[1] / x[5] - 1
    gx[12] = 2 / x[8] - 1
    gx[13] = 2 / x[9] - 1
    gx[14] = x[11] / x[10] - 1

    return fx, gx, hx


def p17(x):
    ## Tension/compression  spring  design (case 1)
    out = constant.benchmark_function(17)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]
    hx = 0
    fx = x[0] ** 2 * x[1] * (x[2] + 2)
    gx = zeros(g)
    gx[0] = 1 - (x[1] ** 3 * x[2]) / (71785 * x[0] ** 4)
    gx[1] = (4 * x[1] ** 2 - x[0] * x[1]) / (12566 * (x[1] * x[0] ** 3 - x[0] ** 4)) + 1 / (5108 * x[0] ** 2) - 1
    gx[2] = 1 - 140.45 * x[0] / (x[1] ** 2 * x[2])
    gx[3] = (x[0] + x[1]) / 1.5 - 1
    return fx, gx, hx


def p18(x):
    ## Update
    out = constant.benchmark_function(18)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]
    x[0] = 0.0625 * round(x[0])
    x[1] = 0.0625 * round(x[1])
    ## Pressure vessel design
    hx = 0
    gx = zeros(g)
    fx = 0.6224 * x[0] * x[2] * x[3] + 1.7781 * x[1] * x[2] ** 2 + 3.1661 * x[0] ** 2 * x[3] + 19.84 * x[0] ** 2 * x[2]
    gx[0] = -x[0] + 0.0193 * x[2]
    gx[1] = -x[1] + 0.00954 * x[2]
    gx[2] = -pi * x[2] ** 2 * x[3] - 4 / 3 * pi * x[2] ** 3 + 1296000
    gx[3] = x[3] - 240
    return fx, gx, hx


def p19(x):
    out = constant.benchmark_function(19)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]
    ## Welded beam design
    fx = 1.10471 * x[0] ** 2 * x[1] + 0.04811 * x[2] * x[3] * (14 + x[1])
    hx = 0
    P = 6000
    L = 14
    delta_max = 0.25
    E = 30 * 1e6
    G = 12 * 1e6
    T_max = 13600
    sigma_max = 30000
    Pc = 4.013 * E * sqrt(x[2] ** 2 * x[3] ** 6 / 30) / L ** 2 * (1 - x[2] / (2 * L) * sqrt(E / (4 * G)))
    sigma = 6 * P * L / (x[3] * x[2] ** 2)
    delta = 6 * P * L ** 3 / (E * x[2] ** 2 * x[3])
    J = 2 * (sqrt(2) * x[0] * x[1] * (x[1] ** 2 / 4 + (x[0] + x[2]) ** 2 / 4))
    R = sqrt(x[1] ** 2 / 4 + (x[0] + x[2]) ** 2 / 4)
    M = P * (L + x[1] / 2)
    ttt = M * R / J
    tt = P / (sqrt(2) * x[0] * x[1])
    t = sqrt(tt ** 2 + 2 * tt * ttt * x[1] / (2 * R) + ttt ** 2)
    ## constraints
    gx = zeros(g)
    gx[0] = t - T_max
    gx[1] = sigma - sigma_max
    gx[2] = x[0] - x[3]
    gx[3] = delta - delta_max
    gx[4] = P - Pc
    return fx, gx, hx


def p20(x):
    out = constant.benchmark_function(20)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]
    ## Three-bar truss design problem
    fx = (2 * sqrt(2) * x[0] + x[1]) * 100
    gx = zeros(g)
    gx[0] = (sqrt(2) * x[0] + x[1]) / (sqrt(2) * x[0] ** 2 + 2 * x[0] * x[1]) * 2 - 2
    gx[1] = x[1] / (sqrt(2) * x[0] ** 2 + 2 * x[0] * x[1]) * 2 - 2
    gx[2] = 1 / (sqrt(2) * x[1] + x[0]) * 2 - 2
    hx = 0
    return fx, gx, hx


def p21(x):
    ## Multiple disk clutch brake design problem
    out = constant.benchmark_function(21)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]

    ## parameters
    Mf = 3
    Ms = 40
    Iz = 55
    n = 250
    Tmax = 15
    s = 1.5
    delta = 0.5
    Vsrmax = 10
    rho = 0.0000078
    pmax = 1
    mu = 0.6
    Lmax = 30
    delR = 20
    Rsr = 2 / 3 * (x[1] ** 3 - x[0] ** 3) / (x[1] ** 2 * x[0] ** 2)
    Vsr = pi * Rsr * n / 30
    A = pi * (x[1] ** 2 - x[0] ** 2)
    Prz = x[3] / A
    w = pi * n / 30
    Mh = 2 / 3 * mu * x[3] * x[4] * (x[1] ** 3 - x[0] ** 3) / (x[1] ** 2 - x[0] ** 2)
    T = Iz * w / (Mh + Mf)
    hx = 0
    gx = zeros(g)
    fx = pi * (x[1] ** 2 - x[0] ** 2) * x[2] * (x[4] + 1) * rho
    gx[0] = -x[1] + x[0] + delR
    gx[1] = (x[4] + 1) * (x[2] + delta) - Lmax
    gx[2] = Prz - pmax
    gx[3] = Prz * Vsr - pmax * Vsrmax
    gx[4] = Vsr - Vsrmax
    gx[5] = T - Tmax
    gx[6] = s * Ms - Mh
    gx[7] = -T
    return fx, gx, hx


def p22(x):
    ## Planetary gear train design optimization problem
    out = constant.benchmark_function(22)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]

    ##parameter Initialization
    x = abs(x).astype(int)
    Pind = [3, 4, 5]
    mind = [1.75, 2, 2.25, 2.5, 2.75, 3.0]
    N1 = x[0]
    N2 = x[1]
    N3 = x[2]
    N4 = x[3]
    N5 = x[4]
    N6 = x[5]
    p = Pind[x[6]-1]
    m1 = mind[x[7]-1]
    m2 = mind[x[8]-1]
    ## objective function
    i1 = N6 / N4
    i01 = 3.11
    i2 = N6 * (N1 * N3 + N2 * N4) / (N1 * N3 * (N6 - N4))
    i02 = 1.84
    iR = -(N2 * N6 / (N1 * N3))
    i0R = -3.11
    fx = max([i1 - i01, i2 - i02, iR - i0R])
    ## constraints
    Dmax = 220
    dlt22 = 0.5
    dlt33 = 0.5
    dlt55 = 0.5
    dlt35 = 0.5
    dlt34 = 0.5
    dlt56 = 0.5
    beta = arccos(((N6 - N3) ** 2 + (N4 + N5) ** 2 - (N3 + N5) ** 2) / (2 * (N6 - N3) * (N4 + N5)))
    gx = zeros(g)
    gx[0] = m2 * (N6 + 2.5) - Dmax
    gx[1] = m1 * (N1 + N2) + m1 * (N2 + 2) - Dmax
    gx[2] = m2 * (N4 + N5) + m2 * (N5 + 2) - Dmax
    gx[3] = abs(m1 * (N1 + N2) - m2 * (N6 - N3)) - m1 - m2
    gx[4] = -((N1 + N2) * sin(pi / p) - N2 - 2 - dlt22)
    gx[5] = -((N6 - N3) * sin(pi / p) - N3 - 2 - dlt33)
    gx[6] = -((N4 + N5) * sin(pi / p) - N5 - 2 - dlt55)
    if beta == beta.real:
        gx[7] = (N3 + N5 + 2 + dlt35) ** 2 - ((N6 - N3) ** 2 + (N4 + N5) ** 2 - 2 * (N6 - N3) * (N4 + N5) * cos(2 * pi / p - beta))
    else:
        gx[7] = 1e6
    gx[8] = -(N6 - 2 * N3 - N4 - 4 - 2 * dlt34)
    gx[9] = -(N6 - N4 - 2 * N5 - 4 - 2 * dlt56)
    hx = remainder(N6 - N4, p)

    return fx, gx, hx


def p23(x):
    ## Step-cone pulley problem
    out = constant.benchmark_function(23)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]
    gx = zeros(g)
    hx = zeros(h)
    ## parameter Initialization
    d1 = x[0] * 1e-3
    d2 = x[1] * 1e-3
    d3 = x[2] * 1e-3
    d4 = x[3] * 1e-3
    w = x[4] * 1e-3
    N = 350
    N1 = 750
    N2 = 450
    N3 = 250
    N4 = 150
    rho = 7200
    a = 3
    mu = 0.35
    s = 1.75 * 1e6
    t = 8 * 1e-3
    ## objective function
    fx = rho * w * pi / 4 * (d1 ** 2 * (1 + (N1 / N) ** 2) + d2 ** 2 * (1 + (N2 / N) ** 2) + d3 ** 2 * (1 + (N3 / N) ** 2) + d4 ** 2 * (1 + (N4 / N) ** 2))
    ## constraint
    C1 = pi * d1 / 2 * (1 + N1 / N) + (N1 / N - 1) ** 2 * d1 ** 2 / (4 * a) + 2 * a
    C2 = pi * d2 / 2 * (1 + N2 / N) + (N2 / N - 1) ** 2 * d2 ** 2 / (4 * a) + 2 * a
    C3 = pi * d3 / 2 * (1 + N3 / N) + (N3 / N - 1) ** 2 * d3 ** 2 / (4 * a) + 2 * a
    C4 = pi * d4 / 2 * (1 + N4 / N) + (N4 / N - 1) ** 2 * d4 ** 2 / (4 * a) + 2 * a
    R1 = exp(mu * (pi - 2 * arcsin((N1 / N - 1) * d1 / (2 * a))))
    R2 = exp(mu * (pi - 2 * arcsin((N2 / N - 1) * d2 / (2 * a))))
    R3 = exp(mu * (pi - 2 * arcsin((N3 / N - 1) * d3 / (2 * a))))
    R4 = exp(mu * (pi - 2 * arcsin((N4 / N - 1) * d4 / (2 * a))))
    P1 = s * t * w * (1 - exp(-mu * (pi - 2 * arcsin((N1 / N - 1) * d1 / (2 * a))))) * pi * d1 * N1 / 60
    P2 = s * t * w * (1 - exp(-mu * (pi - 2 * arcsin((N2 / N - 1) * d2 / (2 * a))))) * pi * d2 * N2 / 60
    P3 = s * t * w * (1 - exp(-mu * (pi - 2 * arcsin((N3 / N - 1) * d3 / (2 * a))))) * pi * d3 * N3 / 60
    P4 = s * t * w * (1 - exp(-mu * (pi - 2 * arcsin((N4 / N - 1) * d4 / (2 * a))))) * pi * d4 * N4 / 60

    gx[0] = -R1 + 2
    gx[1] = -R2 + 2
    gx[2] = -R3 + 2
    gx[3] = -R4 + 2
    gx[4] = -P1 + (0.75 * 745.6998)
    gx[5] = -P2 + (0.75 * 745.6998)
    gx[6] = -P3 + (0.75 * 745.6998)
    gx[7] = -P4 + (0.75 * 745.6998)
    hx[0] = C1 - C2
    hx[1] = C1 - C3
    hx[2] = C1 - C4
    return fx, gx, hx


def OBJ11(x, n):
    a = x[0]
    b = x[1]
    c = x[2]
    e = x[3]
    f = x[4]
    l = x[5]
    Zmax = 99.9999
    P = 100
    if n == 1:
        def fhd(z):
            return P * b * sin(arccos((a ** 2 + (l - z) ** 2 + e ** 2 - b ** 2) / (2 * a * sqrt((l - z) ** 2 + e ** 2))) +
                               arccos((b ** 2 + (l - z) ** 2 + e ** 2 - a ** 2) / (2 * b * sqrt((l - z) ** 2 + e ** 2)))) / \
                   (2 * c * cos(arccos((a ** 2 + (l - z) ** 2 + e ** 2 - b ** 2) / (2 * a * sqrt((l - z) ** 2 + e ** 2))) + arctan(e / (l - z))))

        fhd_func = fhd
    else:
        def fhd(z):
            return -(P * b * sin(arccos((a ** 2 + (l - z) ** 2 + e ** 2 - b ** 2) / (2 * a * sqrt((l - z) ** 2 + e ** 2))) +
                                 arccos((b ** 2 + (l - z) ** 2 + e ** 2 - a ** 2) / (2 * b * sqrt((l - z) ** 2 + e ** 2)))) /
                     (2 * c * cos(arccos((a ** 2 + (l - z) ** 2 + e ** 2 - b ** 2) / (2 * a * sqrt((l - z) ** 2 + e ** 2))) + arctan(e / (l - z)))))

        fhd_func = fhd
    return fminbound(fhd_func, 0, Zmax)


def p24(x):  ## Not done
    ### Robot gripper problem
    out = constant.benchmark_function(24)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]

    a = x[0]
    b = x[1]
    c = x[2]
    e = x[3]
    ff = x[4]
    l = x[5]
    delta = x[6]
    Ymin = 50
    Ymax = 100
    YG = 150
    Zmax = 99.9999
    P = 100
    alpha_0 = arccos((a ** 2 + l ** 2 + e ** 2 - b ** 2) / (2 * a * sqrt(l ** 2 + e ** 2))) + arctan(e / l)
    beta_0 = arccos((b ** 2 + l ** 2 + e ** 2 - a ** 2) / (2 * b * sqrt(l ** 2 + e ** 2))) - arctan(e / l)
    alpha_m = arccos((a ** 2 + (l - Zmax) ** 2 + e ** 2 - b ** 2) / (2 * a * sqrt((l - Zmax) ** 2 + e ** 2))) + arctan(e / (l - Zmax))
    beta_m = arccos((b ** 2 + (l - Zmax) ** 2 + e ** 2 - a ** 2) / (2 * b * sqrt((l - Zmax) ** 2 + e ** 2))) - arctan(e / (l - Zmax))
    ## objective function
    fx = zeros(D)
    for i in range(0, D):
        fx[i] = -1 * OBJ11(x, 2) + OBJ11(x, 1)
    ## constraints
    Yxmin = 2 * (e + ff + c * sin(beta_m + delta))
    Yxmax = 2 * (e + ff + c * sin(beta_0 + delta))
    gx = zeros(g)
    gx[0] = Yxmin - Ymin
    gx[1] = -Yxmin
    gx[2] = Ymax - Yxmax
    gx[3] = Yxmax - YG
    gx[4] = l ** 2 + e ** 2 - (a + b) ** 2
    gx[5] = b ** 2 - (a - e) ** 2 - (l - Zmax) ** 2
    gx[6] = Zmax - l
    hx = 0
    tt = int(imag(fx[0]) != 0)
    fx[tt] = 1e4
    tt = int(imag(gx[0]) != 0)
    gx[tt] = 1e4
    return fx, gx, hx


def p25(x):
    ## Hydro-static thrust bearing design problem
    out = constant.benchmark_function(25)
    D, g, h, xmin, xmax = out["D"], out["g"], out["h"], out["xmin"], out["xmax"]

    R = x[0]
    Ro = x[1]
    mu = x[2]
    Q = x[3]
    gamma = 0.0307
    C = 0.5
    n = -3.55
    C1 = 10.04
    Ws = 101000
    Pmax = 1000
    delTmax = 50
    hmin = 0.001
    gg = 386.4
    N = 750
    P = (log10(log10(8.122 * 1e6 * mu + 0.8)) - C1) / n
    delT = 2 * (10 ** P - 560)
    Ef = 9336 * Q * gamma * C * delT
    h = (2 * pi * N / 60) ** 2 * 2 * pi * mu / Ef * (R ** 4 / 4 - Ro ** 4 / 4) - 1e-5
    Po = (6 * mu * Q / (pi * h ** 3)) * log(R / Ro)
    W = pi * Po / 2 * (R ** 2 - Ro ** 2) / (log(R / Ro) - 1e-5)
    ##  objective function
    fx = (Q * Po / 0.7 + Ef) / 12
    ##  constraints
    gx = zeros(g)
    hx = 0
    gx[0] = Ws - W
    gx[1] = Po - Pmax
    gx[2] = delT - delTmax
    gx[3] = hmin - h
    gx[4] = Ro - R
    gx[5] = gamma / (gg * Po) * (Q / (2 * pi * R * h)) - 0.001
    gx[6] = W / (pi * (R ** 2 - Ro ** 2) + 1e-5) - 5000
    return fx, gx, hx

