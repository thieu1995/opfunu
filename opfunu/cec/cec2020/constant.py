#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:14, 20/09/2020                                                        %
#
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy import ones, zeros, concatenate

D = [9, 11, 7, 6, 9, 38, 48, 2, 3, 3, 7, 7, 5, 10, 7, 14, 3, 4, 4, 2, 5, 9, 5, 7, 4, 22, 10, 10, 4, 3, 4, 5,
     30, 118, 153, 158, 126, 126, 126, 76, 74, 86, 86, 30, 25, 25, 25, 30, 30, 30, 59, 59, 59, 59, 64, 64, 64]
gn = [0, 0, 14, 1, 2, 0, 0, 2, 1, 3, 4, 9, 3, 10, 11, 15, 4, 4, 5, 3, 8, 10, 8, 7, 7, 86, 3, 9, 1, 8, 1, 6, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 105, 24, 24, 24,
      29, 29, 29, 14, 14, 14, 14, 0, 0, 0]
hn = [8, 9, 0, 4, 4, 32, 38, 0, 1, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 108, 148, 148, 116, 116, 116, 76, 74, 76, 76, 0, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 6, 6]

# bound constraint definitions for all 53 test functions
xmin1 = [0, 0, 0, 0, 1000, 0, 100, 100, 100]
xmax1 = [10, 200, 100, 200, 2000000, 600, 600, 600, 900]
xmin2 = [10 ** 4, 10 ** 4, 10 ** 4, 0, 0, 0, 100, 100, 100, 100, 100]
xmax2 = [0.819 * 10 ** 6, 1.131 * 10 ** 6, 2.05 * 10 ** 6, 0.05074, 0.05074, 0.05074, 200, 300, 300, 300, 400]
xmin3 = [1000, 0, 2000, 0, 0, 0, 0]
xmax3 = [2000, 100, 4000, 100, 100, 20, 200]
xmin4 = [0, 0, 0, 0, 1e-5, 1e-5]
xmax4 = [1, 1, 1, 1, 16, 16]
xmin5 = zeros(D[4])
xmax5 = [100, 200, 100, 100, 100, 100, 200, 100, 200]
xmin6 = zeros(D[5])
xmax6 = [90, 150, 90, 150, 90, 90, 150, 90, 90, 90, 150, 150, 90, 90, 150, 90, 150, 90, 150, 90, 1, 1.2,
         1, 1, 1, 0.5, 1, 1, 0.5, 0.5, 0.5, 1.2, 0.5, 1.2, 1.2, 0.5, 1.2, 1.2]
xmin7 = zeros(D[6])
xmin7[23] = xmin7[25] = xmin7[27] = xmin7[30] = 0.849999
xmax7 = ones(D[6])
xmax7[3] = 140
xmax7[24] = xmax7[26] = xmax7[31] = xmax7[34] = xmax7[36] = xmax7[28] = 30
xmax7[1] = xmax7[2] = xmax7[4] = xmax7[12:15] = 90
xmax7[0] = xmax7[5:12] = xmax7[15:20] = 35
xmin8 = [0, -0.51]
xmax8 = [1.6, 1.49]
xmin9 = [0.5, 0.5, -0.51]
xmax9 = [1.4, 1.4, 1.49]
xmin10 = [0.2, -2.22554, -0.51]
xmax10 = [1, -1, 1.49]
xmin11 = [0, 0, 0, 0, -0.51, -0.51, 0]
xmax11 = [20, 20, 10, 10, 1.49, 1.49, 40]
xmin12 = [0, 0, 0, -0.51, -0.51, -0.51, -0.51]
xmax12 = [100, 100, 100, 1.49, 1.49, 1.49, 1.49]
xmin13 = [27, 27, 27, 77.51, 32.51]
xmax13 = [45, 45, 45, 102.49, 45.49]
xmin14 = [0.51, 0.51, 0.51, 250, 250, 250, 6, 4, 40, 10]
xmax14 = [3.49, 3.49, 3.49, 2500, 2500, 2500, 20, 16, 700, 450]
xmin15 = [2.6, 0.7, 17, 7.3, 7.3, 2.9, 5]
xmax15 = [3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5]
xmin16 = 0.001 * ones(D[15])
xmax16 = +5 * ones(D[15])
xmin17 = [0.05, 0.25, 2.00]
xmax17 = [2, 1.3, 15.0]
xmin18 = [0.51, 0.51, 10, 10]
xmax18 = [99.49, 99.49, 200, 200]
xmin19 = [0.125, 0.1, 0.1, 0.1]
xmax19 = [2, 10, 10, 2]
xmin20 = zeros(D[19])
xmax20 = 1 * ones(D[19])
xmin21 = [60, 90, 1, 0, 2]
xmax21 = [80, 110, 3, 1000, 9]
xmin22 = [16.51, 13.51, 13.51, 16.51, 13.51, 47.51, 0.51, 0.51, 0.51]
xmax22 = [96.49, 54.49, 51.49, 46.49, 51.49, 124.49, 3.49, 6.49, 6.49]
xmin23 = [0, 0, 0, 0, 0]
xmax23 = [60, 60, 90, 90, 90]
xmin24 = [10, 10, 100, 0, 10, 100, 1]
xmax24 = [150, 150, 200, 50, 150, 300, 3.14]
xmin25 = [1, 1, 1e-6, 1]
xmax25 = [16, 16, 16 * 1e-6, 16]
xmin26 = concatenate((6.51 * ones(8), 0.51 * ones(14)), axis=0)
xmax26 = concatenate((76.49 * ones(8), 4.49 * ones(4), 9.49 * ones(10)), axis=0)
xmin27 = 0.645e-4 * ones(D[26])
xmax27 = 50e-4 * ones(D[26])
xmin28 = [125, 10.5, 4.51, 0.515, 0.515, 0.4, 0.6, 0.3, 0.02, 0.6]
xmax28 = [150, 31.5, 50.49, 0.6, 0.6, 0.5, 0.7, 0.4, 0.1, 0.85]
xmin29 = [20, 1, 20, 0.1]
xmax29 = [50, 10, 50, 60]
xmin30 = [0.51, 0.6, 0.51]
xmax30 = [70.49, 3, 42.49]
xmin31 = 12. * ones(4)
xmax31 = 60. * ones(4)
xmin32 = [78, 33, 27, 27, 27]
xmax32 = [102, 45, 45, 45, 45]
xmin33 = 0.001 * ones(D[32])
xmax33 = ones(D[32])
xmin34 = -1 * ones(D[33])
xmax34 = +1 * ones(D[33])
xmin35 = -1 * ones(D[34])
xmax35 = +1 * ones(D[34])
xmin36 = -1 * ones(D[35])
xmax36 = +1 * ones(D[35])
xmin37 = -1 * ones(D[36])
xmin37[117: 126] = 0
xmax37 = +1 * ones(D[36])
xmin38 = -1 * ones(D[37])
xmin38[117: 126] = 0
xmax38 = +1 * ones(D[37])
xmin39 = -1 * ones(D[38])
xmin39[117: 126] = 0
xmax39 = +1 * ones(D[38])
xmin40 = -1 * ones(D[39])
xmin40[75: 76] = 0
xmax40 = +1 * ones(D[39])
xmax40[75: 76] = 2
xmin41 = -1 * ones(D[40])
xmax41 = +1 * ones(D[40])
xmin42 = -1 * ones(D[41])
xmin42[75: 76] = 0
xmin42[77: 86] = 0
xmax42 = +1 * ones(D[41])
xmax42[75: 76] = 2
xmax42[77: 86] = 500
xmin43 = -1 * ones(D[42])
xmin43[75: 76] = 0
xmin43[77: 86] = 0
xmax43 = +1 * ones(D[42])
xmax43[75: 76] = 2
xmax43[77: 86] = 500
xmin44 = 40 * ones(D[43])
xmax44 = 1960 * ones(D[43])
xmin45 = zeros(D[44])
xmax45 = +90 * ones(D[44])
xmin46 = zeros(D[45])
xmax46 = +90 * ones(D[45])
xmin47 = zeros(D[46])
xmax47 = +90 * ones(D[46])
xmin48 = zeros(D[47])
xmax48 = +90 * ones(D[47])
xmin49 = zeros(D[48])
xmax49 = +90 * ones(D[48])
xmin50 = zeros(D[49])
xmax50 = +90 * ones(D[49])
xmin51 = zeros(D[50])
xmax51 = 10. * ones(D[50])
xmin52 = zeros(D[51])
xmax52 = 10. * ones(D[51])
xmin53 = zeros(D[52])
xmax53 = 10. * ones(D[52])
xmin54 = zeros(D[53])
xmax54 = 10. * ones(D[53])
xmin55 = zeros(D[54])
xmax55 = 10. * ones(D[54])
xmin56 = zeros(D[55])
xmax56 = 10. * ones(D[55])
xmin57 = zeros(D[56])
xmax57 = 10. * ones(D[56])

def benchmark_function(idx):
    # % prob_k -> Index of problem.
    # % D[idx  -> Dimension of the problem.
    # % par.g  -> Number of inequility constraints.
    # % par.h  -> Number of equality constraints.
    # % par.xmin -> lower bound of decision variables.
    # % par.xmax -> upper bound of decision variables.
    xmin = globals()['xmin' + str(idx)]
    xmax = globals()['xmax' + str(idx)]
    return {"D": D[idx-1], "g": gn[idx-1], "h": hn[idx-1], "xmin": xmin, "xmax": xmax}