#!/usr/bin/env python
# Created by "Thieu" at 14:14, 20/09/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np


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
xmin5 = np.zeros(D[4]).tolist()
xmax5 = [100, 200, 100, 100, 100, 100, 200, 100, 200]
xmin6 = np.zeros(D[5]).tolist()
xmax6 = [90, 150, 90, 150, 90, 90, 150, 90, 90, 90, 150, 150, 90, 90, 150, 90, 150, 90, 150, 90, 1, 1.2,
         1, 1, 1, 0.5, 1, 1, 0.5, 0.5, 0.5, 1.2, 0.5, 1.2, 1.2, 0.5, 1.2, 1.2]
xmin7 = np.zeros(D[6])
xmin7[23] = xmin7[25] = xmin7[27] = xmin7[30] = 0.849999
xmax7 = np.ones(D[6])
xmax7[3] = 140
xmax7[24] = xmax7[26] = xmax7[31] = xmax7[34] = xmax7[36] = xmax7[28] = 30
xmax7[1] = xmax7[2] = xmax7[4] = xmax7[12:15] = 90
xmax7[0] = xmax7[5:12] = xmax7[15:20] = 35
xmin7 = xmin7.tolist()
xmax7 = xmax7.tolist()
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
xmin16 = (0.001 * np.ones(D[15])).tolist()
xmax16 = (+5 * np.ones(D[15])).tolist()
xmin17 = [0.05, 0.25, 2.00]
xmax17 = [2, 1.3, 15.0]
xmin18 = [0.51, 0.51, 10, 10]
xmax18 = [99.49, 99.49, 200, 200]
xmin19 = [0.125, 0.1, 0.1, 0.1]
xmax19 = [2, 10, 10, 2]
xmin20 = np.zeros(D[19]).tolist()
xmax20 = np.ones(D[19]).tolist()
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
xmin26 = np.concatenate((6.51 * np.ones(8), 0.51 * np.ones(14)), axis=0).tolist()
xmax26 = np.concatenate((76.49 * np.ones(8), 4.49 * np.ones(4), 9.49 * np.ones(10)), axis=0).tolist()
xmin27 = (0.645e-4 * np.ones(D[26])).tolist()
xmax27 = (50e-4 * np.ones(D[26])).tolist()
xmin28 = [125, 10.5, 4.51, 0.515, 0.515, 0.4, 0.6, 0.3, 0.02, 0.6]
xmax28 = [150, 31.5, 50.49, 0.6, 0.6, 0.5, 0.7, 0.4, 0.1, 0.85]
xmin29 = [20, 1, 20, 0.1]
xmax29 = [50, 10, 50, 60]
xmin30 = [0.51, 0.6, 0.51]
xmax30 = [70.49, 3, 42.49]
xmin31 = (12. * np.ones(4)).tolist()
xmax31 = (60. * np.ones(4)).tolist()
xmin32 = [78, 33, 27, 27, 27]
xmax32 = [102, 45, 45, 45, 45]
xmin33 = (0.001 * np.ones(D[32])).tolist()
xmax33 = np.ones(D[32]).tolist()
xmin34 = (-1 * np.ones(D[33])).tolist()
xmax34 = (+1 * np.ones(D[33])).tolist()
xmin35 = (-1 * np.ones(D[34])).tolist()
xmax35 = (+1 * np.ones(D[34])).tolist()
xmin36 = (-1 * np.ones(D[35])).tolist()
xmax36 = (+1 * np.ones(D[35])).tolist()
xmin37 = -1 * np.ones(D[36])
xmin37[117: 126] = 0
xmin37 = xmin37.tolist()
xmax37 = (+1 * np.ones(D[36])).tolist()
xmin38 = -1 * np.ones(D[37])
xmin38[117: 126] = 0
xmax38 = +1 * np.ones(D[37])
xmin39 = -1 * np.ones(D[38])
xmin39[117: 126] = 0
xmin39 = xmin39.tolist()
xmax39 = (+1 * np.ones(D[38])).tolist()
xmin40 = -1 * np.ones(D[39])
xmin40[75: 76] = 0
xmin40 = xmin40.tolist()
xmax40 = +1 * np.ones(D[39])
xmax40[75: 76] = 2
xmax40 = xmax40.tolist()
xmin41 = (-1 * np.ones(D[40])).tolist()
xmax41 = (+1 * np.ones(D[40])).tolist()
xmin42 = -1 * np.ones(D[41])
xmin42[75: 76] = 0
xmin42[77: 86] = 0
xmax42 = +1 * np.ones(D[41])
xmax42[75: 76] = 2
xmax42[77: 86] = 500
xmin42 = xmin42.tolist()
xmax42 = xmax42.tolist()

xmin43 = -1 * np.ones(D[42])
xmin43[75: 76] = 0
xmin43[77: 86] = 0
xmax43 = +1 * np.ones(D[42])
xmax43[75: 76] = 2
xmax43[77: 86] = 500
xmin43 = xmin43.tolist()
xmax43 = xmax43.tolist()

xmin44 = (40 * np.ones(D[43])).tolist()
xmax44 = (1960 * np.ones(D[43])).tolist()
xmin45 = np.zeros(D[44]).tolist()
xmax45 = (+90 * np.ones(D[44])).tolist()
xmin46 = np.zeros(D[45]).tolist()
xmax46 = (+90 * np.ones(D[45])).tolist()
xmin47 = np.zeros(D[46]).tolist()
xmax47 = (+90 * np.ones(D[46])).tolist()
xmin48 = np.zeros(D[47]).tolist()
xmax48 = (+90 * np.ones(D[47])).tolist()
xmin49 = np.zeros(D[48]).tolist()
xmax49 = (+90 * np.ones(D[48])).tolist()
xmin50 = np.zeros(D[49]).tolist()
xmax50 = (+90 * np.ones(D[49])).tolist()
xmin51 = np.zeros(D[50]).tolist()
xmax51 = (10. * np.ones(D[50])).tolist()
xmin52 = np.zeros(D[51]).tolist()
xmax52 = (10. * np.ones(D[51])).tolist()
xmin53 = np.zeros(D[52]).tolist()
xmax53 = (10. * np.ones(D[52])).tolist()
xmin54 = np.zeros(D[53]).tolist()
xmax54 = (10. * np.ones(D[53])).tolist()
xmin55 = np.zeros(D[54]).tolist()
xmax55 = (10. * np.ones(D[54])).tolist()
xmin56 = np.zeros(D[55]).tolist()
xmax56 = (10. * np.ones(D[55])).tolist()
xmin57 = np.zeros(D[56]).tolist()
xmax57 = (10. * np.ones(D[56])).tolist()


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
