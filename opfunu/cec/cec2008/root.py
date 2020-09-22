#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 12:31, 20/04/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                  %
#-------------------------------------------------------------------------------------------------------%

import pkg_resources
from pandas import read_csv


class Root:
    def __init__(self, f_name=None, f_shift_data_file=None, f_ext='.txt', f_bias=None):
        self.f_name = f_name
        self.f_shift_data_file = f_shift_data_file
        self.f_ext = f_ext
        self.f_bias = f_bias
        self.current_path = pkg_resources.resource_filename("opfunu", "cec/cec2008/")
        self.support_path_data = pkg_resources.resource_filename("opfunu", "cec/cec2008/support_data/")

    def load_shift_data(self):
        data = read_csv(self.support_path_data + self.f_shift_data_file + self.f_ext, delimiter='\s+', index_col=False, header=None)
        return data.values.reshape((-1))

    def load_matrix_data(self, data_file=None):
        data = read_csv(self.support_path_data + data_file + self.f_ext, delimiter='\s+', index_col=False, header=None)
        return data.values

