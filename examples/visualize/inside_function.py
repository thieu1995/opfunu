#!/usr/bin/env python
# Created by "Thieu" at 16:55, 24/05/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from opfunu.cec_based import F12010

# Visualize opfunu function using method in object
f0 = F12010()
f0.plot_2d(selected_dims=(2, 3), n_points=300)
f0.plot_3d(selected_dims=(2, 3), n_points=300)

## Visualize opfunu function using utility function
from opfunu import draw_2d, draw_3d

draw_2d(f0.evaluate, f0.lb, f0.ub, selected_dims=(2, 3), n_points=300)
draw_3d(f0.evaluate, f0.lb, f0.ub, selected_dims=(2, 3), n_points=300)
