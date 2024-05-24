#!/usr/bin/env python
# Created by "Thieu" at 16:55, 24/05/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from opfunu.cec_based import F12010

# Visualize opfunu function using method in object
f0 = F12010()

f0.plot_2d(selected_dims=(2, 3), n_points=300, ct_cmap="viridis", ct_levels=30, ct_alpha=0.7,
           fixed_strategy="mean", fixed_values=None, title="Contour map of the F1 CEC 2010 function",
           x_label=None, y_label=None, figsize=(10, 8), filename="2d-f12010", exts=(".png", ".pdf"), verbose=True)

f0.plot_3d(selected_dims=(1, 6), n_points=500, ct_cmap="viridis", ct_levels=30, ct_alpha=0.7,
           fixed_strategy="mean", fixed_values=None, title="3D visualization of the F1 CEC 2010 function",
           x_label=None, y_label=None, figsize=(10, 8), filename="3d-f12010", exts=(".png", ".pdf"), verbose=True)

## Visualize opfunu function using utility function
from opfunu import draw_2d, draw_3d

draw_2d(f0.evaluate, f0.lb, f0.ub, selected_dims=(2, 3), n_points=300)
draw_3d(f0.evaluate, f0.lb, f0.ub, selected_dims=(2, 3), n_points=300)
