#!/usr/bin/env python
# Created by "Thieu" at 20:46, 29/06/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import re
from io import BytesIO
from pathlib import Path
import matplotlib.pyplot as plt
import platform
import numpy as np
import requests
from PIL import Image
from matplotlib import cm
from matplotlib.ticker import ScalarFormatter, FuncFormatter


cmap = [(0, '#2f9599'), (0.45, '#eeeeee'), (1, '#8800ff')]
cmap = cm.colors.LinearSegmentedColormap.from_list('Custom', cmap, N=256)
SUPPORTED_ARRAY = (list, tuple, np.ndarray)


def __clean_filename__(filename):
    chars_to_remove = ["`", "~", "!", "@", "#", "$", "%", "^", "&", "*", ":", ",", "<", ">", ";", "+", "|"]
    regular_expression = '[' + re.escape(''.join(chars_to_remove)) + ']'

    temp = filename.encode("ascii", "ignore")
    fname = temp.decode()                           # Removed all non-ascii characters
    fname = re.sub(regular_expression, '', fname)   # Removed all special characters
    fname.replace("_", "-")                         # Replaced _ by -
    return fname


def __check_filepath__(filename):
    filename.replace("\\", "/")                     # For better handling the parent folder
    if "/" in filename:
        list_names = filename.split("/")[:-1]       # Remove last element because it is filename
        filepath = "/".join(list_names)
        Path(filepath).mkdir(parents=True, exist_ok=True)
    return filename


def custom_formatter(x, pos):
    # Define a custom tick formatter function
    if abs(x) >= 1e4:
        return '{:.1e}'.format(x)
    else:
        return '{:.2f}'.format(x)


def draw_latex(latex, title="Latex equation", figsize=(8, 3), dpi=500,
               filename=None, exts=(".png", ".pdf"), verbose=True):
    """
    Draw latex equation.

    Parameters
    ----------
    latex : equation
        Your latex equation, you can test on the website: https://latex.codecogs.com/
    title : str
        Title for the figure
    figsize : tuple, default=(10, 8)
        The figure size with format of tuple (width, height)
    dpi : int, default=500
        The dot per inches (DPI) - indicate the number of dots per inch.
    filename : str, default = None
        Set the file name, If None, the file will not be saved
    exts : list, tuple, np.ndarray
        The list of extensions to save file, for example: (".png", ".pdf", ".jpg")
    verbose : bool
        Show the figure or not. It will not show on linux system.
    """
    base_url = 'https://latex.codecogs.com/png.latex?\dpi{' + str(dpi) + '}'
    url = f'{base_url}{latex}'
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')

    if filename is not None:
        filepath = __check_filepath__(__clean_filename__(filename))
        for idx, ext in enumerate(exts):
            plt.savefig(f"{filepath}{ext}", bbox_inches='tight')
            # img.convert('RGB').save(f'{filepath}{ext}')
    if platform.system() != "Linux" and verbose:
        plt.show()
    plt.close()


def draw_2d(func, lb=None, ub=None, selected_dims=None, n_points=1000,
            ct_cmap="viridis", ct_levels=30, ct_alpha=0.7,
            fixed_strategy="mean", fixed_values=None,
            title="Contour map of the function", x_label=None, y_label=None,
            figsize=(10, 8), filename=None, exts=(".png", ".pdf"), verbose=True):
    """
    Draw 2D contour of the function.

    Parameters
    ----------
    func : callable
        The callable function that is used to calculate the value
    lb : list, tuple, np.ndarray
        The lower bound of the variables, should be a list, tuple, or numpy array.
    ub : list, tuple, np.ndarray
        The upper bound of the variables, should be a list, tuple, or numpy array.
    selected_dims : list, tuple, np.ndarray
        The selected dimensions you want to draw.
        If your function has only 2 dimensions, it will select (1, 2) automatically.
    n_points : int
        The number of points that will be used to draw the contour
    ct_cmap : str
        The cmap of matplotlib.pyplot.contourf function (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html)
    ct_levels : int
        The levels parameter of contourf function
    ct_alpha : float
        The alpha parameter of contourf function
    fixed_strategy : str
        The selected strategy to set values for other dimensions.
        When your function has > 2 dimensions, you need to set a fixed value for other dimensions to be able to calculate value.
        List of available strategy: ["min", "max", "mean', "values", "zero"]
            + min: Set the other dimensions by its lower bound
            + max: Set the other dimensions by its upper bound
            + mean: Set the other dimensions by it average value (lower bound + upper bound) / 2
            + zero: Set the other dimensions by 0
            + values: Set the other dimensions by your passed values through the parameter: `fixed_values`.

    fixed_values : list, tuple, np.ndarray
        Fixed values for all dimensions (length should be the same as lower bound), the selected dimensions will be replaced in the drawing process.
    title : str
        Title for the figure
    x_label : str
        Set the x label
    y_label : str
        Set the y label
    figsize : tuple, default=(10, 8)
        The figure size with format of tuple (width, height)
    filename : str, default = None
        Set the file name, If None, the file will not be saved
    exts : list, tuple, np.ndarray
        The list of extensions to save file, for example: (".png", ".pdf", ".jpg")
    verbose : bool
        Show the figure or not. It will not show on linux system.
    """
    if isinstance(lb, SUPPORTED_ARRAY) and isinstance(ub, SUPPORTED_ARRAY):
        if len(lb) == len(ub):
            lb = np.array(lb)
            ub = np.array(ub)
        else:
            raise ValueError(f"Length of lb and ub should be equal.")
    else:
        raise TypeError(f"Type of lb and ub should be a list, tuple or np.ndarray.")
    if len(lb) == 2 or selected_dims is None:
        selected_dims = (1, 2)
    if isinstance(selected_dims, SUPPORTED_ARRAY) and len(selected_dims) == 2:
        selected_dims = tuple(selected_dims)
    else:
        raise TypeError(f"selected_dims should be a list of 2 selected dimensions.")

    selected_dims = (min(selected_dims), max(selected_dims))
    if selected_dims[0] < 1 or selected_dims[1] > len(lb):
        raise ValueError(f"The selected_dims's values should >= 1 and <= {len(lb)}.")
    idx_dims = (selected_dims[0] - 1, selected_dims[1] - 1)

    # Generate a grid of points for the d1-th and d2-th dimensions
    d1 = np.linspace(lb[idx_dims[0]], ub[idx_dims[0]], n_points)
    d2 = np.linspace(lb[idx_dims[1]], ub[idx_dims[1]], n_points)
    D1, D2 = np.meshgrid(d1, d2)

    # Fix the other dimensions to zero (or another value within the domain)
    if fixed_strategy == "mean":
        mm_values = (lb + ub) / 2
    elif fixed_strategy == "min":
        mm_values = lb
    elif fixed_strategy == "max":
        mm_values = ub
    elif fixed_strategy == "values":
        mm_values = fixed_values
    else:
        mm_values = np.zeros(len(lb))

    # Combine the fixed and varying dimensions into a single array
    solution = np.full((n_points, n_points, len(lb)), np.array(mm_values))
    solution[:, :, idx_dims[0]] = D1  # d1-th dimension
    solution[:, :, idx_dims[1]] = D2  # d2-th dimension

    # Compute the function values using vectorized operations
    Z = np.apply_along_axis(func, axis=-1, arr=solution)

    # Plot the function
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=figsize)

    cont = plt.contourf(D1, D2, Z, levels=ct_levels, cmap=ct_cmap, alpha=ct_alpha)
    cbar = plt.colorbar(cont)
    cbar.set_label('Function Value')
    # cbar.formatter = ScalarFormatter()
    # cbar.formatter.set_scientific(True)
    # cbar.formatter.set_powerlimits((-2, 2))
    # cbar.update_ticks()

    cbar.formatter = FuncFormatter(custom_formatter)
    cbar.update_ticks()

    plt.title(title)
    if x_label is None:
        x_label = f'Dimension X{selected_dims[0]}'
    if y_label is None:
        y_label = f'Dimension X{selected_dims[1]}'
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if filename is not None:
        filepath = __check_filepath__(__clean_filename__(filename))
        for idx, ext in enumerate(exts):
            plt.savefig(f"{filepath}{ext}", bbox_inches='tight')
    if platform.system() != "Linux" and verbose:
        plt.show()
    plt.close()


def draw_3d(func, lb=None, ub=None, selected_dims=None, n_points=1000,
            ct_cmap="viridis", ct_levels=30, ct_alpha=0.7,
            fixed_strategy="mean", fixed_values=None,
            title="3D visualization of the function", x_label=None, y_label=None,
            figsize=(10, 8), filename=None, exts=(".png", ".pdf"), verbose=True):
    """
    Draw 3D of the function.

    Parameters
    ----------
    func : callable
        The callable function that is used to calculate the value
    lb : list, tuple, np.ndarray
        The lower bound of the variables, should be a list, tuple, or numpy array.
    ub : list, tuple, np.ndarray
        The upper bound of the variables, should be a list, tuple, or numpy array.
    selected_dims : list, tuple, np.ndarray
        The selected dimensions you want to draw.
        If your function has only 2 dimensions, it will select (1, 2) automatically.
    n_points : int
        The number of points that will be used to draw the contour
    ct_cmap : str
        The cmap of matplotlib.pyplot.contourf function (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html)
    ct_levels : int
        The levels parameter of contourf function
    ct_alpha : float
        The alpha parameter of contourf function
    fixed_strategy : str
        The selected strategy to set values for other dimensions.
        When your function has > 2 dimensions, you need to set a fixed value for other dimensions to be able to calculate value.
        List of available strategy: ["min", "max", "mean', "values", "zero"]
            + min: Set the other dimensions by its lower bound
            + max: Set the other dimensions by its upper bound
            + mean: Set the other dimensions by it average value (lower bound + upper bound) / 2
            + zero: Set the other dimensions by 0
            + values: Set the other dimensions by your passed values through the parameter: `fixed_values`.

    fixed_values : list, tuple, np.ndarray
        Fixed values for all dimensions (length should be the same as lower bound), the selected dimensions will be replaced in the drawing process.
    title : str
        Title for the figure
    x_label : str
        Set the x label
    y_label : str
        Set the y label
    figsize : tuple, default=(10, 8)
        The figure size with format of tuple (width, height)
    filename : str, default = None
        Set the file name, If None, the file will not be saved
    exts : list, tuple, np.ndarray
        The list of extensions to save file, for example: (".png", ".pdf", ".jpg")
    verbose : bool
        Show the figure or not. It will not show on linux system.
    """
    if isinstance(lb, SUPPORTED_ARRAY) and isinstance(ub, SUPPORTED_ARRAY):
        if len(lb) == len(ub):
            lb = np.array(lb)
            ub = np.array(ub)
        else:
            raise ValueError(f"Length of lb and ub should be equal.")
    else:
        raise TypeError(f"Type of lb and ub should be a list, tuple or np.ndarray.")
    if len(lb) == 2 or selected_dims is None:
        selected_dims = (1, 2)
    if isinstance(selected_dims, SUPPORTED_ARRAY) and len(selected_dims) == 2:
        selected_dims = tuple(selected_dims)
    else:
        raise TypeError(f"selected_dims should be a list of 2 selected dimensions.")

    selected_dims = (min(selected_dims), max(selected_dims))
    if selected_dims[0] < 1 or selected_dims[1] > len(lb):
        raise ValueError(f"The selected_dims's values should >= 1 and <= {len(lb)}.")
    idx_dims = (selected_dims[0] - 1, selected_dims[1] - 1)

    # Generate a grid of points for the d1-th and d2-th dimensions
    d1 = np.linspace(lb[idx_dims[0]], ub[idx_dims[0]], n_points)
    d2 = np.linspace(lb[idx_dims[1]], ub[idx_dims[1]], n_points)
    D1, D2 = np.meshgrid(d1, d2)

    # Fix the other dimensions to zero (or another value within the domain)
    if fixed_strategy == "mean":
        mm_values = (lb + ub) / 2
    elif fixed_strategy == "min":
        mm_values = lb
    elif fixed_strategy == "max":
        mm_values = ub
    elif fixed_strategy == "values":
        mm_values = fixed_values
    else:
        mm_values = np.zeros(len(lb))

    # Combine the fixed and varying dimensions into a single array
    solution = np.full((n_points, n_points, len(lb)), np.array(mm_values))
    solution[:, :, idx_dims[0]] = D1  # d1-th dimension
    solution[:, :, idx_dims[1]] = D2  # d2-th dimension

    # Compute the function values using vectorized operations
    Z = np.apply_along_axis(func, axis=-1, arr=solution)

    # Plot the function
    plt.rcParams.update({'font.size': 14})

    # Create a 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(D1, D2, Z, cmap=ct_cmap, edgecolor='none', alpha=ct_alpha)

    # Add contours
    ax.contour(D1, D2, Z, zdir='z', offset=np.min(Z), cmap=ct_cmap, levels=ct_levels)

    # Add a color bar which maps values to colors
    cbar_ax = fig.add_axes([0.825, 0.175, 0.05, 0.55])
    cbar = fig.colorbar(surf, cax=cbar_ax, label='Function Value', shrink=0.5, aspect=5,)
    # cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Function Value')
    cbar.formatter = ScalarFormatter()
    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((-2, 2))
    cbar.update_ticks()

    # Apply the custom tick formatter to the color bar
    cbar_ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))

    if x_label is None:
        x_label = f'Dimension X{selected_dims[0]}'
    if y_label is None:
        y_label = f'Dimension X{selected_dims[1]}'
    # Set plot title and labels with increased padding
    ax.set_title(title)
    ax.set_xlabel(x_label, labelpad=10)
    ax.set_ylabel(y_label, labelpad=10)
    ax.set_zlabel("Function Value", labelpad=16)

    # Set the elevation and azimuth angles to adjust the viewing angle
    ax.view_init(elev=20, azim=-120)

    # Use scientific notation for large tick values
    formatter = ScalarFormatter()
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    # Apply the custom tick formatter to the z-axis
    ax.zaxis.set_major_formatter(FuncFormatter(custom_formatter))

    if filename is not None:
        filepath = __check_filepath__(__clean_filename__(filename))
        for idx, ext in enumerate(exts):
            plt.savefig(f"{filepath}{ext}", bbox_inches='tight')
    if platform.system() != "Linux" and verbose:
        plt.show()
    plt.close()
