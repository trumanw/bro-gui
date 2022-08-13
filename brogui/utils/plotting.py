import os
from typing import Union, Optional, List

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from nextorch.plotting import add_x_slice_2d, add_y_slice_2d, add_z_slice_2d, colormap, set_axis_values, figformat, backgroundtransparency
from nextorch.utils import unitscale_xv, MatrixLike2d

def sampling_3d(
    Xs: Union[MatrixLike2d, List[MatrixLike2d]], 
    X_ranges: Optional[MatrixLike2d] = None,
    x_indices: Optional[List[int]] = [0, 1, 2],
    X_names: Optional[List[str]] = None, 
    slice_axis: Optional[Union[str, int]] = None, 
    slice_value: Optional[float] = None, 
    slice_value_real: Optional[float] = None, 
    design_names: Optional[Union[str, List[str]]] = None,
    save_fig: Optional[bool] = False,
    save_path: Optional[str] = None):
    """Plot sampling plan(s) in 3 dimensional space
    X must be 3 dimensional

    Parameters
    ----------
    Xs : Union[MatrixLike2d, List[MatrixLike2d]]
        The set of sampling plans in a unit scale,
        Can be a list of matrices or one matrix
    X_ranges : Optional[MatrixLike2d], optional
            list of x ranges, by default None
    x_indices : Optional[List[int]], optional
        indices of three x variables, by default [0, 1, 2]
    X_name: Optional[List(str)], optional
        Names of X varibale shown as x,y,z-labels
    slice_axis : Optional[Union[str, int]], optional
        axis where a 2d slice is made, by default None
    slice_value : Optional[float], optional
        value on the axis where a 2d slide is made, 
        in a unit scale, by default None 
    slice_value_real : Optional[float], optional
        value on the axis where a 2d slide is made, 
        in a real scale, by default None 
    design_names : Optional[List[str]], optional
        Names of the designs, by default None
    save_fig: Optional[bool], optional
        if true save the plot 
        by default False
    save_path: Optional[str], optional
        Path where the figure is being saved
        by default the current directory

    Raises
    ------
    ValueError
        if input axis is defined but the value is not given
    ValueError
        if input axis name is not x, y or z, or 0, 1, 2
    """
    
    # if only one set of design is input, convert to list
    if not isinstance(Xs, list):
        Xs = [Xs]
    # set default design names if none
    if design_names is None:
        design_names = ['design' + str(i) for i in range(len(Xs))]
    if not isinstance(design_names, list):
        design_names = [design_names]
    # set the file name
    # if only one set of design, use that design name
    # else use comparison in the name
    file_name = 'sampling_3d_'
    if not isinstance(design_names, list):
        file_name += design_names # for a single design, include its name
    else:
        file_name += 'comparison' # for multiple designs, use "comparison"

    # Extract two variable indices for plotting
    x_indices = sorted(x_indices) 
    index_0 = x_indices[0]
    index_1 = x_indices[1]
    index_2 = x_indices[2]

    # Set default axis names 
    n_dim = Xs[0].shape[1]
    if X_names is None:
        X_names = ['x' + str(i+1) for i in range(n_dim)]
    # Set default [0,1] range for a unit scale
    if X_ranges is None:
        X_ranges = [[0,1]] * n_dim
    # Set default number of sections
    n_tick_sections  = 5
    
    # set the colors
    colors = colormap(np.linspace(0, 1, len(Xs)))
    # Visualize sampling plan - a 3D scatter plot
    fig  = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(111, projection='3d')
    for Xi, ci, name_i in zip(Xs, colors, design_names):
        ax.scatter(Xi[:, index_0], Xi[:, index_1], Xi[:, index_2], \
            color=ci, marker='o', s = 60, alpha = 0.6, label = name_i)
    # Get axes limits
    xlim_plot = list(ax.set_xlim(0, 1))
    ylim_plot = list(ax.set_ylim(0, 1))
    zlim_plot = list(ax.set_zlim(0, 1))
    
    # Add a 2d slide if required
    if slice_axis is not None:
        if (slice_value is None) and (slice_value_real is None):
            raise ValueError("Input a slice value")
        if (slice_axis == 'x') or (slice_axis == 0): 
            if slice_value is None: # convert the slice value into a unit scale
                slice_value = unitscale_xv(slice_value_real, X_ranges[0])
            add_x_slice_2d(ax, slice_value, [0, 1], [0, 1])
            file_name += '_slice_x'
        elif (slice_axis == 'y') or (slice_axis == 1): 
            if slice_value is None:
                slice_value = unitscale_xv(slice_value_real, X_ranges[1])
            add_y_slice_2d(ax, slice_value, [0, 1], [0, 1])
            file_name += '_slice_y'
        elif (slice_axis == 'z') or (slice_axis == 2): 
            if slice_value is None:
                slice_value = unitscale_xv(slice_value_real, X_ranges[2])
            add_z_slice_2d(ax, slice_value, [0, 1], [0, 1])
            file_name += '_slice_z'
        else: 
            raise ValueError("Input slice_axis is not valid, must be x, y or z, or 0, 1, 2")
    
    # set axis labels and ticks
    ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    ax.set_xlabel(X_names[index_0], labelpad= 15)
    ax.set_ylabel(X_names[index_1],labelpad= 15)
    ax.set_zlabel(X_names[index_2],labelpad=3)
    ax.set_xticks(set_axis_values(xlim_plot, n_tick_sections))
    ax.set_xticklabels(set_axis_values(X_ranges[index_0], n_tick_sections))
    ax.set_yticks(set_axis_values(ylim_plot, n_tick_sections))
    ax.set_yticklabels(set_axis_values(X_ranges[index_1], n_tick_sections))
    ax.set_zticks(set_axis_values(zlim_plot, n_tick_sections))
    ax.set_zticklabels(set_axis_values(X_ranges[index_2], n_tick_sections))
    ax.view_init(30, 45)
    # st.pyplot(fig)
    
    # save the figure as png
    if save_fig:
        if save_path is None: 
            save_path = os.getcwd()
        if not os.path.exists(save_path): os.makedirs(save_path)
        fig.savefig(os.path.join(save_path, file_name + '.' + figformat), 
                    bbox_inches="tight", transparent=backgroundtransparency)
    
    return fig