'''
Created on 13 Jul 2021

@author: Martijn Sparnaaij
'''

import json

from mpl_toolkits.axes_grid1.axes_divider import Divider
import mpl_toolkits.axes_grid1.axes_size as Size

import matplotlib.pyplot as plt
import numpy as np


def get_cdf_of_data(contact_data):
    sorted_data = np.sort(contact_data)
    # calculate the proportional values of samples
    p = 1. * np.arange(len(sorted_data)) / (len(sorted_data) - 1)
    
    return sorted_data, p 


def plot_convergence(exp_info_filename):
    with open(exp_info_filename, 'r') as f:
        exp_info_data = json.load(f)

    step_size = exp_info_data['convergence_config']['steps']
        
    array_names_dict = exp_info_data['combined_data_info']['arrayNamesDict']
        
    x_arrays = {array_name:[] for array_name in array_names_dict.keys()}
    y_arrays = {array_name:[] for array_name in array_names_dict.keys()}
        
    colors = ('tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:olive', 'tab:cyan', 'tab:brown')
        
    for entry in exp_info_data['convergence_stats']:
        repl_nr = entry[0]
        
        for key, p_values in entry[1].items():
            step_ind = 1
            for p_value in p_values:
                x_arrays[key].append(repl_nr - step_size + step_ind)
                y_arrays[key].append(p_value)
                step_ind += 1
    
    fig, (ax, legend_ax) = plt.subplots(2, 1, gridspec_kw={'hspace': 0.05, 'height_ratios': [3.5, 1]})
    
    line_handles = []
    line_labels = []
    
    for label, x in x_arrays.items():
        if array_names_dict[label][0] == 1.0:
            line_style = '--' 
        elif array_names_dict[label][0] == 1.5:
            line_style = '-' 
        elif array_names_dict[label][0] == 2.0:
            line_style = ':' 
        if array_names_dict[label][1] == "staff_customer":
            color_ind_0 = 0
        elif array_names_dict[label][1] == "customer_customer":
            color_ind_0 = 3
        else:
            color_ind_0 = 6
        
        if array_names_dict[label][2] == "weight_over_agents":
            color_ind_1 = 0
        elif array_names_dict[label][2] == "contacts_over_agents":
            color_ind_1 = 1
        else:
            color_ind_1 = 2
    
        line_handle = ax.plot(x, y_arrays[label], label=label, color=colors[color_ind_0 + color_ind_1], linestyle=line_style)
        line_handles.append(line_handle[0])
        line_labels.append(label)
    
    ax.set_title(exp_info_filename.stem)
    ax.grid()
    ax.axhline(0.95, linestyle='--', linewidth=2, color='k')
    leg = legend_ax.legend(line_handles, line_labels, loc='upper center', ncol=3)
    legend_ax.axis('off')
    leg.set_draggable(True)
    plt.show()
    
def plot_cdf_weights_at_probabilities(weights_at_probabilities, array_names_dict):
    colors = ('tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:olive', 'tab:cyan', 'tab:brown')
        
    ax = plt.axes()
    
    for label, data in weights_at_probabilities.items():
        if array_names_dict[label][0] == 1.0:
            line_style = '-' 
        elif array_names_dict[label][0] == 1.5:
            line_style = '--' 
        elif array_names_dict[label][0] == 2.0:
            line_style = ':' 
        if array_names_dict[label][1] == "staff_customer":
            color_ind_0 = 0
        elif array_names_dict[label][1] == "customer_customer":
            color_ind_0 = 3
        else:
            color_ind_0 = 6
        
        if array_names_dict[label][2] == "weight_over_agents":
            color_ind_1 = 0
        elif array_names_dict[label][2] == "contacts_over_agents":
            color_ind_1 = 1
        else:
            color_ind_1 = 2
    
        ax.plot(data[0], data[1], label=label, color=colors[color_ind_0 + color_ind_1], linestyle=line_style)
    
    ax.grid()
    leg = plt.legend(loc='upper center', ncol=3)
    leg.set_draggable(True)
    plt.show()
    
def get_cdf_weights_at_probabilities(contact_data, step_size=0.05):
    x, y = get_cdf_of_data(contact_data)
    probabilities = np.arange(step_size, 1.0+step_size, step_size)
    weights_at_probabilities = {}
    
    for probability in probabilities:
        indices = np.argwhere(y <= probability)
        weights_at_probabilities[probability] = x[indices[-1][0]]
        
    return weights_at_probabilities  
    
def create_cfd_weights_paper_plot(figure_props={}):
    fig = plt.figure(**figure_props)
    fig.canvas.set_window_title('')
    
    main_ax = fig.add_axes((0,0,1,1), 'main_ax')
    divider = Divider(fig, (0,0,1,1), [Size.Fixed(0.5), Size.Scaled(1), Size.Fixed(0.5)], [Size.Fixed(0.5), Size.Scaled(1), Size.Fixed(0.5)], aspect=False)
    main_ax.set_axes_locator(divider.new_locator(nx=1,ny=1))
    
    plt.draw()
    

