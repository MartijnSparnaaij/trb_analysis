'''
Created on 13 Jul 2021

@author: Martijn Sparnaaij
'''

import json

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
    
    ax = plt.axes()
    
    for label, x in x_arrays.items():
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
    
        ax.plot(x, y_arrays[label], label=label, color=colors[color_ind_0 + color_ind_1], linestyle=line_style)
    
    ax.grid()
    ax.axhline(0.95, linestyle='--', linewidth=2, color='k')
    leg = plt.legend()
    leg.set_draggable(True)
    plt.show()
    