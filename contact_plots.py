'''
Created on 13 Jul 2021

@author: Martijn Sparnaaij
'''

import csv
import json
from scipy import stats
from math import sqrt

from mpl_toolkits.axes_grid1.axes_divider import Divider
import mpl_toolkits.axes_grid1.axes_size as Size

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
import numpy as np
import pandas as pd
from copy import deepcopy
import xlsxwriter
from cliffs_delta import cliffs_delta


def get_data_from_files(data_folder, convert_2_minutes=False):
    combined_data_files = get_combined_data_files_from_dir(data_folder)
    
    contact_data = {}
    infra_and_measure_name_2_data = {}
    for combined_data_file in combined_data_files:
        num_data = np.load(combined_data_file, allow_pickle=True)
        arrays_dict = num_data['arrays'].item()
        if convert_2_minutes:
            for key, contact_data_array in arrays_dict.items():
                if 'weight' in key:
                    arrays_dict[key] = contact_data_array/60
        
        scen_name = combined_data_file.stem.replace('_scenario_combined.stats', '')
        name_parts = scen_name.split('_')
        infra_name = '_'.join(name_parts[:2])
        measure_name = '_'.join(name_parts[2:])
        contact_data[scen_name] = {'data':arrays_dict, 'infra_name':infra_name, 'measure_name':measure_name}
        if infra_name not in infra_and_measure_name_2_data:
            infra_and_measure_name_2_data[infra_name] = {}
        if measure_name not in infra_and_measure_name_2_data[infra_name]:
            infra_and_measure_name_2_data[infra_name][measure_name] = []
        infra_and_measure_name_2_data[infra_name][measure_name] = arrays_dict
        
    return contact_data, infra_and_measure_name_2_data

def get_combined_data_files_from_dir(data_folder):
    scen_exts = ['.stats', '.npz']
    combinded_data_files = []
    for entry in data_folder.iterdir():
        if entry.is_file() and entry.suffixes == scen_exts:
            combinded_data_files.append(entry)
        elif entry.is_dir():
            combinded_data_files += get_combined_data_files_from_dir(entry)
    return combinded_data_files

def get_cdf_of_data(contact_data):
    sorted_data = np.sort(contact_data)
    # calculate the proportional values of samples
    p = 1. * np.arange(len(sorted_data)) / (len(sorted_data) - 1)
    
    return sorted_data, p 

def do_chi_square_test(base_array, compare_array, normalize=True):
    base_counts, compare_counts, values = get_value_counts(base_array, compare_array, normalize)
    
    if len(values) == 1:
        return 1.0
    
    res = stats.chisquare(compare_counts, base_counts)
    return res.pvalue    

def get_value_counts(base_array, compare_array, normalize):
    base_values, base_counts = np.unique(base_array, return_counts=True)
    if normalize:
        base_counts = base_counts/sum(base_counts)
    base_counts_dict = dict(zip(base_values, base_counts))
    compare_values, compare_counts = np.unique(compare_array, return_counts=True)
    if normalize:
        compare_counts = compare_counts/sum(compare_counts)
    else:
        compare_counts = compare_counts/sum(compare_counts)*sum(base_counts)
    compare_counts_dict = dict(zip(compare_values, compare_counts))
    values = np.unique(np.concatenate((base_values, compare_values)))
    
    base_counts = [base_counts_dict[value] if value in base_counts_dict else 0 for value in values]
    compare_counts = [compare_counts_dict[value] if value in compare_counts_dict else 0 for value in values]
        
    return base_counts, compare_counts, values

def do_power_divergence_test(base_array, compare_array, normalize=False):
    base_counts, compare_counts, values = get_value_counts(base_array, compare_array, normalize)
    
    if len(values) == 1:
        return 1.0
    
    res = stats.power_divergence(compare_counts, base_counts)
    return res.pvalue    

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
    
def get_colors(color_count):
    colorMap = cm.get_cmap('cubehelix_r') #viridis_r
    start_buffer = 0.18
    end_buffer = 0.15
    colors = []
    step_size = (1 - start_buffer - end_buffer)/(color_count-1)
    for ii in range(color_count):
        colors.append(colorMap(start_buffer + ii*step_size))
    
    return colors
    
def create_cfd_weights_paper_plot(cdf_data, first_inset_cdf_data, second_inset_cdf_data, figure_props={}):
    fig = plt.figure(**figure_props)
    
    
    divider = Divider(fig, (0,0,1,1), [Size.Fixed(0.45), Size.Scaled(1), Size.Fixed(0.03)], [Size.Fixed(0.5), Size.Scaled(1), Size.Fixed(0.08), Size.Fixed(0.4), Size.Fixed(0.05)], aspect=False)
    main_ax = fig.add_axes((0,0,1,1), 'main_ax')
    main_ax.set_axes_locator(divider.new_locator(nx=1,ny=1))
    
    legend_ax = fig.add_axes((0,0,1,1), 'legend_ax')
    legend_ax.set_axes_locator(divider.new_locator(nx=1,ny=3))
    legend_ax.set_axis_off()
        
    #colors = ('tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:cyan')
    colors = get_colors(len(cdf_data))
    line_widths = (3,3,1.5,1.5,1.5)
    line_styles = ('-', '--', '--', '-', '--')
    zorders = (1,2,4,5,3)
    
    line_ind = 0
    line_handles = []
    line_labels = []
    for label, plot_data in cdf_data.items():
        x = plot_data['x']/60 # Converge from seconds to minutes
        y = plot_data['y']
        line_handle = main_ax.plot(x, y, color=colors[line_ind], linewidth=line_widths[line_ind], 
                                       linestyle=line_styles[line_ind], label=f'{label}',
                                       zorder=zorders[line_ind])
        line_handles.append(line_handle[0])
        line_labels.append(label)
        line_ind += 1 
    
    main_ax.set_xlabel('Exposure time [minutes]', fontsize=8)
    main_ax.set_ylabel('Probability [-]', fontsize=8)
    main_ax.tick_params(labelsize=8)
    main_ax.grid()
    
    inset_ax_width = .36
    inset_ax_height = .52
    inset_ax_y = 0.17
    
    
    back_ax = fig.add_axes((.16, inset_ax_y-0.05,inset_ax_width*2+0.105,inset_ax_height+0.1), 'back_ax')
    back_ax.get_xaxis().set_visible(False)
    back_ax.get_yaxis().set_visible(False)
    back_ax.spines['bottom'].set_color('w')
    back_ax.spines['top'].set_color('w') 
    back_ax.spines['right'].set_color('w')
    back_ax.spines['left'].set_color('w')
    back_ax.set_alpha(0.6)
    
    first_inset_ax = fig.add_axes((.20, inset_ax_y,inset_ax_width,inset_ax_height), 'first_inset_ax')
    second_inset_ax = fig.add_axes((.62, inset_ax_y,inset_ax_width,inset_ax_height), 'second_inset_ax_inset_ax')
    
    line_ind = 0
    for label, plot_data in first_inset_cdf_data.items():
        x = plot_data['x']/60 # Converge from seconds to minutes
        y = plot_data['y']
        first_inset_ax.plot(x, y, color=colors[line_ind], linewidth=line_widths[line_ind], 
                                       linestyle=line_styles[line_ind], label=f'{label}',
                                       zorder=zorders[line_ind])
        line_ind += 1 
    
    first_inset_ax.grid()
    first_inset_ax.set_title('Customer-Customer', fontsize=8)
    first_inset_ax.tick_params(labelsize=8)
    
    line_ind = 0
    for label, plot_data in second_inset_cdf_data.items():
        x = plot_data['x']/60 # Converge from seconds to minutes
        y = plot_data['y']
        second_inset_ax.plot(x, y, color=colors[line_ind], linewidth=line_widths[line_ind], 
                                       linestyle=line_styles[line_ind], label=f'{label}',
                                       zorder=zorders[line_ind])
        line_ind += 1 
    
    second_inset_ax.grid()
    second_inset_ax.set_title('Staff-Customer', fontsize=8)
    second_inset_ax.tick_params(labelsize=8)
    
    legend_ax.legend(line_handles, line_labels, loc='upper center', ncol=3, fontsize=8, handlelength=3)
    plt.draw()
    

def create_cfd_contacts_paper_plot(cdf_data, figure_props={}):
    fig = plt.figure(**figure_props)
    
    
    
    divider = Divider(fig, (0,0,1,1), [Size.Fixed(0.45), Size.Scaled(1), Size.Fixed(0.1), Size.Scaled(1), Size.Fixed(0.1), Size.Scaled(1), Size.Fixed(0.03)], [Size.Fixed(0.5), Size.Scaled(1), Size.Fixed(0.21), Size.Fixed(0.5), Size.Fixed(0.02)], aspect=False)
    
    legend_ax = fig.add_axes((0,0,1,1), 'legend_ax')
    legend_ax.set_axes_locator(divider.new_locator(nx=1,ny=3, nx1=6))
    legend_ax.set_axis_off()
        
    colors = ('tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:cyan')
    colors = get_colors(len(list(cdf_data.values())[0]))
    line_widths = (3,3,1.5,1.5,1.5)
    line_styles = ('-', '--', '--', '-', '--')
    zorders = (1,2,4,5,3)

    axes = []
    line_handles = []
    line_labels = []
    ax_ind = 0
    for ax_label, ax_data in cdf_data.items():
        draw_ax = fig.add_axes((0,0,1,1), f'main_ax_{ax_ind}')
        draw_ax.set_axes_locator(divider.new_locator(nx=1+ax_ind*2,ny=1))
        line_ind = 0
        for label, plot_data in ax_data.items():
            x = plot_data['x']
            y = plot_data['y']
            line_handle = draw_ax.plot(x, y, color=colors[line_ind], linewidth=line_widths[line_ind], 
                                       linestyle=line_styles[line_ind], label=f'{label}',
                                       zorder=zorders[line_ind])
            if ax_ind == 0:
                line_handles.append(line_handle[0])
                line_labels.append(label)
            line_ind += 1 
        
        draw_ax.set_xlabel('# contacts [-]', fontsize=8)
        if ax_ind == 0:
            draw_ax.set_ylabel('Probability [-]', fontsize=8)
        else:
            draw_ax.set_yticklabels([])
        
        draw_ax.set_title(ax_label, fontsize=8)
        draw_ax.tick_params(labelsize=8)
        draw_ax.grid()
        ax_ind += 1
        
    legend_ax.legend(line_handles, line_labels, loc='upper center', ncol=3, fontsize=8, handlelength=3)
        
    plt.draw()
    
    
def create_base_case_paper_plot(fig_config, divider_config, plots_config, data):
    fig_title = None
    if 'title' in fig_config:
        fig_title = fig_config['title']
        fig_config.pop('title')
    fig = plt.figure(**fig_config)
    if fig_title is not None:
        fig.canvas.manager.set_window_title(fig_title)
    
    hor_division = [Size.Fixed(divider_config['left']), Size.Scaled(1), Size.Fixed(divider_config['gap_hor']), 
                    Size.Scaled(1), Size.Fixed(divider_config['gap_hor']), Size.Scaled(1), Size.Fixed(divider_config['right'])]
    ver_division = [Size.Fixed(divider_config['bottom']), Size.Scaled(1), Size.Fixed(divider_config['gap_ver']), Size.Scaled(divider_config['top_axes_scale']), Size.Fixed(divider_config['top'])]  
    divider = Divider(fig, (0,0,1,1), hor_division, ver_division, aspect=False)
    
    main_axes = []
    top_axes = []
    for ax_nr in range(3):
        main_ax = fig.add_axes((0,0,1,1), f'main_ax_{ax_nr}')
        main_ax.set_axes_locator(divider.new_locator(nx=1+ax_nr*2,ny=1))
        main_axes.append(main_ax)
        
        top_ax = fig.add_axes((0,0,1,1), f'top_ax_{ax_nr}')
        top_ax.set_axes_locator(divider.new_locator(nx=1+ax_nr*2,ny=3))
        top_axes.append(top_ax)
            
    for ax_nr, plot_config in enumerate(plots_config):
        x = list(range(1, len(plot_config['measure_names']) + 1))
        main_ax = main_axes[ax_nr]
        top_ax = top_axes[ax_nr]
        if plot_config['main_type'] == 'bar_plot':
            bar_plot(main_ax, data, plot_config, x)
        else:
            box_plot(main_ax, data, plot_config, x)
        
        x_ticks = main_ax.get_xticks()
        x_lim = main_ax.get_xlim() 
        plot_config['top_ax_config'].update({'x_lim':x_lim, 'x_ticks':x_ticks})
        
        threshold_quartile_plot(top_ax, data, plot_config, x)
        #mean_diff_plot(top_ax, data, plot_config, x)
           
    plt.draw()
        
def create_and_save_box_paper_plot(fig_config, save_filename, divider_config, plots_config, data):
    fig = plt.figure(**fig_config)
    hor_division = [Size.Fixed(divider_config['left']), Size.Scaled(1), Size.Fixed(divider_config['gap_hor']), 
                    Size.Scaled(1), Size.Fixed(divider_config['gap_hor']), Size.Scaled(1), Size.Fixed(divider_config['right'])]
    ver_division = [Size.Fixed(divider_config['bottom']), Size.Scaled(1), Size.Fixed(divider_config['top'])]  
    divider = Divider(fig, (0,0,1,1), hor_division, ver_division, aspect=False)
    
    for ax_nr, plot_config in enumerate(plots_config):
        ax = fig.add_axes((0,0,1,1), f'main_ax_{ax_nr}')
        ax.set_axes_locator(divider.new_locator(nx=1+ax_nr*2,ny=1))
        
        x = list(range(1, len(plot_config['measure_names']) + 1))
        if plot_config['main_type'] == 'bar_plot':
            bar_plot(ax, data, plot_config, x)
        else:
            box_plot(ax, data, plot_config, x)
        
    plt.savefig(save_filename, dpi=300)
    
        
def box_plot(ax, data, plot_config, x):
    if 'infra_names' in plot_config:
        infra_names = plot_config['infra_names']
    else:
        infra_names = [plot_config['infra_name']]
    
    if 'data_names' in plot_config:
        data_names = plot_config['data_names']
    else:
        data_names = [plot_config['data_name']]
    
    box_data = []
    for measure_name in plot_config['measure_names']:
        for infra_name in infra_names:
            for data_name in data_names:
                box_data.append(data[infra_name][measure_name][data_name])
        
    for x_coord in x:
        ax.axvline(x_coord, **plot_config['v_line_config'])

    ax.boxplot(box_data, positions=x, zorder=3, **plot_config['box_plot_config'])    
    set_ax_config(ax, plot_config['main_ax_config'])
 
def multi_box_plot(ax, data, plot_config, x):
    handles = []
    labels = []
    
    for ii, sub_group in enumerate(plot_config['sub_groups']):
        infra_name = sub_group['infra_name']
        data_name = sub_group['data_name']        
        
        box_data = []
        for measure_name in plot_config['measure_names']:
            box_data.append(data[infra_name][measure_name][data_name])
            
        for x_coord in x[ii]:
            ax.axvline(x_coord, **plot_config['v_line_config'])
    
        box_handles = ax.boxplot(box_data, positions=x[ii], zorder=3, **sub_group['box_plot_config'])    
        handles.append(box_handles['boxes'][0])
        labels.append(sub_group['label'])
        
    set_ax_config(ax, plot_config['main_ax_config'])
    
    return (handles, labels)
 
def bar_border_line_plot(ax, data, plot_config, x):
    handles = []
    labels = []
    
    line_styles = ['-', '--', ':']
    
    for sub_group in plot_config['sub_groups']:
        infra_name = sub_group['infra_name']
        data_name = sub_group['data_name'] 
        categories, _, value_2_rel_counts_per_measure = get_counts_per_value(plot_config['measure_names'], infra_name, data_name, data)
        
        prev_border_y = [0 for _ in plot_config['measure_names']]
        if sub_group["label"] in ('base_case - 1.0', 'busy_corner - 1.0'):
            print(f'{sub_group["label"]}')
            print(categories)
        for ii in range(len(categories) - 1):
            category = categories[ii]
            border_y = [prev_border_y[jj] + value_2_rel_counts_per_measure[measure_name][category] if category in value_2_rel_counts_per_measure[measure_name] else prev_border_y[jj] for jj, measure_name in enumerate(plot_config['measure_names'])]            
            ax.plot(x, border_y, label=f'{sub_group["label"]} - ({categories[ii]} - {categories[ii+1]})', linestyle=line_styles[ii], **sub_group['line_plot_config'])
            if sub_group["label"] in ('base_case - 1.0', 'busy_corner - 1.0'):
                print(f'{sub_group["label"]} - ({categories[ii]} - {categories[ii+1]})')
                print(prev_border_y)
                print(border_y)
                print([value_2_rel_counts_per_measure[measure_name][category] for measure_name in plot_config['measure_names']])
            
            prev_border_y = deepcopy(border_y)
 
    ax.legend(**plot_config['bar_plot_legend_config'])
    set_ax_config(ax, plot_config['main_ax_config'])
 
def get_counts_per_value(measure_names, infra_name, data_name, data): 
    values = []
    value_2_counts_per_measure = {}
    value_2_rel_counts_per_measure = {}
    
    for measure_name in measure_names:
        value_2_counts_per_measure[measure_name] = []
        value_2_rel_counts_per_measure[measure_name] = []
        unique_values, value_counts = np.unique(data[infra_name][measure_name][data_name], return_counts=True)
        values.append(unique_values)
        value_2_counts_per_measure[measure_name] = dict(zip(unique_values, value_counts))
        value_2_rel_counts_per_measure[measure_name] = dict(zip(unique_values, value_counts/np.sum(value_counts)))
                
    categories = np.unique(np.concatenate(values)).tolist()
    
    return categories, value_2_counts_per_measure, value_2_rel_counts_per_measure   
                
def bar_plot(ax, data, plot_config, x):
    if 'infra_names' in plot_config:
        infra_names = plot_config['infra_names']
    else:
        infra_names = [plot_config['infra_name']]
    
    if 'data_names' in plot_config:
        data_names = plot_config['data_names']
    else:
        data_names = [plot_config['data_name']]
    
    values = []
    value_2_counts_per_measure = {}
    value_2_rel_counts_per_measure = {}
    
    for measure_name in plot_config['measure_names']:
        value_2_counts_per_measure[measure_name] = {}
        value_2_rel_counts_per_measure[measure_name] = {}
        for infra_name in infra_names:
            value_2_counts_per_measure[measure_name][infra_name] = {}
            value_2_rel_counts_per_measure[measure_name][infra_name] = {}
            for data_name in data_names:
                unique_values, value_counts = np.unique(data[infra_name][measure_name][data_name], return_counts=True)
                values.append(unique_values)
                value_2_counts_per_measure[measure_name][infra_name][data_name] = dict(zip(unique_values, value_counts))
                value_2_rel_counts_per_measure[measure_name][infra_name][data_name] = dict(zip(unique_values, value_counts/np.sum(value_counts)))
                    
    categories = np.unique(np.concatenate(values)).tolist()
    bar_data = {}
    text_data = {}
    bottoms = {}
    column_count = len(plot_config['measure_names'])*len(infra_names)*len(data_names)
    
    for ii, category in enumerate(categories):
        bar_data[category]  = []
        text_data[category]  = []
        for measure_name in plot_config['measure_names']:
            for infra_name in infra_names:
                for data_name in data_names:
                    rel_counts = value_2_rel_counts_per_measure[measure_name][infra_name][data_name]
                    counts = value_2_counts_per_measure[measure_name][infra_name][data_name]
                    bar_data[category].append(rel_counts[category] if category in rel_counts else 0)
                    text_data[category].append(counts[category] if category in counts else 0)
             
        if ii == 0:
            bottoms[category] = None
        elif ii == 1:
            bottoms[category] = bar_data[categories[ii-1]]
        else:
            bottoms[category] = [bottoms[categories[ii-1]][jj] + bar_data[categories[ii-1]][jj] for jj in range(column_count)]
    
        prev_category = category
        
    for x_coord in x:
        ax.axvline(x_coord, **plot_config['v_line_config'])

    for category in categories:
        ax.bar(x, bar_data[category], label=f'{category}', bottom=bottoms[category], zorder=3, **plot_config['bar_plot_config'])

    # for category in categories:
    #     for ii, x_coord in enumerate(x):
    #         if bottoms[category] is None:
    #             y_bottom = 0
    #         else:
    #             y_bottom = bottoms[category][ii]
    #
    #         y_coord = y_bottom + bar_data[category][ii]/2
    #         count = text_data[category][ii]
    #         ax.text(x_coord, y_coord, f'{count}', zorder=4, **plot_config['bar_plot_text_config'])
            
    ax.set_xticks(x)
    ax.legend(**plot_config['bar_plot_legend_config'])
    set_ax_config(ax, plot_config['main_ax_config'])     
    
       
def mean_diff_plot(ax, data, plot_config, x):
    infra_name = plot_config['infra_name']
    data_name = plot_config['data_name']
        
    base_name = plot_config['measure_names'][0]
    base_mean = np.mean(data[infra_name][base_name][data_name])
    mean_data = []
    
    colors = []
    mean_bar_color_config = plot_config['mean_bar_color_config']
    for measure_name in plot_config['measure_names']:
        other_mean = np.mean(data[infra_name][measure_name][data_name])
        mean_data.append((other_mean - base_mean)/base_mean*100)  
        if mean_data[-1] < 0:
            colors.append(mean_bar_color_config['negative'])
        elif mean_data[-1] > 0: 
            colors.append(mean_bar_color_config['positive'])
        else:
            colors.append(mean_bar_color_config['base'])
            
    for x_coord in x:
        ax.axvline(x_coord, **plot_config['v_line_config'])

    ax.axhline(0, linewidth=2, color='k', linestyle='-', zorder=4)
    ax.bar(x, mean_data, zorder=3, color=colors, **plot_config['mean_bar_config'])
    set_ax_config(ax, plot_config['top_ax_config'])
       
def threshold_quartile_plot(ax, data, plot_config, x):
    infra_name = plot_config['infra_name']
    data_name = plot_config['data_name']
    thresholds_config = plot_config['thresholds_config']
    
    for threshold_config in thresholds_config:
        y = []
        for measure_name in plot_config['measure_names']:
            y.append(compute_quantile_value(data, infra_name, data_name, measure_name, threshold_config['value']))
        ax.plot(x, y, **threshold_config['line_config'])
        
    set_ax_config(ax, plot_config['top_ax_config'])
        
def compute_quantile_value(data, infra_name, data_name, measure_name, threshold_value):
    data_array = data[infra_name][measure_name][data_name]
    return 1 - sum(data_array <= threshold_value)/len(data_array)
    
        
def mean_plot(ax, data, plot_config, x):
    if 'infra_names' in plot_config:
        infra_names = plot_config['infra_names']
    else:
        infra_names = [plot_config['infra_name']]
    
    if 'data_names' in plot_config:
        data_names = plot_config['data_names']
    else:
        data_names = [plot_config['data_name']]
        
    bar_data = []
    
    for measure_name in plot_config['measure_names']:
        for infra_name in infra_names:
            for data_name in data_names:
                bar_data.append(np.mean(data[infra_name][measure_name][data_name]))  
            
    for x_coord in x:
        ax.axvline(x_coord, **plot_config['v_line_config'])

    ax.axhline(0, linewidth=2, color='k', linestyle='-', zorder=4)
    ax.bar(x, mean_data, zorder=3, color=colors, **plot_config['mean_bar_config'])
    set_ax_config(ax, plot_config['top_ax_config'])
        
def set_ax_config(ax, ax_config):
    if 'x_ticks' in ax_config:
        ax.set_xticks(ax_config['x_ticks'])
    if 'x_tick_labels' in ax_config:
        if 'x_tick_labels_config' in ax_config:
            ax.set_xticklabels(ax_config['x_tick_labels'], **ax_config['x_tick_labels_config'])
        else:
            ax.set_xticklabels(ax_config['x_tick_labels'])
    elif 'x_tick_labels_config' in ax_config:
        print(ax.get_xticklabels())
        ax.set_xticklabels(ax.get_xticklabels(), **ax_config['x_tick_labels_config'])

    if 'x_lim' in ax_config:
        ax.set_xlim(ax_config['x_lim'])
    if 'y_lim' in ax_config:
        ax.set_ylim(ax_config['y_lim'])

        
    if 'y_label' in ax_config:
        if 'y_label_config' in ax_config:
            ax.set_ylabel(ax_config['y_label'], **ax_config['y_label_config'])
        else:
            ax.set_ylabel(ax_config['y_label'])
            
    if 'title' in ax_config:
        if 'title_config' in ax_config:
            ax.set_title(ax_config['title'], **ax_config['title_config'])
        else:
            ax.set_title(ax_config['title'])
    if 'grid' in ax_config:
        ax.grid(**ax_config['grid'])


def create_compare_plot(fig_config, divider_config, plots_config, data):
    fig_title = None
    if 'title' in fig_config:
        fig_title = fig_config['title']
        fig_config.pop('title')
    fig = plt.figure(**fig_config)
    if fig_title is not None:
        fig.canvas.manager.set_window_title(fig_title)
    
    hor_division = [Size.Fixed(divider_config['left']), Size.Scaled(1), Size.Fixed(divider_config['gap_hor']), 
                    Size.Scaled(1), Size.Fixed(divider_config['gap_hor']), Size.Scaled(1), Size.Fixed(divider_config['right'])]
    ver_division = [Size.Fixed(divider_config['bottom']), Size.Scaled(1), Size.Fixed(divider_config['gap_ver']), Size.Fixed(divider_config['v_legend']), Size.Fixed(divider_config['top'])]  
    divider = Divider(fig, (0,0,1,1), hor_division, ver_division, aspect=False)
    
    ax_count = 3
    
    legend_ax = fig.add_axes((0,0,1,1), f'legend_ax')
    legend_ax.set_axes_locator(divider.new_locator(nx=1,ny=3,nx1=1+ax_count*2))
    legend_ax.set_axis_off()
    
    main_axes = []
    for ax_nr in range(ax_count):
        main_ax = fig.add_axes((0,0,1,1), f'main_ax_{ax_nr}')
        main_ax.set_axes_locator(divider.new_locator(nx=1+ax_nr*2,ny=1))
        main_axes.append(main_ax)
    
    for ax_nr, plot_config in enumerate(plots_config):
        x, x_labels = get_x_coords(plot_config)
        main_ax = main_axes[ax_nr]
        plot_config['main_ax_config'].update({'x_ticks':x_labels})
        if plot_config['main_type'] == 'bar_border_line':
            bar_border_line_plot(main_ax, data, plot_config, x_labels)
        else:
            legend_info = multi_box_plot(main_ax, data, plot_config, x)
        

    legend_ax.legend(*legend_info, **plot_config['legend_config'])
           
    plt.draw()   
    
def get_x_coords(plot_config):
    if plot_config['main_type'] == 'bar_border_line':
        x = list(range(1, len(plot_config['measure_names']) + 1))
        return x, x
   
    x_per_group = False
    
    if 'sub_groups' in plot_config:
        width = plot_config['sub_groups'][0]['box_plot_config']['widths']
        x_per_group = True
    elif 'widths' not in plot_config['box_plot_config']:
        width = 0.5 # mlp default    
    else:
        width = plot_config['box_plot_config']['widths']
        
    measure_count = len(plot_config['measure_names'])
    if x_per_group:
        set_count = len(plot_config['sub_groups'])
    else:
        if 'infra_names' in plot_config:
            infra_count = len(plot_config['infra_names'])
        else:
            infra_count = 1
        if 'data_names' in plot_config:
            data_count = len(plot_config['data_names'])
        else:
            data_count = 1

        set_count = infra_count*data_count

    set_gap = max(0.2, 1-width)
    set_width = width*set_count

    if x_per_group:
        x = [[] for _ in range(set_count)]
    else:
        x = []
    x_labels = []

    x_coord = width/2
    for _ in range(measure_count):
        x_start = x_coord
        for ii in range(set_count): 
            if x_per_group:
                x[ii].append(x_coord)
            else:
                x.append(x_coord)
            x_coord += width
        x_labels.append(x_start + (x_coord - width -  x_start)/2)
        x_coord += set_gap
    
    return x, x_labels
    
def create_quartile_plot(fig_config, divider_config, plots_config, data):
    fig_title = None
    if 'title' in fig_config:
        fig_title = fig_config['title']
        fig_config.pop('title')
    fig = plt.figure(**fig_config)
    if fig_title is not None:
        fig.canvas.manager.set_window_title(fig_title)
    
    hor_division = [Size.Fixed(divider_config['left']), Size.Scaled(1), Size.Fixed(divider_config['gap_hor']), 
                    Size.Scaled(1), Size.Fixed(divider_config['gap_hor']), Size.Scaled(1), Size.Fixed(divider_config['right'])]
    ver_division = [Size.Fixed(divider_config['bottom']), Size.Scaled(1), Size.Fixed(divider_config['gap_ver']), Size.Fixed(divider_config['v_legend']), Size.Fixed(divider_config['top'])]  
    divider = Divider(fig, (0,0,1,1), hor_division, ver_division, aspect=False)
    
    ax_count = 3
    
    legend_ax = fig.add_axes((0,0,1,1), f'legend_ax')
    legend_ax.set_axes_locator(divider.new_locator(nx=1,ny=3,nx1=1+ax_count*2))
    legend_ax.set_axis_off()

    for ax_nr, plot_config in enumerate(plots_config):
        ax = fig.add_axes((0,0,1,1), f'main_ax_{ax_nr}')
        ax.set_axes_locator(divider.new_locator(nx=1+ax_nr*2,ny=1))
    
        x = list(range(1,len(plot_config['measure_names'])+1))  
        plot_config['ax_config'].update({'x_ticks':x})
        handles, labels = plot_quartile_lines(ax, data, plot_config, x)
           
    legend_ax.legend(handles, labels, **plot_config['legend_config'])
    plt.draw()   
    
    
def plot_quartile_bars(ax, data, plot_config, x):
    quartile_labels = ['Q1', 'Q2', 'Q3', 'Q4']
    handles = []
    labels = []
        
    for jj, bar_set_config in enumerate(plot_config['bar_set_configs']):
        quartiles = [[] for _ in range(len(quartile_labels))]
        bottoms = [[] for _ in range(len(quartile_labels))]
        
        infra_name = bar_set_config['infra_name']
        data_name = bar_set_config['data_name']
        for measure_name in plot_config['measure_names']:
            local_data = data[infra_name][measure_name][data_name] 
            quartiles_local = stats.mstats.mquantiles(local_data)
            bottoms[0].append(0)
            for ii, value in enumerate(quartiles_local):
                bottoms[ii+1].append(value)
                quartiles[ii].append(value - bottoms[ii][-1])
                
            quartiles[3].append(np.max(local_data) - bottoms[3][-1])
               
        for ii in range(len(quartiles)):
            bar_set_config['bar_plot_config']['alpha'] = 1 - 0.2*ii
            bars = ax.bar(x[jj], quartiles[ii], label=f'{bar_set_config["label"]} - {quartile_labels[ii]}', bottom=bottoms[ii], zorder=3, **bar_set_config['bar_plot_config'])
            handles.append(bars[0])
            labels.append(bars._label)
            
    set_ax_config(ax, plot_config['ax_config'])
    return handles, labels

def plot_quartile_lines(ax, data, plot_config, x):
    quartile_labels = ['Q1', 'Q2', 'Q3', 'Q4']
    line_styles = ['-', '--', ':', '-.']
    handles = []
    labels = []
        
    for line_plot_config in plot_config['line_plot_configs']:
        quartiles = [[] for _ in range(len(quartile_labels))]
               
        infra_name = line_plot_config['infra_name']
        data_name = line_plot_config['data_name']
        for measure_name in plot_config['measure_names']:
            local_data = data[infra_name][measure_name][data_name] 
            quartiles_local = stats.mstats.mquantiles(local_data)
            
            for ii, value in enumerate(quartiles_local):
                 quartiles[ii].append(value)
                
            quartiles[3].append(np.max(local_data))
               
        for ii in range(len(quartiles)):
            line_plot_config['line_config']['linestyle'] = line_styles[ii]
            line = ax.plot(x, quartiles[ii], label=f'{line_plot_config["label"]} - {quartile_labels[ii]}', zorder=3, **line_plot_config['line_config'])
            handles.append(line[0])
            labels.append(line[0].get_label())
            
    set_ax_config(ax, plot_config['ax_config'])
    return handles, labels
    
  
    
def get_quartile_x_coords(plot_config):
    width = plot_config['bar_set_configs'][0]['bar_plot_config']['width']
        
    measure_count = len(plot_config['measure_names'])
   
    
    set_gap = plot_config['gap_size']
    set_count = len(plot_config['bar_set_configs'])
    
    set_width = width*set_count

    x = [[] for _ in range(measure_count)]
    x_labels = []
    
    x_coord = width/2
    for _ in range(measure_count):
        x_start = x_coord
        for ii in range(set_count): 
            x[ii].append(x_coord)
            x_coord += width
        x_labels.append(x_start + (x_coord - width -  x_start)/2)
        x_coord += set_gap
    
    return x, x_labels
    
def print_diff_stats(cut_off_dist, interaction_type_all, infra_name, parts_2_data_name, data_types, measures_and_labels, infra_and_measure_name_2_data, max_label_len, normalize=False):
    data_name_slice = parts_2_data_name[cut_off_dist][interaction_type_all]
    print(f'cut-off dist = {cut_off_dist}, interaction type = {interaction_type_all}, infra = {infra_name}')
    for data_type in data_types:
        
        if data_type == 'weight_over_contacts' and interaction_type_all == 'customer_staff':
            data_name = parts_2_data_name[cut_off_dist]['staff_customer'][data_type]            
        else:
            data_name = data_name_slice[data_type]
        print(f'----------- {data_type} -----------')
        
        combs_processed = []
        for measure_name, label in measures_and_labels:
            data_array = infra_and_measure_name_2_data[infra_name][measure_name][data_name]
            for other_measure_name, other_label in measures_and_labels:
                if measure_name == other_measure_name or (other_measure_name, measure_name) in combs_processed:
                    continue
                combs_processed.append((measure_name, other_measure_name))
                other_data_array = infra_and_measure_name_2_data[infra_name][other_measure_name][data_name]
                if 'contacts_over_agents' in data_name and interaction_type_all == 'customer_staff':
                    p_value = do_chi_square_test(data_array, other_data_array, normalize)
                    effect_size = get_effect_size(data_array, other_data_array, categorical=True)
                else:
                    res = stats.cramervonmises_2samp(data_array, other_data_array)
                    p_value = res.pvalue        
                    effect_size = get_effect_size(data_array, other_data_array)
                if p_value < 0.001:
                    sig = 'true'
                else:
                    sig = 'FALSE'

                   
                print(f'{label:>{max_label_len}s} - {other_label:>{max_label_len}s} = {p_value:.5f}, {sig:>5}, {effect_size:>-8.5f}')
        print('\n')
        
        
def print_diff_stats_table(cut_off_dist, interaction_type_all, infra_name, parts_2_data_name, data_types, measures_and_labels, infra_and_measure_name_2_data, normalize=False):
    data_name_slice = parts_2_data_name[cut_off_dist][interaction_type_all]

    columns = []
    print(f',,', end='')
    for ii, (measure_name, label) in enumerate(measures_and_labels):
        if ii == 0:
            continue
        columns.append((measure_name, label))
        print(f',{label},,,,', end='')
    
    print('')
    print(f',,,{",".join(["p_value", "cliff_d_val", "cliff_d_cat", "ks_stat", "ks_p_value"]*(len(measures_and_labels) - 1))}')
    
    print(f'{infra_name} - {cut_off_dist} - {interaction_type_all}', end='')
    for data_type in data_types:
        print(f',{data_type}', end='')
        if data_type == 'weight_over_contacts' and interaction_type_all == 'customer_staff':
            data_name = parts_2_data_name[cut_off_dist]['staff_customer'][data_type]            
        else:
            data_name = data_name_slice[data_type]
        
        printed_data = set()
        for ii, (measure_name, label) in enumerate(measures_and_labels):
            if ii == len(measures_and_labels) - 1:
                continue    
            data_array = infra_and_measure_name_2_data[infra_name][measure_name][data_name]
            if ii == 0:
                print(f',{label}', end='')
            else:
                print(f',,{label}', end='')
            for other_measure_name, other_label in columns:
                if measure_name == other_measure_name or (other_measure_name, measure_name) in printed_data:
                    print(',,,,,', end='')
                    continue
                 
                other_data_array = infra_and_measure_name_2_data[infra_name][other_measure_name][data_name]
                
                p_value = get_p_value_to_print(data_array, other_data_array, data_name, interaction_type_all, normalize=normalize)
                cliff_d_val, cliff_d_cat  = cliffs_delta(data_array, other_data_array)
                ks_res = stats.ks_2samp(data_array, other_data_array)
              
                printed_data.add((measure_name, other_measure_name))
                print(f',{p_value:.4f},{cliff_d_val:.4f},{cliff_d_cat},{ks_res.statistic:.4f},{ks_res.pvalue:.4f}', end='')
                     
            print('')
        
        print('')        
                
                
def print_diff_stats_table_inter_infra(cut_off_dist, interaction_type_all, infra_name_combs, parts_2_data_name, data_types, measures_and_labels, infra_and_measure_name_2_data, normalize=False):
    data_name_slice = parts_2_data_name[cut_off_dist][interaction_type_all]

    columns = []
    print(f',,', end='')
    for measure_name, label in measures_and_labels:
        columns.append((measure_name, label))
        print(f',{label},,,,', end='')
    
    print('')
    print(f',,,{",".join(["p_value", "cliff_d_val", "cliff_d_cat", "ks_stat", "ks_p_value"]*len(measures_and_labels))}')
    
    print(f'{cut_off_dist} - {interaction_type_all}', end='')
    for data_type in data_types:
        print(f',{data_type}', end='')
        if data_type == 'weight_over_contacts' and interaction_type_all == 'customer_staff':
            data_name = parts_2_data_name[cut_off_dist]['staff_customer'][data_type]            
        else:
            data_name = data_name_slice[data_type]
        
        for ii, (infra_name, other_infra_name) in enumerate(infra_name_combs):
            if ii == 0:
                print(f',{infra_name} - {other_infra_name}', end='')
            else:
                print(f',,{infra_name} - {other_infra_name}', end='')
                   
            
            for measure_name, label in measures_and_labels:
                data_array = infra_and_measure_name_2_data[infra_name][measure_name][data_name]
                other_data_array = infra_and_measure_name_2_data[other_infra_name][measure_name][data_name]
                
         
                p_value = get_p_value_to_print(data_array, other_data_array, data_name, interaction_type_all, normalize=normalize)
                cliff_d_val, cliff_d_cat  = cliffs_delta(data_array, other_data_array)
                ks_res = stats.ks_2samp(data_array, other_data_array)
              
                print(f',{p_value:.4f},{cliff_d_val:.4f},{cliff_d_cat},{ks_res.statistic:.4f},{ks_res.pvalue:.4f}', end='')
            print('')    
        
        print('') 
        
def print_diff_stats_table_inter_dist(cut_off_dist_combs, interaction_type_all, infra_name, parts_2_data_name, data_types, measures_and_labels, infra_and_measure_name_2_data, normalize=False):
    columns = []
    print(f',,', end='')
    for measure_name, label in measures_and_labels:
        columns.append((measure_name, label))
        print(f',{label},,,,', end='')
    
    print('')
    print(f',,,{",".join(["p_value", "cliff_d_val", "cliff_d_cat", "ks_stat", "ks_p_value"]*len(measures_and_labels))}')
    
    print(f'{infra_name} - {interaction_type_all}', end='')
    for data_type in data_types:
        print(f',{data_type}', end='')
        for ii, (cut_off_dist, other_cut_off_dist) in enumerate(cut_off_dist_combs):
            if ii == 0:
                print(f',{cut_off_dist} - {other_cut_off_dist}', end='')
            else:
                print(f',,{cut_off_dist} - {other_cut_off_dist}', end='')

            if data_type == 'weight_over_contacts' and interaction_type_all == 'customer_staff':
                data_name = parts_2_data_name[cut_off_dist]['staff_customer'][data_type]
                other_data_name = parts_2_data_name[other_cut_off_dist]['staff_customer'][data_type]            
            else:
                data_name = parts_2_data_name[cut_off_dist][interaction_type_all][data_type]
                other_data_name = parts_2_data_name[other_cut_off_dist][interaction_type_all][data_type]
                       
            for measure_name, label in measures_and_labels:
                data_array = infra_and_measure_name_2_data[infra_name][measure_name][data_name]
                other_data_array = infra_and_measure_name_2_data[infra_name][measure_name][other_data_name]
                
                p_value = get_p_value_to_print(data_array, other_data_array, data_name, interaction_type_all, normalize=normalize)
                cliff_d_val, cliff_d_cat  = cliffs_delta(data_array, other_data_array)
                ks_res = stats.ks_2samp(data_array, other_data_array)
              
                print(f',{p_value:.4f},{cliff_d_val:.4f},{cliff_d_cat},{ks_res.statistic:.4f},{ks_res.pvalue:.4f}', end='')
            print('')    
        print('')
        
                
def get_effect_size_to_print(data_array, other_data_array, data_name, interaction_type_all):
    if 'contacts_over_agents' in data_name and interaction_type_all == 'customer_staff':
        effect_size = get_effect_size(data_array, other_data_array, categorical=True)
    else:
        effect_size = get_effect_size(data_array, other_data_array)
        
    return effect_size

def get_p_value_to_print(data_array, other_data_array, data_name, interaction_type_all, normalize):
    if 'contacts_over_agents' in data_name and interaction_type_all == 'customer_staff':
        p_value = do_chi_square_test(data_array, other_data_array, normalize=normalize)
    else:
        res = stats.cramervonmises_2samp(data_array, other_data_array)
        p_value = res.pvalue    
        
    return p_value

                    
def print_top_line(measures_and_labels): 
    columns = []
    for ii, (measure_name, label) in enumerate(measures_and_labels):
        if ii == 0:
            continue
        print(f',{label}', end='')
        columns.append((measure_name, label))
            
    print('')
    return columns
        
def get_effect_size(data_array, other_data_array, categorical=False):
    # Cohen's d (or w for categorical)
    if categorical:
        base_counts, compare_counts, _ = get_value_counts(data_array, other_data_array, False)
        return sqrt(np.sum([(compare_count/sum(compare_counts) - base_count/sum(base_counts))**2/(base_count/sum(base_counts)) for base_count, compare_count in zip(base_counts, compare_counts)]))
        
    n1 = len(data_array)
    n2 = len(other_data_array)
    s1_pow2 = np.var(data_array, ddof=1)
    s2_pow2 = np.var(other_data_array, ddof=1)
    s = sqrt(((n1 - 1)*s1_pow2 + (n2 - 1)*s2_pow2)/(n1 + n2 - 2))
    
    return (np.mean(data_array) - np.mean(other_data_array))/s

def create_csv(csv_pd, cut_off_dist, interaction_type_all, infra_name, parts_2_data_name, data_types, infra_and_measure_name_2_data, measures_and_labels):
    data_name_slice = parts_2_data_name[cut_off_dist][interaction_type_all]
    cut_off_dist_str = '_'.join(cut_off_dist.split('.'))
    for data_type in data_types:
        if data_type == 'weight_over_contacts' and interaction_type_all == 'customer_staff':
            data_name = parts_2_data_name[cut_off_dist]['staff_customer'][data_type]            
        else:
            data_name = data_name_slice[data_type]
            
        df_dict = {}
        for measure_name, _ in measures_and_labels:
            df_dict[measure_name] = infra_and_measure_name_2_data[infra_name][measure_name][data_name]
            
        df = pd.DataFrame({k:pd.Series(v) for k,v in df_dict.items()})
        csv_filename = csv_pd.joinpath(f'{infra_name}_{interaction_type_all}_{cut_off_dist_str}_{data_type}.csv')
        df.to_csv(csv_filename)
    
def get_effect_sizes_from_csvs(data_folder):
    csv_files = get_combined_csv_files_from_dir(data_folder)
    rows_2_csv = []
    for csv_filename in csv_files:
        rows_2_csv += process_effect_sizes_csv(csv_filename)
    
    output_filename = data_folder.joinpath(f'effect_sizes.xlsx')
    
    workbook = xlsxwriter.Workbook(output_filename)
    worksheet = workbook.add_worksheet()
    
    for row_ind, row in enumerate(rows_2_csv):
        for col_ind, row_entry in enumerate(row):
            worksheet.write(row_ind, col_ind, row_entry)
    
    workbook.close()
    
def read_data_from_csv(csv_filename):
    header = None
    data_rows = []
    with open(csv_filename, newline='') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')
        for row in rows:
            if header is None:
               header = row
               continue
            data_rows.append(row)   
            
    return header, data_rows
            
def process_effect_sizes_csv(csv_filename):
    header, data_rows = read_data_from_csv(csv_filename)
    rows_2_csv = [[csv_filename.name[:-7]],[''] + header[5:] + [''] + header[5:]]
    effect_size_labels = header[1:5]
    effect_size_labels[0] = 'N'
    for data_row in data_rows:
        rows_2_csv.append(process_effect_size_row(data_row, effect_size_labels))
    
    rows_2_csv.append([])
    
    return rows_2_csv

def process_effect_size_row(data_row, effect_size_labels):
    row_data = []
    name = data_row[0]
    effect_size_thresholds = [float(v) for v in data_row[1:5]]
    effect_size_num = [float(v) for v in data_row[5:]]
    effect_size_str = []
    threshold_count = len(effect_size_labels)
    for effect_size in effect_size_num:
        if effect_size < effect_size_thresholds[0]:
            if effect_size_thresholds[0] == 0:
                effect_size = -effect_size
            elif effect_size_thresholds[0] == 0.5 and effect_size_thresholds[1] > effect_size_thresholds[0]: 
                effect_size = 1 - effect_size
        elif effect_size_thresholds[0] == 0.5 and effect_size_thresholds[1] < effect_size_thresholds[0] and effect_size > effect_size_thresholds[0]: 
            effect_size = 1 - effect_size
                
        for ii in range(threshold_count):
            if ii == threshold_count - 1:
                effect_size_str.append(effect_size_labels[-1])
            else:
                if effect_size_thresholds[1] > effect_size_thresholds[0]:
                    if effect_size >= effect_size_thresholds[ii] and effect_size < effect_size_thresholds[ii + 1]:
                        effect_size_str.append(f'{effect_size_labels[ii]} - {effect_size_labels[ii+1]}')
                        break  
                else:
                    if effect_size <= effect_size_thresholds[ii] and effect_size > effect_size_thresholds[ii + 1]:
                        effect_size_str.append(f'{effect_size_labels[ii]} - {effect_size_labels[ii+1]}')
                        break  
        
    return [name] + effect_size_str + [''] + effect_size_num

def get_combined_csv_files_from_dir(data_folder):
    csv_files = []
    for entry in data_folder.iterdir():
        if entry.is_file() and entry.suffix == '.csv' and '_ES' in entry.name:
            csv_files.append(entry)
    return csv_files
