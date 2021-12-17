'''
Created on 13 Jul 2021

@author: Martijn Sparnaaij
'''

import json
from scipy import stats

from mpl_toolkits.axes_grid1.axes_divider import Divider
import mpl_toolkits.axes_grid1.axes_size as Size

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from copy import deepcopy


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

def do_chi_square_test(base_array, compare_array):
    base_values, base_counts = np.unique(base_array, return_counts=True)
    base_counts = base_counts/sum(base_counts)
    base_counts_dict = dict(zip(base_values, base_counts))
    compare_values, compare_counts = np.unique(compare_array, return_counts=True)
    compare_counts = compare_counts/sum(compare_counts)
    compare_counts_dict = dict(zip(compare_values, compare_counts))
    values = np.unique(np.concatenate((base_values, compare_values)))
    
    
    base_counts = [base_counts_dict[value] if value in base_counts_dict else 0 for value in values]
    compare_counts = [compare_counts_dict[value] if value in compare_counts_dict else 0 for value in values]
    
    if len(values) == 1:
        return 1.0
    
    res = stats.chisquare(compare_counts, base_counts)
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
        
        mean_diff_plot(top_ax, data, plot_config, x)
           
    plt.draw()
        
def box_plot(ax, data, plot_config, x):
    infra_name = plot_config['infra_name']
    data_name = plot_config['data_name']
    
    box_data = []
    for measure_name in plot_config['measure_names']:
        box_data.append(data[infra_name][measure_name][data_name])
        
    for x_coord in x:
        ax.axvline(x_coord, **plot_config['v_line_config'])

    ax.boxplot(box_data, positions=x, zorder=3, **plot_config['box_plot_config'])    
    set_ax_config(ax, plot_config['main_ax_config'])
 
def bar_plot(ax, data, plot_config, x):
    infra_name = plot_config['infra_name']
    data_name = plot_config['data_name']
    
    values = []
    value_2_counts_per_measure = {}
    value_2_rel_counts_per_measure = {}
    for measure_name in plot_config['measure_names']:
        unique_values, value_counts = np.unique(data[infra_name][measure_name][data_name], return_counts=True)
        values.append(unique_values)
        value_2_counts_per_measure[measure_name] = dict(zip(unique_values, value_counts))
        value_2_rel_counts_per_measure[measure_name] = dict(zip(unique_values, value_counts/np.sum(value_counts)))
        
        
    categories = np.unique(np.concatenate(values)).tolist()
    bar_data = {}
    text_data = {}
    bottoms = {}
    for ii, category in enumerate(categories):
        bar_data[category] = [value_2_rel_counts_per_measure[measure_name][category] if category in value_2_rel_counts_per_measure[measure_name] else 0 for measure_name in plot_config['measure_names']]
        text_data[category] = [value_2_counts_per_measure[measure_name][category] if category in value_2_rel_counts_per_measure[measure_name] else 0 for measure_name in plot_config['measure_names']]
        if ii == 0:
            bottoms[category] = None
        elif ii == 1:
            bottoms[category] = bar_data[categories[ii-1]]
        else:
            bottoms[category] = [bottoms[categories[ii-1]][jj] + bar_data[categories[ii-1]][jj] for jj in range(len(plot_config['measure_names']))]
        
        prev_category = category
        
    for x_coord in x:
        ax.axvline(x_coord, **plot_config['v_line_config'])

    for category in categories:
        ax.bar(x, bar_data[category], label=f'{category}', bottom=bottoms[category], zorder=3, **plot_config['bar_plot_config'])

    for category in categories:
        for ii, x_coord in enumerate(x):
            if bottoms[category] is None:
                y_bottom = 0
            else:
                y_bottom = bottoms[category][ii]
            
            y_coord = y_bottom + bar_data[category][ii]/2
            count = text_data[category][ii]
            ax.text(x_coord, y_coord, f'{count}', zorder=4, **plot_config['bar_plot_text_config'])
            
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
