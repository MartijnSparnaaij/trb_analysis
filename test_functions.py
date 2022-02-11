#from nomad.NOMAD.activity_scheduler import RestaurantScheduler
#from nomad.NOMAD.input_manager import RestaurantSchedulerInput
import numpy as np
import itertools as it
from contact_analysis import calc_cum_contact_count_for_sample_size

def runTest(schedulerInput, destinations, pedParameterSets, 
                                sources, activities, walkLevels):
    scheduler = RestaurantScheduler(schedulerInput, destinations, pedParameterSets, 
                                    sources, activities, walkLevels)
    
    scheduler.tables2destinations = scheduler._getTables()
    groups, groupsPerTable = scheduler._createGroups()
    
    printResults(groupsPerTable, groups, scheduler)
    
def printResults(groupsPerTable, groups, scheduler): 
    print(f'Mean duration = {scheduler.meanVisitDuration}')
    print(f'Group count = {len(groups)}')
    print(f'\nGroup count per table:')
    for groupID, groupsOfTable in groupsPerTable.items():
        print(f'{groupID} = {len(groupsOfTable)}')
        
    print(f'\nGroup start/end per table:')
    for groupID, groupsOfTable in groupsPerTable.items():
        printStr = f'{groupID} = '
        for group in groupsOfTable:
            printStr += f'({group.startTime:8.2f}, {group.endTime:8.2f}), '        
        print(printStr)
        
    print(f'\n{"="*50}\n') 

def getNeighborhoodsStr(neighborhoods):
    return f'[{", ".join([neighborhood.ID for neighborhood in neighborhoods])}]'
    
def printStaff2neighborhoods(staff2neighborhoods):
    for ped, neighborhoods in staff2neighborhoods.items():
        print(f'{ped.ID}: {getNeighborhoodsStr(neighborhoods)}')

def printSeatCountPerNeighborhood(neighborhoods):
    for neighborhood in neighborhoods:
        print(f'{neighborhood.ID}: {neighborhood.seatCount}')

def printTable2Neighborhood(table2neighborhood):
    for tableID, neighborhood in table2neighborhood.items():
        print(f'{tableID}: {neighborhood.ID}')


def test_get_agent_contact_data():
    from pathlib import Path
    import sys
    
    nomad_path = r"C:\Users\martijnsparnaa\Eclipse_Python_NOMAD\nomad_trb_paper"
    sys.path.insert(1, nomad_path)
    
    analysis_path = r"C:\Users\martijnsparnaa\Jupyterlab_notebooks\trb_analysis"
    sys.path.insert(1, analysis_path)
    
    
    output_filename = Path(r'C:\Users\martijnsparnaa\surfdrive\PhD\Activity scheduler\Simulations\test\contact_data.npz')
    config_filenames = [Path(r'C:\Users\martijnsparnaa\surfdrive\PhD\Activity scheduler\Simulations\2021_12\transfer\long_restaurant_nn_nts_bvd\summarized\long_restaurant_nn_nts_bvd_scenario_experiment_recreated.config.json')]
    
    contact_data_config = {
        'contacts': [
            ((1.0, 1), (1.5, 15, True)),
            ((1.0, 1), (1.5, 10, False)),
            ((1.0, 1), (2.0, 15, False)),
            ((None, None), (1.5, 15, False))            
        ],
        'combs': (2, 3)
        }
    
    from contact_analysis import get_agent_contact_data
    get_agent_contact_data(output_filename, config_filenames, contact_data_config)


def test_comb_cum__time_prob():
    comb_sizes = [1,2,3]
    time_thres = 15
    for comb_size in comb_sizes:
        print(f'{0} - {calc_cum_contact_count_for_sample_size([1], comb_size, time_thres)}')
    for comb_size in comb_sizes:
        print(f'{0} - {calc_cum_contact_count_for_sample_size([1,2], comb_size, time_thres)}')
    for comb_size in comb_sizes:
        print(f'{0} - {calc_cum_contact_count_for_sample_size([1,2,3], comb_size, time_thres)}')
    for comb_size in comb_sizes:
        print(f'{1} - {calc_cum_contact_count_for_sample_size([15], comb_size, time_thres)}')
    correct = [0,1,1]
    for ii, comb_size in enumerate(comb_sizes):
         print(f'{correct[ii]} - {calc_cum_contact_count_for_sample_size([8,7], comb_size, time_thres)}') 
    correct = [0,0,1]
    for ii, comb_size in enumerate(comb_sizes):
        print(f'{correct[ii]} - {calc_cum_contact_count_for_sample_size([4,6, 5], comb_size, time_thres)}')
    array = [8,15,7,1,4,2,3,5,1,0.5,6,1]
    el_cnt = len(array)
    correct = [1,1+1/len(list(it.combinations(range(el_cnt-1),2))),2 + 1/len(list(it.combinations(range(el_cnt-3),3)))]
    for ii, comb_size in enumerate(comb_sizes):
        print(f'{correct[ii]} - {calc_cum_contact_count_for_sample_size(array, comb_size, time_thres)}')
    

if __name__ == '__main__':
    test_comb_cum__time_prob()