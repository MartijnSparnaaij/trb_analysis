#from nomad.NOMAD.activity_scheduler import RestaurantScheduler
#from nomad.NOMAD.input_manager import RestaurantSchedulerInput
import numpy as np

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


if __name__ == '__main__':
    test_get_agent_contact_data()