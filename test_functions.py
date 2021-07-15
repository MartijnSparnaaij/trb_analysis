from nomad.NOMAD.activity_scheduler import RestaurantScheduler
from nomad.NOMAD.input_manager import RestaurantSchedulerInput
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
