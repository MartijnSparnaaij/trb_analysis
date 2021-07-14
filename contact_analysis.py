'''
Created on 13 Jul 2021

@author: Martijn Sparnaaij
'''
import json


def run_experiment(scenario_filename):
    pass
    

def create_cdfs(experiment_config_file):
    pass

def process_replication():
    pass

def process_file(filename, time_step):
    with open(filename) as f:
        contact_data = json.load(f)
    
    weigths_per_connection = {}
    weigths_per_agent = {}
    
    for ID_tuple in contact_data['ID2IDtuple']:
        contacts = contact_data[ID_tuple]
        
        


