'''
Created on 13 Jul 2021

@author: Martijn Sparnaaij
'''
from _collections import defaultdict, deque
from copy import deepcopy
import json
from math import ceil
from pathlib import Path

from pubsub import pub
from scipy import stats

import NOMAD
from NOMAD.activity_scheduler import STAFF_GROUP_NAME
from NOMAD.constants import TIMESTEP_MSG_ID, SEND_TIME_MSG_INTERVAL_SEC
from NOMAD.nomad_model import onlyCreate
import numpy as np
from NOMAD.output_manager import BasicFileOutputManager




MAX_REPLICATIONS = 5000
INIT_REPLICATIONS = 30
CONSECUTIVE_STEPS = 5
P_THRESHOLD_VALUE = 0.95
STEPS = 'steps'
P_THRESHOLD = 'p_threshold'
CONVERGENCE_CONFIG = {STEPS:CONSECUTIVE_STEPS, '':P_THRESHOLD_VALUE}

WEIGHT_OVER_CONTACTS = 'weight_over_contacts'
WEIGHT_OVER_AGENTS = 'weight_over_agents'
CONTACTS_OVER_AGENTS = 'contacts_over_agents'
CDF_TYPES = (WEIGHT_OVER_CONTACTS, WEIGHT_OVER_AGENTS, CONTACTS_OVER_AGENTS)

CUT_OFF_DISTANCES = 'cut_off_distances' 
CUT_OFF_DISTANCES_VALUES = (1.0, 1.5, 2.0) 

STAFF_CUSTOMER = 'staff_customer'
CUSTOMER_CUSTOMER = 'customer_customer'
CUSTOMER_STAFF = 'customer_staff'

INTERACTION_TYPES = (STAFF_CUSTOMER, CUSTOMER_CUSTOMER, CUSTOMER_STAFF)

ARRAY_NAMES_FLD = 'arrayNames'
ARRAY_NAMES_DICT_FLD = 'arrayNamesDict'
INDICES_FLD = 'indices' 

class ExperimentRunner():
    
    def __init__(self, scenario_filename, lrcm_filename, convergence_config=CONVERGENCE_CONFIG, cut_off_distances=CUT_OFF_DISTANCES_VALUES, max_replications=MAX_REPLICATIONS, init_replications=INIT_REPLICATIONS, data_folder=None):
        self.scenario_filename = Path(scenario_filename)
        self.lrcm_filename = lrcm_filename
        self.convergence_config = convergence_config
        self.cut_off_distances = cut_off_distances
        self.max_replications = max_replications
        self.init_replications = init_replications
        self.data_folder = data_folder
        
        self.experiment_filename = self.scenario_filename.parent.joinpath(f'{self.scenario_filename.stem}_experiment.json')
        self.combined_stats_filename = self.scenario_filename.parent.joinpath(f'{self.scenario_filename.stem}_combined.stats')
            
        rng = np.random.default_rng()
        self.seeds_resevoir = rng.choice(np.arange(1,max_replications*2), max_replications, False).tolist()
        
        
        self.combined_data = {cut_off_distance:{STAFF_CUSTOMER:{cdf_type:None for cdf_type in CDF_TYPES}, 
                                                CUSTOMER_CUSTOMER:{cdf_type:None for cdf_type in CDF_TYPES},
                                                CUSTOMER_STAFF:{cdf_type:None for cdf_type in (WEIGHT_OVER_AGENTS, CONTACTS_OVER_AGENTS)}} 
                                      for cut_off_distance in self.cut_off_distances}
        self.combined_data_indices = {cut_off_distance:{STAFF_CUSTOMER:{cdf_type:deque('', self.convergence_config[STEPS] + 1) for cdf_type in CDF_TYPES}, 
                                                        CUSTOMER_CUSTOMER:{cdf_type:deque('', self.convergence_config[STEPS] + 1) for cdf_type in CDF_TYPES},
                                                       CUSTOMER_STAFF:{cdf_type:deque('', self.convergence_config[STEPS] + 1) for cdf_type in (WEIGHT_OVER_AGENTS, CONTACTS_OVER_AGENTS)}} 
                                      for cut_off_distance in self.cut_off_distances} # Of the last 5 entries
        
        self.array_names = {cut_off_distance:{
                            STAFF_CUSTOMER:{cdf_type:get_array_field_name(cut_off_distance, STAFF_CUSTOMER, cdf_type) for cdf_type in CDF_TYPES}, 
                            CUSTOMER_CUSTOMER:{cdf_type:get_array_field_name(cut_off_distance, CUSTOMER_CUSTOMER, cdf_type) for cdf_type in CDF_TYPES},
                            CUSTOMER_STAFF:{cdf_type:get_array_field_name(cut_off_distance, CUSTOMER_STAFF, cdf_type) for cdf_type in (WEIGHT_OVER_AGENTS, CONTACTS_OVER_AGENTS)}} 
                            for cut_off_distance in self.cut_off_distances}
        
        self.array_names_2_dict = {}
        for cut_off_distance, interaction_types in self.array_names.items():
            for interaction_type, cdf_types in interaction_types.items():
                for cdf_type, array_name in cdf_types.items():
                    self.array_names_2_dict[array_name] = (cut_off_distance, interaction_type, cdf_type)
         
        self._init_experiment_file()
        pub.subscribe(self._update_time, TIMESTEP_MSG_ID)
            
    def _update_time(self, timeInd, currentTime): # @UnusedVariable
        print('.', end='')
            
    def _init_experiment_file(self):
        self.experiment_info = {
            'scenario_filename': str(self.scenario_filename),
            'lrcm_filename': str(self.lrcm_filename),
            'combined_stats_filename': str(self.combined_stats_filename),
            'max_replications': self.max_replications,
            'init_replications': self.init_replications,  
            'cut_off_distances': self.cut_off_distances,  
            'convergence_config': self.convergence_config,
            'successful_replication_count': 0,
            'failed_replication_count': 0,
            'successful_replications': [], # (seed, .scen filename)
            'failed_replications': [], # seed
            'combined_data_info': {ARRAY_NAMES_FLD: self.array_names,
                                   ARRAY_NAMES_DICT_FLD: self.array_names_2_dict,
                                   INDICES_FLD:convert_sub_fields_to_list(self.combined_data_indices)},
            'convergence_stats': [] # {replications:#rep, cvm_stats:{cdf_1:[pvalues]}}
            }
    
        self._update_info_file(write_permission='x') 

    def run_experiment(self, replication_start_nr=0):
        self.replication_nr = replication_start_nr
        
        self.last_convergence_check = self.init_replications - 1 - self.convergence_config['steps']
        if self.replication_nr > self.last_convergence_check:
            self.last_convergence_check =+ ceil((self.replication_nr - self.last_convergence_check)/self.convergence_config['steps'])*self.convergence_config['steps']            
        
        self.failed_count = 0
        while self.replication_nr < self.max_replications:
            seed = self.seeds_resevoir[self.replication_nr]
            print(f'Running replication {self.replication_nr} with seed = {seed}')            
            NOMAD.NOMAD_RNG = np.random.default_rng(seed)
            try:
                nomad_model = onlyCreate(self.scenario_filename, lrcmLoadFile=self.lrcm_filename, seed=seed)
                print('='*ceil(nomad_model.duration/SEND_TIME_MSG_INTERVAL_SEC))
                nomad_model.start()
            except:
                del nomad_model
                self._update_info_after_failed(seed)
                continue
            print('\nSimulation done. Processing data... ', end='')
            # Process output accessing it directly via the nomad_model instance
            self._process_data(nomad_model)
                   
            self._update_info_after_success(seed, str(nomad_model.outputManager.scenarioFilename))
            del nomad_model     
            print('done')
            if self.replication_nr - self.failed_count == self.last_convergence_check + self.convergence_config['steps']:
                # Check convergence
                print('Checking convergence... ', end='')
                p_values, has_converged = self._check_convergence()                 
                self._update_info_after_convergence_check(p_values)
                    
                print('done')
                if has_converged:                    
                    self._save_data_2_file()
                    print(f'{"="*40}\n')
                    print(f'FINISHED!!!!!!!!!!\n')
                    print(f'{"="*40}')
                    break 
            
            self._save_data_2_file()                           
            self.replication_nr += 1            
            
    def recreate_from_folder(self):
        # Get a scenario file from folder
        # Get time step from file
        # Get all conncetion files 
        # Process all files
        scen_filenames = get_scenario_files_from_dir(self.data_folder)
        self.replication_nr = 0
        self.last_convergence_check = self.init_replications - 1 - self.convergence_config['steps']
        self.failed_count = 0
        print(len(scen_filenames))
    
        for scen_filename in scen_filenames:
            try:
                print(f'{self.replication_nr:04d} - {scen_filename} ...', end='')
                time_step, seed, connections, ID_2_group_ID = load_data(scen_filename)
            except:
                self._update_info_after_failed(seed)                           
                continue
            
            self._process_data(None, connections, ID_2_group_ID, time_step)
            
            self._update_info_after_success(seed, scen_filename)
            print('done')
            if self.replication_nr == self.last_convergence_check + self.convergence_config['steps']:
                # Check convergence
                print('Checking convergence... ', end='')
                p_values, _ = self._check_convergence() 
                self._update_info_after_convergence_check(p_values)
                print('done')                

            self._save_data_2_file()
            self.replication_nr += 1
    
    def _update_info_after_failed(self, seed):
        self.replication_nr += 1   
        self.failed_count += 1
        self.experiment_info['failed_replication_count'] += 1
        self.experiment_info['failed_replications'].append(seed)
        self._update_info_file()
                
    def _update_info_after_success(self, seed, scen_filename):
        self.experiment_info['combined_data_info'][INDICES_FLD] = convert_sub_fields_to_list(self.combined_data_indices)
        self.experiment_info['successful_replication_count'] += 1
        self.experiment_info['successful_replications'].append((seed, str(scen_filename)))
                
    def _update_info_after_convergence_check(self, p_values):
        self.last_convergence_check = self.replication_nr
        self.failed_count = 0                
        # save combined data 2 file
        self.experiment_info['convergence_stats'].append((self.replication_nr, p_values, convert_sub_fields_to_list(self.combined_data_indices)))
 
    def _save_data_2_file(self):
        print('Saving data to file...', end='')    
        self._save_combined_data()
        self._update_info_file()
        print('done')
    
    def _process_data(self, nomad_model, connections=None, ID_2_group_ID=None, time_step=None):
        if nomad_model is not None:
            connections = nomad_model.outputManager.connections
            ID_2_group_ID = nomad_model.outputManager.ID2groupID
            time_step = nomad_model.timeInfo.timeStep
        
        weigths_per_connection, weigths_per_agent, connections_per_agent = compute_weighted_graph(self.cut_off_distances, connections, ID_2_group_ID, time_step)

        self._add_data_2_combined_data(weigths_per_connection, weigths_per_agent, connections_per_agent)

    def _add_data_2_combined_data(self, weigths_per_connection, weigths_per_agent, connections_per_agent):
        for cut_off_distance in self.cut_off_distances:
            for interaction_type in INTERACTION_TYPES:
      
                if interaction_type in (STAFF_CUSTOMER, CUSTOMER_CUSTOMER):
                    self._extend_array(cut_off_distance, interaction_type, WEIGHT_OVER_CONTACTS, 
                                  weigths_per_connection[cut_off_distance][interaction_type])

                weigths_per_agent_list = list(weigths_per_agent[cut_off_distance][interaction_type].values())
                self._extend_array(cut_off_distance, interaction_type, WEIGHT_OVER_AGENTS, 
                                  weigths_per_agent_list)

                connections_per_agent_list = list(connections_per_agent[cut_off_distance][interaction_type].values())
                self._extend_array(cut_off_distance, interaction_type, CONTACTS_OVER_AGENTS, 
                                  connections_per_agent_list)
                      
    def _extend_array(self, cut_off_distance, interaction_type, cdf_type, newArrayPart):
        try:
            self.combined_data[cut_off_distance][interaction_type][cdf_type] = np.concatenate((self.combined_data[cut_off_distance][interaction_type][cdf_type], 
                                                                                                 newArrayPart))                        
        except (ValueError, TypeError):
            self.combined_data[cut_off_distance][interaction_type][cdf_type] = np.array(newArrayPart)
            
        self.combined_data_indices[cut_off_distance][interaction_type][cdf_type].append(len(self.combined_data[cut_off_distance][interaction_type][cdf_type]))

    def _save_combined_data(self):
        arrays = {}
        for cut_off_distance, interaction_types in self.combined_data.items():
            for interaction_type, cdf_types in interaction_types.items():
                for cdf_type, array in cdf_types.items():
                    array_name = self.array_names[cut_off_distance][interaction_type][cdf_type]
                    arrays[array_name] = array
        
        np.savez(self.combined_stats_filename, **arrays)
        
    def _update_info_file(self, write_permission='w'):
        with open(self.experiment_filename, write_permission) as f:
            json.dump(self.experiment_info, f, indent=4)    
        
    def _check_convergence(self):
        p_values = {}
        has_converged = True
        for cut_off_distance, interaction_types in self.combined_data.items():
            for interaction_type, cdf_types in interaction_types.items():
                for cdf_type, array in cdf_types.items():
                    array_indices = self.combined_data_indices[cut_off_distance][interaction_type][cdf_type]
                    p_values_local, has_converged_local =self._check_convergence_for_array(array, array_indices)
                    has_converged = has_converged and has_converged_local
                    array_name = self.array_names[cut_off_distance][interaction_type][cdf_type]
                    p_values[array_name] = p_values_local
        
        return p_values, has_converged
        
    def _check_convergence_for_array(self, array, array_indices):
        p_values = []
        base_array = array[:array_indices[-1]]
        for ii in range(2,self.convergence_config[STEPS] + 2):
            compare_array = array[:array_indices[-ii]]
            if len(base_array) < 2 or len(compare_array) < 2:
                continue
            
            res = stats.cramervonmises_2samp(base_array, compare_array)
            p_values.append(res.pvalue)
            base_array = compare_array
        
        has_converged = len(p_values) == self.convergence_config[STEPS]
        for p_value in p_values:
            if p_value < self.convergence_config[P_THRESHOLD]:
                has_converged = False
                break
            
        return p_values, has_converged
        
def continue_from_file(experiment_filename, combined_stats_filename):
    # Load experiment file
    with open(experiment_filename, 'r') as f:
        experiment_config_data = json.load(f)
    
    # Create experiment runner
    experiment_runner = ExperimentRunner(Path(experiment_config_data['scenario_filename']), 
                                        Path(experiment_config_data['lrcm_filename']), 
                                        convergence_config=experiment_config_data['convergence_config'], 
                                        cut_off_distances=experiment_config_data['cut_off_distances'],
                                        max_replications=experiment_config_data['max_replications'], 
                                        init_replications=experiment_config_data['init_replications'])
    
    
    for successful_replication in experiment_config_data['successful_replications']:
        experiment_runner.experiment_info['successful_replication_count'] += 1
        experiment_runner.experiment_info['successful_replications'].append(tuple(successful_replication))
        if successful_replication[0] in experiment_runner.seeds_resevoir:
            experiment_runner.seeds_resevoir.remove(successful_replication[0])
     
    for failed_replication_seed in experiment_config_data['failed_replications']:
        experiment_runner.experiment_info['failed_replication_count'] += 1
        experiment_runner.experiment_info['failed_replications'].append(failed_replication_seed)
        if failed_replication_seed in experiment_runner.seeds_resevoir:
            experiment_runner.seeds_resevoir.remove(failed_replication_seed)
       
    
    for convergence_stats in experiment_config_data['convergence_stats']:
        experiment_runner.experiment_info['convergence_stats'].append(convergence_stats)
       
       
    replication_start_nr = experiment_config_data['successful_replication_count']
    experiment_data = np.load(combined_stats_filename)
    
    # Set all experiment runner fields
    experiment_runner.experiment_info['combined_data_info'][INDICES_FLD] = experiment_config_data['combined_data_info'][INDICES_FLD]
    for dist, dist_data in experiment_config_data['combined_data_info'][INDICES_FLD].items():
        dist = float(dist)
        for interaction_type, interaction_data in dist_data.items():
            for cdf_type, indices in interaction_data.items():
                for index in indices:
                    experiment_runner.combined_data_indices[dist][interaction_type][cdf_type].append(index)
                
                combined_label = experiment_runner.array_names[dist][interaction_type][cdf_type]
                experiment_runner.combined_data[dist][interaction_type][cdf_type] = experiment_data[combined_label]
 
    # Init experiment file    
    experiment_runner._update_info_file()
    # Run experiment starting at the provided replication_nr
    experiment_runner.run_experiment(replication_start_nr)
        
def get_array_field_name(cut_off_distance, interaction_type, cdf_type):
    return f'{cut_off_distance}_{interaction_type}_{cdf_type}'

def convert_sub_fields_to_list(data_dict):
    def _convert_entry(dict_part):
        for key, value in dict_part.items():
            if isinstance(value, dict):
                _convert_entry(value)
            else:
                dict_part[key] = list(value)
    
    data_dict_copy = deepcopy(data_dict)
    _convert_entry(data_dict_copy)
    return data_dict_copy
    
def get_scenario_files_from_dir(data_folder):
    scen_ext = f'.{BasicFileOutputManager.SCEN_EXTENSION}'
    return sorted([entry for entry in data_folder.iterdir() if entry.is_file() and entry.suffix == scen_ext])

def load_data(scen_filename):
    scen_data = BasicFileOutputManager.readScenarioDataFile(scen_filename)
    conn_filename = scen_filename.parent.joinpath(scen_data.connFile)
    with open(conn_filename, 'r') as f:
        conn_data = json.load(f)
        
    connections = {tuple(ID_list):conn_data[ID] for ID, ID_list in conn_data['ID2IDtuple'].items()}
    ID_2_group_ID = {int(ID):group_ID for ID, group_ID in conn_data['ID2groupID'].items()}
    
    return scen_data.timeStep, get_seed_from_filename(scen_filename), connections, ID_2_group_ID

def get_seed_from_filename(filename):
    return int(filename.stem.split('_')[-1])

def compute_weighted_graph(cut_off_distances, connections, ID_2_group_ID, time_step):      
    weigths_per_connection = {cut_off_distance:{STAFF_CUSTOMER:[], CUSTOMER_CUSTOMER:[]} for cut_off_distance in cut_off_distances}
    weigths_per_agent = {cut_off_distance:{STAFF_CUSTOMER:defaultdict(float), CUSTOMER_CUSTOMER:defaultdict(float), CUSTOMER_STAFF:defaultdict(float)} for cut_off_distance in cut_off_distances}
    connections_per_agent = {cut_off_distance:{STAFF_CUSTOMER:defaultdict(int), CUSTOMER_CUSTOMER:defaultdict(int), CUSTOMER_STAFF:defaultdict(int)} for cut_off_distance in cut_off_distances}

    for IDtuple, value in connections.items():
        ID_0 = IDtuple[0]
        ID_1 = IDtuple[1]
        if ID_2_group_ID[ID_0] == STAFF_GROUP_NAME:
            interaction_type = STAFF_CUSTOMER
            interaction_type_0 = STAFF_CUSTOMER
            interaction_type_1 = CUSTOMER_STAFF           
        elif ID_2_group_ID[ID_1] == STAFF_GROUP_NAME:
            interaction_type = STAFF_CUSTOMER
            interaction_type_0 = CUSTOMER_STAFF
            interaction_type_1 = STAFF_CUSTOMER
        else:
            interaction_type = CUSTOMER_CUSTOMER
            interaction_type_0 = CUSTOMER_CUSTOMER
            interaction_type_1 = CUSTOMER_CUSTOMER

        valueArray = np.array(value)
        for cut_off_distance in cut_off_distances:
            connectionWeight = time_step*np.sum(valueArray[:,1] <= cut_off_distance)
             
            weigths_per_connection[cut_off_distance][interaction_type].append(connectionWeight)
            weigths_per_agent[cut_off_distance][interaction_type_0][ID_0] += connectionWeight
            weigths_per_agent[cut_off_distance][interaction_type_1][ID_1] += connectionWeight
            connections_per_agent[cut_off_distance][interaction_type_0][ID_0] += 1
            connections_per_agent[cut_off_distance][interaction_type_1][ID_1] += 1

    return weigths_per_connection, weigths_per_agent, connections_per_agent
