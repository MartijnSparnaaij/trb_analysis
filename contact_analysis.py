'''
Created on 13 Jul 2021

@author: Martijn Sparnaaij
'''
from _collections import defaultdict, deque 
from collections import namedtuple
from copy import deepcopy
import json
import logging
from logging.handlers import QueueHandler
from math import ceil, floor
import multiprocessing
from pathlib import Path
from time import sleep
import traceback
import itertools as it

from pubsub import pub
from scipy import stats

import NOMAD
from NOMAD.activity_scheduler import STAFF_GROUP_NAME
from NOMAD.constants import TIMESTEP_MSG_ID, SEND_TIME_MSG_INTERVAL_SEC
from NOMAD.nomad_model import onlyCreate
from NOMAD.output_manager import BasicFileOutputManager, ConnectivityFileOutputManager
from NOMAD.xml_scenario_input import getOutputFolder
import numpy as np


MAX_REPLICATIONS = 5000
INIT_REPLICATIONS = 30
CONSECUTIVE_STEPS = 5
P_THRESHOLD_VALUE = 0.95
STEPS = 'steps'
P_THRESHOLD = 'p_threshold'
CONVERGENCE_CONFIG = {STEPS:CONSECUTIVE_STEPS, P_THRESHOLD:P_THRESHOLD_VALUE}

WEIGHT_OVER_CONTACTS = 'weight_over_contacts'
WEIGHT_OVER_AGENTS = 'weight_over_agents'
CONTACTS_OVER_AGENTS = 'contacts_over_agents'
CDF_TYPES = (WEIGHT_OVER_CONTACTS, WEIGHT_OVER_AGENTS, CONTACTS_OVER_AGENTS)

CUT_OFF_DISTANCES = 'cut_off_distances' 
CUT_OFF_DISTANCES_VALUES = (1.0, 1.5, 2.0) 

STAFF_TYPE = 'staff'
CUSTOMER_TYPE = 'customer'
STAFF_TYPE_NR = 0
CUSTOMER_TYPE_NR = 1

STAFF_CUSTOMER = 'staff_customer'
CUSTOMER_CUSTOMER = 'customer_customer'
CUSTOMER_STAFF = 'customer_staff'

INTERACTION_TYPES = (STAFF_CUSTOMER, CUSTOMER_CUSTOMER, CUSTOMER_STAFF)

ARRAY_NAMES_FLD = 'arrayNames'
ARRAY_NAMES_DICT_FLD = 'arrayNamesDict'
INDICES_FLD = 'indices' 

EXPERIMENT_CONFIG_FILE_ATTRS = (
    'main_dir',
    'scenario_filename',
    'lrcm_filename',
    'combined_stats_filename',
    'simulations_summary_filename',
    'experiment_state_filename',
    'max_replications',
    'init_replications',  
    'cut_off_distances',  
    'convergence_config',
    'array_names',
    'seeds_resevoir'
    )

EXPERIMENT_STATE_FILE_ATTRS = (
    'has_converged',
    'successful_replication_count',
    'failed_replication_count',
    'replication_nr',
    'last_convergence_stats',
    'successful_replications',
    'failed_replications',      
    )

CLOSE_CONTACT_NON_CUM = 0
CLOSE_CONTACT_CUM = 1
CLOSE_CONTACT_BOTH = 2
NON_CLOSE_CONTACT = 3
CLOSE_CONTACT_SET = (CLOSE_CONTACT_NON_CUM, CLOSE_CONTACT_CUM, CLOSE_CONTACT_BOTH)

#LOGGER_NAME = 'contact_experiment'
   

def start_experiments(scenario_filenames, lrcm_filename_per_scenario, log_dir, log_name, convergence_config=CONVERGENCE_CONFIG, cut_off_distances=CUT_OFF_DISTANCES_VALUES, max_replications=MAX_REPLICATIONS, init_replications=INIT_REPLICATIONS):
    run_experiments(scenario_filenames, lrcm_filename_per_scenario, log_dir, log_name, convergence_config, cut_off_distances, max_replications, init_replications)

def run_experiments(scenario_filenames, lrcm_filename_per_scenario, log_dir, log_name, convergence_config, cut_off_distances, max_replications, init_replications, continue_from_file=False):
    createLoggers(log_dir, log_name)
    logger = logging.getLogger(log_name)
    
    exp_runners = []
    logger.info('\n=======================================================================\n')
    if continue_from_file:
        logger.info(f'Continuing experiments for {len(scenario_filenames)} scenarios')
    else:
        logger.info(f'Starting experiments for {len(scenario_filenames)} scenarios')
        
    experiments_state_filename = log_dir.joinpath(f'{log_name}_experiments.state.json')
    
    base_output_folders = [getOutputFolder(scenario_filename) for scenario_filename in scenario_filenames]
    main_dir = get_main_dir(base_output_folders + scenario_filenames + [log_dir])
    seeds_resevoir = create_seed_resevoir(max_replications)
    for scenario_filename in scenario_filenames:
        save_dict = {'main_dir': str(main_dir), 'filenames':[str(scenario_filename.relative_to(main_dir)) for scenario_filename in scenario_filenames], 'lrcm_filename_per_scenario':{str(scenario):str(lrcm_filenanme.relative_to(main_dir)) for scenario, lrcm_filenanme in lrcm_filename_per_scenario.items()}, 'log_dir':str(log_dir.relative_to(main_dir)), 'log_name':log_name, 'convergence_config':convergence_config, 'cut_off_distances':cut_off_distances, 'max_replications':max_replications, 'init_replications':init_replications}
    
        with open(experiments_state_filename, 'w') as f:
            json.dump(save_dict, f, indent=4)    
        
        logger.info(f'Creating experiment runner for {scenario_filename}')        
        exp_runner = ExperimentRunner(main_dir, scenario_filename, lrcm_filename_per_scenario[scenario_filename.stem], log_name, convergence_config, cut_off_distances, max_replications, init_replications)
        exp_runners.append(exp_runner)

        if continue_from_file:
            experiment_config = exp_runner.get_experiment_config()       
            exp_runner.init_experiment_from_state(experiment_config)
        else:
            exp_runner.init_experiment(seeds_resevoir=seeds_resevoir)
            
    queue = multiprocessing.Queue(-1)
    listener = multiprocessing.Process(
        target=listener_process, args=(queue, log_dir, log_name), )
    listener.start()
    h = QueueHandler(queue)
    h.setLevel(logging.INFO)
    logger.addHandler(h)
    
    if continue_from_file:
        pass        
    else:
        with multiprocessing.Pool(processes=3, initializer=pool_process_init, initargs=(queue,)) as pool:
            exp_runners = pool.map(do_initial_replication, exp_runners)
            
    # If continue_from_file finish current replication step
    for exp_runner in exp_runners:
        logger.info(f'{exp_runner.scenario_filename.stem} - {exp_runner.successful_replication_count} - {exp_runner.replication_nr}')
    while not _all_have_converged(exp_runners, logger):      
        with multiprocessing.Pool(processes=3, initializer=pool_process_init, initargs=(queue,)) as pool:
            exp_runners = pool.map(do_one_convergence_step, exp_runners)
     
    return exp_runners

def do_initial_replication(exp_runner):
    exp_runner.do_initial_replications(do_initial_replication.queue)
    return exp_runner
    
def do_one_convergence_step(exp_runner):
    exp_runner.do_one_convergence_step(do_one_convergence_step.queue)
    return exp_runner
    
def _all_have_converged(exp_runners, logger):
    logger.info(f'{"="*40}\n')
    logger.info('Checking convergence....')
    not_converged_count = 0
    for exp_runner in exp_runners:
        if not exp_runner.has_converged:
            not_converged_count += 1
        
    
    if not_converged_count > 0:
        logger.info(f'No convergence. {not_converged_count} out of {len(exp_runners)} have not converged\n')
        return False
    
    logger.info('CONVERGENCE!!!!\n')
    return True
    
def pool_process_init(queue):
    do_initial_replication.queue = queue
    do_one_convergence_step.queue = queue
        
def continue_experiments(experiments_config_filename):
    with open(experiments_config_filename, 'r') as f:
        experiment_config = json.load(f)
    
    main_dir = Path(experiment_config['main_dir'])
    scenario_filenames = [main_dir.joinpath(scenario_filename) for scenario_filename in experiment_config['filenames']]
    lrcm_filename_per_scenario = {}
    for scenario, lrcm_filename in experiment_config['lrcm_filename_per_scenario'].items():
        lrcm_filename_per_scenario[scenario] = main_dir.joinpath(lrcm_filename)
    log_dir = main_dir.joinpath(experiment_config['log_dir'])
    log_name = experiment_config['log_name']
    convergence_config = experiment_config['convergence_config']
    cut_off_distances = experiment_config['cut_off_distances']
    max_replications = experiment_config['max_replications']
    init_replications = experiment_config['init_replications']
            
    run_experiments(scenario_filenames, lrcm_filename_per_scenario, log_dir, log_name, convergence_config, cut_off_distances, max_replications, init_replications, continue_from_file=True)

def listener_process(queue, log_dir, log_name):
    logger = createLoggers(log_dir, log_name)
    while True:
        while not queue.empty():
            record = queue.get()            
            logger.handle(record)
        sleep(1)

class ExperimentRunner():
    
    def __init__(self, main_dir, scenario_filename, lrcm_filename, log_name, convergence_config=CONVERGENCE_CONFIG, cut_off_distances=CUT_OFF_DISTANCES_VALUES, max_replications=MAX_REPLICATIONS, init_replications=INIT_REPLICATIONS, queue=None):
        self.logger = logging.getLogger(log_name)
        self.queue = queue
        self.log_name = log_name 
        self.main_dir = main_dir
        self.scenario_filename = Path(scenario_filename)
        self.lrcm_filename = Path(lrcm_filename)
        self.convergence_config = convergence_config
        self.cut_off_distances = cut_off_distances
        self.max_replications = max_replications
        self.init_replications = init_replications
        self.has_converged = False
        
        self.combined_data = {cut_off_distance:{STAFF_CUSTOMER:{cdf_type:None for cdf_type in CDF_TYPES}, 
                                                CUSTOMER_CUSTOMER:{cdf_type:None for cdf_type in CDF_TYPES},
                                                CUSTOMER_STAFF:{cdf_type:None for cdf_type in (WEIGHT_OVER_AGENTS, CONTACTS_OVER_AGENTS)}} 
                                      for cut_off_distance in self.cut_off_distances}
        self.combined_data_indices = {cut_off_distance:{STAFF_CUSTOMER:{cdf_type:deque('', self.convergence_config[STEPS] + 1) for cdf_type in CDF_TYPES}, 
                                                        CUSTOMER_CUSTOMER:{cdf_type:deque('', self.convergence_config[STEPS] + 1) for cdf_type in CDF_TYPES},
                                                       CUSTOMER_STAFF:{cdf_type:deque('', self.convergence_config[STEPS] + 1) for cdf_type in (WEIGHT_OVER_AGENTS, CONTACTS_OVER_AGENTS)}} 
                                      for cut_off_distance in self.cut_off_distances} # Of the last 5 entries

        #pub.subscribe(self._update_time, TIMESTEP_MSG_ID)
            
    def init_experiment(self, recreated=False, seeds_resevoir=None, base_output_folder=None):
        if base_output_folder is None:
            base_output_folder = getOutputFolder(self.scenario_filename)
        output_folder = base_output_folder.joinpath('summarized') 
        if not output_folder.is_dir():
            output_folder.mkdir(parents=True)
        
        self._base_output_folder = base_output_folder
        
        if recreated:
            recreated_str = '_recreated'
        else:  
            recreated_str = ''
            
        self.experiment_config_filename = output_folder.joinpath(f'{self.scenario_filename.stem}_experiment{recreated_str}.config.json')
        self.simulations_summary_filename = output_folder.joinpath(f'{self.scenario_filename.stem}_simulations_summary{recreated_str}.res')
        self.experiment_state_filename = output_folder.joinpath(f'{self.scenario_filename.stem}_experiment{recreated_str}.state.json')
        self.combined_stats_filename = output_folder.joinpath(f'{self.scenario_filename.stem}_combined{recreated_str}.stats.npz')            
        
        if seeds_resevoir is None:
            rng = np.random.default_rng()
            self.seeds_resevoir = rng.choice(np.arange(1,self.max_replications*2), self.max_replications, False).tolist()
        else:
            self.seeds_resevoir = seeds_resevoir 
        self.array_names = {cut_off_distance:{
                            STAFF_CUSTOMER:{cdf_type:get_array_field_name(cut_off_distance, STAFF_CUSTOMER, cdf_type) for cdf_type in CDF_TYPES}, 
                            CUSTOMER_CUSTOMER:{cdf_type:get_array_field_name(cut_off_distance, CUSTOMER_CUSTOMER, cdf_type) for cdf_type in CDF_TYPES},
                            CUSTOMER_STAFF:{cdf_type:get_array_field_name(cut_off_distance, CUSTOMER_STAFF, cdf_type) for cdf_type in (WEIGHT_OVER_AGENTS, CONTACTS_OVER_AGENTS)}} 
                            for cut_off_distance in self.cut_off_distances}
         
        self.successful_replication_count = 0
        self.failed_replication_count = 0
        self.last_convergence_stats = None
        self.successful_replications = []
        self.failed_replications = [] 
        self.replication_nr = 0
         
        self._create_experiment_config_file()
        self._create_experiment_state_file()
        self._create_simulations_summary_file()        
            
    def init_experiment_from_state(self, experiment_config):
        self.combined_stats_filename = self.main_dir.joinpath(experiment_config['combined_stats_filename'])
        self.simulations_summary_filename = self.main_dir.joinpath(experiment_config['simulations_summary_filename'])
        self.experiment_state_filename = self.main_dir.joinpath(experiment_config['experiment_state_filename'])
        self.array_names = {}
        for cut_off_distance, interaction_types in list(experiment_config['array_names'].items()):
            self.array_names[float(cut_off_distance)] = interaction_types
        
        self.seeds_resevoir = experiment_config['seeds_resevoir']    
    
        with open(self.experiment_state_filename, 'r') as f:
                experiment_state = json.load(f)
            
        self.successful_replication_count = experiment_state['successful_replication_count']
        self.failed_replication_count = experiment_state['failed_replication_count']
        self.last_convergence_stats = experiment_state['last_convergence_stats']
        self.successful_replications = experiment_state['successful_replications']
        self.failed_replications = experiment_state['failed_replications'] 
        self.replication_nr = experiment_state['replication_nr']
        self.has_converged = experiment_state['has_converged']
    
        num_data = np.load(self.combined_stats_filename, allow_pickle=True)
        arrays_dict = num_data['arrays'].item()
        indices_dict = num_data['indices'].item() 

        for cut_off_distance, interaction_types in self.array_names.items():            
            for interaction_type, cdf_types in interaction_types.items():
                for cdf_type, array_name in cdf_types.items():
                    self.combined_data[cut_off_distance][interaction_type][cdf_type] = arrays_dict[array_name]
                    self.combined_data_indices[cut_off_distance][interaction_type][cdf_type] = indices_dict[array_name]
    
    def _update_time(self, timeInd, currentTime): # @UnusedVariable
        print('.', end='')

    def config_logger(self, queue):
        self.logger = logging.getLogger(f'{self.scenario_filename.stem}')
        self.logger.setLevel(logging.INFO)
        h = QueueHandler(queue)
        h.setLevel(logging.INFO)
        self.logger.addHandler(h) 

    def do_initial_replications(self, queue):
        self.config_logger(queue)
        
        for replication_nr in range(self.init_replications):
            self.replication_nr = replication_nr
            self._run_replication()
            
        self.replication_nr += 1
        self._check_convergence(force=True)
            
    def continue_convergence_step(self, queue):
        self.config_logger(queue)
        
        step_count = self.convergence_config['steps'] - (self.replication_nr % self.convergence_config['steps'])
        for _ in range(step_count):
            self._run_replication()
            self.replication_nr += 1
            
        self._check_convergence(force=True)
            
    def do_one_convergence_step(self, queue):
        self.config_logger(queue)
        
        for _ in range(self.convergence_config['steps']):
            self._run_replication()
            self.replication_nr += 1 
            
        self._check_convergence(force=True)
            
    def run_experiment(self):
        self.last_convergence_check = self.init_replications - 1 - self.convergence_config['steps']
        if self.replication_nr - self.failed_replication_count > self.last_convergence_check + self.convergence_config['steps']:
            self.last_convergence_check = floor((self.replication_nr - self.failed_replication_count)/self.convergence_config['steps'])*self.convergence_config['steps']           
        
        while self.replication_nr < self.max_replications:
            self._run_replication()
            
            has_converged = self._check_convergence()
            if has_converged:                
                break
            self.replication_nr += 1            
        
    def _run_replication(self):
        seed = self.seeds_resevoir[self.replication_nr]
        self.logger.info(f'Running replication {self.replication_nr} with seed = {seed}')            
        NOMAD.NOMAD_RNG = np.random.default_rng(seed)
        try:
            nomad_model = onlyCreate(self.scenario_filename, lrcmLoadFile=self.lrcm_filename, seed=seed, logStreamLevel=None) #
            #self.logger.info('='*ceil(nomad_model.duration/SEND_TIME_MSG_INTERVAL_SEC))
            nomad_model.start()
        except:
            self.logger.info(f'{self.replication_nr}: FAILED')
            traceback.print_exc()
            print('')
            try:
                scen_filename = nomad_model.outputManager.scenarioFilename
            except:
                scen_filename = ''
            try:
                del nomad_model
            except: pass
            self.failed_replication_count += 1
            self.failed_replications.append(seed)
            self._update_files_after_sim(seed, scen_filename, False)
            return

        self.logger.info(f'{self.replication_nr}: Simulation done. Processing data... ')
        # Process output accessing it directly via the nomad_model instance
        self._process_data(nomad_model)
        self.successful_replication_count += 1
        self.successful_replications.append(seed)
        self._update_files_after_sim(seed, nomad_model.outputManager.scenarioFilename, True)
        del nomad_model
        self.logger.info(f'{self.replication_nr}: done')
            
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

    def _check_convergence(self, force=False):
        if force or (self.replication_nr - self.failed_replication_count == self.last_convergence_check + self.convergence_config['steps']):
            # Check convergence
            self.logger.info(f'{self.replication_nr}: Checking convergence...')
            p_values, has_converged = self._check_convergence_for_arrays()                 
            self.last_convergence_stats = p_values
            self.last_convergence_check = self.replication_nr    
            self.logger.info('done')
            if has_converged:
                self.has_converged = True                    
                self.logger.info(f'{"="*40}\n')
                self.logger.info(f'{self.replication_nr}: FINISHED!!!!!!!!!!\n')
                self.logger.info(f'{"="*40}')
            
            self._update_experiment_state_file()
            return has_converged                                
        return False
    
    def _check_convergence_for_arrays(self):
        p_values = {}
        has_converged = True
        for cut_off_distance, interaction_types in self.combined_data.items():
            for interaction_type, cdf_types in interaction_types.items():
                for cdf_type, array in cdf_types.items():
                    array_indices = self.combined_data_indices[cut_off_distance][interaction_type][cdf_type]
                    is_counts_array = cdf_type == CONTACTS_OVER_AGENTS
                    p_values_local, has_converged_local = self._check_convergence_for_array(array, array_indices, is_counts_array)
                    has_converged = has_converged and has_converged_local
                    array_name = self.array_names[cut_off_distance][interaction_type][cdf_type]
                    p_values[array_name] = p_values_local
        
        return p_values, has_converged
        
    def _check_convergence_for_array(self, array, array_indices, is_counts_array=False):
        p_values = []
        base_array = array[:array_indices[-1]]
        for ii in range(2,self.convergence_config[STEPS] + 2):
            compare_array = array[:array_indices[-ii]]
            if len(base_array) < 2 or len(compare_array) < 2:
                continue
            
            if is_counts_array:
                p_value = do_chi_square_test(base_array, compare_array)
                p_values.append(p_value)
            else:
                res = stats.cramervonmises_2samp(base_array, compare_array)
                p_values.append(res.pvalue)
            base_array = compare_array
        
        has_converged = len(p_values) == self.convergence_config[STEPS]
        for p_value in p_values:
            if p_value < self.convergence_config[P_THRESHOLD]:
                has_converged = False
                break
            
        return p_values, has_converged
        
    # ====================================================================================================
        
    def _create_experiment_config_file(self):
        save_dict = {}
        for attr_name in EXPERIMENT_CONFIG_FILE_ATTRS:
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, Path):
                if attr_value == self.main_dir:
                    attr_value = str(attr_value)
                else:
                    attr_value = str(attr_value.relative_to(self.main_dir))
            save_dict[attr_name] = attr_value

        with open(self.experiment_config_filename, 'w') as f:
            json.dump(save_dict, f, indent=4)     

    def _create_experiment_state_file(self):
        self._update_experiment_state_file()   

    def _create_simulations_summary_file(self):
        with open(self.simulations_summary_filename, 'w') as f:
            f.write('#Seed, scenario filename, success\n') 

    def _update_files_after_sim(self, seed, scen_filename, success):
        self._update_experiment_state_file()
        self._update_simulations_summary_file(seed, scen_filename, success)
        if success:
            self._update_combined_stats_file()
        
    def _update_experiment_state_file(self):
        save_dict = {}
        for attr_name in EXPERIMENT_STATE_FILE_ATTRS:
            save_dict[attr_name] = getattr(self, attr_name)

        with open(self.experiment_state_filename, 'w') as f:
            json.dump(save_dict, f, indent=4)     

    def _update_simulations_summary_file(self, seed, scen_filename, success):
        try:
            scen_filename_str = str(get_relative_path_to(scen_filename, self.simulations_summary_filename))
        except:
            scen_filename_str = str(scen_filename)
        
        with open(self.simulations_summary_filename, 'a') as f:
            f.write(f'{seed}, {scen_filename_str}, {success}\n') 
    
    def _update_combined_stats_file(self):
        arrays = {}
        indices = {}
        for cut_off_distance, interaction_types in self.combined_data.items():
            for interaction_type, cdf_types in interaction_types.items():
                for cdf_type, array in cdf_types.items():
                    array_name = self.array_names[cut_off_distance][interaction_type][cdf_type]
                    arrays[array_name] = array
                    indices[array_name] = self.combined_data_indices[cut_off_distance][interaction_type][cdf_type]
                    
        data_2_save = {'arrays':arrays, 'indices':indices}
        
        np.savez(self.combined_stats_filename, **data_2_save)

    # ====================================================================================================

    def get_experiment_config(self):
        base_output_folder = getOutputFolder(self.scenario_filename)
        output_folder = base_output_folder.joinpath('summarized') 
        if not output_folder.is_dir():
            output_folder.mkdir(parents=True)
        
        self.experiment_config_filename = output_folder.joinpath(f'{self.scenario_filename.stem}_experiment.config.json')

        if not self.experiment_config_filename.is_file():
            return None

        with open(self.experiment_config_filename, 'r') as f:
            experiment_config = json.load(f)

        return experiment_config 

    # ====================================================================================================

    def recreate_from_folder(self, seeds_resevoir, base_output_folder=None, last_seed=None):
        # Get a scenario file from folder
        # Get time step from file
        # Get all conncetion files 
        # Process all files
        self.init_experiment(recreated=True, seeds_resevoir=seeds_resevoir, base_output_folder=base_output_folder)
        self.last_convergence_check = self.init_replications - 1 - self.convergence_config['steps']
        scen_filenames = get_scenario_files_from_dir(self._base_output_folder)
        scen_filename_seed = {get_seed_from_filename(scen_filename):scen_filename for scen_filename in scen_filenames}
               
        for self.replication_nr, seed in enumerate(self.seeds_resevoir):
            if last_seed is not None and seed == last_seed:
                stop_after = True
            else:
                stop_after = False
                
            self.logger.info(f'Processing {self.replication_nr} with seed = {seed}')   
            try:
                scen_filename = scen_filename_seed[seed]
                time_step, _, connections, ID_2_group_ID = load_data(scen_filename)
            except:
                self.logger.info('FAILED')
                traceback.print_exc()
                self.logger.info('')
                self.failed_replication_count += 1
                self.failed_replications.append(seed)
                try: 
                    self._update_files_after_sim(seed, scen_filename, False)
                except:
                    self._update_files_after_sim(seed, '', False)
                continue
            
            self.logger.info('Loading done. Processing data... ')
            self._process_data(None, connections, ID_2_group_ID, time_step)
            
            self.successful_replication_count += 1
            self.successful_replications.append(seed)
            self._update_files_after_sim(seed, scen_filename, True)
            self.logger.info('done')
            
            has_converged = self._check_convergence()
            if has_converged or stop_after:
                break
            
# ========================================================================================================
# ========================================================================================================       
    
def recreate_from_folder(experiment_config_filename, base_output_folder=None, last_seed=None):
    # Get a scenario file from folder
    # Get time step from file
    # Get all conncetion files 
    # Process all files
    log_name = 'recreation'
    createLoggers(None, log_name)
    with open(experiment_config_filename, 'r') as f:
        experiment_config = json.load(f)
    
    scenario_filename = Path(experiment_config['scenario_filename'])
    lrcm_filename = Path(experiment_config['lrcm_filename'])
    convergence_config = experiment_config['convergence_config']
    cut_off_distances = experiment_config['cut_off_distances']
    max_replications = experiment_config['max_replications']
    init_replications = experiment_config['init_replications']
    seeds_resevoir = experiment_config['seeds_resevoir']
      
    try:
        main_dir = Path(experiment_config['main_dir'])
    except:
        main_dir = get_main_dir([scenario_filename, Path(experiment_config['simulations_summary_filename'])])
     
    exp_runner = ExperimentRunner(main_dir, scenario_filename, lrcm_filename, log_name, convergence_config, cut_off_distances, max_replications, init_replications)
    exp_runner.recreate_from_folder(seeds_resevoir, base_output_folder, last_seed)
    
def get_connection_files_from_dir(data_folder):
    scen_ext = f'.{ConnectivityFileOutputManager.CONNECTIVITY_EXTENSION}.npz'
    return sorted([entry for entry in data_folder.iterdir() if entry.is_file() and entry.suffix == scen_ext])

        
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
    def get_data_with_ID(ID, conv2str):
        if conv2str:
            ID = str(ID)
        return conn_data[ID]
    scen_data = BasicFileOutputManager.readScenarioDataFile(scen_filename)
    conn_filename = scen_filename.parent.joinpath(scen_data.connFile)
    try:
        with open(conn_filename, 'r') as f:
            conn_data = json.load(f)
        ID2IDtuple = conn_data['ID2IDtuple']
        ID2groupID = conn_data['ID2groupID']
        conv2str = False
    except FileNotFoundError:
        conn_data = np.load(f'{str(conn_filename)}.npz', allow_pickle=True)
        ID2IDtuple = conn_data['ID2IDtuple'].item()
        ID2groupID = conn_data['ID2groupID'].item()
        conv2str = True
            
    connections = {tuple(ID_list):get_data_with_ID(ID, conv2str) for ID, ID_list in ID2IDtuple.items()}
    ID_2_group_ID = {int(ID):group_ID for ID, group_ID in ID2groupID.items()}
    
    return scen_data.timeStep, get_seed_from_filename(scen_filename), connections, ID_2_group_ID

def convert_conn_file(conn_filename):
    try: 
        with open(conn_filename, 'r') as f:
            conn_data = json.load(f)
    except:
        return
    
    np.savez(conn_filename, **conn_data)
        
def get_seed_from_filename(filename):
    return int(filename.stem.split('_')[-1])

def do_chi_square_test(base_array, compare_array):
    base_values, base_counts = np.unique(base_array, return_counts=True)
    base_counts = base_counts
    base_counts_dict = dict(zip(base_values, base_counts))
    compare_values, compare_counts = np.unique(compare_array, return_counts=True)
    compare_counts = compare_counts/sum(compare_counts)*sum(base_counts)
    compare_counts_dict = dict(zip(compare_values, compare_counts))
    values = np.unique(np.concatenate((base_values, compare_values)))
    
    
    base_counts = [base_counts_dict[value] if value in base_counts_dict else 0 for value in values]
    compare_counts = [compare_counts_dict[value] if value in compare_counts_dict else 0 for value in values]
    
    if len(values) == 1:
        return 1.0
    
    res = stats.chisquare(compare_counts, base_counts)
    return res.pvalue    

def compute_weighted_graph(cut_off_distances, connections, ID_2_group_ID, time_step):      
    weigths_per_connection = {cut_off_distance:{STAFF_CUSTOMER:[], CUSTOMER_CUSTOMER:[]} for cut_off_distance in cut_off_distances}
    weigths_per_agent = {cut_off_distance:{STAFF_CUSTOMER:defaultdict(float), CUSTOMER_CUSTOMER:defaultdict(float), CUSTOMER_STAFF:defaultdict(float)} for cut_off_distance in cut_off_distances}
    connections_per_agent = {cut_off_distance:{STAFF_CUSTOMER:defaultdict(int), CUSTOMER_CUSTOMER:defaultdict(int), CUSTOMER_STAFF:defaultdict(int)} for cut_off_distance in cut_off_distances}

    for ID_tuple, value in connections.items():
        ID_0 = ID_tuple[0]
        ID_1 = ID_tuple[1]
        interaction_type, interaction_type_0, interaction_type_1 = get_interaction_type_from_IDs(ID_2_group_ID, ID_0, ID_1)

        valueArray = np.array(value)
        for cut_off_distance in cut_off_distances:
            connection_weight = calc_connection_weight(time_step, valueArray, cut_off_distance)
            if connection_weight == 0:
                continue
             
            weigths_per_connection[cut_off_distance][interaction_type].append(connection_weight)
            weigths_per_agent[cut_off_distance][interaction_type_0][ID_0] += connection_weight
            weigths_per_agent[cut_off_distance][interaction_type_1][ID_1] += connection_weight
            connections_per_agent[cut_off_distance][interaction_type_0][ID_0] += 1
            connections_per_agent[cut_off_distance][interaction_type_1][ID_1] += 1

    return weigths_per_connection, weigths_per_agent, connections_per_agent

def calc_connection_weight(time_step, valueArray, cut_off_distance):
    return time_step*np.sum(valueArray[:,1] <= cut_off_distance)

def get_interaction_type_from_IDs(ID_2_group_ID, ID_0, ID_1):
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
        
    return interaction_type, interaction_type_0, interaction_type_1
        
def get_contacts_longer_than_value(data_folder, boundary_value, cut_off_distances, interaction_type):
    scen_filenames = get_scenario_files_from_dir(data_folder)
    print(len(scen_filenames))
    for scen_filename in scen_filenames:
        try:
            time_step, _, connections, ID_2_group_ID = load_data(scen_filename)
        except:
            traceback.print_exc()
            print(f'FAILED - {scen_filename.name}')
            continue
        contacts_longer_than_value = []
        for ID_tuple, value in connections.items():
            interaction_type_ID_tuple, _, _ = get_interaction_type_from_IDs(ID_2_group_ID, ID_tuple[0], ID_tuple[1])
            if interaction_type_ID_tuple != interaction_type:
                continue
            value_array = np.array(value)
            for cut_off_distance in cut_off_distances:
                connection_weight = calc_connection_weight(time_step, value_array, cut_off_distance)
                if connection_weight > boundary_value:
                    contacts_longer_than_value.append((ID_tuple, cut_off_distance, connection_weight, value_array))
                    #print(f'{scen_filename.name} - {cut_off_distance} - {ID_tuple} = {connection_weight}')
                    
        if len(contacts_longer_than_value) == 0:
            print('.')
            continue
        
        scen_data = BasicFileOutputManager.readScenarioDataFile(scen_filename)
        desc_filename = scen_filename.parent.joinpath(scen_data.descFile)
        pedestrians = BasicFileOutputManager.readDescriptionDataFile(desc_filename)
        print(scen_filename.name)
        
        for contact_longer_than_value in contacts_longer_than_value:
            ID_tuple = contact_longer_than_value[0]
            value_array = contact_longer_than_value[3]
            ped_0_data = pedestrians[ID_tuple[0]]
            ped_1_data = pedestrians[ID_tuple[1]]
            #if ped_0_data.activityLog[0].destinationID in ("chair40_eetstoel", "chair21_eetstoel") or \
            #    ped_1_data.activityLog[0].destinationID in ("chair40_eetstoel", "chair21_eetstoel"):
            #    continue
            
            print(f'{ID_tuple} - ({ped_0_data.activityLog[0].destinationID}, {ped_1_data.activityLog[0].destinationID}): {value_array[0,0]:.2f} - {value_array[-1,0]:.2f} - {contact_longer_than_value[2]} - {contact_longer_than_value[1]}')
            #for ii in range(len(value_array)):
            #    print(f'{value_array[ii,0]:.2f} - {value_array[ii,1]:.2f}')
        
        print('')
    
    
def createLoggers(log_dir, log_name):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter("{asctime} - {name} - {levelname} - {message}", '%Y/%m/%d %H:%M:%S', style='{')
    
    import sys    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    if log_dir is None:
        return logger
    
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        log_dir.mkdir(parents=True)

    log_filename = log_dir.joinpath(f'{log_name}.log')
    file_handeler = logging.handlers.RotatingFileHandler(log_filename, maxBytes=1024*1024*30, backupCount=10)
    file_handeler.setFormatter(formatter)
    file_handeler.setLevel(logging.INFO)
    logger.addHandler(file_handeler)
    
    return logger

# =============================================================

def get_main_dir(path_list):
    for main_dir in path_list[0].parents:
        is_relative_to_all = True
        for path_el in path_list:
            if not path_el.is_relative_to(main_dir):
                is_relative_to_all = False
                break
            
        if is_relative_to_all:
            return main_dir
    
    raise Exception('The provided paths have no common parent')

def get_relative_path_to(main_path, reference_path):
    if reference_path.is_file():
        reference_path = reference_path.parent
    
    if main_path.is_relative_to(reference_path):
        return main_path.relative_to(reference_path)
        
    for parent_ind, parent_dir in enumerate(reference_path.parents):
        if main_path.is_relative_to(parent_dir):
            relative_path = main_path.relative_to(parent_dir)
            for _ in range(parent_ind + 1):
                relative_path = Path('../').joinpath(relative_path)
            return relative_path
                
    raise Exception('The provided paths have no common parent')

# ================================================================================================
def create_seed_resevoir(max_replications, return_as_list=True):
    rng = np.random.default_rng()
    seed_resevoir = rng.choice(np.arange(1,max_replications*2), max_replications, False)
    if return_as_list: 
        return seed_resevoir.tolist()
    
    return seed_resevoir

def create_seed_resevoir_file(seeds_filename, max_replications):
    seeds_resevoir = create_seed_resevoir(max_replications, return_as_list=False)
    np.save(seeds_filename,seeds_resevoir)
    print(seeds_filename)
    
def get_seed_resevoir_from_file(seeds_filename):
    return np.load(seeds_filename).tolist()

# ================================================================================================
def start_experiment(scenario_filename, lrcm_filename, log_name, convergence_config=CONVERGENCE_CONFIG, cut_off_distances=CUT_OFF_DISTANCES_VALUES, max_replications=MAX_REPLICATIONS, init_replications=INIT_REPLICATIONS):
    exp_runner = ExperimentRunner(scenario_filename, lrcm_filename, log_name, convergence_config, cut_off_distances, max_replications, init_replications)
    exp_runner.init_experiment()
    exp_runner.run_experiment()
    return exp_runner

def continue_experiment(experiment_config_filename):
    with open(experiment_config_filename, 'r') as f:
        experiment_config = json.load(f)
        
    exp_runner = ExperimentRunner(experiment_config['scenario_filename'], experiment_config['lrcm_filename'], 
                                  experiment_config['convergence_config'], experiment_config['cut_off_distances'], 
                                  experiment_config['max_replications'], experiment_config['init_replications'])    

    exp_runner.init_experiment_from_state(experiment_config)
    exp_runner.run_experiment()
    return exp_runner


# ================================================================================================

def get_agent_count_data(output_filename, config_filenames):
    count_per_set = {}
    
    for config_filename in config_filenames:
        print(f'Processing {config_filename.stem}...')
        count_data, infra_name, measure_name = get_count_data_from_config_file(config_filename)
        if infra_name not in count_per_set:
            count_per_set[infra_name] = {}
        
        print(f'{infra_name} - {measure_name} - {count_data}')
        count_per_set[infra_name][measure_name] = count_data
        
    with open(output_filename, 'w') as f:
        json.dump(count_per_set, f, indent=4)    

    
def get_count_data_from_config_file(config_filename):
    experiment_config, infra_name, measure_name = get_config_file_content(config_filename)
    experiment_state, experiment_state_filename = get_experiment_state(experiment_config)
    scenario_filenames = get_scenario_filenames(experiment_state_filename.parent.parent, experiment_state['successful_replications'])

    agent_count, has_contact_count = get_count_data_from_scen_files(scenario_filenames, experiment_config['cut_off_distances'])

    count_data = {'replication_count':experiment_state['successful_replication_count'],
                  'agent_count':agent_count, 'has_contact_count':has_contact_count}

    return count_data, infra_name, measure_name



def get_count_data_from_scen_files(scenario_filenames, cut_off_distances):
    agent_count = 0
    has_contact_count = defaultdict(int)
    print(f'{"="*len(scenario_filenames)}')
    for scenario_filename in scenario_filenames:
        print('.', end='')
        try:
            peds, conn_data, _ = get_ped_and_conn_data(scenario_filename)
        except:
            continue
        
        agent_count += len(peds)
        ped_ids = list(peds.keys())
                
        IDs_processed = defaultdict(set)
        for ID_tuple, data_matrix in conn_data['connData'].items():
            for cut_off_dist_str in cut_off_distances:
                cut_off_dist = float(cut_off_dist_str)
                if np.any(data_matrix[:,1] <= cut_off_dist):
                    IDs_processed[cut_off_dist_str].add(ID_tuple[0])
                    IDs_processed[cut_off_dist_str].add(ID_tuple[1])
               
        for cut_off_dist_str, peds in IDs_processed.items():
            has_contact_count[cut_off_dist_str] += len(peds)
       
    print('')             
    return agent_count, has_contact_count
            
# ================================================================================================

def get_agent_contact_data(output_filename, config_filenames, contact_data_config):
    process_contact_data_config(contact_data_config)
    contact_data_all = defaultdict(dict)
    for config_filename in config_filenames:
        print(f'Processing {config_filename.stem}...')
        experiment_config, infra_name, measure_name = get_config_file_content(config_filename)
        experiment_state, experiment_state_filename = get_experiment_state(experiment_config)
        scenario_filenames = get_scenario_filenames(experiment_state_filename.parent.parent, experiment_state['successful_replications'])
        contact_data = get_agent_contact_data_from_scen_file(scenario_filenames, contact_data_config)
        contact_data_all[infra_name][measure_name] = contact_data
                        
    np.savez(output_filename, **contact_data_all)
        
def process_contact_data_config(contact_data_config):
    ThresholdSet = namedtuple('ThresholdSet', ['non_cum', 'cum', 'key'])
    Thresholds = namedtuple('Thresholds', ['dist', 'time', 'use_in_combs'])
    contacts = []
    for t_set in contact_data_config['contacts']:
        contacts.append(ThresholdSet(Thresholds(*t_set[0], False), Thresholds(*t_set[1]), get_key(t_set)))
    
    contact_data_config['contacts'] = contacts
   
def get_key(t_set):
    if t_set[0][0] is None:
        return f'nc_None_c_dist={t_set[1][0]:.1f}_time={t_set[1][1]:.1f}'
    else:
        return f'nc_dist={t_set[0][0]:.1f}_time={t_set[0][1]:.1f}_c_dist={t_set[1][0]:.1f}_time={t_set[1][1]:.1f}'
        
def get_agent_contact_data_from_scen_file(scenario_filenames, contact_data_config):
    contact_data_base = {'contacts':{}, 'combs':{}}
    for contact_t_set in contact_data_config['contacts']:
        contact_data_base['contacts'][contact_t_set.key] = {'cc_count_per_agent':[], 
                                                            'cc_count_both':0, 'cc_count_nc':0, 'cc_count_c':0}
            
    contact_data = {'staff_count':0, 'customer_count':0}
    for interaction_type in INTERACTION_TYPES:
        contact_data[interaction_type] = deepcopy(contact_data_base)
    
    file_count = len(scenario_filenames)
    for ii, scenario_filename in enumerate(scenario_filenames):
        print(f'Processing {ii+1} of {file_count}')
        process_scen_file_contact_data(scenario_filename, contact_data, contact_data_config)
                
    return contact_data
        
def process_scen_file_contact_data(scenario_filename, contact_data, contact_data_config):
    try:
        peds, conn_data, time_step = get_ped_and_conn_data(scenario_filename)
    except:
        return
    
    process_contacts(peds, conn_data, contact_data, contact_data_config, time_step)    

def process_contacts(peds, conn_data, contact_data, contact_data_config, time_step):
    group_ids = [ped.groupID for ped in peds.values()]
    staff_count = group_ids.count('staff')
    contact_data['customer_count'] += len(peds) - staff_count
    contact_data['staff_count'] += staff_count

    id_2_group_id = conn_data['ID2groupID']
    
    for t_set in contact_data_config['contacts']:
        for interaction_type in INTERACTION_TYPES:
            contact_data[interaction_type]['contacts'][t_set.key]['cc_count_per_agent_tmp'] = defaultdict(int)
    
    for id_tuple, proximity_data in conn_data['connData'].items():
        interaction_type, group_type_0, group_type_1 = get_interaction_type(id_tuple, id_2_group_id)
        for t_set in contact_data_config['contacts']:
            contact_type, _ = is_close_contact(proximity_data, time_step, t_set.non_cum, t_set.cum, interaction_type)
            if contact_type == CLOSE_CONTACT_BOTH:
                contact_data[interaction_type]['contacts'][t_set.key]['cc_count_both'] += 1
            elif contact_type == CLOSE_CONTACT_NON_CUM:
                contact_data[interaction_type]['contacts'][t_set.key]['cc_count_nc'] += 1
            elif contact_type == CLOSE_CONTACT_CUM:
                contact_data[interaction_type]['contacts'][t_set.key]['cc_count_c'] += 1
                
            if contact_type in CLOSE_CONTACT_SET:
                if interaction_type == STAFF_CUSTOMER:
                    if group_type_0 == CUSTOMER_TYPE:
                        contact_data[CUSTOMER_STAFF]['contacts'][t_set.key]['cc_count_per_agent_tmp'][id_tuple[0]] += 1
                    else:
                        contact_data[interaction_type]['contacts'][t_set.key]['cc_count_per_agent_tmp'][id_tuple[0]] += 1
                    if group_type_1 == CUSTOMER_TYPE:
                        contact_data[CUSTOMER_STAFF]['contacts'][t_set.key]['cc_count_per_agent_tmp'][id_tuple[1]] += 1
                    else:
                        contact_data[interaction_type]['contacts'][t_set.key]['cc_count_per_agent_tmp'][id_tuple[1]] += 1
                else:
                    contact_data[interaction_type]['contacts'][t_set.key]['cc_count_per_agent_tmp'][id_tuple[0]] += 1
                    contact_data[interaction_type]['contacts'][t_set.key]['cc_count_per_agent_tmp'][id_tuple[1]] += 1
    
    for t_set in contact_data_config['contacts']:
        for interaction_type in INTERACTION_TYPES:
            contact_data[interaction_type]['contacts'][t_set.key]['cc_count_per_agent'] += list(contact_data[interaction_type]['contacts'][t_set.key]['cc_count_per_agent_tmp'].values())
            contact_data[interaction_type]['contacts'][t_set.key].pop('cc_count_per_agent_tmp')
        # for comb_info in contact_data_config['combs']:
        #     dist_threshold = comb_info[1]
        #     time_within_dist = get_time_within_dist(proximity_data, dist_threshold, time_step)
        #     if time_within_dist > 0:
        #         if group_type_1 not in combs_data[comb_info][id_tuple[0]]:
        #             combs_data[comb_info][id_tuple[0]][group_type_1] = []
        #             combs_data[comb_info][id_tuple[0]]['group_type'] = group_type_0
        #         if group_type_0 not in combs_data[comb_info][id_tuple[1]]:
        #             combs_data[comb_info][id_tuple[1]][group_type_0] = []
        #             combs_data[comb_info][id_tuple[1]]['group_type'] = group_type_1
        #
        #         combs_data[comb_info][id_tuple[0]][group_type_1].append(time_within_dist)
        #         combs_data[comb_info][id_tuple[1]][group_type_0].append(time_within_dist)
        
    #process_combs(contact_data, contact_data_config, combs_data, id_2_group_id)
            
def process_combs(contact_data, contact_data_config, combs_data, id_2_group_id):
    for comb_info in contact_data_config['combs']:
        comb_size, dist_threshold, time_threshold = comb_info
        for agent_id, agent_combs_data in combs_data[comb_info].items():
            agent_group_type = agent_combs_data['group_type']
            for group_type, group_data in agent_combs_data.items():
                if group_type == 'group_type':
                    continue
                combs = np.array(list(it.combinations(group_data, comb_size)))
                if agent_group_type == CUSTOMER_TYPE and group_type == CUSTOMER_TYPE:
                    interaction_type = CUSTOMER_CUSTOMER
                elif agent_group_type == STAFF_TYPE:
                    interaction_type = STAFF_CUSTOMER
                else:
                    interaction_type = CUSTOMER_STAFF
                
                contact_data[interaction_type]['combs'][comb_info]['combs_count'] += len(combs)
                contact_data[interaction_type]['combs'][comb_info]['count'] += np.sum(np.sum(combs, axis=1) >= time_threshold)
            
def is_close_contact(proximity_data, time_step, non_cum_thresholds, cum_thresholds, interaction_type):
    has_non_cum_close_contact = calc_if_has_non_cum_close_contact(proximity_data, time_step, non_cum_thresholds, interaction_type)
    cum_contact_time = time_step/60*np.sum(proximity_data[:,1] <= cum_thresholds.dist)
    if has_non_cum_close_contact and cum_contact_time > cum_thresholds.time:
        return CLOSE_CONTACT_BOTH, cum_contact_time
    if has_non_cum_close_contact:
        return CLOSE_CONTACT_NON_CUM, cum_contact_time
    if cum_contact_time > cum_thresholds.time:
        return CLOSE_CONTACT_CUM, cum_contact_time
    
    return NON_CLOSE_CONTACT, cum_contact_time
    
def calc_if_has_non_cum_close_contact(proximity_data, time_step, non_cum_thresholds, interaction_type):
    if non_cum_thresholds.dist is None and non_cum_thresholds.time is None: 
        return False
    
    time_steps = proximity_data[proximity_data[:,1] <= non_cum_thresholds.dist,0]
    split_indices = np.argwhere((np.abs(time_steps[1:]-time_steps[:-1] - time_step) < 1e-10) == False).flatten()
    time_periods = []
    ind_count = len(split_indices)
    for ii, ind in enumerate(split_indices):
        if ii == 0:
            time_periods.append(time_step/60*(ind+1))
        
        if ind_count > 1 and ind != split_indices[-1]:
            time_periods.append(time_step/60*(split_indices[ii+1] - ind))
        
        if ind == split_indices[-1]:   
            time_periods.append(time_step/60*(len(time_steps) - ind - 1))
    
    has_non_cum_close_contact = np.any(np.array(time_periods) > non_cum_thresholds.time)
    
    return has_non_cum_close_contact

# ================================================================================================
            
def get_agent_cum_contact_data(output_filename, config_filenames, comb_sizes, dist_thres):
    cur_infra_name = None
    for config_filename in config_filenames:
        print(f'Processing {config_filename.stem}...')
        experiment_config, infra_name, measure_name = get_config_file_content(config_filename)
        if cur_infra_name is None:
            cur_infra_name = infra_name
            cum_contact_data_all = defaultdict(dict)
        elif cur_infra_name != infra_name:
            save_file_name = get_save_file_name(output_filename, cur_infra_name)
            np.savez(save_file_name, **cum_contact_data_all)
            load_file_name = get_save_file_name(output_filename, infra_name)
            if load_file_name.is_file():
                data_from_file = np.load(load_file_name, allow_pickle=True)
                cum_contact_data_all = {}
                for key, val in data_from_file.items(): 
                    cum_contact_data_all[key] = val.item()
            else:
                cum_contact_data_all = defaultdict(dict)
            cur_infra_name = infra_name
                
        experiment_state, experiment_state_filename = get_experiment_state(experiment_config)
        scenario_filenames = get_scenario_filenames(experiment_state_filename.parent.parent, experiment_state['successful_replications'])
        cum_contact_data = get_agent_cum_contact_data_from_scen_file(scenario_filenames, comb_sizes, dist_thres)
        cum_contact_data_all[measure_name] = cum_contact_data    
    
    save_file_name = get_save_file_name(output_filename, cur_infra_name)
    np.savez(save_file_name, **cum_contact_data_all)
         
def get_save_file_name(output_filename, infra_name):
    return output_filename.parent.joinpath(f'{infra_name}_{output_filename.stem}')
            
def get_agent_cum_contact_data_from_scen_file(scenario_filenames, comb_sizes, dist_thres):
    file_count = len(scenario_filenames)
    cum_contact_data = {interaction_type:{comb_size: np.zeros(0, dtype=float) for comb_size in comb_sizes} for interaction_type in INTERACTION_TYPES}
    cum_contact_data['all'] = {comb_size: np.zeros(0, dtype=float) for comb_size in comb_sizes}
    for ii, scenario_filename in enumerate(scenario_filenames):
        print(f'Processing {ii+1} of {file_count}')
        try:
            peds, conn_data, time_step = get_ped_and_conn_data(scenario_filename)
        except:
            continue
        
        contact_times = get_contact_times(conn_data, time_step, dist_thres)
        for comb_size in comb_sizes:
            cum_contact_data_local = get_cum_contact_times_for_sample_size(contact_times, comb_size)
            for interaction_type in INTERACTION_TYPES:
                cum_contact_data[interaction_type][comb_size] = np.concatenate((cum_contact_data[interaction_type][comb_size], np.array(cum_contact_data_local[interaction_type])))
            cum_contact_data['all'][comb_size] = np.concatenate((cum_contact_data['all'][comb_size], np.array(cum_contact_data_local['all'])))
                            
    return cum_contact_data
          
def get_contact_times(conn_data, time_step, dist_thres):
    contact_times = {STAFF_TYPE:{}, CUSTOMER_TYPE:{}}    
    id_2_group_id = conn_data['ID2groupID']
    for id_tuple, proximity_data in conn_data['connData'].items():
        interaction_type, group_type_0, group_type_1 = get_interaction_type(id_tuple, id_2_group_id)    
        cum_contact_time = time_step/60*np.sum(proximity_data[:,1] <= dist_thres)
        if cum_contact_time > 0:
            if group_type_0 == STAFF_TYPE: 
                if id_tuple[0] not in contact_times[STAFF_TYPE]:
                    contact_times[STAFF_TYPE][id_tuple[0]] = []
                contact_times[STAFF_TYPE][id_tuple[0]].append(cum_contact_time)
                if id_tuple[1] not in contact_times[CUSTOMER_TYPE]:
                    contact_times[CUSTOMER_TYPE][id_tuple[1]] = {STAFF_TYPE:[], CUSTOMER_TYPE:[]}
                contact_times[CUSTOMER_TYPE][id_tuple[1]][STAFF_TYPE].append(cum_contact_time)
            else:
                if id_tuple[0] not in contact_times[CUSTOMER_TYPE]:
                    contact_times[CUSTOMER_TYPE][id_tuple[0]] = {STAFF_TYPE:[], CUSTOMER_TYPE:[]}
                
                contact_times[CUSTOMER_TYPE][id_tuple[0]][group_type_1].append(cum_contact_time)
                if group_type_1 == STAFF_TYPE:
                    if id_tuple[1] not in contact_times[STAFF_TYPE]:
                        contact_times[STAFF_TYPE][id_tuple[1]] = []
                    contact_times[STAFF_TYPE][id_tuple[1]].append(cum_contact_time)
            
    return contact_times

def get_cum_contact_times_for_sample_size(contact_times, comb_size):
    cum_contact_data_local = {STAFF_CUSTOMER:[], CUSTOMER_CUSTOMER:[], CUSTOMER_STAFF:[]}
    for contact_time_list in contact_times[STAFF_TYPE].values():
        cum_contact_times = calc_cum_contact_times_for_sample_size(contact_time_list, comb_size)
        cum_contact_data_local[STAFF_CUSTOMER] += cum_contact_times
 
    for contact_times_per_group in contact_times[CUSTOMER_TYPE].values():
        for other_group_type, contact_time_list in contact_times_per_group.items():
            if len(contact_time_list) == 0:
                continue
            cum_contact_times = calc_cum_contact_times_for_sample_size(contact_time_list, comb_size)
            if other_group_type == STAFF_TYPE:
                cum_contact_data_local[CUSTOMER_STAFF] += cum_contact_times
            else:
                cum_contact_data_local[CUSTOMER_CUSTOMER] += cum_contact_times
                
    cum_contact_data_local['all'] = []
    for contact_times_per_group in contact_times[CUSTOMER_TYPE].values():
        contact_time_list = contact_times_per_group[STAFF_TYPE] + contact_times_per_group[CUSTOMER_TYPE]
        cum_contact_times = calc_cum_contact_times_for_sample_size(contact_time_list, comb_size)
        cum_contact_data_local['all'] += cum_contact_times    
    
    return cum_contact_data_local

def calc_cum_contact_times_for_sample_size(contact_time_list, comb_size):
    combs = np.array(list(it.combinations(contact_time_list, comb_size)))
    if len(combs) == 0:
        comb_size -= 1
        while comb_size > 1:
            combs = np.array(list(it.combinations(contact_time_list, comb_size)))
            comb_size -= 1
            if len(combs) > 0:
                break
    if combs.ndim == 1:
        return contact_time_list
    return np.sum(combs, axis=1).tolist()

# ===============================================================================================

def get_agent_cum_contact_count_data(output_filename, config_filenames, comb_sizes, dist_thres, time_thres):
    cum_contact_data_all = {}
    
    for config_filename in config_filenames:
        print(f'Processing {config_filename.stem}...')
        experiment_config, infra_name, measure_name = get_config_file_content(config_filename)
        if infra_name not in cum_contact_data_all:
            cum_contact_data_all[infra_name] = {measure_name:{}}
        if measure_name not in cum_contact_data_all[infra_name]:
            cum_contact_data_all[infra_name][measure_name] = {}
                
        experiment_state, experiment_state_filename = get_experiment_state(experiment_config)
        scenario_filenames = get_scenario_filenames(experiment_state_filename.parent.parent, experiment_state['successful_replications'])
        cum_contact_data = get_agent_cum_contact_counts_from_scen_file(scenario_filenames, comb_sizes, dist_thres, time_thres)        
        cum_contact_data_all[infra_name][measure_name] = cum_contact_data    
    
    np.savez(output_filename, **cum_contact_data_all)
                     
def get_agent_cum_contact_counts_from_scen_file(scenario_filenames, comb_sizes, dist_thres, time_thres):
    file_count = len(scenario_filenames)
    cum_contact_data = {interaction_type:{comb_size: np.zeros(0, dtype=float) for comb_size in comb_sizes} for interaction_type in INTERACTION_TYPES}
    cum_contact_data['all'] = {comb_size: np.zeros(0, dtype=float) for comb_size in comb_sizes}
    for ii, scenario_filename in enumerate(scenario_filenames):
        print(f'Processing {ii+1} of {file_count}')
        try:
            peds, conn_data, time_step = get_ped_and_conn_data(scenario_filename)
        except:
            continue
        
        contact_times = get_contact_times(conn_data, time_step, dist_thres)
        for comb_size in comb_sizes:
            cum_contact_data_local = get_cum_contact_counts_for_sample_size(contact_times, comb_size, time_thres)
            for interaction_type in INTERACTION_TYPES:
                cum_contact_data[interaction_type][comb_size] = np.concatenate((cum_contact_data[interaction_type][comb_size], np.array(cum_contact_data_local[interaction_type])))
            cum_contact_data['all'][comb_size] = np.concatenate((cum_contact_data['all'][comb_size], np.array(cum_contact_data_local['all'])))
                            
    return cum_contact_data
                     
def get_cum_contact_counts_for_sample_size(contact_times, comb_size, time_thres):
    cum_contact_data_local = {STAFF_CUSTOMER:[], CUSTOMER_CUSTOMER:[], CUSTOMER_STAFF:[]}
    for contact_time_list in contact_times[STAFF_TYPE].values():
        cum_contact_count = calc_cum_contact_count_for_sample_size(contact_time_list, comb_size, time_thres)
        cum_contact_data_local[STAFF_CUSTOMER].append(cum_contact_count)
 
    for contact_times_per_group in contact_times[CUSTOMER_TYPE].values():
        for other_group_type, contact_time_list in contact_times_per_group.items():
            if len(contact_time_list) == 0:
                continue
            cum_contact_count = calc_cum_contact_count_for_sample_size(contact_time_list, comb_size, time_thres)
            if other_group_type == STAFF_TYPE:
                cum_contact_data_local[CUSTOMER_STAFF].append(cum_contact_count)
            else:
                cum_contact_data_local[CUSTOMER_CUSTOMER].append(cum_contact_count)
                
    cum_contact_data_local['all'] = []
    cum_contact_data_local['all_counts'] = {STAFF_TYPE:0, CUSTOMER_TYPE:0, CUSTOMER_STAFF:0}
    for contact_times_per_group in contact_times[CUSTOMER_TYPE].values():
        contact_time_list = contact_times_per_group[STAFF_TYPE] + contact_times_per_group[CUSTOMER_TYPE]
        type_indices = [STAFF_TYPE_NR]*len(contact_times_per_group[STAFF_TYPE]) + [CUSTOMER_TYPE_NR]*len(contact_times_per_group[CUSTOMER_TYPE])
        cum_contact_count, type_counts = calc_cum_contact_count_for_sample_size(contact_time_list, comb_size, time_thres, type_indices=type_indices)
        cum_contact_data_local['all'].append(cum_contact_count)  
        for key in type_counts:
            cum_contact_data_local['all_counts'][key] += type_counts[key]
    
    return cum_contact_data_local

def calc_cum_contact_count_for_sample_size(contact_time_list, comb_size, time_thres, type_indices=None):
    def get_return_tuple():
        if type_counts is not None:
            return (close_contact_count, type_counts)
        else:
            return close_contact_count
    
    if type_indices is not None:
        type_counts = {STAFF_TYPE:0, CUSTOMER_TYPE:0, CUSTOMER_STAFF:0}
        type_indices = np.array(type_indices)
    else:
        type_counts = None
    
    contact_time_array = np.array(contact_time_list)
    base_ind = np.where(contact_time_array >= time_thres)[0]
    close_contact_count = len(base_ind)
    if comb_size == 1:
        return get_return_tuple()
    
    type_counts = update_type_counts(type_counts, type_indices, base_ind)
    indices_2_filter_out = base_ind.tolist()
    
    for other_comb_size in range(2, comb_size + 1):
        comb_indices = np.array(list(it.combinations(range(len(contact_time_array)), other_comb_size)))
        combs = contact_time_array[comb_indices]
        type_combs = type_indices[comb_indices]
        if len(combs) == 0:
            return get_return_tuple()   
        
        indices_2_ignore = np.zeros(len(combs), dtype=bool)
        for ind_2_filter_out in indices_2_filter_out:
            if isinstance(ind_2_filter_out, list):
                has_ind_sum = np.zeros(len(combs))
                for ind in ind_2_filter_out:
                    has_ind_sum += np.sum(comb_indices == ind, axis=1)                
                indices_2_ignore = indices_2_ignore | (has_ind_sum == len(ind_2_filter_out))
            else:
                indices_2_ignore = indices_2_ignore | np.sum(comb_indices == ind_2_filter_out, axis=1) == 1
        
        combs = combs[~indices_2_ignore,:]
        
        if len(combs) == 0:
            return get_return_tuple()  
        
        comb_indices = comb_indices[~indices_2_ignore,:]
        close_contact_ind = np.where(np.sum(combs, axis=1) >= time_thres)[0]
        for ind in close_contact_ind:
            indices_2_filter_out.append(comb_indices[ind,:].tolist())
            
        close_contact_count += len(close_contact_ind)/len(combs)
        
        type_combs = type_combs[~indices_2_ignore,:]
        type_counts = update_type_counts(type_counts, type_combs, close_contact_ind)   
   
    return get_return_tuple()

def update_type_counts(type_counts, type_combs, close_contact_ind):
    if type_counts is None:
        return
    
    if len(close_contact_ind) == 0:
        return type_counts
    
    if type_combs.ndim == 1:
        type_counts[STAFF_TYPE] += np.sum(type_combs[close_contact_ind] == STAFF_TYPE_NR)
        type_counts[CUSTOMER_TYPE] += np.sum(type_combs[close_contact_ind] == CUSTOMER_TYPE_NR)
        return type_counts

    type_sums = np.sum(type_combs[close_contact_ind,:], axis=1)
    type_counts[STAFF_TYPE] += np.sum(type_sums == 0)
    type_counts[CUSTOMER_TYPE] += np.sum(type_sums == type_combs.shape[1])
    type_counts[CUSTOMER_STAFF] += np.sum((type_sums > 0) & (type_sums < type_combs.shape[1]))
            
    return type_counts

# ================================================================================================
def process_cum_contact_file(input_filename, output_filename, time_thres):
    data_in_file = np.load(input_filename, allow_pickle=True)
    thres_probs = {}
    for infra_name, infra_data in data_in_file.items():
        print(infra_name)
        thres_probs[infra_name] = {}
        infra_data = infra_data.item()
        for measure_name, measure_data in infra_data.items():
            print(measure_name)
            thres_probs[infra_name][measure_name] = {}
            for interaction_type, intact_data in measure_data.items():
                print(interaction_type)
                thres_probs[infra_name][measure_name][interaction_type] = {}
                for comb_size, contact_time_array in intact_data.items():
                    print(comb_size)
                    prob = np.sum(contact_time_array >= time_thres)/len(contact_time_array)
                    thres_probs[infra_name][measure_name][interaction_type][comb_size] = prob
    
    np.savez(output_filename, **thres_probs)

def process_cum_contact_files(infra_name_2_file, output_filename, time_thres):
    
    thres_probs = {}
    for infra_name, input_filename in infra_name_2_file.items():
        infra_data = np.load(input_filename, allow_pickle=True)
        print(infra_name)
        thres_probs[infra_name] = {}
        for measure_name, measure_data in infra_data.items():
            measure_data = measure_data.item()
            print(measure_name)
            thres_probs[infra_name][measure_name] = {}
            for interaction_type, intact_data in measure_data.items():
                print(interaction_type)
                thres_probs[infra_name][measure_name][interaction_type] = {}
                for comb_size, contact_time_array in intact_data.items():
                    print(comb_size)
                    prob = np.sum(contact_time_array >= time_thres)/len(contact_time_array)
                    thres_probs[infra_name][measure_name][interaction_type][comb_size] = prob
    
    np.savez(output_filename, **thres_probs)


# ================================================================================================

def get_scenario_filenames(pd, seeds):
    if isinstance(seeds[0], str):
        seeds = [int(seed) for seed in seeds]
        
    scenario_filenames = []
    for entry in pd.iterdir():
        if not entry.is_file() or entry.suffix != '.scen':
            continue
        
        name_parts = entry.stem.split('_')
        seed = int(name_parts[-1].replace('.scen', ''))
        if seed in seeds:
            scenario_filenames.append(entry)
            
    return scenario_filenames

def get_config_file_content(config_filename):
    with open(config_filename, 'r') as f:
        experiment_config = json.load(f) 

    if '.json' in config_filename.stem:
        scen_name = config_filename.stem.replace('_scenario_experiment.config.json', '')
    else:
        scen_name = config_filename.stem.replace('_scenario_experiment.config', '')
    name_parts = scen_name.split('_')
    infra_name = '_'.join(name_parts[:2])
    measure_name = '_'.join(name_parts[2:])

    return experiment_config, infra_name, measure_name

def get_experiment_state(experiment_config):
    experiment_state_filename = Path(experiment_config['main_dir']).joinpath(experiment_config['experiment_state_filename'])
    with open(experiment_state_filename, 'r') as f:
        experiment_state = json.load(f) 
        
    return experiment_state, experiment_state_filename

def get_ped_and_conn_data(scenario_filename):
    scen_data =  BasicFileOutputManager.readScenarioDataFile(scenario_filename)
    desc_filename = scenario_filename.parent.joinpath(scen_data.descFile)
    conn_filename = scenario_filename.parent.joinpath(scen_data.connFile)
    peds = BasicFileOutputManager.readDescriptionDataFile(desc_filename)
    
    if conn_filename.suffix != '.npz':
        conn_filename = conn_filename.parent.joinpath(f'{conn_filename.name}.npz')
    
    conn_data = ConnectivityFileOutputManager.readConnectionsDataFile(conn_filename)
    
    return peds, conn_data, scen_data.timeStep

def get_interaction_type(id_tuple, id_2_group_id):
    group_type_0 = STAFF_TYPE if id_2_group_id[id_tuple[0]] == STAFF_TYPE else CUSTOMER_TYPE
    group_type_1 = STAFF_TYPE if id_2_group_id[id_tuple[1]] == STAFF_TYPE else CUSTOMER_TYPE
    
    if group_type_0 == STAFF_TYPE or group_type_1 == STAFF_TYPE:
        return STAFF_CUSTOMER, group_type_0, group_type_1
    else:
        return CUSTOMER_CUSTOMER, group_type_0, group_type_1

