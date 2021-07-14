'''
Created on 13 Jul 2021

@author: Martijn Sparnaaij
'''
import json
from pathlib import Path

import numpy as np
import NOMAD


MAX_REPLICATIONS = 5000
INIT_REPLICATIONS = 30
CONSECUTIVE_STEPS = 5
P_THRESHOLD = 0.95
CONVERGENCE_CONFIG = {'steps':CONSECUTIVE_STEPS, 'p_threshold':P_THRESHOLD}


class ExperimentRunner():
    
    def __init__(self, scenario_filename, lrcm_filename, convergence_config=CONVERGENCE_CONFIG, max_replications=MAX_REPLICATIONS, init_replications=INIT_REPLICATIONS):
        self.scenario_filename = Path(scenario_filename)
        self.lrcm_filename = lrcm_filename
        self.convergence_config = convergence_config
        self.max_replications = max_replications
        self.init_replications = init_replications
        
        self.experiment_filename = self.scenario_filename.parent.joinpath(f'{self.scenario_filename.stem}_experiment.json')
        self.combined_stats_filename = self.scenario_filename.parent.joinpath(f'{self.scenario_filename.stem}_combined.stats')
            
        rng = np.random.default_rng()
        self.seeds = rng.choice(np.arange(1,max_replications*2), max_replications, False).tolist()
            
    def _init_experiment_file(self):
        self.experiment_info = {
            'scenario_filename': self.scenario_filename,
            'lrcm_filename': self.lrcm_filename,
            'combined_stats_filename': self.combined_stats_filename,
            'seeds': self.seeds,
            'max_replications': self.max_replications,
            'init_replications': self.init_replications,    
            'convergence_config': self.convergence_config,
            'successful_replication_count': 0,
            'failed_replication_count': 0,
            'successful_replications': [], # (seed, .scen filename, .stats filename)
            'failed_replications': [], # (seed, .scen filename, .stats filename)
            'convergence_stats': [] # {replications:#rep, cvm_stats:{cdf_1:[pvalues]}}
            }
    
        with open(self.experiment_filename, 'x') as f:
            json.dump(self.experiment_info, f)  

    def _run_experiment(self):
        replication_nr = 0
        not_converged = True
        last_convergence_check = 24
        while replication_nr < self.init_replications and not_converged:
            NOMAD.NOMAD_RNG = np.random.default_rng(self.seeds[replication_nr])
            from NOMAD import NOMAD_RNG
            print(NOMAD_RNG.integers(0, 10, 5))
            print(np.random.default_rng(self.seeds[replication_nr]).integers(0, 10, 5))
            # Process output
            if replication_nr == last_convergence_check + self.convergence_config['steps']:
                # Check convergence
                last_convergence_check = replication_nr
                not_converged = False
                        
            replication_nr += 1
            
        
        pass
        # Reset nomad range
        # Create model
        # Run model
    

    def _process_data(self):
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
        
        


