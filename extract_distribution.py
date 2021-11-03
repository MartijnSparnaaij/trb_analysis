import numpy as np
import pandas as pd



VISIT_PER_TABLE_COLUMS = ['visit_id', 'table_id', 'start_time', 'end_time', 'group_size', 'fully_captured']

def extract_activity_times(df):
    return (df['Departure time'] - df['Arrival time'])/np.timedelta64(1, 's')

def extract_visits_per_table(df_demand_raw):
    df_demand_raw['Captured_since_arrival'] = df_demand_raw['Captured_since_arrival'].astype(bool)
    df_demand_filtered = df_demand_raw[df_demand_raw['Assigned table ID'].notnull()]
    
    visit_id = 0
    cur_visits = {}
    visits = []
        
    def add_visit(table_id, pop=True):
        visits.append([cur_visits[table_id][col_nm] for col_nm in VISIT_PER_TABLE_COLUMS])
        if pop:
            cur_visits.pop(table_id)    
        
    for row_ind in range(df_demand_filtered.shape[0]):
        table_id = df_demand_filtered['Assigned table ID'][row_ind]
        guest_count = df_demand_filtered['No. of guests at the table at the moment'][row_ind]
        start_time = df_demand_filtered['Arrival time'][row_ind]
        end_time = df_demand_filtered['Departure time'][row_ind]
        captured_since_arrival = df_demand_filtered['Captured_since_arrival'][row_ind]

        
        
        if table_id in cur_visits:
            cur_visits[table_id]['guest_count'] += guest_count
            if cur_visits[table_id]['guest_count'] > cur_visits[table_id]['group_size']:
                cur_visits[table_id]['group_size'] = cur_visits[table_id]['guest_count']
                
            if cur_visits[table_id]['guest_count'] == 0:
                cur_visits[table_id]['end_time'] = end_time
                add_visit(table_id)                
                
        else:
            cur_visits[table_id] = {'visit_id': visit_id, 'table_id': table_id, 'start_time': start_time, 'end_time': end_time, 'group_size': guest_count, 'fully_captured': captured_since_arrival, 'guest_count': guest_count}
            visit_id += 1
            if pd.notnull(end_time):                
                cur_visits[table_id]['group_size'] = -guest_count
                add_visit(table_id)
                
    for table_id, visit in cur_visits.items():
        visit['end_time'] = np.nan
        add_visit(table_id, pop=False)
           
    df_demand = pd.DataFrame(visits, columns=VISIT_PER_TABLE_COLUMS)
    df_demand['visit_time'] = df_demand.apply(lambda row: (row['end_time'] - row['start_time'])/np.timedelta64(1, 's') if row['fully_captured'] and pd.notnull(row['end_time']) else np.nan, axis=1)
    return df_demand
     
        
        
        
        
        
        
        
    
    