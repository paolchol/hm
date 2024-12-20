#%% # Description

'''
DRN_SA
Script to run multiple instances of a MODFLOW model simulating a lowland spring using the DRN package

- Modifies the .drn file
- Runs the model
- Saves the outflow in the first cells, until the one where a discharge measure is available
- Export the result as an .xlsx file
'''

#%% # Setup

# Import necessary packages
import datetime
import flopy
import flopy.utils.binaryfile as bf
import numpy as np
import os
import pandas as pd

# Define needed functions
def change_type(df, cols, t):
    """
    Change the dtype of selected columns to a given type

    df: pandas.DataFrame
        the dataframe
    cols: str, list of str
        the columns to be changed
    t: str
        the dtype wanted

    Returns:
    df: pandas.DataFrame
    """
    for col in cols:
        df[col] = df[col].astype(t)
    return df

def load_drn_file(path):
    """
    Load MODFLOW .drn input file

    path: str
        Path to the .drn file to be loaded
    
    Returns:
    df: pandas.DataFrame
        Dataframe containing the information stored in the original .drn file
    """
    f = open(path).readlines()
    df = pd.DataFrame()
    for row in f[4:]:
        r = list(filter(None, row.split(' ')))
        df = pd.concat([df, pd.DataFrame(r).transpose()])
    df.columns = ['layer', 'row', 'column', 'stage', 'conductance', 'node']
    df.node = df.node.str.removesuffix('\n')
    df.reset_index(inplace=True, drop = True)

    # assign the correct dtypes to the functions
    df = change_type(df, ['layer', 'row', 'column', 'node'], 'int')
    df = change_type(df, ['stage', 'conductance'], 'float')    
    return df

#%% # Load DRN characteristics

# Define needed paths and model name
cwd = os.getcwd()   #where test files are stored (github)
drn_data = os.path.join(cwd, 'test_files', 'busca_drain_specifiche_celle.csv') # DRN characteristics
model_ws = os.path.join(cwd, 'test_files', 'drain_model_test')
model_name = 'busca_drain'

# Load original .drn file, load DRAIN cells characteristics and
# transform the drn structure in flopy (required format for "stress_period_data")
drn = load_drn_file(os.path.join(model_ws, 'busca_drain_start.drn'))
drn_sp = pd.read_csv(drn_data)
# flopy adds 1 to layer, row and column, so subtract 1 to them
drn.layer = 0
drn.row = drn.row - 1
drn.column = drn.column - 1

# Generate the DRN package through flopy
# m = flopy.modflow.Modflow(model_name, model_ws=model_ws)
ipakcb = 50 #code for cell-by-cell flow data storage

mf = flopy.modflow.Modflow.load(
    os.path.join(model_ws, f'{model_name}.nam'),
    model_ws = model_ws,
    exe_name='MF2005',
    version = 'mf2005',
    verbose = False
)
mf.drn.ipakcb = ipakcb
mf.drn.filenames = os.path.join(model_ws, f'{model_name}.drn')

#%% # Define loop parameters

# Define the hydraulic conductivities to be tested
k_dict = {
    'kt': [0.0001, 0.0003], 
    'ka': [0.0003, 0.0005]
}

# Define cells until the flow value is extracted from drain[0]
row_limit = 93
column_limit = 76

# Define the cell until the drain cells are considered as the "head" of the fontanile
# to them, kt will be applied
r_t = 93
c_t = 76

#%% # Loop
# Loop to change conductance in the drain package, run the model and extract cbb results

'''
START OF LOOP
'''
start = datetime.datetime.now()
outputs = []
cond = (drn_sp.row <= r_t) & (drn_sp.column <= c_t)
m = 1
drain_outflow_save = pd.DataFrame()
for kt in k_dict['kt']:
    # Change conductance
    drn_sp.loc[cond, 'Conductanc'] = (kt*drn_sp.loc[cond, 'Width']*drn_sp.loc[cond, 'Length'])/drn_sp.loc[cond, 'Thickness']
    for ka in k_dict['ka']:
        drn_sp.loc[~cond, 'Conductanc'] = (ka*drn_sp.loc[~cond, 'Width']*drn_sp.loc[~cond, 'Length'])/drn_sp.loc[~cond, 'Thickness']

        drn.conductance = round(drn_sp.Conductanc, 5)
        # Modify the drn package' stress period data
        stress_period_data = {0: drn.iloc[:, :-1].to_numpy().tolist()} #remove node column, not needed nor supported by flopy's ModflowDrn class
        mf.drn.stress_period_data = stress_period_data
        mf.drn.write_file(check = False)

        success, buff = mf.run_model(silent=True) #False to test the code, then switch to True
        if not success:
            raise Exception("MODFLOW did not terminate normally.")
        
        # Get drain values from cbb output
        cbb = bf.CellBudgetFile(os.path.join(model_ws, f'{model_name}.cbb'))
        drn_cbb = cbb.get_data(text = 'DRAINS')

        # Sum the flux extracted from drain[0]
        sum_drain = np.sum(drn_cbb[0][0, :row_limit+1, :column_limit+1])
        sum_drain_tot = np.sum(drn_cbb[0][0, :,:])
        sum_drain_t = np.sum(drn_cbb[0][0, :r_t+1, :c_t+1])

        # Save the outflow in every cell
        drain_outflow = []
        for r,c in zip(drn.row, drn.column):
            drain_outflow.append(drn_cbb[0][0, r, c])
        drain_outflow_save = pd.concat([drain_outflow_save, pd.DataFrame(drain_outflow)], axis=1)

        # Append the parameters to a list
        outputs.append([f'M{m}', kt, ka, sum_drain, sum_drain_tot, sum_drain_t])
        m += 1
end = datetime.datetime.now()

print('Runs terminated')
print('Number of runs: ', len(k_dict['kt'])*len(k_dict['ka']))
print('Elapsed time (s): ',  f'{(end-start).seconds + round((end-start).microseconds*(10**-6),2)}')

#%% # Save to dataframe and export

drain_outflow_save.columns = [f'M{i}' for i in range(1,len(k_dict['kt'])*len(k_dict['ka'])+1)]
drain_outflow_save.insert(0, column = 'c', value = drn.column)
drain_outflow_save.insert(0, column = 'r', value = drn.row)
df_results = pd.DataFrame(outputs, columns = ['model', 'kt', 'ka', 'drn_until_rc', 'drn_total', 'drn_head'])
df_results.to_excel(os.path.join(model_ws, 'drain_results_multipar.xlsx'))
drain_outflow_save.to_csv(os.path.join(model_ws, 'drain_outflow_multipar.csv'))