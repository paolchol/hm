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

# Define limits of hydraulic conductivity range and number of iterations
ki = 0.01
kf = 0.000001
n = 10 #10 to test the code, then switch to 100
step = np.linspace(ki,kf,n)

k_dict = {
    'kt': [0.0001, 0.0003], 
    'ka': [0.0003, 0.0005]
}

# Define cells until the flow value is extracted from drain[0]
row_limit = 93
column_limit = 76

#%% # Loop
# Loop to change conductance in the drain package, run the model and extract cbb results

'''
START OF LOOP
'''
start = datetime.datetime.now()
outputs = []
for i in range(n):
    k = step[i]
    # Change conductance
    conductance_change = (k*drn_sp.Width*drn_sp.Length)/drn_sp.Thickness
    drn.conductance = round(conductance_change, 5)
    stress_period_data = {0: drn.iloc[:, :-1].to_numpy().tolist()} #remove node column, not needed nor supported by flopy's ModflowDrn class

    # Modify the drn package' stress period data
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

    # Append the parameters to a list
    outputs.append([k, sum_drain, sum_drain_tot])
end = datetime.datetime.now()

print('Runs terminated')
print('Number of runs: ', n)
print('Elapsed time (s): ', f'{(end-start).seconds}.{round((end-start).microseconds*(10**-6),2)}')

#%% Save to dataframe and export

df_results = pd.DataFrame(outputs, columns = ['k', 'drn_until_rc', 'drn_total'])
df_results.to_excel(os.path.join(model_ws, 'drain_results.xlsx'))