#%% Description

'''
DRN_SA
Script to run multiple instances of a MODFLOW model simulating a lowland spring using the DRN package

- Modifies the .drn file
- Runs the model
- Saves the outflow in the first cells, until the one where a discharge measure is available
- Export the result as an .xlsx file
'''

#%% Setup
# Import all necessary packages
import datetime
import flopy
import flopy.utils.binaryfile as bf
import numpy as np
import os
import pandas as pd
import warnings

# Define working directory 
cwd = os.getcwd()   #where test files are stored (github)
cwd_model = os.getcwd()   #where model files are stored (local)

# Function to load .drn file directly
def import_input_file(path):
    f = open(path).readlines()
    df = pd.DataFrame()
    for row in f[4:]:
        r = list(filter(None, row.split(' ')))
        df = pd.concat([df, pd.DataFrame(r).transpose()])
    df.columns = ['layer', 'row', 'column', 'stage', 'conductance', 'node']
    df.node = df.node.str.removesuffix('\n')
    df.layer = df.layer.astype('int')
    df.row = df.row.astype('int')
    df.column = df.column.astype('int')
    df.stage = df.stage.astype('float')
    df.conductance = df.conductance.astype('float')
    df.reset_index(inplace=True, drop = True)
    return df

# Load original .drn file, load DRAIN cells characteristics and
# transform the drn structure in flopy (required format for "stress_period_data")
drn = import_input_file(os.path.join(cwd, 'busca_drain_start.drn'))
drn_sp = pd.read_csv(os.path.join(cwd,'busca_drain_specifiche_celle_v2.csv'))

print(f"Numero di celle in drn: {len(drn)}")
print(f"Numero di celle in drn_sp: {len(drn_sp)}")

# Trova la lunghezza minima tra drn e drn_sp
min_length = min(len(drn), len(drn_sp))

drn.layer = 0
#%%

# Create and load the model class (only useful because it's needed by ModflowDrn class)
# this can be replaced by loading the actual model .nam file (not needed though)
modelpth = os.path.join(cwd_model) #path to model
# mf = flopy.modflow.Modflow(
#     "busca_drain",
#     model_ws = os.path.join(cwd_model),
#     exe_name = "mf2005",
# )
mf = flopy.modflow.Modflow.load(
    os.path.join(cwd_model,'busca_drain_fondo_2_punti.nam'),
    model_ws = os.path.join(cwd_model),
    exe_name='MF2005',
    version = 'mf2005',
    verbose = False
    )
# Define limits of hydraulic conductivity range and number of iterations
ki = 0.00007
kf = 0.00006
n = 10 #10 to test the code, then switch to 100

step = np.linspace(ki,kf,n)
inputs = []
outputs = []

ipakcb = 50 #code for cell-by-cell flow data storage

# define cells untill the value extracted from drain[0]
row_limit = 93
column_limit = 76

#%% Loop
# Loop to change conductance in the drain package, run the model and extract cbb results

'''
START OF LOOP
'''
start = datetime.datetime.now()
for i in range(n):
    k = step[i]
    # Change conductance
    conductance_change = (k*drn_sp.Width*drn_sp.Length)/drn_sp.Thickness
    drn.conductance = round(conductance_change,5)
    stress_period_data = {0: drn.iloc[:, :-1].to_numpy().tolist()} #remove node column, not needed nor supported by flopy's ModflowDrn class

    # Generate the drn package inside the flopy class
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        drain = flopy.modflow.ModflowDrn(mf, ipakcb=ipakcb, stress_period_data=stress_period_data,
                                    filenames=os.path.join(cwd_model, 'busca_drain_fondo_2_punti.drn'))
    drain.write_file(check = False)
    
    # Run the model
    success, buff = mf.run_model(silent=True) #False to test the code, then switch to True
    if not success:
        raise Exception("MODFLOW did not terminate normally.")
    
    # Get drain values from cbb output
    cbb = bf.CellBudgetFile(os.path.join(cwd_model, 'busca_drain_fondo_2_punti.cbb'))
    drain = cbb.get_data(text = 'DRAINS')

    #sum otflow from selected cells
    sum_drain = np.sum(drain[0][0, :row_limit+1, :column_limit+1])

    # Append the input and output to lists
    inputs.append(k)
    outputs.append(sum_drain)
end = datetime.datetime.now()

print('Runs terminated')
print('Number of runs: ', n)
print('Elapsed time (s): ', (start-end).seconds)

#%% Save to dataframe and export

df_results = pd.DataFrame({'k_value': inputs, 'drain_results': outputs})
df_results.to_csv(os.path.join(cwd, 'results.csv'),index=False)