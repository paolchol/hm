#%% Description

'''
SFR_SA
Script to run multiple instances of a MODFLOW model simulating a lowland spring using the SFR package

- Modifies the .sfr file
- Runs the model
- Saves the outflow of a specified cell
- Export the result as an .xlsx file
'''

#%% Setup

# Import necessary packages
import datetime
import flopy
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

def load_streamflow_dat(f, nsp = 1):
    """
    Load the streamflow.dat file generated as output by MODFLOW

    f: str
        path to the streamflow.dat file
    nsp: int, optional
        number of stress periods of the simulation. 1 works also with stationary models.
        Default is 1
    
    Returns:
    df: pandas.DataFrame
        dataframe containing the information stored inside streamflow.dat file
    """
    if nsp == 1:
        df = pd.DataFrame()
        with open(f, 'r') as file:
            for row in file.readlines()[8:]:
                r = list(filter(None, row.split(' ')))
                df = pd.concat([df, pd.DataFrame(r).transpose()])
            df.columns = ['l', 'r', 'c', 'iseg', 'ireach', 'flow_into_reach', 'flow_to_aquifer', 'flow_out_reach', 'overlnd_runoff',
                        'direct_precip', 'stream_et', 'stream_head', 'strem_depth', 'stream_width', 'stream_bed_cond', 'streambed_gradient']
            df.streambed_gradient = df.streambed_gradient.str.removesuffix('\n')
            df = change_type(df, ['l', 'r', 'c', 'iseg', 'ireach'], 'int') 
            df = change_type(df, ['flow_into_reach', 'flow_to_aquifer', 'flow_out_reach', 'overlnd_runoff',
                                    'direct_precip', 'stream_et', 'stream_head', 'strem_depth', 'stream_width',
                                    'stream_bed_cond', 'streambed_gradient'], 'float')
            df.reset_index(inplace = True, drop= True)
    else:
        for sp in range(nsp):
            pass
    return df

#%% Load SFR characteristics

# Define needed paths and model name
cwd = os.getcwd()
sfr_data = os.path.join(cwd, 'test_files', 'busca_sfr2_sfr_data.xlsx') # SFR characteristics
model_ws = os.path.join(cwd, 'test_files', 'sfr_model_test') # Model working directory
model_name = 'busca_sfr2'

# Load general parameters (item 1)
it1 = pd.read_excel(sfr_data, sheet_name = 'ITEM1')

# Load reach data (item 2)
reach_data = pd.read_excel(sfr_data, sheet_name = 'ITEM2')
reach_data = reach_data.apply(pd.to_numeric)
reach_data.columns = ['k', 'i', 'j', 'iseg', 'ireach', 'rchlen', 'strtop', 'slope',  'strthick',  'strhc1']
reach_data = reach_data.loc[:,:].to_records(index = False)
# flopy adds 1 to layer, row and column, so remove 1 here
reach_data.k = reach_data.k - 1
reach_data.i = reach_data.i - 1
reach_data.j = reach_data.j - 1

# Load item 5
it5 = pd.read_excel(sfr_data, sheet_name = 'ITEM5')

# Load segment data (item 6a)
segment_data = pd.read_excel(sfr_data, sheet_name = 'ITEM6abc')
segment_data.columns = [x.lower() for x in segment_data.columns]
segment_data = segment_data.loc[:,:].to_records(index = False)
segment_data = {0: segment_data}

# Generate the SFR package through flopy
unit_number = 27 # define this based on the model

m = flopy.modflow.Modflow(model_name, model_ws = model_ws)
sfr = flopy.modflow.ModflowSfr2(
    m,
    nstrm = it1.NSTRM.values[0],              # number of reaches
    nss = it1.NSS.values[0],                  # number of segments
    const = it1.CONST.values[0],              # constant for manning's equation: 1 for m/s
    dleak = it1.DLEAK.values[0],              # closure tolerance for stream stage computation
    ipakcb = it1.ISTCB1.values[0],            # flag for writing SFR output to cell-by-cell budget (on unit 50)
    istcb2 = it1.ISTCB2.values[0],            # flag for writing SFR output to text file
    dataset_5 = {0: it5.values[0].tolist()},
    unit_number = unit_number,
    isfropt = it1.ISFROPT.values[0],
    segment_data = segment_data,
    reach_data=reach_data
)

#%% Define loop parameters 

# Define the type of SFR structure
# 2SEG
# nSEG
sfr_type = '2SEG'

# Define limits of hydraulic conductivity range and number of iterations
ki = 0.01
kf = 0.000001
n = 10 #10 to test the code, then switch to 100
step = np.linspace(ki,kf,n)
inputs = []
outputs = []

# Define reach and segment from where to get the reach flow
reach = 72
segment = 1

#%% Loop

'''
START OF LOOP
'''
start = datetime.datetime.now()
for i in range(n):
    # Change hydraulic conductivity and assign to sfr package
    k = step[i]
    reach_data.strhc1 = k
    sfr.reach_data = reach_data
    # Write the new .sfr file
    sfr.write_file()
    # Run the model
    success, buff = flopy.mbase.run_model(
                exe_name = os.path.join(model_ws, 'MF2005.exe'),
                namefile = f'{model_name}.nam',
                model_ws = model_ws,
                silent = False #False to test the code, then switch to True
                )
    if not success:
        raise Exception("MODFLOW did not terminate normally.")
    # Load the streamflow.dat file and extract the searched flow
    f = os.path.join(model_ws, f'{model_name}_streamflow.dat')
    df = load_streamflow_dat(f)
    flow = df.loc[(df.ireach == reach) & (df.iseg == segment), 'flow_out_reach'].values[0]
    # Append k and flow
    inputs.append(k)
    outputs.append(flow)
end = datetime.datetime.now()

print('Runs terminated')
print('Number of runs: ', n)
print('Elapsed time (s): ', f'{(end-start).seconds}.{round((end-start).microseconds*(10**-6),2)}')

#%% Save to dataframe and export

df_results = pd.DataFrame({'k_value': inputs, 'sfr_results': outputs})
df_results.to_excel(os.path.join(model_ws, 'sfr_results.xlsx'))