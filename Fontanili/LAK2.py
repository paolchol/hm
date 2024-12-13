'''
LAK - OPTION 2
Script to run instances of a MODFLOW model simulating the head of a lowland spring using the LAK package

- Modifies the hydraulic contuctivity of .lak file
- Runs the model
- Exports the result as an .xlsx file

Important: The SFR package has to be removed from the model and generated externally to avoid issues while loading it

'''
#%% Setup
import os
import datetime
import numpy as np
import pandas as pd
import flopy
import shutil

cwd = "d:\Claudia\MAURICE" # Local directory
model_ws = os.path.join(cwd,"LAKE-2")  # Folder containing the model
lak_file = os.path.join(model_ws, "busca_base_infittito_apr24_sfr_icalc2_lake.lak")
sfr_data = os.path.join(model_ws, 'busca_sfr_data_icalc2_lake.xlsx') # SFR characteristics
model_name = "busca_base_infittito_apr24_sfr_icalc2_lake"

# Backup the original LAK file
backup_lak_file = lak_file + ".bak"
if not os.path.exists(backup_lak_file):
    shutil.copy(lak_file, backup_lak_file)

# Define needed functions
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
                        'direct_precip', 'stream_et', 'stream_head', 'stream_depth', 'stream_width', 'stream_bed_cond', 'streambed_gradient']
            df.streambed_gradient = df.streambed_gradient.str.removesuffix('\n')
            df = change_type(df, ['l', 'r', 'c', 'iseg', 'ireach'], 'int') 
            df = change_type(df, ['flow_into_reach', 'flow_to_aquifer', 'flow_out_reach', 'overlnd_runoff',
                                    'direct_precip', 'stream_et', 'stream_head', 'stream_depth', 'stream_width',
                                    'stream_bed_cond', 'streambed_gradient'], 'float')
            df.reset_index(inplace = True, drop= True)
    else:
        for sp in range(nsp):
            pass
    return df
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

# Load the Existing Model
mf = flopy.modflow.Modflow.load(
    f"{model_name}.nam", 
    model_ws=model_ws, 
    check=False, 
    verbose=True
)

#%% Generate SFR with flopy
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

# Load segment data (item 6abc)
segment_data = pd.read_excel(sfr_data, sheet_name = 'ITEM6abc')
segment_data.columns = [x.lower() for x in segment_data.columns]
segment_data = segment_data.loc[:,:].to_records(index = False)
#segment_data = {0: segment_data}

# Load channel geometry data (item 6d)
item6d = pd.read_excel(sfr_data, sheet_name='ITEM6d')  # Geometry data

# Initialize the channel_geometry_data dictionary
channel_geometry_data = {}

# Group by segment
for segment, group in item6d.groupby("segment"):
    # Extract x and z values for the segment
    x_values = group[group["value"] == "x"].iloc[:, 2:].values.flatten()
    z_values = group[group["value"] == "z"].iloc[:, 2:].values.flatten()
    
    # Create a recarray for this segment
    geometry = np.rec.array(list(zip(x_values, z_values)), dtype=[("x", "f4"), ("z", "f4")])
    
    # Store the segment data under the first index (assuming single stress period `i=0`)
    if 0 not in channel_geometry_data:
        channel_geometry_data[0] = {}
    channel_geometry_data[0][segment] = geometry

# Generate the SFR package through flopy
unit_number = 27 # define this based on the model
sfr = flopy.modflow.ModflowSfr2(
    mf,
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
    reach_data=reach_data,
    channel_geometry_data=channel_geometry_data
) 

#  Write the updated model input files
mf.write_input()
# # Write all model input files, excluding unsupported 'check' argument
# for p in mf.packagelist:
#     if hasattr(p, 'write_file'):
#         p.write_file()

#%% Define loop parameters 
ki = 0.01
kf = 0.000001
n = 10
hydraulic_conductivities = np.linspace(ki, kf, n)  # Define range of K values

# Store results
inputs = []
flows = []
depths = []

# Define reach and segment from where to get the reach flow
reach = 63
segment = 1
#%% LOOP
'''
LOOP
'''
# Open the LAK file outside of the loop 
with open(backup_lak_file, "r") as file:
        lines = file.readlines()

start = datetime.datetime.now()

for k_value in hydraulic_conductivities:
    # Modify the K value in the second line of the LAK file
    lines[1] = f"     0.000        10 {k_value:.3e}\n"  # Update K value

    # Write the updated LAK file
    with open(lak_file, "w") as file:
        file.writelines(lines)

    # Load and run the model
    mf = flopy.modflow.Modflow.load(f"{model_name}.nam", model_ws=model_ws, check=False)
    success, buff = mf.run_model(silent=True)
    if not success:
        print(f"Model run failed for K = {k_value:.3e}")
        inputs.append(k_value)
        flows.append(None)
        depths.append(None)
        continue

    # Load the streamflow.dat file and extract the searched flow
    f = os.path.join(model_ws, f'{model_name}_streamflow.dat')
    df = load_streamflow_dat(f)
    flow = df.loc[(df.ireach == reach) & (df.iseg == segment), 'flow_out_reach'].values[0]
    depth = df.loc[(df.ireach == reach) & (df.iseg == segment), 'stream_depth'].values[0]
    # Append k and flow
    inputs.append(k_value)
    flows.append(flow)
    depths.append(depth)

end = datetime.datetime.now()

print('Runs terminated')
print('Number of runs: ', n)
print('Elapsed time (s): ', f'{(end-start).seconds}.{round((end-start).microseconds*(10**-6),2)}')

#%% Save to dataframe and export
df_results = pd.DataFrame({'k_value': inputs, 'flow': flows, 'depth':depths})
df_results.to_excel(os.path.join(model_ws, 'lake_results.xlsx'))

# %%
