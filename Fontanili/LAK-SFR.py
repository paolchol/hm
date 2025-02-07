'''
LAK - OPTION 2
Script to run instances of a MODFLOW model simulating the head of a lowland spring using the LAK package

- Writes the SFR package based on an excel file
- Modifies the hydraulic contuctivity of .lak file
- Runs the model
- Exports the result as an .xlsx file

Important:
Make sure these parameters are defined within the GWV model, if not, set them and create datasets again
- Check MODFLOW 2000 Packages: the Cell-by-Cell Flow unit numbers have to be different for every package 
  (LPF=50, LAK=60, SFR=55)
- Check MODFLOW Options > Resaturation: make sure that the "Resaturation Capacity is Active" box is checked.
  This allows us to use the "simple" formula for the calculation of the lakebed conductance.

'''
#%% Setup
import os
import datetime
import numpy as np
import pandas as pd
import flopy
import shutil
import math
import pickle

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

def run(i, j, tool, model_ws, model_name, params, params_save,
        flow_save, depth_save, flow_target, depth_target, columns, flowaq_save):
    # Write the new .sfr file transforming tool to reach_data
    reach_data = tool.loc[:,:].to_records(index = False)
    sfr.reach_data = reach_data
    sfr.write_file()
    # Run the model
    # m_code = f'M{i}'
    success, buff = flopy.mbase.run_model(
                exe_name = os.path.join(model_ws, 'MF2005.exe'),
                namefile = f'{model_name}.nam',
                model_ws = model_ws,
                silent = silent #False to test the code, then switch to True
                )
    if not success:
        print(params)
        raise Exception("MODFLOW did not terminate normally.")

    # Load the streamflow.dat file
    df = load_streamflow_dat(os.path.join(model_ws, f'{model_name}_streamflow.dat'))

    # Extract flow and depth in the target reach
    f = df.loc[(df.ireach == reach) & (df.iseg == segment), 'flow_out_reach'].values[0]
    fin = df.loc[(df.ireach == 1) & (df.iseg == 1), 'flow_into_reach'].values[0]
    d = df.loc[(df.ireach == reach) & (df.iseg == segment), 'stream_depth'].values[0]

    # Update the output structures
    params_save.append(params + [f,d,fin])
    # Extract flow and depth in all reaches and add them to the output structures
    flow_save = pd.concat([flow_save, df.flow_out_reach], axis=1)
    depth_save = pd.concat([depth_save, df.stream_depth], axis=1)
    flowaq_save = pd.concat([flowaq_save, df.flow_to_aquifer], axis=1)

    # Save the results after 100 runs
    if i % 100 == 0:

        save(i+1, j, columns, params_save, flow_target,
                depth_target, flow_save, depth_save, reach_data, model_ws, flowaq_save,
                save_100 = True)
        
        # Clear the output structures
        del flow_save, depth_save, flowaq_save

        flow_save, depth_save, flowaq_save = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        j += 100
    
    return params_save, flow_save, depth_save, flowaq_save, j

def save(i, j, columns, params_save, flow_target, depth_target,
         flow_save, depth_save, reach_data, model_ws, flowaq_save, save_100 = True):
    if not save_100:
        # Define column labels
        params_save = pd.DataFrame(params_save, columns = columns)
        # Add columns to params_save
        params_save['flow_diff'] = flow_target - params_save.flow_out_reach
        params_save['depth_diff'] = depth_target - params_save.stream_depth
        # Save as CSV
        params_save.to_csv(os.path.join(model_ws, 'run_output', f'sfr_results_M{i-1}.csv'), index = False)
    else:
        with open(os.path.join(model_ws, 'run_output', 'sfr_results_intermediate.pickle'), 'wb') as f:
            pickle.dump(params_save, f)
    # Set columns in flow_save and depth_save
    flow_save.columns, depth_save.columns, flowaq_save.columns = [f'M{x}' for x in range(j,i)], [f'M{x}' for x in range(j,i)], [f'M{x}' for x in range(j,i)]
    flow_save['ireach'], depth_save['ireach'], flowaq_save['ireach'] = reach_data.ireach, reach_data.ireach, reach_data.ireach
    flow_save['iseg'], depth_save['iseg'], flowaq_save['iseg'] = reach_data.iseg, reach_data.iseg, reach_data.iseg
    # Save as CSV
    flow_save.to_csv(os.path.join(model_ws, 'run_output', f'sfr_reach_flow_out_reach_M{i-1}.csv'), index = False)
    depth_save.to_csv(os.path.join(model_ws, 'run_output', f'sfr_reach_depth_M{i-1}.csv'), index = False)
    flowaq_save.to_csv(os.path.join(model_ws, 'run_output', f'sfr_reach_flow_to_aquifer_M{i-1}.csv'), index = False)

#%% Define paths

cwd = "d:\Claudia\MAURICE" # Local directory
model_ws = os.path.join(cwd,"LAKE_SFR")  # Folder containing the model
lak_file = os.path.join(model_ws, "busca_base_infittito_apr24_sfr_icalc2_lake.lak")
sfr_data = os.path.join(model_ws, 'busca_sfr_data_icalc2_lake.xlsx') # SFR characteristics
model_name = "busca_base_infittito_apr24_sfr_icalc2_lake"

#%% Generate SFR with flopy

# Create a model object just to write the SFR package
mf = flopy.modflow.Modflow(
    modelname = model_name, 
    model_ws=model_ws, 
    check=False, 
    verbose=False
)
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
segment_data = {0: segment_data}

# Load channel geometry data (item 6d)
item6d = pd.read_excel(sfr_data, sheet_name='ITEM6d')  # Geometry data
geom_data = {}
for seg in item6d.segment.unique():
    tool = item6d.loc[item6d.segment == seg, [f'v{i}' for i in range(1,9)]].to_numpy().copy()
    geom_data[int(seg)] = [tool[0].tolist(), tool[1].tolist()]
geom_data = {0: geom_data}

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
    channel_geometry_data=geom_data
) 

#  Write the updated SFR input file
sfr.write_file()

#%% Define LAKE loop parameters

# Load the model now that SFR has been written
mf = flopy.modflow.Modflow.load(
    f"{model_name}.nam", 
    model_ws=model_ws, 
    check=False)

# Access LAK package
lak = mf.lak
lake_mask = lak.lakarr.array   # Extract lake mask array from lakarr
lake_cells = np.argwhere(lake_mask[0] > 0)  # Find lake cells
delr = mf.dis.delr.array  # 1D array of column widths
delc = mf.dis.delc.array  # 1D array of row heights

# Define fixed parameters used to calculate the lakebed conductance
lakebed_thickness = 0.5  
cell_area = np.outer(delc, delr)  # Shape: (nrow, ncol)

#consider doing something like this
lakebed_ks = {
    'kb': [0.0001, 0.003]
}

#%%
# Define SFR loop parameters

# Define hydraulic conductivity parameter dictionary
# kt = t is "testa", the "head" of the fontanile
# ka = a is "asta", the channel of the fontanile
# 
# kt: a list containing the values to test
# ka: a list containing the values to test

k_dict = {
    'ka': [0.0003, 0.0005]
}

silent = False # True: the MODFLOW runs will not be printed in the terminal

n = len(lakebed_ks['kb'])*len(k_dict['ka'])

print('Based on the parameters set:\n')
print('- ' + str(n) + ' runs will be executed')
print(f'- Approximately {n*0.5} s will be needed ({n*0.5/(60*60*24)} days)')

#%%
# General loop parameters

# Define reach and segment from where to get the reach flow
reach = 63 # it is not the right reach
segment = 1

# Define the target values for flow and depth
flow_target = 0.019  # m3/s
depth_target = 0.20   # m

#%% LOOP
'''
LOOP
'''
# Modify lakebed conductance value based on different k values
start = datetime.datetime.now()

# Initialize needed variables
lakebed_conductance = np.zeros((mf.dis.nlay, mf.dis.nrow, mf.dis.ncol), dtype=np.float32)
params_save = []
flow_save, depth_save, flowaq_save = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

columns = ['m_code', 'kb','ka', 'flow_out_reach', 'stream_depth', 'flow_into_reach']

i, j = 1, 1
for kb in lakebed_ks['kb']:
    
    # Calculate conductance only for lake cells (layer, row, column)
    for lay, row, col in lake_cells:
        lakebed_conductance[lay, row, col] = (kb * cell_area[row, col]) / lakebed_thickness

    # Update bdlknc for each layer
    lak.bdlknc = flopy.utils.Transient3d(               # BDLKNC needs to be a transient3d object (same as lakarr)
        model=mf,
        shape=(mf.dis.nlay, mf.dis.nrow, mf.dis.ncol),  # Shape of the array
        dtype=np.float32,
        value=lakebed_conductance,  
        name="bdlknc"
        )
  
    #  Write the updated lak file (mf.write_input() returns the stress periods error)
    lak.write_file()

    for ka in k_dict['ka']:
        tool = pd.DataFrame(reach_data).copy()
        tool.loc[:, 'strhc1'] = ka
        

        params = [f'M{i}', kb, ka]
        params_save, flow_save, depth_save, flowaq_save, j = run(i, j, tool, model_ws, model_name,
                                                                params, params_save, flow_save, depth_save,
                                                                flow_target, depth_target, columns, flowaq_save)
        # Progress the counter to generate the model code
        i += 1

# Save the results
reach_data = tool.loc[:,:].to_records(index = False) # just to print reaches
save(i, j, columns, params_save, flow_target,
        depth_target, flow_save, depth_save, reach_data, model_ws, flowaq_save,
        save_100 = False)

end = datetime.datetime.now()

print('Runs terminated')
print('Number of runs: ', n)
print('Elapsed time (s): ', f'{(end-start).seconds}.{round((end-start).microseconds*(10**-6),2)}')
# %%
