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
# 2SEG: 2 segments, one for the "testa" (the "head" of the fontanile) and one for the "asta" (the channel of the fontanile)
# nSEG: n segments, one for the "testa", multiple for the "asta"
sfr_type = '2SEG'

# Define the segment number of the "testa" and the number of segments of the "asta"
seg_t = 1 # segment number
seg_a = 1 # number of segments

# Define hydraulic conductivity parameter dictionary
# kt = t is "testa", the "head" of the fontanile
# ka = a is "asta", the channel of the fontanile
# 
# kt: a list containing the values to test
# ka:
#   if sfr_type == '2SEG', a list containing the values to test
#   if sfr_type == 'nSEG', a list containing n lists with the values to test

k_dict = {
    'kt': [0.001, 0.003], 
    'ka': [0.003, 0.005]
}

# Define slope parameter dictionary
# st = t is "testa", the "head" of the fontanile
# sa = a is "asta", the channel of the fontanile
# 
# st: a list containing the values to test
# sa:
#   if sfr_type == '2SEG', a list containing the values to test
#   if sfr_type == 'nSEG', a list containing n lists with the values to test

s_dict = {
    'st': [0.0001, 0.00003, 0.00001],
    'sa': [0.0003, 0.00005, 0.00001]
}

# Define reach and segment from where to get the reach flow
reach = 72
segment = 1

# Define the target values for flow and depth
flow_target = 0.0506  # m3/s
depth_target = 0.40   # m

#%% Loop

'''
START OF LOOP
'''

def run_model(model_ws, model_name):
    success, buff = flopy.mbase.run_model(
                            exe_name = os.path.join(model_ws, 'MF2005.exe'),
                            namefile = f'{model_name}.nam',
                            model_ws = model_ws,
                            silent = False #False to test the code, then switch to True
                            )
    if not success:
                raise Exception("MODFLOW did not terminate normally.")

if sfr_type == '2SEG':

    #calculate the number of runs
    n = len(k_dict['kt'])*len(k_dict['ka'])*len(s_dict['st'])*len(s_dict['sa'])
    print(f'{n} runs will be performed')
    i = 1
    params_save = []
    flow_save, depth_save = pd.DataFrame(), pd.DataFrame()

    start = datetime.datetime.now()

    for kt in k_dict['kt']:
        # Transform reach_data to a pandas.DataFrame
        tool = pd.DataFrame(reach_data)
        # Change hydraulic conductivity and slope in the segments
        tool.loc[tool.iseg == seg_t, 'strhc1'] = kt
        for ka in k_dict['ka']:
            tool.loc[tool.iseg != seg_t, 'strhc1'] = ka
            for st in s_dict['st']:
                tool.loc[tool.iseg == seg_t, 'slope'] = st
                for sa in s_dict['sa']:
                    tool.loc[tool.iseg != seg_t, 'slope'] = sa

                # Write the new .sfr file transforming tool to reach_data
                reach_data = tool.loc[:,:].to_records(index = False)
                sfr.reach_data = reach_data
                sfr.write_file()
                # Run the model
                m_code = f'M{i}'
                run_model(model_ws, model_name)
    
                # Load the streamflow.dat file
                f = os.path.join(model_ws, f'{model_name}_streamflow.dat')
                df = load_streamflow_dat(f)

                # Extract flow and depth in the target reach
                f = df.loc[(df.ireach == reach) & (df.iseg == segment), 'flow_out_reach'].values[0]
                d = df.loc[(df.ireach == reach) & (df.iseg == segment), 'stream_depth'].values[0]
                
                
                # Update the output structures
                params = [m_code, kt, ka, st, sa, f, d]
                params_save.append(params)
                # Extract flow and depth in all reaches and add them to the output structures
                flow_save = pd.concat([flow_save, df.flow_out_reach], axis=1)
                depth_save = pd.concat([depth_save, df.stream_depth], axis=1)
                
                # Save the results after 100 runs
                if i % 100 == 0:
                    params_save = pd.DataFrame(params_save, columns = ['m_code', 'kt','ka', 'st', 'sa', 'flow_out_reach', 'stream_depth'])
                    # Add columns to params_save
                    params_save['flow_diff'] = flow_target - params_save.flow_out_reach
                    params_save['depth_diff'] = depth_target - params_save.stream_depth

                    # Save as CSV
                    params_save.to_csv(os.path.join(model_ws, f'sfr_results_M{i}.csv'), index = False)
                    flow_save.to_csv(os.path.join(model_ws, f'sfr_reach_flow_M{i}.csv'), index = False)
                    depth_save.to_csv(os.path.join(model_ws, f'sfr_reach_depth_M{i}.csv'), index = False)
                    
                    # Clear the output structures
                    del params_save, flow_save, depth_save
                    params_save = []
                    flow_save, depth_save = pd.DataFrame(), pd.DataFrame()
                    
                # Progress the counter to generate the model code
                i += 1

        # Save the results
        params_save = pd.DataFrame(params_save, columns = ['m_code', 'kt','ka', 'st', 'sa', 'flow_out_reach', 'stream_depth'])
        # Add columns to params_save
        params_save['flow_diff'] = flow_target - params_save.flow_out_reach
        params_save['depth_diff'] = depth_target - params_save.stream_depth
        # Save as CSV
        params_save.to_csv(os.path.join(model_ws, f'sfr_results_M{i}.csv'), index = False)
        flow_save.to_csv(os.path.join(model_ws, f'sfr_reach_flow_M{i}.csv'), index = False)
        depth_save.to_csv(os.path.join(model_ws, f'sfr_reach_depth_M{i}.csv'), index = False)

    end = datetime.datetime.now()

elif sfr_type == 'nSEG':
    print('sfr_type == nSEG is not developed yet')
    pass
else:
    print('sfr_type variable is not defined correctly')

print('Runs terminated')
print('Number of runs: ', n)
print('Elapsed time (s): ', f'{(end-start).seconds}.{round((end-start).microseconds*(10**-6),2)}')