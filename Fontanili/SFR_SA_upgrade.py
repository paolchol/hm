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
import math
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
        The option is not yet developed
    
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
                        'direct_precip', 'stream_et', 'stream_head', 'stream_depth', 'stream_width', 'streambed_cond', 'streambed_gradient']
            df.streambed_gradient = df.streambed_gradient.str.removesuffix('\n')
            df = change_type(df, ['l', 'r', 'c', 'iseg', 'ireach'], 'int') 
            df = change_type(df, ['flow_into_reach', 'flow_to_aquifer', 'flow_out_reach', 'overlnd_runoff',
                                    'direct_precip', 'stream_et', 'stream_head', 'stream_depth', 'stream_width',
                                    'streambed_cond', 'streambed_gradient'], 'float')
            df.reset_index(inplace = True, drop= True)
    else:
        for sp in range(nsp):
            pass
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
    d = df.loc[(df.ireach == reach) & (df.iseg == segment), 'stream_depth'].values[0]

    # Update the output structures
    params_save.append(params + [f,d])
    # Extract flow and depth in all reaches and add them to the output structures
    flow_save = pd.concat([flow_save, df.flow_out_reach], axis=1)
    depth_save = pd.concat([depth_save, df.stream_depth], axis=1)
    flowaq_save = pd.concat([flowaq_save, df.flow_to_aquifer], axis=1)

    # Save the results after 100 runs
    if i % 100 == 0:

        save(i+1, j, columns, params_save, flow_target,
                depth_target, flow_save, depth_save, reach_data, model_ws, flowaq_save)
        
        # Clear the output structures
        del params_save, flow_save, depth_save, flowaq_save

        params_save = []
        flow_save, depth_save = pd.DataFrame(), pd.DataFrame()
        j += 100
    
    return params_save, flow_save, depth_save, flowaq_save, j

def save(i, j, columns, params_save, flow_target, depth_target,
         flow_save, depth_save, reach_data, model_ws, flowaq_save):
    # Define column labels
    params_save = pd.DataFrame(params_save, columns = columns)
    # Add columns to params_save
    params_save['flow_diff'] = flow_target - params_save.flow_out_reach
    params_save['depth_diff'] = depth_target - params_save.stream_depth
    # Set columns in flow_save and depth_save
    flow_save.columns, depth_save.columns, flowaq_save.columns = [f'M{x}' for x in range(j,i)], [f'M{x}' for x in range(j,i)], [f'M{x}' for x in range(j,i)]
    flow_save['ireach'], depth_save['ireach'], flowaq_save['ireach'] = reach_data.ireach, reach_data.ireach, reach_data.ireach
    flow_save['iseg'], depth_save['iseg'], flowaq_save['iseg'] = reach_data.iseg, reach_data.iseg, reach_data.iseg
    # Save as CSV
    params_save.to_csv(os.path.join(model_ws, f'sfr_results_M{i-1}.csv'), index = False)
    flow_save.to_csv(os.path.join(model_ws, f'sfr_reach_flow_out_reach_M{i-1}.csv'), index = False)
    depth_save.to_csv(os.path.join(model_ws, f'sfr_reach_depth_M{i-1}.csv'), index = False)
    flowaq_save.to_csv(os.path.join(model_ws, f'sfr_reach_flow_to_aquifer_M{i-1}.csv'), index = False)

#%% # Define needed paths and model name

cwd = os.getcwd()
sfr_data = os.path.join(cwd, 'test_files', 'busca_sfr2_sfr_data.xlsx') # SFR characteristics
model_ws = os.path.join(cwd, 'test_files', 'sfr_model_test') # Model working directory
model_name = 'busca_sfr2'

#%% Load SFR characteristics

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
# 1SEG: 1 segment, the "testa" (the "head" of the fontanile) and the "asta" (the channel of the fontanile) are specified by the reach number in reach_t
# nSEG: n segments, one for the "testa", multiple for the "asta"
# sfr_type = '1SEG'
sfr_type = 'nSEG'

# Define the segment number of the "testa" and the number of segments of the "asta"
seg_t = 1               # segment number
seg_a = [1,2,3,4,5,6,7] # number of segments of the asta
tseg = False            # True: the "head" takes the whole seg_t, False: the "head" takes a subset of seg_t, specify the reaches of the "head" in reach_t
reach_t = 9             # reach number of the last reach of the "head"

## CHANGE NEEDED IF DIFFERENT NUMBER OF SEGMENTS ##
# If you have a different number of segments, you will have to change some rows in the Loop section
# Go to line 332 for explanation

# Define hydraulic conductivity parameter dictionary
# kt = t is "testa", the "head" of the fontanile
# ka = a is "asta", the channel of the fontanile
# 
# kt: a list containing the values to test
# ka: a list containing the values to test

k_dict = {
    'kt': [0.0001, 0.0003], 
    'ka': [0.0003, 0.0005]
}

# Define slope parameter dictionary
# st = t is "testa", the "head" of the fontanile
# sa = a is "asta", the channel of the fontanile
# 
# st: a list containing the values to test
# sa:
#   if sfr_type == '2SEG', a list containing the values to test
#   if sfr_type == 'nSEG', a list containing n lists with the values to test

# s_dict = {
#     'st': [0.0001, 0.00003],
#     'sa': [0.0003, 0.00005]
# }

s_dict = {
    'st': [0.0001, 0.00003],
    'sa': [[0.0003, 0.00005],
           [0.0003, 0.00005],
           [0.0003, 0.00005],
           [0.0003, 0.00005],
           [0.0003, 0.00005],
           [0.0003, 0.00005],
           [0.0003, 0.00005]]
}

# Define reach and segment from where to get the reach flow
reach = 72
segment = 1

# Define the target values for flow and depth
flow_target = 0.0506  # m3/s
depth_target = 0.40   # m

# Set the print of model runs to silent
silent = True # True: the MODFLOW runs will not be printed in the terminal

#%% Loop

'''
START OF LOOP
'''

if sfr_type == '1SEG':
    start = datetime.datetime.now()

    # Calculate the number of runs
    n = len(k_dict['kt'])*len(k_dict['ka'])*len(s_dict['st'])*len(s_dict['sa'])
    print(f'{n} runs will be performed')

    # Initialize needed variables
    i, j = 1, 1
    params_save = []
    flow_save, depth_save, flowaq_save = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    columns = ['m_code', 'kt','ka', 'st', 'sa', 'flow_out_reach', 'stream_depth']

    for kt in k_dict['kt']:
        # Transform reach_data to a pandas.DataFrame
        tool = pd.DataFrame(reach_data)
        # Change hydraulic conductivity and slope in the segments
        tool.loc[(tool.iseg == seg_t) & (tool.ireach <= reach_t), 'strhc1'] = kt
        for ka in k_dict['ka']:
            tool.loc[(tool.iseg == seg_t) & (tool.ireach > reach_t), 'strhc1'] = ka
            for st in s_dict['st']:
                tool.loc[(tool.iseg == seg_t) & (tool.ireach <= reach_t), 'slope'] = st
                for sa in s_dict['sa']:
                    tool.loc[(tool.iseg == seg_t) & (tool.ireach > reach_t), 'slope'] = sa
                    
                    params = [f'M{i}', kt, ka, st, sa]
                    params_save, flow_save, depth_save, flowaq_save, j = run(i, j, tool, model_ws, model_name,
                                                                            params, params_save, flow_save, depth_save,
                                                                            flow_target, depth_target, columns, flowaq_save)
                        
                    # Progress the counter to generate the model code
                    i += 1

    # Save the results
    reach_data = tool.loc[:,:].to_records(index = False)
    save(i, j, columns, params_save, flow_target,
            depth_target, flow_save, depth_save, reach_data, model_ws, flowaq_save)

    end = datetime.datetime.now()

elif sfr_type == 'nSEG':
    start = datetime.datetime.now()
    
    # Calculate the number of runs
    n = len(k_dict['kt'])*len(k_dict['ka'])*len(s_dict['st'])*math.prod([len(s_dict['sa'][x]) for x in range(len(s_dict['sa']))])
    print(f'{n} runs will be performed')

    # Initialize needed variables
    i, j = 1, 1
    params_save = []
    flow_save, depth_save, flowaq_save = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    columns = ['m_code', 'kt','ka', 'st'] + [f'sa{x}' for x in range(1, len(seg_a)+1)] + ['flow_out_reach', 'stream_depth']

    for kt in k_dict['kt']:
        # Transform reach_data to a pandas.DataFrame
        tool = pd.DataFrame(reach_data).copy()
        # Change hydraulic conductivity and slope in the segments
        if not tseg:
                    tool.loc[(tool.iseg == seg_t) & (tool.ireach <= reach_t), 'strhc1'] = kt
        else:
            tool.loc[tool.iseg == seg_t, 'strhc1'] = kt
        for ka in k_dict['ka']:
            tool.loc[tool.iseg != seg_t, 'strhc1'] = ka
            if not tseg:
                tool.loc[(tool.iseg == seg_t) & (tool.ireach > reach_t), 'strhc1'] = ka
            for st in s_dict['st']:
                if not tseg:
                    tool.loc[(tool.iseg == seg_t) & (tool.ireach <= reach_t), 'slope'] = st
                else:
                    tool.loc[tool.iseg == seg_t, 'slope'] = st
                ## CHANGE NEEDED BELOW ## 
                # The following for loops will have to be incremented or decreased based of the number
                # of segments in seg_a (and the number of lists seg_dict['sa'])
                for sa1 in s_dict['sa'][0]:
                    if seg_a[0] == seg_t and not tseg:
                        tool.loc[(tool.iseg == seg_a[0]) & (tool.ireach > reach_t), 'slope'] = sa1
                    else:
                        tool.loc[tool.iseg == seg_a[0], 'slope'] = sa1
                    for sa2 in s_dict['sa'][1]:
                        tool.loc[tool.iseg == seg_a[1], 'slope'] = sa2
                        for sa3 in s_dict['sa'][2]:
                            tool.loc[tool.iseg == seg_a[2], 'slope'] = sa3
                            for sa4 in s_dict['sa'][3]:
                                tool.loc[tool.iseg == seg_a[3], 'slope'] = sa4
                                for sa5 in s_dict['sa'][4]:
                                    tool.loc[tool.iseg == seg_a[4], 'slope'] = sa5                                    
                                    for sa6 in s_dict['sa'][5]:
                                        tool.loc[tool.iseg == seg_a[5], 'slope'] = sa6
                                        for sa7 in s_dict['sa'][6]:
                                            tool.loc[tool.iseg == seg_a[6], 'slope'] = sa7

                                            sas = [sa1, sa2, sa3, sa4, sa5, sa6, sa7] # slopes assigned to the segments
                                            params = [f'M{i}', kt, ka] + sas
                                            params_save, flow_save, depth_save, flowaq_save, j = run(i, j, tool, model_ws, model_name,
                                                                                                    params, params_save, flow_save, depth_save,
                                                                                                    flow_target, depth_target, columns, flowaq_save)
                                            # Progress the counter to generate the model code
                                            i += 1
    
    # Save the results
    reach_data = tool.loc[:,:].to_records(index = False) # just to print reaches
    save(i, j, columns, params_save, flow_target,
            depth_target, flow_save, depth_save, reach_data, model_ws, flowaq_save)

    end = datetime.datetime.now()
else:
    print('sfr_type variable is not defined correctly')

print('Runs terminated')
print('Number of runs: ', n)
print('Elapsed time (s): ', f'{(end-start).seconds + round((end-start).microseconds*(10**-6),2)}')