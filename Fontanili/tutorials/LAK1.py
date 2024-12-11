'''
LAK - OPTION 1
Script to run instances of a MODFLOW model simulating the head of a lowland spring using the LAK package

- Modifies the hydraulic contuctivity of .lak file
- Runs the model
- Exports the result as an .xlsx file

'''

#%% Setup

# Import necessary packages
import flopy
import numpy as np
import os
import pandas as pd

# Load the MODFLOW model
dir = "d:\Claudia\MAURICE" # Local directory
model_ws = os.path.join(dir,"busca_base_infittito_aprile24_sfr_icalc2_lake")  # Folder containing the model
model_name = "busca_base_infittito_apr24_sfr_icalc2_lake"
mf = flopy.modflow.Modflow.load(
    os.path.join(model_ws, f'{model_name}.nam'),
    model_ws = model_ws,
    exe_name='MF2005',
    version = 'mf2005',
    check=True,
    verbose = True
)
#%% ISSUE WITH SFR
# check sfr individually
sfr_file = f"{model_name}.sfr"
sfr = flopy.modflow.ModflowSfr2.load(sfr_file, mf, check=True, verbose=True)

#%% Access and Modify the LAK Package
# The hydraulic conductivity is typically associated with the lakebed leakance (lakarr or botm) in the LAK package.
lak = mf.lak  # Access the LAK package

# Get the lakebed conductance array
lak_leakance = lak.lakarr.array

# Modify the hydraulic conductivity (example: scale by a factor)
new_leakance = lak_leakance * 1.1  # Increase by 10%
lak.lakarr = flopy.utils.Util2d(mf, new_leakance.shape, value=new_leakance, name="lakarr")

# Define a range of hydraulic conductivity values
conductivity_factors = np.linspace(0.5, 2.0, 10)  # From 50% to 200% of the original

#%% Define loop parameters 

# Define limits of hydraulic conductivity range and number of iterations
ki = 0.01
kf = 0.000001
n = 10 #10 to test the code, then switch to 100
step = np.linspace(ki,kf,n)
inputs = []
outputs = []


#%% LOOP
'''
LOOP
'''
for factor in conductivity_factors:
    # Update lakebed leakance
    lak.lakarr = flopy.utils.Util2d(mf, lak_leakance.shape, value=lak_leakance * factor, name="lakarr")

    # Write input files
    mf.write_input()

    # Run the model
    success, buff = mf.run_model(silent=True)
    if not success:
        print(f"Model run failed for factor {factor}")
        inputs.append(factor)
        outputs.append(None)
        continue

    # Extract head results
    hds = flopy.utils.HeadFile(f"{model_name}.hds")
    heads = hds.get_data()
    
    # Store the average head for the lake cells
    lake_heads = heads[lak.ibound.array > 0]  # Assuming positive ibound indicates lake cells
    avg_head = lake_heads.mean()
    
    # Append the input and output to lists
    inputs.append(factor)
    outputs.append(avg_head)

#%% Save to dataframe and export
df_results = pd.DataFrame({'factor_value': inputs, 'avg_heads': outputs})
df_results.to_excel(os.path.join(model_ws, 'lake_results.xlsx'))

