'''
LAK - OPTION 2
Script to run instances of a MODFLOW model simulating the head of a lowland spring using the LAK package

- Modifies the hydraulic contuctivity of .lak file
- Runs the model
- Exports the result as an .xlsx file

'''
#%% Setup
import os
import numpy as np
import pandas as pd
import flopy
import shutil

dir = "d:\Claudia\MAURICE" # Local directory
model_ws = os.path.join(dir,"busca_base_infittito_aprile24_sfr_icalc2_lake")  # Folder containing the model
lak_file = os.path.join(model_ws, "busca_base_infittito_apr24_sfr_icalc2_lake.lak")
model_name = "busca_base_infittito_apr24_sfr_icalc2_lake"

# Backup the original LAK file
backup_lak_file = lak_file + ".bak"
if not os.path.exists(backup_lak_file):
    shutil.copy(lak_file, backup_lak_file)

#%% Define loop parameters 

# Iteration parameters
hydraulic_conductivities = np.linspace(1e-4, 1e-2, 10)  # Define range of K values
specific_cell = (0, 10, 10)  # Cell to extract head results from (layer, row, column)

# Results storage
inputs = []
outputs = []

#%% LOOP
'''
LOOP
'''

for k_value in hydraulic_conductivities:
    # Modify the LAK file
    with open(backup_lak_file, "r") as file:
        lines = file.readlines()

    # The hydraulic conductivity is contained on the 2nd line
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
        outputs.append(None)
        continue

    # Extract head results
    hds = flopy.utils.HeadFile(os.path.join(model_ws, f"{model_name}.hds"))
    head = hds.get_data(kstpkper=(0, 0))[specific_cell]  # Assuming first time step and stress period
    
    # Append the input and output to lists
    inputs.append(k_value)
    outputs.append(head)

#%% Save to dataframe and export
df_results = pd.DataFrame({'factor_value': inputs, 'avg_heads': outputs})
df_results.to_excel(os.path.join(model_ws, 'lake_results.xlsx'))
