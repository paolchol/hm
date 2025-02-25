#%% Setup

# Import necessary packages
from datetime import datetime
import flopy
import flopy.utils.binaryfile as bf
import numpy as np
import os
import pandas as pd
import pickle
import re

#%% Define paths and load DRN characteristics

# Define needed paths and model name
cwd = os.getcwd()   # Where this file is saved
model_ws = [os.path.join(cwd, 'RUN0_15P_FONT'), os.path.join(cwd, 'WIRR_15P_FONT')]
model_name = ['RUN0_15P_FONT', 'WIRR_15P_FONT']

# Load and define DRN characteristics
drn = pd.read_excel(os.path.join(cwd, 'DRN_PARS.xlsx'))
# Adjust l,r,c to avoid flopy renumbering
drn.l = 0
drn.r = drn.r - 1
drn.c = drn.c - 1
# Define lenght, width and sediment thickness of all the springs
lenght, width, thickness = 25, 10, 0.5
# Define the code for cell-by-cell DRN flow data storage
ipakcb = 50

#%% Define n k realizations

number_realiz = 50
kmin = 1e-6
kmax = 1e-4
kdf = {}
for i in range(0, number_realiz):
    k = np.random.default_rng().uniform(low=kmin, high=kmax, size = 148)
    kdf[i] = k

# Save the k realizations
kdfout = pd.DataFrame()
for i in kdf.keys():
    kdfout[f'M{i+1}'] = kdf[i]
kdfout.insert(0, 'reach', drn.reach)
kversion = datetime.today().strftime('%y%m%d_%H%M%S')
kdfout.to_csv(os.path.join(cwd, f'k_realizations_{kversion}.csv'), index = False)

#%% Set information on the transient model

n_sp = 20
n_ts = 5
# stress period 1 has only 1 time step, this is addressed directly in the loop at line XX

# Define the timestep and stress period to extract from the results

sp_hds = 9
ts_hds = 5

#%% LOOP

start = datetime.now()
for mn, mws in zip(model_name, model_ws): # Loop over different models
    for ki in range(0,number_realiz): # Loop over different k realizations
        print(f'Executing simulation {ki+1} for model {mn}')
        # Extract ith k realization and write it in the DRN file
        k = kdf[ki]
        drn.conductance = np.round((k*lenght*width)/thickness, 5)
        stress_period_data = {}
        for sp in range(0, n_sp):
            stress_period_data[sp] = drn.iloc[:,:-1].to_numpy().tolist()

        m = flopy.modflow.Modflow(mn, model_ws = mws, version='mf2k')
        mdrn = flopy.modflow.ModflowDrn(m, stress_period_data=stress_period_data)
        mdrn.ipakcb = 50
        mdrn.filenames = os.path.join(mws, f'{mn}.drn')
        mdrn.write_file(check = False)

        # Run the model
        success, buff = flopy.mbase.run_model(
            exe_name = os.path.join(mws, 'modflow-nwt.exe'),
            namefile = f'{mn}.nam',
            model_ws = mws,
            silent = True #False to test the code, then switch to True
        )

        # Read .hds and .cbb files
        hf = flopy.utils.HeadFile(os.path.join(mws, f'{mn}.hds'))
        hds = hf.get_data(kstpkper=(ts_hds-1, sp_hds-1))

        # Initialize the hds 3d save file at the first iteration
        if ki == 0:
            hds3d = np.ndarray((hds.shape[1], hds.shape[2], number_realiz))

        # Save the results in 3d arrays
        hds3d[:,:,ki] = hds[0,:,:]

        # Read .cbb file
        cbb = bf.CellBudgetFile(os.path.join(mws, f'{mn}.cbb'))

        # Extract the DRN flux for all drains at all timesteps
        sps, tss, tsave = [], [], pd.DataFrame()
        for s in range(0, n_sp):
            for t in range(0, n_ts):
                if s == 0 and t > 0: # stress period 1 has only 1 time step
                    pass
                else:
                    drains = cbb.get_data(kstpkper = (t, s), text = 'DRAIN')
                    flux = []
                    for r, c in zip(drn.r, drn.c):
                        flux.append(drains[0][0][r][c]) #first 0: access the array, second: first layer
                    tsave = pd.concat([tsave, pd.DataFrame(flux).transpose()], axis = 0)
                    sps.append(s+1)
                    tss.append(t+1)
        tsave['sp'], tsave['ts'] = sps, tss
        
        # Initialize the hds 3d save file at the first iteration
        if ki == 0:
            drn3d = np.ndarray((tsave.shape[0], tsave.shape[1], number_realiz))
        
        # Save the results in the 3d array
        drn3d[:,:,ki] = tsave

        # Close the .hds and .cbb files
        hf.close()
        cbb.close()

    # Print end of the single model runs
    print('\n\n')
    print(mn, f': {number_realiz} runs completed')
    hours = round(((datetime.now()-start).seconds + (datetime.now()-start).microseconds*(10**-6))/(60*60), 3)
    print('Elapsed time (h): ',  hours, '\n')
    
    # Save the 3d arrays
    with open(os.path.join(mws, f'{mn}_hds3d_{kversion}.pickle'), 'wb') as f:
        pickle.dump(hds3d, f, pickle.HIGHEST_PROTOCOL)
        print(f'3D hds data for SP {sp_hds} and TS {ts_hds} saved at: ',
              os.path.join(mws, f'{mn}_hds3d_{kversion}.pickle'))
    
    with open(os.path.join(mws, f'{mn}_drn3d_{kversion}.pickle'), 'wb') as f:
        pickle.dump(drn3d, f, pickle.HIGHEST_PROTOCOL)
        print(f'3D drn flux data for all SP and TS saved at: ',
              os.path.join(mws, f'{mn}_drn3d_{kversion}.pickle'))
    
# Print final messages
end = datetime.now()
hours = round(((end-start).seconds + (end-start).microseconds*(10**-6))/(60*60), 3)
print('Elapsed time (h): ',  hours)
print('The k version is: ', kversion)
