#%% Setup

# Import necessary packages
import datetime
import flopy
import flopy.utils.binaryfile as bf
import numpy as np
import os
import pandas as pd
import pickle
import rasterio as rio

#%% # Load DRN characteristics

# Define needed paths and model name
cwd = os.getcwd()   #where test files are stored (github)
model_ws = cwd
model_name = ['RUN0_15P_FONT', 'WIRR_15P_FONT']

drn = pd.read_excel(os.path.join(cwd, 'RUN0_15P_FONT_DRN_PARS.xlsx'))

drn.l = 0
drn.r = drn.r - 1
drn.c = drn.c - 1

lenght, width, thickness = 25, 10, 0.5

#%% # Define k realizations

kmin = 1e-6
kmax = 1e-3
kdf = {}
for i in range(0,50):
    k = np.random.default_rng().uniform(low=kmin, high=kmax, size = 148)
    kdf[i] = k

#%% 