#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:41:40 2018

@author: sulem
"""
import logging
import sys
sys.path.insert(0, '/misc/vlgscratch4/BrunaGroup/sulem/chem/HGNN')
#from graph_reader import *
#from preprocessing import *
#from main import *
#from main_testing import *
from preprocessing.loading import load_qm9, load_experiment_sets, load_molecule, split_data
from functions.data_generator import load_graph_sets
import main_gnn

"""
split_data()
logging.info("Datasets successfully saved")
"""

"""
load_qm9(spatial=True)
logging.info("Dataset with spatial information successfully saved as processed data")



load_qm9(charge=True)
logging.info("Dataset with partial charge successfully saved as processed data")

"""
load_qm9(spatial=True, charge=True)
logging.info("Dataset with spatial information and partial charge successfully saved as processed data")


"""
load_graph_sets(n=6000, Nmax=30, c=0.95)
logging.info("Generated dataset successfully saved")
"""

"""
#load_experiment_sets()
load_experiment_sets(Ntrain=8000, Nvalid=1000, Ntest=1000)
logging.info("Experiment sets successfully imported")
"""

"""
load_experiment_sets()
logging.info("Experiment sets successfully imported")
"""

"""
load_molecule(1111)
logging.info("Molecule 1111 successfully saved")
"""

"""
load_N_molecules(100)
logging.warning("Debugging set successfully saved")
"""

"""
load_debug_set()
logging.warning("Debugging set successfully saved as tensors")
"""

"""
mae = main()
logging.warning("Training on the debugging set achieved")
"""

"""
mae, mae2 = test_hyperparameters()
"""

"""
err, mae = main_gnn.main()
"""

"""
load_train_valid_sets(400, 100)
"""

"""
main_ccn.main_ccn()
"""