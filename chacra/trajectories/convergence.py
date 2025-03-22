import MDAnalysis as mda
import numpy as np
import pandas as pd



# Feature of interest is contact probabilities
# t is simulation frames not actual time

# use contact_types read_groups_by_frame to generate blocks of contact probabilities

# Block Averages
    # standard deviations 

# Running Averages / Cumulative Averages

# Autocorrelation

# Stationarity Tests (KPSS, ADF, etc)

# Standard Error - Estimate Future Precision

# fit autocorrelation to Ae-t/tconv
# estimate how long t is needed to get a certain precision


# Bootstrapping - do means all fall within a certain variation?
# Jackknifing - how much does removing a single data point affect the mean?


# fit cumulative average to a converging function and solve for avg(x)t - avg(x)inf < epsilon 
