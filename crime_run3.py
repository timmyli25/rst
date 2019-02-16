import cynet.cynet as sp
import pandas as pd
import numpy as np
import pickle
STOREFILE='crime.p'
LOGFILE='crime.csv'

with open("list1.txt", "rb") as fp:
    tiles = pickle.load(fp)

S2=sp.spatioTemporal(log_store=STOREFILE,
                    types=None,
                    value_limits=[0,1],
                    grid=tiles,
                    init_date='2001-01-01',
                    end_date='2018-12-31',
                    freq='D', EVENT='Arrest',
                    threshold=0.05)
S2.fit(csvPREF='ARREST')
