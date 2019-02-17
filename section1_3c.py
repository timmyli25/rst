import cynet.cynet as cn
import pickle

STOREFILE='crime.p'
CSVFILE='crime.csv'

with open("list1.txt", "rb") as tiles_pickle:
    tiles = pickle.load(tiles_pickle)

S2=cn.spatioTemporal(log_store=STOREFILE,
                    types=None,
                    value_limits=[0,1],
                    grid=tiles,
                    init_date='2001-01-01',
                    end_date='2018-12-31',
                    freq='D', EVENT='Arrest',
                    threshold=0.05)
S2.fit(csvPREF='ARREST')
