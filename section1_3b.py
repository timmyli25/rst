import cynet.cynet as cn
import pickle

STOREFILE='crime.p'
CSVFILE='crime.csv'

with open("tiles.txt", "rb") as tiles_pickle:
    tiles = pickle.load(tiles_pickle)

S01=cn.spatioTemporal(log_store=CSVFILE,
                     types=[['HOMICIDE','ASSAULT','BATTERY']],
                     value_limits=None,
                     grid=tiles,
                     init_date='2001-01-01',
                     end_date='2018-12-31',
                     freq='D',threshold=0.05)
S01.fit(csvPREF='CRIME-')
