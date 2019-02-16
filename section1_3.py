import cynet.cynet as cn
import numpy as np

EPS = 200
STOREFILE='crime.p'
CSVFILE='crime.csv'

grid={'Latitude':np.around(np.linspace(41.5,42.05,EPS),decimals=5),
      'Longitude':np.around(np.linspace(-87.9,-87.2,EPS),decimals=5),
      'Eps':EPS}

tiles=list([[grid['Latitude'][i],grid['Latitude'][i+1],grid['Longitude'][j], grid['Longitude'][j+1]]
            for i in np.arange(len(grid['Latitude'])-1)
            for j in np.arange(len(grid['Longitude'])-1)])


S0=cn.spatioTemporal(log_file=CSVFILE,
                     log_store=STOREFILE,
                     types=[['BURGLARY','THEFT','MOTOR VEHICLE THEFT']],
                     value_limits=None,
                     grid=tiles,
                     init_date='2001-01-01',
                     end_date='2018-12-31',
                     freq='D',
                     EVENT='Primary Type',
                     threshold=0.05)
S0.fit(csvPREF='CRIME-')
tiles=S0.getGrid()

with open("tiles.txt", "wb") as tiles_pickle:
    pickle.dump(tiles,tiles_pickle)
