import cynet.cynet as sp
import pandas as pd
import numpy as np
import pickle
STOREFILE='crime.p'
LOGFILE='crime.csv'

EPS=200
DATES  = []

for year in range(2001, 2017):
    period_start = str(year) + '-01-01'
    period_end = str(year + 2) + '-12-31'
    DATES.append((period_start, period_end))
print(DATES)

grid={'Latitude':np.around(np.linspace(41.5,42.05,EPS),decimals=5),
      'Longitude':np.around(np.linspace(-87.9,-87.2,EPS),decimals=5),
      'Eps':EPS}


tiles=list([[grid['Latitude'][i],grid['Latitude'][i+1],grid['Longitude'][j], grid['Longitude'][j+1]]
            for i in np.arange(len(grid['Latitude'])-1)
            for j in np.arange(len(grid['Longitude'])-1)])


S0=sp.spatioTemporal(log_file=LOGFILE,
                     log_store=STOREFILE,
                     types=[['BURGLARY','THEFT','MOTOR VEHICLE THEFT']],
                     value_limits=None,
                     grid=tiles,
                     init_date='2001-01-01',
                     end_date='2018-12-31',
                     freq='D',threshold=0.05)
S0.fit(csvPREF='CRIME-')
tiles=S0.getGrid()

with open("list1.txt", "wb") as fp:
    pickle.dump(tiles,fp)

'''
S01=sp.spatioTemporal(log_store=STOREFILE,
                     types=[['HOMICIDE','ASSAULT','BATTERY']],
                     value_limits=None,
                     grid=tiles,
                     init_date='2001-01-01',
                     end_date='2018-12-31',
                     freq='D',threshold=0.05)
S01.fit(csvPREF='CRIME-')

S2=sp.spatioTemporal(log_store=STOREFILE,
                    types=None,
                    value_limits=[0,1],
                    grid=tiles,
                    init_date='2001-01-01',
                    end_date='2018-12-31',
                    freq='D', EVENT='Arrest',
                    threshold=0.05)
S2.fit(csvPREF='ARREST')


CSVfile = ['ARREST.csv','CRIME-BURGLARY-THEFT-MOTOR_VEHICLE_THEFT.csv','CRIME-HOMICIDE-ASSAULT-BATTERY.csv']

for period in DATES:
    begin = period[0]
    end = period[1]
    name = 'triplet/' + 'CRIME-'+'_' + begin + '_' + end
    sp.readTS(CSVfile,csvNAME=name,BEG=begin,END=end)
    sp.splitTS(CSVfile, BEG = begin, END = end, dirname = './split', prefix = begin + '_' + end)
'''
