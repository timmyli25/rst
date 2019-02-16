import cynet.cynet as sp
import pandas as pd
import numpy as np
import pickle

CSVfile = ['ARREST.csv','CRIME-BURGLARY-THEFT-MOTOR_VEHICLE_THEFT.csv','CRIME-HOMICIDE-ASSAULT-BATTERY.csv']
DATES  = []

DATES  = []

for year in range(2001, 2017):
    period_start = str(year) + '-01-01'
    period_end = str(year + 2) + '-12-31'
    period_end_extended = str(year + 3) + '-12-31'
    DATES.append((period_start, period_end, period_end_extended))
print(DATES)


for period in DATES:
    begin = period[0]
    end = period[1]
    extended_end = period[2]
    name = 'triplet/' + 'CRIME-'+'_' + begin + '_' + end
    #sp.readTS(CSVfile,csvNAME=name,BEG=begin,END=end)
    sp.splitTS(CSVfile, BEG = begin, END = extended_end, dirname = './split', prefix = begin + '_' + extended_end)