import cynet.cynet as cn

CSVfile = ['ARREST.csv','CRIME-BURGLARY-THEFT-MOTOR_VEHICLE_THEFT.csv','CRIME-HOMICIDE-ASSAULT-BATTERY.csv']
begin = '2015-01-01'
end = '2017-12-31'
extended_end = '2018-12-31'
name = 'triplet/' + 'CRIME-'+'_' + begin + '_' + end

#Generates desired triplets.
cn.readTS(CSVfile,csvNAME=name,BEG=begin,END=end)

#Generates files which contains in sample and out of sample data.
cn.splitTS(CSVfile, BEG = begin, END = extended_end, dirname = './split', prefix = begin + '_' + extended_end)
