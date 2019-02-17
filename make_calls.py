import cynet.cynet as cn

stream = file('config_pypi.yaml', 'r')
settings_dict=yaml.load(stream)

TS_PATH=settings_dict['TS_PATH']
NAME_PATH=settings_dict['NAME_PATH']
LOG_PATH=settings_dict['LOG_PATH']
FILEPATH=settings_dict['FILEPATH']
END=settings_dict['END']
BEG=settings_dict['BEG']
NUM=settings_dict['NUM']
PARTITION=settings_dict['PARTITION']
XgenESeSS=settings_dict['XgenESeSS']
RUN_LOCAL=settings_dict['RUN_LOCAL']

XG = cn.xgModels(TS_PATH,NAME_PATH, LOG_PATH,FILEPATH, BEG, END, NUM, PARTITION, XgenESeSS,RUN_LOCAL)
XG.run(workers=4)
