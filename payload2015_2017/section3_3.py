import cynet.cynet as cn
import yaml
import glob

stream = file('config_pypi.yaml', 'r')
settings_dict = yaml.load(stream)

model_nums = settings_dict['model_nums']
MODEL_GLOB = settings_dict['MODEL_GLOB']
horizon = settings_dict['horizons'][0]
DATA_PATH = settings_dict['DATA_PATH']
RUNLEN = settings_dict['RUNLEN']
RESPATH = settings_dict['RESPATH']
FLEX_TAIL_LEN = settings_dict['FLEX_TAIL_LEN']
VARNAME=list(set([i.split('#')[-1] for i in glob.glob(DATA_PATH+"*")]))+['ALL']

cn.run_pipeline(MODEL_GLOB,model_nums, horizon, DATA_PATH, RUNLEN, VARNAME, RESPATH,\
    FLEX_TAIL_LEN=FLEX_TAIL_LEN,cores=4,gamma=True)
