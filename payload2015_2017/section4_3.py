import cynet.cynet as cn
import yaml

stream = file('config_pypi.yaml', 'r')
settings_dict = yaml.load(stream)
FLEX_TAIL_LEN = settings_dict['FLEX_TAIL_LEN']

cn.flexroc_only_parallel('models/*.log',tpr_threshold=0.85,fpr_threshold=None,FLEX_TAIL_LEN=FLEX_TAIL_LEN, cores=4)
