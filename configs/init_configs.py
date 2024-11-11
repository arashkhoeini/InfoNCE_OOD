import yaml
from easydict import EasyDict as edict
import os.path as osp
from utils.utils import mkdir


def init_config(config_path, args):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    f.close()
    config = easy_dic(config)
    for key,value in vars(args).items():
        if '.' in key:
            key1, key2 = key.split('.')
            config[key1][key2] = value
        else:
            config[key] = value
    # config.snapshot = osp.join(config.snapshot, config.note)
    # mkdir(config.snapshot)
    return config

def easy_dic(dic):
    dic = edict(dic)
    for key, value in dic.items():
        if isinstance(value, dict):
            dic[key] = edict(value)
    return dic

# config parser should also supprot nested dictionary
def config_parser(config, args):
    for arg in args:
        if '=' not in arg:
            continue
        else:
            key, value = arg.split('=')
        
        keys = key.split('.')
        sub_config = config
        for k in keys[:-1]:
            sub_config = sub_config[k]
        
        final_key = keys[-1]
        sub_config[final_key] = type_align(sub_config[final_key], value)
    
    return config

def type_align(source, target):
    if isinstance(source, bool):
        if target == 'False':
            return False
        elif target == 'True':
            return True 
        else:
            raise ValueError
    elif isinstance(source, float):
        return float(target)
    elif isinstance(source, str):
        return target
    elif isinstance(source, int):
        return int(target)
    elif isinstance(source, list):
        return eval(target)
    else:
        print("Unsupported type: {}".format(type(source)))


def show_config(config, sub=False):
    msg = ''
    for key, value in config.items():
        if (key == 'source') or (key == 'target'):
            continue
        if isinstance(value, list):
            value = ' '.join([str(v) for v in value])
            msg += '{:>25} : {:<15}\n'.format(key, value)
        elif isinstance(value, dict):
            msg += show_config(value, sub=True)
        else:
            msg += '{:>25} : {:<15}\n'.format(key, value)
    return msg