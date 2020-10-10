"""
HShake (https://github.com/gabliw)
"""

import os
import json
import urllib.request as request


def known_url_crawling(url_list):
    NotImplemented
    request.urlretrieve(url_list, "../Dataset/not_refine/test.jpg")


# image file rename function
def rename(target_dir):
    from shutil import copy
    files = sorted([os.path.join(target_dir, file) for file in os.listdir(target_dir)])

    for idx, file in enumerate(files[1:]):
        file_name = f"{idx + 1:03d}.png"
        print(file_name)
        copy(file, os.path.join('../Dataset/target', file_name))


# Contributed with Winterchild
class _Dict(dict):

    def __init__(self, *args, **kwargs):
        super(_Dict, self).__init__(*args, **kwargs)
        for k, v in self.items():
            if type(v) is dict:
                self[k] = _Dict(v)

    def __getattr__(self, key):
        def __proxy__(_dict, key):
            for k, v in _dict.items():
                if k == key:
                    return v
                if isinstance(v, _Dict):
                    ret = __proxy__(v, key)
                    if ret is not None:
                        return ret
                    else:
                        continue
        try:
            return __proxy__(self, key)
        except KeyError as e:
            raise AttributeError(e)

    def __setattr__(self, key, value):
        def __proxy__(_dict, key, value):
            for k in _dict.keys():
                if k == key:
                    _dict[k] = value
                    return True
                if isinstance(_dict[k], _Dict):
                    return __proxy__(_dict[k], key, value)
        if __proxy__(self, key, value):
            return
        self[key] = value

    def __delattr__(self, key):
        def __proxy__(_dict, key):
            for k in _dict.keys():
                if k == key:
                    del _dict[k]
                    return
                if isinstance(_dict[k], _Dict):
                    __proxy__(_dict[k], key)
        try:
            __proxy__(self, key)
        except KeyError as e:
            raise AttributeError(e)


class ConfigParser(_Dict):

    def __init__(self, filename):
        self.filename = os.path.abspath(filename)
        super(ConfigParser, self).__init__(self.read_configs())

    def read_configs(self):
        with open(self.filename) as f:
            configs = json.load(f)
        return configs

    def initialize_object(self, name, module, *args, **kwargs):
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_name for k in kwargs])
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)


def print_json(config, indent=''):
    for k, v in config.items():
        if type(config[k]) is _Dict:
            print(indent+f'{k}:')
            print_json(v, indent+'    ')
        else:
            print(indent+f'{k}: {v}')