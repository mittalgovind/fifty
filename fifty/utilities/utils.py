import json
import re

from hyperopt import hp


def json_paramspace2hyperopt_paramspace(d):
    return {k: hp.choice(k, v) for k, v in d.items()}


def make_safe_filename(s: str, replacement='') -> str:
    for c in r'/\<>"/|?*':
        s = s[:200].replace(c, replacement)
    return s


def dict_to_safe_filename(d: dict, sep=',', replacement='') -> str:
    """
    :returns filename-safe string extracted from the dictionary (assuming that dict is json serializable)
    ```
    >>> dict_to_safe_filename({'b': 1, 'a': 0})
    'a=0,b=1'
    ```
    """
    fname = json.dumps(d, sort_keys=True)
    fname = fname.replace(':', '=')
    fname = re.sub('[\\s"]', replacement, fname)
    fname = re.sub(',', sep, fname)
    fname = f'({make_safe_filename(fname)[1:-1]})' # replace {...} with (...)
    return fname