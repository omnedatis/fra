# -*- coding: utf-8 -*-
"""
Created on Friday 11 25 11:14:39 2022

@author: Jeff
"""
import json
import logging
import os
import pickle
from typing import Any

import pandas as pd


def _get_valid_name(table_name: str, column_name: str) -> str:
    if '/' in column_name:
        column_name = column_name.replace('/', '-') + '_MOD'
    return f'{table_name}/{column_name}'


def _force_load(fp: str):
    if not os.path.isfile(fp):
        raise FileNotFoundError(f'file {fp} could not be found')
    ftype = fp.split('.')[-1]
    match ftype:
        case 'csv':
            data = pd.read_csv(fp, encoding='utf-8-sig')
        case 'json':
            data = json.load(open(fp, 'r'))
        case 'pkl':
            data = pickle.load(open(fp, 'rb'))
        case _:
            logging.warning('tring to force loading in binary form')
            data = pickle.load(open(fp, 'rb'))
    return data


def _force_dump(obj: Any, fp: str):
    dir_ = os.path.dirname(fp)
    if dir_ != '.' and not os.path.exists(dir_):
        os.makedirs(dir_)
    ftype = fp.split('.')[-1]
    match ftype:
        case 'csv':
            if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
                obj.to_csv(fp, encoding='utf-8-sig')
            else:
                with open(fp, 'w') as file:
                    file.writelines()
        case 'json':
            json.dump(obj, open(fp, 'w'), ensure_ascii=False)
        case 'pkl':
            pickle.dump(obj, open(fp, 'wb'))
        case _:
            logging.warning('tring to force dumping binary file')
            pickle.dump(obj, open(fp, 'wb'))
