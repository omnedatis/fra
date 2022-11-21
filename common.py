# -*- coding: utf-8 -*-
from collections import namedtuple
import datetime
from enum import Enum
import json
import logging
import os
import pickle
import threading as mt
from typing import NamedTuple, List, Union, Optional, Any, Dict, Callable, Tuple, Literal

import pandas as pd

SCHEMA_COLUMNS = ['code', 'source', 'table', 'label', 'name', 'label2']
OUT_LOC = './_reports'
DATA_LOC = './_data'
DATA_CACHE_LOC = DATA_LOC + '/_cache'
TASK_LOC = DATA_CACHE_LOC + '/_tasks'
DATE_INDEX_NAME = 'PriceDt'
PRICE_ALIAS = {DATE_INDEX_NAME: ['PriceDt']}


DELIMITER = '`'


class ColumnInfo(NamedTuple):
    code: str
    name: str
    freq:str
    source: str
    table: str
    label: str
    label2: str

    @property
    def key(self):
        return DELIMITER.join([self.code, self.table, self.name])


class TableInfo(NamedTuple):
    table_source: str
    column_info: List[ColumnInfo] = []


class EngineConfig(NamedTuple):
    sql_engine: str
    user: str
    password: str
    ip_addr: str
    port: str
    db_name: str
    charset: str


class DataSource(str, Enum):
    SQL = 'SQL_DB'
    MISC = 'MISC_DB'


class Dtype(NamedTuple):
    code: str
    dtype: Union[str, int, float, datetime.date, datetime.datetime]


class Schema(NamedTuple):
    table_name: str
    columns: List[Dtype]
    pk_idxs: List[int]
    data_idxs: Optional[List[int]]


class SQLTables(Schema, Enum):
    HIST_DATA = Schema('ai_pd_historyprice', [
        Dtype('ProCode', str),
        Dtype('PriceDt', datetime.date),
        Dtype('CP', float),
        Dtype('HP', float),
        Dtype('LP', float),
        Dtype('OP', float),
        Dtype('Volume', int),
        Dtype('Amount', int),
        Dtype('Turnover', float),
        Dtype('MODIFY_DT', datetime.date),
        Dtype('CREATE_DT', datetime.date)
    ], pk_idxs=[1, 0], data_idxs=[2])

    def get_columns(self) -> List[str]:
        return [i.code for i in self.columns]

    def get_names(self) -> List[str]:
        ret = [self.columns[i].code for i in self.pk_idxs]
        if self.data_idxs is not None:
            ret += [self.columns[i].code for i in self.data_idxs]
        return ret

    def get_pk_names(self) -> List[str]:
        return [self.columns[i].code for i in self.pk_idxs]

    def get_data_names(self) -> List[str]:
        if self.data_idxs is None:
            raise RuntimeError('data columns are not defined')
        return [self.columns[i].code for i in self.pk_idxs]

    @classmethod
    def get(cls, value: str):
        for each in cls:
            if each.table_name == value:
                return each
        raise RuntimeError(f'table {value} not found')


class ExcelFormats(str, Enum):
    XSLX = '.xlsx'


class Sheet(NamedTuple):
    file: str
    sheet: str
    surfix: ExcelFormats


class LocalTables(Sheet, Enum):
    MAIN_SCHEMA = Sheet('./_docs/schema', '欄位定義', ExcelFormats.XSLX)
    TARGETS = Sheet('./_docs/target_feature_map', '標的列表', ExcelFormats.XSLX)
    FEATURE = Sheet('./_docs/target_feature_map', '因子列表', ExcelFormats.XSLX)
    TF_MAP = Sheet('./_docs/target_feature_map', '標的因子對應表', ExcelFormats.XSLX)
    F_TYPE = Sheet('./_docs/target_feature_map', '因子類型', ExcelFormats.XSLX)

    def get_file_loc(self):
        return self.file+self.surfix.value


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


def get_valid_name(table_name: str, column_name: str) -> str:
    if '/' in column_name:
        column_name = column_name.replace('/', '-') + '_MOD'
    return f'{table_name}/{column_name}'


class ReturnableTread:

    def __init__(self, target: Callable, args: Tuple):
        self._target = target
        self._args = args
        self._tread = mt.Thread(target=self._run)
        self._return = None

    def _run(self):
        if self._args is not None:
            ret = self._target(self._args)
        else:
            ret = self._target()
        self._return = ret


PERIOD_NAME_MAP = {
    'D': '日',
    'W': '週',
    'M': '月',
    'Q': '季'
}

PERIOD_NUM_MAP = {
    'D': 1,
    'W': 5,
    'M': 20,
    'Q': 60
}


class Period(NamedTuple):
    unit: int
    type: Literal['D', 'W', 'M', 'Q']

    @property
    def name(self) -> str:
        return f'{self.unit} {PERIOD_NAME_MAP[self.type]}'

    @property
    def steps(self) -> int:
        return self.unit*PERIOD_NUM_MAP[self.type]


class Periods(Period, Enum):
    TWOWEEK = Period(2, 'W')
    ONEMONTH = Period(1, 'M')
    TWOMONTH = Period(2, 'M')
    ONEQUATAR = Period(1, 'Q')

    @classmethod
    def get_name(cls) -> List[str]:
        return [i.name for i in cls]

    @classmethod
    def get(cls, value) -> Period:
        for i in cls:
            if i.name == value:
                return i
        raise RuntimeError(f'Invalid period value encounterd {value}')