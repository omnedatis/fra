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

SCHEMA_COLUMNS = ['code', 'name', 'freq', 'label',
                  'label2', 'label3', 'source', 'table']
DOC_LOC = '.\\_docs'
OUT_LOC = '.\\_reports'
DATA_LOC = '.\\_data'
DATA_CACHE_LOC = DATA_LOC + '\\_cache'
TASK_LOC = DATA_CACHE_LOC + '\\_tasks'
DATE_INDEX_NAME = 'PriceDt'
PRICE_ALIAS = {DATE_INDEX_NAME: ['PriceDt']}


DELIMITER = '`'


class ColumnInfo(NamedTuple):
    code: str
    name: str
    freq: str
    source: str
    table: str
    label: str
    label2: str
    label3: str

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


class Table(NamedTuple):
    file_table: str
    file_name: str
    sheet: str
    surfix: ExcelFormats

    def get_file_loc(self):
        return self.file_table+'\\'+self.file_name+self.surfix.value


class LocalTables(Table, Enum):
    MAIN_SCHEMA = Table(DOC_LOC, 'schema', '欄位定義', ExcelFormats.XSLX)
    TARGETS = Table(DOC_LOC, 'target_feature_map', '標的列表', ExcelFormats.XSLX)
    TFEATURES = Table(DOC_LOC, 'target_feature_map',
                      '標的因子列表', ExcelFormats.XSLX)
    FEATURES = Table(DOC_LOC, 'target_feature_map', '因子列表', ExcelFormats.XSLX)
    TF_MAP = Table(DOC_LOC, 'target_feature_map', '標的因子對應表', ExcelFormats.XSLX)
    F_TYPE = Table(DOC_LOC, 'target_feature_map', '因子類型', ExcelFormats.XSLX)


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
    TWOQUATAR = Period(2, 'Q')
    FOURQUATAR = Period(4, 'Q')

    @classmethod
    def get_names(cls) -> List[str]:
        return [i.name for i in cls]

    @classmethod
    def get(cls, value) -> Period:
        for i in cls:
            if i.name == value:
                return i
        raise RuntimeError(f'Invalid period value encounterd {value}')


class Feature(NamedTuple):
    cinfo: ColumnInfo
    series: pd.Series


class Task(NamedTuple):
    tname: str
    target: Feature
    features: List[Feature]


ONE_DAY = Period(1, 'D')
