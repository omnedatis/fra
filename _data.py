# -*- coding: utf-8 -*-
from collections import defaultdict, namedtuple
import logging
import json
import os
from typing import Dict, List, Union, NamedTuple

import pandas as pd
from sqlalchemy import create_engine

from common import (
    EngineConfig, LocalTables, ExcelFormats, DataSource, Sheet,
    SQLTables, SCHEMA_COLUMNS, DATA_LOC, DATA_CACHE_LOC, _force_dump, _force_load,
    get_full_name, ColumnInfo, TableInfo
)


def _read_local_table(table: Sheet) -> pd.DataFrame:
    if table.surfix == ExcelFormats.XSLX:
        data = pd.read_excel(table.file+table.surfix, table.sheet)
    else:
        raise RuntimeError(f'invalid local table type {table.surfix}')
    return data


class _MiscDataProvider:

    MISC_DB_LOC = f'{DATA_LOC}/_misc_db'

    def __init__(self) -> None:
        ...

    def _get_all_data(self, table_name: str) -> pd.DataFrame:
        if not os.path.isfile(f'{DATA_CACHE_LOC}/{table_name}.pkl'):
            logging.info('Loading all market data from DB')
            data = _read_local_table(Sheet(self.MISC_DB_LOC+f'/{table_name}', 'data', ExcelFormats.XSLX))
            _force_dump(data, f'{DATA_CACHE_LOC}/{table_name}.pkl')
        else:
            logging.info('Loading all market data from file')
            data = _force_load(f'{DATA_CACHE_LOC}/{table_name}.pkl')
        return data

    def _get_data(self, table_name:str, column_name:str) -> pd.Series:
        if not os.path.isfile(f'{self.MISC_DB_LOC}/{table_name}.xlsx'):
            raise FileNotFoundError(f'table {table_name} not found')
            
        full_name = get_full_name(table_name, column_name)
        if not os.path.isfile(f'{DATA_CACHE_LOC}/{full_name}.pkl'):
            logging.info(f'Loading data on {full_name} from all file')
            table = self._get_all_data(table_name).set_index('PriceDt')
            market_table = table[column_name]
            _force_dump(market_table, f'{DATA_CACHE_LOC}/{full_name}.pkl')
        else:
            logging.info(f'Loading data on {full_name} from file')
            market_table = _force_load(f'{DATA_CACHE_LOC}/{full_name}.pkl')
        return market_table

    def get_data(self, table_name: str, column_name: str) -> pd.Series:
        logging.info(f'Getting data on table: {table_name} column: {column_name}')
        
        market_table = self._get_data(table_name, column_name)
        
        return market_table


class _SQLDBDataProvider:

    ENG_CONFIG_LOC = f'./eng_config.json'
    CONFIG_STR = ('{sql_engine}://{user}:{password}@{ip_addr}'
                  ':{port}/{db_name}?charset={charset}')

    def __init__(self):
        self._config = json.load(open(self.ENG_CONFIG_LOC, 'r'))
        self._engine = create_engine(self.CONFIG_STR.format(
            **EngineConfig(**self._config['db'])._asdict()))
        self._table_cache: Dict[str, pd.DataFrame] = {}

    def _get_all_data(self, sql_table: SQLTables) -> pd.DataFrame:
        if not os.path.isfile(f'{DATA_CACHE_LOC}/{sql_table.table_name}.pkl'):
            logging.info('Loading all market data from DB')
            columns = sql_table.get_names()
            sql = f'SELECT * FROM {sql_table.table_name}'
            data = pd.read_sql_query(sql, self._engine)[columns]
            data = data.set_index(sql_table.get_pk_names()[0])
            data.index = data.index.values.astype('datetime64[D]')
            _force_dump(data, f'{DATA_CACHE_LOC}/{sql_table.table_name}.pkl')
        else:
            logging.info('Loading all market data from file')
            data = _force_load(f'{DATA_CACHE_LOC}/{sql_table.table_name}.pkl')
        return data

    def _get_data(self, table_name:str, column_name:str) -> pd.Series:
        sql_table = SQLTables.get(table_name)
        full_name = get_full_name(table_name, column_name)

        if not os.path.isfile(f'{DATA_CACHE_LOC}/{full_name}.pkl'):
            logging.info(f'Loading data on {full_name} from all file')
            table = self._get_all_data(sql_table)
            market_table = table[table[sql_table.get_pk_names()[1]]==column_name].iloc[:,1]
            _force_dump(market_table, f'{DATA_CACHE_LOC}/{full_name}.pkl')
        else:
            logging.info(f'Loading data on {full_name} from file')
            market_table = _force_load(f'{DATA_CACHE_LOC}/{full_name}.pkl')
        return market_table

    def get_data(self, table_name: str, column_name: str) -> pd.Series:
        logging.info(f'Getting data on table: {table_name} column: {column_name}')
        
        market_table = self._get_data(table_name, column_name)
        
        return market_table

    def drop_cache(self):
        self._table_cache = {}

    @property
    def engine(self):
        return self._engine

    @property
    def engine(self, value):
        if hasattr(self, '_engine'):
            raise RuntimeError('resetting sql engine is not allowed')
        self._config = value

class ColumnInfoManager:
    ...
class DataSet:
    CONFIG_LOC = f'{DATA_LOC}/schema.json'

    def __init__(self):
        self._sql_db = _SQLDBDataProvider()
        self._misc_db = _MiscDataProvider()
        self._gen_config()
        self._ext_cols = {}
        self._series: Dict[str, pd.Series] = {}
        self._get_data()

    def _get_data(self):
        tables: Dict[str, TableInfo] = json.load(open(self.CONFIG_LOC, 'r'))
        for tname, tinfo in tables.items():
            source:str = tinfo[TableInfo._fields[0]]
            column_infos:List[Dict[str, str]] = tinfo[TableInfo._fields[1]]
            for cinfo in column_infos:
                cinfo =  ColumnInfo(**cinfo)
                if source == DataSource.SQL:
                    data = self._sql_db.get_data(
                        cinfo.table, cinfo.code).rename(cinfo.key)
                elif source == DataSource.MISC:
                    data = self._misc_db.get_data(
                        cinfo.table, cinfo.code).rename(cinfo.key)
                else:
                    raise RuntimeError(f'undefined source {tinfo}')
     
                self.update_series(data)
                self.update_col_info(cinfo.key, cinfo) 

    @classmethod
    def _gen_config(cls)  -> None:
        tables = {}
        schema = _read_local_table(
            LocalTables.MAIN_SCHEMA)[list(ColumnInfo._fields)].dropna()
        for each in range(schema.values.shape[0]):
            column_info = ColumnInfo(*schema.values[each, :].tolist())
            if column_info.code:
                if column_info.table not in tables:
                    tables[column_info.table] = TableInfo(column_info.source, [])._asdict()
                tables[column_info.table][TableInfo._fields[1]].append(
                    column_info._asdict())
        json.dump(tables, open(cls.CONFIG_LOC, 'w'), ensure_ascii=False)

    @property
    def series(self):
        copied = self._series.copy()
        return copied

    def update_series(self, value: pd.Series):

        if value.name in self._series:
            raise RuntimeError('reseting data not allowed')
        self._series[value.name] = value

    def update_col_info(self, col_code: str, col_info: ColumnInfo):
        if col_code in self._ext_cols:
            invalid = False if len(col_info) == len(
                self._ext_cols[col_code]) else True
            registered = self._ext_cols[col_code]
            for old, new in zip(registered, col_info):
                invalid &= old != new
            if invalid:
                raise RuntimeError(
                    f'inconsistent definition of {col_code} encountered')
        if not isinstance(col_info, ColumnInfo):
            raise TypeError(
                f'column info can only be of type {ColumnInfo}')
        self._ext_cols[col_code] = col_info
