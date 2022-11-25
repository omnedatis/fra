# -*- coding: utf-8 -*-
from collections import defaultdict, namedtuple
import logging
import json
import os
from typing import Dict, List, Union, NamedTuple

import pandas as pd
from sqlalchemy import create_engine

from _const import (
    EngineConfig, LocalTables, ExcelFormats, DataSource, Table,
    SQLTables, DATA_LOC, DATA_CACHE_LOC, ColumnInfo, TableInfo, Feature
)
from _utils import _force_dump, _force_load, _get_valid_name


def _read_local_table(table: Table) -> pd.DataFrame:
    if table.surfix == ExcelFormats.XSLX:
        data = pd.read_excel(table.get_file_loc(), table.sheet)
    else:
        raise RuntimeError(f'invalid local table type {table.surfix}')
    return data


class _MiscDataProvider:

    MISC_DB_LOC = f'{DATA_LOC}/_misc_db'

    def __init__(self) -> None:
        ...

    @classmethod
    def _get_all_data(cls, table_name: str) -> pd.DataFrame:
        if not os.path.isfile(f'{DATA_CACHE_LOC}/{table_name}.pkl'):
            logging.info('Loading all market data from DB')
            # no interface
            data = _read_local_table(Table(cls.MISC_DB_LOC, table_name, 'data', ExcelFormats.XSLX))
            _force_dump(data, f'{DATA_CACHE_LOC}/{table_name}.pkl')
        else:
            logging.info('Loading all market data from file')
            data = _force_load(f'{DATA_CACHE_LOC}/{table_name}.pkl')
        return data

    @classmethod
    def _get_data(cls, table_name:str, column_name:str) -> pd.Series:
        if not os.path.isfile(f'{cls.MISC_DB_LOC}/{table_name}.xlsx'):
            raise FileNotFoundError(f'table {table_name} not found')
            
        full_name = _get_valid_name(table_name, column_name)
        if not os.path.isfile(f'{DATA_CACHE_LOC}/{full_name}.pkl'):
            logging.info(f'Loading data on {full_name} from all file')
            table = cls._get_all_data(table_name).set_index('PriceDt')
            market_table = table[column_name]
            _force_dump(market_table, f'{DATA_CACHE_LOC}/{full_name}.pkl')
        else:
            logging.info(f'Loading data on {full_name} from file')
            market_table = _force_load(f'{DATA_CACHE_LOC}/{full_name}.pkl')
        return market_table

    @classmethod
    def get_data(cls, table_name: str, column_name: str) -> pd.Series:
        logging.info(f'Getting data on table: {table_name} column: {column_name}')
        
        market_table = cls._get_data(table_name, column_name)
        
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
        full_name = _get_valid_name(table_name, column_name)

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

class DataSet:
    CONFIG_LOC = f'{DATA_LOC}/schema.json'

    def __init__(self):
        self._sql_db = _SQLDBDataProvider()
        self._misc_db = _MiscDataProvider()
        self._features: Dict[str, Feature] = {}
        self._gen_config()
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
                        cinfo.table, cinfo.code).rename(cinfo.name)
                elif source == DataSource.MISC:
                    data = self._misc_db.get_data(
                        cinfo.table, cinfo.code).rename(cinfo.name)
                else:
                    raise RuntimeError(f'undefined source {tinfo}')
     
                self._update_data(data, cinfo)

    @classmethod
    def _gen_config(cls) -> None:
        _tables = {}
        schema = _read_local_table(
            LocalTables.MAIN_SCHEMA)[list(ColumnInfo._fields)]
        for each in range(schema.values.shape[0]):
            column_info = ColumnInfo(*schema.values[each, :].tolist())
            # code exists means data source exists
            if (column_info.code == column_info.code):  
                if column_info.table not in _tables: # defaultdict-like action
                    _tables[column_info.table] = TableInfo(column_info.source, [])._asdict()
                _tables[column_info.table][TableInfo._fields[1]].append(
                    column_info._asdict())
        _tables = json.dumps(_tables, ensure_ascii=False).replace('NaN', 'null')
        tables = json.loads(_tables)
        json.dump(tables, open(cls.CONFIG_LOC, 'w'), ensure_ascii=False)

    @property
    def features(self) -> Dict[str, Feature]:
        copied = self._features
        return copied

    def _update_data(self, value: pd.Series, cinfo:ColumnInfo):
        if cinfo.key in self._features:
            raise RuntimeError('reseting data not allowed')
        self._features[cinfo.key] = Feature(cinfo, value)
