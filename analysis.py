# -*- coding: utf-8 -*-
from collections import defaultdict
import glob
import logging
import os
import threading as mt
from typing import Dict, Literal, Tuple, List, Optional

import numpy as np
import openpyxl
import pandas as pd
from pandas import ExcelFile
import streamlit as st
from _data import DataSet
from common import (
    get_cache_id, clear_cache, _force_dump, _force_load, LocalTables,
    ColumnInfo, DATA_CACHE_LOC, TASK_LOC, DELIMITER, Periods
)

from _states import get_is_analysis, set_is_analysis
# logging setting
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
logging.basicConfig(level=0 ,handlers=[stream_handler], format='%(message)s')

# streamlit config
st.set_page_config(layout='wide')

PAGES = ['標的因子對應資訊', '資料欄位資訊', '因子分析']

@st.cache
def get_cache_tables(cached_id:int) -> dict:
    _cache_tables = {}
    _cache_tables.update(get_target_feature_map(get_cache_id(1)))
    _cache_tables.update(get_schema(get_cache_id(0)))
    return _cache_tables

@st.cache
def get_dataset(cached_id:int) -> DataSet:
    return DataSet()

@st.cache
def get_schema(cached_id:int) -> Dict[Literal['資料欄位資訊'], Dict[str, pd.DataFrame]]:
    ret = {}
    sheet_names = openpyxl.load_workbook(LocalTables.MAIN_SCHEMA.get_file_loc()).sheetnames
    with ExcelFile(LocalTables.MAIN_SCHEMA.get_file_loc()) as ex_file:
        for sheet in sheet_names:
            ret[sheet] = pd.read_excel(ex_file, sheet)

    return {PAGES[1]:ret} 

@st.cache
def get_column_info(cached_id:int) -> Dict[str, ColumnInfo]:
    data:pd.DataFrame = get_schema(get_cache_id(0))[PAGES[1]]['欄位定義'][list(ColumnInfo._fields)].dropna()
    names = data['name'].values
    data = data.values

    return {n:ColumnInfo(*i.tolist()) for n, i in zip(names, data)}

@st.cache
def get_target_feature_map(cached_id:int) -> Dict[Literal['標的因子對應資訊'], Dict[str, pd.DataFrame]]:
    ret={}
    with ExcelFile(LocalTables.TF_MAP.get_file_loc()) as ex_file:
        for sheet, name in zip(ex_file.book.worksheets, ex_file.book.sheetnames):
            if sheet.sheet_state == 'visible':
                ret[name] = pd.read_excel(ex_file, name)
    return {PAGES[0]:ret}


@st.cache
def get_tf_map(cached_id:int) -> Dict[str, List[str]]:
    data = get_target_feature_map(get_cache_id(
        1))[PAGES[0]]['標的因子對應表'][['target_name', 'feature_name']].dropna()
    ret = defaultdict(list)
    for target, feature in data.values.tolist():
        ret[target].append(feature)
    return ret
    
@st.cache
def _get_single_tasks(cache_id) -> Dict[str, Dict[str, Tuple[pd.Series, pd.Series]]]:
    cinfo_map = get_column_info(get_cache_id(0))
    tf_map = get_tf_map(get_cache_id(1))
    esp_names = []
    single_feature = defaultdict(dict)

    for target, features in tf_map.items():
        try:
            t_data = dataset.series[cinfo_map[target].key]
        except Exception as esp:
            esp_names.append(target)
        for f in features:
            try:
                f_data = dataset.series[cinfo_map[f].key]
            except Exception as esp:
                esp_names.append(target)
            single_feature[target][f] = t_data, f_data

    if esp_names:
        logging.warning(f'invalid feature {esp_names} encontered')

    return single_feature

@st.cache
def _get_multi_task(cache_id):
    cinfo_map = get_column_info(get_cache_id(0))
    tf_map = get_tf_map(get_cache_id(1))
    esp_names = []
    multi_features = {}
    feat = []
    for target, features in tf_map.items():
        try:
            t_data = dataset.series[cinfo_map[target].key]
        except Exception as esp:
            esp_names.append(target)
        for f in features:
            try:
                f_data = dataset.series[cinfo_map[f].key]
            except Exception as esp:
                esp_names.append(target)
            feat.append(f_data)
                
        multi_features[target] = t_data, feat
    if esp_names:
        logging.warning(f'invalid feature {esp_names} encontered')

    return multi_features


def _single_feature_corr(tasks:Dict[str, Dict[str, Tuple[pd.Series, pd.Series]]],
        target_name:str) -> dict:

    ret = defaultdict(dict)
    for t_name, task in tasks.items():
        if t_name != target_name:
            continue
        for name, (X, Y) in task.items():
            ts = pd.concat([X, Y], axis=1, sort=True).dropna()
            ret[target_name][name] = np.corrcoef(ts.values.T)[0,1].tolist()
    return ret[target_name]


@st.cache
def get_disable_state(cache_id) -> bool:
    return get_is_analysis()

def _handle_analysis_on_click(is_analysis):
    set_is_analysis(is_analysis)


if __name__ == '__main__':
    cache_id = get_cache_id(0)
    dataset = get_dataset(cache_id)
    _cache_tables = get_cache_tables(cache_id)
    cinfo_map = get_column_info(cache_id)
    if get_is_analysis():
        page = PAGES[2]
    else:
        page = st.sidebar.selectbox('選擇頁面', options=[i for i in _cache_tables])

    match page:
        case '標的因子對應資訊':
            containers = st.tabs(_cache_tables[page])
            for idx, sheet in enumerate(_cache_tables[page]):
                table = _cache_tables[page][sheet]
                if sheet == LocalTables.TARGETS.sheet:
                    targets = st.sidebar.multiselect('選擇標的', table['target_name'].values.tolist())
                    def _styler(x:pd.DataFrame, color:str) -> np.ndarray:
                        ret = np.full(x.shape, False)
                        names = table['target_name'].values
                        _targets = np.array(targets)
                        names, _targets = np.ix_(names, _targets)
                        ret[(names==_targets).sum(axis=1)>0] =True
                        return np.where(ret, f'background-color: {color};', None)
                    if not targets:
                        targets = table['target_name'].values.tolist()
                        with containers[idx]:
                            st.dataframe(table)
                    else:
                        with containers[idx]:
                            _table = table.style.apply(_styler, color='#fcec3d', axis=None)
                            st.dataframe(_table)
                elif sheet == LocalTables.TF_MAP.sheet:
                    with containers[idx]:
                        names = table['target_name'].values
                        targets = np.array(targets)
                        names, targets = np.ix_(names, targets)
                        _table = table[(names == targets).sum(axis=1)>0]
                        st.dataframe(_table)
            st.sidebar.button('執行分析', key='analysis', on_click=lambda : _handle_analysis_on_click(True))

        case '資料欄位資訊':
            index_name = None
            sheet_name =_cache_tables[page]['欄位定義']
            
            index_names = st.sidebar.multiselect('因子清單', options=[i for i in cinfo_map])
            length = len(index_names)
            if length > 0:
                for row in range((length//4)+1):
                    cols = st.columns(4,)
                    col_names = index_names[row*4:(row+1)*4]
                    for idx, each in enumerate(col_names):
                        cinfo = cinfo_map[each]
                        cols[idx].dataframe(dataset.series[cinfo.key])
            st.sidebar.button('執行分析', key='analysis', on_click=lambda : _handle_analysis_on_click(True))

        case '因子分析':
            st.sidebar.button('返回設定頁', key='return', on_click=lambda : _handle_analysis_on_click(False))
            st.sidebar.selectbox('選擇領先期別', Periods.get_list())
            st.markdown('## 執行分析項目')
            targets = _cache_tables[PAGES[0]][LocalTables.TARGETS.sheet]['target_name'].tolist()
            s_task = _get_single_tasks(cache_id)
            containers = st.tabs(targets)
            for idx, name in  enumerate(targets):
                with containers[idx]:
                    ret = []
                    items = sorted(_single_feature_corr(
                        s_task, name).items(), key = lambda x: x[1], reverse=True)
                    for name, value in items:
                        ret.append({**cinfo_map[name]._asdict(), **{"coef":value}})
                    df = pd.DataFrame(ret)
                    st.dataframe(df)

        case _:
            raise RuntimeError(f'unrecognizable fable {page}')

    
    # with st.sidebar.expander('!!!'):
    #     cmd = st.text_input(' ')
    #     if cmd == 'clear':
    #         clear_cache(0)
   

        
      
