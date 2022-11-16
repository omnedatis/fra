# -*- coding: utf-8 -*-
from collections import defaultdict
import glob
import logging
import os
from typing import Dict, Literal, Tuple, List

import numpy as np
import openpyxl
import pandas as pd
from pandas import ExcelFile
import streamlit as st
from _data import DataSet
from common import (
    get_cache_id, clear_cache, _force_dump, _force_load, LocalTables, ColumnInfo,
    Task, DATA_CACHE_LOC, TASK_LOC, DELIMITER
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
def get_dataset(cache_id):
    return DataSet()

@st.cache
def get_target_feature_map(cache_id:int) -> Dict[Literal['標的因子對應資訊'], Dict[str, pd.DataFrame]]:
    ret={}
    sheet_names = openpyxl.load_workbook(LocalTables.TF_MAP.get_file_loc()).sheetnames
    with ExcelFile(LocalTables.TF_MAP.get_file_loc()) as ex_file:
        for sheet in sheet_names:
            ret[sheet] = pd.read_excel(ex_file, sheet)
    return {PAGES[0]:ret}

@st.cache
def get_schema(cache_id:int) -> Dict[Literal['資料欄位資訊'], Dict[str, pd.DataFrame]]:
    ret = {}
    sheet_names = openpyxl.load_workbook(LocalTables.MAIN_SCHEMA.get_file_loc()).sheetnames
    with ExcelFile(LocalTables.MAIN_SCHEMA.get_file_loc()) as ex_file:
        for sheet in sheet_names:
            ret[sheet] = pd.read_excel(ex_file, sheet)

    return {PAGES[1]:ret} 

@st.cache
def get_column_info(cache_id:int) -> Dict[str, ColumnInfo]:
    data:pd.DataFrame = get_schema(cache_id)[PAGES[1]]['欄位定義'][list(ColumnInfo._fields)].dropna()
    names = data['name'].values
    data = data.values

    return {n:ColumnInfo(*i.tolist()) for n, i in zip(names, data)}

@st.cache
def get_tf_map(cache_id) -> Dict[str, str]:
    data = get_target_feature_map(
        cache_id)[PAGES[0]]['標的因子對應表'][['target_name', 'feature_name']].dropna()
    ret = defaultdict(list)
    for target, feature in data.values.tolist():
        ret[target].append(feature)
    return ret
    
@st.cache
def _gen_tasks(cache_id):
    cinfo_map = get_column_info(cache_id)
    tf_map = get_tf_map(cache_id)
    tasks:List[Task] = []
    esp_names = []
    task_names = {}
    for target, features in tf_map.items():
        try:
            task_names[target] = [f for f in features]
            tasks.append(Task(dataset.series[cinfo_map[target].key], [dataset.series[cinfo_map[i].key] for i in features]))
        except Exception as esp:
            try:
                for i in features:
                    cinfo_map[i]
            except Exception as esp:
                esp_names.append(i)
    if esp_names:
        logging.warning(f'invalid feature {esp_names} encontered')
    for t in tasks:
        tname = str(t.target.name).split(DELIMITER)[-1]
        for idx, f in enumerate(t.features):
            df = pd.concat([t.target, f], axis=1, sort=True).astype('float32')
            _force_dump((f.astype('float32'), t.target.astype('float32')), f'{TASK_LOC}/{tname}/pkl/{tname}--{task_names[tname][idx]}.pkl')
            _force_dump(df, f'{TASK_LOC}/{tname}/csv/{tname}--{task_names[tname][idx]}.csv')
        _force_dump((t.target.astype('float32'),  [f.astype('float32') for f in t.features]), f'{TASK_LOC}/{tname}/{tname}.pkl')

@st.cache
def _single_feature_analysis(cache_id):
    # x transform
    tasks = glob.glob(f'{TASK_LOC}/*/pkl/*.pkl')
    datasets = [_force_load(i) for i in tasks]
    for X, Y in datasets:
        print(X, Y)
        # model:somemodel@Model = Model(x,y)
        # model.fit()
        
    # y transfrom
    # dataset transform

@st.cache
def get_disable_state(cache_id):
    return get_is_analysis()

def _handle_analysis_on_click(is_analysis):
    set_is_analysis(is_analysis)
    if is_analysis:
        _gen_tasks(cache_id)

if __name__ == '__main__':
    cache_id = get_cache_id()
    _cache_tables = {}
    _cache_tables.update(get_target_feature_map(cache_id))
    _cache_tables.update(get_schema(cache_id))
    dataset = get_dataset(cache_id)
    if get_is_analysis():
        page = PAGES[2]
    else:
        page = st.sidebar.selectbox('選擇頁面', options=[i for i in _cache_tables])

    match page:
        case '標的因子對應資訊':
            containers = st.tabs(_cache_tables[page])
            for idx, sheet in enumerate(_cache_tables[page]):
                table = _cache_tables[page][sheet]
                with containers[idx]:
                    st.dataframe(table)
                if sheet == LocalTables.TARGETS.sheet:
                    target = st.sidebar.selectbox('選擇標的', table['target_name'].values.tolist())
                if sheet == LocalTables.FEATURE.sheet:
                    features = st.sidebar.multiselect('選擇因子', table['feature_name'].values.tolist())
            st.sidebar.button('執行分析', key='analysis', on_click=lambda : _handle_analysis_on_click(True))

        case '資料欄位資訊':
            index_name = None
            sheet_name =_cache_tables[page]['欄位定義']
            cinfo_map = get_column_info(cache_id)
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
            st.sidebar.button('返回設定頁', key='return',  on_click=lambda : _handle_analysis_on_click(False))
        case _:
            raise RuntimeError(f'unrecognizable fable {page}')

    
    with st.sidebar.expander('!!!'):
        cmd = st.text_input('')
        if cmd == 'clear':
            clear_cache()
    # clear = st.sidebar.button('清空快取', key='clear_cache')
    # if clear:
    #     clear_cache()
   

        
      
