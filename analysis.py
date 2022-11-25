# -*- coding: utf-8 -*-
"""
Created on Thursday 11 24 12:45:04 2022

@author: Jeff
"""
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
from pandas import ExcelFile
import streamlit as st

from _const import (
    SCHEMA_COLUMNS, OUT_LOC, LocalTables, ColumnInfo, Task, Periods, Feature
)
from _utils import _force_dump, _force_load
from _data import DataSet
from _model import SingleCorrModel, PeriodTransformer
from  _states import (
    get_cache_id, get_tf_map_table, set_tf_map_table, 
    get_targets, set_targets, get_reports, set_reports, get_period, set_period
)

st.set_page_config(layout='wide')

@st.cache  # immutable
def get_dataset(cache_id: int) -> DataSet:
    return DataSet()

@st.cache # immutable
def get_tables(cache_id) ->Dict[str, Dict[str, pd.DataFrame]]:
    ret = defaultdict(dict)
    for each in LocalTables:
        with ExcelFile(each.get_file_loc()) as file:
            ws_name = file.book.sheetnames
            ws_ = file.book.worksheets
            for name, sheet in zip(ws_name, ws_):
                if sheet.sheet_state == 'visible':
                    ret[each.file_name][name] = pd.read_excel(file, name)
    return ret 

@st.cache # immutable
def get_features(cache_id) -> Dict[str, ColumnInfo]:
    _schema = LocalTables.MAIN_SCHEMA
    feats:pd.DataFrame  = get_tables(get_cache_id(0))[_schema.file_name][_schema.sheet]
    ret = {}
    ridx  = feats.shape[0]
    for each in range(ridx):
        _row = feats.iloc[each,:]
        ret[_row['name']] = ColumnInfo(**_row[SCHEMA_COLUMNS].to_dict())
    return ret
    

@st.cache # immutable
def get_tf_map_from_file(cache_id) -> Dict[str, Dict[str, List[str]]]:
    tables = get_tables(get_cache_id(0))
    tfeatures = tables[LocalTables.TFEATURES.file_name][LocalTables.TFEATURES.sheet]
    features = tables[LocalTables.FEATURES.file_name][LocalTables.FEATURES.sheet]
    ret = defaultdict(lambda: defaultdict(list))
    for t, n, tf in tfeatures[['target_code', 'target_name', 'name']].values.tolist():
        for t_, f in features[['target_code', 'name']].values.tolist():
            if t == t_:
                ret[n][tf].append(f)
    return ret

@st.cache #1
def get_tf_map(cache_id):
    return get_tf_map_table() or get_tf_map_from_file(get_cache_id(0))

@st.cache #1
def _gen_task(cache_id) -> Dict[str, List[Task]]:
    ret = defaultdict(list)
    tf_map = get_tf_map(get_cache_id(1))
    for target, tvalue in tf_map.items():
        for tfeature, tfvalue in tvalue.items():
            _feautres = [DATA.features[FEATURES[i].key] for i in tfvalue]
            task = Task(tfeature, DATA.features[FEATURES[tfeature].key], _feautres)
            ret[target].append(task)
    return ret


def _get_corr(task:Task):
    target = task.target
    features = task.features
    model = SingleCorrModel(
                y_transformer=PeriodTransformer(get_period()))
    ret = []
    for each in features:
        value = model.get_corr(each, target)[0, 1].tolist()
        ret.append({**each.cinfo._asdict(), **{'coef':value}})
    return pd.DataFrame(ret).set_index('name')



def _handle_report_on_click():
    reports = get_reports()
    if reports:
        for name, each in reports.items():
            _force_dump(each, f'{OUT_LOC}/{name}.csv')
        set_reports({})

PAGES = ['基本資訊檢視', '資料檢視', '更新標的因子對應表', '因子分析', '領先期別分析']

if __name__ == '__main__':
    FEATURES = get_features(get_cache_id(0))
    TABLES = get_tables(get_cache_id(0))
    DATA = get_dataset(get_cache_id(0))
    TF_MAP = get_tf_map(get_cache_id(1))
    TASKS = {target.tname:target for task in _gen_task(get_cache_id(1)).values() for target in task}
    page = st.sidebar.selectbox('選擇頁面', PAGES)
    
    if page == PAGES[0]:
        containers = st.tabs([f'{bn}_{sn}' for bn, b in TABLES.items() for sn, _ in b.items()])
        _idx = 0
        for bname, book in TABLES.items():
            for sname, sheet in book.items():
                with containers[_idx]:
                    st.dataframe(sheet)
                _idx += 1
    elif page == PAGES[1]:
        _COL_NUM = 3
        labels = list(set([i.label for i  in FEATURES.values()]))
        label2s = list(set([i.label2 for i  in FEATURES.values()]))
        label3s = list(set([i.label3 for i  in FEATURES.values()]))
        sls = st.sidebar.multiselect('選擇類別一', labels)
        sl2s = st.sidebar.multiselect('選擇類別二', label2s)
        sl3s = st.sidebar.multiselect('選擇類別三', label3s)
        _feautres = FEATURES
        if sls:
            _feautres = {k:v for k, v in _feautres.items() if v.label in sls}
        if sl2s:
            _feautres = {k:v for k, v in _feautres.items() if v.label2 in sl2s}
        if sl3s:
            _feautres = {k:v for k, v in _feautres.items() if v.label3 in sl3s}
        _sf = st.sidebar.multiselect('選擇因子', _feautres)
        length = len(_sf)
        if length:
            for row in range((length//_COL_NUM)+1):
                cols = st.columns(_COL_NUM)
                _d_name = _sf[row*_COL_NUM:(row+1)*_COL_NUM]
                for idx, each_dn in enumerate(_d_name):
                    with cols[idx]:
                        _data = DATA.features[FEATURES[each_dn].key].series
                        st.dataframe(_data)
    elif page == PAGES[2]:
        st.markdown('## 現有對應關係檢視')
        col1, col2 = st.columns([1, 2])
        with col1:
            st.json(TF_MAP)
    elif page == PAGES[3]:
        st.markdown('## 執行分析項目')
        
        period = st.sidebar.selectbox('選擇領先期別', Periods.get_names(),index=4)
        col1, col2 = st.sidebar.columns(2)
        col1.button('產生報表', on_click=_handle_report_on_click)
        set_period(Periods.get(period))
        _reports = []
        tnames = [i for i in TASKS]
        containers = st.tabs(tnames)
        for idx, _task in enumerate(TASKS.values()):
            data = _get_corr(_task)
            with containers[idx]:
                st.dataframe(data)
            _reports[f'{period}\\{_task.tname}'] = data
        set_reports({**get_reports(), **_reports})
    elif page == PAGES[4]:
        targets = get_targets()
        if not targets:
            set_targets([i for i in TASKS])
        target = st.sidebar.selectbox('選擇標的', get_targets())
        col1, col2 = st.sidebar.columns(2)
        col1.button('產生報表', on_click=_handle_report_on_click)
        ret = []
        for period in Periods:
            set_period(period)
            srs = _get_corr(TASKS[target])['coef'].rename(f'{period.name}')
            ret.append(srs)
        data = pd.concat(ret, axis=1)
        info = [FEATURES[i]._asdict() for i in  data.index.values.tolist()]
        info = pd.DataFrame(info).set_index('name')
        data = pd.concat([info, data], axis=1)
        st.dataframe(data)
        set_reports({**get_reports(), **{target:data}})
    else:
        raise RuntimeError(f'unknown page {page}')
    

