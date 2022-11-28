# -*- coding: utf-8 -*-
from typing import List, Union, Dict
import pandas as pd
from _const import Period

_is_analysis = False

_targets = []

_reports = {}

_cache_0 = 0

_cache_1 = 0

_cache_2 = 0

_period = None

_tf_map_table = {}

_page_index = 0


def get_is_analysis() -> bool:
    global _is_analysis
    return _is_analysis


def set_is_analysis(state: bool) -> bool:
    global _is_analysis
    _is_analysis = state
    return _is_analysis


def get_targets() -> List[str]:
    global _targets
    return _targets.copy()


def set_targets(values: List[str]) -> List[str]:
    global _targets
    _targets = values


def clear_cache_id(which: int) -> int:
    global _cache_0, _cache_1, _cache_2
    if which < 3:
        _cache_2 += 1
    if which < 2:
        _cache_1 += 1
    if which < 1:
        _cache_0 += 1


def get_cache_id(which: int) -> int:
    global _cache_2, _cache_1, _cache_0
    return [_cache_0, _cache_1, _cache_2][which]


def get_reports() -> Dict[str, Union[pd.DataFrame, pd.Series]]:
    global _reports
    return _reports.copy()


def set_reports(values:Dict[str, Union[pd.DataFrame, pd.Series]]):
    global _reports
    _reports = values

def get_period() -> Union[Period, None]:
    global _period
    return _period

def set_period(value:Period):
    global _period
    _period = value
    return _period

def get_task_map_table() -> Dict[str, Dict[str, List[str]]]:
    global _tf_map_table
    return _tf_map_table

def set_task_map_table(value:pd.DataFrame):
    global _tf_map_table
    _tf_map_table = value

def get_page_index() -> int:
    global _page_index
    return _page_index

def set_page_index(value:int):
    global _page_index
    _page_index = value
