# -*- coding: utf-8 -*-

_is_analysis = False

def get_is_analysis() -> bool:
    global _is_analysis
    return _is_analysis

def set_is_analysis(state:bool) -> bool:
    global _is_analysis
    _is_analysis = state
    return _is_analysis
    