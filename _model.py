# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import glob
from typing import Optional, List

import numpy as np

from common import _force_load

files = glob.glob('./_data/_cache/_tasks/S&P 500 INDEX/**/*.pkl', recursive=True)

tasks = {}
for f in files:
    name = f.split('\\')[-1][:-4]
    tasks[name] = _force_load(f)

class BaseTransformer(metaclass=ABCMeta):
    
    @abstractmethod
    def fit():...

    @abstractmethod
    def transform():...

    @abstractmethod
    def fit_transform():...


class Model:
    
    def __init__(self, X:np.ndarray, Y:np.ndarray, 
            x_transformer:Optional[List[BaseTransformer]]=None, 
            y_transformer:Optional[List[BaseTransformer]]=None):
        self._X = X
        self._Y = Y
        self._x_transformers = []
        self._y_transformers = []
    
    @property
    def x_transformers(self):
        return self._x_transformers

    @property
    def x_transformers(self):
        return self._x_transformers

    def compile(self,):
        ...

    def fit(self):
        ...
