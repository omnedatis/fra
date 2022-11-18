# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
from typing import Optional

from common import Period

class TransformerBase(metaclass=ABCMeta):

    @abstractmethod
    def transform(self):
        ...

class PeriodTransformer(TransformerBase):

    def __init__(self, period:Period):
        self._period = period

    def transform(self, data:pd.Series):
        if self._period is None:
            return data
        return data.shift(-self._period.steps)


class SingleCorrModel:
    
    def __init__(self, *, x_transformer:Optional[TransformerBase]=None,
            y_transformer:Optional[TransformerBase]=None):
        self._x_transformer = x_transformer
        self._y_transformer = y_transformer
    
    @property
    def x_transformers(self):
        return self._x_transformer

    @property
    def y_transformers(self):
        return self._y_transformer

    def _fit(self, x_data:pd.Series, y_data:pd.Series) -> pd.DataFrame:
        if self._x_transformer is not None:
            x_data = self._x_transformer.transform(x_data)
        if self._y_transformer is not None:
            y_data = self._y_transformer.transform(y_data)
        data = pd.concat([x_data, y_data], axis=1, sort=True)
        data.iloc[:,1] = data.iloc[:,1].ffill()
        return data.dropna()

    def get_corr(self, x_data:pd.Series, y_data:pd.Series):
        data = self._fit(x_data, y_data)
        return np.corrcoef(data.values.T)