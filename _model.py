# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
from typing import Optional

from common import Period, Feature, ONE_DAY

class TransformerBase(metaclass=ABCMeta):

    @abstractmethod
    def transform(self):
        ...

class PeriodTransformer(TransformerBase):

    def __init__(self, period:Period):
        self._period = period

    def transform(self, data:pd.Series) -> pd.Series:
        if self._period is None:
            return data
        data.index = data.index - pd.Timedelta(self._period.steps, 'd')
        return data


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

    def _fit(self, x_data:Feature, y_data:Feature) -> pd.DataFrame:
        x_info, y_info = x_data.cinfo, y_data.cinfo
        x_data, y_data = x_data.series.copy(), y_data.series.copy()
        if self._x_transformer is not None:
            x_data = self._x_transformer.transform(x_data)
        if self._y_transformer is not None:
            y_data = self._y_transformer.transform(y_data)
        data:pd.DataFrame = pd.concat([x_data, y_data], axis=1, sort=True)
        if x_info.freq != ONE_DAY.unit:
            data.iloc[:,1] = data.iloc[:,1].ffill()
        return data.dropna()

    def get_corr(self, x_data:Feature, y_data:Feature):
        data = self._fit(x_data, y_data)
        return np.corrcoef(data.values.T)