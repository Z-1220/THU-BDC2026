from __future__ import annotations

from copy import deepcopy

from qlib.data.dataset import DatasetH, TSDatasetH, TSDataSampler
from qlib.data.dataset.handler import DataHandlerLP


class TSDatasetHWithFill(TSDatasetH):
    def __init__(self, *args, fillna_type: str = "none", **kwargs):
        self.fillna_type = fillna_type
        super().__init__(*args, **kwargs)

    def _prepare_seg(self, slc, **kwargs):
        if not isinstance(slc, slice):
            slc = slice(*slc)
        dtype = kwargs.pop("dtype", None)
        fillna_type = kwargs.pop("fillna_type", self.fillna_type)
        if (flt_col := kwargs.pop("flt_col", None)) is None:
            flt_col = self.flt_col

        ext_slice = self._extend_slice(slc, self.cal, self.step_len)
        data = DatasetH._prepare_seg(self, ext_slice, **kwargs)

        flt_kwargs = deepcopy(kwargs)
        if flt_col is not None:
            flt_kwargs["col_set"] = flt_col
            flt_data = DatasetH._prepare_seg(self, ext_slice, **flt_kwargs)
            assert len(flt_data.columns) == 1
        else:
            flt_data = None

        tsds = TSDataSampler(
            data=data,
            start=slc.start,
            end=slc.stop,
            step_len=self.step_len,
            fillna_type=fillna_type,
            dtype=dtype,
            flt_data=flt_data,
        )
        return tsds

    def __setstate__(self, state: dict):
        self.__dict__.update(state)
        if not hasattr(self.handler, "_infer"):
            self.handler.setup_data(init_type=DataHandlerLP.IT_FIT_SEQ)
