from __future__ import annotations

import math
import statistics
import typing


def round_(x: float, bucket_size: float) -> float:
    """Bin `x` into buckets."""
    if math.isinf(x) or math.isnan(x):
        return x

    # NOTE: Without additional rounding, the results can sometimes be not exact: 1.1500000000000001
    ndigits = max(str(bucket_size)[::-1].find("."), 0)
    return round(bucket_size * round(x / bucket_size), ndigits=ndigits)


def mean(list_: typing.Iterable[float]) -> float:
    """Mean function that does not return an error if `list_` is empty."""
    list_ = list(list_)
    if len(list_) == 0:
        return math.nan
    return statistics.mean(list_)  # NOTE: `statistics.mean` returns an error for an empty list
