import typing

import polars as pl


def _check_and_squeeze(lst: typing.List[typing.Any]) -> typing.Any:
    if len(lst) == 0:
        raise ValueError("No items to squeeze!")
    if len(lst) > 1:
        raise ValueError(f"Expected exactly one item, got {len(lst)}.")
    return lst[0]


def squeeze_int(expr: pl.Expr) -> pl.Expr:
    """Returns a Polars expression that:
    1) checks that there is exactly one unique value,
    2) returns that single value,
    3) raises ValueError otherwise.
    """
    return expr.map_batches(
        lambda series: _check_and_squeeze(series.to_list()),
        return_dtype=int,
        returns_scalar=True,
    )


def squeeze_str(expr: pl.Expr) -> pl.Expr:
    """Returns a Polars expression that:
    1) checks that there is exactly one unique value,
    2) returns that single value,
    3) raises ValueError otherwise.
    """
    return expr.map_batches(
        lambda series: _check_and_squeeze(series.to_list()),
        return_dtype=str,
        returns_scalar=True,
    )
