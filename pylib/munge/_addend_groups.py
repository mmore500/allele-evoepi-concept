import typing

import polars as pl


def addend_groups(
    df: pl.DataFrame,
    group_by: typing.Union[typing.Sequence[str], str],
    aggs: typing.Sequence[pl.Expr] = (pl.all().last(),),
    *,
    inner_only: bool = False,
):
    """Add a row to each group in a DataFrame with values set via user-defined
    aggregation.

    Groups a DataFrame by specified columns, increments a designated time
    column, and concatenates the resulting rows with the original DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame.
    group_by : str or sequence of str
        Column(s) used to define groups to be addended.
    aggs : sequence of any, optional
        Aggregations to apply on each group, to set values in addended row.

        Use `(pl.col(time_column) + 1).max().alias(time_column)`, for example,
        to set time in the addended row to one past from the last observed time.

        Use `pl.all().last()` to set all columns to the last observed value.
        Use `pl.exclude(time_column).last()` to set all columns except time to
        the last observed value, to prevent a duplicate aggregation error.

        Defaults to the last value in all columns.
    inner_only : bool, defaultFalse
        If True, only return rows from the original DataFrame that are also in
        the agggregation result, by default False.

    Returns
    -------
    pl.DataFrame
        A concatenated DataFrame containing the original rows
        and the newly created rows with the incremented time column.

    Notes
    -----
    Order within groups is preserved by polars (i.e., if using `first` or
    `last`).
    """
    addendum = df.group_by(group_by).agg(*aggs)

    if inner_only:
        df = df.select(addendum.columns)

    return pl.concat([df, addendum], how="diagonal_relaxed")
