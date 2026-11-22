import typing

import polars as pl


# adapted from https://github.com/pola-rs/polars/issues/5159#issuecomment-1943839127
def union_upsample(
    df: pl.DataFrame,
    *,
    upsample: str,
    group_by: typing.Union[list, str],
    fill_null_ops: typing.Sequence = tuple(),
) -> pl.DataFrame:
    """Perform a union-based upsample by ensuring the union of unique upsample
    column values across group_by groups.

    Parameters
    ----------
    df : polars.DataFrame
        The input DataFrame.
    upsample : str
        The name of the column to upsample.
    group_by : str or Sequence[str]
        The column(s) to group by.
    fill_null_ops : Sequence[Tuple[polars.Expr, Dict[str, Any]]], optional
        A sequence of fill_null operations, where each element is a tuple of
        (target_expr, fill_null_kwargs).

        The `target_expr` is a Polars expression specifying the column to fill,
        and `fill_null_kwargs` is a dictionary of keyword arguments passed to
        Polars `fill_null`. Pass `pl.all()` for `target_expr` to fill all
        columns.

    Returns
    -------
    polars.DataFrame
        A new DataFrame with upsampled categories and optionally filled null values.
    """
    upsample_union = df[upsample].unique().sort().to_list()
    upsample_dtype = df[upsample].dtype

    if isinstance(group_by, str):
        group_by = [group_by]

    df = df.lazy()

    # upsample
    df = (
        df.select(*group_by, upsample)
        .group_by(group_by)
        .first()
        .with_columns(
            pl.lit(
                upsample_union,
                dtype=pl.List(upsample_dtype),
            ).alias(upsample)
        )
        .explode(upsample)
        .join(df, how="left", on=[*group_by, upsample])
    )

    # fill null
    for target, fill_null_kws in fill_null_ops:
        df = df.with_columns(
            target.fill_null(
                **fill_null_kws
            ).over(  # keeps sorted order within groups
                group_by,
            ),
        )

    return df.collect()
