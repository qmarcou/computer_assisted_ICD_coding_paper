"""Some utility functions to wrap around base panda."""
import copy
import sys
from pathlib import Path
from _collections_abc import Iterable
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def read_filter_by_chunk(read_func, filepath_or_buffer, filter_func,
                         chunksize, *args, **kwargs):
    """
    Read data by chunk using Pandas and apply a filter function on each chunk.

    Pandas does not provide a way to filter rows based on data characteristics
    upon reading. This can be very inconvenient upon working with large files
    since out of the box Pandas will load the whole file into memory before
    being able to filter rows, end my end up saturating computer memory while
    in the end only end a small fraction of the rows was needed.
    This function is a wrapper around Pandas file reading function and allows
    to read file by chunk and apply a rows filtering operation on each chunk.

    Parameters:
    read_func : Callable
            a Pandas reading function
    filepath_or_buffer : str, path object or file-like object
            Same as required for Pandas.read_* function
    filter_func : Callable
            a function taking a Pandas Dataframe as input and
            and returning a Dataframe with the same columns but possibly
            different number of rows. No check is made about columns
            consistency, and filtered chunks are concatenated via the
            Pandas.concat method which does not enforce columns consistency and
            could result into
    chunksize : int
            The size of the parsed chunks. The size of the chunk will incluence
            the dtypes inferred by Pandas, it is thus advised to specify dtypes
            for the reading function to prevent erratic behavior of the
            filtered function.
    *args and **kwargs: positional and keywords arguments for the reading
     function

    Returns:
    Pandas.Dataframe: a Dataframe with rows filtered based on the filter_func
    function

    """

    first_chunk = True
    with read_func(filepath_or_buffer, *args, **kwargs,
                   chunksize=chunksize) as reader:
        for chunk in reader:
            if first_chunk:
                final_df = filter_func(chunk)
                first_chunk = False
            else:
                tmp_df = filter_func(chunk)
                final_df = pd.concat(
                    [final_df, tmp_df], ignore_index=True, sort=False)
    return final_df


def read_filter_csv_by_chunk(filepath_or_buffer, filter_func, chunksize,
                             *args, **kwargs):
    """Read and filter csv file. See read_filter_by_chunk for details."""
    from pandas import read_csv
    return read_filter_by_chunk(read_csv, filepath_or_buffer, filter_func,
                                chunksize, *args, **kwargs)


def read_csv_defaultdtype(filepath_or_buffer, defaultdtype,
                          dtypes_dict, *args, **kwargs):
    """Call pd.read_csv with a default dtype overriden by dtypes_dict."""
    # Make a first minimal read to get the columns names
    # TODO nrows=max(header)
    kwargs_copy = kwargs.copy()
    kwargs_copy.pop("nrows", None)
    kwargs_copy.pop("engine", None)
    file_cols = pd.read_csv(filepath_or_buffer, nrows=1,
                            engine='python', *args,
                            **kwargs_copy).columns
    # Introduce default values when not specified in the dtypes_dict
    for colname in file_cols:
        if not dtypes_dict.__contains__(colname):
            dtypes_dict[colname] = defaultdtype
    return pd.read_csv(filepath_or_buffer, dtype=dtypes_dict, *args, **kwargs)


def lists_to_rows(data: pd.DataFrame, colname: str):
    # work on a new index in there's already duplicated row indices
    init_cols = data.columns
    init_index_names = [name for name in data.index.names]
    if len(init_index_names) == 1:
        init_index_names = init_index_names[0]
    data = data.reset_index()
    new_df = pd.merge(data.drop(columns=colname),
                      pd.DataFrame(data[colname].to_list()).stack().droplevel(
                          1).squeeze().rename(colname),
                      right_index=True, left_index=True)
    # Reset index to the original one with possibly more duplicates now
    # Use the old index columns
    new_df.set_index(
        keys=new_df.columns[~new_df.columns.isin(init_cols)].to_list(),
        inplace=True)
    # rename in case the old index was not named or name was
    # already taken by another col
    new_df.index.rename(init_index_names,
                        inplace=True)
    # Reorder columns to the original order before return
    return new_df[init_cols]

# def list_to_rows(data, cols: str | Iterable[str]):
#     """Expand a list column to individual rows."""
#     # from https://ianh6ll6n.medium.com/expanding-a-pandas-list-column-to-rows-41c69aaf9488
#     import numpy as np
#     orig_cols = data.columns
#     if isinstance(cols, str):
#         cols = [cols]
#     for col in cols:
#         item_col = f"{col}_list_item$$"
#         ren_col = {item_col: col}
#         data = (
#             pd.DataFrame(data[col].to_list())
#                 .replace([None], np.nan)  # convert any Nones to NaN
#                 .merge(data, right_index=True, left_index=True)
#                 .melt(id_vars=orig_cols, value_name=item_col)
#                 .dropna(subset=[item_col])  # get rid of rows with NaNs
#                 # drop our unneeded src column plus the melt "variable"
#                 .drop([col, "variable"], axis=1)
#                 .rename(columns=ren_col)
#         )
#     return data
