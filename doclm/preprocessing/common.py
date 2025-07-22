import pandas as pd


def get_headers(data_frame):
    """
    return top most not nan row index
    :param data_frame: pandas data frame
    :return: index (int)
    """
    # var = data_frame.index[data_frame.notna().any(axis=1).dropna(axis=0)].tolist()
    var = data_frame.ffill().count(axis=1).drop_duplicates().index.tolist()
    cols = data_frame.loc[var].\
        transpose().ffill().transpose().values
    try:
        return var, [' '.join(i) for i in pd.MultiIndex.from_arrays(cols)]
    except ValueError :
        return var, data_frame.columns
    except TypeError:
        return var, data_frame.columns


def get_file_info_from_reader(reader):
    fields = ["author", "creator", "producer", "subject", "title"]
    info = {}
    if not reader or not hasattr(reader, "metadata") or not reader.metadata:
        for f in fields:
            info[f] = None
        info["pages"] = None
        return info
    info["pages"] = len(reader.pages) if hasattr(reader, "pages") else None

    return get_file_info_from_meta_data(reader.metadata, fields, info)


def get_file_info_from_meta_data(metadata, fields, info):
    meta = metadata
    for f in fields:
        info[f] = getattr(meta, f) if hasattr(meta, f) else None
    return info

