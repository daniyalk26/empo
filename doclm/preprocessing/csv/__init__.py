import logging

import pandas as pd
from tempfile import SpooledTemporaryFile
from typing import Optional, Union, IO

from ..common import get_headers, get_file_info_from_reader
log = logging.getLogger("doclogger")


def simple_processor(
        stream: Optional[Union[IO[bytes], SpooledTemporaryFile]] = None,
        **kwargs,
):
    """Partitions Microsoft csv Documents in .csv format.

    Parameters
    ----------
    stream
        A string defining the target filename path.
        :param stream:
    """

    df = pd.read_csv(stream, header=None, encoding='unicode_escape')

    meta_info = get_file_info_from_reader(stream)

    header_indices, cols = get_headers(df[:10])
    df.columns = cols
    df.drop(index=header_indices, inplace=True)
    csv_text = df.to_csv(na_rep="")

    return meta_info, [csv_text]
