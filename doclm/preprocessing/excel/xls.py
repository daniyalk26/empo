import logging
import yaml
import pandas as pd
from tempfile import SpooledTemporaryFile
from typing import List, Optional, Union, IO

from ..common import get_headers, get_file_info_from_reader

log = logging.getLogger("doclogger")


def xls_processor(stream,**kwargs,
):
    """Partitions Microsoft Excel Documents in .xls format into its document elements.

    """

    try:
        sheets = pd.read_excel(stream, sheet_name=None, header=None)
    except ValueError:
        log.warning("Excel file format cannot be determined, using 'openpyxl' as default")
        sheets = pd.read_excel(stream, sheet_name=None, header=None, engine='openpyxl')

    meta_info = get_file_info_from_reader(stream)
    meta_info['pages'] = len([1 for _, df in sheets.items() if not df.empty])
    return meta_info, extract_sheets(sheets)


def extract_sheets(sheets):
    for sheet_name, df in sheets.items():
        header_indices, cols = get_headers(df[:10])
        df.columns = cols
        log.debug('extracting sheet %s', sheet_name)

        if df.empty:
            log.warning('%s is empty', sheet_name)
            continue
        df.drop(index=header_indices, inplace=True)
        csv_text = df.to_csv(na_rep="", index=False)
        # header_loc = df[df == 'Row Labels'].dropna(axis=1, how='all').dropna(how='all')
        # text = lxml.html.document_fromstring(html_text).text_content()
        yield sheet_name,  yaml.dump([csv_text])

