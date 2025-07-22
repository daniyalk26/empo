import logging

from tempfile import SpooledTemporaryFile
from typing import Union, IO

import html2text
from .common import get_file_info_from_reader

log = logging.getLogger("doclogger")


def html_processor(stream: Union[IO[bytes], SpooledTemporaryFile], **kwargs):
    """Partitions Microsoft csv Documents in .csv format.

    Parameters
    ----------
    stream
        A string defining the target filename path.
        :param stream:
    """
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True

    response = stream.read().decode()
    # df = pd.read_csv(stream, header=None, encoding='unicode_escape')
    markdown_document = h.handle(response)
    meta_info = get_file_info_from_reader(stream)

    return meta_info, [markdown_document]


