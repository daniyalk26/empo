import re
import logging
import itertools
import unicodedata
from copy import deepcopy
from typing import Iterable
from functools import lru_cache
from collections import ChainMap

import numpy as np
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_pages
from langchain_core.documents import Document

# from concurrent.futures import ThreadPoolExecutor
from .util import get_file_info_from_reader

log = logging.getLogger("doclogger")
log.disabled = False


# pylint: disable=invalid-name,

def lev_dist(a, b):
    @lru_cache(None)  # for memorization
    def min_dist(s1, s2):
        if s1 == len(a) or s2 == len(b):
            return len(a) - s1 + len(b) - s2

        # no change required
        if a[s1] == b[s2]:
            return min_dist(s1 + 1, s2 + 1)

        return 1 + min(
            min_dist(s1, s2 + 1),  # insert character
            min_dist(s1 + 1, s2),  # delete character
            min_dist(s1 + 1, s2 + 1),  # replace character
        )

    return min_dist(0, 0)


def ham_dist(s1, s2):
    if len(s1) != len(s2):
        raise ValueError("Undefined")
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))


def search_min_dist_substr(source, search, distance_func=lev_dist):
    l = len(search)
    index = 0
    min_dist = l
    # min_substring = source[:l]
    for i in range(len(source) - l + 1):
        if source[i] == search[0]:
            d = distance_func(search, source[i : i + l])
            if d < min_dist:
                min_dist = d
                index = i
                # min_substring = source[i: i + l]
            if d < 3:
                break
    return index, min_dist, search


def find_fonts_for_document(page, page_num):
    def find_font_recursively(
        current_obj, parent_obj="root", text_lines=None, page_num=page_num
    ):
        # try:
        #     log.debug(parent_obj.get_text())
        # except:
        #     log.debug(parent_obj)
        # log.debug(f"current object {current_obj}, parent object {parent_obj}")
        # TODO: improve this function
        if (
            (not isinstance(current_obj, Iterable))
            and hasattr(current_obj, "fontname")
            and hasattr(current_obj, "size")
            and hasattr(parent_obj, "get_text")
            and len(current_obj.get_text().strip()) > 0
        ):
            if all(
                True if unicodedata.category(c) not in {"Co"} else False
                for c in current_obj.get_text()
            ):
                # if unicodedata.category(current_obj.get_text()) not in {'Co'}:
                # TODO: improve this line
                line_text = re.sub(r"\s+", " ", parent_obj.get_text().strip())
                if (line_text, page_num) in text_lines:
                    text_lines[(line_text, page_num)].append(
                        (current_obj.size, current_obj.get_text())
                    )
                else:
                    text_lines[(line_text, page_num)] = [
                        (current_obj.size, current_obj.get_text())
                    ]

        if isinstance(current_obj, Iterable):
            for child_obj in current_obj:
                find_font_recursively(
                    child_obj,
                    parent_obj=current_obj,
                    text_lines=text_lines,
                    page_num=page_num,
                )

    # pages = extract_pages(file_path)
    text_lines = {}
    # page_num = 0
    # for page in pages:
    #     page_num += 1
    #     # print(page_num)
    find_font_recursively(page, page_num=page_num, text_lines=text_lines)
    return text_lines


def get_headings_from_text_font_size_dict(font_size_dict, font_size_index):
    """
    keys of font_size_dict are supposed to contain heading text and page number tuple
    :param font_size_dict: dict fot tuple
    :param font_size_index:
    :return: list of keys of font_size_dict which meet the heading level criteria
    """

    text_font_size_dict = deepcopy(font_size_dict)
    font_sizes_sorted = sorted(
        list(
            set(
                    char_details[0]
                    for text_line in text_font_size_dict
                    for char_details in text_font_size_dict[text_line]
            )
        ),
        reverse=True,
    )

    if isinstance(font_size_index, int):
        text_lines_keys = text_font_size_dict.keys()
        for text_line in text_lines_keys:
            text_font_size_dict[text_line] = [
                char_details
                for char_details in text_font_size_dict[text_line]
                if char_details[0]
                >= font_sizes_sorted[min(font_size_index, len(font_sizes_sorted) - 1)]
            ]
        headings = {
            (k[0], k[1], font_sizes_sorted.index(v[0][0]) + 1): v
            for k, v in text_font_size_dict.items()
            if (len(v) != 0)
            and (unicodedata.category(k[0][0]) not in {"Ll", "Ps", "P"})
            and (3 < len(k[0]) < 80)  # Avoid small-case characters as starting char
        }
        # print(headings)
    headings_count = {}
    candidate_headings_list = list(headings.keys())
    for heading, _, _ in candidate_headings_list:
        if heading in headings_count:
            headings_count[heading] += 1
        else:
            headings_count[heading] = 1
    return [
        heading_tuple
        for heading_tuple in candidate_headings_list
        # Just to avoid title from the header from repeating
        if headings_count[heading_tuple[0]] <= 3
    ]


def pypdf_extract_text(page):
    return page.extract_text()


def filter_headings_list_relative_to_num_pages(headings_levels_list, num_pages):
    filter_criteria = 3 * num_pages
    for level, headings_list in enumerate(headings_levels_list):
        if level == 0 and len(headings_list) > filter_criteria:
            return [], -1  # no headings found case
        if len(headings_list) > filter_criteria:
            return headings_levels_list[level - 1], level

    return headings_levels_list[-1], len(headings_levels_list)


def find_headings_offset_in_page(page_text, page_headings_list):
    headings_list_level = []
    for heading, heading_level, page_num in page_headings_list:
        off_set, min_dist, min_substring = search_min_dist_substr(page_text, heading)
        if min_dist < 3:
            headings_list_level.append(
                (page_num, off_set, min_dist, heading_level, min_substring)
            )
    return headings_list_level


def get_text_against_headings(headings_offset_pagewise, pages):
    len_of_each_page = []
    len_of_each_page_cum_sum = []
    headings_text = []
    num_of_headings = len(headings_offset_pagewise)
    # if num_of_headings == 0:
    #     return pages, list(range(1,len(pages)+1))
    combined_doc = ""
    for page in pages:
        combined_doc += page
        len_of_each_page.append(len(page))
        len_of_each_page_cum_sum.append(len(combined_doc))

    section_end_page = []
    for heading_idx, headings_tuple in enumerate(headings_offset_pagewise):
        page_num, page_off_set, _, _, _ = headings_tuple
        doc_offset = (
            page_off_set
            if page_num == 1
            else len_of_each_page_cum_sum[page_num - 2] + page_off_set
        )
        if heading_idx + 1 != num_of_headings:
            next_headings_tuple = headings_offset_pagewise[heading_idx + 1]
            next_page_num, next_page_off_set, _, _, _ = next_headings_tuple
            next_doc_offset = (
                next_page_off_set
                if next_page_num == 1
                else len_of_each_page_cum_sum[next_page_num - 2] + next_page_off_set
            )

            headings_text.append(combined_doc[doc_offset:next_doc_offset])
            section_end_page.append(next_page_num)
        else:
            headings_text.append(combined_doc[doc_offset:])
            section_end_page.append(len(pages))
    # print(sum([len(heading_text) for heading_text in headings_text]))
    return headings_text, section_end_page, len_of_each_page


def merge_headings_text(headings_offset_pagewise, headings_texts):
    num_lines_in_section_texts = [text.count("\n") for text in headings_texts]
    num_of_headings = len(headings_offset_pagewise)
    is_next_level_bigger = [0] * len(headings_offset_pagewise)
    is_section_line_count_large = [0] * len(headings_offset_pagewise)

    for heading_idx, heading_tuple in enumerate(headings_offset_pagewise):
        _, _, _, heading_level, heading, _ = heading_tuple
        is_section_line_count_large[heading_idx] = (
            1 if num_lines_in_section_texts[heading_idx] >= 4 else 0
        )
        if heading_idx + 1 != num_of_headings:
            is_next_level_bigger[heading_idx] = (
                1 if heading_level > headings_offset_pagewise[heading_idx + 1][3] else 0
            )
    is_mergable_with_next_heading = [
        i or j for i, j in zip(is_next_level_bigger, is_section_line_count_large)
    ]
    is_mergable_with_next_heading = [
        is_mergable_with_next_heading[0]
    ] + is_mergable_with_next_heading[:-1]
    is_mergable_with_next_heading = np.cumsum(is_mergable_with_next_heading).tolist()

    _headings_tuples = []
    _headings_texts = []

    for heading_num in sorted(list(set(is_mergable_with_next_heading))):
        max_idx = max(
            index
            for index, _heading_num in enumerate(is_mergable_with_next_heading)
            if _heading_num == heading_num
        )
        min_idx = min(
            index
            for index, _heading_num in enumerate(is_mergable_with_next_heading)
            if _heading_num == heading_num
        )
        _heading_text = "".join(
            list(text for text in headings_texts[min_idx : max_idx + 1])
        )
        _page_num = headings_offset_pagewise[min_idx][0]
        _off_set = headings_offset_pagewise[min_idx][1]
        _min_dist = max(
            heading_tuple[2]
            for heading_tuple in headings_offset_pagewise[min_idx : max_idx + 1]
        )
        _level = min(
            heading_tuple[3]
            for heading_tuple in headings_offset_pagewise[min_idx : max_idx + 1]
        )
        _headings = "\n".join(
            [
                heading_tuple[4]
                for heading_tuple in headings_offset_pagewise[min_idx : max_idx + 1]
            ]
        )
        _section_end_page = headings_offset_pagewise[max_idx][5]

        _headings_tuples.append(
            (_page_num, _off_set, _min_dist, _level, _headings, _section_end_page)
        )
        _headings_texts.append(_heading_text)
    return _headings_tuples, _headings_texts


def segregate_section_to_pagewise(
    section_texts, headings_offset_pagewise, len_of_each_page
):
    _section_texts = []
    for heading_idx, heading_tuple in enumerate(headings_offset_pagewise):
        # print(f"heading number{heading_idx}")
        # print(
        #     f"section heading:{heading_tuple[-2]}, start page:{heading_tuple[0]}, end page:{heading_tuple[-1]}"
        # )
        texts = []

        if heading_tuple[-1] == heading_tuple[0]:
            texts.append(section_texts[heading_idx])
        else:
            offset = heading_tuple[1]
            for idx, section_page_num in enumerate(
                range(heading_tuple[0], heading_tuple[-1] + 1)
            ):
                if idx == 0:
                    texts.append(
                        section_texts[heading_idx][
                            0 : len_of_each_page[section_page_num - 1] - offset
                        ]
                    )
                    offset = len_of_each_page[section_page_num - 1] - offset
                else:
                    texts.append(
                        section_texts[heading_idx][
                            offset : len_of_each_page[section_page_num - 1] + offset
                        ]
                    )
                    offset = len_of_each_page[section_page_num - 1] + offset
        _section_texts.append(texts)
    return _section_texts


def structured_doc_processor(stream, file_path, initial_num_pages, source_tag,**kwargs):
    log.info(" extracting text form PDF_reader file %s", file_path)

    (
        meta_info,
        initial_page_text,
        headings_offset_pagewise,
        section_texts,
    ) = structured_parser(stream, initial_num_pages)
    docs = []
    # format of tuples in headings_offset_pagewise
    # page_num, off_set, min_dist, heading_level, heading_text, section_end_page
    # if len(section_texts):
    for section_num, texts in enumerate(section_texts):
        for page_offset, text in enumerate(texts):
            if len(text):
                doc = Document(
                    page_content=text,
                    metadata={
                        # source_tag: file_path,
                        "section_start_page": headings_offset_pagewise[section_num][0],
                        "page": headings_offset_pagewise[section_num][0] + page_offset,
                        "heading_level": headings_offset_pagewise[section_num][3],
                        "heading": headings_offset_pagewise[section_num][4],
                        "section_end_page": headings_offset_pagewise[section_num][5],
                    },
                )
                docs.append(doc)
    return initial_page_text, docs, meta_info


def structured_parser(file_path, initial_num_pages):
    # pool_obj = multiprocessing.Pool(os.cpu_count())
    # log.info("inside parser, created worker pool successfully")

    try:
        pdfminer_pages = extract_pages(file_path)
        pypdf_reader = PdfReader(file_path)
        pypdf_pages = pypdf_reader.pages
        num_pages = len(pypdf_pages)
        # _pypdf_pages = pool_obj.map(pypdf_extract_text, pypdf_pages)
        _pypdf_pages = []
        for page in pypdf_pages:
            _pypdf_pages.append(pypdf_extract_text(page))
        pypdf_pages = _pypdf_pages

        # text_lines = pool_obj.starmap(
        #     find_fonts_for_document, zip(pdfminer_pages, range(1, num_pages + 1))
        # )
        text_lines = []
        for page, idx in zip(pdfminer_pages, range(1, num_pages + 1)):
            text_lines.append(find_fonts_for_document(page, idx))

        log.info("found fonts from document successfully")
        text_lines = dict(ChainMap(*text_lines))

        # heading, page_num, level
        # headings_list_levels_list = pool_obj.starmap(
        #     get_headings_from_text_font_size_dict,
        #     [(text_lines, font_level_index) for font_level_index in range(1, 10)],
        # )
        headings_list_levels_list = []
        for font_level_index in range(1, 10):
            headings_list_levels_list.append(
                get_headings_from_text_font_size_dict(text_lines, font_level_index)
            )

        log.info("got headings based on font size")
        log.debug("headings_list_levels_list: %s", headings_list_levels_list)
        # heading, page_num, level
        ################NOT SORTED
        headings_page_list, _ = filter_headings_list_relative_to_num_pages(
            headings_list_levels_list, len(pypdf_pages)
        )
        log.info(
            "filtered headings successfully, num of filtered headings %s",
            len(headings_page_list),
        )
        log.debug("headings_page_list: %s", headings_page_list)

        # page_num, off_set, min_dist, heading_level, min_substring
        # headings_offset_pagewise = pool_obj.starmap(
        #     find_headings_offset_in_page,
        #     [
        #         (
        #             page,
        #             [
        #                 (heading, heading_level, page_num)
        #                 for heading, page_num, heading_level in headings_page_list
        #                 if page_num == page_idx + 1
        #             ],
        #         )
        #         for page_idx, page in enumerate(pypdf_pages)
        #     ],
        # )
        headings_offset_pagewise = []
        for page_idx, page in enumerate(pypdf_pages):
            for heading, page_num, heading_level in headings_page_list:
                if page_num == page_idx + 1:
                    headings_offset_pagewise.append(
                        find_headings_offset_in_page(
                            page, [(heading, heading_level, page_num)]
                        )
                    )

        log.info("Found heading offset in document successfully")
        log.debug("%s", headings_offset_pagewise)
        headings_offset_pagewise = list(
            itertools.chain.from_iterable(headings_offset_pagewise)
        )

        headings_offset_pagewise = sorted(
            headings_offset_pagewise, key=lambda x: (x[0], x[1])
        )
        if len(headings_offset_pagewise):
            headings_offset_pagewise.insert(
                0, (1, 0, 0, 1, "")
            )  # to handle initial text before the first heading
            (
                section_texts,
                section_end_page,
                len_of_each_page,
            ) = get_text_against_headings(headings_offset_pagewise, pypdf_pages)
            log.info("Got section wise texts")

            # page_num, off_set, min_dist, heading_level, min_substring, section_end_page
            headings_offset_pagewise = [
                (i[0], i[1], i[2], i[3], i[4], j)
                for i, j in zip(headings_offset_pagewise, section_end_page)
            ]

            headings_offset_pagewise, section_texts = merge_headings_text(
                headings_offset_pagewise, section_texts
            )
            log.info("merged small sections together")
            assert sum(len(text) for text in section_texts) == sum(
                len_of_each_page
            ), AssertionError("Bug in section text extraction")
            _section_texts = segregate_section_to_pagewise(
                section_texts, headings_offset_pagewise, len_of_each_page
            )
        else:
            log.warning("failed to find any heading using structured parser")
            raise Exception("failed to find any heading using structured parser")
        # pool_obj.close()
        initial_page_text = "".join(pypdf_pages[:initial_num_pages])
        meta_info = get_file_info_from_reader(pypdf_reader)
        return meta_info, initial_page_text, headings_offset_pagewise, _section_texts
    except Exception as e:
        log.error("Exception occurred", exc_info=True)
        # pool_obj.close()
        raise ValueError(e) from e
    finally:
        # pool_obj.close()
        pass


# if __name__ == "__main__":
#     file_path = "/home/in01-nbk-741/Downloads/taimoor1/cpk 1.pdf"
#     initial_num_pages = 3
#     structured_parser(file_path, initial_num_pages)
