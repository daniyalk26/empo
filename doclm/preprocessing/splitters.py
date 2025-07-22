import io
from copy import deepcopy
from typing import List

import numpy as np
import pandas as pd
import yaml
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter


def custom_uniform_splitter(in_docs, chunk_size, chunk_overlap):
    metadata_list = [doc.metadata for doc in in_docs]
    in_len_of_each_page_content = [len(doc.page_content) for doc in in_docs]
    in_total_characters=sum(in_len_of_each_page_content)
    out_len_of_each_doc_content=[]
    while in_total_characters:
        if in_total_characters>chunk_size-chunk_overlap:
            out_len_of_each_doc_content.append(chunk_size-chunk_overlap)
            in_total_characters=in_total_characters-(chunk_size-chunk_overlap)
        else:
            out_len_of_each_doc_content.append(in_total_characters)
            in_total_characters=0
    
    in_ending_offset_each_page_content = []
    in_page_char_counter = 0
    for in_len_page_content in in_len_of_each_page_content:
        in_page_char_counter += in_len_page_content
        in_ending_offset_each_page_content.append(in_page_char_counter)
    in_starting_offset_each_page_content = [0]+in_ending_offset_each_page_content[:-1]
    
    out_ending_char_offset_doc_content = []
    out_doc_char_counter = 0
    for out_len_doc_content in out_len_of_each_doc_content:
        out_doc_char_counter += out_len_doc_content
        out_ending_char_offset_doc_content.append(out_doc_char_counter)
    out_starting_char_offset_doc_content = [0]+out_ending_char_offset_doc_content[:-1]
    out_ending_char_offset_doc_content = [doc_content_length+chunk_overlap for doc_content_length in out_ending_char_offset_doc_content[:-1]]+[out_ending_char_offset_doc_content[-1]]
    out_len_of_each_doc_content = [doc_content_length+chunk_overlap for doc_content_length in out_len_of_each_doc_content[:-1]]+[out_len_of_each_doc_content[-1]]
    
    combined_document = "".join([doc.page_content for doc in in_docs])
    out_docs=[
        Document(page_content=combined_document[out_start_offset:out_end_offset], metadata={}) for out_start_offset, out_end_offset in zip(out_starting_char_offset_doc_content,out_ending_char_offset_doc_content)
        ]
    
    out_starting_page_offset_doc_content=[]
    current_in_doc_idx = 0
    for out_doc_idx, out_doc_start_offset in enumerate(out_starting_char_offset_doc_content):
        for in_doc_idx in range(current_in_doc_idx,len(in_docs)):
            if out_doc_start_offset<in_ending_offset_each_page_content[in_doc_idx]:
                out_docs[out_doc_idx].metadata = deepcopy(metadata_list[in_doc_idx])
                out_starting_page_offset_doc_content.append(in_doc_idx)
                break
            else:
                current_in_doc_idx+=1
    
    out_docs_page_breaks=[]
    for out_doc_idx, (out_doc_start_offset,out_doc_end_offset) in enumerate(zip(out_starting_char_offset_doc_content,out_ending_char_offset_doc_content)):
        out_docs_page_break=[]
        for in_doc_idx in range(out_starting_page_offset_doc_content[out_doc_idx],len(in_docs)):
            if (in_ending_offset_each_page_content[in_doc_idx]>=out_doc_end_offset):
                out_docs_page_breaks.append(out_docs_page_break)
                out_docs[out_doc_idx].metadata['page_breaks_char_offset']=out_docs_page_break
                break
            else:
                out_docs_page_break.append(in_ending_offset_each_page_content[in_doc_idx]-out_doc_start_offset)


    return out_docs


def custom_uniform_structured_splitter(in_docs, chunk_size, chunk_overlap):
    out_docs = []
    if len(in_docs)==0:
        return out_docs
    section_grouped_docs = []
    heading = in_docs[0].metadata["heading"]
    for doc in in_docs:
        if doc.metadata["heading"] == heading:
            section_grouped_docs.append(doc)
        else:
            out_docs += custom_uniform_splitter(
                section_grouped_docs, chunk_size, chunk_overlap
            )
            section_grouped_docs = [doc]
            heading = doc.metadata["heading"]
    out_docs += custom_uniform_splitter(section_grouped_docs, chunk_size, chunk_overlap)
    return out_docs


def token_splitter_wrapper(in_docs, chunk_size, chunk_overlap):
    text_split = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # separators=["\n\n", "\n\r", "\r\n", "\r\r", " ", ""],
    )
    out_docs = text_split.split_documents(in_docs)
    return out_docs


def recursive_splitter_wrapper(in_docs, chunk_size, chunk_overlap):
    text_split = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n\r", "\r\n", "\r\r", " ", ""],
    )
    out_docs = text_split.split_documents(in_docs)
    return out_docs


def _overlap_between_strings(current_str, next_str):
    for ele in range(1, len(current_str)):
        if next_str.startswith(current_str[ele:]):
            return current_str[ele:]
    return ""

# def _overlap_between_strings(current_str, next_str):
#     for ele in range(1, len(current_str)):
#         if next_str.startswith(current_str[ele:]):
#             return current_str[ele:]
#     return ""

def markdown_splitter_wrapper(docs: List[Document], chunk_size: int, chunk_overlap):
    """
    splits list of document objects into smaller chunk sized documents
    :param docs: List[Documents]
    :param chunk_size: int size of maximum allowable chunk
    :return: List[Documents]
    """
    headers_to_split_on = [
        ("#", "heading1"),
        ("##", "heading2"),
        ("###", "heading3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, return_each_line=False)
    dock_chunks = []

    assert len(docs) == 1, 'check html processor getting multiple markdowns'

    for document in docs:
        split_docs = markdown_splitter.split_text(document.page_content)
        for docx in split_docs:
            docx.metadata.update(document.metadata)
            docx.metadata['heading'] = '\n'.join([
                docx.metadata.get(h) for h in ['heading1', 'heading2', 'heading3']
                if docx.metadata.get(h)])

        final_split_docs = custom_uniform_structured_splitter(split_docs, chunk_size, chunk_overlap)
        additional_processing(final_split_docs)
        dock_chunks.extend(final_split_docs)
    return dock_chunks


def additional_processing(docs):
    for idx, doc in enumerate(docs):
        doc.metadata["chunk_num"] = idx + 1
        doc.metadata["chunk_len_in_chars"] = len(doc.page_content)
        # if (idx > 0) and (
        #     doc.metadata.get("heading", "") == docs[idx - 1].metadata.get("heading", "")
        # ):
        #     overlap_length = len(
        #         _overlap_between_strings(docs[idx - 1].page_content, doc.page_content)
        #     )
        #     doc.metadata["previous_chunk_text"] = docs[idx - 1].page_content[
        #         : len(docs[idx - 1].page_content) - overlap_length
        #     ]
        # if (idx + 1 < len(docs)) and (
        #     doc.metadata.get("heading", "") == docs[idx + 1].metadata.get("heading", "")
        # ):
        #     overlap_length = len(
        #         _overlap_between_strings(doc.page_content, docs[idx + 1].page_content)
        #     )
        #     doc.metadata["next_chunk_text"] = docs[idx + 1].page_content[
        #         overlap_length:
        #     ]
    return docs


def extract_split_indexes(doc_page, chunk_size):
    sums = doc_page.applymap(lambda x: len(str(x))) \
               .sum(axis=1) \
               .cumsum() / chunk_size
    return np.ceil(sums).drop_duplicates(keep='last').index.dropna().astype(int).to_list()


def table_text_splitter(docs: List[Document], chunk_size: int, chunk_overlap):
    """
    splits list of document objects into smaller chunk sized documents
    :param docs: List[Documents]
    :param chunk_size: int size of maximum allowable chunk
    :return: List[Documents]
    """
    data_to_return = []
    table_num = 0
    chunk_num = 0

    for doc in docs:
        doc_list = yaml.full_load(doc.page_content)
        if not isinstance(doc_list, list):
            doc_list = [doc_list]

        for data_in_string in doc_list:
            table_num += 1
            doc_page = pd.read_csv(io.StringIO(data_in_string))
            indices = extract_split_indexes(doc_page, chunk_size)
            prev_ind = 0
            for i, next_ind in enumerate(indices):
                chunk_num += 1
                html_text = doc_page.loc[prev_ind:next_ind].to_csv(na_rep="", index=False)
                meta = doc.metadata.copy()
                meta['chunk_num'] = chunk_num
                meta['table_num'] = table_num
                meta['table'] = html_text
                # todo: add previos_shunk next chunk

                data_to_return.append(
                    Document(
                        page_content=html_text,
                        metadata=meta
                    )
                )
                prev_ind = next_ind
    return data_to_return
