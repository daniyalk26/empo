from __future__ import annotations

import os
import logging
from .encryption import AESCipher, get_db_psk, get_psk_by_name

log = logging.getLogger("doclogger")

class Schema:
    """
    Basing metaclass contains meta names in vector store, tobe used by other modules
    """
    processing_step = "processing_step"
    file_tag = "remote_id"
    source_tag = "source"
    id_tag = 'id'
    tenant_id_tag= "tenant_id"
    name_tag = "name"
    encrypt_tag = "encrypted"
    lang_param = "lang"
    chunk_num_tag = "chunk_num"
    chunk_len_in_chars_tag = "chunk_len_in_chars"
    front_end_required_keys = [name_tag, id_tag, "format", "original_format"]
    page_tag = "page"
    summary_tag = "document_summary"
    next_chunk_tag = "next_chunk_text"
    page_break_tag = 'page_breaks_char_offset'
    author_tag="author"
    # next_chunk_page_tag = "next_chunk_page"
    next_chunk_page_breaks_tag = "next_chunk_page_breaks"
    current_chunk_uuid="uuid"
    original_format_tag="original_format"
    format_tag="format"
    # previous_chunk_page_tag = "previous_chunk_page"
    previous_chunk_page_breaks_tag = "previous_chunk_page_breaks"
    previous_chunk_tag = "previous_chunk_text"
    encrypt_meta_keys = ["document_summary", "previous_chunk_text", "next_chunk_text"]
    citation_tag = 'document_id'
    encryption_enable = os.getenv("ENCRYPT", "")
    chunk_summary_tag = 'document_summary'
    chunk_table_tag = 'table'
    chunk_heading_tag = 'heading'
    chunk_tag = "chunk_num"

    parma_map = {
        "Temperature": "temperature",
        "Max length (tokens)": "max_tokens",
        "Stop sequences": "stop_sequence",
        "Top probabilities": "top_porb",
        "Frequency penalty": "freq_penalty",
        "Presence penalty": "present_penalty",
    }
    conv_ops = {
        "int": int,
        "float": float,
        "text": str,
    }

    @staticmethod
    def encrypt(text, key=None, name=None):
        if Schema.encryption_enable.lower() == "true":
            assert not (key and name), 'key and name both can not be present '
            if name:
                return AESCipher(get_psk_by_name(name)).encrypt(text.encode("utf-8")).decode("utf-8")
            if key:
                return (
                    AESCipher(key.encode("utf-8"))
                    .encrypt(text.encode("utf-8"))
                    .decode("utf-8")
                )
            return AESCipher(get_db_psk()).encrypt(text.encode("utf-8")).decode("utf-8")
        return text

    @staticmethod
    def decrypt(text, key=None, name=None):
        if Schema.encryption_enable.lower() == "true":
            try:
                assert not (key and name), 'key and name both can not be present '
                if name:
                    return AESCipher(get_psk_by_name(name)).decrypt(text)
                if key:
                    return AESCipher(key.encode("utf-8")).decrypt(text)
                return AESCipher(get_db_psk()).decrypt(text)
            except Exception as e_x:
                log.error("unable to decrypt %s", text)
                log.error(str(e_x), exc_info=True)
                raise Exception(e_x) from e_x
        return text


    @staticmethod
    def make_filter(files):
        log.info("parsing filter request for db backend")
        # af, ar, bg, bn, ca, cs, cy, da, de, el, en, es, et, fa, fi, fr, gu, he,
        # hi, hr, hu, id, it, ja, kn, ko, lt, lv, mk, ml, mr, ne, nl, no, pa, pl,
        # pt, ro, ru, sk, sl, so, sq, sv, sw, ta, te, th, tl, tr, uk, ur, vi, zh - cn, zh - tw
        # corpus_lang = list({d.get('lang') for d in files if d.get('lang') is not None})
        if files:
            filters = {
                Schema.id_tag: [f[Schema.id_tag] for f in files],
            }
        else:
            filters = {}
        return  filters


    @staticmethod
    def extract_context(files):
        context=''
        if len(files) <= 20:
            context='\n\t- '.join([f['summary'] for f in files])
        return  context