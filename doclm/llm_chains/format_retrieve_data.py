"""
   contains implementation for document retrival from vector store
"""
from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List, Optional, Callable, Tuple
import logging
from pydantic import Extra, Field

import numpy as np

from langchain.chains.base import Chain
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.retrievers import BaseRetriever, Document
from langchain_core.runnables import RunnableConfig
from collections import OrderedDict

from ..schema import Schema

log = logging.getLogger("doclogger")
log.disabled = False
# pylint: disable=C0103,R0914,R0903:

#TODO: Why are these variables declared here.


# source_tag = "source"
class FormatRetrieverBaseChain(Chain, ABC):

    retriever: BaseRetriever = Field(exclude=True)
    """Prompt object to use."""
    output_key_text: str = "context"  #: :meta private:
    output_key_source: str = "source"
    token_calculator: Callable

    @staticmethod
    def _overlap_between_strings(current_str: str, next_str: str):
        for ele in range(1, len(current_str)):
            if next_str.startswith(current_str[ele:]):
                return current_str[ele:]
        return ""

    @staticmethod
    def _remove_overlap_from_prev_string(prev_str: str, current_str: str):
        result = prev_str
        if len(current_str) == 0:
            return result, 0
        overlap_length = len(FormatRetrieverBaseChain._overlap_between_strings(prev_str, current_str))
        result = prev_str[: len(prev_str) - overlap_length]
        return result, len(prev_str) - overlap_length

    @staticmethod
    def _remove_overlap_from_next_string(current_str: str, next_str: str):
        result = next_str
        if len(current_str) == 0:
            return result, 0
        overlap_length = len(FormatRetrieverBaseChain._overlap_between_strings(current_str, next_str))
        result = next_str[overlap_length:]
        return result, overlap_length

    # This function does not handle heading boundaries which would occur in case of structured parser.
    @staticmethod
    def _remove_chunk_text_overlap_from_doc(doc: Document):
        # metadata = doc.metadata
        doc.metadata[
            Schema.previous_chunk_tag], previous_chunk_overlap_offset = FormatRetrieverBaseChain._remove_overlap_from_prev_string(
            doc.metadata[Schema.previous_chunk_tag], doc.page_content)
        doc.metadata[
            Schema.next_chunk_tag], next_chunk_overlap_offset = FormatRetrieverBaseChain._remove_overlap_from_next_string(
            doc.page_content, doc.metadata[Schema.next_chunk_tag])
        doc.metadata[Schema.previous_chunk_page_breaks_tag] = [pg_break for pg_break in
                                                        doc.metadata[Schema.previous_chunk_page_breaks_tag] if
                                                        pg_break < previous_chunk_overlap_offset]
        doc.metadata[Schema.next_chunk_page_breaks_tag] = [pg_break - next_chunk_overlap_offset for pg_break in
                                                    doc.metadata[Schema.next_chunk_page_breaks_tag] if
                                                    pg_break - next_chunk_overlap_offset > 0]
        return doc

    @staticmethod
    def _reduce_tokens_below_limit(token_calculator: Callable, max_tokens: int, docs: List[Document]) -> List[Document]:
        num_docs = len(docs)
        log.info("got %s docs", num_docs)
        extracted_document_format_template_tokens = 40
        tokens = [
            token_calculator(doc.page_content) +
            token_calculator(doc.metadata[Schema.summary_tag]) +
            extracted_document_format_template_tokens +
            token_calculator(doc.metadata.get(Schema.next_chunk_tag, "")) +
            token_calculator(doc.metadata.get(Schema.previous_chunk_tag, ""))
            for doc in docs
        ]
        token_count = sum(tokens)
        while token_count > max_tokens:
            num_docs -= 1
            token_count -= tokens[num_docs]

        log.info("returning %s docs", num_docs)

        return docs[:num_docs]
    @staticmethod
    def merge_chunks(chunks_and_scores):
        def merge_chunks_list(chunk_list: List[Document]) -> Document:
            if len(chunk_list) == 1:
                return chunk_list[0]
            result_chunk = chunk_list[0].copy()
            current_chunk_idx = 0
            for chunk in chunk_list[1:]:
                if chunk.metadata[Schema.chunk_tag] - chunk_list[current_chunk_idx].metadata[Schema.chunk_tag] == 1:
                    current_chunk_pg_content_length = len(result_chunk.page_content)
                    result_chunk.metadata[Schema.page_break_tag] += [pg_break + current_chunk_pg_content_length for pg_break in
                                                              result_chunk.metadata[Schema.next_chunk_page_breaks_tag]]
                    result_chunk.page_content += result_chunk.metadata[Schema.next_chunk_tag]
                    result_chunk.metadata[Schema.next_chunk_page_breaks_tag] = chunk.metadata[Schema.next_chunk_page_breaks_tag]
                    result_chunk.metadata[Schema.next_chunk_tag] = chunk.metadata[Schema.next_chunk_tag]
                    current_chunk_idx += 1
                elif chunk.metadata[Schema.chunk_tag] - chunk_list[current_chunk_idx].metadata[Schema.chunk_tag] == 2:
                    current_chunk_pg_content_length = len(result_chunk.page_content)
                    result_chunk.metadata[Schema.page_break_tag] += [pg_break + current_chunk_pg_content_length for pg_break in
                                                              result_chunk.metadata[Schema.next_chunk_page_breaks_tag]]
                    result_chunk.page_content += result_chunk.metadata[Schema.next_chunk_tag]

                    overlap_length = len(
                        FormatRetrieverBaseChain._overlap_between_strings(result_chunk.page_content, chunk.page_content))
                    chunk.metadata[Schema.page_break_tag] = [pg_break - overlap_length for pg_break in
                                                      chunk.metadata[Schema.page_break_tag] if (pg_break - overlap_length) > 0]
                    current_chunk_pg_content_length = len(result_chunk.page_content)
                    result_chunk.metadata[Schema.page_break_tag] += [pg_break + current_chunk_pg_content_length for pg_break in
                                                              chunk.metadata[Schema.page_break_tag]]
                    result_chunk.page_content += chunk.page_content[overlap_length:]

                    result_chunk.metadata[Schema.next_chunk_page_breaks_tag] = chunk.metadata[Schema.next_chunk_page_breaks_tag]
                    result_chunk.metadata[Schema.next_chunk_tag] = chunk.metadata[Schema.next_chunk_tag]
                    current_chunk_idx += 1
                else:
                    current_chunk_pg_content_length = len(result_chunk.page_content)
                    result_chunk.metadata[Schema.page_break_tag] += [pg_break + current_chunk_pg_content_length for pg_break in
                                                              result_chunk.metadata[Schema.next_chunk_page_breaks_tag]]
                    result_chunk.page_content += result_chunk.metadata[Schema.next_chunk_tag]

                    overlap_length = len(FormatRetrieverBaseChain._overlap_between_strings(result_chunk.page_content,
                                                                                           chunk.metadata[
                                                                                           Schema.previous_chunk_tag]))
                    chunk.metadata[Schema.previous_chunk_tag] = chunk.metadata[Schema.previous_chunk_tag][overlap_length:]
                    chunk.metadata[Schema.previous_chunk_page_breaks_tag] = [pg_break - overlap_length for pg_break in
                                                                      chunk.metadata[Schema.previous_chunk_page_breaks_tag] if
                                                                      (pg_break - overlap_length) > 0]
                    current_chunk_pg_content_length = len(result_chunk.page_content)
                    result_chunk.metadata[Schema.page_break_tag] += [pg_break + current_chunk_pg_content_length for pg_break in
                                                              chunk.metadata[Schema.previous_chunk_page_breaks_tag]]
                    result_chunk.page_content += chunk.metadata[Schema.previous_chunk_tag]

                    current_chunk_pg_content_length = len(result_chunk.page_content)
                    result_chunk.metadata[Schema.page_break_tag] += [pg_break + current_chunk_pg_content_length for pg_break in
                                                              chunk.metadata[Schema.page_break_tag]]
                    result_chunk.page_content += chunk.page_content[overlap_length:]

                    result_chunk.metadata[Schema.next_chunk_page_breaks_tag] = chunk.metadata[Schema.next_chunk_page_breaks_tag]
                    result_chunk.metadata[Schema.next_chunk_tag] = chunk.metadata[Schema.next_chunk_tag]
                    current_chunk_idx += 1

            return result_chunk

        def doc_scores_map(chunks_and_scores: List[Tuple])->Dict:
            docs_score_accumulation={}
            for chunk, score in chunks_and_scores:
                if chunk.metadata[Schema.id_tag] not in docs_score_accumulation:
                    docs_score_accumulation[chunk.metadata[Schema.id_tag]]=[]
                docs_score_accumulation[chunk.metadata[Schema.id_tag]].append(score)
            return {id:np.sum(v) for id,v in docs_score_accumulation.items()}

        def doc_chunk_map(chunks_and_scores: List[Tuple]) -> Dict:
            docs_chunk_mapping = {}
            for chunk, score in chunks_and_scores:
                if chunk.metadata[Schema.id_tag] not in docs_chunk_mapping:
                    docs_chunk_mapping[chunk.metadata[Schema.id_tag]] = []
                docs_chunk_mapping[chunk.metadata[Schema.id_tag]].append(chunk)
            return {id: sorted(v, key=lambda x: x.metadata[Schema.chunk_tag]) for id, v in docs_chunk_mapping.items()}

        def doc_mergeable_chunk_map(doc_chunk_dict: Dict) -> Dict:
            docs_mergeable_chunk_mapping = {}
            for doc_id, chunk_list in doc_chunk_dict.items():
                docs_mergeable_chunk_mapping[doc_id] = []
                current_chunk_idx = 0
                current_chunk_list = [chunk_list[0]]
                for chunk in chunk_list[1:]:
                    if (chunk.metadata[Schema.chunk_tag] - chunk_list[current_chunk_idx].metadata[Schema.chunk_tag] in [1, 2, 3]) and \
                            (chunk.metadata.get(Schema.chunk_heading_tag, "") == chunk_list[current_chunk_idx].metadata.get(
                                Schema.chunk_heading_tag, "")):
                        current_chunk_list.append(chunk)
                        current_chunk_idx += 1
                        continue
                    else:
                        docs_mergeable_chunk_mapping[doc_id].append(current_chunk_list)
                        current_chunk_list = [chunk]
                        current_chunk_idx += 1
                docs_mergeable_chunk_mapping[doc_id].append(current_chunk_list)
            return docs_mergeable_chunk_mapping

        def doc_merged_chunk_map(doc_mergeable_chunk_dict: Dict) -> Dict:
            docs_merged_chunk_mapping = {}
            for doc_id, mergeable_chunk_list in doc_mergeable_chunk_dict.items():
                docs_merged_chunk_mapping[doc_id] = []
                for mergeable_chunks in mergeable_chunk_list:
                    docs_merged_chunk_mapping[doc_id].append(merge_chunks_list(mergeable_chunks))
            return docs_merged_chunk_mapping

        def doc_page_wise_split_merged_chunk_map(doc_merged_chunk_dict: Dict) -> Dict:
            doc_page_wise_split_chunk_mapping = {}
            for doc_id, merged_chunk_list in doc_merged_chunk_dict.items():
                doc_page_wise_split_chunk_mapping[doc_id] = []
                for merged_chunk in merged_chunk_list:
                    len_previous_chunk = len(merged_chunk.metadata[Schema.previous_chunk_tag])
                    len_current_chunk = len(merged_chunk.page_content)
                    starting_page_number = int(merged_chunk.metadata[Schema.page_tag]) - len(
                        merged_chunk.metadata[Schema.previous_chunk_page_breaks_tag])
                    combined_chunk_content = merged_chunk.metadata[Schema.previous_chunk_tag] + merged_chunk.page_content + \
                                             merged_chunk.metadata[Schema.next_chunk_tag]
                    combined_pg_breaks = merged_chunk.metadata[Schema.previous_chunk_page_breaks_tag] + \
                                         [pg_break + len_previous_chunk for pg_break in
                                          merged_chunk.metadata[Schema.page_break_tag]] + \
                                         [pg_break + len_previous_chunk + len_current_chunk for pg_break in
                                          merged_chunk.metadata[Schema.next_chunk_page_breaks_tag]]
                    if not len(combined_pg_breaks):
                        combined_pg_breaks = [len(combined_chunk_content)]
                    if combined_pg_breaks[-1] != len(combined_chunk_content):
                        combined_pg_breaks += [len(combined_chunk_content)]
                    page_starting_offset_in_chars = 0
                    # merged_chunk_pg_wise_splits=[]
                    for idx, pg_break_end_offset in enumerate(combined_pg_breaks):
                        metadata:dict = merged_chunk.metadata.copy()
                        metadata.update({Schema.page_tag: starting_page_number + idx,
                                         Schema.chunk_heading_tag: merged_chunk.metadata.get(Schema.chunk_heading_tag, ''),
                                         Schema.name_tag: merged_chunk.metadata.get(Schema.name_tag),
                                         })
                        doc_page = Document(
                            page_content=combined_chunk_content[page_starting_offset_in_chars: pg_break_end_offset],
                            metadata=metadata)
                        # merged_chunk_pg_wise_splits.append(doc_page)
                        doc_page_wise_split_chunk_mapping[doc_id].append(doc_page)
                        page_starting_offset_in_chars = pg_break_end_offset
                # doc_page_wise_split_chunk_mapping[doc_id].append(merged_chunk_pg_wise_splits)
            return doc_page_wise_split_chunk_mapping

        def doc_same_page_content_combined_map(doc_page_wise_split_chunk_dict: Dict):
            doc_same_page_content_combined_mapping = {}
            for doc_id, imperfect_page_wise_chunk_list in doc_page_wise_split_chunk_dict.items():
                doc_same_page_content_combined_mapping[doc_id] = [imperfect_page_wise_chunk_list[0]]
                current_idx = 0
                for imperfect_page_wise_chunk in imperfect_page_wise_chunk_list[1:]:
                    if [imperfect_page_wise_chunk.metadata[Schema.page_tag]] == \
                            imperfect_page_wise_chunk_list[current_idx].metadata[Schema.page_tag]:
                        doc_same_page_content_combined_mapping[doc_id][
                            current_idx].page_content += '\n\n...\n\n' + imperfect_page_wise_chunk.page_content
                    else:
                        doc_same_page_content_combined_mapping[doc_id].append(imperfect_page_wise_chunk)
                        current_idx += 1
            return doc_same_page_content_combined_mapping

        docid_score_tuple_sorted=[(doc_id, score) for doc_id,score in doc_scores_map(chunks_and_scores).items()]
        docid_score_tuple_sorted = sorted(docid_score_tuple_sorted, key=lambda x: x[1], reverse=True)
        doc_chunk_dict = doc_chunk_map(chunks_and_scores)
        doc_mergeable_chunk_dict = doc_mergeable_chunk_map(doc_chunk_dict)
        doc_merged_chunk_dict = doc_merged_chunk_map(doc_mergeable_chunk_dict)
        doc_page_wise_split_merged_chunk_dict = doc_page_wise_split_merged_chunk_map(doc_merged_chunk_dict)
        doc_same_page_content_combined_dict = doc_same_page_content_combined_map(doc_page_wise_split_merged_chunk_dict)

        sorted_result_dict=OrderedDict()
        for doc_id,_ in docid_score_tuple_sorted:
            sorted_result_dict[doc_id]=doc_same_page_content_combined_dict[doc_id]

        # return docs_page_wise_split_n_sorted_list
        return sorted_result_dict


class FormatRetrieverDocumentChain(FormatRetrieverBaseChain):
    """
    An example of a custom chain.
    """
    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True
        # allow_population_by_field_name = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return ["query"]

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key_text, self.output_key_source]


    #TODO: Need to find a faster way for checking overlap

    def _get_docs(self, input_question: str, lang: str, tenant_id: int, filter:List,k:int, callbacks) -> List[Document]:
        log.info("getting relevant documents form retriever")
        config = RunnableConfig(callbacks=callbacks,
                                metadata={"stage_name": 'doc_chunk_retriever',
                                          "retriever": self.retriever},
                           )
        return self.retriever.invoke(input={
            'query': input_question,
            'lang': lang,
            'tenant_id': tenant_id,
            'filter': filter,
            'k': k}, config=config
        )


    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        log.info("Entering FormatRetrieveDataChain")
        log.info(inputs)
        docs_and_scores = self._get_docs(
            inputs["query"], inputs["lang"], inputs["tenant_id"], inputs["filter"],inputs["k"],run_manager.get_child() if run_manager else None
        )
        log.info(f"originally got doc names {[doc.metadata['name'] for doc, _ in docs_and_scores]}")
            
        # for doc, score in docs_and_scores:
        if inputs['reranker_bool']==True:
            log.info("Reranking documents")
            try:
                docs_and_scores = inputs["reranker"].rerank_documents(inputs["query"], [doc for doc, _ in docs_and_scores])
            except Exception as e:
                log.error(e, exc_info=True)
        log.info(f"after rerank got doc names {[doc.metadata['name'] for doc, _ in docs_and_scores]}")
        docs_and_scores_without_overlap = [(self._remove_chunk_text_overlap_from_doc(doc), score) for doc, score in docs_and_scores]

        docs = self._reduce_tokens_below_limit(self.token_calculator, inputs["max_tokens"], [doc for doc,_ in docs_and_scores])

        docs_and_scores = docs_and_scores_without_overlap[:len(docs)]
        docs_and_scores = docs_and_scores[:inputs["k"]]
        log.info(f"after _reduce_tokens_below_limit got doc names {[doc.metadata['name'] for doc, _ in docs_and_scores]}")
        
        log.debug('merging chunks')
        docs_dict = self.merge_chunks(docs_and_scores)
        if set(docs_dict.keys()) != set([doc.metadata['id'] for doc, _ in docs_and_scores]):
            log.error("difference before and after merging, probably wrong merging")
            log.error(docs_dict.keys(), set([doc.metadata['id'] for doc, _ in docs_and_scores]))
        response = ""
        if len(docs) == 0:
            log.info("No relevant documents found: leaving")

            return {
                self.output_key_text: "no relevant documents found".upper(),
                self.output_key_source: [],
            }

        log.info("joining document in formatted_doc")
        # return [{k_y: doc.metadata.get(k_y) for k_y in keys} for doc in docs]
        resource_source = []
        source_keys = Schema.front_end_required_keys + [Schema.page_tag]  #, source_tag]

        excerpt_counter = 1
        for oc_id, doc_list in docs_dict.items():
            response += '<Document Start>\n'
            response += f'Document name: {doc_list[0].metadata[Schema.name_tag]}\n'
            response += f'Document authors: {doc_list[0].metadata[Schema.author_tag]}\n'
            response += f'Document description: {doc_list[0].metadata[Schema.summary_tag]}\n'

            for doc in doc_list:
                response += f'\n[Excerpt: {excerpt_counter}]\n'
                if len(doc.metadata.get(Schema.chunk_heading_tag, '')):
                    response += f'Excerpt heading: {doc.metadata.get(Schema.chunk_heading_tag)}\n'
                response += f'{doc.page_content}\n'
                _source = {k_y: doc.metadata.get(k_y) for k_y in source_keys}
                _source[Schema.citation_tag] = str(excerpt_counter)
                resource_source.append(_source)
                excerpt_counter += 1
            # _source.update({name_tag: doc.name})
            response += '<Document End>'
        log.info(f"sources dict before leaving Format Retriever: {resource_source}")
        log.info("leaving function")

        return {self.output_key_text: response, self.output_key_source: resource_source}

    async def _acall(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        prompt_value = self.prompt.format_prompt(**inputs)
        docs_and_scores = []
        if inputs['reranker_bool']==True:
            docs_and_scores = await input["reranker"].arerank_documents(inputs["query"], [doc for doc, _ in docs_and_scores])
        # Whenever you call a language model, or another chain, you should pass
        # a callback manager to it. This allows the inner run to be tracked by
        # any callbacks that are registered on the outer run.
        # You can always obtain a callback manager for this by calling
        # `run_manager.get_child()` as shown below.
        response = await self.llm.agenerate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )
    
        # If you want to log something about this run, you can do so by calling
        # methods on the `run_manager`, as shown below. This will trigger any
        # callbacks that are registered for that event.
        if run_manager:
            await run_manager.on_text("Log something about this run")

        return {self.output_key: response.generations[0][0].text}

    @property
    def _chain_type(self) -> str:
        return "summarization_chain"


class FormatRetrieverWebChain(FormatRetrieverBaseChain):
    """
       An example of a custom chain.
       """
    # run_manager: Optional[CallbackManagerForChainRun] = None,

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True
        # allow_population_by_field_name = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return ["query"]

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key_text, self.output_key_source]

    # TODO: Need to find a faster way for checking overlap
    #
    # def _get_docs(self, input_question: str, lang: str, tenant_id: int, callbacks) -> List[Document]:
    #     log.info("getting relevant documents form retriever")
    #     return self.retriever.get_relevant_documents(
    #         query=input_question,
    #         lang=lang,
    #         callbacks=callbacks,
    #         metadata={"retriever": self.retriever},
    #         tenant_id=tenant_id
    #     )

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:

        log.info("Entering formmater cahin")
        config = RunnableConfig(callbacks=run_manager.get_child(),
                                metadata={"stage_name": 'format retriever'},
                                # metadata={'stream': True}
                                configurable={"output_token_number": inputs["max_tokens"]}
                                )
        docs_and_scores = self.retriever.invoke(
            inputs, config)

        # inputs["query"], inputs["lang"]
        if len(docs_and_scores) == 0:
            log.info("No relevant documents found: leaving")
            return {self.output_key_text: "no relevant documents found".upper(),
                    self.output_key_source: []}

        docs = [FormatRetrieverBaseChain._remove_chunk_text_overlap_from_doc(doc) for doc, _ in docs_and_scores]
        docs = FormatRetrieverBaseChain._reduce_tokens_below_limit(self.token_calculator,  inputs["max_tokens"], docs)
        docs_and_scores = docs_and_scores[:len(docs)]

        log.debug('merging chunks')
        docs_dict = FormatRetrieverBaseChain.merge_chunks(docs_and_scores)

        response = ""
        resource_source = []

        log.info("joining document in formatted_doc")
        excerpt_counter = 1
        for doc_id, doc_list in docs_dict.items():
            response += '<Document Start>\n'
            response += 'Document URL: '+doc_list[0].metadata['source']+'\n'
            response += f'Document description: {doc_list[0].metadata[Schema.summary_tag]}\n'
            response += f'\n[Excerpt: {excerpt_counter}] '

            for doc in doc_list:
                # if len(doc.metadata.get(Schema.chunk_heading_tag, '')):
                #     response += f'Excerpt heading: {doc.metadata.get(Schema.chunk_heading_tag)}\n'
                response += f'{doc.page_content}\n'
            _source = doc_list[0].metadata.copy()
            _source[Schema.citation_tag] = str(excerpt_counter)
            resource_source.append(_source)
            excerpt_counter += 1
            # _source.update({name_tag: doc.name})
            response += '<Document End>\n\n'

        log.info("leaving function")

        return {self.output_key_text: response, self.output_key_source: resource_source}
