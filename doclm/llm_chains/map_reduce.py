import json
import logging
from abc import ABC
from typing import Any, Dict, List, Optional, Tuple
from pydantic import Extra, Field

from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)

from langchain.base_language import BaseLanguageModel
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import BaseRetriever, Document
from langchain.chains.openai_functions import create_structured_output_chain

from .qa_chain import chat_title_generate
from .format_retrieve_data import MergedDataRetrieverChain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain

log = logging.getLogger("doclogger")


class MapSubjectChain(BaseCombineDocumentsChain, ABC):
    """ Class to implements parallel calls to llm and then reduces the
    result as single answer
    """
    common_llm: BaseLanguageModel
    large_context_llm: BaseLanguageModel
    map_prompt: BasePromptTemplate
    reducer_prompt: BasePromptTemplate
    title_prompt: BasePromptTemplate
    subject: str = "a specific field"
    retriever: BaseRetriever = Field()
    document_formatting_prompt: BasePromptTemplate
    reduce_document_formatting_prompt: BasePromptTemplate
    map_function: Any
    document_variable_name: str = 'context'
    output_key: str = "text"  #: :meta private:

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
        return ["input"]

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.
        :meta private:
        """
        return [self.output_key]

    def _call(
            self,
            inputs: Dict[str, List[Document]],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        """Prepare inputs, call combine docs, prepare outputs."""
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()

        chat_subject = None

        if not inputs.get("chat_subject"):
            chat_subject = chat_title_generate(inputs["input"], self.common_llm,
                                               self.title_prompt, run_manager)

        document_combine_chain = MergedDataRetrieverChain(
            document_formatting_prompt=self.document_formatting_prompt,
            retriever=self.retriever,
            verbose=False,
        )

        log.info("Getting relevant documents")
        document_output_dict = document_combine_chain(
            {"query": inputs["input"]},
            callbacks=run_manager.get_child() if run_manager else None,
        )
        # docs = document_output_dict["text"]
        # doc_source = document_output_dict["source"]
        docs = list(
            zip(document_output_dict["text"], document_output_dict["source"])
        )
        # docs = inputs[self.input_key]
        # Other keys are assumed to be needed for LLM prediction
        # other_keys = {k: v for k, v in inputs.items() if k != self.input_key}
        response, doc_source = self.combine_docs(docs, callbacks=_run_manager.get_child(), subject=self.subject,
                                                 question=inputs["input"])

        return {
            self.output_key: json.dumps(
                {
                    "chat_subject": chat_subject,
                    "response": response,
                    "files": doc_source
                    # 'answer': json.dumps({'response': response, "files": doc_source})})
                }
            )
        }

        #
        # # extra_return_dict[self.output_key] = output
        # return output, extra_return_dict

    def combine_docs(
            self,
            docs: List[Any],
            token_max: Optional[int] = None,
            callbacks=None,
            **kwargs: Any,
    ) -> Tuple[str, list]:
        """Combine documents in a map reduce manner.

        Combine by mapping first chain over all documents, then reducing the results.
        This reducing can be done recursively if needed (if there are many documents).
        """
        # map_chain = LLMChain(llm=self.common_llm, prompt=self.map_prompt)
        map_chain = create_structured_output_chain(
            self.map_function, self.common_llm, self.map_prompt)

        map_results = map_chain.apply(
            # FYI - this is parallelized and so it is fast.
            [{self.document_variable_name: d[0], **kwargs} for d in docs],
            callbacks=callbacks,
        )
        question_result_key = map_chain.output_key
        filtered_sources = []
        doc_string = ''
        for i, r in enumerate(map_results):
            answer_exist = r[question_result_key].values()

            if answer_exist:
                filtered_sources.append(docs[i][1])
                # translated document may be need to support multilingual for now
                doc_string += self.reduce_document_formatting_prompt.format_prompt(
                    **{
                        "content": docs[i][0]
                    }).to_string()
                # doc_string += context
                # doc_string += '\n'
        # ===============================================
        reduce_chain = LLMChain(
            prompt=self.reducer_prompt,
            llm=self.large_context_llm,
        )
        log.info("running answer chain with")
        response = reduce_chain.run(
            {
                # "subject": self.subject,
                "question": kwargs["question"],
                "context": doc_string,
            },
            callbacks=callbacks,
            metadata={"stage_name": 'answer_generation_chain'}
        )

        return response, filtered_sources

    async def acombine_docs(
            self,
            docs: List[Document],
            token_max: Optional[int] = None,
            callbacks=None,
            **kwargs: Any,
    ) -> Tuple[str, dict]:
        """Combine documents in a map reduce manner.

        Combine by mapping first chain over all documents, then reducing the results.
        This reducing can be done recursively if needed (if there are many documents).
        """
        map_results = await self.common_llm.apply(
            # FYI - this is parallelized and so it is fast.
            [{**{self.document_variable_name: d.page_content}, **kwargs} for d in docs],
            callbacks=callbacks,
        )
        question_result_key = self.common_llm.output_key
        result_docs = [
            Document(page_content=r[question_result_key], metadata=docs[i].metadata)
            # This uses metadata from the docs, and the textual results from `results`
            for i, r in enumerate(map_results)
        ]
        result, extra_return_dict = await self.reduce_documents_chain.acombine_docs(
            result_docs, token_max=token_max, callbacks=callbacks, **kwargs
        )
        if self.return_intermediate_steps:
            intermediate_steps = [r[question_result_key] for r in map_results]
            extra_return_dict["intermediate_steps"] = intermediate_steps
        return result, extra_return_dict

    # def _call(
    #         self,
    #         inputs: Dict[str, Any],
    #         run_manager: Optional[CallbackManagerForChainRun] = None,
    # ) -> Dict[str, str]:
    #     log.info("in SubjectChain")
    #
    #     chat_subject = None
    #     if not inputs.get("chat_subject"):
    #         chat_subject = chat_title_generate(inputs["input"], self.llm,
    #                                            self.title_prompt, run_manager)
    #     # dummy_answer_chain = LLMChain(llm=self.llm, prompt=self.dummy_prompt)
    #     # dummy_answer = dummy_answer_chain.run(
    #     #     inputs["input"], callbacks=run_manager.get_child() if run_manager else None
    #     # metadata = {"stage_name": 'dummy_answer'}
    #     # )
    #
    #     document_combine_chain = SimpleDataRetrieveChain(
    #         document_formatting_prompt=self.document_formatting_prompt,
    #         retriever=self.retriever,
    #         verbose=False,
    #     )
    #
    #     log.info("Getting relevant documents")
    #     document_output_dict = self.document_retriver_chain(
    #         {"query": inputs["input"]},
    #         callbacks=run_manager.get_child() if run_manager else None,
    #     )
    #
    #     formatted_docs = document_output_dict["text"]
    #     doc_source = document_output_dict["source"]
    #     answer_generation_chain = LLMChain(
    #         prompt=self.extractor_prompt,
    #         llm=self.chat_llm,
    #     )
    #     log.info("running answer chain with")
    #     response = answer_generation_chain.run(
    #         {
    #             "subject": self.subject,
    #             "question": inputs["input"],
    #             "context": formatted_docs,
    #         },
    #         callbacks=run_manager.get_child() if run_manager else None,
    #         metadata={"stage_name": 'answer_generation_chain'}
    #     )
    #
    #     return {
    #         self.output_key: json.dumps(
    #             {
    #                 "chat_subject": chat_subject,
    #                 "response": response,
    #                 "files": doc_source
    #                 # 'answer': json.dumps({'response': response, "files": doc_source})})
    #             }
    #         )
    #     }

# reduce_prompt = PromptTemplate.from_template(
#     "Combine these summaries: {context}"
# )
# reduce_llm_chain = LLMChain(llm=llm, prompt=reduce_prompt)
#
# combine_documents_chain = StuffDocumentsChain(
#     llm_chain=reduce_llm_chain,
#     document_prompt=document_prompt,
#     document_variable_name=document_variable_name
# )
# reduce_documents_chain = ReduceDocumentsChain(
#     combine_documents_chain=combine_documents_chain,
# # )
# llm_chain = LLMChain(llm=self.chat_llm, prompt=self.extractor_prompt)
# chain = MapReduceDocumentsChain(
#     llm_chain=llm_chain,
#     reduce_documents_chain=reduce_documents_chain,
# )
# val = chain.run()
