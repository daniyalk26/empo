import re
import os
import logging
from datetime import date

from typing import Optional, Type, List, Any, Union,Tuple
from typing_extensions import Annotated
from pydantic import BaseModel, Field


from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import InjectedState
from langchain.tools import BaseTool
from langchain_core.tools import InjectedToolArg
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
    dispatch_custom_event
)

from langchain_core.messages import HumanMessage

from ..llm_chains.qa_chain import GeneralChain, build_title_chain
from ..templates.util import get_templates
from ..schema import Schema
from ..tokenizer_cal import num_tokens_from_string
from ..config import supported_templates
from ..llm_chains.format_retrieve_data import FormatRetrieverDocumentChain, FormatRetrieverWebChain
from ..templates.planner import joinner_citation_instructions
from ..models.models import OutFEChatMetaMsg


log = logging.getLogger("doclogger")
log.disabled = False
MAX_TOKENS_ROUTER = int(os.getenv('MAX_TOKENS_ROUTER', 150))
MAX_TOKENS_REPHRASE = int(os.getenv('MAX_TOKENS_REPHRASE', 500))
MAX_TOKENS_CHAT = int(os.getenv('MAX_TOKENS_CHAT', 3000))
MAX_TOKENS_TITLE = int(os.getenv('MAX_TOKENS_TITLE', 100))

def get_token_calculator_for_model(model_name):
    def token_calculator(msg):
        return num_tokens_from_string(model_name, msg)

    return token_calculator

def prepare_question(runtime_state: BaseModel, task_id:str)->str:
    tasks_dict={str(task.id):task for task in runtime_state.plan.task_list}
    additional_context=''
    depends_on_task_list=tasks_dict[task_id].depends_on
    current_task_description=tasks_dict[task_id].description
    current_task=f'Current Task to answer; **{current_task_description}**\n'
    # question=runtime_state.user_question
    # depends_on_tasks_sources={}
    # llm_question=self.state.task_data[task_id].description
    additional_context=f'User original question: {runtime_state.user_question}'
    if len(depends_on_task_list)>0:
        # additional_context+='\n**Execution Plan:**\n'
        # additional_context+=f'Overall Goal: {runtime_state.plan.goal}\n'
        additional_context+='**Previous Tasks:**\n'
        for dependent_task_id in depends_on_task_list:
            dependent_task_description=runtime_state.task_data[str(dependent_task_id)].description
            dependent_task_result=runtime_state.task_results[str(dependent_task_id)].content['response']
            additional_context+=f'Task {dependent_task_id}. {dependent_task_description}\nTask {dependent_task_id} result. {dependent_task_result}\n\n'
        # current_task=additional_context+current_task
    return current_task+additional_context


def regex_filter(response, files) -> list:
    try:
        # response = re.sub(r"\[(?i)AI.+?:\s*(\d+.*?)\]", "", response)
        # task_id_sources_map={}
        reg_exp_get_content_between_braces = r'\[Task:(.*?\d),.*?:(.*?\d+)\]'
        reg_exp_get_doc_nums = r'\d+'
        cited_docs = []
        for task_match, excerpt_match in re.findall(reg_exp_get_content_between_braces, response):
            tasks = [task for task in re.findall(reg_exp_get_doc_nums, task_match)]
            excerpts = [task for task in re.findall(reg_exp_get_doc_nums, excerpt_match)]
            for task in tasks:
                for excerpt in excerpts:
                    cited_docs.append((task,excerpt))
            
        log.debug('extracted documents %s', str(cited_docs))

        log.debug("document parse in regex")
        log.debug(cited_docs)
        cited_docs = set(cited_docs)

        cited_sources = []
        for task_id in files:
            for file in files[task_id]:
                if (str(task_id), file[Schema.citation_tag]) in cited_docs:  # and file not in results:
                    cited_sources.append(file)
                else:
                    log.debug(f'{task_id}, {file[Schema.citation_tag]} not in {cited_docs}')
        return cited_sources

    except Exception as e:
        log.error('Can not apply regex', exc_info=True)
        return []


class SearchInput(BaseModel):
    """Parameters to be used for searching the required information"""
    query: str = Field(description="should be the search query")
    task_id: Annotated[str, InjectedToolArg]
    runtime_state: Annotated[BaseModel, InjectedToolArg]


class JoinerInput(BaseModel):
    """Parameters to be used for generating response to the task at hand"""
    query: str = Field(description="Should be the brief task description.")
    task_id: Annotated[str, InjectedToolArg]
    runtime_state: Annotated[BaseModel, InjectedToolArg]


class SearchInputWithFileFilters(SearchInput):
    llm_file_filters: Optional[List[int]] = Field(description="should be the list of document ids to narrow the search scope")


class SearchEnterpriseKnowledgeTool(BaseTool):
    name: str = "enterprise"
    # TODO: need to update this
    description: str = f"Use this tool for retrieving documents from the enterprise knowledge base"
    args_schema: Type[BaseModel] = SearchInput
    state: Annotated[BaseModel, InjectedState]

    def _run(
        self, query: str, task_id: str, runtime_state: BaseModel, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> dict[str, Any]:
        """Use the tool."""
        log.info("in SearchEnterpriseKnowledgeTool")
        model_name = self.state.tools_llm.metadata.get('model') or self.state.tools_llm.default.model_name
        use_templates=get_templates(['SUBJECT'], 'planner',
                                   model_name)
        k=20
        search_query=query
        llm_question=prepare_question(runtime_state=runtime_state, task_id=task_id)

        chat_llm=self.state.tool_args[self.name]["chat_llm"]
        extractor_prompt=use_templates['extractor_prompt']
        retriever=self.state.tool_args[self.name]["retriever"]
        token_calculator = get_token_calculator_for_model(model_name)

        reranker=self.state.tool_args[self.name]["reranker"]
        reranker_bool=self.state.tool_args[self.name]["reranker_bool"]

        tenant_id=self.state.tenant_id
        lang = self.state.lang
        filter=self.state.tool_args[self.name]["filter"]
        user_profile=self.state.additional_metadata["user_profile"]
        lang_expanded = supported_templates[lang]

        log.info('question: %s ,language: %s', search_query, lang)
        
        subject = "provided documents"
        
        human_msg = extractor_prompt[-1].format(context='', 
                                                lang=lang_expanded, 
                                                question=llm_question, 
                                                subject=subject, 
                                                **user_profile,
                                                **self.state.tool_args[self.name], 
                                                **chat_llm.metadata['model_profile_for_generation'])
        assert isinstance(human_msg, HumanMessage) , 'the message should be a human template, check prompt template'
        human_msg_tokens = chat_llm.get_num_tokens_from_messages([human_msg])

        config = RunnableConfig(callbacks=run_manager.get_child(),
                                metadata={"stage_name": 'answer_generation_chain',
                                          "tool_name": self.name,
                                          "task_id": task_id},
                                configurable={"output_token_number": MAX_TOKENS_CHAT}
                                )

        document_combine_chain = FormatRetrieverDocumentChain(
            retriever=retriever,
            verbose=False,
            token_calculator=token_calculator
        )

        document_output_dict = document_combine_chain.invoke(
            {"query": search_query, 
             "lang":lang,
             "max_tokens": chat_llm.metadata["retriever_token_limit_calculator"](human_msg_tokens),
             "tenant_id": tenant_id,
             "filter":filter,
             "k":k,
             "reranker_bool":reranker_bool,
             "reranker":reranker},
            config=config,
        )

        formatted_docs = document_output_dict["context"]
        formatted_docs = formatted_docs.replace('[Excerpt:', f'[Task: {task_id}, Excerpt:')
        tasks_sources= {completed_task_id: completed_task.content.get('files',[]) for completed_task_id,completed_task in runtime_state.task_results.items()}
        tasks_sources[int(task_id)] = document_output_dict["source"]
        log.info(f"Format retriver returned: {tasks_sources[int(task_id)]}")
        answer_generation_chain = extractor_prompt | chat_llm

        log.info("running answer chain with")
        inputs={}
        inputs.update({
            "subject": subject,
            "question": llm_question,
            "context": formatted_docs,
            "lang": lang_expanded}
        )
        inputs.update(chat_llm.metadata.get('model_profile_for_generation'))
        inputs.update({'user_name':user_profile.get('user_name','Not Provided'),
                       'user_designation':user_profile.get('user_designation','Not Provided'),
                       'user_department':user_profile.get('user_department','Not Provided'),
                       'user_personalization':user_profile.get('user_personalization','Not Provided'),
                       'user_company':user_profile.get('user_company','Not Provided'),
                       'user_specified_behaviour':user_profile.get('user_specified_behaviour',''),
                       'user_country':user_profile.get('user_country','Not Provided')})
        inputs.update({'current_date': date.today()})
        inputs.update({'chat_history':runtime_state.user_chat_history})

        if chat_llm.metadata.get('stream_able', True):
            response = ''.join([chunk.content for chunk in answer_generation_chain.stream(
                inputs, config=config)])

        else:
            AIMessage = answer_generation_chain.invoke(inputs, config=config)
            response = AIMessage.content
        # docx = len(set([x['name'] for x in sources]))
        # pages = len([x['page'] for x in sources])
        files = tasks_sources
        result_sources = regex_filter(response,files)
        # result_sources = regex_filter(response,doc_source)

        dispatch_custom_event('chat_meta',
                              data= OutFEChatMetaMsg(
                                  task_id=task_id, step_name='task_final', tool_name=self.name,
                                  response=response, files= result_sources, search_phrase=search_query, chat_subject='')
                              )
        return {"response": response,
                "files": result_sources,
                "search_phrase": search_query,
                # 'answer': json.dumps({'response': response, "files": doc_source})})
                }

    def update_description_for_planning(self, **kwargs) -> None:
        """Update the description of the tool"""
        callbacks = [c for c in kwargs['callbacks'] if c.__class__.__name__ == 'TokenCalculationCallback']
       
        lang = self.state.lang
        filter=self.state.tool_args[self.name]["filter"]
        tenant_id=self.state.tenant_id

        adv_search_retriever = self.state.tool_args[self.name]['adv_search_retriever']
        config = RunnableConfig(callbacks=callbacks,
                                metadata={"stage_name": 'adv_search_retriever',
                                          "retriever": adv_search_retriever},
                           )
        retrieved_docs = adv_search_retriever.invoke(input={"query":self.state.query
            ,"k": 25, "filter": filter,"lang": lang,
            "tenant_id": tenant_id
            }, config=config
            )
        doc_summaries="\n\t- ".join([doc.doc_summary for doc in retrieved_docs])
        self.description = f"Use this tool for retrieving documents from the enterprise knowledge base. Summaries of some of the documents available in this knowledge base are provided below: \n\t- {doc_summaries}"

    def update_description_for_execution(self, **kwargs) -> None:
        """Update the description of the tool"""
        callbacks = [c for c in kwargs['callbacks'] if c.__class__.__name__ == 'TokenCalculationCallback']

        advance_search_retriever_search_query=self.state.plan.description+'\n '+\
            self.state.plan.goal+'\n '+\
            '\n '.join([task.description for task in self.state.plan.task_list])
        
        lang = self.state.lang
        filter=self.state.tool_args[self.name]["filter"]
        tenant_id=self.state.tenant_id

        adv_search_retriever = self.state.tool_args[self.name]['adv_search_retriever']
        config = RunnableConfig(callbacks=callbacks,
                                metadata={"stage_name": 'adv_search_retriever',
                                          "retriever": adv_search_retriever},
                           )
        retrieved_docs = adv_search_retriever.invoke(input={"query":advance_search_retriever_search_query
            ,"k": 25, "filter": filter,"lang": lang,
            "tenant_id": tenant_id
            }, config=config
            )
        doc_summaries="\n\t- ".join([doc.doc_summary for doc in retrieved_docs])
        self.description = f"Use for retrieving documents from the enterprise knowledge base. Summaries of some of the documents available in this knowledge base are provided below: \n\t- {doc_summaries}"



class SearchEnterpriseKnowledgeWithFiltersTool(BaseTool):
    name: str = "enterprise_knowledge_with_known_filters"
    description: str = f"useful for when you need to answer Enterprise specific questions like; {name} when the file filters are known"
    args_schema: Type[BaseModel] = SearchInputWithFileFilters
    state: Annotated[BaseModel, InjectedState]
    # subject_chain: Annotated[Chain, InjectedToolArg]

    def _run(
        self, query: str, task_id: str, runtime_state: BaseModel, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        raise NotImplementedError
    def update_description_for_planning(self, **kwargs) -> None:
        """Update the description of the tool"""
        pass
    def update_description_for_execution(self, **kwargs) -> None:
        """Update the description of the tool"""
        pass


class SearchWebTool(BaseTool):
    name:str = "web_search"
    description:str = """Use this tool when the query requires searching the internet, \
        obtaining current or publicly available information online, \
        or referencing recent news, weather, named entities, or the latest research."""
    args_schema: Type[BaseModel] = SearchInput
    state: Annotated[BaseModel, InjectedState]

    def _run(
        self, query: str,  task_id: str, runtime_state: BaseModel, run_manager: Optional[CallbackManagerForToolRun] = None, **kwargs
    ) -> dict[str, Any]:
        """Use the tool."""

        web_retriever=self.state.tool_args[self.name]["retriever"]
        large_context_llm=self.state.tool_args[self.name]["chat_llm"]
        model_name = large_context_llm.metadata.get('model', None) or self.large_context_llm.model_name

        use_templates=get_templates(['WEBCHAT'], 'planner',
                                    model_name)
        subject= "internet topic"

       
        token_calculator = get_token_calculator_for_model(model_name)
        search_query = query
        llm_question=prepare_question(runtime_state=runtime_state, task_id=task_id)
        # llm_question=self.state.task_data[task_id].description
        # additional_context=''
        # depends_on_task_list=self.state.task_data[task_id].depends_on
        # # depends_on_tasks_sources={}

        # if len(depends_on_task_list)>0:
        #     additional_context='\n**Context:**\n'
        #     additional_context+=f'Overall Goal: {runtime_state.plan.goal}\n'
        #     for dependent_task_id in depends_on_task_list:
        #         additional_context+=str(dependent_task_id)+'. '
        #         additional_context+=runtime_state.task_data[str(dependent_task_id)].description+'\n'
        #         additional_context+=str(runtime_state.task_results[str(dependent_task_id)].content['response'])+'\n\n'
        #     llm_question=additional_context+str(task_id)+'. '+llm_question
        response_template=use_templates["web_search"]
        user_profile=self.state.additional_metadata["user_profile"]

        lang = self.state.lang
        lang_expanded = supported_templates[lang]

        human_msg: HumanMessage = response_template[-1].format(context='',
                                                # current_date=current_date,
                                                lang=lang_expanded, 
                                                question=llm_question, 
                                                subject=subject, 
                                                **user_profile,
                                                **self.state.tool_args[self.name], 
                                                **large_context_llm.metadata['model_profile_for_generation'])
        assert isinstance(human_msg, HumanMessage) , 'the message should be a human template, check prompt template'

        human_msg_tokens = large_context_llm.get_num_tokens_from_messages([human_msg])
        config = RunnableConfig(callbacks=run_manager.get_child(),
                                metadata={"stage_name": 'web_search_chain',
                                          "tool_name": self.name,
                                          "task_id": task_id},
                                # metadata={'stream': True}
                                configurable={"output_token_number": MAX_TOKENS_CHAT}
                                )

        web_document_combine_retrieve_chain = FormatRetrieverWebChain(
            retriever=web_retriever,
            token_calculator=token_calculator,
            verbose = False,
        )
        formatted_document: dict = web_document_combine_retrieve_chain.invoke(
            {"query": search_query,
             "max_tokens": large_context_llm.metadata["retriever_token_limit_calculator"](human_msg_tokens),
             "lang": lang},
            config=config
        )
        tasks_sources= {completed_task_id: completed_task.content.get('files',[]) for completed_task_id,completed_task in runtime_state.task_results.items()}
        context, tasks_sources[int(task_id)] = formatted_document["context"], formatted_document["source"]
        context = context.replace('[Excerpt:', f'[Task: {task_id}, Excerpt:')

        question_chain = response_template | large_context_llm


        log.info("running response with %s", type(large_context_llm))
        inputs={}
        inputs.update(
            {
                "question": llm_question,
                "chat_history": [],
                "context": context,
            }
        )
        inputs.update(large_context_llm.metadata.get('model_profile_for_generation'))
        inputs.update({'user_name':user_profile.get('user_name','Not Provided'),
                       'user_designation':user_profile.get('user_designation','Not Provided'),
                       'user_department':user_profile.get('user_department','Not Provided'),
                       'user_personalization':user_profile.get('user_personalization','Not Provided'),
                       'user_company':user_profile.get('user_company','Not Provided'),
                       'user_specified_behaviour':user_profile.get('user_specified_behaviour',''),
                       'user_country':user_profile.get('user_country','Not Provided')})
        inputs.update({'current_date': date.today()})
        inputs.update({'chat_history':runtime_state.user_chat_history})

        if large_context_llm.metadata.get('stream_able', True):
            response = ''.join([chunk.content for chunk in question_chain.stream(inputs, config=config)])
        else:
            AIMessage = question_chain.invoke(inputs, config=config)
            response = AIMessage.content
        files = tasks_sources
        result_sources = regex_filter(response,files)
        # result_sources = regex_filter(response,doc_source)

        dispatch_custom_event('chat_meta',
                              data=OutFEChatMetaMsg(
                                  task_id=task_id, step_name='task_final', tool_name=self.name,
                                  response=response, files=result_sources, search_phrase=search_query, chat_subject='')
                              )
        return {
                    "response": response,
                    "files": result_sources,
                    "search_phrase": search_query,
                }

    def update_description_for_planning(self, **kwargs) -> None:
        """Update the description of the tool"""
        pass
    def update_description_for_execution(self, **kwargs) -> None:
        """Update the description of the tool"""
        pass



class SearchURLTool(BaseTool):
    name:str = "get_url_data"
    # description = get_prompt_template_prompty("tools", "SearchEnterpriseKnowledgeTool)
    description:str = "useful to fetch data from a specific url"
    args_schema: Type[BaseModel] = SearchInput
    planning_state: Annotated[BaseModel, InjectedState]
    execution_state: Annotated[BaseModel, InjectedState]

    def _run(
        self, query: str, task_id:str, runtime_state: BaseModel, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        raise NotImplementedError
    def update_description_for_planning(self, **kwargs) -> None:
        """Update the description of the tool"""
        pass
    def update_description_for_execution(self, **kwargs) -> None:
        """Update the description of the tool"""
        pass


class SearchAttachmentsTool(BaseTool):
    name: str = "attached_documents"
    # description = get_prompt_template_prompty("tools", "SearchEnterpriseKnowledgeTool)
    description: str = "Useful for when you need to answer questions using documents attached by user."
    args_schema: Type[BaseModel] = SearchInput
    state: Annotated[BaseModel, InjectedState]

    def _run(
        self, query: str,task_id:str, runtime_state: BaseModel, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Union[dict[str, Any], str]:
        """Use the tool."""
        log.info("in SearchAttachmentsTool")
        model_name = self.state.tools_llm.metadata.get('model') or self.state.tools_llm.model_name
        search_query=query
        llm_question=prepare_question(runtime_state=runtime_state, task_id=task_id)
        use_templates=get_templates(['SUBJECT'], 'planner',
                                    model_name)
        k=20

        chat_llm=self.state.tool_args[self.name]["chat_llm"]
        extractor_prompt=use_templates['extractor_prompt']
        retriever=self.state.tool_args[self.name]["retriever"]
        token_calculator = get_token_calculator_for_model(model_name)

        reranker=self.state.tool_args[self.name]["reranker"]
        reranker_bool=self.state.tool_args[self.name]["reranker_bool"]

        tenant_id=self.state.tenant_id
        lang = self.state.lang
        filter=self.state.tool_args[self.name]["filter"]
        user_profile=self.state.additional_metadata["user_profile"]
        lang_expanded = supported_templates[lang]

        log.info('question: %s ,language: %s', search_query, lang)

        subject = "the provided attached documents"
        
        human_msg = extractor_prompt[-1].format(context='', 
                                                lang=lang_expanded, 
                                                question=llm_question, 
                                                subject=subject, 
                                                **user_profile,
                                                **self.state.tool_args[self.name], 
                                                **chat_llm.metadata['model_profile_for_generation'])
        assert isinstance(human_msg, HumanMessage) , 'the message should be a human template, check prompt template'
        human_msg_tokens = chat_llm.get_num_tokens_from_messages([human_msg])

        config = RunnableConfig(callbacks=run_manager.get_child(),
                                metadata={"stage_name": 'answer_generation_chain',
                                          "tool_name":self.name,
                                          "task_id":task_id},
                                configurable={"output_token_number": MAX_TOKENS_CHAT}
                                )
        document_combine_chain = FormatRetrieverDocumentChain(
            retriever=retriever,
            verbose=False,
            token_calculator=token_calculator
        )

        document_output_dict = document_combine_chain.invoke(
            {"query": search_query, 
             "lang":lang,
             "max_tokens": chat_llm.metadata["retriever_token_limit_calculator"](human_msg_tokens),
             "tenant_id": tenant_id,
             "filter":filter,
             "k":k,
             "reranker_bool":reranker_bool,
             "reranker":reranker},
            config=config
        )

        formatted_docs = document_output_dict["context"]
        formatted_docs = formatted_docs.replace('[Excerpt:', f'[Task: {task_id}, Excerpt:')
        doc_source = document_output_dict["source"]
        # doc_source.extend(tasks_sources)
        tasks_sources= {completed_task_id: completed_task.content.get('files',[]) for completed_task_id,completed_task in runtime_state.task_results.items()}
        tasks_sources[int(task_id)]=doc_source
        log.info(f"Format retriver returned: {doc_source}")
        answer_generation_chain = extractor_prompt | chat_llm


        log.info("running answer chain with")
        inputs={}
        inputs.update({
            "subject": subject,
            "question": llm_question,
            "context": formatted_docs,
            "lang": lang_expanded}
        )
        inputs.update(chat_llm.metadata.get('model_profile_for_generation'))
        inputs.update({'user_name':user_profile.get('user_name','Not Provided'),
                       'user_designation':user_profile.get('user_designation','Not Provided'),
                       'user_department':user_profile.get('user_department','Not Provided'),
                       'user_personalization':user_profile.get('user_personalization','Not Provided'),
                       'user_company':user_profile.get('user_company','Not Provided'),
                       'user_specified_behaviour':user_profile.get('user_specified_behaviour',''),
                       'user_country':user_profile.get('user_country','Not Provided')})
        inputs.update({'current_date': date.today()})
        inputs.update({'chat_history':runtime_state.user_chat_history})

        if chat_llm.metadata.get('stream_able', True):
            response = ''.join([chunk.content for chunk in answer_generation_chain.stream(
                inputs, config=config)])

        else:
            AIMessage = answer_generation_chain.invoke(inputs, config=config)
            response = AIMessage.content
        result_sources = regex_filter(response,tasks_sources)

        dispatch_custom_event('chat_meta',
                              data=OutFEChatMetaMsg(
                                  task_id=task_id, step_name='task_final', tool_name=self.name,
                                  response=response, files=result_sources, search_phrase=query, chat_subject='')
                              )

        return {
                    "response": response,
                    "files": result_sources,
                    "search_phrase": query,
                    # 'answer': json.dumps({'response': response, "files": doc_source})})
                }
        
        
    def update_description_for_planning(self, **kwargs) -> None:
        """Update the description of the tool"""
        self.description=f"""Use this tool if question seems directly or indirectly related user provided documents whose summaries are provided below; \
        
Summaries: 
    - {self.state.tool_args[self.name]["attached_file_context"]}.
In case user question contains any demonstrative pronouns vague actions assume it to be related."""

    def update_description_for_execution(self, **kwargs) -> None:
        """Update the description of the tool"""
        self.description=f"""Use this tool if question seems directly or indirectly related user provided documents whose summaries are provided below; \
        
Summaries: 
    {self.state.tool_args[self.name]["attached_file_context"]}.
In case user question contains any demonstrative pronouns vague actions assume it to be related."""




class ParametricTool(BaseTool):
    name:str = "default"
    # description = get_prompt_template_prompty("tools", "SearchEnterpriseKnowledgeTool)
    description:str = "should be called when no other tool is suitable for use. Typical uses include formatting already available info, translation, etc."
    args_schema: Type[BaseModel] = JoinerInput
    state: Annotated[BaseModel, InjectedState]
    def _run(
        self, query: str, task_id:str, runtime_state: BaseModel, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> dict[str, Any]:
        """Use the tool."""
        use_templates=get_templates(['PARAMETRIC','TITLE'], 'planner',
                                    self.state.tools_llm.metadata.get('model') or self.state.tools_llm.model_name)

        # question=query
        question=self.state.task_data[task_id].description
        additional_context=''
        depends_on_task_list=self.state.task_data[task_id].depends_on
        previous_tasks_sources= {completed_task_id: completed_task.content.get('files',[]) for completed_task_id,completed_task in runtime_state.task_results.items()}

        additional_context=''
        depends_on_task_list=self.state.task_data[task_id].depends_on
        depends_on_tasks_sources={}
        if len(depends_on_task_list)>0:
            additional_context='\n**Context:**\n'
            for dependent_task_id in depends_on_task_list:
                depends_on_tasks_sources[str(dependent_task_id)]= runtime_state.task_results[str(dependent_task_id)].content['files']
                additional_context+=str(dependent_task_id)+'. '
                additional_context+=runtime_state.task_data[str(dependent_task_id)].description+'\n'
                additional_context+=str(runtime_state.task_results[str(dependent_task_id)].content['response'])+'\n'
        question+=additional_context

        direct_chain = GeneralChain(
        llm=self.state.tool_args['default']["chat_llm"],
        response_template=use_templates["response_template"],
        title_chain=build_title_chain(self.state.tool_args['default']["llm"], use_templates["title_prompt"]),
        # verbose=verbose
        )
        chain_result = direct_chain.invoke(
            input={
                "chat_subject": "Not required", 
                'question':question,
                'chat_history': [],
                "subject": "a specific field",
                 "user_profile": self.state.additional_metadata["user_profile"],
                 "current_date": date.today()

            },
            config=RunnableConfig(callbacks=run_manager.get_child(),
                                  metadata={"tool_name": self.name,
                                            "task_id": task_id},
                                  tags=[f'{self.name}:{task_id}']
                                  )
        )

        chain_result = chain_result[direct_chain.output_key]
        response=chain_result['response']
        files = depends_on_tasks_sources
        result = self.regex_filter(response,files)
        return {"response": result[0],
                'files':  result[1]}

    def update_description_for_planning(self, **kwargs) -> None:
        """Update the description of the tool"""
        pass

    def update_description_for_execution(self, **kwargs) -> None:
        """Update the description of the tool"""
        pass

    def regex_filter(self,response,files):
        try:
            task_id_sources_map={}
            reg_exp_get_content_between_braces = r'\[Task:(.*?\d),.*?:(.*?\d+)\]'
            reg_exp_get_doc_nums = r'\d+'
            cited_docs = []
            for task_match, excerpt_match in re.findall(reg_exp_get_content_between_braces, response):
                tasks = [task for task in re.findall(reg_exp_get_doc_nums, task_match)]
                excerpts = [task for task in re.findall(reg_exp_get_doc_nums, excerpt_match)]
                for task in tasks:
                    for excerpt in excerpts:
                        cited_docs.append((task,excerpt))
                
            log.debug('extracted documents %s', str(cited_docs))

            log.debug("document parse in regex")
            log.debug(cited_docs)
            cited_docs = set(cited_docs)

            cited_sources = []
            for task_id in files:
                for file in files[task_id]:
                    if (task_id, file[Schema.citation_tag]) in cited_docs:  # and file not in results:
                        cited_sources.append(file)
                    else:
                        log.debug(f'{task_id}, {file[Schema.citation_tag]} not in {cited_docs}')
            return response, cited_sources

        except Exception as e:
            log.error('Can not apply regex', exc_info=True)
            return response, []
    



class JoinnerTool(BaseTool):
    name:str = "joinner"
    # description = get_prompt_template_prompty("tools", "SearchEnterpriseKnowledgeTool)
    description:str = "Use when you need to combine the results of previous tasks. Should be used when no other tool is suitable for use."
    args_schema: Type[BaseModel] = JoinerInput
    state: Annotated[BaseModel, InjectedState]

    def _run(
        self, query: str, task_id:str, runtime_state: BaseModel, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> dict[str, Any]:
        """Use the tool."""
        use_templates=get_templates(['JOINER','TITLE'], 'planner',
                                    self.state.tools_llm.metadata.get('model') or self.state.tools_llm.default.model_name)
        len_citation_tools = len(set([v.tool_name for v in runtime_state.task_data.values()]).intersection({'web_search', 'enterprise','attached_documents'}))
        citation_instructions = joinner_citation_instructions if len_citation_tools>0 else ''
        question=prepare_question(runtime_state=runtime_state, task_id=task_id)

        direct_chain = GeneralChain(
        llm=self.state.tool_args[self.name]["chat_llm"],
        response_template=use_templates["joinner_prompt"],
        title_chain=build_title_chain(self.state.tool_args[self.name]["llm"], use_templates["title_prompt"]),
        # verbose=verbose
        )
        
        user_profile = self.state.additional_metadata["user_profile"]
        self.state.additional_metadata["user_profile"].update({'user_name':user_profile.get('user_name','Not Provided'),
                       'user_designation':user_profile.get('user_designation','Not Provided'),
                       'user_department':user_profile.get('user_department','Not Provided'),
                       'user_personalization':user_profile.get('user_personalization','Not Provided'),
                       'user_company':user_profile.get('user_company','Not Provided'),
                       'user_specified_behaviour':user_profile.get('user_specified_behaviour',''),
                       'user_country':user_profile.get('user_country','Not Provided')})
        chain_result = direct_chain.invoke(
            input={
                "chat_subject": "Not required", 
                'question':question,
                'chat_history': runtime_state.user_chat_history,
                "subject": "a specific field",
                "user_profile": self.state.additional_metadata["user_profile"],
                "current_date": date.today(),
                "joinner_citation_instructions":citation_instructions,
                **self.state.additional_metadata["user_profile"]

            },
            config=RunnableConfig(callbacks=run_manager.get_child(),
                                  metadata={"tool_name": self.name,
                                            "task_id": task_id},
                                  tags=[f'{self.name}:{task_id}']
                                  )
        )

        chain_result = chain_result[direct_chain.output_key]
        response = chain_result['response']
        tasks_sources= {completed_task_id: completed_task.content.get('files',[]) for completed_task_id,completed_task in runtime_state.task_results.items()}
        # files = tasks_sources
        result_sources = regex_filter(response,tasks_sources)

        dispatch_custom_event('chat_meta',
                              data=OutFEChatMetaMsg(
                                  task_id=task_id, step_name='task_final', tool_name=self.name,
                                  response=response, files=result_sources, search_phrase=query, chat_subject='')
                              )


        return {"response": response,
                "tool_input":query,
                'files':  result_sources}

    def update_description_for_planning(self, **kwargs) -> None:
        """Update the description of the tool"""
        pass

    def update_description_for_execution(self, **kwargs) -> None:
        """Update the description of the tool"""
        pass