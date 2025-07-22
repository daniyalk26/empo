
from typing_extensions import Annotated
from typing import Any, Dict, Tuple, Optional, Callable, Type, Union

from typing import Any
from pydantic import BaseModel, Field

from langgraph.prebuilt import InjectedState
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

# Load and chunk contents of the blog
from ..llm_chains.format_retrieve_data import FormatRetrieverDocumentChain, FormatRetrieverWebChain
from ..llm_chains.qa_chain import detect_query_language
from ..agents.plan_and_execute import PlanAndExecute
from ..llm_chains.qa_chain import get_token_calculator_for_model
from ..external_endpoints.azure_llm import get_chat_model, get_llm_common
from ..util import get_templates


class ToolInput(BaseModel):
    query: str = Field(description="should be a search query")
    state: Annotated[dict, InjectedState] = Field(default_factory=dict)


class SearchKwargs(BaseModel):
    k: int
    filter: Union[Dict[Any, Any] ,None] = None
    tenant_id: int
    reranker_bool: Union[bool, None] = None
    reranker:  Union[Any, None] = None


class EnterpriseTool(BaseTool):
    name: str = "enterprise"
    description: str = "Use for retrieving information from documents in the enterprise knowledge base"
    answer_generation_required: bool = True
    lang_detector: Any
    search_kwargs: SearchKwargs
    template: Any
    document_retriever_chain: FormatRetrieverDocumentChain
    args_schema: Type[BaseModel] = ToolInput
    example: str = '''
    example;
    ```json
    {{
        "tool_name": "enterprise",
        "args" :{{
            'query': '<standalone search string in users input language>'
        }}
    }}
    ``` '''
    def __init__(self, retriever, chat_model, enterprise_doc_summaries='', **kwargs: Any):

        description = "Use for retrieving information from documents in the enterprise knowledge base"

        if len(enterprise_doc_summaries) > 0 :
            description = f"""Use this tool for retrieving documents from the enterprise knowledge base. Summaries of some of the documents available in this knowledge base are provided below: \n\t- {enterprise_doc_summaries}
Use this tool for answering queries from these or other such documents. Prefer this tool to `default` wherever appropriate."""

        document_retriever_chain = FormatRetrieverDocumentChain(
            retriever=retriever,
            token_calculator=get_token_calculator_for_model(chat_model),
            verbose=False,
        )

        super().__init__(search_kwargs=SearchKwargs(**kwargs),
                         document_retriever_chain=document_retriever_chain,
                         description=description,
                         **kwargs)


    def _run(
        self, query: str,
            state: Annotated[dict, InjectedState],
            run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> dict[str, Any]:

        """Use the tool."""
        input_state = state.copy()
        chat_history = state.get('chat_history', [])
        input_state.update(state.get('profile_kwargs'))

        input_state['lang'] = detect_query_language(query, chat_history, self.lang_detector, input_state['question'])

        config = RunnableConfig(callbacks=run_manager.get_child(),
                                metadata={"tool_name": self.name,
                                          "task_id": '0'})

        document_output_dict = self.document_retriever_chain.invoke({"query": query,
                                                                "lang": input_state['lang'],
                                                                "max_tokens": input_state['retriever_token_limit'],
                                                                 **self.search_kwargs.model_dump()
                                                                }, config=config)

        document_output_dict['rephrase_query'] = query
        return document_output_dict


class WebTool(BaseTool):
    name: str = "web_search"
    description: str =  """Use this tool when the query requires searching the internet, 
    obtaining current or publicly available information online, or referencing recent news, weather, named entities, or the latest research.
    Prefer this tool to `default` wherever appropriate."""

    answer_generation_required: bool = True
    lang_detector: Any
    search_kwargs: SearchKwargs
    template: Any
    web_document_combine_retrieve_chain: FormatRetrieverWebChain
    args_schema: Type[BaseModel] = ToolInput
    example : str= '''
    example;
    ```json
    {{
        "tool_name": "web_search",
        "args" :{{
            'query': '<standalone search string in users input language>'
        }}
    }}
    ```'''

    def __init__(self, retriever, chat_model, **kwargs: Any):
        web_document_combine_retrieve_chain = FormatRetrieverWebChain(
            retriever=retriever,
            token_calculator=get_token_calculator_for_model(chat_model),
            verbose=False,
        )
        super().__init__(search_kwargs=SearchKwargs(**kwargs),
                         web_document_combine_retrieve_chain=web_document_combine_retrieve_chain,
                         **kwargs)

    def _run(
        self, query: str,
            state: Annotated[dict, InjectedState],
            run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> dict[str, Any]:

        input_state = state.copy()
        chat_history = state.get('chat_history', [])
        input_state.update(state.get('profile_kwargs'))

        input_state['lang'] = detect_query_language(query, chat_history, self.lang_detector, input_state['question'])
        # config['metadata']['stage_name'] = 'rephrase'
        config = RunnableConfig(callbacks=run_manager.get_child(),
                                metadata={"tool_name": self.name,
                                          "task_id": '0'})

        formatted_document = self.web_document_combine_retrieve_chain.invoke(
            {"query": query, "lang": input_state["lang"],
            #  **self.search_kwargs.model_dump(),
             "max_tokens": state['retriever_token_limit']
             },
            config=config,
        )

        formatted_document['rephrase_query'] = query

        return formatted_document


class AttachmentTool(BaseTool):
    name: str = "attached_documents"
    description: str
    answer_generation_required: bool = True
    # rephrase_chain: Callable
    lang_detector: Any
    search_kwargs: SearchKwargs
    template: Any
    document_retriever_chain: FormatRetrieverDocumentChain
    args_schema: Type[BaseModel] = ToolInput
    example : str= '''
    example;
    ```json
    {{
        "tool_name": "attached_documents",
        "args" :{{
            'query': '<standalone search string in users input language>'
        }}
    }}
    ```'''

    def __init__(self, retriever, chat_model, file_context, **kwargs: Any):
        description = f"""Use this tool if question seems directly or indirectly related user provided documents whose summaries are provided below; 
    Summaries: {file_context}.
    In case user question contains any demonstrative pronouns vague actions assume it to be related.
    Prefer this tool to `default` wherever appropriate."""
        document_retriever_chain = FormatRetrieverDocumentChain(
            retriever=retriever,
            token_calculator=get_token_calculator_for_model(chat_model),
            verbose=False,
        )

        super().__init__(search_kwargs=SearchKwargs(**kwargs),
                         document_retriever_chain=document_retriever_chain,
                         description=description,
                         **kwargs)

    def _run(
            self, query: str,
            state: Annotated[dict, InjectedState],
            run_manager: Optional[CallbackManagerForToolRun] = None
            ) -> Any:

        input_state = state.copy()
        chat_history = state.get('chat_history', [])
        input_state.update(state.get('profile_kwargs'))

        input_state['lang'] = detect_query_language(query, chat_history, self.lang_detector, input_state['question'])

        config = RunnableConfig(callbacks=run_manager.get_child(),
                                metadata={"tool_name": self.name,
                                          "task_id": '0'})

        document_output_dict = self.document_retriever_chain.invoke({"query": query,
                                                                "lang": input_state['lang'],
                                                                 **self.search_kwargs.model_dump(),
                                                                "max_tokens": state['retriever_token_limit']
                                                                }, config=config)

        document_output_dict['rephrase_query'] = query
        return document_output_dict


class PlannerTool(BaseTool):
    name :str = 'plan_and_execute'
    description :str = """Use this tool if:
        - The query can be broken down into multiple logical parts.
        - If the query requires a more step by step approach to be answered comprehensively.
        - A comparison is requested (e.g., comparing multiple products, multiple pieces of data, or multiple sets of information).
        - The query is requiring details that spans multiple tools."""
    answer_generation_required: bool = False
    lang_detector: Any
    plan_and_execute_chain: PlanAndExecute
    args_schema: Type[BaseModel] = ToolInput
    example : str= '''
    example;
    ```json
    {{
        "tool_name": "plan_and_execute",
        "args" :{{
            'query': '<actual search string>'
        }}
    }}
    ```'''

    def __init__(self, web_retriever, doc_retriever, advanced_search_retriever, reranker, reranker_bool, tenant_id, chat_model,
                  tools_list, doc_filter, attached_files_filter, attached_file_context, **kwargs):
        tools_list_copy = tools_list.copy()
        try:
            tools_list_copy.remove('plan_and_execute')
        except:
            pass
        tools_list_copy = tools_list_copy + ['joinner']
        templates_to_use = get_templates(['PLANNER'], "planner", chat_model, user_images=None)

        plan_and_execute_chain = PlanAndExecute(
            web_retriever=web_retriever,
            doc_retriever=doc_retriever,
            advanced_search_retriever=advanced_search_retriever,
            planner_template=templates_to_use['initial_question_planner_template'],
            planner_with_chat_history_template=templates_to_use['condenser_planner_template'],
            assign_tools_template=templates_to_use['tool_selector_planner_template'],
            tenant_id=tenant_id,
            planner_llm=get_llm_common(max_tokens=3000, stream=False),
            # executor_llm = get_chat_model(name=chat_model, **kwargs),
            tools_llm=get_chat_model(name=chat_model, **kwargs),
            chat_model=chat_model,
            # lang=lang,
            tools_list=tools_list_copy,
            token_calculator=get_token_calculator_for_model(chat_model),
            reranker=reranker,
            reranker_bool=reranker_bool,
            doc_filter=doc_filter,
            attached_files_filter=attached_files_filter,
            attached_file_context=attached_file_context,
            **kwargs
        )
        super().__init__(plan_and_execute_chain=plan_and_execute_chain,
                         **kwargs)

    def _run(self, query: str,
            state: Annotated[dict, InjectedState],
            run_manager: Optional[CallbackManagerForToolRun] = None )-> dict[str, Any]:
        input_state = state.copy()
        chat_history = state.get('chat_history', [])
        input_state.update(state.get('profile_kwargs'))

        input_state['lang'] = detect_query_language(query, chat_history, self.lang_detector, input_state['question'])
        config = RunnableConfig(callbacks=run_manager.get_child(),
                                metadata={"tool_name": self.name,
                                          "task_id": '0'})

        planner_output_dict = self.plan_and_execute_chain.invoke(
            {
                "query": state['question'],
                "chat_history": chat_history,
                # "tenant_id": self.tenant_id,
                # "tools_list": self.tools_list_copy,
                "lang": input_state['lang'],
                "max_tokens": input_state['retriever_token_limit'],
                "user_profile": state.get('profile_kwargs')
            },
            config=config
        )

        # document_output_dict['rephrase_query'] = rephrase_query
        return planner_output_dict


class GreetingTool(BaseTool):
    name: str = "greeting"
    description: str = "Use this tool for greetings phrases (like hello, hi etc.) and farewell phrases (like thankyou, goodbye, etc.)"
    answer_generation_required: bool = True
    template: Any
    args_schema: Type[BaseModel] = ToolInput
    example: str = '''
    example;
    ```json
    {{
        "tool_name": "greeting",
        "args" :{{
            'query': '<actual search string>'
        }}
    }}
    ``` '''

    def _run(
        self, query: str,
            state: Annotated[dict, InjectedState],
            run_manager: Optional[CallbackManagerForToolRun] = None
    ):
        return dict()
