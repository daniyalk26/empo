
from functools import partial
import logging

from datetime import date

from typing import Optional, List, Dict, Any, Union
from typing_extensions import Annotated
from pydantic import BaseModel, Field

from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables import Runnable
from langchain.tools import BaseTool
from langgraph.graph import StateGraph, END, START

from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks.manager import (
    dispatch_custom_event,
)
from langgraph.graph.graph import CompiledGraph

from langchain_openai import AzureChatOpenAI
from langchain_core.retrievers import BaseRetriever

from ..external_endpoints.reranker import InfinityRerank
from ..tools.plan_and_execute_tools import SearchEnterpriseKnowledgeTool, SearchAttachmentsTool, SearchWebTool, ParametricTool, JoinnerTool
from ..external_endpoints.azure_llm import get_llm_common, get_chat_model
from ..models.models import OutFETaskStepMsg
from ..config import supported_templates


log = logging.getLogger("doclogger")
log.disabled = False


def dict_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    if not dict1:
        return dict2
    if not dict2:
        return dict1

    return {**dict1, **dict2}


def list_merge(list1: List[Any], list2: List[Any]) -> List[Any]:
    if not list1:
        return list2
    if not list2:
        return list1
    return list1 + list2


class Task(BaseModel):
    """Individual tasks in the execution plan"""
    id: int = Field(description="The task number, an integer starting from 1 and incrementing by 1 for each task")
    objective: str = Field(description="should be the objective that this task fulfills")
    description: str = Field(description="The action to be taken in the task")
    depends_on: List[int] = Field(default_factory=list,
                                  description="A list of task ids that must be completed before this task")
    tool_name: Union[str,None] = None


class TaskResult(BaseModel):
    """Result of a task execution"""
    id: int = Field(description="The task number whose result it contains")
    content: Union[str, dict] = Field(description="Actually result of the tool")
    metadata: List[Dict[Any, Any]] = Field(default_factory=list)


class Plan(BaseModel):
    """Complete execution plan"""
    description: str = Field(description="Overall plan description and strategy")
    goal: str = Field(description="Unambiguous goal that the plan is supposed to accomplish")
    task_list: List[Task] = Field(description="Ordered list of tasks to be executed")


class PlannerState(BaseModel):
    """State for the planning stage with requirements gathering and message history"""

    class Config:
        arbitrary_types_allowed = True
        extra="allow"

    messages: Annotated[List[Dict[str, str]], list_merge] = Field(default_factory=list)  # Chat history
    query: str
    tenant_id: int
    context: Annotated[Dict[str, Any], dict_merge] = Field(default_factory=dict)
    tools: Annotated[List[BaseTool],list_merge] = Field(default_factory=list)
    selected_plan: Optional[Plan] = None
    final_answer: Optional[str] = None
    user_chat_history: Optional[List[Any]] = Field(default_factory=list)
    additional_metadata: Annotated[Dict[str, Any], dict_merge] = Field(default_factory=dict)
    planner_llm: Any = Field(description="LLM to use for planning")
    tool_args: Annotated[Dict[str, Any], dict_merge] = Field(default_factory=dict)
    lang: str = Field(description="Language in which response is expected")


class RuntimeState(BaseModel):
    """Runtime state for execution stage with message history support"""

    class Config:
        arbitrary_types_allowed = True

    messages: Annotated[List[Dict[str, str]], list_merge] = Field(default_factory=list)  # Chat history using list_merge
    plan: Plan
    tools: Annotated[List[BaseTool], list_merge] = Field(default_factory=list)
    user_chat_history: List = Field(default_factory=[])
    user_question: str = Field(default='')
    task_data: Dict[str, Task] = Field(default_factory=dict)
    tenant_id: int = Field(description='tenant id')
    task_results: Annotated[Dict[str, TaskResult], dict_merge] = Field(default_factory=dict)
    tools_dict: Annotated[Dict[str, BaseTool], dict_merge] = Field(default_factory=dict)
    final_answer: Union[str,Any] = Field(description='Final result after the execution of all task', default="")
    tools_llm: Any = Field(description="LLM to use for tool's final response")
    additional_metadata: Dict[str, Any] = Field(default_factory=dict)
    tool_args: Dict[str, Any] = Field(default_factory=dict)
    lang: str = Field(description="Language in which response is expected")


class PlanAndExecute:
    def __init__(
            self, web_retriever: BaseRetriever,
            doc_retriever: BaseRetriever,
            advanced_search_retriever: BaseRetriever,
            reranker: InfinityRerank,
            tenant_id: int,
            planner_llm: AzureChatOpenAI,
            tools_llm: AzureChatOpenAI,
            reranker_bool: bool,
            planner_template: Runnable,
            planner_with_chat_history_template: Runnable,
            assign_tools_template: Runnable,
            doc_filter: Dict,
            attached_files_filter: Dict,
            tools_list: List[str],
            attached_file_context: List,
            **kwargs):
        self.web_retriever = web_retriever
        self.doc_retriever = doc_retriever
        self.advanced_search_retriever=advanced_search_retriever
        self.reranker = reranker
        self.reranker_bool = reranker_bool
        self.tenant_id = tenant_id
        self.planner_llm = planner_llm
        self.tools_list = tools_list
        self.planner_template = planner_template
        self.tools_llm = tools_llm
        self.assign_tools_template = assign_tools_template
        self.doc_filter = doc_filter
        self.attached_files_filter = attached_files_filter
        self.planner_with_chat_history_template = planner_with_chat_history_template
        self.attached_file_context = attached_file_context
        self.extra_args = kwargs

        self.__post_init__()

    def __post_init__(self):

        pass

    def build_tool(self, tool: str, state):
        if tool == 'joinner':
            if tool not in state.tool_args:
                state.tool_args[tool] = {}
                state.tool_args[tool]['llm'] = get_llm_common()
                state.tool_args[tool]['chat_llm'] = self.tools_llm
                state.tool_args[tool]['retriever'] = self.web_retriever
            return JoinnerTool(state=state)
        elif tool == 'enterprise':
            if tool not in state.tool_args:
                state.tool_args[tool] = {}
                state.tool_args[tool]['llm'] = get_llm_common()
                state.tool_args[tool]["chat_llm"] = self.tools_llm
                state.tool_args[tool]['reranker'] = self.reranker
                state.tool_args[tool]['reranker_bool'] = self.reranker_bool
                state.tool_args[tool]['filter'] = self.doc_filter
                state.tool_args[tool]['retriever'] = self.doc_retriever
                state.tool_args[tool]['adv_search_retriever']=self.advanced_search_retriever
            return SearchEnterpriseKnowledgeTool(state=state)

        elif tool == 'attached_documents':
            if tool not in state.tool_args:
                state.tool_args[tool] = {}
                state.tool_args[tool]['llm'] = get_llm_common()
                state.tool_args[tool]["chat_llm"] = self.tools_llm
                state.tool_args[tool]['reranker'] = self.reranker
                state.tool_args[tool]['reranker_bool'] = self.reranker_bool
                state.tool_args[tool]['retriever'] = self.doc_retriever
                state.tool_args[tool]['attached_file_context'] = self.attached_file_context
                state.tool_args[tool]['filter'] = self.attached_files_filter
            return SearchAttachmentsTool(state=state)

        elif tool == 'web_search':
            if tool not in state.tool_args:
                state.tool_args[tool] = {}
                state.tool_args[tool]['llm'] = get_llm_common()
                state.tool_args[tool]['chat_llm'] = self.tools_llm
                state.tool_args[tool]['retriever'] = self.web_retriever
            return SearchWebTool(state=state)

        elif tool == 'default':
            if tool not in state.tool_args:
                state.tool_args[tool] = {}
                state.tool_args[tool]['llm'] = get_llm_common()
                state.tool_args[tool]['chat_llm'] = self.tools_llm
                state.tool_args[tool]['retriever'] = self.web_retriever
            return ParametricTool(state=state)
        raise ValueError(f"tool {tool} not implement")

    def build_planner_graph(self) -> CompiledGraph:
        graph = StateGraph(PlannerState)
        graph.add_node("generate_plan", self.generate_plan)
        graph.set_entry_point("generate_plan")
        graph.add_edge("generate_plan", END)
        return graph.compile()

    def build_executor_graph(self, plan: Plan) -> CompiledGraph:
        graph = StateGraph(RuntimeState)
        task_data = {str(task.id): task for task in plan.task_list}
        # logger.debug(f"Initialized step_data with keys: {list(step_data.keys())}")
        log.info(f"Initialized step_data with keys: {list(task_data.keys())}")
        for task in plan.task_list:
            node_name = f"task_{task.id}"
            node_fn = partial(
                self.execute_step,
                task_id=str(task.id)
            )
            graph.add_node(node_name, node_fn)
            log.info(f"Added node: {node_name}")
        for task in plan.task_list:
            node_name = f"task_{task.id}"

            if task.depends_on:
                # Add edges from dependencies
                dep_node_name_list = [f"task_{dep_id}" for dep_id in task.depends_on]
                graph.add_edge(dep_node_name_list, node_name)
                log.info(f"Added edge: {dep_node_name_list} -> {node_name}")
            else:
                # Connect to START if no dependencies
                graph.add_edge(START, node_name)
                log.info(f"Added edge: START -> {node_name}")
        all_dependencies = {dep for task in plan.task_list for dep in task.depends_on}
       
        terminal_tasks_name_list=[
            f"task_{task.id}" for task in plan.task_list
            if task.id not in all_dependencies
        ]
        if len(terminal_tasks_name_list)==0:
            raise RuntimeError("Unable to generate valid graph")
        graph.add_edge(terminal_tasks_name_list, END)
        log.info(f"Added edge: {terminal_tasks_name_list} -> END")

        return graph.compile()

    def invoke(self, inputs, config: Optional[RunnableConfig] = None) -> dict[str, Union[dict[Any, Any], Any]]:
        query = inputs['query']
        lang = inputs['lang']
        chat_history = inputs['chat_history']

        tools_list = self.tools_list            # inputs['tools_list']
        callbacks=config['callbacks'].handlers
        planner_state: PlannerState = PlannerState(query=query, lang=lang, user_chat_history=chat_history,tenant_id=self.tenant_id,
                                                   planner_llm=self.planner_llm,additional_metadata={'user_profile':inputs["user_profile"]})
        planner_tools_list = [self.build_tool(_tool, planner_state) for _tool in tools_list]
        _ = [tool.update_description_for_planning(callbacks=callbacks) for tool in planner_tools_list]
        planner_state.tools = planner_tools_list
        planner_graph: CompiledGraph = self.build_planner_graph()
        planner_result = planner_graph.invoke(planner_state, config=config)
        plan = planner_result['selected_plan']

        task_data = {str(task.id): task for task in plan.task_list}

        execution_state: RuntimeState = RuntimeState(plan=plan, tenant_id=self.tenant_id, task_data=task_data, user_chat_history=chat_history, user_question=query,
                                                     tools_llm=self.tools_llm, lang=lang)
        executor_tools_list = [self.build_tool(_tool, execution_state) for _tool in tools_list]
        _ = [tool.update_description_for_execution(callbacks=callbacks) for tool in executor_tools_list]
        execution_state.tools = executor_tools_list
        execution_state.additional_metadata['user_profile'] = inputs["user_profile"]

        execution_state.tools_dict = {_tool.name: _tool for _tool in executor_tools_list}

        # config['callbacks'].on_custom_event('plan', data=plan)
        dispatch_custom_event('chat_step',
                              data=OutFETaskStepMsg(response=plan.model_dump_json(), task_id='0', step_name='plan',
                                                    tool_name='plan_and_execute')
                              )

        executor_graph: CompiledGraph = self.build_executor_graph(plan)
        executor_result = executor_graph.invoke(execution_state, config=config)
        return {
            "plan":executor_result['plan'].model_dump(), 
            "task_info":{k:v.model_dump() for k,v in executor_result['task_data'].items()}, 
            "task_result":{k:v.model_dump() for k,v in executor_result["task_results"].items()},
            "final_answer":executor_result["final_answer"]
            }
                                            

    def generate_plan(self, state: PlannerState) -> Dict[str, Any]:
        try:
            
            # parser = JsonOutputParser()
            if len(state.user_chat_history) == 0:
                chain = self.planner_template | self.planner_llm| StrOutputParser()|self.planner_llm.with_structured_output(Plan,method='function_calling')
            else:
                chain = self.planner_with_chat_history_template | self.planner_llm | StrOutputParser() | self.planner_llm.with_structured_output(Plan,method='function_calling')
            tools_place_holder_str = self.get_tools_description_str(state.tools)
            lang_expanded = supported_templates[state.lang]
            chain_inputs = {
                "query": state.query,
                "tools_and_descriptions": tools_place_holder_str,
                "chat_history": state.user_chat_history,
                "lang":lang_expanded
            }
            chain_inputs.update({'current_date':date.today()})
            chain_inputs.update(state.additional_metadata['user_profile'])
            chain_inputs.update(self.planner_llm.metadata.get('model_profile_for_generation'))
            plan = chain.invoke(chain_inputs)
            # plan=Plan(**plan)
            log.info(f"plan is {plan.model_dump()}")
            if not plan.task_list:
                raise ValueError("Generated plan contains no steps")
            for task in plan.task_list:
                if not task.description or not task.objective:
                    raise ValueError(f"Invalid step structure in generated plan: {task}")

                # Validate dependencies exist
                if task.depends_on:
                    step_ids = {s.id for s in plan.task_list}
                    invalid_deps = [dep for dep in task.depends_on if dep not in step_ids]
                    if invalid_deps:
                        raise ValueError(f"Step {task.id} has invalid dependencies: {invalid_deps}")

            return {"selected_plan": plan}
        except Exception as e:
            log.warning(f"Error creating plan: {e}", exc_info=True)
            try:
                # Create minimal fallback plan
                fallback_plan = Plan(
                    goal="To generate the most precise answer to user question",
                    task_list=[
                        Task(
                            id=1,
                            objective=state.query,
                            description="Analyze gathered requirements",
                        )
                    ],
                    description="Fallback execution plan"
                )

                # Validate fallback plan
                if not fallback_plan.task_list:
                    raise ValueError("Failed to create valid fallback plan")

                # logger.info("Created fallback plan due to original plan failure")
                return {"selected_plan": fallback_plan}

            except Exception as fallback_error:
                log.error(f"Failed to create fallback plan: {fallback_error}", exc_info=True)
                raise ValueError(f"Could not create plan or fallback: {str(e)} -> {str(fallback_error)}")

    def assign_tool(self, state: RuntimeState, task_id: str) -> BaseTool:
        tools_list = state.tools
        structured_llm = self.planner_llm.bind_tools(tools_list, tool_choice=True)
        chain = self.assign_tools_template | structured_llm
        tools_str = "\n- ".join(
            [
                f"{t.name}: {t.description}\n\t[args:{str(convert_to_openai_tool(t)['function']['parameters']['properties']).replace('{', '{{').replace('}', '}}')}]"
                for t in tools_list])
        overall_goal = state.plan.goal
        context = ""
        for pre_req_task in state.task_data[task_id].depends_on:
            pre_req_task_description = state.task_data[str(pre_req_task)].description
            pre_req_task_description_result = state.task_results[str(pre_req_task)].content
            context += f"Task: {pre_req_task_description}\nResult: {pre_req_task_description_result}"
        result = chain.invoke({"context": context if len(context) else "No context available",
                               "overall_goal": overall_goal,
                               "task_description": state.task_data[task_id].model_dump_json(),
                               "tools_place_holder": tools_str})
        tool_call_details = result.tool_calls[0]
        return tool_call_details

    def is_task_ready(self, state: RuntimeState, task_id: str) -> bool:
        task = state.task_data[task_id]
        pre_req_tasks = task.depends_on
        for pre_req_task in pre_req_tasks:
            if str(pre_req_task) not in state.task_results:
                log.warning(f'pending task {pre_req_task}, cannot execute {task_id}')
                return False
        return True

    def execute_step(self, state: RuntimeState, task_id: str) -> Dict[str, Any]:
        # get details for the current task
        if not self.is_task_ready(state, task_id):
            log.info(f"Waiting in task_{task_id}")
            log.warning("Task invoked before dependencies resolved", state.task_results)
            raise RuntimeError('failed as invoked before pre-req execution')
            # else:
            #     break

        log.info(f"Executing task_{task_id}")
        tool_call_details = self.assign_tool(state, task_id)
        tool_selected = state.tools_dict[tool_call_details["name"]]
        state.task_data[task_id].tool_name=tool_call_details["name"]
        tool_call_details["args"]["task_id"] = task_id
        tool_call_details["args"]["runtime_state"]=state
        tool_result = tool_selected.invoke(tool_call_details["args"])
        result = {"task_results": {task_id: TaskResult(id=int(task_id), content=tool_result, metadata=[{}])}}
        if int(task_id) == max([int(s_id) for s_id in state.task_data.keys()]):
            result["final_answer"] = tool_result
            #files, response,
            # result.update(tool_result)
        return result

    def get_tools_description_str(self, tools_list: List[BaseTool]) -> str:
        tools_place_holder_str = ""
        for idx, tool in enumerate(tools_list, start=1):
            tool_place_holder_str = """{idx}. Name: {name}
Description: {description}

""".format(idx=idx, name=tool.name, description=tool.description)
            tools_place_holder_str += tool_place_holder_str
        return tools_place_holder_str
