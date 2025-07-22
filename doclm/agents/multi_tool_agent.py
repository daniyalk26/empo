
import json
from datetime import date
from typing import TypedDict, Annotated, Dict, Literal, Union
from typing import Optional, List, Any

from langgraph.graph import StateGraph, START, END
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import add_messages
from langchain_core.runnables.config import RunnableConfig, merge_configs
from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.messages import convert_to_messages
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.callbacks.manager import (
    dispatch_custom_event,
)
from ..llm_chains.qa_chain import reduce_chats_below_limit, get_token_calculator_for_model
from ..models.models import OutFETaskStepMsg
from ..tools.multi_tool_tools import EnterpriseTool

class GraphExecutor:
    def __init__(self,
                 tool_llm,
                 chat_llm,
                 tools,
                 router_template,
                 default_template,
                 title_chain,
                 ):
        self.graph = self.build_graph()
        self.router_chain = router_template | tool_llm #| JsonOutputParser()
        self.tools = tools
        self.chat_llm = chat_llm

        self.tools_by_name = {t.name: t for t in tools}
        self.title_chain = title_chain
        self.default_template = default_template

    class AgentState(TypedDict):
        message_board: Annotated[list, add_messages]
        question: str
        lang: str
        tool_name: Union[str, None]
        chat_history: List[Any]
        subject: str
        chat_subject: str
        # attachments: Dict[str,Any]
        profile_kwargs: Dict[str, str]
        results: Dict[Any, Any]
        retriever_token_limit: int
        human_msg_tokens: int

    def invoke(self, *args, **kwargs):
        return self.graph.invoke(*args, **kwargs)

    def route_executor(self, state: AgentState, config: RunnableConfig):
        # question = state['question']
        input_ = state.copy()
        input_['chat_history'] = input_['chat_history'][-6:]  # safety check to limit chat history
        input_.update(state.get('profile_kwargs'))
        input_.update(self.chat_llm.metadata.get('model_profile_for_generation'))
        decision = self.router_chain.invoke(input=input_, config=config)
        message = AIMessage(content='', **JsonOutputParser().invoke(decision))

        template = self.default_template
        if  message.tool_name == 'default' and EnterpriseTool.model_fields['name'].default in self.tools_by_name:
            message.tool_name = EnterpriseTool.model_fields['name'].default

        if message.tool_name != 'default':
            if self.tools_by_name[message.tool_name].answer_generation_required:
                template = self.tools_by_name[message.tool_name].template
        dispatch_custom_event('chat_step',
                              data=OutFETaskStepMsg(response=message.tool_name, task_id='0', step_name='route_selected',
                                                    tool_name=message.tool_name)
                              )
        human_msg = template[-1].format(context='', **input_)
        human_msg_tokens = self.chat_llm.get_num_tokens_from_messages([human_msg])

        return {"message_board": message,
                "retriever_token_limit": self.chat_llm.metadata["retriever_token_limit_calculator"](human_msg_tokens),
                "human_msg_tokens": human_msg_tokens,
                "tool_name": message.tool_name
                }

    def tool_execution_node(self, state: AgentState, config: RunnableConfig):
        if messages := state.get("message_board", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        name = message.tool_name
        args = message.args
        for tool_arg in convert_to_openai_tool(self.tools_by_name[name])['function']['parameters']['properties']:
            if arg_temp := args.get(tool_arg) :
                if isinstance(arg_temp, dict):
                    # is a fail-safe
                    args[tool_arg] = arg_temp.get('description', json.dumps(arg_temp))


        _id = message.id
        args['state'] = state
        tool_result = self.tools_by_name[name].invoke(args, config=config)

        return {"message_board": [ToolMessage(content='',
                                              additional_kwargs=tool_result,
                                              # kwargs=kwargs,
                                              tool_name=name,
                                              tool_call_id=_id
                                              )]
                }

    @staticmethod
    def tool_decision_node(state: AgentState, config: RunnableConfig):

        if isinstance(state, list):
            ai_message = state[-1]

        elif messages := state.get("message_board", []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
        if hasattr(ai_message, "tool_name"):
            if ai_message.tool_name != 'default':
                return "tools"
        return 'no_tool'

    def answer_generation_required(self, state: AgentState, config: RunnableConfig):
        if messages := state.get("message_board", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        if self.tools_by_name[message.tool_name].answer_generation_required:
            return 'answer_node'

        return self.title_decision(state)

    def executor_node(self, state: AgentState, config: RunnableConfig):

        if messages := state.get("message_board", []):
            message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")

        input_ = state.copy()
        input_.update(state.get('profile_kwargs'))
        input_.update(self.chat_llm.metadata.get('model_profile_for_generation'))
        context = {}

        template = self.default_template
        if isinstance(message, ToolMessage):

            context = message.content
            if hasattr(message, "additional_kwargs"):
                context = message.additional_kwargs
                input_.update(context)
            else:
                input_['context'] = context

            template = self.tools_by_name[message.tool_name].template

        model_name = self.chat_llm.metadata.get('model', None) or self.chat_llm.model_name

        input_["chat_history"] = reduce_chats_below_limit(
            input_['chat_history'], get_token_calculator_for_model(model_name),
            self.chat_llm.metadata["chat_history_limit_calculator"](input_["human_msg_tokens"]))
        input_["current_date"] = date.today()
        message = template.invoke(input_)
        # answer = self.chat_llm.invoke(message, config=config)
        config = merge_configs(config,
                               RunnableConfig(metadata={'tool_name': state['tool_name'], 'task_id': "0"})
                               )

        if self.chat_llm.metadata.get('stream_able', True):
            content = ''.join(
                [chunk.content for chunk in self.chat_llm.stream(message, config=config,
                                                                 )])

        else:
            ai_message = self.chat_llm.invoke(message, config=config)
            content = ai_message.content
        # [{'response': response, "context":context}]

        return {"message_board": convert_to_messages([dict(role='ai', content=content, **context)])}

    @staticmethod
    def title_decision(state: AgentState):
        if state['chat_subject'] == "New Chat":
            return 'yes'
        return 'no'

    def title_generation(self, state: AgentState, config: RunnableConfig):
        chat_subject = self.title_chain.invoke(state, config=config)
        return {'chat_subject': chat_subject}

    @staticmethod
    def post_process_output(state: AgentState):

        result = {}
        if messages := state.get("message_board", []):
            final_message = messages[-1]
            result['response'] = final_message.content

            if hasattr(final_message, 'additional_kwargs'):
                additional_kwargs = final_message.additional_kwargs
                result['search_phrase'] = additional_kwargs.get("query", None)
                result['lang'] = additional_kwargs.get("lang")
                result['files'] = additional_kwargs.get("source", [])

                result.update(final_message.additional_kwargs)
                if 'final_answer' in additional_kwargs:
                    result.update(additional_kwargs['final_answer'])

        result['chat_subject'] = state.get("chat_subject", None)

        return {"results": result}

    def build_graph(self) -> CompiledGraph:
        graph_builder = StateGraph(self.AgentState)
        graph_builder.add_node("router_node", self.route_executor)
        graph_builder.add_node("answer_node", self.executor_node)
        graph_builder.add_node("tools", self.tool_execution_node)
        graph_builder.add_node("title", self.title_generation)
        graph_builder.add_node("post_process", self.post_process_output)

        graph_builder.set_entry_point("router_node")
        graph_builder.add_conditional_edges(
            "router_node",
            self.tool_decision_node,
            # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
            # It defaults to the identity function, but if you
            # want to use a node named something else apart from "tools",
            # You can update the value of the dictionary to something else
            # e.g., "tools": "my_tools"
            {"tools": "tools",
             "no_tool": "answer_node", },
        )

        graph_builder.add_conditional_edges("tools",
                                            self.answer_generation_required,
                                            {'answer_node': "answer_node",
                                             "yes": "title",
                                             "no": "post_process"})

        graph_builder.add_conditional_edges(
            "answer_node",
            self.title_decision,
            {"yes": "title", "no": "post_process"},
        )

        graph_builder.add_edge("title", "post_process")
        graph_builder.add_edge("post_process", END)

        return graph_builder.compile()
