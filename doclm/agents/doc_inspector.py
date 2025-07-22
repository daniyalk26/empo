import os

from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph

from typing import TypedDict
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI


SCORE_THRESHOLD = os.getenv("AGENT_SCORE_THRESHOLD", "9")


class Reflection(BaseModel):
    critique: str = Field(description="Critique or Recommendation on generated answer")
    question: str = Field(description="Original or updated query if it was necessary")
    score: int = Field(description="Generated Score on quality of Answer")


class DocInspector:

    def __init__(self, answer_chain, reflection_llm: AzureChatOpenAI, reflection_template, max_revisions=1):
        # def build_graph(self, answer_chain, reflection_model, reflection_template) -> CompiledGraph:
        self.answer_chain = answer_chain
        structured_llm = reflection_llm.with_structured_output(Reflection)
        # assert isinstance(chain, AzureChatOpenAI), "Only AzureChatOpenAI models are allowed"

        self.reflection_chain = reflection_template | structured_llm

        self.reflection_template = reflection_template
        self.max_revisions = max_revisions
        # self.max_tokens = max_tokens
        self.graph = self.build_graph()

    class AgentState(TypedDict):
        question: str
        lang: str
        answer: str
        critique: str
        answer_score: str
        files: list
        revision_number: int
        max_revisions: int
        subject_chain_kwargs: dict


    def generate_answer_node(self, state: AgentState) -> dict:
        assert isinstance(state['question'], str), "question must be a string"
        
        input_params = state['subject_chain_kwargs']
        input_params.update({"question": state['question']})

        result = self.answer_chain.invoke(input_params)
        response = result['results']
        return {'answer': response['response'], 'files': response['files']}

    def reflection_node(self, state: AgentState) -> dict:
        response = self.reflection_chain.invoke(
            {"question": state['question'],
             "answer": state['answer']}
        )
        return {
            "critique": response.critique,
            "question": response.question,
            "answer_score": response.score,
            "revision_number": state.get("revision_number", 0) + 1
        }

    @staticmethod
    def should_continue(state: AgentState) -> str:
        assert isinstance(state["revision_number"], int), "revision_number must be an integer"
        assert isinstance(state["max_revisions"], int), "max_revisions must be an integer"

        if state["revision_number"] > state["max_revisions"] or int(state['answer_score']) >= int(SCORE_THRESHOLD):
            return "end"
        return "answer_generate"

    def build_graph(self) -> CompiledGraph:
        builder = StateGraph(self.AgentState)
        builder.add_node("answer_generate", self.generate_answer_node)

        # builder.add_node("reflect", self.reflection_node)
        builder.set_entry_point("answer_generate")
        builder.add_edge("answer_generate", END)
        # builder.add_conditional_edges(
        #     "answer_generate",  # , reflect
        #     self.should_continue,
        #     {"end": END, "answer_generate": "answer_generate"}
        # )
        # builder.add_edge("answer_generate", "reflect")
        return builder.compile()

