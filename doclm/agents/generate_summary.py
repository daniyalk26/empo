import logging
from typing import TypedDict, Any, Dict

from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from langchain_core.pydantic_v1 import BaseModel, Field

log = logging.getLogger("doclogger")


class OutputFormat(BaseModel):
    critique: str = Field(description="Critique or Recommendation on provided summary")
    score: int = Field(description="Generated Score on quality of summary")


def string_truncation_to_token_limit(text, tokenizer, token_limit):
    encoded = tokenizer.encode(text)
    truncated_encoded = encoded[:token_limit]
    return tokenizer.decode(truncated_encoded)


class SummaryGenerator:
    def __init__(self, model, summary_template, reflection_template, tokenizer, max_tokens, max_revisions=2):
        self.model = model
        self.summary_chain = summary_template | model
        model_with_output_formatter = model.default.with_structured_output(OutputFormat)
        self.reflection_chain = reflection_template | model_with_output_formatter
        self.max_revisions = max_revisions
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.graph = self.build_graph()

    class AgentState(TypedDict):
        input_data: str
        draft: str
        critique: str
        revision_number: int
        max_revisions: int
        doc_lang: str
        score: int

    def generate_summary_node(self, state: AgentState) -> Dict[str, Any]:
        try:
            critique = state.get('critique', '')
            draft = state.get('draft', '')

            # summary_prompt = SUMMARY_PROMPT.format(language=state['doc_lang'])
            content = f"""**Document text:** \n'{state['input_data']}'"""
            if critique != '':
                content = content + f" \n\nHere is the critique: \n{critique}"
            if draft != '':
                content = content + f"\n\nHere is the initial document description: \n{draft}"
                # content = f"""Here is the extracted text:\n{state['input_data']}
                #           \n\nHere is the user-provided critique:\n\n{critique}
                #           \n\nHere is the draft generated in last try:\n\n{draft}"""

            response = self.summary_chain.invoke({"language": state['doc_lang'],
                                                  "content": content})
            log.debug("draft summary %s", response.content)
            return {"draft": response.content,
                    "revision_number": state["revision_number"] + 1,
                    }
        except Exception as e:
            log.error("Error in generate_summary_node: ", exc_info=True)
            raise e

    def reflection_node(self, state: AgentState) -> Dict[str, Any]:
        try:
            content = f"""**Text:** \n{state['input_data']}\n\n
                          **Generated Document Description:**\n{state['draft']}"""
            # reflection_prompt = REFLECTION_PROMPT.format(language=state['doc_lang'])

            response: OutputFormat = self.reflection_chain.invoke({"language": state['doc_lang'],
                                                                   "content": content})
            log.debug("Generated critique %s", response.critique)
            return {
                "critique": response.critique,
                "score": response.score
            }
        except Exception as e:
            log.error("Error in reflection_node: ", exc_info=True)
            raise e

    @staticmethod
    def generate_summary_if_low_score(state: AgentState):

        if state["score"] < 8:
            log.debug("selected to generate summary")
            return "summary"

        log.debug("decided to end ")

        return "end"

    @staticmethod
    def generate_critique_if_max_revision_available(state: AgentState) -> str:
        # if state["revision_number"] > state["max_revisions"] or int(state['answer_score']) >= int(SCORE_THRESHOLD):
        # state["revision_number"] += 1

        if state["revision_number"] < state["max_revisions"]:
            log.debug("selected to evaluated summary generated")
            return "reflect"
        log.debug("decided to end ")

        return "end"

    def build_graph(self) -> CompiledGraph:
        builder = StateGraph(self.AgentState)
        builder.add_node("summary", self.generate_summary_node)
        builder.add_node("reflect", self.reflection_node)
        builder.set_entry_point("summary")
        builder.add_conditional_edges(
            "summary",
            self.generate_critique_if_max_revision_available,
            {"end": "__end__", "reflect": "reflect"}
        )
        builder.add_conditional_edges(
            "reflect",
            self.generate_summary_if_low_score,
            {"end": "__end__", "summary": "summary"}
        )

        # builder.add_edge("reflect", "summary")
        return builder.compile()

    async def agenerate_summary(self, first_pages, doc_lang, config) -> str:

        state = {
            'input_data': self.limit_text_(first_pages),
            'draft': '',
            'critique': '',
            'revision_number': 0,
            'max_revisions': self.max_revisions,
            'doc_lang': doc_lang,
        }
        try:
            result = await self.graph.ainvoke(state, config=config)
            return result['draft']
        except Exception as e:
            log.error("Error in generate_summary: ", exc_info=True)
            raise e


    def generate_summary(self, first_pages, doc_lang, config) -> str:

        state = {
            'input_data': self.limit_text_(first_pages),
            'draft': '',
            'critique': '',
            'revision_number': 0,
            'max_revisions': self.max_revisions,
            'doc_lang': doc_lang,
        }
        try:
            result = self.graph.invoke(state, config=config)
            return result['draft']
        except Exception as e:
            log.error("Error in generate_summary: ", exc_info=True)
            raise e

    def limit_text_(self, text):
        model_name = self.model.default.metadata["model"]
        tokenizer = self.tokenizer(model_name)

        # margin + first_draft_tokens + final_summary_tokens + prompt_tokens + reflection_tokens
        fixed_tokens = 500 + 500 + 500 + 500 + 700

        context_window_tokens = self.model.default.metadata['context_length']
        token_limit = min(3000, (context_window_tokens - fixed_tokens))
        text = string_truncation_to_token_limit(text, tokenizer, token_limit)

        return text
