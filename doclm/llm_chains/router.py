"""
router chain implementation for selecting between given chains
"""
from __future__ import annotations
import logging

from typing import Any, Dict, List, Mapping, Optional
from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.chains.router.base import MultiRouteChain, RouterChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.prompts import PromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain.chains.base import Chain

log = logging.getLogger("doclogger")
log.disabled = False

# pylint: disable=R0914

class MultiPromptChainCustom(MultiRouteChain):
    """A multi-route chain that uses an LLM router chain to choose amongst prompts."""

    router_chain: RouterChain
    """Chain for deciding a destination chain and the input to it."""
    destination_chains: Mapping[str, Chain]
    """Map of name to candidate chains that inputs can be routed to."""
    default_chain: Chain
    """Default chain to use when router doesn't map input to one of the destinations."""

    @property
    def output_keys(self) -> List[str]:
        return ["text"]

    @classmethod
    def from_prompts(
        cls,
        llm: BaseLanguageModel,
        prompt_infos: List[Dict[str, str]],
        default_chain: Optional[LLMChain] = None,
        **kwargs: Any,
    ) -> MultiPromptChainCustom:
        """Convenience constructor for instantiating from destination prompts."""
        destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
        destinations_str = "\n".join(destinations)
        router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(
            destinations=destinations_str
        )
        router_prompt = PromptTemplate(
            template=router_template,
            input_variables=["input"],
            output_parser=RouterOutputParser(),
        )
        router_chain = LLMRouterChain.from_llm(llm, router_prompt)
        destination_chains = {}
        log.info('Parsing prompt_infos')
        for p_info in prompt_infos:
            name = p_info["name"]
            prompt_template = p_info["prompt_template"]
            prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
            chain = LLMChain(llm=llm, prompt=prompt)
            destination_chains[name] = chain
        _default_chain = default_chain or ConversationChain(llm=llm, output_key="text")

        log.info('Returning router chain')
        return cls(
            router_chain=router_chain,
            destination_chains=destination_chains,
            default_chain=_default_chain,
            **kwargs,
        )
