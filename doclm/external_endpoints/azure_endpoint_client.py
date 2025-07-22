import json
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Type,
    cast,
)
from langchain_core.messages import (
    AIMessageChunk,
)

from langchain_community.chat_models.azureml_endpoint import (
    AzureMLChatOnlineEndpoint,
    AzureMLEndpointApiType,
    CustomOpenAIChatContentFormatter,
    ChatGeneration,
    _convert_delta_to_message_chunk
)

from langchain_core.messages import BaseMessage
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


class AzureMLChatEndpoint(AzureMLChatOnlineEndpoint):
    stop: str = None
    max_tokens: int = None

    def __init__(self, stop=None, max_tokens=None, **kwargs):
        super().__init__(**kwargs)
        self.stop = stop
        self.max_tokens: int = max_tokens

    def _stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        self.endpoint_url = self.endpoint_url.replace("/chat/completions", "")
        timeout = None if "timeout" not in kwargs else kwargs["timeout"]

        import openai

        params = {}
        client_params = {
            "api_key": self.endpoint_api_key.get_secret_value(),
            "base_url": self.endpoint_url,
            "timeout": timeout,
            "default_headers": None,
            "default_query": None,
            "http_client": None,
        }

        client = openai.OpenAI(**client_params)
        message_dicts = [
            CustomOpenAIChatContentFormatter._convert_message_to_dict(m)
            for m in messages
        ]
        params = {"stream": True, "stop": self.stop, "model": None,
                  'max_tokens': self.max_tokens, **kwargs}

        default_chunk_class = AIMessageChunk

        for chunk in client.chat.completions.create(messages=message_dicts, **params):
            if not isinstance(chunk, dict):
                chunk = chunk.dict()
            if len(chunk["choices"]) == 0:
                continue
            choice = chunk.pop("choices")[0]

            generation_info = {}
            if finish_reason := choice.get("finish_reason"):
                generation_info["finish_reason"] = finish_reason
                generation_info.update(chunk)
            logprobs = choice.get("logprobs")
            if logprobs:
                generation_info["logprobs"] = logprobs
            chunk = _convert_delta_to_message_chunk(
                choice["delta"], default_chunk_class
            )
            default_chunk_class = chunk.__class__
            chunk = ChatGenerationChunk(message=chunk,
                                        generation_info=generation_info or None
                                        )
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk


class CustomFormatter(CustomOpenAIChatContentFormatter):

    def format_response_payload(
            self,
            output: bytes,
            api_type: AzureMLEndpointApiType = AzureMLEndpointApiType.dedicated,
    ) -> ChatGeneration:
        """Formats response"""
        if api_type in [
            AzureMLEndpointApiType.dedicated,
            AzureMLEndpointApiType.realtime,
        ]:
            try:
                choice = json.loads(output)["output"]
            except (KeyError, IndexError, TypeError) as e:
                raise ValueError(self.format_error_msg.format(api_type=api_type)) from e
            return ChatGeneration(
                message=BaseMessage(
                    content=choice.strip(),
                    type="assistant",
                ),
                generation_info=None,
            )
        if api_type == AzureMLEndpointApiType.serverless:
            try:
                output_parsed = json.loads(output)
                choice = output_parsed.pop("choices")[0]
                if not isinstance(choice, dict):
                    raise TypeError(
                        "Endpoint response is not well formed for a chat "
                        "model. Expected `dict` but `{type(choice)}` was received."
                    )
            except (KeyError, IndexError, TypeError) as e:
                raise ValueError(self.format_error_msg.format(api_type=api_type)) from e
            return ChatGeneration(
                message=BaseMessage(
                    content=choice["message"]["content"].strip(),
                    type=choice["message"]["role"],
                ),
                generation_info=dict(
                    finish_reason=choice.get("finish_reason"),
                    logprobs=choice.get("logprobs"),
                    **output_parsed
                ),
            )
        raise ValueError(f"`api_type` {api_type} is not supported by this formatter")
