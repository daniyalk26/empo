import logging
from typing import List, Dict, Any
# from transformers import AutoTokenizer

import tiktoken

from langchain_openai.chat_models.base import _url_to_size, _count_image_tokens, _convert_message_to_dict


log = logging.getLogger("doclogger")
log.disabled = False

TOKENIZERS = {}


def _get_encoding_model(model) -> tiktoken.Encoding:
    global TOKENIZERS
    if model not in TOKENIZERS:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            log.warning(f"encoding for model {model} not fount using default model. using  cl100k_base encoding ")
            if 'o1' in model or 'gpt-4' in model:
                model = "o200k_base"
            else:
                model = "cl100k_base"

            if model in TOKENIZERS:
                return TOKENIZERS[model]
            encoding = tiktoken.get_encoding(model)

        TOKENIZERS[model] = encoding

    return TOKENIZERS[model]


def num_tokens_from_messages(
        messages:List[Any],
        model='gpt-4o',
        tools = None,
) -> int:
    """Calculate num tokens for gpt-3.5-turbo and gpt-4 with tiktoken package.

    **Requirements**: You must have the ``pillow`` installed if you want to count
    image tokens if you are specifying the image as a base64 string, and you must
    have both ``pillow`` and ``httpx`` installed if you are specifying the image
    as a URL. If these aren't installed image inputs will be ignored in token
    counting.

    OpenAI reference: https://github.com/openai/openai-cookbook/blob/
    main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb

    Args:
        messages: The message inputs to tokenize.
        model: The message inputs to tokenize.
        tools: If provided, sequence of dict, BaseModel, function, or BaseTools
            to be converted to tool schemas.
    """
    # TODO: Count bound tools as part of input.
    if tools is not None:
        log.warning(
            "Counting tokens in tool schemas is not yet supported. Ignoring tools."
        )
    encoding = _get_encoding_model(model)
    if model.startswith("gpt-3.5-turbo-0301"):
        # every message follows <im_start>{role/name}\n{content}<im_end>\n
        tokens_per_message = 4
        # if there's a name, the role is omitted
        tokens_per_name = -1
    elif model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        tokens_per_name = 1
        tokens_per_message = 3

    num_tokens = 0
    messages_dict = [_convert_message_to_dict(m) for m in messages]
    for message in messages_dict:
        num_tokens += tokens_per_message
        for key, value in message.items():
            # This is an inferred approximation. OpenAI does not document how to
            # count tool message tokens.
            if key == "tool_call_id":
                num_tokens += 3
                continue
            if isinstance(value, list):
                # content or tool calls
                for val in value:
                    if isinstance(val, str) or val["type"] == "text":
                        text = val["text"] if isinstance(val, dict) else val
                        num_tokens += len(encoding.encode(text))
                    elif val["type"] == "image_url":
                        if val["image_url"].get("detail") == "low":
                            num_tokens += 85
                        else:
                            image_size = _url_to_size(val["image_url"]["url"])
                            if not image_size:
                                continue
                            num_tokens += _count_image_tokens(*image_size)
                    # Tool/function call token counting is not documented by OpenAI.
                    # This is an approximation.
                    elif val["type"] == "function":
                        num_tokens += len(
                            encoding.encode(val["function"]["arguments"])
                        )
                        num_tokens += len(encoding.encode(val["function"]["name"]))
                    elif val["type"] == "file":
                        log.warning(
                            "Token counts for file inputs are not supported. "
                            "Ignoring file inputs."
                        )
                        pass
                    else:
                        raise ValueError(
                            f"Unrecognized content block type\n\n{val}"
                        )
            elif not value:
                continue
            else:
                # Cast str(value) in case the message value is not a string
                # This occurs with function messages
                num_tokens += len(encoding.encode(str(value)))
            if key == "name":
                num_tokens += tokens_per_name
    # every reply is primed with <im_start>assistant
    num_tokens += 3
    return num_tokens


def num_tokens_from_string(text: str, model_name):
    encoding = _get_encoding_model(model_name)

    return len(encoding.encode(text))
