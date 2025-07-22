# our name: openai _names

# formula to calculate token limits
#     context window - Margin - prompt_tokens - out tokens = remaining tokens
#     3:5 :: chat history: retrieved docs =remaining tokens
import math

margin_tokens=500
longest_output_tokens=4096
longest_prompt_tokens=800

chat_model_context_lengths = {
    # "gpt-3.5-turbo": 4096,
    # "gpt-3.5-turbo-16k": 16000,
    # "gpt-4-32k" : 32000,
    # "gpt-4-turbo": 128000,
    "gpt-4-1": 1047576,
    "gpt-4o" : 128000,
    # "mistral-large-2" : 128000,
    "mistral-large-2411" : 128000,
    "llama-3-70b-instruct" : 8000,
    "llama-3-1-70b-instruct" : 128000,
    "llama-3-1-8b-instruct" : 128000,
    "llama-3-1-408b-instruct": 128000,
    "llama-3-2-90b-instruct": 128000,
    "cohere" : 128000,
    # "o1-preview": 128000,
    "o1": 128000,
    # "o1-mini": 128000,
    "o4-mini": 128000,
    "deepseek-v3": 128000,
    "gemini-2.0-flash": 1000000,
}

def get_chat_history_token_limit(model, _longest_output_tokens=longest_output_tokens):
    def wrapper(input_tokens):
        remaining_tokens = chat_model_context_lengths[model] - input_tokens
        return math.floor(3*(remaining_tokens-margin_tokens-_longest_output_tokens-longest_prompt_tokens)/8)
    return wrapper


def get_retrieved_docs_token_limit(model, _longest_output_tokens=longest_output_tokens):
    def wrapper(input_tokens):
        remaining_tokens = chat_model_context_lengths[model] - input_tokens

        return math.floor(5*(remaining_tokens-margin_tokens-_longest_output_tokens-longest_prompt_tokens)/8)

    return wrapper

default_chat_roles = {
    'instructions': 'system',
    'user': 'human',
    'llm': 'assistant'
}
chat_model_config = {
    "o1": {
        "type": "azure-openai",
        "retriever_token_limit_calculator": get_retrieved_docs_token_limit("o1", 20000),
        "chat_history_limit_calculator": get_chat_history_token_limit("o1", 20000),
        # "model_name": "o1-preview",
        "stream_able": True,

        "model_profile_for_generation": {
            "knowledge_cutoff_date": ' October 1, 2023',
            "model_preffered_name": "o1",
            "model_developed_by": "OpenAI",
            "model_provided_through": "Azure"
        },
        "model_kwargs": {
            "deployment_name_envar": "O1_DEPLOYMENT_NAME",
            "model_name_envar": "O1_MODEL_NAME",
            "temperature": 1,
            "stream": False,
            "max_completion_tokens":20000,
            "tiktoken_model_name": "gpt-4o" #o1 uses same o200k_base encoding model
        },
        "chat_roles": {
            "instructions": "assistant",
            "user": "human",
            "llm": "assistant"
        },

    },
    "o4-mini": {
        "type": "azure-openai",
        "retriever_token_limit_calculator": get_retrieved_docs_token_limit("o4-mini", 20000),
        "chat_history_limit_calculator": get_chat_history_token_limit("o4-mini", 20000),
        # "model_name": "o1-mini",
        "stream_able": True,
        "model_profile_for_generation": {
            "knowledge_cutoff_date": 'October 1, 2023',
            "model_preffered_name": "o1-mini",
            "model_developed_by": "OpenAI",
            "model_provided_through": "Azure"
        },
        "model_kwargs": {
            "deployment_name_envar": "O4_MINI_DEPLOYMENT_NAME",
            "model_name_envar": "O4_MINI_MODEL_NAME",
            "temperature": 1,
            "max_completion_tokens": 20000,
            "tiktoken_model_name": "gpt-4o"  #this uses same o200k_base encoding model
        },
        "chat_roles": {
            "instructions": "assistant",
            "user": "human",
            "llm": "assistant"
        },
    },

    "gpt-4-1": {
        "type": "openai",
        "retriever_token_limit_calculator": get_retrieved_docs_token_limit("gpt-4-1", 32768) ,
        "chat_history_limit_calculator": get_chat_history_token_limit("gpt-4-1", 32768),
        "model_profile_for_generation": {
            "knowledge_cutoff_date":"June 2024",
            "model_preffered_name": "GPT 4.1",
            "model_developed_by": "OpenAI",
            "model_provided_through": "Azure"
        },
        "model_kwargs": {
            "deployment_name_envar": "DIRECT_CHAT_DEPLOYMENT_NAME_4_1",
            "model_name_envar": "DIRECT_CHAT_MODEL_NAME_4_1",
            "temperature": 0.0,
            "stream": True,
            "max_tokens": 3000,
            "tiktoken_model_name": "gpt-4o" #this uses same o200k_base encoding model
        }
    },
    "gpt-4o": {
        "type": "openai",
        "retriever_token_limit_calculator": get_retrieved_docs_token_limit("gpt-4o"),
        "chat_history_limit_calculator": get_chat_history_token_limit("gpt-4o"),
        "context_length": 128000,
        "model_profile_for_generation": {
            "knowledge_cutoff_date":'October 2023',
            "model_preffered_name": "GPT-4o (Omni)",
            "model_developed_by": "OpenAI",
            "model_provided_through": "Azure",
        },
        "model_kwargs": {
            "deployment_name_envar": "DIRECT_CHAT_DEPLOYMENT_NAME_4o",
            "model_name_envar": "DIRECT_CHAT_MODEL_NAME_4o",
            "temperature": 0.0,
            "stream": True,
            "max_tokens": 3000

        }
    },
    "gemini-2.0-flash": {
        "type": "google",
        "retriever_token_limit_calculator": get_retrieved_docs_token_limit("gemini-2.0-flash"),
        "chat_history_limit_calculator": get_chat_history_token_limit("gemini-2.0-flash"),
        "context_length": 1000000,
        "model_profile_for_generation": {
            "knowledge_cutoff_date": 'December, 2024',
            "model_preffered_name": "Gemini 2.0 Flash",
            "model_developed_by": "Google",
            "model_provided_through": "GCP",
        },
        "model_kwargs": {
            "model_name_envar": "DIRECT_CHAT_MODEL_NAME_flash_2o",
            "temperature": 0.0,
            "stream": True,
        }
    },
    "mistral-large-2411": {
        "type": "azure",
        "retriever_token_limit_calculator": get_retrieved_docs_token_limit("mistral-large-2411"),
        "chat_history_limit_calculator": get_chat_history_token_limit("mistral-large-2411"),
        "model_profile_for_generation": {
            "knowledge_cutoff_date":'Unknown',
            "model_preffered_name": "Mistral-Large-2411",
            "model_developed_by": "Mistral",
            "model_provided_through": "Azure"
        },
        "model_kwargs": {
            "base_url_envar": "MISTRAL_END_POINT",
            "api_key_envar": "MISTRAL_API_KEY",
            "stop": None,
            "deployment_name": "Mistral-Large-2",
            "end_point_version": '/v1/chat/completions',
            "max_tokens": 3000
        }
    },
    "llama-3-70b-instruct": {
        "type": "azure",
        "retriever_token_limit_calculator": get_retrieved_docs_token_limit("llama-3-70b-instruct"),
        "chat_history_limit_calculator": get_chat_history_token_limit("llama-3-70b-instruct"),
        "model_profile_for_generation": {
            "knowledge_cutoff_date":'December 2023',
            "model_preffered_name": "Llama 3 70B",
            "model_developed_by": "Meta",
            "model_provided_through": "Azure"
        },
        "model_kwargs": {
            "base_url_envar": "LLAMA_END_POINT",
            "api_key_envar": "LLAMA_API_KEY",
            "stop": None,
            "deployment_name": "llama-3-70B-Instruct",
            "end_point_version": '/v1/chat/completions',
            "max_tokens": 3000
        }
    },
    "llama-3-1-70b-instruct": {
        "type": "azure",
        "retriever_token_limit_calculator": get_retrieved_docs_token_limit("llama-3-1-70b-instruct"),
        "chat_history_limit_calculator": get_chat_history_token_limit("llama-3-1-70b-instruct"),
        "model_profile_for_generation": {
            "knowledge_cutoff_date": 'December 1, 2023',
            "model_preffered_name": "Llama 3.1 70B",
            "model_developed_by": "Meta",
            "model_provided_through": "Azure"
        },
        "model_kwargs": {
            "base_url_envar": "LLAMA_31_70B_END_POINT",
            "api_key_envar": "LLAMA_31_70B_API_KEY",
            "stop": None,
            "deployment_name": "llama-3-1-70B-Instruct",
            "end_point_version": '/v1/chat/completions',
            "max_tokens": 3000
        }
    },
    "llama-3-1-8b-instruct": {
        "type": "azure",
        "retriever_token_limit_calculator": get_retrieved_docs_token_limit("llama-3-1-8b-instruct"),
        "chat_history_limit_calculator": get_chat_history_token_limit("llama-3-1-8b-instruct"),
        "model_profile_for_generation": {
            "knowledge_cutoff_date":'December 2023',
            "model_preffered_name": "Llama 3.1 8B",
            "model_developed_by": "Meta",
            "model_provided_through": "Azure"
        },
        "model_kwargs": {
            "base_url_envar": "LLAMA_31_END_POINT",
            "api_key_envar": "LLAMA_31_API_KEY",
            "stop": None,
            "deployment_name": "llama-3-1-8B-Instruct",
            "end_point_version": '/v1/chat/completions',
            "max_tokens": 3000
        }
    },
    "llama-3-1-408b-instruct": {
        "type": "azure",
        "retriever_token_limit_calculator": get_retrieved_docs_token_limit("llama-3-1-408b-instruct"),
        "chat_history_limit_calculator": get_chat_history_token_limit("llama-3-1-408b-instruct"),
        "model_profile_for_generation": {
            "knowledge_cutoff_date": 'December 2023',
            "model_preffered_name": "Llama-3-1-405B",
            "model_developed_by": "Meta",
            "model_provided_through": "Azure"
        },
        "model_kwargs": {
            "base_url_envar": "LLAMA_31_408_END_POINT",
            "api_key_envar": "LLAMA_31_408_API_KEY",
            # "stop": None,
            "deployment_name": "llama-3-1-408B-Vision-Instruct",
            "end_point_version": '/v1/chat/completions',
            "max_tokens": 3000
        },

    },
    "llama-3-2-90b-instruct": {
        "type": "azure",
        "retriever_token_limit_calculator": get_retrieved_docs_token_limit("llama-3-2-90b-instruct"),
        "chat_history_limit_calculator": get_chat_history_token_limit("llama-3-2-90b-instruct"),
        "model_profile_for_generation": {
            "knowledge_cutoff_date": 'December 1, 2023',
            "model_preffered_name": "Llama-3-2-90B-Vision-Instruct",
            "model_developed_by": "Meta",
            "model_provided_through": "Azure"
        },
        "model_kwargs": {
            "base_url_envar": "LLAMA_32_90_END_POINT",
            "api_key_envar": "LLAMA_32_90_API_KEY",
            "stop": None,
            "deployment_name": "llama-3-2-90B-Vision-Instruct",
            "end_point_version": '/v1/chat/completions',
            "max_tokens": 3000
        }
    },

    "cohere": {
        "type": "azure",
        "retriever_token_limit_calculator": get_retrieved_docs_token_limit("cohere"),
        "chat_history_limit_calculator": get_chat_history_token_limit("cohere"),
        "model_profile_for_generation": {
            "knowledge_cutoff_date":'August 2024',
            "model_preffered_name": "Cohere Command R+ 08-2024",
            "model_developed_by": "Cohere",
            "model_provided_through": "Azure"
        },
        "model_kwargs": {
            "base_url_envar": "COHERE_END_POINT",
            "api_key_envar": "COHERE_API_KEY",
            "stop": None,
            "deployment_name": "Cohere-command-r",
            "end_point_version": '/v1/chat/completions',
            "max_tokens": 3000
        }
    },
    "deepseek-v3": {
        "type": "azure",
        "retriever_token_limit_calculator": get_retrieved_docs_token_limit("deepseek-v3", 20000),
        "chat_history_limit_calculator": get_chat_history_token_limit("deepseek-v3", 20000),
        "model_profile_for_generation": {
            "knowledge_cutoff_date": 'July 2023',
            "model_preffered_name": "DeepSeek-V3-0324",
            "model_developed_by": "deepSeek",
            "model_provided_through": "Azure"
        },
        "model_kwargs": {
            "base_url_envar": "DEEPSEEK_V3_END_POINT",
            "api_key_envar": "DEEPSEEK_V3_API_KEY",
            "stop": None,
            "deployment_name": "DeepSeek-v3",
            "end_point_version": '/v1/chat/completions',
            "max_tokens": 3000,
        },
    "stream": True,

    },

}

backend_azure_map = {
    # "previewo1": "o1-preview",
    "o1": "o1",
    # "minio1": "o1-mini",
    "minio4": "o4-mini",
    'gpt-4o': 'gpt-4o',
    # 'gpt-4-turbo': 'gpt-4-turbo',
    # 'gpt-4-32k': 'gpt-4-32k',
    'gpt-4-1': 'gpt-4-1',
    'text-embedding-ada-002': 'text-embedding-ada-002',
    'text-embedding-3-large': 'text-embedding-3-large',
    # 'gpt-3.5-turbo': 'gpt-3.5-turbo',
    # 'gpt-3.5-turbo-16k': "gpt-3.5-turbo-16k",
    'Llama-3-70B-Instruct': 'llama-3-70b-instruct',
    'Llama-3-1-70B-Instruct': 'llama-3-1-70b-instruct',
    "Llama-3-1-8b-Instruct":  "llama-3-1-8b-instruct",
    "Llama-3-2-90B": "llama-3-2-90b-instruct",
    "Llama-3-1-405B": "llama-3-1-408b-instruct",
    'Mistral-large-02': 'mistral-large-2411',
    'cohere': 'cohere',  # azure send model name to empty string only happens to cohere-command-r
    'deepseek-v3': 'deepseek-v3',
    'gemini-2.0-flash': 'gemini-2.0-flash'  # azure send model name to empty string only happens to cohere-command-r
}

inv_catalog = {v: k for k, v in backend_azure_map.items()}

fe_display_messages = {
    # "rephrase_chain": "Understanding input",
    "retriever_chain": "Searching data library for relevant content",
    "web_search": "Searching Web for relevant content",
    "rephrase" : "{}",
    "documents": "{},{}",
    "web_links": "{}"
    # "response_generation": "Generating Response",
}


def map_to_local(model_name:str):
    return backend_azure_map[model_name]


def map_to_backend(model_name):
    return inv_catalog[model_name]


true = True


def k_chunk(name):
    return chat_model_config[name]["k_chunk"]

web_supported_languages = ["fr", "en", "de", "nl", "da", "fi", "hu", "it", "no", "pt", "ro", "ru", "es", "sv", "tr"]
supported_templates = {
    'en': "English",
    'de': "German",
    'it': "Italian",
}

supported_images = {
    'png': 'PNG',
    'jpeg': 'JPEG',
    'jpg': 'JPG',
    'gif': 'GIF'
}

supported_lang_abbreviation = {v: k for k, v in supported_templates.items()}


def language_abbreviation(language):
    return supported_lang_abbreviation[language]

