"""
  contains utility function used all across the template module
"""
import importlib
import logging
from typing import List, Dict, Any, Union

from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate

from ..config import chat_model_config, default_chat_roles

log = logging.getLogger('doclogger')
log.disabled = False

# pylint: disable=invalid-name,E1120, C0301, C0305
DEFAULT_TEMPLATES = {
    "SUMMARIZATION": {
        # 'description_prompt': 'DOCUMENT_DESCRIPTION_PROMPT',
        "summary_template": ("SUMMARY_PROMPT_SYSTEM", "SUMMARY_PROMPT_HUMAN", False),
        "reflection_template": ("SUMMARY_REFLECTION_PROMPT_SYSTEM", "SUMMARY_PROMPT_HUMAN", False)
    },

    "SUBJECT":{
        "extractor_prompt": (
            {"base_template":"chat_qa_system_template",
             "base_values":{"instructions": "chat_qa_default_instructions"},
            },
            "extractor_template",
            True
        ),
    },

    "REPHRASE": {
        "initial_question_template": ("initial_rephrase_system_template", "initial_rephrase_human_template", False),
        # "chat_condenser_prompt": "CONDENSE_QUESTION_PROMPT",
        "chat_condenser_prompt": ("condenser_template_system", "condenser_template_human", False),
    },
    "SEARCH": {
        "search_prompt": ("search_prompt_system", "search_prompt_human", True),
              },
    "IMAGE": {
        "response_template": (
            "initial_image_system_template",
            "user_image_template_function",
            True)
    },

    "WEBCHAT": {
        "web_search": (
            {"base_template":"web_qa_system_prompt",
             "base_values":{"instructions": "web_default_instructions"},
            },
            "extractor_template",
            True
        ),
    },

    "ROUTER": {
        'router_prompt': 'ROUTER_PROMPT'
    },

    "TITLE": {
        'title_prompt': 'TITLE_PROMPT'
    },

    "DOC_ANALYZER": {
        'reflection': ("REFLECTION_PROMPT_SYSTEM", "REFLECTION_PROMPT_HUMAN", False)
    },

    "PARAMETRIC": {
        "response_template":(
            {"base_template":"direct_response_template_system",
             "base_values":{"instructions": "direct_response_default_instructions"},
            },
            "direct_response_template_human", True)
    },
    "ASSISTANT":{
        "enterprise_knowledge": (
            {"base_template":"assistant_system_template",
             "base_values":{"instructions": "assistant_system_template_subject"},
            },
            "extractor_template",
            True),
        "web_search": (
            {"base_template":"assistant_system_template",
             "base_values":{"instructions": "web_default_instructions"},
            },
            "extractor_template",
            True),
        "default": (
            {"base_template":"assistant_system_template",
             "base_values":{"instructions": "direct_response_default_instructions"},
            },
            "direct_response_template_human",
            True),
    },
    "TRANSLATION":{
        "response_template":("translation_system", "translation_human", False)
    },
    "MULTITOOL": {
        'router': ('multitool_router_system_template', "multitool_router_human_template", True)

    }
}


PLANNER_TEMPLATES = {
    "SUBJECT":{
        "extractor_prompt": (
            {"base_template":"chat_qa_system_template",
             "base_values":{"instructions": "chat_qa_default_instructions"},
            },
            "extractor_template",
            True
        ),
    },
    "IMAGE": {
        "response_template": (
            "initial_image_system_template",
            "user_image_template_function",
            True)
    },

    "WEBCHAT": {
        "web_search": (
            {"base_template":"web_qa_system_prompt",
             "base_values":{"instructions": "web_default_instructions"},
            },
            "extractor_template",
            True
        ),
    },

    "DOC_ANALYZER": {
        'reflection': ("REFLECTION_PROMPT_SYSTEM", "REFLECTION_PROMPT_HUMAN", False)
    },

    "PARAMETRIC": {
        "response_template":(
            {"base_template":"direct_response_template_system",
             "base_values":{"instructions": "direct_response_default_instructions"},
            },
            "direct_response_template_human", True)
    },
    "ASSISTANT":{
        "enterprise_knowledge": (
            {"base_template":"assistant_system_template",
             "base_values":{"instructions": "assistant_system_template_subject"},
            },
            "extractor_template",
            True),
        "web_search": (
            {"base_template":"assistant_system_template",
             "base_values":{"instructions": "web_default_instructions"},
            },
            "extractor_template",
            True),
        "default": (
            {"base_template":"assistant_system_template",
             "base_values":{"instructions": "direct_response_default_instructions"},
            },
            "direct_response_template_human",
            True),
    },
    "TITLE": {
        'title_prompt': 'TITLE_PROMPT'
    },
    "JOINER": {
        "joinner_prompt": ("joinner_response_template_system", "joinner_response_template_human", True)
    },
    "PLANNER":{
        "initial_question_planner_template":("planner_system_prompt","planner_human_prompt", False),
        "condenser_planner_template":("planner_with_chat_history_system_prompt", "planner_with_chat_history_human_prompt", True),
        "tool_selector_planner_template":("planner_tool_selector_system_prompt","planner_tool_selector_user_prompt",False)
    }
}

module_template_map={'default': DEFAULT_TEMPLATES,
                     'planner': PLANNER_TEMPLATES}

def build_chat_templated(chat_qa_system_template, chat_qa_human_template,
                         model_name, history_place_holder=False) -> ChatPromptTemplate:

    chat_roles = chat_model_config[model_name].get("chat_roles", default_chat_roles)
    messages = [(chat_roles['instructions'] , chat_qa_system_template)]
    if history_place_holder:
        messages.append(MessagesPlaceholder("chat_history"))
    messages.append( (chat_roles['user'], chat_qa_human_template))

    return ChatPromptTemplate.from_messages(messages)


def build_router_prompt_template(conversational_desc, subject_desc, multi_prompt_router_template):
    prompt_info = [
        {
            "name": "greetings",
            "description": conversational_desc,
        },
        {
            "name": "enterprise_specific",
            "description": subject_desc,
        },
    ]
    destinations_str = "\n".join([f"{p['name']}: {p['description']}" for p in prompt_info])
    router_template = multi_prompt_router_template.format(destinations=destinations_str)

    return router_template


def get_module(lang):
    # if lang not in supported_templates:
    #     raise NotSupportedException(lang, code="unsupported_language")
    try:
        return importlib.import_module(f'.{lang}', __name__.rsplit('.', 1)[0])
    except ImportError as i_e:
        log.error(i_e, exc_info=True)
        raise ImportError(i_e) from i_e


def attachment_router_prompt():
    messages = [
        ('system', """Given the user question, summaries for user attached documents and categories, classify the user question into suitable category,
- `greetings`: if user question is related to greetings phrases (like hello, hi etc.) or farewell phrases (like thank you, goodbye, etc.)
- `attachments` : User has attached some documents, if user question seems directly or indirectly related these documents (summarized below), then select this category. In case user question contains any demonstrative pronouns \ vague actions assume it is related to these attachments. Summaries:
        - {context} 
- `other`: if the user question is clearly and unambiguously not related to `attachments`
Optionally, you may be provided with some chat history prior to the current question, between the User (HumanMessage) and AI (AIMessage). In case there is some vagueness/ambiguity in user question, categorize as `attachments` category.
Respond with just a single word output, i.e. `category name`.
"""),
        MessagesPlaceholder("chat_history"),
        ('human',"question: {question}")
    ]

    return  ChatPromptTemplate.from_messages(messages)


def get_router_template(route_type='S'):
    """

    :param route_type: can be `S` or `W` S for subject points to RAG, W for web search
    :return:
    """
    assert route_type in ['S', 'W']
    package = get_module('en')
    conversational_desc = getattr(package, 'CONVERSATIONAL_DESC')
    subject = getattr(package, 'SUBJECT_DESC')
    if route_type == 'W':
        subject = getattr(package, 'WEB_DESC')

    router_template = getattr(package, 'MULTI_PROMPT_ROUTER_TEMPLATE')
    router_prompt = build_router_prompt_template(conversational_desc, subject, router_template)
    messages = [
        ('system', router_prompt),
        MessagesPlaceholder("chat_history"),
        ('human',"text input: {question}")
        ]
    return ChatPromptTemplate.from_messages(messages)

def build_image_template(images):
    image_template = []
    for k, v in images.items():
        img_data = v['file_data']
        image_type = v['original_format']
        # if image_type and image_type not in supported_images:
        #     return False
        image_template.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/{image_type};base64, {img_data}"},
        })
    image_template.append({
                            "type": "text",
                            "text": "{question}"
                        }
    )
    return image_template


def get_templates(template_names: List[str], module: str, model_name, user_images=None
                  ) -> Dict[str, ChatPromptTemplate]:

    _module = get_module(module)
    chat_templates = {}
    module_templates = module_template_map[module]

    def load_value_from_module(module, template: Union[Dict[Any, Any], str]):
        if isinstance(template, str):
            if "user_image_template_function" == template:
                return build_image_template(user_images)

            return getattr(module, template)

        elif isinstance(template, dict):
            base_string: str = getattr(module, template["base_template"])
            return base_string.format(**{k_y: getattr(module, val) for k_y, val in template["base_values"].items()})
        raise ValueError(f"{template} is not supported")

    for template_name in template_names:
        templates = module_templates[template_name]
        for key, val in templates.items():
            if isinstance(val, tuple):

                history_place_holder = False
                try:
                    history_place_holder = val[2]
                except IndexError:
                    log.warning("%s does not have 3rd index assuming to not include chat history", val)

                system_prompt_string = load_value_from_module(_module, val[0])
                human_prompt_string = load_value_from_module(_module, val[1])
                chat_templates[key] = build_chat_templated(system_prompt_string, human_prompt_string,
                                                           model_name, history_place_holder)

            elif isinstance(val, str):
                chat_templates[key] = getattr(_module, val)
            else:
                raise ValueError

    return chat_templates
