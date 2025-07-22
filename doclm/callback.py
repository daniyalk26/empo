
import logging
import math
import json

import os
import time
from uuid import UUID
from typing import Any, Dict, Optional, List, Sequence

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from azure.messaging.webpubsubservice import WebPubSubServiceClient
from azure.messaging.webpubsubservice.aio import (
    WebPubSubServiceClient as WebPubSubServiceClientAsync,
)
from langchain_core.callbacks.base import BaseCallbackHandler, AsyncCallbackHandler
from langchain_core.callbacks import FileCallbackHandler
from langchain_core.utils import print_text
from langchain_community.callbacks.openai_info import (
    OpenAICallbackHandler,
    # get_openai_token_cost_for_model,
    # MODEL_COST_PER_1K_TOKENS,
)
from langchain_core.outputs import LLMResult
from .models.models import OutFETaskStepMsg, OutFEStreamMsg
from .tokenizer_cal import num_tokens_from_string, num_tokens_from_messages
from .config import map_to_backend, fe_display_messages

log = logging.getLogger("doclogger")
log.disabled = False


# pylint: disable=E1101,R0902,W0223,C0116
def standardize_model_name_custom(
        model_name: str,
) -> str:
    if "35" in model_name:
        return model_name.replace("35", "3.5")
    try:
        model_name = map_to_backend(model_name)
    except Exception as e:
        log.error(e)
        raise e
    return model_name


# def get_token_calculation(response):
#     token_usage = response["usage"]
#     model_name = standardize_model_name_custom(response.get("model", ""))
#
#     return {
#         "completion_tokens": token_usage.get("completion_tokens", 0),
#         "prompt_tokens": token_usage.get("prompt_tokens", 0),
#         "model_name": model_name,
#     }


# pylint: disable=R0901,C0103
# TODO: There are issues in the way this pubsub class is implemented and used.
class AzureWebPubSubCallback(BaseCallbackHandler):
    def __init__(self, chat_type, encrypt_func, encryption_key=None, **kwargs):
        self.connection_id = kwargs.get("wps_conn_id", 1)
        self.chat_id = kwargs.get("chat_id", 1)
        self.qid = kwargs.get("qid", 2)
        self.userid = kwargs.get("userid", 2)
        self.rid = kwargs.get("rid", 2)
        self.encrypt_func = encrypt_func
        self.encryption_key = encryption_key
        self.service = self.get_azurewebpubsub_service()
        self.pipeline_stage = ""
        self.chat_type = chat_type.upper()

    # @staticmethod
    def get_azurewebpubsub_service(self):
        access_key = os.getenv("AZ_WPS_ACCESS_KEY")
        endpoint = os.getenv("AZ_WPS_ENDPOINT")
        hub_name = os.getenv("AZ_WPS_HUB")
        azure_pubsub_connection_str = (
            f"Endpoint={endpoint};AccessKey={access_key};Version=1.0;"
        )
        return WebPubSubServiceClient.from_connection_string(
            azure_pubsub_connection_str, hub=hub_name
        )

    def on_llm_start(
            self,
            serialized: Dict[str, Any],
            prompts: List[str],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ) -> Any:
        try:
            self.pipeline_stage = serialized['kwargs']['deployment_name']
        except Exception:
            self.pipeline_stage = serialized['id'][-1]

    def on_chain_start(
            self,
            serialized: Dict[str, Any],
            inputs: Dict[str, Any],
            end: str = "\n",
            **kwargs: Any,
    ) -> None:
        self.pipeline_stage = kwargs.get('name', '')
        # if 'o1' in inputs.get('model_preffered_name', ''):
        #     self.send_msg(
        #         {"response": 'Thinking', "rid": self.rid,
        #          "chat_id": self.chat_id, 'step': 3, "type": self.chat_type},
        #         event="chat_step",
        #     )
        if self.pipeline_stage in ["FormatRetrieverWebChain", "FormatRetrieverDocumentChain"]:
            self.send_msg(
            {"response": inputs['query'], "rid": self.rid,
             "chat_id": self.chat_id, 'step': 1, "type": self.chat_type},
            event="chat_step",
        )


    def on_llm_new_token(self, token, **kwargs) -> None:
        self.send_msg(
            {"response": token, "rid": self.rid, "chat_id": self.chat_id},
            event="stream",
        )

    def on_llm_error(self, error, **kwargs):
        log.error(error)

    def on_chain_error(
            self, error, **kwargs: Any
    ) -> Any:
        log.error(error, exc_info=True)

    def send_msg(self, msg: dict, event=None):
        try:
            msg["event"] = event
            encrypted_message = self.encrypt_func(json.dumps(msg), self.encryption_key)
            self.service.send_to_connection(
                connection_id=self.connection_id, message=encrypted_message
            )
        except Exception as e:
            log.debug(str(msg))
            log.error(e, exc_info=True)

    def on_chain_end(
            self,
            outputs: Dict[str, Any],
            # *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
    ) -> None:
        """Print out that we finished a chain."""
        # if 'o1' in self.pipeline_stage:
        #     self.send_msg(
        #         {"response": outputs.content, "rid": self.rid, "chat_id": self.chat_id},
        #         event=kwargs.get('event', 'stream'),
        #     )
        if self.pipeline_stage not in ["FormatRetrieverWebChain", "FormatRetrieverDocumentChain"]:
            return
        log.info(f"Format retriever AzureWebPubSub callback got outputs dict {outputs}")
        if sources := outputs.get("source"):
            docx = len(set([x['name'] for x in sources]))
            pages = len([x['page'] for x in sources])
        else:
            return

        self.send_msg(
            {"response":  [docx, pages], "rid": self.rid,
             "chat_id": self.chat_id, 'step': 2, "type": self.chat_type},
            event="chat_step",
        )


    def close(self):
        try:
            self.service.close()
        except Exception as e:
            log.error("Error while closing stream service object %s", str(e),
                      exc_info=True)



class AzureWebPubSubCallbackMultiTool(AzureWebPubSubCallback):

    def __init__(self, chat_type, encrypt_func, **kwargs):
        super().__init__(chat_type, encrypt_func, **kwargs)
        self.task_map = {}

    def send_msg(self, msg: dict, event=None):

        try:
            if event:
                msg["event"] = event
            msg['rid'] = self.rid
            msg['chat_id'] = self.chat_id

            encrypted_message = self.encrypt_func(json.dumps(msg), self.encryption_key)
            self.service.send_to_connection(
                connection_id=self.connection_id, message=encrypted_message
            )
        except Exception as e:
            log.debug(str(msg))
            log.error(e, exc_info=True)

    def on_chain_start(
            self,
            serialized: Dict[str, Any],
            inputs: Dict[str, Any],
            end: str = "\n",
            **kwargs: Any,
    ) -> None:
        self.pipeline_stage = kwargs.get('name', '')

        print('---')

    def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        if isinstance(data, BaseModel):
            self.send_msg(msg=data.model_dump(), event=name)


    def on_tool_start(self,
                      serialized: dict[str, Any],
                      input_str: str,
                      *,
                      run_id: UUID,
                      parent_run_id: Optional[UUID] = None,
                      tags: Optional[list[str]] = None,
                      metadata: Optional[dict[str, Any]] = None,
                      inputs: Optional[dict[str, Any]] = None,
                      **kwargs: Any,
                      ) -> Any:

        tool_name = serialized.get('name','')
        default_task_id = '0' if metadata.get('stage_name') == 'main_chain' else ''
        task_id = inputs.get('task_id', default_task_id)

        if task_id:
            msg = OutFETaskStepMsg(task_id=task_id, tool_name=tool_name,
                                   response=inputs['query'], step_name='tool_input')
            self.send_msg(msg=msg.model_dump(), event='chat_step')


    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        inputs: dict,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:

        tool_name = metadata.get('tool_name', '')
        task_id = metadata.get('task_id', '')
        self.task_map[run_id] = {'tool_name': tool_name,
                                 'task_id': task_id}

        if tool_name:
            msg = OutFETaskStepMsg(task_id=task_id, tool_name=tool_name,
                                   response=inputs['query'], step_name='search_phrase',)
            self.send_msg(msg=msg.model_dump(), event='chat_step')


    def on_llm_start(
            self,
            serialized: Dict[str, Any],
            prompts: List[str],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ) -> Any:
        tool_name = metadata.get('tool_name', '')
        task_id = metadata.get('task_id', '')
        buffered_tokens: List[str] = []
        # buffer_size=
        if tool_name:
            self.task_map[run_id] = {'tool_name': tool_name,
                                     'task_id': task_id,
                                     "tokens_emmited":0,
                                     "buffered_tokens":buffered_tokens,
                                     "buffer_current_limit":0}

    def on_llm_new_token(self, token, **kwargs) -> None:
        try:
            run_id = kwargs['run_id']
            task_info = self.task_map[run_id]
            task_info['buffered_tokens'].append(token)
            
            if len(task_info['buffered_tokens'])>task_info['buffer_current_limit']:
                msg = OutFETaskStepMsg(response="".join(task_info['buffered_tokens']), task_id=task_info['task_id'],
                                    tool_name=task_info['tool_name'], step_name='answer_generation')
                task_info['tokens_emmited']+=len(task_info['buffered_tokens'])
                task_info['buffer_current_limit']=self.get_buffer_size_for_new_llm_token(task_info['tokens_emmited'])
                self.send_msg(msg=msg.model_dump(), event="stream")
                task_info['buffered_tokens']=[]
        except Exception as e:

            log.debug(str(token))
            log.error(e, exc_info=True)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM ends running.

        Args:
            response (LLMResult): The response which was generated.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            kwargs (Any): Additional keyword arguments.
        """
        try:
            if run_id in self.task_map:
                task_info = self.task_map[run_id]
                final_response = "".join(task_info['buffered_tokens'])

                if response.llm_output:
                    if 'o1' in response.llm_output['model_name']:
                        final_response = response.generations[-1][-1].text

                msg = OutFETaskStepMsg(response=final_response, task_id=task_info['task_id'],
                                            tool_name=task_info['tool_name'], step_name='answer_generation')
                self.send_msg(msg=msg.model_dump(), event="stream")


        except Exception as e:
            log.debug(str(response))
            log.error(e, exc_info=True)

    @staticmethod
    def get_buffer_size_for_new_llm_token(tokens_emmited: int):
        return min(15, int(1 + 5*math.log(tokens_emmited)))

    def on_chain_start(
            self,
            serialized: Dict[str, Any],
            inputs: Dict[str, Any],
            end: str = "\n",
            **kwargs: Any,
    ) -> None:

        pass
        print(kwargs)

    def on_chain_end(
            self,
            outputs: Dict[str, Any],
            # *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
    ) -> None:
        """Print out that we finished a chain."""
        if 'o1' in self.pipeline_stage:
            run_id = kwargs['run_id']
            task_info = self.task_map[run_id]
            msg = OutFETaskStepMsg(response=outputs.content, task_id=task_info['task_id'],
                                   tool_name=task_info['tool_name'], step_name='answer_generation')
            self.send_msg(msg=msg.model_dump(), event="stream")


    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        task_info = self.task_map[run_id]
        msg = OutFETaskStepMsg(task_id=task_info['task_id'], tool_name=task_info['tool_name'],
                               response='', step_name='document_received', )
        self.send_msg(msg=msg.model_dump(), event='chat_step')

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        # print("hi")
        pass

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:

        log.error(kwargs)
        log.error(error, exc_info=True)
        raise error


class AsyncAzureWebPubSubCallback(AsyncCallbackHandler):
    def __init__(self, encrypt_func, encryption_key=None, **kwargs):
        self.connection_id = kwargs.get("connection_id", 1)
        self.chat_id = kwargs.get("chat_id", 1)
        self.qid = kwargs.get("qid", 2)
        self.userid = kwargs.get("userid", 2)
        self.rid = kwargs.get("rid", 2)
        self.encrypt_func = encrypt_func
        self.encryption_key = encryption_key
        self.service = self.get_azurewebpubsub_service()

    @staticmethod
    def get_azurewebpubsub_service():
        access_key = os.getenv("AZ_WPS_ACCESS_KEY")
        endpoint = os.getenv("AZ_WPS_ENDPOINT")
        hub_name = os.getenv("AZ_WPS_HUB")
        azure_pubsub_connection_str = (
            f"Endpoint={endpoint};AccessKey={access_key};Version=1.0;"
        )
        return WebPubSubServiceClientAsync.from_connection_string(
            azure_pubsub_connection_str, hub=hub_name
        )

    async def on_llm_new_token(self, token, **kwargs) -> None:
        await self.send_msg(
            {"response": token, "rid": self.rid, "chat_id": self.chat_id},
            event="stream",
        )

    async def send_msg(self, msg: dict, event=None):
        msg["event"] = event
        log.debug(str(msg))
        encrypted_message = self.encrypt_func(json.dumps(msg), self.encryption_key)
        await self.service.send_to_connection(
            connection_id=self.connection_id, message=encrypted_message
        )

    def close(self):
        self.service.close()


class MapEncoder(json.JSONEncoder):
    def default(self, z):
        try:
            return super().default(z)
        except Exception:
            return str(z)


class LoggingCallback(FileCallbackHandler):
    @staticmethod
    def parser(key_val, stage):
        key_val['stage'] = stage
        return key_val

    def on_chain_start(
            self,
            serialized: Dict[str, Any],
            inputs: Dict[str, Any],
            end: str = "\n",
            **kwargs: Any,
    ) -> None:
        data_dict = self.parser(
            kwargs,
            "start",
        )
        if serialized:
            data_dict["meta"] = {
                "class name": serialized["id"][-1] if serialized else kwargs['name'],
                "kwargs": serialized.get("kwargs"),
            }
        print_text(self.parse_dict(data_dict), color="blue", end=end, file=self.file)

    def on_text(
            self,
            text: str,
            color: Optional[str] = None,
            end: str = "\n",
            **kwargs: Any,
    ) -> None:
        data_dict = self.parser(kwargs, "on_text")
        data_dict["meta"] = {"text": text}
        print_text(self.parse_dict(data_dict), color="yellow", end=end, file=self.file)

    @staticmethod
    def parse_dict(dic):
        return json.dumps(dic, cls=MapEncoder)

    def on_chain_end(
            self,
            outputs: Dict[str, Any],
            end: str = "\n",
            **kwargs: Any,
    ) -> None:
        """Print out that we finished a chain."""
        data_dict = self.parser(kwargs, "end")
        data_dict["meta"] = outputs
        print_text(self.parse_dict(data_dict), color="pink", end=end, file=self.file)


class TokenCalculationCallback(OpenAICallbackHandler):
    def __init__(self):
        super().__init__()
        self.token_usage_history = []
        self.model_tokens = {}
        self.estimated_tokens = {}
        # self.encoding = tiktoken.get_encoding('cl100k_base')

    @property
    def tokens_usage(self) -> Dict:
        return self.model_tokens

    def on_embedding(self, model: str, tokens: List[List[float]]):

        model_name = map_to_backend(model)
        if model_name not in self.model_tokens:
            self.model_tokens[model_name] = {"prompt_tokens": 0,
                                             "completion_tokens": 0,
                                             "model_name": model_name}
        self.model_tokens[model_name]['prompt_tokens'] += sum(len(i) for i in tokens)

    def on_retriever_start(
            self,
            serialized: Dict[str, Any],
            query: str,
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ) -> Any:

        retriever = metadata.get('retriever')
        retriever_type = kwargs['name']

        if retriever_type in ["DocumentPGKeywordRetriever", "WebResearchRetriever"]:

            return
        elif retriever_type in ["DocumentPGHybridRetriever"]:
            model_name = retriever.store.vector_store.embeddings.model
        else:
            model_name = retriever.store.embeddings.model

        if not model_name:
            raise ValueError('could not find the model')

        if model_name not in self.model_tokens:
            self.model_tokens[model_name] = {"prompt_tokens": 0,
                                             "completion_tokens": 0,
                                             "model_name": model_name}
        # embedding is generated for the query for semantic search
        if isinstance(query, dict):
            self.model_tokens[model_name]['prompt_tokens'] += num_tokens_from_string(query['query'], model_name)
        else:
            self.model_tokens[model_name]['prompt_tokens'] += num_tokens_from_string(query, model_name)

    def on_chat_model_start(
            self,
            serialized: Dict[str, Any],
            messages,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[List[str]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ) -> Any:

        """Run when Chat Model starts running."""
        # Do this only for streaming models
        invocation_params = kwargs.get('invocation_params')
        model = metadata.get('model', None) or invocation_params.get('model', None)
        model_name = map_to_backend(model)
        if model_name not in self.model_tokens:
            self.model_tokens[model_name] = {"prompt_tokens": 0,
                                             "completion_tokens": 0,
                                             "model_name": model_name}

        self.estimated_tokens[run_id] = {"prompt_tokens": num_tokens_from_messages(messages[0], model),
                                         "completion_tokens": 0,
                                         "model_name": model_name}

    def on_llm_new_token(self, token, **kwargs) -> None:
        try:
            run_id = kwargs['run_id']
            self.estimated_tokens[run_id]["completion_tokens"]+=1
        except Exception as e:
            log.error(e, exc_info=True)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Collect token usage."""

        run_id = kwargs.get('run_id')
        token_usage = None
        model_name = self.estimated_tokens[run_id]["model_name"]  # generation_info["model_name"]

        if isinstance(response.llm_output, dict):  # doing this for openai models
            generation_info = response.llm_output
            # log.info(generation_info)
            token_usage = generation_info["token_usage"]

        # For Azure-end-point models doing this
        elif response.generations[0][0].generation_info.get('finish_reason', '') == 'stop':
            generation_info = response.generations[0][0].generation_info
            # log.info(generation_info)
            token_usage = response.generations[0][0].generation_info.get('usage')
            # token_usage = response.generations[0][0].message.usage_metadata  # needed for gpt35/4 due to masking of langchain
        else:
            generation_info = response.generations[0][0].generation_info

        log.info('generation info %s', generation_info)
        self.successful_requests += 1


        # This is for models that give the tokens usage in response
        if token_usage:
            log.debug("%s", response)
            # token_usage = response.llm_output["token_usage"]
            completion_tokens = token_usage.get("completion_tokens") or token_usage.get("output_tokens")
            prompt_tokens = token_usage.get("prompt_tokens") or token_usage.get("input_tokens")

            # model_name = map_to_backend(model)

            self.model_tokens[model_name]['prompt_tokens'] += prompt_tokens
            self.model_tokens[model_name]['completion_tokens'] += completion_tokens
            return
        # if there is for models no tokens usage in response

        else:
            if run_id in self.estimated_tokens:
                self.model_tokens[model_name]['prompt_tokens'] +=  self.estimated_tokens[run_id]["prompt_tokens"]
                self.model_tokens[model_name]['completion_tokens'] +=  self.estimated_tokens[run_id]["completion_tokens"]

        return


class ResponseTimeCalculation(BaseCallbackHandler):

    def __init__(self):
        self.time_stamp_log = []

    @property
    def time_log(self) -> List:
        return self.time_stamp_log

    def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        print(f"on_llm_start {serialized}")
        print(f"on_llm_start {kwargs}")
        if kwargs.get('metadata'):
            self.time_stamp_log.append({
                'name': kwargs['metadata']['stage_name'],
                'start_time': time.time()
            })
        # if not self.chat_start:
        # self.chat_start = time.time()

    # def on_chain_start(
    #         self,
    #         serialized: Dict[str, Any],
    #         inputs: Dict[str, Any],
    #         end: str = "\n",
    #         **kwargs: Any,
    # ) -> None:
    #     print(f"on_chain_start {serialized}")
    #     print(f"on_chain_start {kwargs}")
    #     self.time_stamp_log[str(kwargs['run_id'])] = {
    #         'name': serialized['id'],
    #         'start': time.time()
    #     }
    # self.time_stamp_log[serialized['id']] = time.time()
    # self.time_stamp_log[serialized['id']] = time.time()
    #


log_dir = os.getenv("LOGDIR", None)

if log_dir:
    os.makedirs(log_dir, exist_ok=True)


def get_logger(chat_id, name_string: str = "chat_logs_id_"):
    logfile = os.path.join(log_dir, f"{name_string}{chat_id}.log")

    return LoggingCallback(logfile)
