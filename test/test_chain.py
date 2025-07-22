import json
import logging
import unittest
from unittest.mock import Mock, patch

from dotenv import load_dotenv
from langchain.schema.embeddings import Embeddings

if True:
    load_dotenv()

from doclm.util import Filemeta
from doclm.util import gtp_bot
from doclm.vector_store.custom_pgvector import DocumentPGVector

from langchain.vectorstores.base import VectorStoreRetriever, VectorStore
from langchain.docstore.document import Document
from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from openai.openai_object import OpenAIObject

embedding_length = 1536
mock_subject = 'mock title'
stream_message = ['mes', 'sage', ' ', 'rece', 'ived']


def mock_stream(model, api_key, api_version, organization, engine, message):
    message = [' '] + message
    for i, m in enumerate(message):
        if i == 0:
            delta = {
                "role": "assistant",
                # "content": "m"
            }
        else:
            delta = {
                # "role": "assistant",
                "content": m
            }
        data = {
            "id": "chatcmpl-8KLXC5zHJQ3rNAJTCRKdnINyPGIVD",
            "created": 1699860030,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "delta": delta,
                    "content_filter_results": {}
                }
            ],
            "usage": None
        }
        resp = OpenAIObject.construct_from(
            data,
            api_key=api_key,
            api_version=api_version,
            organization=organization,
            response_ms=None,
            engine=engine,
        )
        yield resp


def dummy_response_function(api_key=None,
                            api_base=None,
                            api_type=None,
                            request_id=None,
                            api_version=None,
                            organization=None,
                            messages=None,
                            stream=None,
                            **params):
    deployment_id = params.pop("deployment_id", None)
    engine = params.pop("engine", deployment_id)
    model = params.get("model", None)
    timeout = params.pop("timeout", None)
    headers = params.pop("headers", None)
    request_timeout = params.pop("request_timeout", None)

    if stream:
        return mock_stream(model, api_key, api_version, organization, engine, stream_message)

    content = None
    for i in messages:
        if 'subject' in i['content']:
            content = mock_subject
        elif 'field_specific' in i['content']:
            content = "{\n    \"destination\": \"field_specific\"\n}"
        elif 'how are you' in i['content']:
            content = (
                "I am an AI language model, so I don't have feelings. How can I assist you "
                "today?")
        elif 'dummy answer' in i['content']:
            content = 'how are you doing'
        else:
            content = 'sorry'

    resp = {'id': 'chatcmpl-8IumvWM1q0AnUW1HoEX1DbqlglYIn',
            'model': model,
            'prompt_filter_results': [{'prompt_index': 0, 'content_filter_results':
                {'hate': {'filtered': False, 'severity': 'safe'}, 'self_harm':
                    {'filtered': False, 'severity': 'safe'},
                 'sexual': {'filtered': False, 'severity': 'safe'},
                 'violence': {'filtered': False, 'severity': 'safe'}}}],
            'choices': [{'index': 0, 'finish_reason': 'stop', 'message':
                {'role': 'assistant',
                 'content': content},
                         'content_filter_results': {
                             'hate': {'filtered': False, 'severity': 'safe'},
                             'self_harm': {'filtered': False,
                                           'severity': 'safe'},
                             'sexual': {'filtered': False,
                                        'severity': 'safe'},
                             'violence': {'filtered': False,
                                          'severity': 'safe'}}}],
            'usage': {'prompt_tokens': 5, 'completion_tokens': 5, 'total_tokens': 10}}

    return OpenAIObject.construct_from(
        resp,
        api_key=api_key,
        api_version=api_version,
        organization=organization,
        response_ms=None,
        engine=engine,
    )


class embeddings(Embeddings):
    model = 'cl100k_base'

    def embed_documents(self, texts):
        return [[0.1] * embedding_length for _ in texts]

    def embed_query(self, text: str):
        return None


class TestChain(unittest.TestCase):

    # @patch.object(SomeClass, 'attribute', sentinel.attribute)
    # @patch(SomeClass, 'attribute', sentinel.attribute)
    # @patch('sqlalchemy.create_engine')
    # @patch('langchain.vectorstores.pgvector.PGVector.add_embeddings')
    # from langchain.vectorstores.base import VectorStoreRetriever, VectorStore
    # @patch('openai.api_resources.abstract.engine_api_resource.EngineAPIResource.create',
    #        side_effect=dummy_response_function)
    @patch('langchain.chat_models.openai.ChatOpenAI.completion_with_retry',
           side_effect=dummy_response_function)
    @patch('sqlalchemy.create_engine')
    @patch('langchain.vectorstores.base.VectorStoreRetriever._get_relevant_documents')
    def test_something(self, doc_retriver, sql_conn, mock_openai_responce):
        # mock_openai_responce = MagicMock()
        embed = embeddings()
        db = DocumentPGVector(connection_string='sss', embedding_function=embed)
        retriver = db.as_retriever()
        mock_ret_doc = [Document(page_content='some document ',
                                 metadata={
                                     'id': file,
                                     'source': f'../file{file}.pdf',
                                     'name': f'original_name_{file}',
                                     'page': str(chunk),
                                     'format': 'pdf',
                                     'chunk_num': chunk,
                                     'document_summary': f'summary_{file}',
                                 }
                                 )
                        for file, chunk in zip([1, 1, 1, 2, 2, 2], [1, 3, 4, 1, 2, 5])
                        ]
        doc_retriver.return_value = mock_ret_doc
        chain = gtp_bot(retriver)
        result = chain.run({
            "question": 'how are you',
            "subject": 'a specific',
            "chat_history": '',
            "": None,
        })
        print(result)
        result = json.loads(result)

        self.assertEqual(result['chat_subject'], mock_subject)

        self.assertEqual(result['response'], ''.join(stream_message))
        self.assertGreater(len(result['files']), 0)
        for f_l in result['files']:
            for k_y in f_l:
                self.assertIn(f_l[k_y], [m_r.metadata[k_y] for m_r in mock_ret_doc])
