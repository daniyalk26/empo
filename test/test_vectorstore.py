
import unittest
from unittest.mock import Mock, patch, DEFAULT
from unittest.mock import MagicMock
# from unittest.mock import ANY
import numpy as np

from langchain.schema.embeddings import Embeddings
from langchain.docstore.document import Document

from dotenv import load_dotenv
import sys

sys.path.append('../..')
if True:
    load_dotenv()
from doclm.vector_store.custom_pgvector import DocumentPGVector
# from doclm.util import Schema


connection_string = 'localhost:5432'


# def import_module_from_path(path: str) -> types.ModuleType:
#     """Import a module from the given path."""
#     module_path = pathlib.Path(path).resolve()
#     print(module_path)
#     module_name = module_path.stem  # 'path/x.py' -> 'x'
#     spec = importlib.util.spec_from_file_location(module_name, module_path)
#     module = importlib.util.module_from_spec(spec)
#     sys.modules[module_name] = module
#     spec.loader.exec_module(module)
#     importlib.util.find_spec()
#     return module
# def test_import(self):
#         my_module = import_module_from_path('../doclm/vector_store/custom_chroma.py')
#         my_module = import_module_from_path('/home/ahmad/Codes/BitBucketRepository/documentlm/doclm/vector_store/')
#         path = '../doclm/vector_store/'
#         module_path = pathlib.Path(path).resolve()
#         print(module_path)
#
#         importlib.import_module('CustomPGVector', '/home/ahmad/Codes/BitBucketRepository/documentlm/doclm/vector_store/custom_pgvector.py')

embedding_length = 1536
class embeddings(Embeddings):
    def embed_documents(self, texts):
        return [[0.1] * embedding_length for i in texts]

    def embed_query(self, text: str):
        return None


class TestChroma(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_add_(self):
        pass


class TestPGvector(unittest.TestCase):

    # @patch('doclm.Interactive')
    @patch('sqlalchemy.create_engine')
    def test_connection_string(self, mock_sqlalchemy_connection):
        db = DocumentPGVector(connection_string=connection_string, embedding_function=embeddings())
        mock_sqlalchemy_connection.assert_called_once()
        mock_sqlalchemy_connection.assert_called_with(connection_string)

    # @patch('sqlalchemy.orm.Query.filter')
    # @patch("sqlalchemy.orm.Query")
    @patch('langchain.vectorstores.pgvector.PGVector.add_embeddings')
    @patch('sqlalchemy.create_engine')
    def test_add_data(self, mock_sql, pg_vector_mock):
        embed = embeddings()
        db = DocumentPGVector(connection_string=connection_string, embedding_function=embed)
        text = ['ssss']
        metadata = [{'name': 'test', 'document_summary': 'summary'}]

        db.add_texts(text, metadata)
        pg_vector_mock.assert_called_with(
            texts=text, embeddings=embed.embed_documents(text),
            metadatas=metadata, ids=[i['name'] for i in metadata])

        # expected = [['fake', 'row', 1], ['fake', 'row', 2]]
        # mock_con = mock_connect.return_value  # result of psycopg2.connect(**connection_stuff)
        # mock_cur = mock_con.cursor.return_value  # result of con.cursor(cursor_factory=DictCursor)
        # mock_cur.fetchall.return_value = expected  #
        # db.update_file_name('test', 'test', 1)
        # mock_connect.assert_called_with(**connection_stuff)
        # mock_con.cursor.asset_called_with(cursor_factory=DictCursor)
        # mock_cur.execute.assert_called_with("Super duper SQL query")

    @patch('langchain.vectorstores.pgvector.PGVector.add_embeddings')
    @patch('sqlalchemy.create_engine')
    # @patch.object('_create_engine')
    def test_delete(self, mock_sqlalchemy, pg_vector_mock):
        embed = embeddings()
        db = DocumentPGVector(connection_string='test', embedding_function=embed)
        text = ['ssss']
        metadata = [{'name': 'test', 'document_summary': 'summary'}]
        # db.delete_file()

        ids, join_text = db._pre_process(text, metadata, None)
        self.assertEqual(ids, [i['name'] for i in metadata])

        self.assertEqual(join_text,
                         [m['document_summary'] + "\n" + m.get('heading', '') + "\n" + t
                          for t, m in zip(list(text), metadata)])
        result_in = []
        for loop in [1, 1, 3, 4]:
            result = MagicMock()
            result.EmbeddingStore.cmetadata = {
                'id': loop,
                'source': f'../file{loop}.pdf',
                'name': f'original_name_{loop}',
                'page': loop,
                'document_summary': f'summary_{loop}',
                'encrypted': False
                                                  }
            result.EmbeddingStore.distance = 0.5
            result.EmbeddingStore.document = 'some string'
            result.EmbeddingStore.custom_id = f'file_{loop}'
            result_in.append(result)

        # result.EmbeddingStore =
        tr_result = db._results_to_docs_and_scores(result_in)
        self.assertIsInstance(tr_result, list)

        for r, g in zip(tr_result, result_in):
            self.assertIsInstance(r[0], Document)
            self.assertEqual(r[0].page_content, g.EmbeddingStore.document)
            self.assertEqual(r[0].metadata['name'], g.EmbeddingStore.custom_id)

        embedding_list = embed.embed_documents(list(range(2)))
        for dif in range(len(result_in)-len(embedding_list)):
            embedding_list.append(np.random.random(embedding_length))
        # embedding_list.append(np.random.random(embedding_length))

        div_results = db._diverse_docs_filter(tr_result, embedding_list)
        self.assertLessEqual(len(div_results), len(result_in))

        r_scale_results = db._rescale_scores(div_results)
        self.assertAlmostEqual(sum([s[1] for s in r_scale_results]), 1)

        fil_results = db._filter_results(r_scale_results)
        self.assertLess(len(fil_results), len(r_scale_results))
    #     db.similarity_search_with_score_by_vector()


#     args = {'DB': 'test'}
#     # db.get_all_pos(args)
#     mock_sqlalchemy.assert_called_once()
#     mock_sqlalchemy.assert_called_with({'DB': 'test'})
# @patch.object(SomeClass, 'attribute', sentinel.attribute)

# @patch('sqlalchemy.create_engine.connect')
# @patch.object(CustomPGVector, 'connect')
# def test_something(self, test_result):
#     db = CustomPGVector('localhost:5432', 'sss', )
#     # self.assertEqual()
#
#     print(test_result)
#     # instance = v1.return_value

# test_result.assert_called()
# test_result.assert_called_with('localhost:5432')
# instance.assert_called_with('localhost:5432')


if __name__ == "__main__":
    unittest.main()
