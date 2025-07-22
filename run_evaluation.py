import os
import csv

# os.environ['AZ_CONT_NAME'] = 'pdf-docs'
# os.environ[
#     'AZ_CONN_STRING'] = "DefaultEndpointsProtocol=https;AccountName=ahmadpoc;AccountKey=IcS+/AbS7Mub97Mdt/kr2laQrh32OX1lYh5LflJ3gVJqEdHULcWCNJVVPhpqVwdxa0Xdru5s05D0Dt3Lhvx3QQ==;EndpointSuffix=core.windows.net"
#
os.environ['LOCAL'] = 'True'
# os.environ['POSTGRES_CONN_STRING'] = "postgresql://postgres:postgres@localhost:5432/postgres"
os.environ['AZ_CONT_NAME'] = 'pdf-docs'
os.environ['AZ_CONN_STRING'] = "DefaultEndpointsProtocol=https;AccountName=ahmadpoc;AccountKey=IcS+/AbS7Mub97Mdt/kr2laQrh32OX1lYh5LflJ3gVJqEdHULcWCNJVVPhpqVwdxa0Xdru5s05D0Dt3Lhvx3QQ==;EndpointSuffix=core.windows.net"

from dotenv import load_dotenv
load_dotenv()

from test import Evaluation
from doclm import obj
import pandas as pd


def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r


def create_evaluation_set(evaluation_obj, files_path, num_questions, chunk_size, target_file):
    print('Generating Evaluation set')
    eval_set = evaluation_obj.generate_qa_set(files_path,
                                              num_questions=num_questions,
                                              chunk_size=chunk_size)
    df = pd.DataFrame(eval_set)
    df.to_csv(target_file)
    return df


if __name__ == "__main__":

    # Specify Language for evaluation
    evaluation_name = 'no_summary_gpt_3.5_chunk_1500_chroma'
    language = 'en'
    response = {'en': 'INCORRECT', 'de': 'FALSCH'}
    folder_path = './test_doc/english/'
    pdf_files = list_files(folder_path)
    evaluation_obj = Evaluation(language=language)

    # df = create_evaluation_set(evaluation_obj, pdf_files, num_questions=400,
    #                            chunk_size=2,                               # Chunk size is based on number of pages now
    #                            target_file='results/german_qa_complete.csv')

    file = [{'remote_id': file_name} for file_name in pdf_files]
    # Read already generated eval_set
    eval_set_1 = pd.read_csv('results/qa_set')  # .to_dict()
    eval_set = [{'question': row['question'], 'answer': row['answer']} for index, row in eval_set_1.iterrows()]

    print('Parsing document to vector store')
    # Parse documents and store it in vecrtor store
    obj.add_document(file)
    retriever = obj.store_db.as_retriever()

    # Specify filenames from where answer should be searched
    print('Running Evaluation')
    graded_answers, latency, predictions = evaluation_obj.execute_evaluation(eval_set, retriever, obj, file, language)
    data_comp = []

    for eval, pred, grade in zip(eval_set, predictions, graded_answers):
        data_comp.append({'Question': eval['question'],
                          'True_Answer': eval['answer'],
                          'Predicted_Answer': pred['answer'],
                          'Grade': grade['text']
                          })

    data_comp_df = pd.DataFrame(data_comp)
    data_comp_df.to_csv('results/' + evaluation_name + '.csv')
    d = pd.DataFrame(predictions)
    d['answer score'] = [g['text'] for g in graded_answers]
    d['latency'] = latency
    # Summary Statistics
    mean_latency = d['latency'].mean()
    correct_answer_count = len([text for text in d['answer score'] if response[language] not in text])
    percentage_answer = (correct_answer_count / len(graded_answers)) * 100

    print('correct_answer_count', correct_answer_count)
    print('percentage_answer', percentage_answer)
    print('Mean_Latency', mean_latency)
    overall_performance = {
        'Total_Questions': len(eval_set),
        'Correct_answer_count': correct_answer_count,
        'Percentage__correct_answer': percentage_answer,
        'Mean_Latency': mean_latency
    }
    # print(overall_performance)
    with open('results/' + evaluation_name + '_performance', 'w') as f:
        w = csv.writer(f)
        w.writerows(overall_performance.items())

