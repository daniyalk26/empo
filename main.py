import os
import time
import json
import logging
import urllib
import hashlib
import requests
from azure.storage.queue import QueueClient

from tenacity import retry, stop_after_attempt, wait_fixed
from doclm.processing import get_processor
from doclm.schema import Schema

log = logging.getLogger("doclogger")

MAX_ATTEMPTS = int(os.getenv('MAX_RETRIES', 3))
BASE_URL = os.environ['API_SERVER_URL']
# Authorization: encrypted(api_key:api_secret: ts)
API_KEY = os.environ['DLM_API_KEY']
ENCRYPT_KEY_NAME = os.environ['DLM_ENCRYPTION_KEYNAME']
API_SECRET = hashlib.sha512((os.environ['DLM_API_SECRET']).encode('utf-8')).hexdigest()


def get_headers():
    ts = str(int(time.time()))
    return {
        'X-DLM-Request-ID': ts,
        'Authorization': Schema.encrypt(f"{API_KEY}:{API_SECRET}:{ts}", name=ENCRYPT_KEY_NAME)
    }


@retry(wait=wait_fixed(3), stop=(stop_after_attempt(MAX_ATTEMPTS)))
def send_data(url, headers, files):
    response = requests.post(url, headers=headers, files=files)


def callback_function(files, index=None, **kwargs):
    index = kwargs.get('index', index)
    url = BASE_URL + '/jobs/files/processing/status'

    json_data = json.dumps({
        'files': files,
        'index': index,
    },
    )
    files_list = {
        'data': (None, Schema.encrypt(json_data, name=ENCRYPT_KEY_NAME)),
    }
    headers = get_headers()
    send_data(url, headers, files_list)
    # for i in range(retries):
    #     try:
    #         response = requests.post(url, headers=get_headers(), files=files_list)
    #
    #         if response.status_code == 200:
    #             return True
    #         log.error('Can not connect to url %s, status code %s', url, response.status_code)
    #     except Exception as e:
    #         log.error(e, exc_info=True)
    #         log.info('%s tries', i)

        # time.sleep(60)
    # raise ConnectionError('unable to send acknowledge')


def get_parsed_att_from_files(file, name):
    return urllib.parse.quote(
        Schema.encrypt(str(file[name]), name=ENCRYPT_KEY_NAME)
    )


def limit_available(file, **kwargs):
    app_id = get_parsed_att_from_files(file, 'application_id')
    tenant_id = get_parsed_att_from_files(file, 'tenant_id')

    url = BASE_URL + f'/jobs/license/info?app_id={app_id}&tenant_id={tenant_id}'

    response = requests.get(url, headers=get_headers())
    r_data = json.loads(response.content)
    # response['ret_val']['data']
    # response['ret_val']['data']['credit_limit']
    # response['ret_val']['data']['consumed_credit_limit']
    status_file = True
    status_details = None
    if r_data['ret_val']['data']['consumed_word_limit'] > r_data['ret_val']['data']['word_limit']:
        status_file = False
        status_details =  {"code": 'word limit', "description": "unavailable word limit"}
    if r_data['ret_val']['data']['consumed_credit_limit'] > r_data['ret_val']['data']['credit_limit']:
        status_file = False
        status_details =  {"code": 'credit limit', "description": "unavailable credit limit"}


    file['status'] = status_file
    file['status_details_update'] = status_details

    return status_file


def main(q_name):
    files_list = None
    message = None
    connection_string = os.environ['AZ_CONN_STRING']
    queue_client = QueueClient.from_connection_string(
        conn_str=connection_string,
        queue_name=q_name,
    )
    processed_files_idx = []

    try:
        # key = get_psk_by_name(encrypt_key)
        message = queue_client.receive_message(visibility_timeout=int(os.getenv('VISIBILITY_TIMEOUT', 60)))
        if message is not None:
            queue_client.delete_message(message)
            val = Schema.decrypt(message.content, name=ENCRYPT_KEY_NAME)
            log.debug("Got message %s ", val)
            files_list = json.loads(val)
            assert isinstance(files_list, list), '%s is not of type list' % type(files_list)
            obj = get_processor()
            for idx, file in enumerate(files_list):
                file['status']=True
                if limit_available(file):
                # check if data exits?
                    if not obj.store_db.if_exists([file]):
                        obj.sync_add_document(files=[file])
                callback_function([file])
                processed_files_idx.append(idx)


    except Exception as ex:
        log.error(str(ex), exc_info=True)
        if files_list:
            new_files_list = [file for idx, file in enumerate(files_list) if idx not in processed_files_idx]
            payload = Schema.encrypt(json.dumps(new_files_list), name=ENCRYPT_KEY_NAME)
            queue_client.send_message(payload)

        elif message:
            queue_client.send_message(message.content)
        raise ex

    finally:
        queue_client.close()


if __name__ == '__main__':
    que_name = os.getenv('QUEUE_NAME', "doclm-simple")
    if que_name:
        main(que_name)
