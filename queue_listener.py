import os
from azure.storage.queue import QueueClient
import logging
import time
from subprocess import Popen
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]:%(levelname)s:%(name)s:%(funcName)s: %(message)s')

def main():
    conn_str = os.environ['AZ_CONN_STRING']
    queue_name = os.environ.get('QUEUE_NAME', "doclm-simple")
    queue_client = QueueClient.from_connection_string(
        conn_str=conn_str,
        queue_name=queue_name,
    )
    logging.info(f"Queue client created for {queue_name}")
    while True:
        try:
            logging.info("Checking the queue")
            props = queue_client.get_queue_properties()
            logging.info(f"Queue {queue_name} has {props.approximate_message_count} messages")
            if props and props.approximate_message_count:
                logging.info("Starting the processing job")
                process= Popen(['python', 'main.py'])
                process.wait()
                logging.info("Processing job completed")
            time.sleep(30)

        except Exception as ex:
            logging.error(str(ex), exc_info=True)

if __name__ == "__main__":
    main()
