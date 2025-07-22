import asyncio
import sys
import websockets

from azure.messaging.webpubsubservice import WebPubSubServiceClient


async def connect(url):
    async with websockets.connect(url) as ws:
        print("connected")
        while True:
            ws_token_recved = await ws.recv()
            print(ws_token_recved)
            print(dir(ws_token_recved))
            print("Recieved Message: " + ws_token_recved)


if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print('Usage: python subscribe.py <connection-string> <hub-name>')
    #     exit(1)

    # connection_string = "Endpoint=https://documentlm.webpubsub.azure.com;AccessKey=8yIwvIxhdc3XRpCXv042fKfNwzXQSd8+VcupNIP9JYk=;Version=1.0;"
    # # hub_name = sys.argv[2]

    # service = WebPubSubServiceClient.from_connection_string(
    #     connection_string, hub="Hub"
    # )
    # roles = ["webpubsub.joinLeaveGroup.stream"]
    # token = service.get_client_access_token(
    #     userId="user1", roles=roles, groups=["stream"]
    # )
    # token2 = service.get_client_access_token(userId="user2")
    try:
        url = "wss://documentlm.webpubsub.azure.com/client/hubs/Hub?access_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJodHRwczovL2RvY3VtZW50bG0ud2VicHVic3ViLmF6dXJlLmNvbS9jbGllbnQvaHVicy9IdWIiLCJpYXQiOjE2OTM5OTQzNjIsImV4cCI6MTY5Mzk5Nzk2Miwic3ViIjoidXNlcjIiLCJyb2xlIjpbIndlYnB1YnN1Yi5qb2luTGVhdmVHcm91cC5zdHJlYW0iXX0.fUIVxvKPtx6KHeuZiUZqqErZ7ifZ_09rONDjH-5EHA0"
        # asyncio.get_event_loop().run_until_complete(connect(token2["url"]))
        asyncio.get_event_loop().run_until_complete(connect(url))

    except KeyboardInterrupt as e:
        raise e
