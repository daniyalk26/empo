

This repository package contains implementations to do document level  querying using OpenApi language model

API requirements

features addition


```
from doclm import interactive
obj = interactive()

obj.add_document(
        :param files: list[dict(with 'url' as key)]
        :param num_pages: optional number of page used to store 
                          in references 
        :param **kwargs: 'cb' as call back function for be called 
                         after execution
        ) 
        calls 'cb' after execution/document addition with metadata 
         of document inserted
obj.ask_question(  
    :param user_input: str
    :param chat_history: str default ''
    :param files: default None, list[dict(with 'url' as key)]
    :param kwargs: 
     ) return-> str
     
obj.delete_document(
        :param files: list[dict(with 'url' as key)]
        )
```