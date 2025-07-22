
# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

# Envioment variables for testing

- AZ_WPS_ACCESS_KEY=<your key>
- AZ_WPS_HUB="Hub"
- CHUNK_SIZE='1200'
- CHUNK_OVERLAP='250'
- CHUNK_EXTEND='title'
- PARSER_TYPE='simple'
- LOGDIR=./log
- AZ_CONN_STRING=<your key>
- AZ_CONT_NAME=<your name>
- AZ_WPS_ACCESS_KEY=<your key>
- AZ_WPS_ENDPOINT=<your name>
- AZ_WPS_HUB=<your name>
- AZURE_OPENAI_ENDPOINT=<your address>
- AZURE_OPENAI_API_KEY=<your key>
- DIRECT_CHAT_DEPLOYMENT_NAME_4_1=gpt-4.1
- DIRECT_CHAT_MODEL_NAME_4_1=gpt-4.1
- DIRECT_CHAT_DEPLOYMENT_NAME_4o=gpt4o-fast
- DIRECT_CHAT_MODEL_NAME_4o=gpt-4o
- O1_DEPLOYMENT_NAME=o1
- O1_MODEL_NAME=o1
- O4_MINI_DEPLOYMENT_NAME=o1=4-mini
- O4_MINI_MODEL_NAME=o4-mini
- EMBED_DEPLOYMENT_NAME=embedding_ada
- EMBED_MODEL_NAME=text-embedding-ada-002
- LLM_COMMON_DEPLOYMENT_NAME=gpt-4o
- LLM_COMMON_MODEL_NAME=gpt-4o
- LLM_TRANSLATION_DEPLOYMENT_NAME=doc_llm_large_3_5
- LLM_TRANSLATION_MODEL_NAME=gpt-4-32k
- MISTRAL_END_POINT=<your endpoint>
- MISTRAL_API_KEY==<your key>
- LLAMA_31_70B_END_POINT=<your endpoint>
- LLAMA_31_70B_API_KEY=<your key>
- LLAMA_31_408_END_POINT=<your endpoint>
- LLAMA_31_408_API_KEY=<your key>
- DEEPSEEK_V3_END_POINT=<your endpoint>
- DEEPSEEK_V3_API_KEY=<your key>
- COHERE_END_POINT=<your endpoint>
- COHERE_API_KEY=<your key>
- LLM_COMMON=gpt-4o
- OPENAI_API_TYPE=azure
- OPENAI_API_VERSION=2024-02-01
- OPENAI_CHAT_API_VERSION=2024-02-01
- SERPER_API_KEY= `make account on web  `https://serper.dev/` `
- WEB_SEARCH_REGION=en
- ENCRYPT=True


## [0.3.4]rc - 2025-06-17
### Added
  - new models added
    - mistral 2411
    - cohere r+ 08-2024
    - o1
    - o4-mini
    - gpt 4.1
    - deepseek v3-0324

  * more detail [here](https://intechww.atlassian.net/wiki/spaces/~55705861222359b93c4fdb8d4e5a67e4c0efb4/pages/1525088260/Model+Retirement+Dates+Replacements)


## [0.2.5]c - 2024-05-27
### Added
 env variable added LLM_COMMON=gpt-3.5-turbo to used for general work load


## [0.2.5]c - 2024-05-27
Envs to limit output token for each chain
- MAX_TOKENS_ROUTER default 150
- MAX_TOKENS_REPHRASE default 500
- MAX_TOKENS_CHAT  default 3000
- MAX_TOKENS_TITLE default 100
- MAX_TOKENS_SUMMARY default 500

## [0.2.5]b - 2024-05-15
### Added
cross-lingual updates,
fast-text added in requirements
  - LLM_TRANSLATION_DEPLOYMENT_NAME
  - LLM_TRANSLATION_MODEL_NAME

 
## [0.2.5]b - 2024-05-15

### Added
- New models support in rag with **'model_name'** arg in ask_question
### changed
  unified model selection for direct/rag models

## [0.2.5] - 2024-05-14

### Fixed
  - Streaming issues 
  - Token extraction from response (rest endpoint)
  - logging issues (jazib jameel)

### changed
- upgraded langcahin to 0.1.14 and openai 1.28.0
- changed envs
  - OPENAI_API_KEY --> AZURE_OPENAI_API_KEY 
  - OPENAI_API_BASE --> AZURE_OPENAI_ENDPOINT
  - OPENAI_API_VERSION = 2024-02-01

### Added
- cross-lingual support (saad hassan)
- Error propagation for frontend
- **'model_name'** arg in  ask_question for frontend access can be selected from 
[Mistral-Large, Llama-3-70B-Instruct, gpt-3.5-turbo-16k,  gpt-4-32k]
- New model support with envs required (ahmad khalid)
  - Mistral
    - MISTRAL_DEPLOY_NAME
    - MISTRAL_END_POINT
  - llama
    - LLAMA_DEPLOY_NAME 
    - LLAMA_END_POINT
  - gpt3.5-15k
    - DIRECT_CHAT_DEPLOYMENT_NAME_35
    - DIRECT_CHAT_MODEL_NAME_35
  - gpt-4-32k
    - DIRECT_CHAT_DEPLOYMENT_NAME_4
    - DIRECT_CHAT_MODEL_NAME_4


## [0.0.3] - 2023-10-16
 
Here we write upgrading notes for brands. It's a team effort to make them as
straightforward as possible.
 
### Added
- Router Chain Implementation updated. Implemented custom router to decouple detection and calling step
- aadd_document async support
- structure aware parsing support 
- filter relevant sources from answer using open ai function calling api
- filter relevant sources from answer using regex
- evaluation code added for internal evaluation
- token tracking for different openai model calls
- unittest added for sanity check
- streaming support to front end
- support for 16k context model
- aask_question async support
- support for local file store reading 
- post_processing of chuck after db retrieval 
- ppt support using python-pptx
 
### Changed
- refactor whole code base
- replaced completion model with chat model
- langchain migration to latest version
- async execution of ask_question, using thread pool
- Token splitter back to Recursive splitter
- Instead of comparing just dummy answer to the chunks in db, we now compare rephrased question and dummy answer both
- Implemented fall back strategy for structured parser to use simple parser if no heading were extracted successfully


### Fixed
- log generation streamlined
- LLM prompt templated updated
- uuid removed from file name for display to end user
- changed argument name wps_conn_id 
- Encryption before sending to pub/sub server
- Encryption related fixes
- minor fixes for ask_question results handling 
- updated chat history to only include first 300 words of each answer
- updated sql filter for fast data retrieval and corresponding filter building