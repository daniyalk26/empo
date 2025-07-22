
from typing import Optional, Type, List, Dict, Any, TypedDict, Union
from pydantic import BaseModel, Field


# class OutBaseMsg(BaseModel):
#     tool_name: Union[str, None]=None
#     task_id: str
#     response: Union[str, None]=None
#     rid: Any
#     chat_id: Any

class FEBaseMsg(BaseModel):
    task_id: str
    response: str = '' # Union[str, None]=None


class OutBEMsgMultiTool(BaseModel):
    files: List=Field(default_factory=list)
    additional_steps: Dict = Field(default_factory=dict)
    tool_name: Union[str, None]=None
    task_id: str
    response: Union[str, None]=None
    search_phrase: Union[str, None] = None


class OutBEMultiTool(BaseModel): #payload
    chat_id: Union[int,None]
    rid: Union[int,None]
    msg: Union[OutBEMsgMultiTool,Any]=None
    time_stamps: List[Dict[str, Any]]=Field(default_factory=list)
    chat_subject: Optional[str]=Field(default=None)
    tokens_usage: List[Dict]=Field(default_factory=list)
    new_subject: Union[str, None]=Field(default=None)
    status: bool=Field(default=True)
    status_details: Union[str,Dict,None]=Field(default=None)
    extras: Dict = Field(default_factory=dict)
    

class InBEMultiTool(BaseModel):
    pass

# class OutPlanMsg(FEBaseMsg):
#     event:str = 'plan'
#     tool_name: Union[str, None]=None


class OutFEErrorMsg(FEBaseMsg):
    code: Union[str,None]
    description: str
    task_id:str =None
    tool_name:str=None

# class planneroutput(Base):
#     plan
#     steps

class OutFEChatMetaMsg(FEBaseMsg):
    # event: str = 'chatmeta'
    class Config:
        extra = 'ignore'
    chat_subject: str
    step_name: str
    additional_steps: Dict = Field(default_factory=dict)
    files: List[Any] = []
    tool_name: str
    search_phrase: Union[str, None]


class OutFEStreamMsg(FEBaseMsg):
    event: str = 'stream'
    step_name: str #rephrase/plan/etc
    tool_name: str

# class OutFETaskStepMsg(FEBaseMsg):
#     step_name: str #rephrase/plan/etc
#     tool_name:str

class OutFETaskStepMsg(FEBaseMsg):
    step_name: str #rephrase/plan/etc
    tool_name:str

# class OutFEModel(FEBaseMsg):
#     rid: Any
#     chat_id: Any
#     event: str #chat_step, chat_meta, error, stream
#     # msg: Union[OutFEErrorMsg,OutFEChatMetaMsg,OutFEStreamMsg,OutFETaskStepMsg]
