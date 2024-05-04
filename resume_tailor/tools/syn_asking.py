from typing import Optional, Type
import streamlit as st

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool


class AskingInput(BaseModel):
    question: str = Field(description="the specific question with detailed question background to applicant")


class UserAskingTool(BaseTool):
    name = "asking_applicant"
    description = "Can use to ask applicant a specific question with detailed question background"
    args_schema: Type[BaseModel] = AskingInput

    def _run(
            self, question: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        with st.chat_message("assistant"):
            st.markdown(question)
        answer=input("\n\nQuestion:\n"+question+"\n\nYour answer:\n")
        with st.chat_message("user"):
            st.markdown(answer)
        return answer

    async def _arun(
            self, question: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("asking_applicant does not support async")
