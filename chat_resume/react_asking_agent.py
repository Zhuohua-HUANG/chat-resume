import enum
import json

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.language_models.base import BaseLanguageModel
import streamlit as st
from chat_resume.tools.syn_asking import UserAskingTool

class ExperienceType(enum.IntEnum):
    project = 1
    work = 2


def get_prompt(e_type: ExperienceType) -> str:
    if e_type == ExperienceType.project:
        prompt = hub.pull("absurd/get-applicant-project-experience")
    elif e_type == ExperienceType.work:
        prompt = hub.pull("absurd/get-applicant-work-experience")
    else:
        raise ValueError(f"experience type {e_type} not valid")
    return prompt


class ReactAskingAgent(object):

    def __init__(self, llm: BaseLanguageModel, experience_type: ExperienceType):
        self.tools = [UserAskingTool()]
        prompt = get_prompt(experience_type)
        self.agent = create_react_agent(llm, self.tools, prompt)

    def get_experience_document(self, experience: json, experience_type: ExperienceType):
        if experience_type==ExperienceType.project:
            st.subheader("Chat with Resume Assistant on Experience of " + experience["name"]+" at "+experience["type"])
        elif experience_type==ExperienceType.work:
            st.subheader("Chat with Resume Assistant on Experience of "+experience["role"]+" at "+experience["company"])
        st.write("Input your answer on command line! \nIf you want to stop the conversation of this experience then "
                 "input \"NO MORE INFO\". ")
        agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
        input_output_dict = agent_executor.invoke(
            {
                "input": experience,
            }
        )
        document = input_output_dict["output"]
        st.markdown("---")
        return document
