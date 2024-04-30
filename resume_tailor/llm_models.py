import os
import re

import google.generativeai as genai
import pandas as pd
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from bigdl.llm.langchain.embeddings import TransformersEmbeddings
from bigdl.llm.langchain.llms import TransformersLLM
from langchain.vectorstores import FAISS
from openai import OpenAI as OriginalOpenAI
from langchain_openai import OpenAI as LangchainOpenAI
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from resume_tailor.utils import parse_json_markdown
from langchain_experimental.llms import LMFormatEnforcer
from transformers import pipeline
from bigdl.llm.transformers import AutoModel, AutoModelForCausalLM
from transformers import AutoTokenizer, LlamaTokenizer


BIGDL = "BigDL-Llama-2-7b-chat"
GPT_3_5 = "gpt-3.5-turbo-0125"
GPT_4 = "gpt-4-1106-preview"
GEMINI_PRO = "gemini-pro"

LOCAL_LLM_VERSION = "llama-2-7b-chat-hf-INT4"

BigDL_INSTANCE = None
HF_MODEL_INSTANCE = None
BigDL_EMBEDDINGS = None

class Certification(BaseModel):
    name: str
    by: str
    link: HttpUrl
class certifications(BaseModel):
    certifications: List[Certification]
class achievements(BaseModel):
    achievements: List[str]
class Project(BaseModel):
    name: str
    type: str
    link: HttpUrl
    from_date: str  
    to: str
    description: List[str]

class projects(BaseModel):
    projects: List[Project]

class Education(BaseModel):
    degree: str
    university: str
    from_date: str
    to: str
    grade: str
    coursework: List[str] = None
    classes: List[str] = None
class education(BaseModel):
    education: List[Education]
class SkillSection(BaseModel):
    name: Optional[str] = None
    skills: Optional[List[str]] = None

class skill_section(BaseModel):
    skill_section: List[SkillSection]
class WorkExperience(BaseModel):
    role: str
    company: str
    location: str
    from_date: str  
    to: str
    description: List[str]

class work_experience(BaseModel):
    work_experience: List[WorkExperience]

class RESUME_DATA_SCHEMA (BaseModel):
  name: str
  summary: str
  phone: str
  email: str
  github: str
  linkedin: str
  education: List[Education]
  skill_section: List[SkillSection]
  work_experience: List[WorkExperience]
  projects: List[Project]
  certifications: List[Certification]
  achievements: List[str]
  
class Job_Detail_SCHEMA (BaseModel):
    title: str 
    keywords: List[str]
    purpose: str
    duties_responsibilities: List[str]
    required_qualifications: List[str]
    preferred_qualifications: List[str]
    company_name: str
    company_info: str


SECTION_TO_MODEL = {
    "RESUME_DATA_SCHEMA": RESUME_DATA_SCHEMA,
    "education": education,
    "skill_section": skill_section,
    'work_experience': work_experience,
    'projects': projects,
    'certifications': certifications,
    'achievements': achievements,
    "Job_Detail_SCHEMA": Job_Detail_SCHEMA,
}

checkpoints_folder_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),'checkpoints')
def get_bigdl_llm():
    global BigDL_INSTANCE
    if BigDL_INSTANCE is None:
        BigDL_INSTANCE = TransformersLLM.from_model_id_low_bit(
            os.path.join(
                checkpoints_folder_path,
                LOCAL_LLM_VERSION
            )
        )
        BigDL_INSTANCE.streaming = False
    return BigDL_INSTANCE

def get_bigdl_hf_model():
    global HF_MODEL_INSTANCE
    model_path = os.path.join(
        checkpoints_folder_path,
        LOCAL_LLM_VERSION
    )
    if HF_MODEL_INSTANCE is None:
        model = AutoModelForCausalLM.load_low_bit(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        HF_MODEL_INSTANCE = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=4096)
        HF_MODEL_INSTANCE.streaming = False
    return HF_MODEL_INSTANCE

def get_miniLM_embedding():
    global BigDL_EMBEDDINGS
    if BigDL_EMBEDDINGS is None:
        embedding_path = os.path.join(
            checkpoints_folder_path,
            'all-MiniLM-L12-v2'
        )
        BigDL_EMBEDDINGS = TransformersEmbeddings.from_model_id(
            model_id=embedding_path)
    return BigDL_EMBEDDINGS

class BigDL_LLM:
    def __init__(self, system_prompt):
        system_prompt = system_prompt.strip()
        if system_prompt:
            self.system_prompt = system_prompt

        self.embeddings = get_miniLM_embedding()
        self.llm=None
        self.hf_model=None
        self.generation_kwargs = {
            "max_new_tokens": 4096,
            "top_p": 0.9,  # delete the low possibility answer
            "temperature": 0.4,
            "repetition_penalty": 1.2,
            "do_sample": True,
        }

    def get_llm(self):
        if self.llm is None:
            self.llm = get_bigdl_llm()
        return self.llm

    def get_hf(self):
        if self.hf_model is None:
            self.hf_model =get_bigdl_hf_model()
        return self.hf_model

    def get_response(self, prompt, expecting_longer_output=False, need_json_output=False, section_schema=None):
        print("function: get_response")
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        prompt = f"{B_INST} {B_SYS} {self.system_prompt} {E_SYS} {prompt} {E_INST}"
        
        if need_json_output:
            if section_schema is None:
                raise ValueError("Section_schema should not be None")
            section_model = SECTION_TO_MODEL[section_schema]
            lm_format_enforcer = LMFormatEnforcer(json_schema=section_model.schema(), pipeline=self.get_hf())
            content = lm_format_enforcer.invoke(prompt)
            index = content.find("[/INST]")
            # 获取特定子字符串之后的字符串
            json_str = content[index + len("[/INST]"):]
            print("model output json:\n", json_str)
            try:
                return parse_json_markdown(json_str)
            except Exception as e:
                print("BigDL_LLM.get_response")
                print("model output json:\n", json_str)
                print(e)
                return None
        else:
            content = self.get_llm().invoke(prompt, **self.generation_kwargs)
            return content

    def get_text_embedding(self, text):
        try:
            vector_embedding = FAISS.from_texts(texts=text, embedding=self.embeddings)
            return vector_embedding
        except Exception as e:
            print(e)
            return None


class ChatGPT:
    def __init__(self, gpt_type, api_key, system_prompt):
        self.gpt_type = gpt_type
        if system_prompt.strip():
            self.system_prompt = {"role": "system", "content": system_prompt}
        self.api_key = api_key
        self.llm_for_chat = OriginalOpenAI(api_key=self.api_key)
        self.embeddings=OpenAIEmbeddings(openai_api_key=self.api_key)

    def get_llm(self):
        """get general llm for specific task propose"""
        return LangchainOpenAI(openai_api_key=self.api_key)

    def get_response(self, prompt, expecting_longer_output=False, need_json_output=False, section_schema=None):
        if section_schema is not None:
            raise ValueError("Section schema is not supported in OpenAI.")

        user_prompt = {"role": "user", "content": prompt}
        try:
            completion = self.llm_for_chat.chat.completions.create(
                model=self.gpt_type,
                messages=[self.system_prompt, user_prompt],
                temperature=0.2,
                # Lower values for temperature result in more consistent outputs (e.g. 0.2), while higher values generate more diverse and creative results (e.g. 1.0). Select a temperature value based on the desired trade-off between coherence and creativity for your specific application. The temperature can range is from 0 to 2.
                max_tokens=4000 if expecting_longer_output else None,
                response_format={"type": "json_object"} if need_json_output else None  # enable json mode
            )

            response = completion.choices[0].message
            content = response.content.strip()

            if need_json_output:
                return parse_json_markdown(content)
            else:
                return content

        except Exception as e:
            print(e)
            st.error(f"Error in OpenAI API, {e}")
            st.markdown("<h3 style='text-align: center;'>Please try again!</h3>", unsafe_allow_html=True)

    def get_text_embedding(self, text, model="text-embedding-ada-002", task_type="retrieval_document"):
        try:
            text = text.replace("\n", " ")
            return self.llm_for_chat.embeddings.create(input=[text], model=model).data[0].embedding
        except Exception as e:
            print(e)


# class Gemini:
#     def __init__(self, api_key, system_prompt):
#         genai.configure(api_key=api_key)
#         self.system_prompt = "System Prompt\n======\n" + system_prompt if system_prompt.strip() else ""
#
#     def get_response(self, prompt, expecting_longer_output=False, need_json_output=False):
#         try:
#             user_prompt = "\n\nUser Prompt\n======\n" + prompt
#             entire_prompt = self.system_prompt + user_prompt
#
#             model = genai.GenerativeModel('gemini-pro')
#             content = model.generate_content(
#                 entire_prompt,
#                 generation_config={
#                     "temperature": 0.7,
#                     "max_output_tokens": 4000 if expecting_longer_output else None,
#                 }
#             )
#
#             if need_json_output:
#                 result = parse_json_markdown(content.text)
#             else:
#                 result = content.text
#
#             if result is None:
#                 st.write("LLM Response")
#                 st.markdown(f"```json\n{content.text}\n```")
#
#             return result
#
#         except Exception as e:
#             print(e)
#             st.error(f"Error in Gemini API, {e}")
#             st.markdown("<h3 style='text-align: center;'>Please try again!</h3>", unsafe_allow_html=True)
#             return None
#
#     def get_embedding(self, content, model="models/embedding-001", task_type="retrieval_document"):
#         try:
#             def embed_fn(data):
#                 result = genai.embed_content(
#                     model=model,
#                     content=data,
#                     task_type=task_type,
#                     title="Embedding of json text" if task_type in ["retrieval_document", "document"] else None)
#
#                 return result['embedding']
#
#             df = pd.DataFrame(content)
#             df.columns = ['chunk']
#             df['embedding'] = df.apply(lambda row: embed_fn(row['chunk']), axis=1)
#
#             return df
#
#         except Exception as e:
#             print(e)
#
#
# class Llama2:
#     def __init__(self, hf_token, system_prompt):
#         # !pip install sentencepiece==0.1.99
#         # !pip install transformers==4.31.0
#         # !pip install accelerate==0.21.0
#         # !pip install bitsandbytes==0.41.1
#         # https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212
#
#         from transformers import LlamaForCausalLM, LlamaTokenizer
#
#         self.system_prompt = system_prompt
#         self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=hf_token)
#         self.model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", load_in_8bit=True,
#                                                       device_map="auto", token=hf_token)
#         self.generation_kwargs = {
#             "max_new_tokens": 512,
#             "top_p": 0.9,
#             "temperature": 0.6,
#             "repetition_penalty": 1.2,
#             "do_sample": True,
#         }
#
#     def get_response(self, prompt_text, need_json_output=False):
#         B_INST, E_INST = "[INST]", "[/INST]"
#         B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
#
#         # Special format required by the Llama2 Chat Model where we can use system messages to provide more context about the task
#         prompt = f"{B_INST} {B_SYS} {self.system_prompt} {E_SYS} {prompt_text} {E_INST}"
#
#         prompt_ids = self.tokenizer(prompt, return_tensors="pt")
#         prompt_size = prompt_ids['input_ids'].size()[1]
#
#         generate_ids = self.model.generate(prompt_ids.input_ids.to(self.model.device), **self.generation_kwargs)
#         generate_ids = generate_ids.squeeze()
#
#         response = self.tokenizer.decode(generate_ids.squeeze()[prompt_size + 1:], skip_special_tokens=True).strip()
#
#         if need_json_output:
#             return parse_json_markdown(response)
#         else:
#             return response
#
#         return response

# DO: https://ai.google.dev/tutorials/python_quickstart#use_embeddings
# def compute_embedding(self, chunks):
#     try:
#         embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         vector_embedding = FAISS.from_texts( texts = chunks, embedding=embeddings)
#         return vector_embedding
#     except Exception as e:
#         print(e)
#         return None

# Define a function to compute embeddings for the text   
# def compute_embedding(self, text):
#     response = openai.Embed(
#         input=text,
#         model="text-davinci-003-001",
#         max_tokens=50
#     )
#     return response['embedding']
