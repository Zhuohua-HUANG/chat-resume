import os
import re

import google.generativeai as genai
import pandas as pd
import streamlit as st
from bigdl.llm.langchain.embeddings import TransformersEmbeddings
from bigdl.llm.langchain.llms import TransformersLLM
from langchain.vectorstores import FAISS
from openai import OpenAI

from resume_tailor.utils import parse_json_markdown

BIGDL = "BigDL-Llama-2-7b-chat"
GPT_3_5 = "gpt-3.5-turbo-0125"
GPT_4 = "gpt-4-1106-preview"
GEMINI_PRO = "gemini-pro"

LOCAL_LLM_VERSION = "llama-2-7b-chat-hf-INT4"

BigDL_INSTANCE = None
BigDL_EMBEDDINGS = None


def get_bigdl_instance_and_embedding():
    global BigDL_INSTANCE, BigDL_EMBEDDINGS
    resume_tailor_folder_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    if BigDL_INSTANCE is not None and BigDL_EMBEDDINGS is not None:
        return BigDL_INSTANCE, BigDL_EMBEDDINGS
    else:
        model_path = os.path.join(
            resume_tailor_folder_path,
            'checkpoints/' + LOCAL_LLM_VERSION
        )
        if BigDL_INSTANCE is None:
            BigDL_INSTANCE = TransformersLLM.from_model_id_low_bit(model_path)
            BigDL_INSTANCE.streaming = False
        if BigDL_EMBEDDINGS is None:
            embedding_path = os.path.join(
                resume_tailor_folder_path,
                'checkpoints/' + 'all-MiniLM-L12-v2'
            )
            BigDL_EMBEDDINGS = TransformersEmbeddings.from_model_id(
                model_id=embedding_path)
        return BigDL_INSTANCE, BigDL_EMBEDDINGS


class BigDL_LLM:
    def __init__(self, system_prompt):
        system_prompt = system_prompt.strip()
        if system_prompt:
            self.system_prompt = system_prompt

        self.llm, self.embeddings = get_bigdl_instance_and_embedding()

        self.generation_kwargs = {
            "max_new_tokens": 4096,
            "top_p": 0.9,  # delete the low possibility answer
            "temperature": 0.4,
            "repetition_penalty": 1.2,
            "do_sample": True,
        }

    def get_response(self, prompt, expecting_longer_output=False, need_json_output=False):
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        prompt = f"{B_INST} {B_SYS} {self.system_prompt} {E_SYS} {prompt} {E_INST}"

        content = self.llm.invoke(prompt, **self.generation_kwargs)
        if need_json_output:
            # Use regular expression to find JSON string
            match = re.search(r'\{.*\}', content, flags=re.DOTALL)
            if match:
                # If match found, use the matched JSON string
                json_str = match.group(0)
                try:
                    return parse_json_markdown(json_str)
                except Exception as e:
                    print("BigDL_LLM.get_response")
                    print("model output json:\n", json_str)
                    print(e)
                    return None
            else:
                raise Exception("The output of BigDL-LLM does not contain JSON.")
                return None
        else:
            return content

    def get_embedding(self, text):
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
        self.client = OpenAI(api_key=api_key)

    def get_response(self, prompt, expecting_longer_output=False, need_json_output=False):
        user_prompt = {"role": "user", "content": prompt}

        try:
            completion = self.client.chat.completions.create(
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

    def get_embedding(self, text, model="text-embedding-ada-002", task_type="retrieval_document"):
        try:
            text = text.replace("\n", " ")
            return self.client.embeddings.create(input=[text], model=model).data[0].embedding
        except Exception as e:
            print(e)


class Gemini:
    def __init__(self, api_key, system_prompt):
        genai.configure(api_key=api_key)
        self.system_prompt = "System Prompt\n======\n" + system_prompt if system_prompt.strip() else ""

    def get_response(self, prompt, expecting_longer_output=False, need_json_output=False):
        try:
            user_prompt = "\n\nUser Prompt\n======\n" + prompt
            entire_prompt = self.system_prompt + user_prompt

            model = genai.GenerativeModel('gemini-pro')
            content = model.generate_content(
                entire_prompt,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 4000 if expecting_longer_output else None,
                }
            )

            if need_json_output:
                result = parse_json_markdown(content.text)
            else:
                result = content.text

            if result is None:
                st.write("LLM Response")
                st.markdown(f"```json\n{content.text}\n```")

            return result

        except Exception as e:
            print(e)
            st.error(f"Error in Gemini API, {e}")
            st.markdown("<h3 style='text-align: center;'>Please try again!</h3>", unsafe_allow_html=True)
            return None

    def get_embedding(self, content, model="models/embedding-001", task_type="retrieval_document"):
        try:
            def embed_fn(data):
                result = genai.embed_content(
                    model=model,
                    content=data,
                    task_type=task_type,
                    title="Embedding of json text" if task_type in ["retrieval_document", "document"] else None)

                return result['embedding']

            df = pd.DataFrame(content)
            df.columns = ['chunk']
            df['embedding'] = df.apply(lambda row: embed_fn(row['chunk']), axis=1)

            return df

        except Exception as e:
            print(e)


class Llama2:
    def __init__(self, hf_token, system_prompt):
        # !pip install sentencepiece==0.1.99
        # !pip install transformers==4.31.0
        # !pip install accelerate==0.21.0
        # !pip install bitsandbytes==0.41.1
        # https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212

        from transformers import LlamaForCausalLM, LlamaTokenizer

        self.system_prompt = system_prompt
        self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=hf_token)
        self.model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", load_in_8bit=True,
                                                      device_map="auto", token=hf_token)
        self.generation_kwargs = {
            "max_new_tokens": 512,
            "top_p": 0.9,
            "temperature": 0.6,
            "repetition_penalty": 1.2,
            "do_sample": True,
        }

    def get_response(self, prompt_text, need_json_output=False):
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

        # Special format required by the Llama2 Chat Model where we can use system messages to provide more context about the task
        prompt = f"{B_INST} {B_SYS} {self.system_prompt} {E_SYS} {prompt_text} {E_INST}"

        prompt_ids = self.tokenizer(prompt, return_tensors="pt")
        prompt_size = prompt_ids['input_ids'].size()[1]

        generate_ids = self.model.generate(prompt_ids.input_ids.to(self.model.device), **self.generation_kwargs)
        generate_ids = generate_ids.squeeze()

        response = self.tokenizer.decode(generate_ids.squeeze()[prompt_size + 1:], skip_special_tokens=True).strip()

        if need_json_output:
            return parse_json_markdown(response)
        else:
            return response

        return response

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
