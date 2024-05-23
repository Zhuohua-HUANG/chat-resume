import json
import os
import random

import numpy as np
import streamlit as st

from chat_resume import prompt_path, demo_data_path
from chat_resume.data_extraction import extract_plain_text_from_pdf, get_url_content
from chat_resume.latex_ops import json_to_latex_to_pdf
from chat_resume.llm_models import BIGDL, GPT_4, GPT_3_5, GEMINI_PRO, BigDL_LLM, ChatGPT
from chat_resume.react_asking_agent import ExperienceType, ReactAskingAgent
from chat_resume.utils import get_default_download_folder, get_prompt, measure_execution_time, read_json, \
    job_doc_name, DocumentType, write_json, write_file, text_to_pdf, store_experience
from chat_resume.vector_store import VectorStore

class ResumeBuilder:
    """
    A class that represents an chat resume for job applications.

    Args:
        api_key (str): The OpenAI API key.
        downloads_dir (str, optional): The directory to save downloaded files. Defaults to the default download folder.

    Attributes:
        api_key (str): The OpenAI API key.
        downloads_dir (str): The directory to save downloaded files.

    Methods:
        get_prompt(system_prompt_path: str) -> str: Returns the system prompt from the specified path.
        resume_to_json(pdf_path: str) -> dict: Extracts resume details from the specified PDF path.
        user_data_extraction(user_data_path: str) -> dict: Extracts user data from the specified path.
        job_details_extraction(url: str) -> dict: Extracts job details from the specified job URL.
        resume_builder(job_details: dict, user_data: dict) -> dict: Generates a resume based on job details and user data.
        cover_letter_generator(job_details: dict, user_data: dict) -> str: Generates a cover letter based on job details and user data.
        resume_cv_pipeline(job_url: str, user_data_path: str) -> None: Runs the Auto Apply Pipeline.
    """

    def __init__(
        self, api_key: str, provider: str, downloads_dir: str = get_default_download_folder()
    ):
        # default is BigDL
        if provider is None or provider.strip() == "":
            self.provider = BIGDL
        else:
            self.provider = provider

        if self.provider == BIGDL:
            self.api_key = None
        else:
            if api_key is None or api_key.strip() == "os":
                if provider == GPT_4 or provider == GPT_3_5:
                    self.api_key = os.environ.get("OPENAI_API_KEY")
                elif provider == GEMINI_PRO:
                    self.api_key = os.environ.get("GEMINI_API_KEY")
            else:
                self.api_key = api_key

        if downloads_dir is None or downloads_dir.strip() == "":
            self.downloads_dir = get_default_download_folder()
        else:
            self.downloads_dir = downloads_dir

    # def load_and_split_documents(self, data, chunk_size=1024, chunk_overlap=100):
    #     try:
    #         # DO: Decide apt chunk size and overlap. start small(128/256) for granular semnatic info to large(512/1024) chunks for broad context.
    #         text_splitter = RecursiveCharacterTextSplitter(
    #             chunk_size=chunk_size,
    #             chunk_overlap=chunk_overlap,
    #             length_function=len
    #         )
    #         chunks = text_splitter.split_text(data)
    #         return chunks
    #     except Exception as e:
    #         print(e)
    #         return None

    # Define a function to perform similarity search between user and job description
    def find_similar_points(self, user_embeddings, job_embeddings):
            try:
                relevant_points = set()
                for embedding in job_embeddings['embedding']:
                    dot_products = np.dot(np.stack(user_embeddings['embedding']), embedding)
                    idx = np.argmax(dot_products)
                    relevant_points.add(user_embeddings.iloc[idx]['chunk'])

                return relevant_points
            except Exception as e:
                print(e)
                return None

    def resume_pdf_to_json(self, pdf_path):
        """
        Converts a resume in PDF format to JSON format.
        Prompts: extract-resume.txt

        Args:
            pdf_path (str): The path to the PDF file.

        Returns:
            dict: The resume data in JSON format.
        """
        if self.provider == BIGDL:
            system_prompt = get_prompt(
                os.path.join(prompt_path, "extract-resume_BigDL.txt")
            )
            section_schema="RESUME_DATA_SCHEMA"
        else:
            system_prompt = get_prompt(
                os.path.join(prompt_path, "extract-resume.txt")
            )
            section_schema = None
        llm = self.get_llm_instance(system_prompt)
        resume_text = extract_plain_text_from_pdf(pdf_path)

        resume_json = llm.get_response(resume_text, need_json_output=True, section_schema=section_schema)
        return resume_json

    def get_llm_instance(self, system_prompt):
        if self.provider == BIGDL:
            return BigDL_LLM(system_prompt=system_prompt)
        elif self.provider == GPT_3_5 or self.provider == GPT_4:
            return ChatGPT(gpt_type= self.provider, api_key=self.api_key, system_prompt=system_prompt)
        # elif self.provider == GEMINI_PRO:
        #     return Gemini(api_key=self.api_key, system_prompt=system_prompt)
        else:
            raise Exception("Invalid LLM Provider")


    @measure_execution_time
    def user_data_extraction(self, user_data_path: str = demo_data_path, is_st=False):
        """
        Extracts user data from the given file path.

        Args:
            user_data_path (str): The path to the user data file.

        Returns:
            dict: The extracted user data in JSON format.
        """
        print("\nFetching user data...")

        if user_data_path is None or (type(user_data_path) is str and user_data_path.strip() == ""):
            user_data_path = demo_data_path

        # Read user data
        if os.path.splitext(user_data_path)[1] == ".pdf":
            user_data = self.resume_pdf_to_json(user_data_path)
        elif os.path.splitext(user_data_path)[1] == ".json":
            user_data = read_json(user_data_path)
        else:
            raise Exception("unknown user data format")
        write_json(os.path.join(self.downloads_dir, "extracted_resume_data.json"), user_data)
        return user_data

    def user_experience_asking(self, user_data: dict, download_resume_path):
        llm_instance = self.get_llm_instance("")
        vstore_faiss=VectorStore(llm_instance.embeddings, download_resume_path)
        for (section, e_type) in [('work_experience', ExperienceType.work), ('projects', ExperienceType.project)]:
            experiences = user_data[section]
            react_agent = ReactAskingAgent(llm_instance.get_llm(), e_type)
            for experience in experiences:
                document = react_agent.get_experience_document(experience, e_type)
                vstore_faiss.store_experience(document,e_type)
        vstore_faiss.construct_and_save_local()
        return vstore_faiss

    def get_exist_vectorstore(self, download_resume_path):
        llm_instance = self.get_llm_instance("")
        vstore_faiss = VectorStore(llm_instance.embeddings, download_resume_path)
        vstore_faiss.load_local()
        return vstore_faiss

    @measure_execution_time
    def job_details_extraction(self, url: str=None, job_site_content: str=None, is_st=False):
        """
        Extracts job details from the specified job URL.
        Prompts: persona-job-llm.txt + extract-job-detail.txt

        Args:
            url (str): The URL of the job posting.
            job_site_content (str): The content of the job posting.

        Returns:
            dict: A dictionary containing the extracted job details.
        """

        print("\nExtracting job details...")

        try:
            system_prompt = get_prompt(
                os.path.join(prompt_path, "persona-job-llm.txt")
            ) + get_prompt(
                os.path.join(prompt_path, "extract-job-detail.txt")
            )

            if url is not None and url.strip() != "":
                job_site_content = get_url_content(url)
                if job_site_content is None:
                    raise Exception("Unable to web scrape the job description.")

            llm = self.get_llm_instance(system_prompt)
            section_schema=None
            if self.provider == BIGDL:
                section_schema="Job_Detail_SCHEMA"
            job_details = llm.get_response(job_site_content, need_json_output=True,section_schema=section_schema)
            if url is not None and url.strip() != "":
                job_details["url"] = url
            job_details["random_number"]= str(random.randint(1, 999999))
            jd_path = job_doc_name(job_details, self.downloads_dir, DocumentType.JobDescription)

            write_json(jd_path, job_details)
            print(f"Job Details JSON generated at: {jd_path}")

            if url is not None and url.strip() != "":
                del job_details['url']

            return job_details, jd_path

        except Exception as e:
            print(e)
            st.write("Please try pasting the job description text instead of the URL.")
            st.error(f"Error in Job Details Parsing, {e}")
            return None, None

    @measure_execution_time
    def cover_letter_generator(self, job_details: dict, user_data: dict, need_pdf: bool = True, is_st=False):
        """
        Generates a cover letter based on the provided job details and user data.
        Prompts: persona-job-llm.txt + generate-cover-letter.txt

        Args:
            job_details (dict): A dictionary containing the job description.
            user_data (dict): A dictionary containing the user's resume or work information.

        Returns:
            str: The generated cover letter.

        Raises:
            None
        """
        print("\nGenerating Cover Letter...")

        try:
            system_prompt = get_prompt(
                os.path.join(prompt_path, "persona-job-llm.txt")
            ) + get_prompt(
                os.path.join(prompt_path, "generate-cover-letter.txt")
            )
            query = f"""Provided Job description delimited by triple backticks(```) and \
                        my resume or work information below delimited by triple dashes(---).
                        ```
                        {json.dumps(job_details)}
                        ```

                        ---
                        {json.dumps(user_data)}
                        ---
                    """

            llm = self.get_llm_instance(system_prompt)
            cover_letter = llm.get_response(query, expecting_longer_output=True)
            cv_path = job_doc_name(job_details, self.downloads_dir, DocumentType.CoverLetter)
            write_file(cv_path, cover_letter)
            print("Cover Letter generated at: ", cv_path)
            if need_pdf:
                text_to_pdf(cover_letter, cv_path.replace(".txt", ".pdf"))
                print("Cover Letter PDF generated at: ", cv_path.replace(".txt", ".pdf"))

            return cover_letter, cv_path.replace(".txt", ".pdf")
        except Exception as e:
            print(e)
            st.write("Error: \n\n",e)
            return None, None


    @measure_execution_time
    def resume_builder(self, job_details: dict, user_data: dict, vstore_faiss: VectorStore, is_st=False):
        """
        Builds a resume based on the provided job details and user data.

        Args:
            job_details (dict): A dictionary containing the job description.
            user_data (dict): A dictionary containing the user's resume or work information.

        Returns:
            dict: The generated resume details.

        Raises:
            FileNotFoundError: If the system prompt files are not found.
        """
        try:
            print("\nGenerating Resume Details...")
            if is_st: st.toast("Generating Resume Details...")

            resume_details_dict = dict()
            system_prompt = get_prompt(os.path.join(prompt_path, "persona-job-llm.txt"))

            # Personal Information Section
            if is_st:
                st.toast("Processing Resume's Personal Info Section...")
            resume_details_dict["personal"] = {
                "name": user_data["name"],
                "phone": user_data["phone"],
                "email": user_data["email"],
                "github": user_data["github"],
                "linkedin": user_data["linkedin"]
            }

            st.markdown("**Personal Info Section**")
            st.write(resume_details_dict)

            string_job_details = json.dumps(job_details)

            experience_list = vstore_faiss.get_top_k_experiences(5, string_job_details)
            if len(experience_list) != 5:
                raise Exception("Wrong number of experiences:"+str(len(experience_list)))

            work_experience_list = []
            project_list = []
            for experience in experience_list:
                type = experience.metadata["type"]
                content = experience.page_content
                if type == ExperienceType.work:
                    work_experience_list.append(content)
                elif type == ExperienceType.project:
                    project_list.append(content)

            user_data['work_experience'] = work_experience_list
            user_data['projects'] = project_list

            # Other Sections
            for section in ['education', 'skill_section', 'work_experience', 'projects', 'certifications', 'achievements']:
                section_log = f"Processing Resume's {section.upper()} Section..."
                if is_st:
                    st.toast(section_log)
                query = get_prompt(os.path.join(prompt_path, "sections", f"{section}.txt"))
                query = (
                    query.replace(
                        "<SECTION_DATA>",
                        json.dumps(user_data[section])
                        ).replace(
                        "<JOB_DESCRIPTION>",
                        string_job_details
                        )
                )

                llm = self.get_llm_instance(system_prompt)
                section_schema=None
                if self.provider == BIGDL:
                    section_schema = section
                response = llm.get_response(query, expecting_longer_output=True, need_json_output=True, section_schema=section_schema)
                if response is not None and isinstance(response, dict):
                    if section in response:
                        if response[section]:
                            if section == "skill_section":
                                resume_details_dict[section] = [i for i in response['skill_section'] if len(i['skills'])]
                            else:
                                resume_details_dict[section] = response[section]

                if is_st:
                    st.markdown(f"**{section.upper()} Section**")
                    st.write(response)

            if is_st:
                st.write("Finish Resume Building")

            resume_details_dict['keywords'] = job_details['keywords']

            resume_json_path = job_doc_name(job_details, self.downloads_dir, DocumentType.Resume)

            write_json(resume_json_path, resume_details_dict)
            resume_pdf_path = resume_json_path.replace(".json", ".pdf")
            # st.write(f"resume_path: {resume_path}")
            if is_st:
                st.write("Converting JSON resume to latex and to pdf")
            resume_pdf_path, resume_tex_path, resume_latex = json_to_latex_to_pdf(resume_details_dict, resume_pdf_path)
            # st.write(f"resume_pdf_path: {resume_pdf_path}")

            return resume_pdf_path, resume_details_dict
        except Exception as e:
            print(e)
            if is_st:
                st.write("Error: \n\n",e)
            return resume_pdf_path, resume_details_dict
