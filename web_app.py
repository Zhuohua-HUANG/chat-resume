import multiprocessing
import os
import json
import shutil
import time

import streamlit as st

from chat_resume.resume_builder import ResumeBuilder
from chat_resume.utils import display_pdf, download_pdf, read_file, read_json
from chat_resume.metrics import jaccard_similarity, overlap_coefficient, cosine_similarity
from chat_resume.llm_models import BIGDL, GPT_3_5, GPT_4, GEMINI_PRO

st.set_page_config(
    page_title="Chat Resume",
    page_icon="📑",
    menu_items={
        # 'Get help': 'https://www.youtube.com',
        'About': 'https://github.com/Zhuohua-HUANG/chat-resume?tab=readme-ov-file',
        'Report a bug': "https://github.com/Zhuohua-HUANG/chat-resume/issues",
    }
)

try:
    st.header(":green[Chat] :orange[Resume]", divider='rainbow')
    st.subheader("Your Personalized Resume Assistant")
    col_text, col_url,_,_ = st.columns(4)
    with col_text:
        st.write("Job Description Text")
    with col_url:
        # is_url_button = st.toggle('Job URL', False)
        is_url_button = False

    url, text = "", ""
    if is_url_button:
        url = st.text_input("Enter job posting URL:", placeholder="Enter job posting URL here...", label_visibility="collapsed")
    else:
        text = st.text_area("Paste job description text:", max_chars=5500, height=200, placeholder="Paste job description text here...", label_visibility="collapsed")

    is_use_existing_userdata_button = st.toggle('Use Existing Data', False)
    if not is_use_existing_userdata_button:
        file = st.file_uploader("Upload your resume or any work-related data(PDF, JSON).", type=["json", "pdf"])

    col_1, col_2 = st.columns(2)
    with col_1:
        provider = st.selectbox("Select LLM provider([OpenAI](https://openai.com/blog/openai-api), [BigDL](https://github.com/intel-analytics/BigDL)):", [BIGDL, GPT_4, GPT_3_5])
    api_key = ""
    if provider in [GPT_4,GPT_3_5, GEMINI_PRO]:
        with col_2:
            api_key = st.text_input("Enter API key:", type="password")
    st.markdown("<sub><sup>💡 GPT-4 is recommended for better results.</sup></sub>", unsafe_allow_html=True)

    # Buttons side-by-side with styling
    col1, col2, col3 = st.columns(3)
    with col1:
        get_resume_button = st.button("Get Resume", key="get_resume", type="primary", use_container_width=True)

    with col2:
        get_cover_letter_button = st.button("Get Cover Letter", key="get_cover_letter", type="primary", use_container_width=True)

    with col3:
        get_both = st.button("Resume + Cover letter", key="both", type="primary", use_container_width=True)
        if get_both:
            get_resume_button = True
            get_cover_letter_button = True

    if get_resume_button or get_cover_letter_button:
        # check valid input
        if not is_use_existing_userdata_button and file is None:
            st.toast(":red[Upload user's resume or work related data to get started]", icon="⚠️")
            st.stop()
        
        if url == "" and text == "":
            st.toast(":red[Please enter a job posting URL or paste the job description to get started]", icon="⚠️") 
            st.stop()
        
        if provider in [GPT_4,GPT_3_5, GEMINI_PRO] and api_key == "":
            st.toast(":red[Please enter the API key to get started]", icon="⚠️")
            st.stop()
        # run the workflow if inputs are valid
        if (is_use_existing_userdata_button or file is not None) and (url != "" or text != ""):
            download_resume_path = os.path.join(os.path.dirname(__file__), "output")

            # st.write(f"download_resume_path: {download_resume_path}")

            resume_llm = ResumeBuilder(api_key=api_key, provider=provider, downloads_dir=download_resume_path)

            vstore = None
            if not is_use_existing_userdata_button:
                # Save the uploaded file
                os.makedirs("uploads", exist_ok=True)
                file_path = os.path.abspath(os.path.join("uploads", file.name))
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
        
                # Extract user data
                with st.status("Extracting user data..."):
                    user_data = resume_llm.user_data_extraction(file_path, is_st=True)
                    st.write(user_data)

                shutil.rmtree(os.path.dirname(file_path))

                with st.status("Chatting about your experience..."):
                    vstore = resume_llm.user_experience_asking(user_data, download_resume_path)
            else:
                user_data = read_json(os.path.join(download_resume_path, "extracted_resume_data.json"))
                vstore = resume_llm.get_exist_vectorstore(download_resume_path)



            if user_data is None or vstore is None:
                st.error("User data can not be processed. "
                         "Please select \"Use existing user data\" button and upload a valid file")
                st.markdown("<h3 style='text-align: center;'>Please try again</h3>", unsafe_allow_html=True)
                st.stop()

            # Extract job details
            with st.status("Extracting job details..."):
                if url != "":
                    job_details, jd_path = resume_llm.job_details_extraction(url=url, is_st=True)
                elif text != "":
                    job_details, jd_path = resume_llm.job_details_extraction(job_site_content=text, is_st=True)
                st.write(job_details)

            if job_details is None:
                st.error("Please paste job description. Job details not able process.")
                st.markdown("<h3 style='text-align: center;'>Please paste job description text and try again!</h3>", unsafe_allow_html=True)
                st.stop()

            # Build Resume
            if get_resume_button:
                with st.status("Building resume..."):
                    resume_path, resume_details = resume_llm.resume_builder(job_details, user_data, vstore, is_st=True)
                    # st.write("Outer resume_path: ", resume_path)
                    # st.write("Outer resume_details is None: ", resume_details is None)
                resume_col_1, resume_col_2 = st.columns([0.7, 0.3])
                with resume_col_1:
                    st.subheader("Generated Resume")
                with resume_col_2:
                    pdf_data = read_file(resume_path, "rb")

                    st.download_button(label="Download Resume ⬇",
                                        data=pdf_data,
                                        file_name=os.path.basename(resume_path),
                                        # on_click=download_pdf(resume_path),
                                        key="download_pdf_button",
                                        mime="application/pdf",
                                        use_container_width=True)

                display_pdf(resume_path, type="image")
                st.toast("Resume generated successfully!", icon="✅")

                # Calculate metrics
                st.subheader("Resume Metrics")
                for metric in ['overlap_coefficient', 'cosine_similarity']:
                    user_personlization = globals()[metric](json.dumps(resume_details), json.dumps(user_data))
                    job_alignment = globals()[metric](json.dumps(resume_details), json.dumps(job_details))
                    job_match = globals()[metric](json.dumps(user_data), json.dumps(job_details))
                    job_improvement = job_alignment-job_match

                    if metric == "overlap_coefficient":
                        title = "Overlap Coefficient"
                        help_text = "The overlap coefficient is a measure of the overlap between two sets, and is defined as the size of the intersection divided by the smaller of the size of the two sets."
                    elif metric == "cosine_similarity":
                        title = "Cosine Similarity"
                        help_text = "The cosine similarity is a measure of the similarity between two non-zero vectors of an inner product space that measures the cosine of the angle between them."

                    st.caption(f"## **:rainbow[{title}]**", help=help_text)
                    col_m_1, col_m_2, col_m_3 = st.columns(3)
                    col_m_1.metric(label=":orange[ImprovedResume vs JobDetail]", value=f"{job_alignment:.3f}",
                                   delta=f"{job_improvement:.6f}", delta_color="normal")
                    col_m_2.metric(label=":blue[OriginalResume vs JobDetail]", value=f"{job_match:.3f}")
                    col_m_3.metric(label=":green[ImprovedResume vs OriginalResume]", value=f"{user_personlization:.3f}")
                st.markdown("---")

            # Build Cover Letter
            if get_cover_letter_button:
                with st.status("Building cover letter..."):
                    cv_details, cv_path = resume_llm.cover_letter_generator(job_details, user_data, is_st=True)
                cv_col_1, cv_col_2 = st.columns([0.7, 0.3])
                with cv_col_1:
                    st.subheader("Generated Cover Letter")
                with cv_col_2:
                    cv_data = read_file(cv_path, "rb")
                    st.download_button(label="Download CV ⬇",
                                    data=cv_data,
                                    file_name=os.path.basename(cv_path),
                                    # on_click=download_pdf(cv_path),
                                    key="download_cv_button",
                                    mime="application/pdf",
                                    use_container_width=True)
                st.markdown(cv_details, unsafe_allow_html=True)
                st.markdown("---")
                st.toast("cover letter generated successfully!", icon="✅")

            st.toast(f"Done", icon="👍🏻")
            st.success(f"Done", icon="👍🏻")
            st.balloons()

            refresh = st.button("Refresh")

            if refresh:
                st.caching.clear_cache()
                st.rerun()
        
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.markdown("<h3 style='text-align: center;'>Please try again!</h3>", unsafe_allow_html=True)
    st.stop()
