import os

from resume_tailor.latex_ops import json_to_latex_to_pdf
from resume_tailor.utils import (
    get_default_download_folder,
    key_value_chunking,
    measure_execution_time,
    read_json,
    write_file,
    write_json,
    job_doc_name,
    text_to_pdf,
    get_prompt,
    DocumentType
)
from resume_tailor.metrics import jaccard_similarity, overlap_coefficient, cosine_similarity, vector_embedding_similarity


module_dir = os.path.dirname(__file__)
demo_data_path = os.path.join(module_dir, "demo_data", "user_profile.json")
prompt_path = os.path.join(module_dir, "prompts")


