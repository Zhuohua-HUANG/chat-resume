from bigdl.llm.transformers import AutoModelForCausalLM
from huggingface_hub import snapshot_download
from transformers import LlamaTokenizer

# LLM download
snapshot_download(repo_id='meta-llama/Llama-2-7b-chat-hf',
                  local_dir="./checkpoints/Llama-2-7'"
                            "'-chat-hf", token="YOU-NEED-TO-ADD-HF-TOKEN")

# Embeddings
snapshot_download(repo_id='sentence-transformers/all-MiniLM-L12-v2',
                  local_dir="./checkpoints/all-MiniLM-L12-v2")

# optimize llm
llm = AutoModelForCausalLM.from_pretrained("./checkpoints/Llama-2-7b-chat-hf", load_in_4bit=True)
llm.save_low_bit("./checkpoints/llama-2-7b-chat-hf-INT4")

# optimize tokenizer
tokenizer = LlamaTokenizer.from_pretrained("./checkpoints/Llama-2-7b-chat-hf")
tokenizer.save_pretrained("./checkpoints/llama-2-7b-chat-hf-INT4")
