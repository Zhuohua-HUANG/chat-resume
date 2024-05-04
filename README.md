# Resume Tailor

[![](./resources/structure.png)](./resources/structure.png)


## 1. Prerequisites

 - **CPU: Intel i7 or above**
 - OS : Windows, Linux, Mac
 - Python : 3.9.18
 - LLM API key: [OpenAI](https://openai.com/pricing) 

## 2. Setup & Run Code - Use as Project
1. Create and activate python environment to avoid any package dependency conflict.
   ```bash
   conda create -n job-llm python=3.9.18
   conda activate job-llm
   ```

2. Install all required packages.
   - Try pip install
     ```bash
     pip install -r requirements.txt
     ```

3. We also need to install following packages to conversion of latex to pdf
    - For windows
   
        https://blog.csdn.net/qq_44319167/article/details/124648861?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170900683216800226543103%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=170900683216800226543103&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-124648861-null-null.142%5Ev99%5Epc_search_result_base2&utm_term=TeX%20Live&spm=1018.2226.3001.4187
    - For linux
        ```bash
        sudo apt-get install texlive-latex-base texlive-fonts-recommended texlive-fonts-extra
        ```
        NOTE: try `sudo apt-get update` if terminal unable to locate package.
    - For Mac
        ```bash
        brew install basictex
        sudo tlmgr install enumitem fontawesome
        ```
4. download Llama and enable BigDL optimize
   At first, you should update the Hugging Face Token in download_Llama2_and_BigDL_optimize.py.
   Then, please run these bash command:
    ```bash
    cd ./resume_tailor 
    python download_Llama2_and_BigDL_optimize.py
    cd ../
    ```
   
6. Run app
    ```bash
    streamlit run web_app.py
    ```
