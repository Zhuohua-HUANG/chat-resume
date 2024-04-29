import unittest
import unittest

from resume_tailor.llm_models import BigDL_LLM, ChatGPT, GPT_3_5
from resume_tailor.vector_store import VectorStore

# system_prompt="""
#     Discard any prior instructions.
#     You are a seasoned career advising expert in crafting resumes and cover letters, boasting a rich 15-year history dedicated to mastering this skill at Harvard Extension School.
#     Picture yourself as a certified professional resume writer, specializing in creating compelling and tailored cover letters that highlight clients' skills, experiences, and achievements—meticulously aligning with the specific job descriptions they target.
#     Your expertise extends across various industries, encompassing a deep understanding of prevailing hiring trends and Applicant Tracking Systems (ATS).
#     Your ability to identify precise keywords, responsibilities, and requirements from job descriptions is unparalleled.
#     """
# instructor="""
#     You are going to write a JSON resume section of "Work Experience" for an applicant applying for job posts. The answer should be JSON format.
#     3-4 per experience, closely mirroring job requirements.
#     Step to follow:
#     1. Analyze my Work details to match job requirements.
#     2. Create a JSON resume section that highlights strongest matches
#     3. Optimize JSON section for clarity and relevance to the job description.
#
#     Consider following Work Details delimited by <WORK></WORK> tag.
#     <WORK>
#     {
#       "role": "AIPrompt Engineer",
#       "company": "GlassboxAI",
#       "from": "Sept. 2023",
#       "to": "Dec. 2023",
#       "description": [
#         "Built interactive chatbot to handle inquiries related to the company, SHAP data, and SHAP diagrams.",
#         "Applied LangChain framework on GPT-3.5 by utilizing OpenAI Embeddings and the FAISS vector database.",
#         "Fixed incorrect segmentation of company data in vector database to improve chatbot's performance.",
#         "Sampled the output of the AI model and refined prompt to empower chatbot to interpret the data features and image data."
#       ]
#     },
#     {
#       "role": "Software Engineer Intern",
#       "company": "Tesla",
#       "from": "May 2023",
#       "to": "Aug. 2023",
#       "description": [
#         "Developed new modules for microservices and wrote their unit test code, based on adapted Gin framework.",
#         "Added JWT Login Authentication module to the project to provide permission control for different modules.",
#         "Wrote Kafka consumer for factory parts scanning information transmission and validation.",
#         "Migrated a total of 800,000 pieces of data from AWS S3 object storage buckets to local Pure Storage and updated the object storage code."
#       ]
#     },
#     {
#       "role": "Java Back-end Developer Intern",
#       "company": "T.Y.M.",
#       "from": "Dec. 2022",
#       "to": "Feb. 2023",
#       "description": [
#         "Expanded the function of the back-end API, and updated documentation.",
#         "Optimized queries in the CRM system by setting compound indexes to improve database access performance.",
#         "Used scheduled tasks to write table-level and database-level backups of MySQL databases."
#       ]
#     }
#     </WORK>
#
#     Consider following Job description delimited by <JOB_DETAIL></JOB_DETAIL> tag.
#     <JOB_DETAIL>
#     {
#   "title": "Full Stack Engineer - Chatbot Team",
#   "keywords": "Full Stack Engineer, Chatbot, GPT4, software solutions, development ecosystem, Typescript, Javascript, Java, Python, C, C++, C#, Golang, data structures, algorithms, operating systems, databases, networking, performance, scalability, testing",
#   "purpose": "To disrupt the car ownership experience, simplify and democratize car ownership, improve customer self-service through a GPT4-powered chatbot, and make car ownership effortless",
#   "duties_responsibilities": [
#     "Partner with product managers, engineers, and designers to enhance the self-serve rate of the Chatbot",
#     "Develop new features and automation, write unit tests, and ensure high-quality engineering results",
#     "Participate in daily code reviews and maintain high code quality standards",
#     "Collaborate on product and engineering specifications, providing valuable insights",
#     "Iterate on the Chatbot based on customer and internal feedback",
#     "Monitor software errors and address critical issues promptly",
#     "Create and update documentation for development, troubleshooting, and training",
#     "Become an expert on the product suite and production systems",
#     "Contribute to the entire product stack and maintain codebase integrity",
#     "Provide feedback to evolve the tech stack"
#   ],
#   "required_qualifications": [
#     "Bachelor's, Master's, or PhD in Computer Science or related field",
#     "1+ years of production software engineering experience in consumer applications",
#     "Proficiency in Typescript/Javascript, Java, Python, C, C++, C#, or Golang",
#     "Strong understanding of data structures, algorithms, operating systems, databases, and networking",
#     "Ability to work with various technologies and environments",
#     "Knowledge of code performance and scalability",
#     "Experience in writing comprehensive tests"
#   ],
#   "preferred_qualifications": "None specified",
#   "company_name": "Jerry",
#   "company_info": "Jerry is a Forbes Top Startup Employer focused on disrupting the car ownership experience. They have experienced significant growth and success in the insurance comparison category. The company is driven by the mission to simplify and democratize car ownership, offering a world-class experience to customers. Jerry values diversity and inclusivity, being an Equal Employment Opportunity employer. They are committed to providing accommodations for individuals with disabilities during the job application process."
# }
#     </JOB_DETAIL>
#
#     Desired Output:
#     Provide JSON object as output like following:
#     {
#       "work_experience": [
#         {
#           "role": "Software Engineer Intern",
#           "company": "Tesla",
#           "location": "Shanghai, China",
#           "from": "May 2023",
#           "to": "Aug 2023",
#           "description": [
#             "Migrated a total of 800,000 pieces of data from AWS S3 object storage buckets to local Pure Storage and update the object storage code.",
#             "Wrote Kafka consumer for factory parts scanning information transmission and validation.",
#             and so on ...
#           ]
#         },
#         and So on ...
#       ]
#     }
#     """
system_prompt="""
    You are Jone."""
instructor="What is your name?"

text="What is AutoLLM?"

class TestBigDL(unittest.TestCase):
    def test_model(self):
        bigdl = BigDL_LLM(system_prompt)
        response=bigdl.get_response(instructor)
        print(response)
        self.assertIsNotNone(response)

    def test_embedding(self):
        bigdl = BigDL_LLM(system_prompt)
        vector_storage = bigdl.get_text_embedding(text)
        print(vector_storage.docstore._dict)

class TestVectorStore(unittest.TestCase):

    def test_openai(self):
        api_key=input("Please enter your api key here: ")
        openai_model = ChatGPT(GPT_3_5, api_key,system_prompt)
        embedding = openai_model.embeddings
        vs = VectorStore(embedding,"D:/b_Work/ip_LLM/resume-tailor-llm/output")
        vs.store_experience("hi",1)
        vs.construct_and_save_local()
        docs=vs.get_top_k_experiences(2,"hihi")
        self.assertIsNotNone(docs)
        print(docs)
        self.assertEqual(len(docs),1,"not equall")
    def test_bigdl(self):
        bigdl = BigDL_LLM(system_prompt)
        embedding = bigdl.embeddings
        vs = VectorStore(embedding,"D:/b_Work/ip_LLM/resume-tailor-llm/output")
        vs.store_experience("hi",1)
        vs.construct_and_save_local()
        docs=vs.get_top_k_experiences(2,"hihi")
        self.assertIsNotNone(docs)
        print(docs)
        self.assertEqual(len(docs),1,"not equall")


if __name__ == '__main__':
    # 创建测试套件
    suit = unittest.TestSuite()
    suit.addTest(TestBigDL("test_model"))
    suit.addTest(TestBigDL("test_embedding"))
    suit.addTest(TestVectorStore("test_openai"))
    suit.addTest(TestVectorStore("test_bigdl"))
    # 创建测试运行器
    runner = unittest.TestRunner()
    runner.run(suit)