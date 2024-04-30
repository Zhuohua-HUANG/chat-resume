from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document
from resume_tailor.react_asking_agent import ExperienceType
class VectorStore:
    def __init__(self, embeddings, download_resume_path):
        self.vectorstore = None
        self.docs=[]
        self.embeddings=embeddings
        self.vs_folder_name = download_resume_path + "/vectorstore"
        self.vs_index_name = "user_experience"

    def store_experience(self, experience: str,experience_type: ExperienceType):
        print("Storing experience: ", experience)
        document = Document(experience, metadata={"type": experience_type})
        self.docs.append(document)

    def construct_and_save_local(self):
        self.vectorstore = FAISS.from_documents(self.docs, self.embeddings)
        self.vectorstore.save_local(folder_path=self.vs_folder_name, index_name=self.vs_index_name)

    def load_local(self):
        self.vectorstore = FAISS.load_local(folder_path=self.vs_folder_name, index_name=self.vs_index_name, embeddings=self.embeddings,
                                       allow_dangerous_deserialization=True)

    def get_top_k_experiences(self, k, query) -> Document:
        if self.vectorstore is None:
            raise Exception("VectorStore did not be loaded")
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        retrieved_docs = self.retriever.invoke(query)
        return retrieved_docs