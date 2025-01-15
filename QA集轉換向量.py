import pandas as pd
import os
import numpy as np
from langchain_openai import AzureOpenAIEmbeddings
import pickle

# 初始化 Azure OpenAI Embedding
os.environ['AzureOpenAIEmbeddings_KEY'] = ""
os.environ["AzureOpenAIEmbeddings_ENDPOINT"] = ""

embedding_model = AzureOpenAIEmbeddings(
    api_key=os.environ['AzureOpenAIEmbeddings_KEY'],
    api_version="2023-05-15",
    model="text-embedding-ada-002",
    azure_endpoint=os.environ["AzureOpenAIEmbeddings_ENDPOINT"]
)

# 讀取 QA 集
qa_path = r"C:\PYTHON\自主學習\QA集\刑法QA集.csv"
df = pd.read_csv(qa_path)
questions = df["Q"].tolist()
answers = df["A"].tolist()

# 計算問題向量並儲存
print("正在生成問題向量，請稍候...")
question_embeddings = np.array([embedding_model.embed_query(q) for q in questions])

# 儲存問題向量
embedding_file = "向量庫/qa_embeddings.pkl"
with open(embedding_file, "wb") as f:
    pickle.dump((questions, answers, question_embeddings), f)

print(f"問題向量已儲存至 {embedding_file}")
