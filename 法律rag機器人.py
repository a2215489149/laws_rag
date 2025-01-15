import os
import re
import pandas as pd 
import random
import logging
import json
import copy
import numpy as np
from typing import Sequence, List
from ckip_transformers.nlp import CkipWordSegmenter
from rank_bm25 import BM25Okapi
# dbms.security.procedures.unrestricted=jwt.security.*,gds.*,apoc.*
# #dbms.security.procedures.allowlist=apoc.coll.*,apoc.load.*,gds.*
# Flask imports
from flask import Flask, request, abort
from joblib import dump, load
# LineBot imports
from linebot.v3 import WebhookHandler
from linebot.models import MessageEvent
from waitress import serve
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage as LineTextMessage
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.exceptions import InvalidSignatureError

# LangChain imports
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder

# Neo4j imports
from neo4j import GraphDatabase
from graphdatascience import GraphDataScience
import pickle
from sklearn.metrics.pairwise import cosine_similarity
# Environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["NEO4J_API_KEY"] = ""
os.environ["NEO4J_URL"] = ""
os.environ["DB_PATH"] = ""
os.environ["LINEBOT_ACCESS_TOKEN"] = ""
os.environ["WebhookHandler"] = ""

# schoolapi
os.environ['AzureOpenAIEmbeddings_KEY'] = ""
os.environ["AzureOpenAIEmbeddings_ENDPOINT"] = ""
os.environ['AzureChatOpenAI_KEY'] = ""
os.environ["AzureChatOpenAI_ENDPOINT"] = ""

# Configuration
PERSIST_HISTORY = True  # Set to False for in-memory storage
STORAGE_PATH = "./chat_histories"
NEO4J_AUTH = ("neo4j", os.environ['NEO4J_API_KEY'])

gds = GraphDataScience(os.environ['NEO4J_URL'], auth=NEO4J_AUTH)
configuration = Configuration(access_token=os.environ["LINEBOT_ACCESS_TOKEN"])
handler = WebhookHandler(os.environ["WebhookHandler"])
txt_file=r"csv_json\stop_words.txt"

#引入bm25向量和原資料庫
csv_path=r"C:\PYTHON\自主學習\csv_json\刑法_去除停用詞.csv"
bm25_model=r"C:\PYTHON\自主學習\向量庫\bm25_model.joblib"
df=pd.read_csv(csv_path)

# 引入QA集向量
QA_embedding_file = r"C:\PYTHON\自主學習\向量庫\qa_embeddings.pkl"


# 日誌紀錄
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)
txt_file = r"csv_json\stop_words.txt"
with open(txt_file, "r", encoding="utf-8") as f:
    stop_words = set(f.read().splitlines())
# Global variables
store = {}  # Memory store
greetings_responses = [
    "你好！我是 Alice 黃律師，你的專屬法律顧問。",
    "你好，我是 Alice 黃律師。",
    "哈囉！我是你的專屬法律顧問，Alice 黃律師。",
    "您好！我是 Alice 黃律師。",
    "嗨！歡迎您，我是 Alice 黃律師。"
]

# LangChain prompt setup
prompt = ChatPromptTemplate.from_messages([
    ("system", """
        你是一名專業的法律顧問，回答問題時應清楚、準確，並直接提供具體資訊。
        需在不破壞查詢到法律文本的語意表達的前提下回答問題。
        請用親切的方式回答問題，回答應具體且準確，所有內容以繁體中文表達，字數限一百字以內。回復中禁止包含「根據我查到的資料」或類似敘述，請直接引用法律條文或案例，僅提供答案。
        以下是參考資料：
        1.BM25比對後相近文章:{bm25_context}
        2.知識圖譜向量比對後相近文章:{neo4j_context}
        3.可能相似的題型:{QA_context}"""
    ),
    MessagesPlaceholder(variable_name="messages")
])

query_transform_prompt = ChatPromptTemplate.from_messages([
    ("""system", "根據以上歷史對話內容，將本次省略的問句內容補充完整，使其成為一個清楚的問句。
    僅生成補充完整的問句，不要破壞語意，不要回答該問句，必須保留問題的重點，比如問題的起因,重點與希望得到的答案要與甚麼相關聯。
    當有歷史對話等攏長字句出現才刪除過多的內文，否則盡量保留原語意，並以繁體中文作答。"""),
    MessagesPlaceholder(variable_name="messages")
])
prompt_keywords= ChatPromptTemplate.from_messages([
    ("system", """
        幫我整理出以下關於刑法問題的關鍵詞，只要其中最重要的三至七個。
        每個關鍵詞必須以空格分開，必須從裡面抓取，最好適用BM25。
        以下是客戶提問:{transformed_query}""")
])
# Azure OpenAI Setup
embeddings = AzureOpenAIEmbeddings(
    api_key=os.environ['AzureOpenAIEmbeddings_KEY'],
    api_version="2023-05-15",
    model="text-embedding-ada-002",
    azure_endpoint=os.environ["AzureOpenAIEmbeddings_ENDPOINT"]
)

llm = AzureChatOpenAI(
    deployment_name="gpt-4",
    azure_endpoint=os.environ["AzureChatOpenAI_ENDPOINT"],
    api_key=os.environ['AzureChatOpenAI_KEY'],
    api_version="2024-08-01-preview"
)

# Chain Setup
prompt_chain = query_transform_prompt | llm
chat_chain = prompt | llm
keywords_chain =prompt_keywords | llm

class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, storage_path: str, session_id: str,record : bool):
        self.storage_path = storage_path
        self.session_id = session_id
        self.record=record
        self._ensure_storage()
    
    def _ensure_storage(self):
        """Ensure the storage file exists and is valid."""
        path = os.path.join(self.storage_path, f"{self.session_id}.json")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump([], f)  # Initialize with an empty list
        else:
            # Ensure the file contains valid JSON; reset if invalid
            try:
                with open(path, "r", encoding="utf-8") as f:
                    json.load(f)
            except json.JSONDecodeError:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump([], f)

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve all messages from the history."""
        path = os.path.join(self.storage_path, f"{self.session_id}.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [
                HumanMessage(content=entry["content"]) if entry["type"] == "human" else AIMessage(content=entry["content"])
                for entry in data
            ]
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add multiple messages to the history."""
        path = os.path.join(self.storage_path, f"{self.session_id}.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing_messages = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            existing_messages = []
        if self.record:
            new_messages = [
                {"type": "human" if isinstance(msg, HumanMessage) else "ai", "content": msg.content}
                for msg in messages
            ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing_messages + new_messages, f, ensure_ascii=False, indent=4)

    def clear(self) -> None:
        """Clear all messages from the history."""
        path = os.path.join(self.storage_path, f"{self.session_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump([], f)


def main():
    
    
    def is_greeting(message: str) -> bool:
        """Check if the message is a greeting."""
        pattern = r'\b(你好|嗨|hello|hi|HI|HELLO|早安|晚安|午安|妳好|機器人妳好|機器人你好|喂|哈囉|早|晚上好|下午好)\b'
        return re.search(pattern, message, re.IGNORECASE) is not None

    def dont_session_history(session_id: str,record:bool) -> BaseChatMessageHistory:
        """Retrieve or initialize session history."""
        if session_id not in store:
            if PERSIST_HISTORY:
                store[session_id] = FileChatMessageHistory(storage_path=STORAGE_PATH, session_id=session_id,record=record)
            else:
                store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]
    def get_session_history(session_id: str,record:bool) -> BaseChatMessageHistory:
        """Retrieve or initialize session history."""
        if session_id not in store:
            if PERSIST_HISTORY:
                store[session_id] = FileChatMessageHistory(storage_path=STORAGE_PATH, session_id=session_id,record=record)
            else:
                store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    if not os.path.exists(STORAGE_PATH):
        os.makedirs(STORAGE_PATH)

    app = Flask(__name__)
    @app.route("/callback", methods=['POST'])
    def callback():
        """Handle incoming webhook requests."""
        signature = request.headers['X-Line-Signature']
        body = request.get_data(as_text=True)
        try:
            handler.handle(body, signature)
        except InvalidSignatureError:
            app.logger.info("Invalid signature. Please check your channel access token/channel secret.")
            abort(400)
        return 'OK'
    
    def load_bm25_model(filename):
        bm25 = load(filename)  
        print(f"BM25 模型已從 {filename} 載入")
        return bm25
    bm25 = load_bm25_model(bm25_model)

    # 文字分詞及過濾停用詞
    def split_words(text: str, stop_words):
        # 這裡使用 CkipWordSegmenter 來進行中文分詞
        ws_driver = CkipWordSegmenter(model="bert-base")
        # 替換掉不必要的標點和換行符，統一為句點
        text = re.sub(r"[,\.\r\n ]", "。", text)
        ws_result = ws_driver([text])
        
        # 去除符號並過濾停用詞
        filtered_words = [re.sub(r'[^\w\s]', '', word).strip() for word in ws_result[0] if word.strip()]
        return [word for word in filtered_words if word not in stop_words]

    # BM25 搜尋，查詢文檔
    def BM25_search(bm25, query, top_k, df):
        # 分詞後的查詢詞
        query_words = split_words(query, stop_words)
        print("切割後進行bm25:", query_words)
        
        if not query_words:
            return [], []
        
        # 計算分數
        doc_scores = bm25.get_scores(query_words).tolist()
        
        # 取得 top_k 文本
        top_n_text = bm25.get_top_n(query_words, df['CKIP Text'].tolist(), n=top_k)
        
        # 返回相關文本
        original_top_n = [
            df[df['CKIP Text'] == chunk]['ArticleContent'].values[0]
            for chunk in top_n_text if chunk in df['CKIP Text'].values
        ]
        
        return original_top_n, doc_scores
    #引入QA向量資料
    def QA_load(QA_embedding_file):
        with open(QA_embedding_file, "rb") as f:
            questions, answers, question_embeddings = pickle.load(f)
            print("成功引入QA向量...")
        return questions, answers, question_embeddings
    #計算QA相似度
    def QA_search(transformed_query_COPY,questions, answers, question_embeddings,threshold=0.8):
        
        QA_context=""
        query_embedding = np.array(embeddings.embed_query(transformed_query_COPY))
        cosine_scores = cosine_similarity([query_embedding], question_embeddings)
        top_k = 3
        top_indices = np.argsort(cosine_scores[0])[::-1][:top_k]
        top_qa_pairs = []
        for idx in top_indices:
            score = cosine_scores[0][idx]
            if score >= threshold:  # 檢查相似度是否超過閥值
                top_qa_pairs.append((questions[idx], answers[idx]))
            else:
                break
        for i,(question, answer) in enumerate(top_qa_pairs):
            QA_context+=f"\n第 {i+1}種類題:\n問題: {question}\n答案: {answer}\n"
        print("QA:",QA_context)
        return QA_context
    @handler.add(MessageEvent, message=TextMessageContent)
    def handle_message(event):
        """Handle incoming messages and generate responses."""
        user_id = event.source.user_id
        session_id = str(user_id)  # Using user_id directly as session_id
        
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            received_message = event.message.text

            print("收到訊息:", received_message)

            # Check for greeting message
            if is_greeting(received_message):
                reply_msg = random.choice(greetings_responses)
            else:
                try:
                    # 搜索歷史對話紀錄
                    
                    session_history = get_session_history(session_id=session_id,record=True)
                    no_session_history = dont_session_history(session_id=session_id,record=False)
                    messages = session_history.messages
                except Exception as e:
                    logging.error(f"Error fetching session history: {e}")
                    print("歷史對話紀錄error")
                    reply_msg = "抱歉，我無法回答您的問題。請稍後再試。"
                    
                try:
                    # 結合歷史對話給gpt生成清晰的問題

                    query_transforming = RunnableWithMessageHistory(
                        prompt_chain,
                        lambda session_id: no_session_history,  # 傳遞已經初始化的 session_history
                        input_messages_key="messages"
                    )

                    # 配置
                    config = {"configurable": {"session_id": session_id}}

                    # 呼叫並獲取結果
                    transformed_query = query_transforming.invoke(
                        {"messages": [HumanMessage(content=received_message)] + messages}, config=config
                    ).content
                except Exception as e:
                    logging.error(f"Error transforming query: {e}")
                    print("結合歷史對話給gpt生成清晰的問題error")
                    reply_msg = "抱歉，我無法回答您的問題。請稍後再試。"
                    
                print("結合歷史對話",transformed_query)  
                #複製結合歷史後的問句
                transformed_query_COPY = copy.deepcopy(transformed_query)
                #取得關鍵字
                response = keywords_chain.invoke({"transformed_query": transformed_query_COPY}).content
                keywords = response.split()
                print("問題關鍵詞:", keywords) 

                #計算QA集COSIN相似度
                questions, answers, question_embeddings=QA_load(QA_embedding_file)
                QA_context=QA_search(transformed_query_COPY,questions, answers, question_embeddings)
                
                #bm25相似度比對
                original_top_n, doc_scores = BM25_search(bm25,response, 3, df)
                bm25_context = ' '.join(original_top_n) if original_top_n else '沒檢索到相關文本'
                print("bm25查詢:",bm25_context)
                
                try:
                    embedding_vector = embeddings.embed_query(transformed_query_COPY)
                except Exception as e:
                    logging.error(f"Error generating embedding: {e}")
                    embedding_vector = None
                    reply_msg = "抱歉，我無法回答您的問題。請稍後再試。"
                # Neo4j query and message response handling
                try:
                    if embedding_vector is not None:
                        query = """
                            MATCH (article:Article)-[:belong]-(ckip:CKIP)
                            WITH article, ckip, gds.similarity.cosine($vector, article.vector) AS similarity
                            WHERE similarity >= $threshold AND 
                                ANY(keyword IN $keywords WHERE ckip.text CONTAINS keyword OR article.content CONTAINS keyword)
                            RETURN article.number AS ArticleNumber, article.content AS ArticleContent, similarity
                            ORDER BY similarity DESC
                            LIMIT 3
                        """
                        THRESHOLD = 0.7
                        params = {"vector": embedding_vector, "threshold": THRESHOLD, "keywords": keywords}
                        results = gds.run_cypher(query, params)
                        
                        
                        # 檢查 Neo4j 查詢結果是否為空
                        if results.empty:
                            neo4j_context = '沒檢索到相關文本'
                        else:
                            # 提取結果並構建neo4j_context
                            neo4j_context = ' '.join([row['ArticleContent'] for _, row in results.iterrows()])
                            print("知識圖譜:",neo4j_context)

                    else:
                        neo4j_context = '沒檢索到相關文本'
                except Exception as e:
                    logging.error(f"Error querying Neo4j: {e}")
                    neo4j_context = '沒檢索到相關文本'

                # Final response generation
                try:
                    with_message_history = RunnableWithMessageHistory(
                        chat_chain,
                        lambda session_id: session_history,
                        input_messages_key="messages"
                    )
                    config = {"configurable": {"session_id": session_id}}
                    reply_msg = with_message_history.invoke(
                        {"messages": [HumanMessage(content=received_message)], 'bm25_context': bm25_context, 'neo4j_context': neo4j_context,"QA_context":QA_context},
                        config=config
                    ).content
                except Exception as e:
                    logging.error(f"Error generating final reply: {e}")
                    reply_msg = "抱歉，我無法回答您的問題。請稍後再試。"

        try:
            line_bot_api.reply_message_with_http_info(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[LineTextMessage(text=reply_msg)]
                )
            )
        except Exception as e:
            logging.error(f"Error replying to message: {e}")
    app.run(host="0.0.0.0", port=8080, debug=True)

if __name__ == "__main__":
    main()