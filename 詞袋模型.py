import pandas as pd
import pickle
import re
from ckip_transformers.nlp import CkipWordSegmenter
from rank_bm25 import BM25Okapi
from joblib import dump, load

# 設定 CSV 檔案路徑和停用詞
csv_path = r"C:\PYTHON\自主學習\csv_json\刑法_去除停用詞.csv"  # 替換成實際 CSV 檔案路徑
with open("C:/PYTHON/自主學習/csv_json/stop_words.txt", "r", encoding="utf-8") as f:
    stop_words = set(f.read().splitlines())

# 初始化 CKIP 分詞器
ws_driver = CkipWordSegmenter(model="bert-base", device=0)

# 讀取 CSV 並處理文本
df = pd.read_csv(csv_path)
df['ArticleContent'] = df['ArticleContent'].astype(str)

# 分詞和過濾停用詞
def split_words(text):
    text = re.sub(r"[,\.\r\n ]", "。", text)  # 標點與空白替換為句點
    ws_result = ws_driver([text])
    filtered_words = [re.sub(r'[^\w\s]', '', word).strip() for word in ws_result[0] if word.strip()]
    return [word for word in filtered_words if word not in stop_words]

# 將所有文本分詞
corpus = [split_words(doc) for doc in df['ArticleContent']]

# 建立 BM25 模型
bm25 = BM25Okapi(corpus)

# 儲存 BM25 模型
dump(bm25, "bm25_model.joblib")
print("BM25 模型已儲存至 bm25_model.joblib")

# 載入 BM25 模型
bm25 = load("bm25_model.joblib")
print("BM25 模型已從 bm25_model.joblib 載入")

# 搜尋 Query
query = "請問刑法第一百條是什麼內容"
top_k = 3

# 分詞處理查詢
query_words = split_words(query)
print("切割後進行bm25:", query_words)

# 若查詢詞不為空，執行 BM25 搜尋
if query_words:
    doc_scores = bm25.get_scores(query_words).tolist()
    top_n_text = bm25.get_top_n(query_words, df['CKIP Text'].tolist(), n=top_k)
    original_top_n = [
        df[df['CKIP Text'] == chunk]['ArticleContent'].values[0]
        for chunk in top_n_text if chunk in df['CKIP Text'].values
    ]
else:
    doc_scores = []
    original_top_n = []

# 輸出結果
print("搜尋結果:", original_top_n)
print("文檔分數:", doc_scores)
