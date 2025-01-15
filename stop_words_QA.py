import pandas as pd
import re
from ckip_transformers.nlp import CkipWordSegmenter

# 初始化 CKIP 分詞器（只需初始化一次）
ws_driver = CkipWordSegmenter(model="bert-base",device=0)

# 文件路徑
input_file = r"C:\PYTHON\自主學習\QA集\刑法QA集.csv"  # 原始文件
txt_file = r"csv_json\stop_words.txt"  # 停用詞文件
output_file = r"C:\PYTHON\自主學習\QA集\刑法QA集.csv"  # 處理後的文件

# 讀取 CSV 和停用詞
df = pd.read_csv(input_file)
with open(txt_file, "r", encoding="utf-8") as f:
    stop_words = set(f.read().splitlines())  # 讀取停用詞，並將其儲存為集合

target_column = ["Q","A"]
processed_texts = []
i=0
    # 先移除換行符號並統一標點
for columns in target_column:
    for text in df[columns]:
        # 替換不必要的符號
        text = text.replace("\n", "。").replace("\r", "。").replace(",", "。").replace(".", "。").replace(" ", "。")

        # CKIP 分詞
        ws_result = ws_driver([text])

        # 移除標點符號並清理，確保每個詞是有效的
        filtered_words = [word for word in ws_result[0] if word.strip() and re.sub(r'[^\w\s]', '', word).strip() not in stop_words]

        # 用 " || " 連接每個詞，並移除多餘空格
        processed_text = ' || '.join(filtered_words).strip()

        # 加入處理後的文本
        processed_texts.append(processed_text)
    i=i+1
    # 新增處理後的列
    if i == 1:
        df["CKIP Q"] = processed_texts
    elif i == 2:
        df["CKIP A"] = processed_texts
    processed_texts=[]
# 保存處理後的文件
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"文本處理完成，保存為 {output_file}")
