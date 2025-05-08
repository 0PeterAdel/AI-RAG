import os
# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb import Client
from chromadb.config import Settings

# 1) إعداد نموذج المتجهات للعمل على CPU
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# 2) إعداد مخزن المتجهات المحلي (Chroma)
client = Client(settings=Settings(persist_directory="02.Vector-Store"))
collection = client.get_or_create_collection(name="hr_docs")

# 3) مسار النصوص الخام المولَّدة
RAW_DIR = Path("02.Raw-Text")

# 4) مُجزّئ النصوص: قطع بحجم ~500 حرف مع تداخل صغير
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)

# 5) اجتز كل ملف JSON واحتسب المتجهات
for json_path in RAW_DIR.rglob("*.json"):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    source = data.get("source", "")
    text = data.get("text", "")

    # تقطيع النص إلى قطع صغيرة
    chunks = splitter.split_text(text)
    # تخطى الملفات الخالية
    if not chunks:
        continue

    # احتساب المتجهات دفعة واحدة
    embeddings = embed_model.encode(chunks, show_progress_bar=True)

    # إعداد الميتاداتا والمعرفات
    metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]
    ids = [f"{json_path.stem}-{i}" for i in range(len(chunks))]

    # إضافة القطع إلى المخزن
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=ids
    )

print("✅ Finished chunking and indexing all documents.")

