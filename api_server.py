import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 1) إعداد FastAPI
app = FastAPI(title="HR RAG QA API")

# 2) إعداد نموذج embeddings
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# 3) إعداد مخزن المتجهات Chroma
client = Client(settings=Settings(persist_directory="02.Vector-Store"))
collection = client.get_or_create_collection(name="hr_docs")

# 4) تحميل نموذج LLM مجاني عبر HuggingFace (GPT-J 6B)
model_name = "EleutherAI/gpt-j-6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map={"": "cpu"}
)

hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    do_sample=True,
    temperature=0.7
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# 5) إعداد Prompt
prompt_template = """
You are an HR data assistant. Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer in a concise, informative way with source references if possible.
"""
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)
chain = LLMChain(llm=llm, prompt=prompt)

# 6) نموذج Pydantic للسؤال
class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(q: Query):
    if not q.question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    # 1) embedding للسؤال
    q_emb = embed_model.encode([q.question])[0]

    # 2) بحث متجهات لاسترجاع أعلى 5 نتائج
    results = collection.query(query_embeddings=[q_emb], n_results=5)
    docs = results['documents'][0]
    metadatas = results['metadatas'][0]

    # 3) دمج النصوص المسترجعة
    context = "\n---\n".join([f"Source: {m['source']}\n{text}" for text, m in zip(docs, metadatas)])

    # 4) توليد الإجابة
    answer = chain.run(context=context, question=q.question)

    return {"answer": answer}

# لتشغيل الخادم:
# uvicorn api_server:app --host 0.0.0.0 --port 8000

