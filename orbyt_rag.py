#!/usr/bin/env python
# coding: utf-8

import os, sys, logging, shutil, tempfile, requests
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank, CohereEmbeddings

# -------------------------------------------------------------------
# Encoding e logging
# -------------------------------------------------------------------
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# -------------------------------------------------------------------
# Variáveis de ambiente
# -------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Defina a variável de ambiente OPENAI_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("Defina a variável de ambiente COHERE_API_KEY")

# Configs ajustáveis
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1500))   # bom p/ instância 512MB
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 100))
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
VECTOR_DIR = os.environ.get("VECTOR_DIR", "orbyt_vector_db")
RETRIEVER_K = int(os.environ.get("RETRIEVER_K", 8))
RERANK_TOP_N = int(os.environ.get("RERANK_TOP_N", 3))

os.makedirs(VECTOR_DIR, exist_ok=True)

# -------------------------------------------------------------------
# Modelos
#   - Embeddings: Cohere (foge do /v1/embeddings da OpenAI que estava 520)
#   - LLM: OpenAI, com retry/timeout
# -------------------------------------------------------------------
embeddings_model = CohereEmbeddings(
    model="embed-multilingual-v3.0",
    cohere_api_key=COHERE_API_KEY
)

llm = ChatOpenAI(
    model_name=OPENAI_MODEL,
    max_tokens=800,
    temperature=0.2,
    api_key=OPENAI_API_KEY,
    max_retries=5,     # tenta de novo em caso de 5xx
    timeout=60         # evita travar
)

# -------------------------------------------------------------------
# Prompts base
# -------------------------------------------------------------------
TUTOR_TEMPLATE = """
Você é o Orby, um assistente de estudos que responde com base no material enviado pelo usuário.
Responda sempre em português, de forma clara e objetiva.

Pergunta:
{question}

Contexto:
{context}
"""

FLASHCARDS_TEMPLATE = """
Você é o Orby, um tutor de estudos. Gere flashcards concisos com base EXCLUSIVA no Contexto.
Formato de saída (JSON válido): 
[
  {{ "front": "pergunta ou termo", "back": "resposta objetiva" }},
  ...
]

Quantidade aproximada: {n_cards} cards.

Tópico/Objetivo do aluno:
{question}

Contexto:
{context}
"""

EXERCISES_TEMPLATE = """
Você é o Orby. Crie {n_questions} questões de múltipla escolha, baseadas EXCLUSIVAMENTE no Contexto.
Nível: {difficulty}. Inclua gabarito e explicação breve.

Formato (JSON válido):
{{
  "questions": [
    {{
      "question": "enunciado",
      "options": ["A) ...","B) ...","C) ...","D) ..."],
      "answer": "A",
      "explanation": "por quê a resposta está correta"
    }}
  ]
}}

Contexto:
{context}
"""

STUDY_PLAN_TEMPLATE = """
Você é o Orby. Gere um plano de estudos baseado EXCLUSIVAMENTE no Contexto. 
Distribua conteúdos por {days} dias, considerando cerca de {minutes_per_day} minutos por dia.

Formato (JSON válido):
{{
  "plan": [
    {{
      "day": 1,
      "topics": ["...", "..."],
      "goals": ["...", "..."],
      "tasks": ["...", "..."],
      "estimated_minutes": {minutes_per_day}
    }}
  ]
}}

Objetivo do aluno:
{question}

Contexto:
{context}
"""

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _collection_path(user_id: str, collection_id: str, base_dir: str = VECTOR_DIR) -> str:
    path = os.path.join(base_dir, f"user_{user_id}", f"collection_{collection_id}")
    os.makedirs(path, exist_ok=True)
    return path

def _materialize_pdf(src: str) -> str:
    if src.lower().startswith("http"):
        fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        logging.info(f"Baixando PDF de {src}")
        with requests.get(src, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
        return tmp_path
    return src

def _build_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True
    )

def _combine_docs(docs) -> str:
    """Transforma lista de Document em texto legível no prompt."""
    if not docs:
        return "NENHUM TRECHO ENCONTRADO."
    parts = []
    for i, d in enumerate(docs[:8]):  # limita contexto bruto no prompt
        txt = (d.page_content or "").strip()
        if not txt:
            continue
        parts.append(f"Trecho {i+1}:\n{txt}")
    return "\n\n".join(parts) if parts else "NENHUM TRECHO ENCONTRADO."

def _build_reranked_retriever(vectordb: Chroma) -> ContextualCompressionRetriever:
    retriever = vectordb.as_retriever(search_kwargs={"k": RETRIEVER_K})
    rerank = CohereRerank(
        model="rerank-multilingual-v3.0",
        top_n=RERANK_TOP_N,
        cohere_api_key=COHERE_API_KEY
    )
    return ContextualCompressionRetriever(base_compressor=rerank, base_retriever=retriever)

# -------------------------------------------------------------------
# Indexação
# -------------------------------------------------------------------
def process_collection(user_id: str, collection_id: str, pdf_sources: List[str]) -> ContextualCompressionRetriever:
    coll_dir = _collection_path(user_id, collection_id, VECTOR_DIR)
    splitter = _build_splitter()
    all_chunks = []

    for src in pdf_sources:
        try:
            local_pdf = _materialize_pdf(src)
            loader = PyPDFLoader(local_pdf, extract_images=False)
            pages = loader.load_and_split()
            if not pages:
                logging.warning(f"PDF sem texto processável: {src}")
                continue
            chunks = splitter.split_documents(pages)
            all_chunks.extend(chunks)
            logging.info(f"PDF {src} → {len(pages)} páginas, {len(chunks)} chunks")
        except Exception as e:
            logging.exception(f"Erro ao processar PDF {src}: {e}")

    if not all_chunks:
        raise ValueError("Nenhum PDF válido para indexar.")

    if os.path.exists(coll_dir) and os.listdir(coll_dir):
        vectordb = Chroma(persist_directory=coll_dir, embedding_function=embeddings_model)
        vectordb.add_documents(all_chunks)
        logging.info(f"Coleção {collection_id} atualizada com {len(all_chunks)} chunks")
    else:
        vectordb = Chroma.from_documents(all_chunks, embeddings_model, persist_directory=coll_dir)
        logging.info(f"Coleção {collection_id} criada com {len(all_chunks)} chunks")

    return _build_reranked_retriever(vectordb)

def load_collection_retriever(user_id: str, collection_id: str) -> ContextualCompressionRetriever:
    coll_dir = _collection_path(user_id, collection_id, VECTOR_DIR)
    vectordb = Chroma(persist_directory=coll_dir, embedding_function=embeddings_model)
    return _build_reranked_retriever(vectordb)

def delete_collection(user_id: str, collection_id: str) -> None:
    coll_dir = _collection_path(user_id, collection_id, VECTOR_DIR)
    if os.path.exists(coll_dir):
        shutil.rmtree(coll_dir)
        logging.info(f"Coleção {collection_id} deletada")

# -------------------------------------------------------------------
# Chain RAG
# -------------------------------------------------------------------
def create_rag_chain(compressor_retriever: ContextualCompressionRetriever, prompt_template: Optional[str] = None):
    template = prompt_template or TUTOR_TEMPLATE
    prompt = ChatPromptTemplate.from_template(template)

    setup_retrieval = RunnableParallel({
        "question": RunnablePassthrough(),
        "context": compressor_retriever | RunnableLambda(_combine_docs)
    })
    output_parser = StrOutputParser()
    return setup_retrieval | prompt | llm | output_parser

# -------------------------------------------------------------------
# Funcionalidades
# -------------------------------------------------------------------
def ask_question(retriever: ContextualCompressionRetriever, question: str) -> str:
    chain = create_rag_chain(retriever, prompt_template=TUTOR_TEMPLATE)
    return chain.invoke(question)

def generate_flashcards(retriever: ContextualCompressionRetriever, objective: str, n_cards: int = 10) -> str:
    tmpl = FLASHCARDS_TEMPLATE.replace("{n_cards}", str(n_cards))
    chain = create_rag_chain(retriever, prompt_template=tmpl)
    return chain.invoke(objective)

def generate_exercises(retriever: ContextualCompressionRetriever, objective: str, n_questions: int = 6, difficulty: str = "médio") -> str:
    tmpl = EXERCISES_TEMPLATE.replace("{n_questions}", str(n_questions)).replace("{difficulty}", difficulty)
    chain = create_rag_chain(retriever, prompt_template=tmpl)
    return chain.invoke(objective)

def generate_study_plan(retriever: ContextualCompressionRetriever, objective: str, days: int = 7, minutes_per_day: int = 60) -> str:
    tmpl = (STUDY_PLAN_TEMPLATE
            .replace("{days}", str(days))
            .replace("{minutes_per_day}", str(minutes_per_day)))
    chain = create_rag_chain(retriever, prompt_template=tmpl)
    return chain.invoke(objective)
