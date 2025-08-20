#!/usr/bin/env python
# coding: utf-8

import os, sys, logging, shutil, tempfile, requests
from typing import List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

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
# Variáveis de ambiente (NÃO hardcode chaves!)
# -------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Defina a variável de ambiente OPENAI_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("Defina a variável de ambiente COHERE_API_KEY")

# Configs ajustáveis por ambiente
CHUNK_SIZE     = int(os.environ.get("CHUNK_SIZE", 1500))   # menor para reduzir tokens
CHUNK_OVERLAP  = int(os.environ.get("CHUNK_OVERLAP", 100))
OPENAI_MODEL   = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
VECTOR_DIR     = os.environ.get("VECTOR_DIR", "orbyt_vector_db")
RETRIEVER_K    = int(os.environ.get("RETRIEVER_K", 8))
RERANK_TOP_N   = int(os.environ.get("RERANK_TOP_N", 3))

os.makedirs(VECTOR_DIR, exist_ok=True)

# -------------------------------------------------------------------
# Modelos
# -------------------------------------------------------------------
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

# timeout + retries ajudam quando a OpenAI devolve 520/instabilidades
llm = ChatOpenAI(
    model_name=OPENAI_MODEL,
    max_tokens=800,
    temperature=0.2,
    api_key=OPENAI_API_KEY,
    timeout=60,
    max_retries=3,  # tenta novamente em erro transitório
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
  { "front": "pergunta ou termo", "back": "resposta objetiva" }
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
{
  "questions": [
    {
      "question": "enunciado",
      "options": ["A) ...","B) ...","C) ...","D) ..."],
      "answer": "A",
      "explanation": "por quê a resposta está correta"
    }
  ]
}

Contexto:
{context}
"""

STUDY_PLAN_TEMPLATE = """
Você é o Orby. Gere um plano de estudos baseado EXCLUSIVAMENTE no Contexto. 
Distribua conteúdos por {days} dias, considerando cerca de {minutes_per_day} minutos por dia.

Formato (JSON válido):
{
  "plan": [
    {
      "day": 1,
      "topics": ["...", "..."],
      "goals": ["...", "..."],
      "tasks": ["...", "..."],
      "estimated_minutes": {minutes_per_day}
    }
  ]
}

Objetivo do aluno:
{question}

Contexto:
{context}
"""

# -------------------------------------------------------------------
# Utils coleção
# -------------------------------------------------------------------
def _collection_path(user_id: str, collection_id: str, base_dir: str = VECTOR_DIR) -> str:
    path = os.path.join(base_dir, f"user_{user_id}", f"collection_{collection_id}")
    os.makedirs(path, exist_ok=True)
    return path

def _download_pdf(url: str) -> Tuple[str, bool]:
    """Baixa URL para arquivo temporário e retorna (path, is_temp=True)."""
    fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    logging.info(f"Baixando PDF de {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    return tmp_path, True

def _materialize_pdf(src: str) -> Tuple[str, bool]:
    """
    Retorna (caminho_local, is_temp). Se for URL, baixa e marca is_temp=True.
    """
    if src.lower().startswith("http"):
        return _download_pdf(src)
    return src, False

def _build_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True
    )

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
    total_pages = 0

    for src in pdf_sources:
        local_pdf = None
        is_temp = False
        try:
            local_pdf, is_temp = _materialize_pdf(src)
            loader = PyPDFLoader(local_pdf, extract_images=False)
            pages = loader.load_and_split()
            if not pages:
                logging.warning(f"[{collection_id}] PDF sem texto processável: {src}")
                continue
            total_pages += len(pages)
            chunks = splitter.split_documents(pages)
            all_chunks.extend(chunks)
            logging.info(f"PDF {src} → {len(pages)} páginas, {len(chunks)} chunks")
        except Exception as e:
            logging.exception(f"Erro ao processar PDF {src}: {e}")
        finally:
            if is_temp and local_pdf and os.path.exists(local_pdf):
                try:
                    os.remove(local_pdf)
                except Exception:
                    pass

    if not all_chunks:
        raise ValueError("Nenhum PDF válido para indexar.")

    if os.path.exists(coll_dir) and os.listdir(coll_dir):
        vectordb = Chroma(persist_directory=coll_dir, embedding_function=embeddings_model)
        vectordb.add_documents(all_chunks)
        logging.info(f"Coleção {collection_id} atualizada: +{len(all_chunks)} chunks (total páginas: {total_pages})")
    else:
        vectordb = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings_model,
            persist_directory=coll_dir
        )
        logging.info(f"Coleção {collection_id} criada: {len(all_chunks)} chunks (total páginas: {total_pages})")

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
# Chains RAG
# -------------------------------------------------------------------
def create_rag_chain(compressor_retriever: ContextualCompressionRetriever, prompt_template: Optional[str] = None):
    template = prompt_template or TUTOR_TEMPLATE
    prompt = ChatPromptTemplate.from_template(template)

    setup_retrieval = RunnableParallel({
        "question": RunnablePassthrough(),
        "context": compressor_retriever
    })
    output_parser = StrOutputParser()
    return setup_retrieval | prompt | llm | output_parser

# -------------------------------------------------------------------
# Funcionalidades
# -------------------------------------------------------------------
def ask_question(retriever: ContextualCompressionRetriever, question: str) -> str:
    try:
        chain = create_rag_chain(retriever, prompt_template=TUTOR_TEMPLATE)
        return chain.invoke(question)
    except Exception as e:
        logging.exception(f"Erro em ask_question: {e}")
        raise

def generate_flashcards(retriever: ContextualCompressionRetriever, objective: str, n_cards: int = 10) -> str:
    try:
        tmpl = FLASHCARDS_TEMPLATE.replace("{n_cards}", str(n_cards))
        chain = create_rag_chain(retriever, prompt_template=tmpl)
        return chain.invoke(objective)
    except Exception as e:
        logging.exception(f"Erro em generate_flashcards: {e}")
        raise

def generate_exercises(retriever: ContextualCompressionRetriever, objective: str, n_questions: int = 6, difficulty: str = "médio") -> str:
    try:
        tmpl = EXERCISES_TEMPLATE.replace("{n_questions}", str(n_questions)).replace("{difficulty}", difficulty)
        chain = create_rag_chain(retriever, prompt_template=tmpl)
        return chain.invoke(objective)
    except Exception as e:
        logging.exception(f"Erro em generate_exercises: {e}")
        raise

def generate_study_plan(retriever: ContextualCompressionRetriever, objective: str, days: int = 7, minutes_per_day: int = 60) -> str:
    try:
        tmpl = (STUDY_PLAN_TEMPLATE
                .replace("{days}", str(days))
                .replace("{minutes_per_day}", str(minutes_per_day)))
        chain = create_rag_chain(retriever, prompt_template=tmpl)
        return chain.invoke(objective)
    except Exception as e:
        logging.exception(f"Erro em generate_study_plan: {e}")
        raise
