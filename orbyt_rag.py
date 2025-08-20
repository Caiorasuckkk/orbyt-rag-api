#!/usr/bin/env python
# coding: utf-8

import os, sys, logging, shutil, tempfile, requests, time
from typing import List, Optional, Tuple, Iterable

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
# Variáveis de ambiente
# -------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Defina a variável de ambiente OPENAI_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("Defina a variável de ambiente COHERE_API_KEY")

# Configs (ajuste à vontade no /etc/orbyt-rag.env)
CHUNK_SIZE     = int(os.environ.get("CHUNK_SIZE", 1200))   # menor → menos tokens
CHUNK_OVERLAP  = int(os.environ.get("CHUNK_OVERLAP", 120))
OPENAI_MODEL   = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
VECTOR_DIR     = os.environ.get("VECTOR_DIR", "orbyt_vector_db")
RETRIEVER_K    = int(os.environ.get("RETRIEVER_K", 8))
RERANK_TOP_N   = int(os.environ.get("RERANK_TOP_N", 3))
EMB_BATCH_SIZE = int(os.environ.get("EMB_BATCH_SIZE", 32))  # lotes pequenos p/ reduzir 520
EMB_RETRIES    = int(os.environ.get("EMB_RETRIES", 6))
EMB_SLEEP_BASE = float(os.environ.get("EMB_SLEEP_BASE", 0.8))  # backoff base

os.makedirs(VECTOR_DIR, exist_ok=True)

# -------------------------------------------------------------------
# Modelos
# -------------------------------------------------------------------
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
    timeout=60,       # importante para rede instável
    max_retries=EMB_RETRIES
)

llm = ChatOpenAI(
    model_name=OPENAI_MODEL,
    max_tokens=800,
    temperature=0.2,
    api_key=OPENAI_API_KEY,
    timeout=60,
    max_retries=3
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
Você é o Orby, um tutor de estudos. Gere flashcards concisos com base EXCLUSIVAMENTE no Contexto.
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
# Helpers
# -------------------------------------------------------------------
def _collection_path(user_id: str, collection_id: str, base_dir: str = VECTOR_DIR) -> str:
    path = os.path.join(base_dir, f"user_{user_id}", f"collection_{collection_id}")
    os.makedirs(path, exist_ok=True)
    return path

def _download_pdf(url: str) -> Tuple[str, bool]:
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

def _batched(seq: List, size: int) -> Iterable[List]:
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

# -------------------------------------------------------------------
# Indexação com lotes e backoff (mitiga 520)
# -------------------------------------------------------------------
def process_collection(user_id: str, collection_id: str, pdf_sources: List[str]) -> ContextualCompressionRetriever:
    coll_dir = _collection_path(user_id, collection_id, VECTOR_DIR)
    splitter = _build_splitter()
    all_chunks = []
    total_pages = 0

    for src in pdf_sources:
        local_pdf, is_temp = None, False
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

    # -------------------------------------------------------------------
    # Filtro e saneamento de chunks (evita 520 da OpenAI)
    # -------------------------------------------------------------------
    valid_chunks = []
    for i, doc in enumerate(all_chunks, start=1):
        txt = (doc.page_content or "").strip()
        if not txt or len(txt) < 10:
            logging.warning(f"[{collection_id}] Chunk #{i} descartado (vazio/curto)")
            continue
        if len(txt) > 7000:  # limite seguro p/ embeddings
            logging.warning(f"[{collection_id}] Chunk #{i} muito grande ({len(txt)} chars), truncando...")
            doc.page_content = txt[:7000]
        valid_chunks.append(doc)

    all_chunks = valid_chunks
    if not all_chunks:
        raise ValueError("Todos os chunks foram descartados (conteúdo inválido).")

    # cria ou reabre o índice
    if os.path.exists(coll_dir) and os.listdir(coll_dir):
        vectordb = Chroma(persist_directory=coll_dir, embedding_function=embeddings_model)
    else:
        # cria com o primeiro lote
        first_batch = next(_batched(all_chunks, EMB_BATCH_SIZE))
        logging.info(f"[{collection_id}] Criando coleção com {len(first_batch)} chunks iniciais...")
        vectordb = Chroma.from_documents(
            documents=first_batch,
            embedding=embeddings_model,
            persist_directory=coll_dir
        )
        remaining = all_chunks[len(first_batch):]
        if not remaining:
            logging.info(f"Coleção {collection_id} criada: {len(all_chunks)} chunks (total págs: {total_pages})")
            return _build_reranked_retriever(vectordb)
        all_chunks = remaining  # prossegue adicionando

    # adiciona em lotes com pequenos backoffs
    added = 0
    for batch_idx, batch in enumerate(_batched(all_chunks, EMB_BATCH_SIZE), start=1):
        for attempt in range(EMB_RETRIES):
            try:
                logging.info(f"[{collection_id}] Enviando batch {batch_idx} com {len(batch)} chunks (tentativa {attempt+1})...")
                vectordb.add_documents(batch)
                added += len(batch)
                break
            except Exception as e:
                wait = EMB_SLEEP_BASE * (2 ** attempt)
                logging.warning(
                    f"[{collection_id}] Erro embeddings (tentativa {attempt+1}/{EMB_RETRIES}). "
                    f"Aguardando {wait:.1f}s. Detalhe: {e}"
                )
                time.sleep(wait)
        else:
            # todas as tentativas falharam
            raise RuntimeError(f"[{collection_id}] Falha ao gerar embeddings (erros repetidos do provedor).")

    logging.info(f"Coleção {collection_id} atualizada/criada: +{added} chunks (total págs: {total_pages})")
    return _build_reranked_retriever(vectordb)

    coll_dir = _collection_path(user_id, collection_id, VECTOR_DIR)
    splitter = _build_splitter()
    all_chunks = []
    total_pages = 0

    for src in pdf_sources:
        local_pdf, is_temp = None, False
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

    # cria ou reabre o índice
    if os.path.exists(coll_dir) and os.listdir(coll_dir):
        vectordb = Chroma(persist_directory=coll_dir, embedding_function=embeddings_model)
    else:
        # cria com o primeiro lote
        first_batch = next(_batched(all_chunks, EMB_BATCH_SIZE))
        vectordb = Chroma.from_documents(
            documents=first_batch,
            embedding=embeddings_model,
            persist_directory=coll_dir
        )
        remaining = all_chunks[len(first_batch):]
        if not remaining:
            logging.info(f"Coleção {collection_id} criada: {len(all_chunks)} chunks (total págs: {total_pages})")
            return _build_reranked_retriever(vectordb)
        all_chunks = remaining  # prossegue adicionando

    # adiciona em lotes com pequenos backoffs
    added = 0
    for batch in _batched(all_chunks, EMB_BATCH_SIZE):
        for attempt in range(EMB_RETRIES):
            try:
                vectordb.add_documents(batch)
                added += len(batch)
                break
            except Exception as e:
                wait = EMB_SLEEP_BASE * (2 ** attempt)
                logging.warning(f"Embeddings 520/erro transitório (tentativa {attempt+1}/{EMB_RETRIES}). "
                                f"Aguardando {wait:.1f}s. Detalhe: {e}")
                time.sleep(wait)
        else:
            # todas as tentativas falharam
            raise RuntimeError("Falha ao gerar embeddings (erros repetidos do provedor). Tente novamente em instantes.")

    logging.info(f"Coleção {collection_id} atualizada/criada: +{added} chunks (total págs: {total_pages})")
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
