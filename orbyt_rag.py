#!/usr/bin/env python
# coding: utf-8

import os, sys, logging, shutil, tempfile, requests
from typing import List, Optional

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
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

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
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 3000))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
VECTOR_DIR = os.environ.get("VECTOR_DIR", "orbyt_vector_db")  # para Chroma persistido em disco
RETRIEVER_K = int(os.environ.get("RETRIEVER_K", 10))
RERANK_TOP_N = int(os.environ.get("RERANK_TOP_N", 3))

# Garantir diretório base
os.makedirs(VECTOR_DIR, exist_ok=True)

# -------------------------------------------------------------------
# Modelos
# -------------------------------------------------------------------
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

llm = ChatOpenAI(
    model_name=OPENAI_MODEL,
    max_tokens=800,
    temperature=0.2,  # respostas mais focadas (estudo)
    api_key=OPENAI_API_KEY
)

# -------------------------------------------------------------------
# Prompts base (podem ser trocados no futuro)
# -------------------------------------------------------------------
TUTOR_TEMPLATE = """
Você é o Orby, um assistente de estudos que responde com base no material enviado pelo usuário.
Seja claro, objetivo e, quando útil, use exemplos simples. Responda sempre em português.

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
Você é o Orby, um tutor. Crie {n_questions} questões de múltipla escolha, baseadas EXCLUSIVAMENTE no Contexto.
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

Observação: evite conteúdo fora do contexto.
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

Observação: seja realista e priorize os tópicos mais importantes do Contexto.

Objetivo do aluno:
{question}

Contexto:
{context}
"""

# -------------------------------------------------------------------
# Utilitários de coleção (multi-PDF por prova/tema)
# -------------------------------------------------------------------
def _collection_path(user_id: str, collection_id: str, base_dir: str = VECTOR_DIR) -> str:
    """
    Retorna o caminho local da coleção (por usuário + coleção).
    """
    path = os.path.join(base_dir, f"user_{user_id}", f"collection_{collection_id}")
    os.makedirs(path, exist_ok=True)
    return path

def _materialize_pdf(src: str) -> str:
    """
    Aceita caminho local OU URL (ex.: Firebase Storage / S3 presigned URL).
    Se for URL, baixa para um arquivo temporário e retorna o caminho local.
    """
    if src.lower().startswith("http"):
        fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
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

def _build_reranked_retriever(vectordb: Chroma) -> ContextualCompressionRetriever:
    retriever = vectordb.as_retriever(search_kwargs={"k": RETRIEVER_K})
    rerank = CohereRerank(
        model="rerank-multilingual-v3.0",
        top_n=RERANK_TOP_N,
        cohere_api_key=COHERE_API_KEY
    )
    return ContextualCompressionRetriever(base_compressor=rerank, base_retriever=retriever)

# -------------------------------------------------------------------
# Indexação (coleção com 1..N PDFs)
# -------------------------------------------------------------------
def process_collection(user_id: str, collection_id: str, pdf_sources: List[str]) -> ContextualCompressionRetriever:
    """
    Cria/atualiza a coleção (multi-PDF) do usuário, gerando embeddings e persistindo no disco.
    Retorna um retriever com reranking pronto para consultas RAG.
    """
    coll_dir = _collection_path(user_id, collection_id, VECTOR_DIR)
    splitter = _build_splitter()
    all_chunks = []

    for src in pdf_sources:
        try:
            local_pdf = _materialize_pdf(src)
            loader = PyPDFLoader(local_pdf, extract_images=False)
            pages = loader.load_and_split()
            if not pages:
                logging.warning(f"[{collection_id}] PDF sem texto processável: {src}")
                continue
            all_chunks.extend(splitter.split_documents(pages))
        except Exception as e:
            logging.exception(f"Falha ao processar PDF ({src}): {e}")

    if not all_chunks:
        raise ValueError("Nenhum PDF válido para indexar.")

    # Se já existe índice, adiciona; senão, cria
    if os.path.exists(coll_dir) and os.listdir(coll_dir):
        vectordb = Chroma(persist_directory=coll_dir, embedding_function=embeddings_model)
        vectordb.add_documents(all_chunks)
    else:
        vectordb = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings_model,
            persist_directory=coll_dir
        )

    return _build_reranked_retriever(vectordb)

def load_collection_retriever(user_id: str, collection_id: str) -> ContextualCompressionRetriever:
    """
    Reabre uma coleção existente sem reprocessar PDFs.
    """
    coll_dir = _collection_path(user_id, collection_id, VECTOR_DIR)
    vectordb = Chroma(persist_directory=coll_dir, embedding_function=embeddings_model)
    return _build_reranked_retriever(vectordb)

def delete_collection(user_id: str, collection_id: str) -> None:
    """
    Remove completamente a coleção (botão 'Encerrar estudos').
    """
    coll_dir = _collection_path(user_id, collection_id, VECTOR_DIR)
    if os.path.exists(coll_dir):
        shutil.rmtree(coll_dir)

# -------------------------------------------------------------------
# Compat: processamento de UM PDF isolado (se precisar)
# -------------------------------------------------------------------
def process_pdf(pdf_path_or_url: str) -> ContextualCompressionRetriever:
    """
    Mantido por compatibilidade: processa um único PDF em um índice "global".
    Prefira process_collection para o fluxo do Orbyt.
    """
    loader = PyPDFLoader(pdf_path_or_url, extract_images=False)
    pages = loader.load_and_split()
    if not pages:
        logging.error("PDF vazio ou ilegível.")
        raise ValueError("O PDF não contém texto processável.")

    splitter = _build_splitter()
    chunks = splitter.split_documents(pages)

    os.makedirs(VECTOR_DIR, exist_ok=True)
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings_model, persist_directory=VECTOR_DIR)
    return _build_reranked_retriever(vectordb)

# -------------------------------------------------------------------
# Chains RAG
# -------------------------------------------------------------------
def create_rag_chain(compressor_retriever: ContextualCompressionRetriever, prompt_template: Optional[str] = None):
    """
    Cria pipeline RAG com o retriever informado e um prompt (padrão: tutor).
    """
    template = prompt_template or TUTOR_TEMPLATE
    prompt = ChatPromptTemplate.from_template(template)

    setup_retrieval = RunnableParallel({
        "question": RunnablePassthrough(),
        "context": compressor_retriever
    })
    output_parser = StrOutputParser()
    return setup_retrieval | prompt | llm | output_parser

# -------------------------------------------------------------------
# Funcionalidades do Orbyt (tutor, flashcards, exercícios, plano)
# -------------------------------------------------------------------
def ask_question(retriever: ContextualCompressionRetriever, question: str) -> str:
    """
    Tutor: responde perguntas com base na coleção (RAG).
    """
    chain = create_rag_chain(retriever, prompt_template=TUTOR_TEMPLATE)
    return chain.invoke(question)

def generate_flashcards(retriever: ContextualCompressionRetriever, objective: str, n_cards: int = 10) -> str:
    """
    Gera flashcards (JSON de front/back) com base na coleção.
    Retorna string JSON (parse no app se quiser).
    """
    tmpl = FLASHCARDS_TEMPLATE
    chain = create_rag_chain(retriever, prompt_template=tmpl.replace("{n_cards}", str(n_cards)))
    return chain.invoke(objective)

def generate_exercises(retriever: ContextualCompressionRetriever, objective: str, n_questions: int = 6, difficulty: str = "médio") -> str:
    """
    Gera exercícios de múltipla escolha (JSON com questions/options/answer/explanation).
    """
    tmpl = EXERCISES_TEMPLATE.replace("{n_questions}", str(n_questions)).replace("{difficulty}", difficulty)
    chain = create_rag_chain(retriever, prompt_template=tmpl)
    return chain.invoke(objective)

def generate_study_plan(retriever: ContextualCompressionRetriever, objective: str, days: int = 7, minutes_per_day: int = 60) -> str:
    """
    Gera plano de estudos (JSON com dias, tópicos, metas e tarefas).
    """
    tmpl = (STUDY_PLAN_TEMPLATE
            .replace("{days}", str(days))
            .replace("{minutes_per_day}", str(minutes_per_day)))
    chain = create_rag_chain(retriever, prompt_template=tmpl)
    return chain.invoke(objective)
