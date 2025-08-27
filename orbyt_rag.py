#!/usr/bin/env python
# coding: utf-8

import os, sys, logging, shutil, tempfile, requests
import json, re  # ✅ ADICIONADO
from typing import List, Optional

# Desativa ruído de telemetria do Chroma
os.environ["ANONYMIZED_TELEMETRY"] = "false"

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank, CohereEmbeddings, ChatCohere
from fastapi import HTTPException  # ✅ necessário no endpoint /exercises

# -------------------------------------------------------------------
# Encoding e logging
# -------------------------------------------------------------------
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------------------------------------------------------
# Variáveis de ambiente
# -------------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Defina a variável de ambiente OPENAI_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("Defina a variável de ambiente COHERE_API_KEY")

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1500))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 100))
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo-0125")
VECTOR_DIR = os.environ.get("VECTOR_DIR", "orbyt_vector_db")
RETRIEVER_K = int(os.environ.get("RETRIEVER_K", 8))
RERANK_TOP_N = int(os.environ.get("RERANK_TOP_N", 3))

# Controle de contexto enviado ao LLM
MAX_DOCS_IN_PROMPT = int(os.environ.get("MAX_DOCS_IN_PROMPT", 4))
MAX_CONTEXT_CHARS   = int(os.environ.get("MAX_CONTEXT_CHARS", 3000))

# Preferência de LLM e modelo Cohere
LLM_PRIMARY = os.environ.get("LLM_PRIMARY", "openai").lower()   # "openai" ou "cohere"
COHERE_MODEL = os.environ.get("COHERE_MODEL", "command-r")

os.makedirs(VECTOR_DIR, exist_ok=True)

# -------------------------------------------------------------------
# Modelos
# -------------------------------------------------------------------
embeddings_model = CohereEmbeddings(
    model="embed-multilingual-v3.0",
    cohere_api_key=COHERE_API_KEY
)

openai_llm = ChatOpenAI(
    model_name=OPENAI_MODEL,
    max_tokens=800,
    temperature=0.2,
    api_key=OPENAI_API_KEY,
    max_retries=5,
    timeout=60,
)

cohere_llm = ChatCohere(
    model=COHERE_MODEL,
    cohere_api_key=COHERE_API_KEY,
    max_tokens=800,
    temperature=0.2,
)

def _select_llm_with_fallback():
    if LLM_PRIMARY == "cohere":
        return cohere_llm.with_fallbacks([openai_llm])
    return openai_llm.with_fallbacks([cohere_llm])

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
    if not docs:
        return "NENHUM TRECHO ENCONTRADO."
    out = []
    used = 0
    for i, d in enumerate(docs[:MAX_DOCS_IN_PROMPT]):
        txt = (getattr(d, "page_content", "") or "").strip()
        if not txt:
            continue
        txt = " ".join(txt.split())
        remaining = MAX_CONTEXT_CHARS - used
        if remaining <= 0:
            break
        snippet = txt[:remaining]
        out.append(f"Trecho {i+1}:\n{snippet}")
        used += len(snippet)
        if used >= MAX_CONTEXT_CHARS:
            break
    return "\n\n".join(out) if out else "NENHUM TRECHO ENCONTRADO."

def _build_reranked_retriever(vectordb: Chroma) -> ContextualCompressionRetriever:
    retriever = vectordb.as_retriever(search_kwargs={"k": RETRIEVER_K})
    rerank = CohereRerank(
        model="rerank-multilingual-v3.0",
        top_n=RERANK_TOP_N,
        cohere_api_key=COHERE_API_KEY
    )
    return ContextualCompressionRetriever(base_compressor=rerank, base_retriever=retriever)

# -------------------------------------------------------------------
# 🔧 Correções de normalização para EXERCÍCIOS
# -------------------------------------------------------------------
CODE_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```", re.I)

def _strip_code_fences(s: str) -> str:
    return CODE_FENCE_RE.sub("", s).strip()

def _safe_json_loads(s: str):
    s = _strip_code_fences(s)
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"(\{.*\}|\[.*\])", s, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return {}
        return {}

def _normalize_exercises_payload(result):
    """
    Recebe `result` (string, dict ou list) e retorna:
    { "questions": [ {question, options[], answer, explanation} ] }
    """
    # 1) Converte string em JSON
    if isinstance(result, str):
        obj = _safe_json_loads(result)
    else:
        obj = result

    # 2) Caso venha aninhado como {"exercises": "..."} ou {"exercises": {...}}
    if isinstance(obj, dict) and "exercises" in obj:
        ex = obj["exercises"]
        if isinstance(ex, str):
            obj = _safe_json_loads(ex)
        else:
            obj = ex

    # 3) Extrai lista de questões
    questions_raw = []
    if isinstance(obj, dict):
        if "questions" in obj and isinstance(obj["questions"], list):
            questions_raw = obj["questions"]
        else:
            # Pode vir como {"Q1": {...}, "Q2": {...}}
            questions_raw = list(obj.values())
    elif isinstance(obj, list):
        questions_raw = obj

    # 4) Normaliza cada item
    norm = []
    for it in questions_raw or []:
        if not isinstance(it, dict):
            continue

        q_text = (it.get("question") or "").strip()

        # options pode ser list/str/dict
        opts = it.get("options", [])
        if isinstance(opts, dict):
            ordered = [opts.get(k) for k in ["A", "B", "C", "D"] if opts.get(k)]
            opts = ordered
        elif isinstance(opts, str):
            parts = [p.strip() for p in re.split(r"[\n;]", opts) if p.strip()]
            opts = parts
        elif isinstance(opts, list):
            opts = [str(o) for o in opts]
        else:
            opts = []

        ans = it.get("answer")
        if isinstance(ans, str):
            m = re.match(r"\s*([ABCD])", ans.strip(), re.I)
            if m:
                ans = m.group(1).upper()

        norm.append({
            "question": q_text if q_text else "Pergunta não encontrada",
            "options": opts,
            "answer": ans,  # "A"/"B"/"C"/"D" (o App converte p/ índice)
            "explanation": (it.get("explanation") or "").strip()
        })

    return {"questions": norm}

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
    llm_chain = _select_llm_with_fallback()
    return setup_retrieval | prompt | llm_chain | output_parser

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

