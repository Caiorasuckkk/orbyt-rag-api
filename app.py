from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

from orbyt_rag import (
    process_collection, load_collection_retriever, delete_collection,
    create_rag_chain, ask_question, generate_flashcards,
    generate_exercises, generate_study_plan
)

# ---------------------------------------
# Configuração do FastAPI + CORS
# ---------------------------------------
app = FastAPI(title="Orbyt RAG API", version="1.0.0")

# Libere a origem do seu app (ou "*", durante o MVP)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # coloque o domínio/URL do seu app em produção
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache opcional em memória (melhora latência após indexar)
retrievers_cache = {}  # chave: (user_id, collection_id) -> retriever

# ---------------------------------------
# Schemas (Pydantic)
# ---------------------------------------
class IndexPayload(BaseModel):
    pdf_urls: List[str] = Field(..., description="URLs dos PDFs (Firebase/S3 presigned, etc.)")

class AskPayload(BaseModel):
    question: str = Field(..., description="Pergunta do usuário")
    prompt_template: Optional[str] = Field(None, description="Opcional: prompt customizado")

class FlashcardsPayload(BaseModel):
    objective: str = Field(..., description="Tópico/objetivo para gerar cards")
    n_cards: int = Field(10, ge=1, le=50, description="Quantidade aproximada de cards")
    prompt_template: Optional[str] = None

class ExercisesPayload(BaseModel):
    objective: str = Field(..., description="Tema/objetivo das questões")
    n_questions: int = Field(6, ge=1, le=30)
    difficulty: str = Field("médio", description="fácil | médio | difícil")
    prompt_template: Optional[str] = None

class StudyPlanPayload(BaseModel):
    objective: str = Field(..., description="Objetivo do aluno (ex.: prova tal)")
    days: int = Field(7, ge=1, le=60)
    minutes_per_day: int = Field(60, ge=15, le=480)
    prompt_template: Optional[str] = None

# ---------------------------------------
# Health-check
# ---------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------------------------------
# Indexar/atualizar coleção (multi-PDF)
# ---------------------------------------
@app.post("/collections/{user_id}/{collection_id}/index")
def index_collection(user_id: str, collection_id: str, payload: IndexPayload):
    try:
        retriever = process_collection(user_id, collection_id, payload.pdf_urls)
        retrievers_cache[(user_id, collection_id)] = retriever
        return {"status": "indexed", "pdf_count": len(payload.pdf_urls)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ---------------------------------------
# Tutor (chat RAG)
# ---------------------------------------
@app.post("/collections/{user_id}/{collection_id}/ask")
def ask(user_id: str, collection_id: str, payload: AskPayload):
    try:
        retriever = retrievers_cache.get((user_id, collection_id)) or load_collection_retriever(user_id, collection_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Coleção não encontrada. Faça o index primeiro.")

    # Se veio prompt customizado, usamos ele; senão usamos o padrão interno
    if payload.prompt_template:
        chain = create_rag_chain(retriever, prompt_template=payload.prompt_template)
        answer = chain.invoke(payload.question)
    else:
        answer = ask_question(retriever, payload.question)

    return {"answer": answer}

# ---------------------------------------
# Flashcards
# ---------------------------------------
@app.post("/collections/{user_id}/{collection_id}/flashcards")
def flashcards(user_id: str, collection_id: str, payload: FlashcardsPayload):
    try:
        retriever = retrievers_cache.get((user_id, collection_id)) or load_collection_retriever(user_id, collection_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Coleção não encontrada.")

    if payload.prompt_template:
        # sobrepor prompt
        chain = create_rag_chain(retriever, prompt_template=payload.prompt_template)
        result = chain.invoke(payload.objective)
    else:
        result = generate_flashcards(retriever, payload.objective, n_cards=payload.n_cards)

    return {"flashcards": result}

# ---------------------------------------
# Exercícios
# ---------------------------------------
@app.post("/collections/{user_id}/{collection_id}/exercises")
def exercises(user_id: str, collection_id: str, payload: ExercisesPayload):
    try:
        retriever = retrievers_cache.get((user_id, collection_id)) or load_collection_retriever(user_id, collection_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Coleção não encontrada.")

    if payload.prompt_template:
        chain = create_rag_chain(retriever, prompt_template=payload.prompt_template)
        result = chain.invoke(payload.objective)
    else:
        result = generate_exercises(
            retriever, payload.objective,
            n_questions=payload.n_questions,
            difficulty=payload.difficulty
        )

    return {"exercises": result}

# ---------------------------------------
# Plano de estudos
# ---------------------------------------
@app.post("/collections/{user_id}/{collection_id}/study-plan")
def study_plan(user_id: str, collection_id: str, payload: StudyPlanPayload):
    try:
        retriever = retrievers_cache.get((user_id, collection_id)) or load_collection_retriever(user_id, collection_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Coleção não encontrada.")

    if payload.prompt_template:
        chain = create_rag_chain(retriever, prompt_template=payload.prompt_template)
        result = chain.invoke(payload.objective)
    else:
        result = generate_study_plan(
            retriever, payload.objective,
            days=payload.days, minutes_per_day=payload.minutes_per_day
        )

    return {"study_plan": result}

# ---------------------------------------
# Encerrar estudos (apagar coleção)
# ---------------------------------------
@app.delete("/collections/{user_id}/{collection_id}")
def delete(user_id: str, collection_id: str):
    delete_collection(user_id, collection_id)
    retrievers_cache.pop((user_id, collection_id), None)
    return {"status": "deleted"}
