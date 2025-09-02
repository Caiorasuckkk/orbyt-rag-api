from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from orbyt_rag import (
    EXERCISES_TEMPLATE_STRICT,
    _normalize_exercises_payload, _normalize_oral_payload,
    process_collection, load_collection_retriever, delete_collection,
    create_rag_chain, ask_question,
    generate_exercises, generate_study_plan, generate_oral_questions
)


app = FastAPI(title="Orbyt RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retrievers_cache: Dict[tuple, Any] = {}


class IndexPayload(BaseModel):
    pdf_urls: List[str] = Field(..., description="URLs dos PDFs (Firebase/S3 presigned, etc.)")

class AskPayload(BaseModel):
    question: str = Field(..., description="Pergunta do usuário")
    prompt_template: Optional[str] = Field(None, description="Opcional: prompt customizado")

class ExercisesPayload(BaseModel):
    objective: str = Field(..., description="Tema/objetivo das questões (ou RESUMO se usar contexto fixo)")
    n_questions: int = Field(6, ge=1, le=30)
    difficulty: str = Field("médio", description="fácil | médio | difícil")
    prompt_template: Optional[str] = None
    use_objective_as_context: bool = False  
class StudyPlanPayload(BaseModel):
    objective: str = Field(..., description="Objetivo do aluno (ex.: prova tal)")
    days: int = Field(7, ge=1, le=60)
    minutes_per_day: int = Field(60, ge=15, le=480)
    prompt_template: Optional[str] = None

class OralQuestionsPayload(BaseModel):
    summary: str = Field(..., description="Resumo LIMPO (PII scrub) vindo do app")
    count: int = Field(10, ge=1, le=30, description="Quantidade de perguntas")
    use_summary_as_context: bool = Field(True, description="Se True, usa somente o summary como contexto fixo")


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


@app.post("/collections/{user_id}/{collection_id}/ask")
def ask(user_id: str, collection_id: str, payload: AskPayload):
    try:
        retriever = retrievers_cache.get((user_id, collection_id)) or load_collection_retriever(user_id, collection_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Coleção não encontrada. Faça o index primeiro.")

    if payload.prompt_template:
        chain = create_rag_chain(retriever, prompt_template=payload.prompt_template)
        answer = chain.invoke(payload.question)
    else:
        answer = ask_question(retriever, payload.question)

    return {"answer": answer}


@app.post("/collections/{user_id}/{collection_id}/exercises")
def exercises(user_id: str, collection_id: str, payload: ExercisesPayload):
    try:
        retriever = retrievers_cache.get((user_id, collection_id)) or load_collection_retriever(user_id, collection_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Coleção não encontrada.")

   
    template_str = EXERCISES_TEMPLATE_STRICT \
        .replace("{n_questions}", str(payload.n_questions)) \
        .replace("{difficulty}", payload.difficulty)

    if payload.use_objective_as_context:
       
        chain = create_rag_chain(None, prompt_template=template_str, fixed_context=payload.objective)
        result = chain.invoke("Gerar exercícios a partir do resumo")
    else:
        # Usa o retriever (contexto já é higienizado em _combine_docs)
        chain = create_rag_chain(retriever, prompt_template=template_str)
        result = chain.invoke(payload.objective)

    
    normalized = _normalize_exercises_payload(result)
    return {"exercises": normalized}


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


@app.post("/collections/{user_id}/{collection_id}/oral-questions")
def oral_questions(user_id: str, collection_id: str, payload: OralQuestionsPayload):
    retriever = None
    if not payload.use_summary_as_context:
        try:
            retriever = retrievers_cache.get((user_id, collection_id)) or load_collection_retriever(user_id, collection_id)
        except Exception:
            raise HTTPException(status_code=404, detail="Coleção não encontrada.")

    result = generate_oral_questions(
        retriever,
        payload.summary,
        n_questions=payload.count,
        use_objective_as_context=payload.use_summary_as_context
    )
    items = _normalize_oral_payload(result)
    return {"items": items}

@app.post("/oral_questions")
def oral_questions_root(payload: dict):
    try:
        user_id = str(payload.get("userId") or payload.get("user_id") or "")
        collection_id = str(payload.get("collectionId") or payload.get("collection_id") or "")
        summary = str(payload.get("summary") or "")
        count = int(payload.get("count") or 10)
    except Exception:
        raise HTTPException(status_code=400, detail="Payload inválido")

    if not summary:
        raise HTTPException(status_code=400, detail="summary obrigatório")

    use_summary_as_context = bool(payload.get("use_summary_as_context", True))
    retriever = None
    if not use_summary_as_context:
        try:
            retriever = retrievers_cache.get((user_id, collection_id)) or load_collection_retriever(user_id, collection_id)
        except Exception:
            retriever = None
            use_summary_as_context = True

    result = generate_oral_questions(
        retriever,
        summary,
        n_questions=count,
        use_objective_as_context=use_summary_as_context
    )
    items = _normalize_oral_payload(result)
    return items  # aqui retornamos a lista diretamente


@app.delete("/collections/{user_id}/{collection_id}")
def delete(user_id: str, collection_id: str):
    delete_collection(user_id, collection_id)
    retrievers_cache.pop((user_id, collection_id), None)
    return {"status": "deleted"}
