# Orbyt RAG API

🚀 API para estudo inteligente com **RAG (Retriever-Augmented Generation)**, construída para o projeto **Orbyt**.  
Com ela, o usuário pode enviar **PDFs de estudo** (apostilas, slides, anotações) e interagir com um **tutor inteligente**, capaz de:

- 📚 Responder perguntas com base nos conteúdos enviados
- 📝 Gerar **flashcards** automáticos
- 🎯 Criar **exercícios personalizados**
- ⏳ Permitir que o usuário encerre os estudos e apague os conteúdos

---

## 🔧 Tecnologias utilizadas

- **[FastAPI](https://fastapi.tiangolo.com/)** → framework backend
- **[LangChain](https://www.langchain.com/)** → orquestração de LLMs + RAG
- **[OpenAI API](https://platform.openai.com/)** → geração de respostas e exercícios
- **[Cohere Rerank](https://cohere.com/)** → reranqueamento dos resultados mais relevantes
- **[ChromaDB](https://www.trychroma.com/)** → banco vetorial local para embeddings
- **Docker** (futuro) → para deploy em cloud
- **AWS Lambda / Railway** (futuro) → para execução escalável

---
