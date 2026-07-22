"""
Chat service for RAG-based conversations with Multi-Document Routing.
"""
from typing import Dict, List, Optional, Tuple
import logging
import json

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq

from backend.config import settings
from backend.models.schemas import Citation
from backend.services.vector_service import vector_service

logger = logging.getLogger(__name__)

NO_CONTEXT_RESPONSE = (
    "I could not find any relevant information in the uploaded document to "
    "answer your question. Please try rephrasing, or ask something that is "
    "covered in the document."
)

_CONTEXTUALIZE_SYSTEM = (
    "Given a chat history and the latest user question which might reference "
    "context in the chat history, formulate a standalone question that can be "
    "understood without the chat history. "
    "If the question is a greeting or entirely unrelated to the document, "
    "return it unchanged. "
    "Do NOT answer the question — only reformulate if needed."
)

_QA_SYSTEM = (
    "You are a precise document analysis assistant. "
    "You answer questions exclusively based on the context extracted from the "
    "uploaded document provided below.\n\n"
    "STRICT RULES — follow without exception:\n"
    "1. Use ONLY the information present in the context below.\n"
    "2. If the context does not contain enough information to answer the "
    "question, respond ONLY with:\n"
    "   'I could not find relevant information in the document to answer "
    "this question.'\n"
    "3. Do NOT use any prior knowledge, general knowledge, or information "
    "outside the provided context.\n"
    "4. Do NOT speculate, infer beyond what is stated, or fabricate details.\n"
    "5. When answering, reference the page or section where the information "
    "was found (e.g., 'According to page 3...').\n"
    "6. Use markdown formatting (bullet points, bold) where it aids clarity.\n\n"
    "Context from the document:\n"
    "──────────────────────────\n"
    "{context}\n"
    "──────────────────────────"
)

_ROUTER_SYSTEM = (
    "You are a document routing assistant. Your job is to select the most relevant document "
    "from a list of available documents based on the user's question.\n"
    "Available Documents:\n"
    "{documents}\n\n"
    "Output EXACTLY and ONLY the filename of the most relevant document. "
    "If none of the documents seem relevant, output 'None'. Do not output any other text or explanation."
)

class ChatService:
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
        self.router_llm = ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name=settings.LLM_MODEL,
            temperature=0.0, # Most deterministic
            max_tokens=50,
        )
        self.session_store: Dict[str, BaseChatMessageHistory] = {}
        self._chain_cache: Dict[str, RunnableWithMessageHistory] = {}
        logger.info("ChatService initialized with Multi-Document Routing")

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.session_store:
            self.session_store[session_id] = ChatMessageHistory()
            logger.info(f"Created new chat history for session {session_id}")
        return self.session_store[session_id]

    async def _route_question(self, prompt: str, filenames: List[str]) -> Optional[str]:
        if not filenames:
            return None
        if len(filenames) == 1:
            return filenames[0]
            
        logger.info(f"Routing question across {len(filenames)} documents: {filenames}")
        
        docs_str = "\n".join([f"- {f}" for f in filenames])
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", _ROUTER_SYSTEM),
            ("human", "{question}")
        ])
        
        chain = prompt_template | self.router_llm
        response = await chain.ainvoke({"documents": docs_str, "question": prompt})
        
        chosen = response.content.strip()
        logger.info(f"Router selected document: {chosen}")
        
        if chosen in filenames:
            return chosen
        return None

    def _build_chain(self, session_id: str, filename: str) -> RunnableWithMessageHistory:
        cache_key = f"{session_id}_{filename}"
        if cache_key in self._chain_cache:
            return self._chain_cache[cache_key]

        logger.info(f"Building RAG chain for session {session_id}, file {filename}")
        vectorstore = vector_service.get_vectorstore(session_id, filename)
        
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": settings.RETRIEVAL_K,
                "score_threshold": settings.SIMILARITY_SCORE_THRESHOLD,
            },
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", _CONTEXTUALIZE_SYSTEM),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", _QA_SYSTEM),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        document_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

        conversational_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        self._chain_cache[cache_key] = conversational_chain
        return conversational_chain

    async def chat(self, prompt: str, session_id: str) -> Tuple[str, Optional[List[Citation]], bool]:
        try:
            # 1. Routing
            filenames = vector_service.get_document_filenames(session_id)
            if not filenames:
                raise ValueError("No vectorstore exists for this session.")
                
            target_filename = await self._route_question(prompt, filenames)
            
            if not target_filename:
                logger.warning(f"Router returned None. Returning no-context response.")
                return NO_CONTEXT_RESPONSE, None, True
                
            # 2. Retrieval & Generation
            chain = self._build_chain(session_id, target_filename)

            response = await chain.ainvoke(
                {"input": prompt},
                config={"configurable": {"session_id": session_id}},
            )

            retrieved_docs = response.get("context", [])

            if not retrieved_docs:
                return NO_CONTEXT_RESPONSE, None, True

            # 3. Score Lookup
            vectorstore = vector_service.get_vectorstore(session_id, target_filename)
            scored_docs = vectorstore.similarity_search_with_relevance_scores(
                prompt,
                k=settings.RETRIEVAL_K,
                score_threshold=settings.SIMILARITY_SCORE_THRESHOLD,
            )
            score_map = {doc.page_content[:80]: score for doc, score in scored_docs}

            # 4. Citation Building
            citations: List[Citation] = []
            for doc in retrieved_docs:
                meta = doc.metadata
                citations.append(
                    Citation(
                        chunk_id=meta.get(
                            "chunk_id",
                            f"{session_id}_p{meta.get('page', 0)}_cUnknown",
                        ),
                        filename=meta.get("filename", meta.get("source", "Unknown")),
                        page=int(meta.get("page", 0)),
                        chunk_index=int(meta.get("chunk_index", 0)),
                        total_chunks=int(meta.get("total_chunks", 0)),
                        content_preview=doc.page_content[:200].strip(),
                        score=score_map.get(doc.page_content[:80]),
                    )
                )

            return response["answer"], citations, False

        except ValueError as e:
            logger.error(f"Vectorstore error: {e}")
            raise
        except Exception as e:
            logger.error(f"Chat processing failed: {e}")
            raise Exception(f"Failed to generate response: {e}")

    def clear_history(self, session_id: str) -> None:
        if session_id in self.session_store:
            self.session_store[session_id] = ChatMessageHistory()

    def delete_session(self, session_id: str) -> None:
        self.session_store.pop(session_id, None)
        # Clear cache for any chain associated with this session
        keys_to_delete = [k for k in self._chain_cache.keys() if k.startswith(f"{session_id}_")]
        for k in keys_to_delete:
            self._chain_cache.pop(k, None)

    def get_message_count(self, session_id: str) -> int:
        if session_id not in self.session_store:
            return 0
        return len(self.session_store[session_id].messages)

    def session_exists(self, session_id: str) -> bool:
        return session_id in self.session_store


chat_service = ChatService()
