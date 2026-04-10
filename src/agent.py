from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # 1. Retrieve
        results = self.store.search(question, top_k=top_k)
        
        # 2. Augment
        context_parts = []
        for r in results:
            source = r["metadata"].get("source", "Unknown")
            context_parts.append(f"[Nguồn: {source}]\n{r['content']}")
            
        context_text = "\n---\n".join(context_parts)
        
        prompt = f"""Bạn là một trợ lý ảo thông vụ sinh viên của trường Đại học Bách Khoa Phương Nam (BKPN).
Hãy trả lời câu hỏi của sinh viên DỰA TRÊN ngữ cảnh được cung cấp bên dưới.

QUY TẮC:
1. Nếu thông tin không có trong văn bản, hãy nói: "Xin lỗi, hiện tại tôi không có thông tin về vấn đề này trong hệ thống dữ liệu học vụ."
2. Hãy trả lời một cách chuyên nghiệp, lịch sự và chính xác bằng tiếng Việt.
3. Không tự ý thêm thắt thông tin nằm ngoài nội dung được cung cấp.

NGỮ CẢNH:
{context_text}

CÂU HỎI: {question}
TRẢ LỜI:"""

        # 3. Generate
        return self.llm_fn(prompt)
