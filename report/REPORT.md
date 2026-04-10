# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Thành Đại Khánh
**Mã sinh viên:** 2A202600404
**Ngày nộp:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Nghĩa là hai vector trong không gian đa chiều đang chỉ về cùng một hướng, thể hiện rằng hai đoạn văn bản đó có sự tương đồng rất cao về mặt ngữ nghĩa hoặc ngữ cảnh.

**Ví dụ HIGH similarity:**
- Sentence A: "Quy trình xét học bổng cho sinh viên năm nhất."
- Sentence B: "Hướng dẫn thủ tục nhận hỗ trợ học phí cho tân sinh viên."
- Tại sao tương đồng: Mặc dù dùng từ ngữ khác nhau nhưng cả hai đều nói về chính sách hỗ trợ tài chính cho người mới vào trường.

**Ví dụ LOW similarity:**
- Sentence A: "Giờ đóng cửa của thư viện là 20:00."
- Sentence B: "Món phở bò là đặc sản của Hà Nội."
- Tại sao khác: Một bên nói về quy định học vụ, một bên nói về ẩm thực, không có mối liên hệ nào.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Vì Cosine Similarity tập trung vào **góc** giữa các vector thay vì độ dài. Điều này giúp hệ thống so sánh được sự tương đồng về nội dung mà không bị ảnh hưởng bởi việc một văn bản dài hàng nghìn chữ hay chỉ ngắn vài câu.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* 
> - Bước nhảy (Step) = chunk_size - overlap = 500 - 50 = 450 ký tự.
> - Số lượng chunk = ceil(10,000 / 450) ≈ 22.22.
> *Đáp án:* 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Khi overlap tăng lên 100, bước nhảy còn 400 ký tự, số lượng chunk tăng lên 25. Chúng ta muốn overlap nhiều hơn để đảm bảo ngữ cảnh ở đoạn cuối của chunk trước không bị cắt đứt đột ngột, giúp AI hiểu được sự liên kết giữa các đoạn.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Hệ thống hỗ trợ tra cứu thông tin học vụ và đời sống sinh viên (Student Academic Support System).

**Tại sao nhóm chọn domain này?**
> Nhóm chọn bộ dữ liệu về học đường vì tính thiết thực và gần gũi với người dùng. Các tài liệu này chứa nhiều thông tin chi tiết về quy định, mốc thời gian và hướng dẫn quy trình, rất thích hợp để kiểm tra độ chính xác của việc truy xuất thông tin cụ thể (Fact-checking).

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | 01_faq_hoc_vu.txt | BKPN | ~1,500 | academic, faq |
| 2 | 02_quy_che_sinh_vien_ktx.txt | BKPN | ~2,200 | dorm, rules |
| 3 | 03_huong_dan_hoc_bong.txt | BKPN | ~1,800 | finance, scholarship |
| 4 | 04_thuc_tap_khoa_luan.txt | BKPN | ~2,500 | process, academic |
| 5 | 05_thu_vien_ho_tro.txt | BKPN | ~2,000 | support, library |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `source` | string | "data/02_ktx.txt" | Giúp AI biết chính xác thông tin được trích dẫn từ file nào. |
| `doc_id` | string | "03_hoc_bong" | Dễ dàng quản lý và xóa toàn bộ chunk thuộc về một tài liệu khi cần update. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| 03_hoc_bong | best baseline (Sentence) | 34 | 403.0 | Trung bình (Dễ mất ngữ cảnh) |
| 03_hoc_bong | **của tôi (Recursive)** | 97 | 140.0 | **Cao (Tìm đúng mục chi tiết)** |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | Recursive + OpenAI | 10 | Hiểu ngữ nghĩa, chính xác tuyệt đối | Cần Internet và API Key |
| Nguyễn Tiến Thành | FixedSize | 6 | Tốc độ xử lý nhanh, đơn giản | Cắt văn bản vụn, mất nghĩa |
| Hà Hưng Phước | Sentence | 7.5 | Giữ câu trọn vẹn, dễ đọc | Bị giới hạn bởi cách ngắt câu |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> RecursiveChunker là tốt nhất vì tài liệu quy chế/FAQ trường học có cấu trúc phân tầng. Việc ngắt theo phân đoạn logic (xuống dòng) giúp mỗi chunk là một đơn vị kiến thức độc lập, hỗ trợ tốt nhất cho việc search ngữ nghĩa.

### Strategy Của Tôi

**Loại:** `RecursiveChunker` (chunk_size=600).

**Mô tả cách hoạt động:**
> Algorithm sẽ ưu tiên tìm các dấu ngắt lớn như `\n\n` (ngắt đoạn) để chia nhỏ văn bản trước. Nếu đoạn vẫn dài, nó sẽ tìm đến `\n` (xuống dòng) và cuối cùng là `. ` (ngắt câu). Điều này đảm bảo mỗi chunk là một khối thông tin trọn vẹn về ý nghĩa.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Vì tài liệu quy chế trường học luôn được trình bày theo dạng các Điều, Khoản (có dấu xuống dòng). `RecursiveChunker` sẽ giúp giữ nguyên một Điều hoặc một Khoản trong cùng một chunk thay vì xẻ đôi nó.

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng Regex `(?<=[.!?])(?:\s+|\n)` để tìm các dấu kết thúc câu kèm theo khoảng trắng hoặc xuống dòng. Cách này xử lý được trường hợp câu kết thúc ngay cuối dòng mà không có dấu cách.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Sử dụng đệ quy (Recursion) để duyệt qua danh sách các separator ưu tiên. Nếu một đoạn văn bản sau khi split vẫn vượt quá `chunk_size`, hàm sẽ tự gọi lại chính nó với separator tiếp theo trong danh sách cho đến khi đạt kích thước mong muốn.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Lưu trữ dưới dạng một danh sách các Dictionary chứa `embedding` (vector) và `metadata`. Khi Search, hệ thống tính tích vô hướng (Dot Product) giữa vector câu hỏi và tất cả các vector trong store, sau đó sắp xếp giảm dần.

**`search_with_filter` + `delete_document`** — approach:
> Thực hiện kỹ thuật **Pre-filtering**: Lọc những bản ghi thỏa mãn metadata trước khi tính similarity để tiết kiệm tài nguyên. Xóa tài liệu bằng cách lọc bỏ các chunk có `doc_id` tương ứng khỏi list.

### KnowledgeBaseAgent

**`answer`** — approach:
> Xây dựng Prompt tiếng Việt chuyên nghiệp với cấu trúc Persona (Trợ lý trường BKPN). Inject nội dung Context vào giữa các nhãn `[Nguồn: ...]` để AI có thể trích dẫn nguồn chính xác trong câu trả lời.

### Test Results

```text

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED                                                [  2%] 
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                                                         [  4%] 
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                                                  [  7%] 
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                                                   [  9%] 
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                                                        [ 11%] 
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED                                        [ 14%] 
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED                                              [ 16%] 
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED                                               [ 19%] 
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED                                             [ 21%] 
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                                                               [ 23%] 
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED                                               [ 26%] 
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                                                          [ 28%] 
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                                                      [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                                                                [ 33%] 
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED                                       [ 35%] 
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED                                           [ 38%] 
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED                                     [ 40%] 
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED                                           [ 42%] 
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                                                               [ 45%] 
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                                                 [ 47%] 
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                                                   [ 50%] 
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                                                         [ 52%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                                              [ 54%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                                                [ 57%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED                                    [ 59%] 
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                                                 [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                                          [ 64%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                                         [ 66%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                                                    [ 69%] 
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                                                [ 71%] 
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED                                           [ 73%] 
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED                                               [ 76%] 
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                                                     [ 78%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED                                               [ 80%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED                            [ 83%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED                                          [ 85%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED                                         [ 88%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED                             [ 90%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED                                        [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED                                 [ 95%] 
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED                       [ 97%] 
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED                           [100%] 

=========================================================== 42 passed in 0.08s ============================================================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Chính sách bảo mật | Quy định an toàn | High | 0.81 | Đúng |
| 2 | Học bổng sinh viên | Hỗ trợ tài chính | High | 0.78 | Đúng |
| 3 | Thư viện | Phòng đào tạo | Low | 0.12 | Đúng |
| 4 | Mật khẩu | Học bổng | Low | 0.05 | Đúng |
| 5 | Thủ tục mượn sách | Đăng ký thẻ | High | 0.74 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Điểm score tăng vọt từ 0.3 lên 0.8 sau khi tôi đổi sang OpenAI Embeddings. Điều này chứng minh embeddings xịn không chỉ so sánh mặt chữ mà thực sự mã hóa được cái "hồn" (ngữ nghĩa) của từ ngữ vào không gian vector.

---

## 6. Results — Cá nhân (10 điểm)

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Điều kiện xét học bổng? | GPA ≥ 2.0, Rèn luyện ≥ 50 |
| 2 | Giờ đóng cửa KTX? | 23:00 hàng ngày |
| 3 | Thủ tục mượn sách? | Kích hoạt thẻ tại quầy lễ tân |
| 4 | Thời gian nộp khóa luận? | Trong vòng 2 tuần sau thực tập |
| 5 | Điều kiện xét tốt nghiệp? | Hoàn thành tín chỉ và chuẩn đầu ra |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Học bổng | A. Học bổng định kỳ... | 0.825 | Có | GPA từ 2.0 trở lên... |
| 2 | Giờ KTX | Điều 9. Giờ giấc ra vào... | 0.762 | Có | KTX đóng cửa lúc 23:00... |
| 3 | Mượn sách | 1.2. Thẻ thư viện... | 0.741 | Có | Xuất trình thẻ sinh viên tại quầy... |
| 4 | Khóa luận | Sau khi kết thúc thực tập... | 0.789 | Có | Nộp báo cáo trong vòng 2 tuần... |
| 5 | Tốt nghiệp | Q: Điều kiện để được xét tốt nghiệp | 0.814 | Có | Hoàn thành toàn bộ số tín chỉ... |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Tôi học được cách các bạn gán metadata như `priority` cho các tài liệu quan trọng để AI ưu tiên đọc những nguồn đó trước khi trả lời.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Tôi thấy một nhóm bạn đã tích hợp được cả file PDF thay vì chỉ file thô .txt, điều này mở rộng khả năng ứng dụng thực tế rất lớn.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ thêm bước "Vệ sinh dữ liệu" (Data Cleaning) kĩ hơn, loại bỏ các ký tự đặc biệt thừa để giúp việc Embedding diễn ra chính xác hơn nữa.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |

---
