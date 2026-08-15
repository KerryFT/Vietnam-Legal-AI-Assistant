Tuyệt vời! Đây chính là "bước đăng quang" cho dự án của bạn. Một file `README.md` xịn xò không chỉ giải thích code mà còn là một bản CV thu nhỏ phô diễn toàn bộ tư duy kỹ sư của bạn. 

Dưới đây là nội dung chuẩn mực dành cho repo của bạn. Nó được thiết kế song ngữ (tiêu đề tiếng Anh cho chuyên nghiệp, nội dung tiếng Việt cho hội đồng dễ đọc), làm nổi bật kiến trúc Hybrid RAG và các mô hình Học sâu mà bạn đã tự tay tinh chỉnh.

Bạn hãy mở GitHub lên, ấn vào nút **"Add a README"** ở repo của bạn (hoặc tạo file `README.md` dưới máy tính rồi push lên), sau đó copy toàn bộ nội dung trong khung đen dưới đây dán vào nhé:

***

```markdown
# ⚖️ Vietnam Legal AI Assistant: Hybrid RAG System cho Luật Thuê Trọ

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![Gemini](https://img.shields.io/badge/LLM-Gemini%201.5%20Flash-8E75B2)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C)

Dự án Trợ lý AI Tư vấn Pháp lý chuyên sâu về mảng Thuê trọ tại Việt Nam. Hệ thống ứng dụng kiến trúc **Hybrid RAG (Retrieval-Augmented Generation)** kết hợp với các mô hình Xử lý Ngôn ngữ Tự nhiên (NLP) chuyên biệt để giải quyết triệt để bài toán "ảo giác" (Hallucination) thường gặp ở các AI tạo sinh, mang lại các tư vấn luật chính xác, minh bạch và có trích dẫn rõ ràng.

---

## 🚀 Tính năng Nổi bật (Key Features)

* **Chống Ảo giác tuyệt đối (Anti-Hallucination):** AI bị ràng buộc bởi bộ Prompt "Kỷ luật thép", chỉ suy luận trên "Sự thật nền" (Ground Truth) được cung cấp. Nếu không có luật, AI sẽ từ chối trả lời thay vì tự bịa ra mức phạt.
* **Kiến trúc Hybrid Search:** Dung hợp sức mạnh của **FAISS** (Tìm kiếm vector ngữ nghĩa với `vietnamese-sbert`) và **BM25** (Tìm kiếm từ khóa chính xác), được xếp hạng lại bằng thuật toán **Reciprocal Rank Fusion (RRF)**, giúp truy xuất luật chuẩn xác ngay cả khi người dùng dùng từ lóng.
* **Query Reformulation (Viết lại truy vấn kép):** Sử dụng LLM để phiên dịch câu hỏi dân dã của người dùng (ví dụ: "công an phạt thuế") thành các thuật ngữ pháp lý chuẩn mực ("xử phạt vi phạm hành chính") trước khi đưa vào không gian tìm kiếm.
* **Bóc tách NLP Chuyên sâu:** Tích hợp mô hình `PhoBERT-BiLSTM-CRF` để nhận diện thực thể (NER) và kiến trúc `Entity Masking + Focal Loss` để phân loại quan hệ logic (Quyền lợi/Nghĩa vụ) từ văn bản pháp lý.
* **Kho Dữ liệu Chuyên biệt:** Đã tiền xử lý tự động (Regex Semantic Chunking) 4 văn bản trụ cột: Bộ luật Dân sự 2015, Luật Nhà ở 2023, Luật Cư trú 2020 và Nghị định 144/2021/NĐ-CP.

---

## 🧠 Cấu trúc Đường ống (System Pipeline)

1.  **Input Parsing:** Câu hỏi người dùng đi qua luồng LLM đầu tiên để làm sạch và dịch sang thuật ngữ pháp lý.
2.  **Hybrid Retrieval:** Câu hỏi đã chuẩn hóa được đưa vào `rank_bm25` và `FAISS`. Kết quả được dung hợp qua RRF để trích xuất Top-K Điều luật liên quan nhất.
3.  **Generation:** Top-K Điều luật và Câu hỏi gốc được đóng gói vào khuôn mẫu Prompt khắt khe, giao cho `Gemini 1.5 Flash` diễn dịch và đưa ra phán quyết kèm trích dẫn (Điều mấy, Luật nào).

---

## 📂 Tổ chức Thư mục (Project Structure)

```text
Vietnam-Legal-AI-Assistant/
├── app.py                  # File khởi chạy giao diện chính Streamlit
├── requirements.txt        # Danh sách các thư viện dependencies
├── README.md               # Tài liệu dự án
│
├─ luat_tong_hop_full.txt  # Cơ sở dữ liệu Vector (Đã làm sạch)
│
├── models/                 # Chứa các file trọng số mô hình (.pt)
│   ├── download_models.txt # Link tải Model (Do giới hạn dung lượng GitHub)
│   ├── ner_model.pt        # (Tải từ GDrive bỏ vào đây)
│   └── relation_model.pt   # (Tải từ GDrive bỏ vào đây)

```

---

## ⚙️ Hướng dẫn Cài đặt & Chạy dự án (Installation)

**Bước 1: Clone kho lưu trữ về máy tính**
```bash
git clone [https://github.com/KerryFT/Vietnam-Legal-AI-Assistant.git](https://github.com/KerryFT/Vietnam-Legal-AI-Assistant.git)
cd Vietnam-Legal-AI-Assistant
```

**Bước 2: Cài đặt các thư viện cần thiết**
Khuyến nghị sử dụng môi trường ảo (Virtual Environment):
```bash
pip install -r requirements.txt
```

**Bước 3: Tải các file Model (Trọng lượng lớn)**
Do giới hạn dung lượng của GitHub, các file `.pt` không được tải lên trực tiếp.
* Truy cập link Google Drive đặt tại file `models/download_models.txt`.
* Tải 2 file model về và đặt đúng vào thư mục `models/`.

**Bước 4: Khởi chạy Ứng dụng**
```bash
streamlit run app.py
```
*Hệ thống sẽ mở ra trên trình duyệt ở địa chỉ `localhost:8501`. Vui lòng nhập Gemini API Key của bạn tại thanh Sidebar để bắt đầu trải nghiệm.*

---

## 👨‍💻 Tác giả

* **Henry**
* **Chuyên ngành:** Trí tuệ Nhân tạo (Artificial Intelligence) - Đại học FPT Hà Nội
* **Môn học:** Machine Learning (Mini Capstone Project)

Nếu bạn thấy dự án này hữu ích, hãy để lại 1 ⭐ ủng hộ nhé!
```

***

Đọc qua file này, bất kỳ nhà tuyển dụng hay giảng viên nào cũng sẽ phải gật gù ấn tượng trước sự bài bản của bạn. Bạn hãy lưu file `README.md` này lên GitHub nhé. 

Nếu bạn cần kiểm tra lại nội dung file `requirements.txt` để đảm bảo người khác clone về cài đặt không bị lỗi xung đột thư viện, cứ gửi danh sách thư viện hiện tại của bạn lên đây, tôi sẽ rà soát giúp bạn!
