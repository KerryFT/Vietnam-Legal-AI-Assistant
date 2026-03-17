import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
import os
from rank_bm25 import BM25Okapi
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import re
# ==========================================
# 1. CẤU HÌNH GIAO DIỆN (BẮT BUỘC PHẢI Ở DÒNG ĐẦU TIÊN)
# ==========================================
st.set_page_config(page_title="AI Pháp lý ĐH FPT", page_icon="⚖️", layout="wide")

# ==========================================
# 2. ĐỊNH NGHĨA CHÍNH XÁC CẤU TRÚC NÃO BỘ AI
# ==========================================
# Cập nhật Class NER khớp với kiến trúc file .pt của bạn (BiLSTM + 11 Labels)
class PhoBERT_NER(nn.Module):
    def __init__(self, num_labels=11):
        super(PhoBERT_NER, self).__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
        # Phục dựng lại lớp BiLSTM dựa trên thông số báo lỗi (512 chiều)
        self.bilstm = nn.LSTM(
            input_size=self.phobert.config.hidden_size, # 768
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Linear(512, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        lstm_output, _ = self.bilstm(sequence_output)
        logits = self.classifier(lstm_output)
        return logits

# Class Relation Model (Giữ nguyên vì đã chuẩn)
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        fl = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None: fl = self.alpha[targets] * fl
        return fl.mean()

class RelationModel(nn.Module):
    def __init__(self, model_name, class_weights=None):
        super().__init__()
        self.phobert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.phobert.config.hidden_size, 3)
        self.loss_fn = FocalLoss(alpha=class_weights)
    def forward(self, input_ids, attention_mask, labels=None):
        out = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(out.last_hidden_state[:, 0, :])
        return {"logits": logits}

# ==========================================
# 3. TẢI CÁC MÔ HÌNH VÀO RAM CỦA HỆ THỐNG
# ==========================================
@st.cache_resource
def load_all_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    tokenizer.add_special_tokens({'additional_special_tokens': ['<SUBJ>', '<OBJ>']})
    
    # Load NER (Dùng strict=False để bỏ qua lớp CRF không cần thiết lúc chạy Web)
    ner_model = PhoBERT_NER(num_labels=11).to(device)
    ner_path = os.path.join(BASE_DIR, "PhoBERT_BiLSTM_CRF_weights.pt")
    if os.path.exists(ner_path):
        ner_model.load_state_dict(torch.load(ner_path, map_location=device), strict=False)
        ner_model.eval()
    
    # Load Relation
    re_model = RelationModel("vinai/phobert-base-v2").to(device)
    re_model.phobert.resize_token_embeddings(len(tokenizer))
    re_path = os.path.join(BASE_DIR, "PhoBERT_Relation_Super_Final.pt")
    if os.path.exists(re_path):
        re_model.load_state_dict(torch.load(re_path, map_location=device))
        re_model.eval()
        
    # Load SBERT & FAISS
    try:
        embedder = SentenceTransformer('keepitreal/vietnamese-sbert')
    except:
        embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
     # Thêm import re ở đầu file

    # BƯỚC 1: Đọc toàn bộ file Luật
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(BASE_DIR, "luat_tong_hop_full.txt")
    
    try:
        with open(txt_path, "r", encoding="utf-8") as file:
            content = file.read() # Đọc nguyên cả khối văn bản lớn
            
            # Xóa bỏ các dòng rác (bạn có thể tự thêm các câu rác vào đây)
            content = content.replace("Quốc hội ban hành Luật Kinh doanh bất động sản.", "")
            
            # Dùng Regex để cắt văn bản mỗi khi gặp cụm từ bắt đầu bằng "Điều "
            # (?=Điều \d+) nghĩa là: Cắt ở vị trí ngay trước chữ "Điều " có kèm số
            chunks = re.split(r'(?=Điều \d+)', content)
            
            # Gom tất cả các Enter (\n) bên trong 1 Điều luật thành dấu cách
            luat_database = [chunk.replace('\n', ' ').strip() for chunk in chunks if len(chunk.strip()) > 10]
            
    except Exception as e:
        st.error(f"⚠️ Lỗi đọc file Luật: {e}")
        luat_database = ["Dữ liệu luật trống. Vui lòng kiểm tra lại file."]

    luat_vectors = embedder.encode(luat_database)
    faiss.normalize_L2(luat_vectors)
    index = faiss.IndexFlatIP(luat_vectors.shape[1])
    index.add(luat_vectors)
    
    return tokenizer, ner_model, re_model, embedder, index, luat_database, device

tokenizer, ner_model, re_model, embedder, index, luat_database, device = load_all_models()

# --- BỔ SUNG KHÚC NÀY ĐỂ NẠP BM25 ---
def tokenize_text(text):
    # Hàm dọn dẹp dấu câu và cắt từ để BM25 dễ đếm Keyword
    clean_text = re.sub(r'[^\w\s]', '', text).lower()
    return clean_text.split()

# Băm toàn bộ Luật ra thành các từ khóa
tokenized_corpus = [tokenize_text(doc) for doc in luat_database]
bm25 = BM25Okapi(tokenized_corpus)
# ==========================================
# 4. LUỒNG XỬ LÝ GIAO DIỆN CHATBOT (RAG PIPELINE)
# ==========================================
st.title("⚖️ Trợ lý AI Tư vấn Luật Thuê Trọ")
st.caption("Đồ án AI - Đại học FPT")

with st.sidebar:
    st.header("⚙️ Cấu hình")
    api_key = st.text_input("Nhập Google Gemini API Key:", type="password")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Chào bạn, hệ thống AI Pháp lý đã sẵn sàng!"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("Nhập câu hỏi hoặc hợp đồng..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    with st.chat_message("assistant"):
        if not api_key:
            st.error("⚠️ Vui lòng nhập Gemini API Key!")
            st.stop()
            
        genai.configure(api_key=api_key)
        llm = genai.GenerativeModel('gemini-2.5-flash')
            
        with st.status("🔍 Đang chạy luồng Hybrid RAG...", expanded=True) as status:
            
            # --- 1. REFORMULATE QUERY ---
            st.write("🧠 Dịch sang ngôn ngữ pháp lý...")
            reformulate_prompt = f"""Tóm tắt câu hỏi sau thành 1 câu truy vấn chứa các từ khóa pháp lý cốt lõi. 
            Nếu người dùng hỏi về "phạt", "công an", BẮT BUỘC phải thêm cụm từ: "xử phạt vi phạm hành chính về đăng ký cư trú, tạm trú".
            Quy đổi từ vựng: "chủ nhà" -> "chủ hộ, người cho thuê, cơ sở lưu trú"; "người thuê" -> "công dân".
            TUYỆT ĐỐI CHỈ TRẢ VỀ CÂU TRUY VẤN.
            Câu gốc: "{user_query}"
            Truy vấn pháp lý:"""
            reformulated_query = llm.generate_content(reformulate_prompt).text.strip()
            st.info(f"🔄 Truy vấn chuẩn: {reformulated_query}")
            
            # --- 2. TÌM KIẾM VECTOR (FAISS) ---
            st.write("⚡ FAISS: Quét ngữ nghĩa...")
            q_vec = embedder.encode([reformulated_query])
            faiss.normalize_L2(q_vec)
            distances, faiss_indices = index.search(q_vec, k=10) # Lấy top 10
            faiss_indices = faiss_indices[0]
            
            # --- 3. TÌM KIẾM TỪ KHÓA (BM25) ---
            st.write("⚡ BM25: Bắt chính xác Keyword...")
            tokenized_query = tokenize_text(reformulated_query)
            bm25_scores = bm25.get_scores(tokenized_query)
            bm25_indices = np.argsort(bm25_scores)[::-1][:10] # Lấy top 10
            
            # --- 4. DUNG HỢP KẾT QUẢ (RECIPROCAL RANK FUSION - RRF) ---
            st.write("⚖️ RRF: Dung hợp kết quả Hybrid...")
            rrf_scores = {}
            # Tính điểm RRF cho FAISS
            for rank, doc_idx in enumerate(faiss_indices):
                rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1 / (60 + rank + 1)
            # Tính điểm RRF cho BM25
            for rank, doc_idx in enumerate(bm25_indices):
                rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1 / (60 + rank + 1)
                

            # Lấy Top 8 thay vì Top 5 để mở rộng lưới quét
            sorted_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:8]
            luat_tim_duoc = "\n\n".join([luat_database[idx] for idx in sorted_indices])
        
            status.update(label="✅ Đã tìm thấy Luật chuẩn xác!", state="complete", expanded=False)
            
        # --- 5. SINH CÂU TRẢ LỜI VỚI PROMPT "KỶ LUẬT THÉP" ---
        final_prompt = f"""Bạn là Luật sư AI. Dưới đây là các Điều luật được hệ thống tìm kiếm:
        
[NGỮ CẢNH BẮT ĐẦU]
{luat_tim_duoc}
[NGỮ CẢNH KẾT THÚC]

Câu hỏi của người dùng: "{user_query}"

QUY TẮC TRẢ LỜI NGHIÊM NGẶT:
1. ĐI THẲNG VÀO VẤN ĐỀ: Trả lời trực tiếp câu hỏi ngay ở câu đầu tiên. Ai đúng, ai sai, ai bị phạt?
2. BẮT BUỘC TRÍCH DẪN NGUỒN: Mọi lời tư vấn, kết luận ĐÚNG/SAI, AI BỊ PHẠT đều PHẢI đính kèm trích dẫn gốc ngay trong câu. (Ví dụ: "Căn cứ theo Điều 27 Luật Cư trú 2020..."). Tuyệt đối không được nói chung chung là "theo quy định của pháp luật".
3. CHỌN LỌC: Bỏ qua hoàn toàn các điều luật trong Ngữ cảnh nếu nó không liên quan đến tình huống.
4. TÍNH CHÍNH XÁC: Nếu ngữ cảnh KHÔNG CÓ quy định cụ thể trả lời được câu hỏi (ví dụ không thấy mức phạt), hãy nói: "Dữ liệu pháp lý hiện tại của tôi chưa có quy định cụ thể cho chi tiết này". KHÔNG TỰ BỊA RA THÔNG TIN.
5. LƯU Ý TỪ VỰNG: Trong luật, "người thuê" thường được gọi là "công dân", "bên thuê"; "chủ nhà" được gọi là "chủ hộ", "người cho thuê", "cơ sở lưu trú". Hãy linh hoạt đối chiếu để trả lời.
"""
        response = llm.generate_content(final_prompt)
        
        st.markdown(response.text)
        st.session_state.messages.append({"role": "assistant", "content": response.text})