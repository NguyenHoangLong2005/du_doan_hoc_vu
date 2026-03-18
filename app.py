import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re

# ==============================
# 1. CẤU HÌNH TRANG & LOAD MODEL
# ==============================
st.set_page_config(
    page_title="Dự đoán Cảnh báo học vụ",
    page_icon="🎓",
    layout="centered"
)

@st.cache_resource
def load_model_assets():
    try:
        with open("model_assets.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("❌ Không tìm thấy file 'model_assets.pkl'. Vui lòng upload file này lên cùng thư mục với app.py")
        return None

assets = load_model_assets()

# ==============================
# 2. CÁC HÀM HỖ TRỢ (UTILITIES)
# ==============================
def clean_text(text, teencode_dict):
    if not text: return ""
    text = text.lower()
    # Thay thế teencode
    words = [teencode_dict.get(w, w) for w in text.split()]
    # Loại bỏ ký tự đặc biệt
    text = re.sub(r"[^a-zA-Zà-ỹ\s]", " ", " ".join(words))
    # Loại bỏ khoảng trắng thừa
    return re.sub(r"\s+", " ", text).strip()

# ==============================
# 3. GIAO DIỆN NGƯỜI DÙNG (UI)
# ==============================
st.title("🎓 Hệ thống Dự đoán Cảnh báo học vụ")
st.markdown("---")

if assets:
    st.subheader("📝 Nhập thông tin sinh viên")
    
    col1, col2 = st.columns(2)
    
    with col1:
        essay = st.text_area(
            "Bài luận cá nhân (Personal Essay):", 
            placeholder="Ví dụ: Em hứa sẽ cố gắng học tập tốt trong học kỳ tới...",
            height=150
        )
    
    with col2:
        notes = st.text_area(
            "Ghi chú của cố vấn (Advisor Notes):", 
            placeholder="Ví dụ: Sinh viên thường xuyên vắng mặt không lý do...",
            height=150
        )

    st.markdown("---")
    attendance = st.slider("Điểm chuyên cần trung bình (0.0 - 1.0):", 0.0, 1.0, 0.8)

    # Nút dự đoán
    if st.button("🚀 Phân tích nguy cơ", use_container_width=True):
        # TIỀN XỬ LÝ DỮ LIỆU ĐẦU VÀO
        clean_e = clean_text(essay, assets['teencode'])
        clean_n = clean_text(notes, assets['teencode'])
        
        # 1. Tạo các tính năng số (Numerical Features)
        essay_word_count = len(clean_e.split())
        advisor_note_words = len(clean_n.split())
        
        # 2. Tạo tính năng TF-IDF cho văn bản
        tfidf_features = assets['tfidf'].transform([clean_e]).toarray()
        tfidf_df = pd.DataFrame(tfidf_features, columns=assets['tfidf'].get_feature_names_out())
        
        # 3. Tạo DataFrame đầu vào đúng cấu trúc lúc Train
        # Khởi tạo row với toàn giá trị 0
        input_row = pd.DataFrame(0.0, index=[0], columns=assets['feature_names'])
        
        # Gán giá trị cho các cột cơ bản
        input_row['essay_word_count'] = float(essay_word_count)
        input_row['advisor_note_words'] = float(advisor_note_words)
        input_row['attendance_mean'] = float(attendance)
        
        # Gán giá trị cho các cột TF-IDF (chỉ gán nếu cột đó tồn tại trong feature_names)
        for col in tfidf_df.columns:
            if col in input_row.columns:
                input_row[col] = tfidf_df[col].values[0]

        # THỰC HIỆN DỰ ĐOÁN
        prediction = assets['model'].predict(input_row)[0]
        probability = assets['model'].predict_proba(input_row)[0][1]

        # HIỂN THỊ KẾT QUẢ
        st.markdown("### Kết quả dự đoán:")
        if prediction == 1:
            st.error(f"⚠️ **NGUY CƠ CAO**: Sinh viên này có khả năng bị Cảnh báo học vụ.")
            st.progress(probability)
            st.write(f"Xác suất rủi ro: **{probability*100:.2f}%**")
        else:
            st.success(f"✅ **AN TOÀN**: Sinh viên hiện tại ở trạng thái Bình thường.")
            st.progress(float(probability))
            st.write(f"Xác suất rủi ro: **{probability*100:.2f}%**")

else:
    st.info("💡 Vui lòng đảm bảo file `model_assets.pkl` đã được đặt cạnh file `app.py` để ứng dụng hoạt động.")