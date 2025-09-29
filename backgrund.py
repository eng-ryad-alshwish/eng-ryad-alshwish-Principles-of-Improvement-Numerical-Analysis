# welcome_app.py
import streamlit as st
import base64
from pathlib import Path

# ---------- إعداد الصفحة ----------
st.set_page_config(
    page_title="Numerical Analysis Project",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- دالة لتحويل الصورة إلى base64 ----------
def get_base64_of_bin_file(bin_file):
    BASE_DIR = Path(__file__).parent  # مسار ملف current file
    file_path = BASE_DIR / bin_file

    if not file_path.exists():
        st.warning(f"⚠️ File not found: {file_path}")
        return None

    with open(file_path, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# ---------- قراءة الصورة وتحويلها ----------
img_base64 = get_base64_of_bin_file("6.jfif")  # ضع صورتك هنا

if img_base64:
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: 100% 100%;
        background-position: center;
        background-repeat: no-repeat;
        min-height: 100vh;
    }}
    [data-testid="stHeader"], [data-testid="stToolbar"] {{
        background: rgba(0,0,0,0);
    }}
    .overlay {{
        background: rgba(180, 170, 150, 0.75);
        padding: 50px;
        border-radius: 20px;
        text-align: center;
        max-width: 800px;
        margin: auto;
        top: 50%;
        transform: translateY(50%);
    }}
    .start-btn {{
        font-size: 1.3em;
        padding: 0.7em 2.5em;
        margin-top: 2em;
        background-color: #ff7f50;
        color: white;
        border-radius: 15px;
        border: none;
        transition: all 0.3s ease;
    }}
    .start-btn:hover {{
        background-color: #ff4500;
        transform: scale(1.05);
        cursor: pointer;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# ---------- محتوى الواجهة ----------
st.markdown(
    """
    <div class="overlay">
        <h1 style="font-size: 3em; color: #2C3E50;"> أهلاً بك في مشروع التحليل العددي</h1>
        <h3 style="color: #555;">حل المعادلات غير الخطية</h3>
        <p style="font-size: 1.2em; color: #333; margin-top: 20px;">
            استكشف طرقًا عددية لإيجاد حلول تقريبية للمعادلات غير الخطية باستخدام:
            <br>• Bisection
            <br>• Newton–Raphson
            <br>• Secant
        </p>
        <button class="start-btn" onclick="window.location.href='app.py'">ابدأ المشروع</button>
    </div>
    """,
    unsafe_allow_html=True
)
