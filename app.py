import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="RFM Intelligence", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "retail_dw.db"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Noto+Sans+Thai:wght@400;500;600;700&display=swap');
    :root { --ink: #18212b; --muted: #45545f; --line: #d9e1e4; --mint: #dff5eb; --green: #167a5a; --paper: #f8faf9; }
    .stApp { background: var(--paper); color: var(--ink); font-family: 'Noto Sans Thai', 'DM Sans', sans-serif; }
    [data-testid="stSidebar"] { background: #17252a; border-right: 0; }
    [data-testid="stSidebar"] * { color: #eaf5f1; }
    [data-testid="stSidebar"] .stRadio label { padding: .55rem .7rem; border-radius: 10px; font-size: .88rem; }
    [data-testid="stSidebar"] .stRadio label:hover { background: rgba(223,245,235,.1); }
    .brand { padding: 1rem 0 2.2rem; }
    .brand-mark { color: #9ee5c8; font-size: 1.4rem; font-weight: 700; letter-spacing: .08em; }
    .brand-sub { color: #9cafb2; font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; margin-top: .25rem; }
    .eyebrow { color: var(--green); font-size: .74rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; margin-bottom: .5rem; }
    h1, h2, h3, p, label, .stMarkdown, [data-testid="stMetric"] { font-family: 'Noto Sans Thai', 'DM Sans', sans-serif; }
    h1 { color: var(--ink) !important; font-size: clamp(2rem, 4vw, 3.35rem) !important; font-weight: 700 !important; letter-spacing: -.02em; line-height: 1.2 !important; margin-bottom: .55rem !important; }
    .hero-copy { color: var(--muted); font-size: 1.05rem; font-weight: 500; max-width: 38rem; line-height: 1.75; }
    .hero { padding: 1.2rem 0 1.8rem; }
    .section-title { font-size: 1.15rem; font-weight: 700; margin: 1.8rem 0 .8rem; }
    .section-note { color: var(--muted); font-size: .92rem; font-weight: 500; margin-top: -.55rem; margin-bottom: 1rem; }
    [data-testid="stMetric"] { background: white; border: 1px solid var(--line); border-radius: 14px; padding: 1.05rem 1.2rem; box-shadow: 0 8px 25px rgba(24,33,43,.04); }
    [data-testid="stMetricLabel"] { color: var(--muted); font-size: .84rem; font-weight: 600; white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
    [data-testid="stMetricValue"] { color: var(--ink); font-size: clamp(1.05rem, 2vw, 1.65rem); line-height: 1.15; white-space: normal !important; overflow-wrap: anywhere; }
    .insight { background: var(--mint); border-radius: 14px; padding: 1.1rem 1.25rem; color: #1d4437; min-height: 5rem; }
    .insight strong { display: block; font-size: 1rem; margin-bottom: .25rem; }
    .prediction-intro { display: flex; align-items: flex-end; justify-content: space-between; gap: 2rem; margin: .5rem 0 1.5rem; }
    .model-status { background: #e4f6ee; border: 1px solid #bde5d3; border-radius: 999px; color: #176b50; font-size: .78rem; font-weight: 700; padding: .45rem .8rem; white-space: nowrap; }
    .form-panel, .result-panel { background: white; border: 1px solid var(--line); border-radius: 18px; padding: 1.35rem; box-shadow: 0 12px 32px rgba(24,33,43,.06); }
    .form-panel-title { color: var(--ink); font-size: 1.05rem; font-weight: 700; margin-bottom: .2rem; }
    .form-panel-note { color: var(--muted); font-size: .83rem; margin-bottom: 1.15rem; }
    .input-label { color: var(--ink); font-size: .86rem; font-weight: 700; margin: .2rem 0 .25rem; }
    .input-help { color: var(--muted); font-size: .74rem; margin: -.3rem 0 .35rem; }
    .preview-kicker { color: var(--green); font-size: .73rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .preview-title { color: var(--ink); font-size: 1.3rem; font-weight: 700; margin: .2rem 0 1rem; }
    .preview-row { border-top: 1px solid var(--line); display: flex; justify-content: space-between; padding: .75rem 0; }
    .preview-row span:first-child { color: var(--muted); font-size: .82rem; }
    .preview-row span:last-child { color: var(--ink); font-size: .9rem; font-weight: 700; }
    .result-banner { background: #dff7e8; border: 1px solid #a9dfbb; border-radius: 14px; color: #135d3f; margin-top: 1rem; padding: 1rem 1.15rem; }
    .result-banner strong { display: block; font-size: 1.05rem; margin-bottom: .2rem; }
    .stForm { border: 0 !important; padding: 0 !important; }
    [data-testid="stFormSubmitButton"] { margin-top: .85rem; }
    @media (max-width: 800px) { .prediction-intro { align-items: flex-start; flex-direction: column; gap: .7rem; } .form-panel, .result-panel { padding: 1rem; } }
    .stButton > button, .stFormSubmitButton > button { background: var(--green); border: 0; border-radius: 10px; color: white; font-weight: 600; min-height: 2.7rem; transition: all .2s ease; }
    .stButton > button:hover, .stFormSubmitButton > button:hover { background: #105c44; transform: translateY(-1px); box-shadow: 0 6px 16px rgba(22,122,90,.2); }
    .stTextInput input, .stNumberInput input, [data-baseweb="select"] > div { border-radius: 10px; border-color: var(--line); background: white; color: var(--ink) !important; font-weight: 600; }
    .stTextInput input::placeholder, .stNumberInput input::placeholder { color: #71808a !important; opacity: 1 !important; }
    .stNumberInput input:focus, .stTextInput input:focus { color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important; }
    [data-baseweb="select"] * { color: var(--ink) !important; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
    .footer { color: #91a0a4; font-size: .72rem; border-top: 1px solid var(--line); margin-top: 3rem; padding: 1rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_db_connection():
    return sqlite3.connect(DB_PATH)


@st.cache_resource
def load_ml_assets():
    return joblib.load(BASE_DIR / "rfm_model.joblib"), joblib.load(BASE_DIR / "scaler.joblib")


def load_rfm_data():
    if not DB_PATH.exists():
        return None
    with get_db_connection() as conn:
        return pd.read_sql_query("SELECT * FROM dim_customer_rfm", conn)


model, scaler = load_ml_assets()
rfm = load_rfm_data()

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">RFM ◈</div><div class="brand-sub">Customer intelligence</div></div>', unsafe_allow_html=True)
    st.caption("WORKSPACE")
    menu = st.radio("ไปยังหน้าที่ต้องการ", [
        "ภาพรวมธุรกิจ",
        "สำรวจข้อมูลลูกค้า",
        "ทำนายลูกค้ารายใหม่",
    ], label_visibility="collapsed")
    st.markdown("<div style='height: 45vh'></div>", unsafe_allow_html=True)
    st.caption("DATA WAREHOUSE")
    st.caption("SQLite · RFM Model v1")

if rfm is None:
    st.markdown('<div class="eyebrow">Setup required</div><h1>ยังไม่มีข้อมูลสำหรับแสดงผล</h1>', unsafe_allow_html=True)
    st.info("กรุณารัน `etl_pipeline.py` เพื่อสร้าง retail_dw.db และเตรียมข้อมูล RFM ก่อนเปิดหน้านี้")
    st.stop()

if rfm.empty:
    st.warning("ฐานข้อมูลพร้อมใช้งาน แต่ยังไม่มีรายการลูกค้า")
    st.stop()

if menu == "ภาพรวมธุรกิจ":
    st.markdown('<div class="hero"><div class="eyebrow">Customer intelligence / Overview</div><h1>เข้าใจลูกค้าได้ในมุมเดียว</h1><div class="hero-copy">ภาพรวมประสิทธิภาพลูกค้าและโอกาสทางธุรกิจจากข้อมูล RFM ที่ผ่านการจัดกลุ่มด้วย Machine Learning</div></div>', unsafe_allow_html=True)

    total_revenue = rfm["Monetary"].sum()
    avg_revenue = rfm["Monetary"].mean()
    vip_count = rfm["Customer_Segment"].str.contains("VIP", na=False).sum()
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("ลูกค้าทั้งหมด", f"{len(rfm):,}", "ราย")
    kpi2.metric("ยอดขายรวม", f"£ {total_revenue:,.0f}")
    kpi3.metric("มูลค่าเฉลี่ย / คน", f"£ {avg_revenue:,.0f}")
    kpi4.metric("ลูกค้า VIP", f"{vip_count:,}", f"{vip_count / len(rfm):.1%} ของทั้งหมด")

    st.markdown('<div class="section-title">ภาพรวมการแบ่งกลุ่ม</div><div class="section-note">ดูสัดส่วนลูกค้าเพื่อจัดลำดับความสำคัญของแคมเปญถัดไป</div>', unsafe_allow_html=True)
    chart_col, insight_col = st.columns([1.6, 1], gap="large")
    segment_counts = rfm["Customer_Segment"].value_counts().reset_index(name="จำนวนลูกค้า")
    segment_counts.columns = ["กลุ่มลูกค้า", "จำนวนลูกค้า"]
    fig = px.bar(segment_counts, x="จำนวนลูกค้า", y="กลุ่มลูกค้า", color="กลุ่มลูกค้า", orientation="h", text="จำนวนลูกค้า", color_discrete_sequence=["#167a5a", "#50a88a", "#d4a84f", "#d27661"])
    fig.update_layout(showlegend=False, height=340, margin=dict(l=0, r=10, t=10, b=10), plot_bgcolor="white", paper_bgcolor="white", font=dict(family="Noto Sans Thai, DM Sans", color="#18212b", size=13), xaxis_title=None, yaxis_title=None, xaxis=dict(tickfont=dict(color="#18212b", size=12), gridcolor="#e7ecef", zerolinecolor="#cbd5d9"), yaxis=dict(tickfont=dict(color="#18212b", size=12), gridcolor="white"))
    fig.update_traces(textposition="outside", textfont=dict(color="#18212b", size=13), hovertemplate="%{y}: %{x} ราย<extra></extra>")
    with chart_col:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with insight_col:
        st.markdown('<div class="insight"><strong>จุดสังเกต</strong>กลุ่ม VIP มีลูกค้า {:,} ราย ควรใช้เป็นกลุ่มหลักสำหรับ retention และสิทธิพิเศษเฉพาะบุคคล</div>'.format(vip_count), unsafe_allow_html=True)
        st.markdown('<div style="height: 12px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="insight"><strong>พร้อมลงมือทำ</strong>ใช้เมนู “สำรวจข้อมูลลูกค้า” เพื่อดูรายชื่อและมูลค่าของแต่ละ segment</div>', unsafe_allow_html=True)

elif menu == "สำรวจข้อมูลลูกค้า":
    st.markdown('<div class="hero"><div class="eyebrow">Customer intelligence / Explorer</div><h1>สำรวจลูกค้าแบบเจาะจง</h1><div class="hero-copy">กรอง segment ที่ต้องการ แล้วเรียงลำดับลูกค้าตามมูลค่าเพื่อหาโอกาสทางธุรกิจได้ทันที</div></div>', unsafe_allow_html=True)
    segments = sorted(rfm["Customer_Segment"].dropna().unique().tolist())
    selected_segment = st.selectbox("เลือกกลุ่มลูกค้า", segments)
    result_df = rfm.loc[rfm["Customer_Segment"] == selected_segment, ["CustomerID", "Recency", "Frequency", "Monetary", "Customer_Segment"]].sort_values("Monetary", ascending=False)
    st.markdown(f'<div class="section-title">{selected_segment}</div><div class="section-note">พบลูกค้า {len(result_df):,} รายการ · เรียงตามยอดใช้จ่ายรวมจากมากไปน้อย</div>', unsafe_allow_html=True)
    st.dataframe(result_df, width="stretch", hide_index=True, column_config={"Monetary": st.column_config.NumberColumn("Monetary (£)", format="£ %.2f"), "Recency": st.column_config.NumberColumn("Recency (วัน)"), "Frequency": st.column_config.NumberColumn("Frequency (ครั้ง)")})

else:
    st.markdown('<div class="prediction-intro"><div><div class="eyebrow">Prediction workspace</div><h1>จัดกลุ่มลูกค้ารายใหม่</h1><div class="hero-copy">กรอกพฤติกรรมการซื้อ แล้วให้โมเดลช่วยระบุกลุ่มลูกค้าที่เหมาะสม</div></div><div class="model-status">● MODEL READY · RFM V1</div></div>', unsafe_allow_html=True)
    form_col, result_col = st.columns([1.15, .85], gap="large")
    with form_col:
        st.markdown('<div class="form-panel-title">ข้อมูลสำหรับวิเคราะห์</div><div class="form-panel-note">ทุกช่องจำเป็นต่อการคำนวณผลลัพธ์</div>', unsafe_allow_html=True)
        with st.form("prediction_form", border=True):
            st.markdown('<div class="input-label">รหัสลูกค้า</div><div class="input-help">ใช้รหัสช่วง 90000 - 99999 สำหรับลูกค้าใหม่</div>', unsafe_allow_html=True)
            customer_id = st.number_input("รหัสลูกค้า", min_value=90000, max_value=99999, value=90001, step=1, label_visibility="collapsed")
            recency_col, frequency_col = st.columns(2, gap="medium")
            with recency_col:
                st.markdown('<div class="input-label">Recency</div><div class="input-help">วันที่ไม่ได้ซื้อล่าสุด</div>', unsafe_allow_html=True)
                recency_input = st.number_input("Recency", min_value=1, value=15, step=1, label_visibility="collapsed")
            with frequency_col:
                st.markdown('<div class="input-label">Frequency</div><div class="input-help">จำนวนครั้งที่ซื้อ</div>', unsafe_allow_html=True)
                frequency_input = st.number_input("Frequency", min_value=1, value=6, step=1, label_visibility="collapsed")
            st.markdown('<div class="input-label">Monetary (£)</div><div class="input-help">ยอดใช้จ่ายรวมตลอดอายุลูกค้า</div>', unsafe_allow_html=True)
            monetary_input = st.number_input("Monetary", min_value=1.0, value=1200.0, step=50.0, label_visibility="collapsed")
            submit_btn = st.form_submit_button("วิเคราะห์และบันทึกข้อมูล", width="stretch")
    with result_col:
        st.markdown('<div class="result-panel"><div class="preview-kicker">Live preview</div><div class="preview-title">ข้อมูลที่จะส่งเข้าโมเดล</div><div class="preview-row"><span>Customer ID</span><span>{:,}</span></div><div class="preview-row"><span>Recency</span><span>{:,} วัน</span></div><div class="preview-row"><span>Frequency</span><span>{:,} ครั้ง</span></div><div class="preview-row"><span>Monetary</span><span>£ {:,.2f}</span></div></div>'.format(customer_id, recency_input, frequency_input, monetary_input), unsafe_allow_html=True)
        st.markdown('<div class="insight" style="margin-top: 1rem;"><strong>โมเดลพร้อมใช้งาน</strong>ระบบจะวิเคราะห์ทั้ง 3 ค่าเพื่อจัดกลุ่มลูกค้า และบันทึกผลลงฐานข้อมูลเมื่อกดยืนยัน</div>', unsafe_allow_html=True)

    if submit_btn:
        user_data = np.array([[recency_input, frequency_input, monetary_input]])
        prediction = model.predict(scaler.transform(user_data))[0]
        try:
            with get_db_connection() as conn:
                conn.execute("INSERT INTO dim_customer_rfm (CustomerID, Recency, Frequency, Monetary, Cluster, Customer_Segment) VALUES (?, ?, ?, ?, ?, ?)", (customer_id, recency_input, frequency_input, monetary_input, 99, prediction))
            st.markdown(f'<div class="result-banner"><strong>จัดกลุ่มสำเร็จ · {prediction}</strong>ลูกค้า {customer_id} ถูกบันทึกลงฐานข้อมูลเรียบร้อยแล้ว</div>', unsafe_allow_html=True)
        except sqlite3.IntegrityError:
            st.error(f"ไม่สามารถบันทึกได้: รหัสลูกค้า {customer_id} มีอยู่ในระบบแล้ว")
        except sqlite3.Error as error:
            st.error(f"ไม่สามารถบันทึกข้อมูลได้: {error}")

st.markdown('<div class="footer">RFM Intelligence · Customer segmentation workspace</div>', unsafe_allow_html=True)
        