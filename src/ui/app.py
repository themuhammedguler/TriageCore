# ==============================================================================
# TriageCore - Faz 6: Modern Kurumsal Klinik Karar Destek & Vaka Simülatörü Arayüzü
# ==============================================================================
# Bu modul; yapay zeka kliselerinden uzak, gercek bir Hastane Bilgi Yonetim
# Sistemi (HBYS / CDSS) ve OSCE Tip Egitim Simulatoru olarak tasarlanmis
# yuksek performansli Streamlit web konsoludur.
#
# Ozellikler:
# 1. Linear & Modern Healthtech estetik standartlarinda Derin Obsidiyen Dark Mode
# 2. Asla kirilmayan/kaymayan Sistem Telemetrisi ve Canli Ag Nabiz Indikatoru
# 3. Sol Menude hizli, anlik (0.01 sn) gecis saglayan modern Klinik Modul Secici
# 4. Elektronik Hasta Dosyasi (EHR Banner) ve gercekci Hekim-Hasta diyalog baloncuklari
# 5. Hizli anamnez/muayene istem haplari ve 100 uzerinden klinik teshis puanlama
# ==============================================================================

# JSON serilestirme islemleri icin json modulu yuklenir
import json

# Rastgele vaka secimi icin random modulu yuklenir
import random

# Sistem yollarini yonetmek icin sys modulu yuklenir
import sys

# Tekil oturum kimligi uretmek icin uuid modulu yuklenir
import uuid

# Dosya yollari islemleri icin Path modulu yuklenir
from pathlib import Path

# Tip ipuclari icin Any, Dict, List, Optional ve Tuple yuklenir
from typing import Any, Dict, List, Optional, Tuple

# Proje kok dizinini Python modul arama yoluna (sys.path) ekler
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# HTTP istekleri gondermek icin requests kutuphanesi yuklenir
import requests  # noqa: E402

# Web arayuzu olusturmak icin Streamlit kutuphanesi yuklenir
import streamlit as st  # noqa: E402

# Yerel vaka bankasi ve klinik degerlendirme fonksiyonlari ice aktarilir
from src.api.cases import (  # noqa: E402
    build_patient_system_prompt,
    evaluate_doctor_submission,
    list_public_cases,
    simulate_patient_response,
)

# Gomulu kural tabanli klinik triyaj motoru ice aktarilir
from src.api.engine import engine  # noqa: E402

# Klinik sohbet ve denetim loglayicisi ice aktarilir
from src.api.logger import audit_logger  # noqa: E402


def configure_page():
    """Sayfa basligini, genis ekran yerlesimini ve profesyonel klinik CSS stillerini uygular."""
    # Streamlit sayfa yapilandirmasi kurumsal medikal sembol ile baslatilir
    st.set_page_config(
        page_title="TriageCore | Klinik Karar Destek & OSCE Simülatörü",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Kurumsal hastane yazilimi gorunumu saglayan ozel Dark Mode CSS kurallari
    clinical_css = """
    <style>
        /* Genel arka plan ve tipografi - Kurumsal Obsidiyen Dark Palet */
        .stApp {
            background-color: #090D16;
            color: #F8FAFC;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        /* Sidebar Koyu Zemin ve Kenarlik */
        section[data-testid="stSidebar"] {
            background-color: #0D1322;
            border-right: 1px solid #1E293B;
        }

        /* Sidebar Navigasyon Butonlari - Modern Klinik Kartlar */
        div[data-testid="stSidebar"] div[role="radiogroup"] > label {
            background: #111A2E !important;
            border: 1px solid #1E2E4A !important;
            border-radius: 8px !important;
            padding: 10px 14px !important;
            margin-bottom: 8px !important;
            transition: all 0.2s ease !important;
            cursor: pointer !important;
            color: #F8FAFC !important;
            font-weight: 600 !important;
        }
        div[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
            border-color: #38BDF8 !important;
            background: #16243F !important;
            color: #38BDF8 !important;
        }
        div[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {
            background: linear-gradient(135deg, #152A4A 0%, #112038 100%) !important;
            border-color: #0EA5E9 !important;
            box-shadow: 0 0 12px rgba(14, 165, 233, 0.3) !important;
        }

        /* Ust Ana Marka & Baslik Cubugu */
        .brand-header {
            background: linear-gradient(135deg, #0F172A 0%, #111D35 100%);
            border: 1px solid #1E2E4A;
            border-left: 4px solid #0EA5E9;
            border-radius: 10px;
            padding: 18px 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .brand-header-left h1 {
            margin: 0;
            font-size: 1.45rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #F8FAFC;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .brand-header-left p {
            margin: 4px 0 0 0;
            font-size: 0.88rem;
            color: #94A3B8;
        }
        .brand-badge {
            background: #1E293B;
            color: #38BDF8;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 20px;
            border: 1px solid #334155;
            letter-spacing: 0.05em;
        }

        /* Klinik Bilgi Kartlari */
        .clinical-card {
            background: #111A2E;
            border: 1px solid #1E2E4A;
            border-radius: 10px;
            padding: 18px 20px;
            margin-bottom: 16px;
            color: #E2E8F0;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        .clinical-card h4 {
            margin-top: 0;
            margin-bottom: 12px;
            font-size: 0.95rem;
            color: #38BDF8;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border-bottom: 1px solid #1E293B;
            padding-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* Canli Nabiz Sinyali Animasyonu */
        @keyframes pulse-glow {
            0% {
                box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
            }
            70% {
                box-shadow: 0 0 0 7px rgba(16, 185, 129, 0);
            }
            100% {
                box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
            }
        }
        .telemetry-pulse {
            display: inline-block;
            width: 9px;
            height: 9px;
            background-color: #10B981;
            border-radius: 50%;
            margin-right: 6px;
            animation: pulse-glow 2s infinite;
        }

        /* Sol Sidebar Sistem Telemetri Kartı */
        .telemetry-card {
            background: linear-gradient(180deg, #131E35 0%, #0D1527 100%);
            border: 1px solid #1E293B;
            border-radius: 10px;
            padding: 14px;
            margin-top: 14px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
        }
        .telemetry-header {
            display: flex;
            align-items: center;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid #1E293B;
        }
        .telemetry-status-text {
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            color: #34D399;
        }
        .telemetry-tag {
            margin-left: auto;
            font-size: 0.68rem;
            padding: 2px 7px;
            border-radius: 10px;
            background: #1E293B;
            color: #38BDF8;
            border: 1px solid #334155;
            font-weight: 700;
        }
        .telemetry-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        .telemetry-row:last-child {
            margin-bottom: 0;
        }
        .telemetry-label {
            font-size: 0.74rem;
            font-weight: 600;
            color: #94A3B8;
            white-space: nowrap;
            letter-spacing: 0.02em;
        }
        .telemetry-pill {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.72rem;
            font-weight: 600;
            color: #38BDF8;
            background: #080D1A;
            border: 1px solid #1E2E4A;
            padding: 3px 8px;
            border-radius: 6px;
            white-space: nowrap;
            max-width: 140px;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        /* Elektronik Hasta Kabul Dosyasi (EHR Banner) */
        .ehr-banner {
            background: linear-gradient(135deg, #131E35 0%, #16243F 100%);
            border: 1px solid #233554;
            border-left: 5px solid #38BDF8;
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 18px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
        }
        .ehr-patient-title {
            font-size: 1.15rem;
            font-weight: 800;
            color: #F8FAFC;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .ehr-meta {
            font-size: 0.82rem;
            color: #94A3B8;
            margin-top: 4px;
        }
        .ehr-difficulty-badge {
            font-size: 0.76rem;
            font-weight: 700;
            padding: 4px 12px;
            border-radius: 20px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .diff-kolay {
            background: rgba(16, 185, 129, 0.2);
            color: #34D399;
            border: 1px solid #10B981;
        }
        .diff-orta {
            background: rgba(245, 158, 11, 0.2);
            color: #FBBF24;
            border: 1px solid #F59E0B;
        }
        .diff-zor {
            background: rgba(239, 68, 68, 0.2);
            color: #F87171;
            border: 1px solid #EF4444;
        }

        /* Konusma Balonlari - Hekim vs Hasta / Sistem */
        .msg-doctor {
            background-color: #152238;
            border: 1px solid #233554;
            border-left: 4px solid #38BDF8;
            padding: 14px 18px;
            border-radius: 8px;
            margin-bottom: 12px;
            font-size: 0.94rem;
            color: #F8FAFC;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }
        .msg-doctor .msg-header {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            color: #38BDF8;
            margin-bottom: 6px;
            letter-spacing: 0.05em;
        }
        .msg-patient {
            background-color: #0E2226;
            border: 1px solid #134E4A;
            border-left: 4px solid #10B981;
            padding: 14px 18px;
            border-radius: 8px;
            margin-bottom: 12px;
            font-size: 0.94rem;
            color: #F0FDF4;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }
        .msg-patient .msg-header {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            color: #34D399;
            margin-bottom: 6px;
            letter-spacing: 0.05em;
        }
        .msg-system {
            background-color: #121927;
            border: 1px solid #1E2E4A;
            border-left: 4px solid #6366F1;
            padding: 14px 18px;
            border-radius: 8px;
            margin-bottom: 12px;
            font-size: 0.94rem;
            color: #F1F5F9;
        }
        .msg-system .msg-header {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            color: #818CF8;
            margin-bottom: 6px;
            letter-spacing: 0.05em;
        }

        /* ESI Aciliyet Rozetleri - Parlayan Dinamik Koyu Tonlar */
        .esi-badge-1 {
            background: linear-gradient(135deg, #DC2626 0%, #7F1D1D 100%);
            color: #FFFFFF;
            padding: 12px 16px;
            border-radius: 8px;
            font-weight: 800;
            font-size: 0.95rem;
            box-shadow: 0 0 16px rgba(220, 38, 38, 0.45);
            border: 1px solid #EF4444;
            text-align: center;
        }
        .esi-badge-2 {
            background: linear-gradient(135deg, #EA580C 0%, #9A3412 100%);
            color: #FFFFFF;
            padding: 12px 16px;
            border-radius: 8px;
            font-weight: 800;
            font-size: 0.95rem;
            box-shadow: 0 0 16px rgba(234, 88, 12, 0.45);
            border: 1px solid #F97316;
            text-align: center;
        }
        .esi-badge-3 {
            background: linear-gradient(135deg, #D97706 0%, #92400E 100%);
            color: #FFFFFF;
            padding: 12px 16px;
            border-radius: 8px;
            font-weight: 800;
            font-size: 0.95rem;
            box-shadow: 0 0 14px rgba(217, 119, 6, 0.35);
            border: 1px solid #F59E0B;
            text-align: center;
        }
        .esi-badge-4 {
            background: linear-gradient(135deg, #2563EB 0%, #1E40AF 100%);
            color: #FFFFFF;
            padding: 12px 16px;
            border-radius: 8px;
            font-weight: 800;
            font-size: 0.95rem;
            border: 1px solid #3B82F6;
            text-align: center;
        }
        .esi-badge-5 {
            background: linear-gradient(135deg, #059669 0%, #065F46 100%);
            color: #FFFFFF;
            padding: 12px 16px;
            border-radius: 8px;
            font-weight: 800;
            font-size: 0.95rem;
            border: 1px solid #10B981;
            text-align: center;
        }

        /* Streamlit Metrik Duzenlemeleri */
        div[data-testid="stMetricValue"] {
            color: #38BDF8 !important;
            font-weight: 800;
        }
        div[data-testid="stMetricLabel"] {
            color: #94A3B8 !important;
            font-size: 0.80rem !important;
        }

        /* Butonlarin Koyu Temaya Uygun Estetik Koseleri */
        button[kind="secondary"] {
            border-radius: 8px !important;
            border: 1px solid #233554 !important;
            background: #111A2E !important;
            color: #E2E8F0 !important;
            transition: all 0.2s ease;
        }
        button[kind="secondary"]:hover {
            border-color: #38BDF8 !important;
            color: #38BDF8 !important;
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.2) !important;
        }
        button[kind="primary"] {
            border-radius: 8px !important;
            background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
            border: none !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 12px rgba(2, 132, 199, 0.35) !important;
        }
    </style>
    """
    # CSS stillerini sayfaya gom
    st.markdown(clinical_css, unsafe_allow_html=True)


def init_session_state():
    """Uygulama oturum degiskenlerini (session_state) ilk calismada hazirlar."""
    # Varsayilan calisma modulu (Sol Menu)
    if "current_module" not in st.session_state:
        st.session_state.current_module = "🩺 Klinik Triyaj"

    # Triyaj modu oturum kimligi
    if "triage_session_id" not in st.session_state:
        st.session_state.triage_session_id = f"TRG-{uuid.uuid4().hex[:6].upper()}"
    # Triyaj modu mesaj gecmisi
    if "triage_messages" not in st.session_state:
        st.session_state.triage_messages = []
    # Son hesaplanan triyaj durumu
    if "last_triage_meta" not in st.session_state:
        st.session_state.last_triage_meta = None

    # Doktor Vaka Simulasyonu oturum kimligi
    if "sim_session_id" not in st.session_state:
        st.session_state.sim_session_id = f"SIM-{uuid.uuid4().hex[:6].upper()}"
    # Secili klinik vaka kimligi
    if "active_case_id" not in st.session_state:
        st.session_state.active_case_id = "CASE-001"
    # Doktor-Hasta diyalog gecmisi
    if "sim_messages" not in st.session_state:
        st.session_state.sim_messages = []
    # Son vaka degerlendirme sonucu
    if "sim_eval_result" not in st.session_state:
        st.session_state.sim_eval_result = None


# ONBELLEKLEME DEKORATORU: Sekme gecislerinde 5 saniye beklemeyi engeller, anlik (0.01 sn) gecis saglar
@st.cache_data(ttl=15, show_spinner=False)
def check_endpoint_health(api_url: str) -> Dict[str, Any]:
    """Belirtilen API sunucusunun saglik durumunu 15 saniyelik onbellekleme ve kisa timeout ile sorgular."""
    clean_url = api_url.rstrip("/")
    try:
        # Kisa 600 ms zaman asimi: Sunucu yanit vermese bile sayfanin donmasini onler
        resp = requests.get(f"{clean_url}/health", timeout=0.6)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    # Baglanti saglanamazsa guvenli gomulu yerel mantik moduna doner
    return {
        "status": "local_embedded",
        "model_loaded": True,
        "engine_type": "embedded_engine",
        "device": "local-cpu",
    }


def call_chat_endpoint(
    api_url: str,
    session_id: str,
    mode: str,
    messages: List[Dict[str, str]],
    case_id: Optional[str] = None,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """Mesaj gecmisini uzak Colab Ngrok sunucusuna veya yerel gomulu motora iletir."""
    clean_url = api_url.rstrip("/")
    payload = {
        "session_id": session_id,
        "mode": mode,
        "messages": messages,
        "case_id": case_id,
        "temperature": temperature,
        "max_tokens": 512,
    }

    # Eger doktor simulasyon modundaysak ve uzak LLM'e gidiyorsa gizli hasta rolunu ekle
    if mode == "simulation" and case_id:
        sys_prompt = build_patient_system_prompt(case_id)
        payload["messages"] = [{"role": "system", "content": sys_prompt}] + messages

    try:
        resp = requests.post(f"{clean_url}/v1/chat", json=payload, timeout=25)
        if resp.status_code == 200:
            data = resp.json()
            # Eger uzak Colab sunucusundan geliyorsa yerel denetim dosyasina da kaydet
            if "127.0.0.1" not in clean_url and "localhost" not in clean_url:
                audit_logger.log_interaction(
                    session_id=session_id,
                    mode=mode,
                    case_id=case_id,
                    user_message=messages[-1]["content"] if messages else "",
                    assistant_reply=data.get("reply", ""),
                    triage_level=data.get("triage_level"),
                    urgency_score=data.get("urgency_score"),
                    inference_time_ms=data.get("inference_time_ms", 0.0),
                )
            return data
    except Exception:
        pass

    # Uzak sunucuya erisilemezse kesintisiz klinik deneyim icin yerel motor devreye girer
    last_msg = messages[-1]["content"] if messages else ""
    if mode == "simulation":
        reply_text = simulate_patient_response(case_id or "CASE-001", last_msg)
        audit_logger.log_interaction(
            session_id=session_id,
            mode="simulation",
            case_id=case_id,
            user_message=last_msg,
            assistant_reply=reply_text,
            inference_time_ms=12.0,
        )
        return {
            "session_id": session_id,
            "mode": "simulation",
            "reply": reply_text,
            "inference_time_ms": 12.0,
        }

    # Triyaj yerel analizi
    combined_text = " ".join([m["content"] for m in messages if m["role"] == "user"])
    level, score, is_emerg, flags, assess, action = engine._analyze_clinical_urgency(combined_text)
    reply_text = f"**Klinik Değerlendirme:** {assess}\n\n**Önerilen Aksiyon Planı:** {action}"
    audit_logger.log_interaction(
        session_id=session_id,
        mode="triage",
        user_message=last_msg,
        assistant_reply=reply_text,
        triage_level=level.value,
        urgency_score=score,
        inference_time_ms=15.0,
    )
    return {
        "session_id": session_id,
        "mode": "triage",
        "reply": reply_text,
        "triage_level": level.value,
        "urgency_score": score,
        "is_emergency": is_emerg,
        "red_flags": flags,
        "inference_time_ms": 15.0,
    }


def render_sidebar() -> Tuple[str, str, float]:
    """Sol kontrol panelini (Navigasyon, yapay zeka baglanti ayarlari ve sistem durumunu) cizer."""
    with st.sidebar:
        # Marka Basligi
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
                <span style="font-size:1.6rem;">🩺</span>
                <div>
                    <h3 style="margin:0; font-size:1.25rem; font-weight:800; color:#F8FAFC; letter-spacing:-0.01em;">TriageCore</h3>
                    <p style="margin:0; font-size:0.75rem; color:#94A3B8; font-weight:600;">Klinik Karar Destek & CDSS</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 1. MODUL NAVIGASYONU (KULLANICI ISTEGI DOGRULTUSUNDA SOL MENUDE)
        st.markdown(
            "<p style='font-size:0.75rem; font-weight:700; color:#94A3B8; margin-bottom:6px; letter-spacing:0.05em;'>KLİNİK ÇALIŞMA ALANI</p>",
            unsafe_allow_html=True,
        )
        selected_module = st.radio(
            "Çalışma Modülü",
            options=[
                "🩺 Klinik Triyaj",
                "👨‍⚕️ Doktor Simülatörü",
                "📋 Denetim Kayıtları",
            ],
            index=0,
            label_visibility="collapsed",
        )
        st.divider()

        # 2. YAPAY ZEKA MODEL BAGLANTISI (ANLASILIR ISIMLENDIRME ILE)
        st.markdown("#### 🤖 AI Model Bağlantısı")
        st.caption("Google Colab (Ngrok) veya Yerel Sunucu URL:")
        api_url = st.text_input(
            "AI Sunucu Adresi",
            value="http://127.0.0.1:8000",
            help="Google Colab son hücresinde üretilen Ngrok veya Cloudflared HTTPS adresini buraya yapıştırın.",
            label_visibility="collapsed",
        )

        # Determinizm / Kararlilik (Temperature) ayar cubugu
        st.markdown(
            "<p style='font-size:0.78rem; color:#94A3B8; margin-top:10px; margin-bottom:4px;'>Klinik Kararlılık (Temperature)</p>",
            unsafe_allow_html=True,
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.05,
            max_value=0.80,
            value=0.20,
            step=0.05,
            help="Düşük değerler protokollere tam bağlı ve tutarlı klinik yanıtlar üretir.",
            label_visibility="collapsed",
        )

        # Onbellekli hizli sunucu telemetri kontrolu
        health_info = check_endpoint_health(api_url)
        raw_engine = health_info.get("engine_type", "embedded_engine")
        engine_label = "Llama-3 (Colab)" if "colab" in raw_engine else "Yerel Mantık Motoru"
        raw_device = health_info.get("device", "local-cpu")
        device_label = raw_device if len(raw_device) <= 12 else raw_device[:12] + "..."
        is_remote = "127.0.0.1" not in api_url and "localhost" not in api_url
        conn_label = "Uzak HTTPS" if is_remote else "Yerel / Gomulu"

        # ASLA SATIR BOLUNMESI VEYA KAYMA YAPMAYAN YENI TELEMETRI KARTI
        st.markdown(
            f"""
            <div class="telemetry-card">
                <div class="telemetry-header">
                    <span class="telemetry-pulse"></span>
                    <span class="telemetry-status-text">SİSTEM ÇEVRİMİÇİ</span>
                    <span class="telemetry-tag">v2.0</span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-label">Model Motoru</span>
                    <span class="telemetry-pill" title="{raw_engine}">⚡ {engine_label}</span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-label">İşlemci / Donanım</span>
                    <span class="telemetry-pill" title="{raw_device}">💻 {device_label}</span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-label">Bağlantı Türü</span>
                    <span class="telemetry-pill">🔒 {conn_label}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("© 2026 TriageCore • Acil Servis Triyaj & Anamnez Destek Konsolu")

        return selected_module, api_url, temperature


def render_triage_module(api_url: str, temperature: float):
    """Mod 1: Ust uste soru sorulabilen Cok Turlu Klinik Triyaj ve Konsultasyon ekrani."""
    st.markdown(
        """
        <div class="brand-header">
            <div class="brand-header-left">
                <h1><span>🩺</span> Klinik Triyaj ve Çok Turlu Konsültasyon</h1>
                <p>Hasta semptomlarını iletin; sistem ESI aciliyetini derecelendirsin ve takip sorularıyla ayırıcı tanıyı derinleştirsin.</p>
            </div>
            <div class="brand-badge">ACİL SERVİS CDSS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_chat, col_panel = st.columns([2.1, 1.0], gap="large")

    with col_panel:
        st.markdown('<div class="clinical-card"><h4>📋 Klinik Triyaj Özeti</h4>', unsafe_allow_html=True)
        st.caption(f"Aktif Protokol: `{st.session_state.triage_session_id}`")

        meta = st.session_state.last_triage_meta
        if meta and meta.get("urgency_score"):
            score = meta["urgency_score"]
            level_text = meta.get("triage_level", f"ESI-{score}")
            st.markdown(
                f'<div class="esi-badge-{score}">{level_text}</div>',
                unsafe_allow_html=True,
            )
            st.write("")
            st.metric("Aciliyet Derecesi (ESI)", f"Seviye {score} / 5")

            red_flags = meta.get("red_flags", [])
            if red_flags:
                st.markdown("**🚨 Saptanan Kırmızı Bayraklar:**")
                for rf in red_flags:
                    st.error(f"• {rf}")
            else:
                st.info("Aktif hayati kırmızı bayrak saptanmadı.")
            inf_ms = meta.get("inference_time_ms") or 0
            st.caption(f"⚡ Yanıt Süresi: {inf_ms} ms")
        else:
            st.info("Henüz semptom analizi yapılmadı. Hızlı senaryolardan birini seçin veya aşağıdan şikayet girin.")

        st.divider()
        if st.button("🔄 Yeni Triyaj Oturumu Başlat", use_container_width=True):
            st.session_state.triage_session_id = f"TRG-{uuid.uuid4().hex[:6].upper()}"
            st.session_state.triage_messages = []
            st.session_state.last_triage_meta = None
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_chat:
        # Hizli ornek senaryo butonlari
        st.markdown(
            "<p style='font-size:0.85rem; font-weight:700; color:#94A3B8; margin-bottom:8px;'>HIZLI KLİNİK SENARYO SEÇİN:</p>",
            unsafe_allow_html=True,
        )
        q_col1, q_col2, q_col3 = st.columns(3)
        preset_prompt = None
        if q_col1.button("⚡ Akut Göğüs Ağrısı & Terleme", use_container_width=True):
            preset_prompt = "54 yaşında erkek hasta, 30 dakikadır göğüs ortasında sol kola ve çeneye yayılan baskı tarzı ağrı ve soğuk terleme tarifliyor."
        if q_col2.button("⚡ Sağ Alt Kadran Karın Ağrısı", use_container_width=True):
            preset_prompt = "Göbek çevresinden başlayıp sağ alt kadrana yerleşen şiddetli karın ağrısı, iştahsızlık ve 38.2 derece ateş mevcut."
        if q_col3.button("⚡ İdrarda Yanma & Sık İdrar", use_container_width=True):
            preset_prompt = "3 gündür idrar yaparken yanma ve sık idrara çıkma şikayeti var, ateş veya böğür ağrısı yok."

        # Mesaj gecmisini kurumsal Hekim vs Sistem kartlari ile goster
        for msg in st.session_state.triage_messages:
            if msg["role"] == "user":
                st.markdown(
                    f"""
                    <div class="msg-doctor">
                        <div class="msg-header"><span>👨‍⚕️</span> Hekim / Semptom Bildirimi</div>
                        {msg["content"]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="msg-system">
                        <div class="msg-header"><span>➕</span> TriageCore Klinik Değerlendirme</div>
                        {msg["content"]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Alt kisimdaki cok turlu konsültasyon chat kutusu
        user_input = st.chat_input("Hasta şikayetini veya takip sorunuzu yazın (Örn: Hangi EKG ve kan tetkikleri istenmeli?)...")
        active_query = preset_prompt or user_input

        if active_query:
            st.session_state.triage_messages.append({"role": "user", "content": active_query})
            res = call_chat_endpoint(
                api_url=api_url,
                session_id=st.session_state.triage_session_id,
                mode="triage",
                messages=st.session_state.triage_messages,
                temperature=temperature,
            )
            st.session_state.triage_messages.append({"role": "assistant", "content": res["reply"]})
            st.session_state.last_triage_meta = res
            st.rerun()


def render_simulation_module(api_url: str, temperature: float):
    """Mod 2: Kullanicinin Doktor oldugu, hastaya sorular sorarak teshis koydugu OSCE Simulatoru."""
    cases = list_public_cases()
    case_map = {c["case_id"]: c for c in cases}
    current_case = case_map[st.session_state.active_case_id]

    # Ust Marka ve Vaka Basligi
    st.markdown(
        """
        <div class="brand-header">
            <div class="brand-header-left">
                <h1><span>👨‍⚕️</span> Klinik Vaka Simülatörü — Hekim Anamnez & Tanı Konsolu</h1>
                <p>Bu simülasyonda siz <b>Muayene Eden Hekimsiniz</b>. Karşınızdaki hastaya sorular yöneltin, fizik muayene ve tetkik isteyip kesin tanınızı koyun.</p>
            </div>
            <div class="brand-badge">OSCE SINAV MODU</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Vaka Secim & Yonetim Cubugu
    top_c1, top_c2, top_c3 = st.columns([2.3, 1.0, 1.0])
    with top_c1:
        selected_id = st.selectbox(
            "Aktif Klinik Vaka:",
            options=list(case_map.keys()),
            format_func=lambda cid: f"{case_map[cid]['title']} — [Zorluk: {case_map[cid]['difficulty']}]",
            index=list(case_map.keys()).index(st.session_state.active_case_id),
            label_visibility="collapsed",
        )
        if selected_id != st.session_state.active_case_id:
            st.session_state.active_case_id = selected_id
            st.session_state.sim_messages = []
            st.session_state.sim_eval_result = None
            st.session_state.sim_session_id = f"SIM-{uuid.uuid4().hex[:6].upper()}"
            st.rerun()

    with top_c2:
        if st.button("🎲 Rastgele Vaka Getir", use_container_width=True):
            random_case = random.choice(list(case_map.keys()))
            st.session_state.active_case_id = random_case
            st.session_state.sim_messages = []
            st.session_state.sim_eval_result = None
            st.session_state.sim_session_id = f"SIM-{uuid.uuid4().hex[:6].upper()}"
            st.rerun()

    with top_c3:
        if st.button("🔄 Vakayı Sıfırla", use_container_width=True):
            st.session_state.sim_messages = []
            st.session_state.sim_eval_result = None
            st.session_state.sim_session_id = f"SIM-{uuid.uuid4().hex[:6].upper()}"
            st.rerun()

    # Vaka baslangicinda hastanin ilk basvuru sikayetini otomatik yukle
    if not st.session_state.sim_messages:
        st.session_state.sim_messages.append(
            {"role": "assistant", "content": current_case["initial_complaint"]}
        )

    # Zorluk derecesine gore CSS sinifi belirle
    diff_class = "diff-orta"
    if current_case["difficulty"] == "Kolay":
        diff_class = "diff-kolay"
    elif current_case["difficulty"] == "Zor":
        diff_class = "diff-zor"

    # ELEKTRONIK HASTA KABUL DOSYASI (EHR CHART BANNER)
    st.markdown(
        f"""
        <div class="ehr-banner">
            <div>
                <div class="ehr-patient-title">
                    <span>👤</span> {current_case['patient_name']} ({current_case['age']} Yaş, {current_case['gender']})
                </div>
                <div class="ehr-meta">
                    <b>Kabul Birimi:</b> Acil Servis Triyaj Masası &nbsp;|&nbsp;
                    <b>Protokol No:</b> <code>{st.session_state.sim_session_id}</code> &nbsp;|&nbsp;
                    <b>Mevcut Durum:</b> Hekim Anamnezi Alınıyor
                </div>
            </div>
            <div>
                <span class="ehr-difficulty-badge {diff_class}">Zorluk: {current_case['difficulty']}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_anamnesis, col_diagnosis = st.columns([2.1, 1.0], gap="large")

    with col_anamnesis:
        # Hizli Hekim Sorgulari ve Muayene Istemi Butonlari
        st.markdown(
            "<p style='font-size:0.85rem; font-weight:700; color:#94A3B8; margin-bottom:8px;'>HIZLI HEKİM SORGULARI VE MUAYENE İSTEMLERİ:</p>",
            unsafe_allow_html=True,
        )
        b1, b2, b3, b4, b5 = st.columns(5)
        quick_action = None
        if b1.button("📍 Ağrı Karakteri", use_container_width=True):
            quick_action = "Ağrının tam yeri neresi, başka bir yere yayılıyor mu ve şiddeti nasıl?"
        if b2.button("🌡️ Ek Semptomlar", use_container_width=True):
            quick_action = "Ateş, bulantı, kusma, iştahsızlık veya başka bir şikayetiniz var mı?"
        if b3.button("💊 İlaç & Geçmiş", use_container_width=True):
            quick_action = "Bilinen kronik bir hastalığınız, düzenli kullandığınız ilaç veya ameliyat geçmişiniz var mı?"
        if b4.button("🩺 Fizik Muayene", use_container_width=True):
            quick_action = "Hastanın vital bulgularını ölç ve hedefe yönelik detaylı fizik muayenesini yap."
        if b5.button("🧪 Lab & Tetkik", use_container_width=True):
            quick_action = "Tam kan sayımı, biyokimya ve ilgili radyolojik görüntüleme tetkiklerini iste."

        # Hekim (Siz) vs Hasta Diyalog Akisi
        for msg in st.session_state.sim_messages:
            if msg["role"] == "user":
                st.markdown(
                    f"""
                    <div class="msg-doctor">
                        <div class="msg-header"><span>👨‍⚕️</span> Hekim (Siz)</div>
                        {msg["content"]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="msg-patient">
                        <div class="msg-header"><span>👤</span> Hasta ({current_case['patient_name']}) / Klinik Bulgular</div>
                        {msg["content"]}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        doctor_input = st.chat_input("Hastaya sorunuzu yöneltin veya klinik tetkik isteyin...")
        final_question = quick_action or doctor_input

        if final_question:
            st.session_state.sim_messages.append({"role": "user", "content": final_question})
            res = call_chat_endpoint(
                api_url=api_url,
                session_id=st.session_state.sim_session_id,
                mode="simulation",
                messages=st.session_state.sim_messages,
                case_id=st.session_state.active_case_id,
                temperature=temperature,
            )
            st.session_state.sim_messages.append({"role": "assistant", "content": res["reply"]})
            st.rerun()

    with col_diagnosis:
        st.markdown('<div class="clinical-card"><h4>🎯 Kesin Tanı ve Triyaj Kararı</h4>', unsafe_allow_html=True)
        st.write("Anamnez, fizik muayene ve laboratuvar bulgularına dayanarak tanınızı belirleyin:")

        diag_input = st.text_input(
            "Kesin Tanınız (Örn: Akut Apandisit)",
            placeholder="Klinik tanıyı buraya yazın...",
        )
        esi_input = st.selectbox(
            "Tahmini ESI Aciliyet Seviyesi",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: f"ESI-{x} ({'Hayati / Kırmızı Alan' if x <= 2 else 'Acil / Sarı-Yeşil Alan'})",
            index=1,
        )

        # Sorulan soru sayisini hesapla
        doctor_q_count = sum(1 for m in st.session_state.sim_messages if m["role"] == "user")
        st.caption(f"🩺 Yapılan Hekim Sorgusu / Muayene Sayısı: **{doctor_q_count}**")

        if st.button("🏆 Tanıyı Doğrula & Puanla", type="primary", use_container_width=True):
            if not diag_input.strip():
                st.warning("Lütfen değerlendirme öncesi bir tanı ifadesi yazın.")
            else:
                eval_res = evaluate_doctor_submission(
                    case_id=st.session_state.active_case_id,
                    submitted_diagnosis=diag_input,
                    submitted_esi=esi_input,
                    question_count=doctor_q_count,
                )
                audit_logger.log_interaction(
                    session_id=st.session_state.sim_session_id,
                    mode="case_evaluation",
                    case_id=st.session_state.active_case_id,
                    user_message=f"Hekim Tanısı: {diag_input} (ESI-{esi_input})",
                    assistant_reply=f"Skor: {eval_res['score']}/100 | Doğru Tanı: {eval_res['correct_diagnosis']}",
                    urgency_score=eval_res["expected_esi"],
                )
                st.session_state.sim_eval_result = eval_res

        # Puanlama sonucu karti
        if st.session_state.sim_eval_result:
            ev = st.session_state.sim_eval_result
            st.divider()
            st.metric("Klinik Başarı Puanı", f"{ev['score']} / 100")
            if ev["is_correct"]:
                st.success(ev["feedback"])
            else:
                st.error(ev["feedback"])
            st.markdown(f"**💎 Klinik İpuçları (Clinical Pearls):**\n\n{ev['clinical_pearls']}")
        st.markdown("</div>", unsafe_allow_html=True)


def render_audit_logs_module():
    """Mod 3: Kaydedilen tum sohbet ve vaka degerlendirme loglarini inceleme ekrani."""
    st.markdown(
        """
        <div class="brand-header">
            <div class="brand-header-left">
                <h1><span>📋</span> Klinik Sohbet Kayıtları ve Denetim Logları</h1>
                <p>Sistem üzerinden yapılan tüm konsültasyonlar, vaka simülasyonları ve üretilen cevaplar yasal denetim için kayıt altındadır.</p>
            </div>
            <div class="brand-badge">AUDIT TRAIL</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    f_col1, f_col2, f_col3 = st.columns([1.5, 1.5, 1.0])
    with f_col1:
        mode_filter = st.selectbox(
            "Modül Filtresi",
            options=["Tümü", "triage", "simulation", "case_evaluation"],
        )
    with f_col2:
        limit_val = st.selectbox("Gösterilecek Kayıt Sayısı", options=[25, 50, 100, 250], index=1)

    selected_mode = None if mode_filter == "Tümü" else mode_filter
    logs = audit_logger.get_logs(limit=limit_val, mode=selected_mode)

    with f_col3:
        st.write("")
        st.download_button(
            label="📥 Logları İndir (JSON)",
            data=json.dumps(logs, ensure_ascii=False, indent=2),
            file_name="triagecore_audit_logs.json",
            mime="application/json",
            use_container_width=True,
        )

    # Ust Metrikler (None-safe guvenli hesaplamalar ile TypeError hatasini engeller)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Toplam Denetim Kaydı", f"{len(logs)} Adet")
    with m2:
        # NoneType hatalarini onleyen guvenli gecikme suresi toplami
        avg_time = round(
            sum((i.get("inference_time_ms") or 0.0) for i in logs) / max(1, len(logs)),
            1,
        )
        st.metric("Ortalama Yanıt Süresi", f"{avg_time} ms")
    with m3:
        # TypeError: '<=' not supported between 'NoneType' and 'int' hatasini onleyen guvenli filtreleme
        emergency_count = sum(
            1
            for i in logs
            if i.get("urgency_score") is not None and i["urgency_score"] <= 2
        )
        st.metric("Kritik Vaka / Acil Oranı", f"{emergency_count} Kayıt")

    st.divider()

    if not logs:
        st.info("Henüz kaydedilmiş bir klinik görüşme bulunmuyor.")
        return

    for item in logs:
        ts = item.get("timestamp", "")
        sess = item.get("session_id", "")
        mode_str = item.get("mode", "").upper()
        case_str = f" | Vaka: {item['case_id']}" if item.get("case_id") else ""
        esi_str = f" | {item['triage_level']}" if item.get("triage_level") else ""
        inf_time = item.get("inference_time_ms") or 0

        with st.expander(f"[{ts}] Protokol: {sess} | Mod: {mode_str}{case_str}{esi_str}"):
            st.markdown(f"**Hekim / Kullanıcı Girdisi:**\n> {item.get('user_message', '')}")
            st.markdown(f"**Sistem / Hasta Yanıtı:**\n\n{item.get('assistant_reply', '')}")
            st.caption(f"⚡ Yanıt Süresi: {inf_time} ms")


def main():
    """Ana Streamlit uygulama akisini ve sol menuden anlik calisma alanini baslatir."""
    configure_page()
    init_session_state()
    # Sol menuden secilen modul, API adresi ve sicaklik ayari aninda alinir
    selected_module, api_url, temperature = render_sidebar()

    # Sol menude secilen modulu ekrana aninda yukle (5 sn bekleme yok)
    if selected_module == "🩺 Klinik Triyaj":
        render_triage_module(api_url, temperature)
    elif selected_module == "👨‍⚕️ Doktor Simülatörü":
        render_simulation_module(api_url, temperature)
    else:
        render_audit_logs_module()


if __name__ == "__main__":
    main()
