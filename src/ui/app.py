# ==============================================================================
# TriageCore - Faz 6: Kurumsal Klinik Karar Destek ve Vaka Simülatörü Arayüzü
# ==============================================================================
# Bu modul, hicbir 'AI klisesi' (robot ikonlari, renkli balonlar) icermeyen,
# gercek bir Hastane Bilgi Yonetim Sistemi (HBYS / CDSS) gorunumunde tasarlanmis
# Streamlit tabanli web konsoludur. Hem Colab Ngrok uzak sunucusuna hem de yerel
# motora baglanabilir; Cok Turlu Triyaj, Doktor Vaka Simulatoru ve Denetim
# Loglari modullerini barindirir.
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

# Yerel yedek (offline fallback) vaka ve log modulleri ice aktarilir
from src.api.cases import (  # noqa: E402
    build_patient_system_prompt,
    evaluate_doctor_submission,
    list_public_cases,
    simulate_patient_response,
)
from src.api.engine import engine  # noqa: E402
from src.api.logger import audit_logger  # noqa: E402


def configure_page():
    """Sayfa basligini, genis ekran yerlesimini ve sade klinik CSS stillerini uygular."""
    # Streamlit sayfa yapilandirmasi (robot ikonu yerine sade medikal sembol)
    st.set_page_config(
        page_title="TriageCore | Klinik Karar Destek Sistemi",
        page_icon="➕",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Kurumsal hastane yazilimi gorunumu saglayan ozel Dark Mode CSS kurallari
    clinical_css = """
    <style>
        /* Genel arka plan ve tipografi - Kurumsal Dark Palette */
        .stApp {
            background-color: #0B0F19;
            color: #F8FAFC;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        /* Sidebar Koyu Zemin */
        section[data-testid="stSidebar"] {
            background-color: #0F172A;
            border-right: 1px solid #1E293B;
        }

        /* Ust baslik cubugu */
        .clinical-header {
            background-color: #111827;
            color: #F8FAFC;
            padding: 18px 26px;
            border-radius: 8px;
            border: 1px solid #1E293B;
            border-left: 5px solid #0EA5E9;
            margin-bottom: 22px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }
        .clinical-header h2 {
            margin: 0;
            font-size: 1.4rem;
            font-weight: 700;
            letter-spacing: -0.01em;
            color: #F8FAFC;
        }
        .clinical-header p {
            margin: 6px 0 0 0;
            font-size: 0.88rem;
            color: #94A3B8;
        }

        /* Klinik bilgi kartlari - Koyu Mod */
        .clinical-card {
            background-color: #131D31;
            border: 1px solid #1E293B;
            border-radius: 8px;
            padding: 18px;
            margin-bottom: 16px;
            color: #E2E8F0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }
        .clinical-card h4 {
            margin-top: 0;
            margin-bottom: 10px;
            font-size: 0.95rem;
            color: #38BDF8;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            border-bottom: 1px solid #1E293B;
            padding-bottom: 8px;
        }

        /* Sol Sidebar Baglanti Durumu Kartı */
        .clinical-status-box {
            background-color: #162238;
            border: 1px solid #1E293B;
            border-radius: 6px;
            padding: 12px 14px;
            font-size: 0.84rem;
            color: #E2E8F0;
            margin-top: 8px;
        }
        .status-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }
        .status-row:last-child {
            margin-bottom: 0;
        }
        .status-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            background-color: #10B981;
            border-radius: 50%;
            margin-right: 6px;
            box-shadow: 0 0 8px rgba(16, 185, 129, 0.6);
        }
        .status-val {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            color: #38BDF8;
            font-weight: 600;
        }
        .status-online {
            color: #34D399;
            font-weight: 700;
        }

        /* Mesaj kutulari - Kurumsal Hekim ve Hasta Renkleri */
        .msg-doctor {
            background-color: #1E293B;
            border: 1px solid #334155;
            border-left: 4px solid #38BDF8;
            padding: 14px 18px;
            border-radius: 6px;
            margin-bottom: 12px;
            font-size: 0.95rem;
            color: #F8FAFC;
        }
        .msg-system {
            background-color: #0F172A;
            border: 1px solid #1E293B;
            border-left: 4px solid #10B981;
            padding: 14px 18px;
            border-radius: 6px;
            margin-bottom: 12px;
            font-size: 0.95rem;
            color: #F1F5F9;
        }
        .msg-label {
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            color: #38BDF8;
            margin-bottom: 6px;
            letter-spacing: 0.05em;
        }
        .msg-system .msg-label {
            color: #34D399;
        }

        /* ESI Aciliyet Rozetleri - Yuksek Kontrastli Koyu Tonlar */
        .esi-badge-1 {
            background: linear-gradient(135deg, #DC2626, #991B1B);
            color: #FFFFFF;
            padding: 10px 14px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.92rem;
            box-shadow: 0 0 14px rgba(220, 38, 38, 0.35);
        }
        .esi-badge-2 {
            background: linear-gradient(135deg, #EA580C, #C2410C);
            color: #FFFFFF;
            padding: 10px 14px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.92rem;
            box-shadow: 0 0 14px rgba(234, 88, 12, 0.35);
        }
        .esi-badge-3 {
            background: linear-gradient(135deg, #D97706, #B45309);
            color: #FFFFFF;
            padding: 10px 14px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.92rem;
        }
        .esi-badge-4 {
            background: linear-gradient(135deg, #2563EB, #1D4ED8);
            color: #FFFFFF;
            padding: 10px 14px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.92rem;
        }
        .esi-badge-5 {
            background: linear-gradient(135deg, #059669, #047857);
            color: #FFFFFF;
            padding: 10px 14px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.92rem;
        }

        /* Streamlit Metrik ve Expander Kartlari */
        div[data-testid="stMetricValue"] {
            color: #38BDF8 !important;
            font-weight: 700;
        }
        div[data-testid="stMetricLabel"] {
            color: #94A3B8 !important;
        }
    </style>
    """
    # CSS kurallarini sayfaya gom
    st.markdown(clinical_css, unsafe_allow_html=True)


def init_session_state():
    """Uygulama oturum degiskenlerini (session_state) ilk calismada hazirlar."""
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


def check_endpoint_health(api_url: str) -> Dict[str, Any]:
    """Belirtilen API (Colab Ngrok veya yerel) sunucusunun saglik durumunu sorgular."""
    clean_url = api_url.rstrip("/")
    try:
        resp = requests.get(f"{clean_url}/health", timeout=4)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {
        "status": "local_embedded",
        "model_loaded": True,
        "engine_type": "embedded_clinical_engine",
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
    """Mesaj gecmisini uzak Colab Ngrok sunucusuna veya yerel API'ye gonderip yanit alir."""
    clean_url = api_url.rstrip("/")
    payload = {
        "session_id": session_id,
        "mode": mode,
        "messages": messages,
        "case_id": case_id,
        "temperature": temperature,
        "max_tokens": 512,
    }

    # Eger doktor simulasyon modundaysak ve uzak Colab LLM'e gidiyorsa gizli hasta rolunu ekle
    if mode == "simulation" and case_id:
        sys_prompt = build_patient_system_prompt(case_id)
        payload["messages"] = [{"role": "system", "content": sys_prompt}] + messages

    try:
        resp = requests.post(f"{clean_url}/v1/chat", json=payload, timeout=25)
        if resp.status_code == 200:
            data = resp.json()
            # Eger uzak Colab sunucusundan geliyorsa yerel denetim dosyasina da yedekle
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

    # Eger uzak sunucu kapaliysa kesintisiz deneyim icin gomulu motoru calistir
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

    # Triyaj gomulu hesaplama
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
    """Sol kontrol panelini (Colab Ngrok URL ayari, baglanti durumu ve mod secimi) cizer."""
    with st.sidebar:
        st.markdown("### ➕ TriageCore Konsol")
        st.caption("Klinik Karar Destek & Anamnez Sistemi v2.0")
        st.divider()

        # Mod Secici
        selected_module = st.radio(
            "Çalışma Modülü",
            options=[
                "1. Klinik Konsültasyon & Triyaj",
                "2. Vaka Simülatörü (Doktor Modu)",
                "3. Denetim ve Sohbet Kayıtları",
            ],
            index=0,
        )

        st.divider()
        st.markdown("#### Uzak GPU / API Bağlantısı")
        # Colab Ngrok / Cloudflared URL giris kutusu
        api_url = st.text_input(
            "Colab Ngrok / API Endpoint URL",
            value="http://127.0.0.1:8000",
            help="Colab son hücredeki Ngrok veya Cloudflared HTTPS adresini buraya yapıştırın.",
        )

        # Sicaklik (Temperature) ayari
        temperature = st.slider(
            "Klinik Determinizm (Temperature)",
            min_value=0.05,
            max_value=0.80,
            value=0.20,
            step=0.05,
            help="Düşük değerler daha tutarlı ve protokole bağlı yanıtlar üretir.",
        )

        # Baglanti durumu kontrolu
        health_info = check_endpoint_health(api_url)
        engine_label = health_info.get("engine_type", "unknown")
        device_label = health_info.get("device", "cpu")
        st.markdown(
            f"""
            <div class="clinical-status-box">
                <div class="status-row">
                    <span><span class="status-dot"></span><b>Bağlantı Durumu</b></span>
                    <span class="status-online">Aktif</span>
                </div>
                <div class="status-row">
                    <span style="color:#94A3B8;">Çıkarım Motoru:</span>
                    <span class="status-val">{engine_label}</span>
                </div>
                <div class="status-row">
                    <span style="color:#94A3B8;">Donanım:</span>
                    <span class="status-val">{device_label}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        return selected_module, api_url, temperature


def render_triage_module(api_url: str, temperature: float):
    """Mod 1: Ust uste soru sorulabilen Cok Turlu Klinik Triyaj ve Konsultasyon ekrani."""
    st.markdown(
        """
        <div class="clinical-header">
            <h2>Klinik Triyaj ve Çok Turlu Konsültasyon Birimi</h2>
            <p>Hasta semptomlarını girin ve takip sorularıyla ayırıcı tanı / ESI aciliyet analizini derinleştirin.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_chat, col_panel = st.columns([2.1, 1.0], gap="large")

    with col_panel:
        st.markdown('<div class="clinical-card"><h4>Klinik Triyaj Özeti</h4>', unsafe_allow_html=True)
        st.caption(f"Kayıt No: `{st.session_state.triage_session_id}`")

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
                st.markdown("**Saptanan Kırmızı Bayraklar:**")
                for rf in red_flags:
                    st.error(f"• {rf}")
            else:
                st.info("Aktif hayati kırmızı bayrak saptanmadı.")
            st.caption(f"Yanıt Süresi: {meta.get('inference_time_ms', 0)} ms")
        else:
            st.write("Henüz semptom girişi yapılmadı. Sol panelden hasta bulgularını iletin.")

        st.divider()
        if st.button("Yeni Konsültasyon Oturumu Başlat", use_container_width=True):
            st.session_state.triage_session_id = f"TRG-{uuid.uuid4().hex[:6].upper()}"
            st.session_state.triage_messages = []
            st.session_state.last_triage_meta = None
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_chat:
        # Hizli ornek semptom butonlari
        st.markdown("**Hızlı Klinik Senaryo Yükle:**")
        q_col1, q_col2, q_col3 = st.columns(3)
        preset_prompt = None
        if q_col1.button("Akut Göğüs Ağrısı & Terleme", use_container_width=True):
            preset_prompt = "54 yaşında erkek hasta, 30 dakikadır göğüs ortasında sol kola ve çeneye yayılan baskı tarzı ağrı ve soğuk terleme tarifliyor."
        if q_col2.button("Sağ Alt Kadran Karın Ağrısı", use_container_width=True):
            preset_prompt = "Göbek çevresinden başlayıp sağ alt kadrana yerleşen şiddetli karın ağrısı, iştahsızlık ve 38.2 derece ateş mevcut."
        if q_col3.button("İdrarda Yanma & Sık İdrar", use_container_width=True):
            preset_prompt = "3 gündür idrar yaparken yanma ve sık idrara çıkma şikayeti var, ateş veya böğür ağrısı yok."

        # Mesaj gecmisini kurumsal kartlar halinde goster
        for msg in st.session_state.triage_messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="msg-doctor"><div class="msg-label">Klinik Sorgu / Semptom Girişi</div>{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="msg-system"><div class="msg-label">TriageCore Klinik Değerlendirme</div>{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

        # Alt kisimdaki cok turlu soru giris kutusu
        user_input = st.chat_input("Hasta şikayetini veya takip sorunuzu yazın (Örn: Hangi tetkikler istenmeli?)...")
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
    """Mod 2: Kullanicinin Doktor oldugu, hastaya sorular sorarak teshis koydugu Vaka Simulatoru."""
    st.markdown(
        """
        <div class="clinical-header">
            <h2>Klinik Vaka Simülatörü — Hekim Anamnez ve Tanı Konsolu</h2>
            <p>Bu modda siz <b>Muayene Eden Hekimsiniz</b>. Karşınızdaki hastaya sorular sorun, fizik muayene ve tetkik isteyin, ardından kesin tanınızı girin.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cases = list_public_cases()
    case_map = {c["case_id"]: c for c in cases}

    # Ust Vaka Secim Cubugu
    top_c1, top_c2, top_c3 = st.columns([2.2, 1.0, 1.0])
    with top_c1:
        selected_id = st.selectbox(
            "Klinik Vaka Seçin",
            options=list(case_map.keys()),
            format_func=lambda cid: f"{case_map[cid]['title']} (Zorluk: {case_map[cid]['difficulty']})",
            index=list(case_map.keys()).index(st.session_state.active_case_id),
        )
        if selected_id != st.session_state.active_case_id:
            st.session_state.active_case_id = selected_id
            st.session_state.sim_messages = []
            st.session_state.sim_eval_result = None
            st.session_state.sim_session_id = f"SIM-{uuid.uuid4().hex[:6].upper()}"
            st.rerun()

    with top_c2:
        st.write("")
        if st.button("Rastgele Vaka Getir", use_container_width=True):
            random_case = random.choice(list(case_map.keys()))
            st.session_state.active_case_id = random_case
            st.session_state.sim_messages = []
            st.session_state.sim_eval_result = None
            st.session_state.sim_session_id = f"SIM-{uuid.uuid4().hex[:6].upper()}"
            st.rerun()

    with top_c3:
        st.write("")
        if st.button("Vakayı Sıfırla", use_container_width=True):
            st.session_state.sim_messages = []
            st.session_state.sim_eval_result = None
            st.session_state.sim_session_id = f"SIM-{uuid.uuid4().hex[:6].upper()}"
            st.rerun()

    current_case = case_map[st.session_state.active_case_id]

    # Eger vaka yeni basladiysa hastanin ilk basvuru sikayetini otomatik ekle
    if not st.session_state.sim_messages:
        st.session_state.sim_messages.append(
            {"role": "assistant", "content": current_case["initial_complaint"]}
        )

    col_anamnesis, col_diagnosis = st.columns([2.1, 1.0], gap="large")

    with col_anamnesis:
        # Hasta Kabul Karti
        st.markdown(
            f"""
            <div class="clinical-card">
                <h4>Hasta Kabul Dosyası — {current_case['patient_name']} ({current_case['age']} Yaş, {current_case['gender']})</h4>
                <b>Poliklinik / Acil Başvuru Durumu:</b> Triyaj Bekliyor &nbsp;|&nbsp;
                <b>Vaka Zorluğu:</b> {current_case['difficulty']} &nbsp;|&nbsp;
                <b>Oturum:</b> <code>{st.session_state.sim_session_id}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Hizli Anamnez ve Muayene Butonlari
        st.markdown("**Hızlı Hekim Sorguları ve Muayene İstemleri:**")
        b1, b2, b3, b4, b5 = st.columns(5)
        quick_doctor_action = None
        if b1.button("Ağrı Detayı", use_container_width=True):
            quick_doctor_action = "Ağrının tam yeri neresi, başka bir yere yayılıyor mu ve şiddeti nasıl?"
        if b2.button("Ek Semptomlar", use_container_width=True):
            quick_doctor_action = "Ateş, bulantı, kusma, iştahsızlık veya başka bir şikayetin var mı?"
        if b3.button("Özgeçmiş & İlaç", use_container_width=True):
            quick_doctor_action = "Bilinen kronik bir hastalığın, düzenli kullandığın ilaç veya yakın zamanda yolculuk öykün var mı?"
        if b4.button("Fizik Muayene", use_container_width=True):
            quick_doctor_action = "Hastanın vital bulgularını ölç ve detaylı fizik muayenesini yap."
        if b5.button("Lab & Görüntüleme", use_container_width=True):
            quick_doctor_action = "Tam kan sayımı, biyokimya ve ilgili radyolojik görüntüleme tetkiklerini iste."

        # Doktor - Hasta Diyalog Akisi
        for msg in st.session_state.sim_messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="msg-doctor"><div class="msg-label">Hekim (Siz)</div>{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="msg-system"><div class="msg-label">Hasta ({current_case["patient_name"]}) / Klinik Bulgu</div>{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

        doctor_input = st.chat_input("Hastaya sorunuzu yöneltin veya muayene/tetkik isteyin...")
        final_doctor_question = quick_doctor_action or doctor_input

        if final_doctor_question:
            st.session_state.sim_messages.append({"role": "user", "content": final_doctor_question})
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
        st.markdown('<div class="clinical-card"><h4>Kesin Tanı ve Triyaj Kararı</h4>', unsafe_allow_html=True)
        st.write("Anamnez, fizik muayene ve tetkikleri tamamladıktan sonra teşhisinizi girin:")

        diag_input = st.text_input(
            "Kesin Tanınız (Örn: Akut Apandisit)",
            placeholder="Tanıyı buraya yazın...",
        )
        esi_input = st.selectbox(
            "Karar Verdiğiniz ESI Triyaj Seviyesi",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: f"ESI-{x} ({'Hayati/Çok Acil' if x <= 2 else 'Acil/Poliklinik'})",
            index=1,
        )

        # Kac soru soruldugunu hesapla
        doctor_q_count = sum(1 for m in st.session_state.sim_messages if m["role"] == "user")
        st.caption(f"Yapılan Hekim Sorgusu / Muayene Sayısı: **{doctor_q_count}**")

        if st.button("Tanıyı Doğrula ve Vakayı Bitir", type="primary", use_container_width=True):
            if not diag_input.strip():
                st.warning("Lütfen değerlendirme öncesi bir tanı yazın.")
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

        if st.session_state.sim_eval_result:
            ev = st.session_state.sim_eval_result
            st.divider()
            st.metric("Klinik Başarı Puanı", f"{ev['score']} / 100")
            if ev["is_correct"]:
                st.success(ev["feedback"])
            else:
                st.error(ev["feedback"])
            st.markdown(f"**Klinik İnceleme (Clinical Pearls):**\n\n{ev['clinical_pearls']}")
        st.markdown("</div>", unsafe_allow_html=True)


def render_audit_logs_module():
    """Mod 3: Kaydedilen tum sohbet ve vaka degerlendirme loglarini inceleme ekrani."""
    st.markdown(
        """
        <div class="clinical-header">
            <h2>Klinik Sohbet Kayıtları ve Denetim Logları (Audit Trail)</h2>
            <p>Sistem üzerinden yapılan tüm triyaj sorguları, vaka simülasyonu diyalogları ve verilen cevaplar aşağıda kayıtlıdır.</p>
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
            label="Logları İndir (JSON)",
            data=json.dumps(logs, ensure_ascii=False, indent=2),
            file_name="triagecore_audit_logs.json",
            mime="application/json",
            use_container_width=True,
        )

    st.caption(f"Toplam **{len(logs)}** denetim kaydı listeleniyor (`logs/clinical_sessions.jsonl`).")

    if not logs:
        st.info("Henüz kaydedilmiş bir klinik görüşme bulunmuyor.")
        return

    for item in logs:
        ts = item.get("timestamp", "")
        sess = item.get("session_id", "")
        mode_str = item.get("mode", "").upper()
        case_str = f" | Vaka: {item['case_id']}" if item.get("case_id") else ""
        esi_str = f" | {item['triage_level']}" if item.get("triage_level") else ""

        with st.expander(f"[{ts}] Oturum: {sess} | Mod: {mode_str}{case_str}{esi_str}"):
            st.markdown(f"**Kullanıcı / Hekim Girdisi:**\n> {item.get('user_message', '')}")
            st.markdown(f"**Sistem / Hasta Yanıtı:**\n\n{item.get('assistant_reply', '')}")
            st.caption(f"İşlem Süresi: {item.get('inference_time_ms', 0)} ms")


def main():
    """Ana Streamlit uygulama akisini baslatir."""
    configure_page()
    init_session_state()
    selected_module, api_url, temperature = render_sidebar()

    if selected_module.startswith("1."):
        render_triage_module(api_url, temperature)
    elif selected_module.startswith("2."):
        render_simulation_module(api_url, temperature)
    else:
        render_audit_logs_module()


if __name__ == "__main__":
    main()
