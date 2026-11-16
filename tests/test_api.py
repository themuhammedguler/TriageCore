# ==============================================================================
# TriageCore - Faz 4: API Endpoint ve Streaming Entegrasyon Testleri
# ==============================================================================
# Bu test dosyasi, FastAPI uygulamasini Starlette TestClient uzerinden test eder;
# HTTP durumlari, Pydantic sema dogrulamasi, ESI siniflandirmasi ve SSE streaming
# davranislarini otomatik dogrular.
# ==============================================================================

# Pytest test cercevesi ice aktarilir
import pytest

# FastAPI ve Starlette test istemcisi ice aktarilir
from starlette.testclient import TestClient

# Test edilecek FastAPI uygulamasi ice aktarilir
from src.api.main import app

# Aciliyet seviyeleri enum sabiti ice aktarilir
from src.api.schemas import EmergencySeverityLevel


@pytest.fixture
def client():
    """FastAPI lifespan yasam dongusunu aktif eden test client fixture'i."""
    # Lifespan ile model yukleme islemi baslatilir
    with TestClient(app) as test_client:
        # Test fonksiyonuna istemci verilir
        yield test_client


def test_root_endpoint(client):
    """Karsilama endpointinin 200 kodu ve servis bilgilerini dondurdugunu test eder."""
    # GET istegi gonderilir
    response = client.get("/")
    # HTTP 200 durumu dogrulanir
    assert response.status_code == 200
    # Donen JSON sozluk yapisi okunur
    data = response.json()
    # Servis adinin donduruldugu dogrulanir
    assert "TriageCore" in data["service"]
    # Durumun online oldugu dogrulanir
    assert data["status"] == "online"


def test_health_check_endpoint(client):
    """Saglik denetimi endpointinin saglikli durumu ve model bilgisini dondurdugunu test eder."""
    # /health endpointine GET istegi atilir
    response = client.get("/health")
    # HTTP 200 durumu dogrulanir
    assert response.status_code == 200
    # JSON verisi ayristirilir
    data = response.json()
    # Sunucu durumunun 'healthy' oldugu dogrulanir
    assert data["status"] == "healthy"
    # Modelin basariyla yuklendigi dogrulanir
    assert data["model_loaded"] is True
    # Cikarim motoru tipinin donduruldugu dogrulanir
    assert "engine_type" in data


def test_triage_critical_emergency_esi_1(client):
    """Akut gogus agrisi vakasinin ESI-1 Hayati Tehlike olarak siniflandigini test eder."""
    # Kritik kalp krizi semptomlarini iceren test yukumu
    payload = {
        "patient_symptoms": "Gogsumun tam ortasinda sol koluma ve ceneme vuran ezici bir baski var, soguk terliyorum.",
        "patient_age": 52,
        "patient_gender": "Erkek",
        "existing_conditions": ["Hipertansiyon"],
    }
    # /v1/triage endpointine POST istegi yapilir
    response = client.post("/v1/triage", json=payload)
    # HTTP 200 durumu dogrulanir
    assert response.status_code == 200
    # JSON cevabi okunur
    data = response.json()
    # Aciliyet seviyesinin ESI-1 Resuscitation oldugu dogrulanir
    assert data["triage_level"] == EmergencySeverityLevel.LEVEL_1_RESUSCITATION.value
    # Aciliyet skorunun 1 oldugu dogrulanir
    assert data["urgency_score"] == 1
    # is_emergency bayraginin True oldugu dogrulanir
    assert data["is_emergency"] is True
    # Kirmizi bayrak listesinin dolu oldugu dogrulanir
    assert len(data["red_flags"]) > 0
    # Onerilen aksiyonda 112/911 acil cagri uyarisi oldugu dogrulanir
    assert "112" in data["recommended_action"]


def test_triage_non_urgent_esi_5(client):
    """Hafif semptomlarin ESI-5 Acil Olmayan vaka olarak dogru siniflandigini test eder."""
    # Hafif bogaz agrisi semptomu
    payload = {
        "patient_symptoms": "Dunden beri hafif bir bogaz kasintim var, baska hicbir sikayetim yok.",
        "patient_age": 28,
    }
    # POST istegi gonderilir
    response = client.post("/v1/triage", json=payload)
    # HTTP 200 durumu dogrulanir
    assert response.status_code == 200
    # JSON cevabi ayristirilir
    data = response.json()
    # Aciliyet skorunun 5 (en dusuk) oldugu dogrulanir
    assert data["urgency_score"] == 5
    # is_emergency bayraginin False oldugu dogrulanir
    assert data["is_emergency"] is False


def test_triage_validation_rejects_empty_symptoms(client):
    """Gecersiz (cok kisa veya bosluk) semptomlarin 422 hatasi verdigini test eder."""
    # 5 karakterden kisa gecersiz semptom yukumu
    payload = {
        "patient_symptoms": "   ",
    }
    # POST istegi gonderilir
    response = client.post("/v1/triage", json=payload)
    # HTTP 422 Unprocessable Entity durumu dogrulanir
    assert response.status_code == 422


def test_triage_streaming_sse_endpoint(client):
    """Server-Sent Events (SSE) streaming endpointinin token akisi sagladigini test eder."""
    # Test semptomu
    payload = {
        "patient_symptoms": "Ayagimi burktum lateral kisimda sisme var fakat uzerine basabiliyorum.",
    }
    # /v1/triage/stream endpointine POST istegi yapilir
    response = client.post("/v1/triage/stream", json=payload)
    # HTTP 200 basarili baglanti dogrulanir
    assert response.status_code == 200
    # Icerik tipinin text/event-stream oldugu dogrulanir
    assert "text/event-stream" in response.headers["content-type"]
    # Govdede SSE protokolune ait 'data:' satirlarinin geldigi dogrulanir
    assert "data:" in response.text
    # Akis sonunda final bildiriminin geldigi dogrulanir
    assert '"is_final": true' in response.text



def test_multi_turn_triage_chat(client):
    """Ust uste sorulan takip sorularinin (multi-turn) baglami koruyarak yanitlandigini test eder."""
    # Ilk mesajda karin agrisi, ikinci mesajda hangi tahlillerin istenecegi sorulur
    payload = {
        "session_id": "test-multi-turn-01",
        "mode": "triage",
        "messages": [
            {"role": "user", "content": "Siddetli karin agrim var, sag alt tarafa vuruyor."},
            {"role": "assistant", "content": "Acil degerlendirme gerekir."},
            {"role": "user", "content": "Hangi kan tahlilleri yapilmali?"},
        ],
    }
    # /v1/chat endpointine POST istegi atilir
    response = client.post("/v1/chat", json=payload)
    # HTTP 200 durumu dogrulanir
    assert response.status_code == 200
    data = response.json()
    # Aciliyet skorunun 2 (ESI-2) oldugu dogrulanir
    assert data["urgency_score"] == 2
    # Takip sorusuna yanit olarak tetkik onerisi eklendigi dogrulanir
    assert "Hemogram" in data["reply"] or "Tetkikler" in data["reply"]


def test_doctor_simulation_cases_and_patient_reply(client):
    """Doktor simulasyonu vaka listesini ve hasta yanit motorunu test eder."""
    # 1. Vaka listesi cekilir
    cases_resp = client.get("/v1/cases")
    assert cases_resp.status_code == 200
    cases = cases_resp.json()["cases"]
    # En az 5 klinik vakanin hazir oldugu dogrulanir
    assert len(cases) >= 5
    # Gizli teshisin (correct_diagnosis) kullaniciya sizdirilmadigi dogrulanir
    assert "correct_diagnosis" not in cases[0]

    # 2. CASE-001 (Apandisit) hastasina fizik muayene sorusu yoneltilir
    sim_payload = {
        "session_id": "test-sim-01",
        "mode": "simulation",
        "case_id": "CASE-001",
        "messages": [
            {"role": "user", "content": "Hastanin batin fizik muayenesini yap ve vital bulgulara bak."},
        ],
    }
    sim_resp = client.post("/v1/chat", json=sim_payload)
    assert sim_resp.status_code == 200
    sim_data = sim_resp.json()
    # Muayene bulgularinda McBurney ve Rebound ibarelerinin yer aldigi dogrulanir
    assert "McBurney" in sim_data["reply"]


def test_doctor_diagnosis_evaluation(client):
    """Doktorun koydugu teshisin ve ESI kararinin dogru puanlandigini test eder."""
    # CASE-001 icin dogru teshis (Akut Apandisit) ve dogru ESI (2) gonderilir
    eval_payload = {
        "session_id": "test-sim-01",
        "case_id": "CASE-001",
        "submitted_diagnosis": "Akut Apandisit",
        "submitted_esi": 2,
        "question_count": 4,
    }
    resp = client.post("/v1/cases/evaluate", json=eval_payload)
    assert resp.status_code == 200
    data = resp.json()
    # Teshisin dogru kabul edildigi ve tam puan (100) aldigi dogrulanir
    assert data["is_correct"] is True
    assert data["score"] == 100
    assert "Alvarado" in data["clinical_pearls"]


def test_audit_logs_endpoint(client):
    """Yapilan konusmalarin log dosyasina kaydedildigini ve /v1/logs ile okundugunu test eder."""
    # Log listesi sorgulanir
    resp = client.get("/v1/logs?limit=10")
    assert resp.status_code == 200
    data = resp.json()
    # Onceki testlerde yapilan konusmalarin loglandigi dogrulanir
    assert data["count"] > 0
    assert len(data["logs"]) > 0
