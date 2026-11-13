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
