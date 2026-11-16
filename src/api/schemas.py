# ==============================================================================
# TriageCore - Faz 4: API İstek ve Yanıt Pydantic Şemaları
# ==============================================================================
# Bu modul, istemciden gelen istekleri (Request) dogrulamak ve yapilandirilmis
# klinik triyaj ciktisini (Response) tip guvenlikli JSON olarak dondurmek icin
# Pydantic modellerini tanimlar.
# ==============================================================================

# Enum sabitleri icin enum modulu yuklenir
from enum import Enum

# Tip ipuclari icin List ve Optional yuklenir
from typing import List, Optional

# Pydantic tabanli dogrulayicilar ve veri modelleri
from pydantic import BaseModel, Field, field_validator


class EmergencySeverityLevel(str, Enum):
    """Acil Servis Triyaj Seviyeleri (Emergency Severity Index - ESI 1-5)."""

    LEVEL_1_RESUSCITATION = "ESI-1: Resuscitation (Hayati Tehlike / Derhal Mudahale)"
    LEVEL_2_EMERGENT = "ESI-2: Emergent (Cok Acil / Dakikalar Icinde)"
    LEVEL_3_URGENT = "ESI-3: Urgent (Orta Derece Acil / 1 Saat Icinde)"
    LEVEL_4_LESS_URGENT = "ESI-4: Less Urgent (Dusuk Aciliyet / Poliklinik)"
    LEVEL_5_NON_URGENT = "ESI-5: Non-Urgent (Acil Degil / Rutin Bakim)"


class TriageRequest(BaseModel):
    """Istemciden gelen hasta semptom degerlendirme istegi."""

    # Hastanin sikayet ve semptom metni (zorunlu alan)
    patient_symptoms: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Hastanin detayli semptom ve sikayet aciklamasi",
        examples=["Gogsumun ortasinda 30 dakikadir sol koluma vuran siddetli bir agri ve soguk terleme var."],
    )

    # Hastanin yasi (opsiyonel)
    patient_age: Optional[int] = Field(
        default=None,
        ge=0,
        le=125,
        description="Hastanin yasi",
        examples=[45],
    )

    # Hastanin cinsiyeti (opsiyonel)
    patient_gender: Optional[str] = Field(
        default=None,
        description="Hastanin cinsiyeti (Erkek / Kadin / Diger)",
        examples=["Erkek"],
    )

    # Hastanin bilinen kronik hastaliklari listesi (opsiyonel)
    existing_conditions: Optional[List[str]] = Field(
        default_factory=list,
        description="Diyabet, hipertansiyon, astim gibi bilinen tanilar",
        examples=[["Hipertansiyon", "Tip 2 Diyabet"]],
    )

    # Uretim sicakligi (temperature): Dusuk degerler klinik determinizmi artirir
    temperature: Optional[float] = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Model uretim sicakligi (0.0 en determinist, 1.0 en yaratici)",
    )

    # Uretilecek maksimum token sayisi
    max_tokens: Optional[int] = Field(
        default=512,
        ge=64,
        le=1024,
        description="Uretilecek maksimum token adedi",
    )

    @field_validator("patient_symptoms")
    @classmethod
    def validate_symptoms_not_blank(cls, v: str) -> str:
        """Semptom metninin sadece bosluklardan olusmadigini dogrular."""
        cleaned = v.strip()
        if len(cleaned) < 5:
            raise ValueError("Semptom aciklamasi en az 5 anlamli karakter icermelidir.")
        return cleaned


class TriageResponse(BaseModel):
    """Klinik triyaj degerlendirmesini iceren yapilandirilmis API yaniti."""

    # ESI standartlarinda aciliyet seviyesi
    triage_level: EmergencySeverityLevel
    # Sayisal aciliyet skoru (1: En acil, 5: En dusuk aciliyet)
    urgency_score: int = Field(ge=1, le=5)
    # Acil durum bayragi (ESI 1 ve 2 icin True)
    is_emergency: bool
    # Tespit edilen kritik kirmizi bayrak semptomlar
    red_flags: List[str]
    # Klinik on degerlendirme ve olasi patolojiler
    clinical_assessment: str
    # Onerilen acil veya poliklinik aksiyon plani
    recommended_action: str
    # Yasal klinik feragatname metni
    disclaimer: str
    # Istegin islenme suresi (milisaniye cinsinden)
    inference_time_ms: float


class HealthResponse(BaseModel):
    """Sunucu saglik ve hazirlik durumu yaniti."""

    status: str = Field(description="Sunucu durumu (healthy / degraded)")
    model_loaded: bool = Field(description="Modelin bellekte hazir olup olmadigi")
    engine_type: str = Field(description="Calisan cikarim motoru (llama.cpp / mock)")
    device: str = Field(description="Kullanilan donanim (cuda / cpu)")
    version: str = Field(description="API surum bilgisi")



class ChatTurn(BaseModel):
    """Cok turlu (multi-turn) diyalogdaki tek bir mesaj adimi."""

    # Mesaj rolu: system, user veya assistant
    role: str = Field(..., description="Mesaj rolu (system / user / assistant)")
    # Mesaj metni
    content: str = Field(..., min_length=1, description="Mesaj metni")


class ChatSessionRequest(BaseModel):
    """Cok turlu klinik triyaj veya Doktor-Hasta vaka simulasyonu istegi."""

    # Konusma oturum kimligi (log takibi icin)
    session_id: str = Field(default="default-session", description="Tekil oturum kimligi")
    # Calisma modu: 'triage' (Hasta -> Doktor) veya 'simulation' (Doktor -> Hasta)
    mode: str = Field(default="triage", description="Calisma modu: triage veya simulation")
    # Konusma gecmisini iceren mesaj listesi
    messages: List[ChatTurn] = Field(..., min_length=1, description="Onceki ve guncel mesajlar")
    # Simulasyon modundaysa secilen vaka kimligi (orn. CASE-001)
    case_id: Optional[str] = Field(default=None, description="Doktor simulasyon vaka kimligi")
    # Model uretim sicakligi
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    # Maksimum uretilecek token adedi
    max_tokens: int = Field(default=512, ge=64, le=1024)


class ChatSessionResponse(BaseModel):
    """Cok turlu klinik diyalog yanit modeli."""

    # Oturum kimligi
    session_id: str
    # Calisma modu
    mode: str
    # Modelin veya hasta simulatorunun urettigi yanit metni
    reply: str
    # Triyaj modunda hesaplanan guncel ESI seviyesi
    triage_level: Optional[EmergencySeverityLevel] = None
    # Triyaj modunda sayisal aciliyet skoru (1-5)
    urgency_score: Optional[int] = None
    # Acil durum bayragi
    is_emergency: bool = False
    # Saptanan kirmizi bayraklar
    red_flags: List[str] = Field(default_factory=list)
    # Yanit uretim suresi (ms)
    inference_time_ms: float


class CaseEvaluationRequest(BaseModel):
    """Doktor vaka simulasyonu sonunda kullanicinin girdigi teshis degerlendirme istegi."""

    # Oturum kimligi
    session_id: str = Field(default="sim-session")
    # Cozulmeye calisilan vaka kimligi
    case_id: str = Field(..., description="Degerlendirilecek vaka ID (orn. CASE-001)")
    # Doktorun (kullanicinin) koydugu teshis
    submitted_diagnosis: str = Field(..., min_length=2, description="Kullanicinin yazdigi teshis")
    # Doktorun belirledigi ESI triyaj aciliyet seviyesi (1-5)
    submitted_esi: int = Field(..., ge=1, le=5, description="Kullanicinin sectigi ESI seviyesi")
    # Vaka boyunca hastaya sorulan anamnez soru sayisi
    question_count: int = Field(default=1, ge=0)


class CaseEvaluationResponse(BaseModel):
    """Doktorun koydugu teshisin puanlama ve klinik geri bildirim yaniti."""

    case_id: str
    is_correct: bool
    score: int
    submitted_diagnosis: str
    correct_diagnosis: str
    submitted_esi: int
    expected_esi: int
    feedback: str
    clinical_pearls: str
