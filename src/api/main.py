# ==============================================================================
# TriageCore - Faz 4 & 6: FastAPI Uretim Seviyesi Web Servisi
# ==============================================================================
# Bu modul, istemcilerden gelen HTTP isteklerini karsilayan, CORS politikalarini
# yoneten, saglik denetimi yapan, cok turlu sohbet, Doktor-Hasta vaka simulasyonu
# ve denetim loglari endpointlerini sunan ana FastAPI uygulamasidir.
# ==============================================================================

# Asenkron yasam dongusu (lifespan) yonetimi icin contextmanager yuklenir
from contextlib import asynccontextmanager

# Opsiyonel sorgu parametreleri icin Optional yuklenir
from typing import Optional

# FastAPI uygulama sinifi ve HTTP hata bilesenleri yuklenir
from fastapi import FastAPI, HTTPException, status

# FastAPI CORS (Cross-Origin Resource Sharing) guvenlik katmani yuklenir
from fastapi.middleware.cors import CORSMiddleware

# SSE streaming destegi icin StreamingResponse yuklenir
from fastapi.responses import StreamingResponse

# Doktor simulasyonu vaka ve degerlendirme fonksiyonlari ice aktarilir
from src.api.cases import evaluate_doctor_submission, list_public_cases

# Konfigurasyon ayarlari ice aktarilir
from src.api.config import settings

# Cikarim motoru tekil nesnesi ice aktarilir
from src.api.engine import engine

# Denetim loglayici nesnesi ice aktarilir
from src.api.logger import audit_logger

# Istek ve yanit Pydantic semalari ice aktarilir
from src.api.schemas import (
    CaseEvaluationRequest,
    CaseEvaluationResponse,
    ChatSessionRequest,
    ChatSessionResponse,
    HealthResponse,
    TriageRequest,
    TriageResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama ayaga kalkarken modeli belleğe yukler, kapanirken kaynaklari temizler."""
    # Sunucu baslarken model yukleme fonksiyonu calistirilir
    print("[*] FastAPI sunucusu baslatiliyor. Cikarim motoru hazirlaniyor...")
    engine.load_model()
    # Sunucu istekleri kabul etmeye baslar
    yield
    # Sunucu kapanirken kaynaklar temizlenir
    print("[*] FastAPI sunucusu kapatiliyor. Kaynaklar serbest birakildi.")


# FastAPI uygulama nesnesi olusturuluyor
app = FastAPI(
    title=settings.app_name,  # Uygulama basligi
    version=settings.app_version,  # Uygulama versiyonu
    description="Acik agirlikli LLM tabanli, ESI uyumlu Tibbi Triyaj, Cok Turlu Konsultasyon ve Doktor Vaka Simulatoru API'si.",
    lifespan=lifespan,  # Yasam dongusu yoneticisi
)

# Tum istemcilere (Web/Mobil/Streamlit) izin veren CORS ayarlari eklenir
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Herhangi bir kaynaktan gelen isteklere izin ver
    allow_credentials=True,  # Kimlik bilgilerine izin ver
    allow_methods=["*"],  # Tum HTTP metodlarina izin ver (GET, POST, OPTIONS vb.)
    allow_headers=["*"],  # Tum basliklara izin ver
)


@app.get("/", tags=["General"])
async def root():
    """Karsilama ve genel API bilgi endpointi."""
    # API temel bilgisi dondurulur
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "documentation": "/docs",
        "health": "/health",
        "status": "online",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Sunucu saglik, model yuklenme ve donanim durumunu donduren denetim endpointi."""
    # Sunucu ve motor durumu dondurulur
    return HealthResponse(
        status="healthy" if engine.is_loaded else "degraded",
        model_loaded=engine.is_loaded,
        engine_type=engine.engine_type,
        device=engine.device,
        version=settings.app_version,
    )


@app.post("/v1/triage", response_model=TriageResponse, tags=["Triage"])
async def evaluate_triage(request: TriageRequest):
    """Hasta semptomlarini analiz ederek yapilandirilmis ESI triyaj ciktisi dondurur."""
    try:
        # Asenkron cikarim motoruna istek iletilir
        response = await engine.predict_triage(request)
        # Yapilandirilmis Pydantic nesnesi dondurulur
        return response
    except Exception as exc:
        # Beklenmedik hatalarda 500 kodu dondurulur
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Triyaj cikarimi sirasinda hata olustu: {str(exc)}",
        )


@app.post("/v1/chat", response_model=ChatSessionResponse, tags=["Multi-Turn & Simulation"])
async def multi_turn_chat(request: ChatSessionRequest):
    """Ust uste sorulan takip sorularini (triage) veya Doktor-Hasta vaka diyalogunu (simulation) yanitlar."""
    try:
        # Cok turlu diyalog motoru cagirilir
        response = await engine.predict_chat_session(request)
        return response
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Diyalog cikarimi sirasinda hata olustu: {str(exc)}",
        )


@app.get("/v1/cases", tags=["Simulation"])
async def get_simulation_cases():
    """Doktor simulasyonu modu icin mevcut hasta vakalarini (gizli tanilari gizleyerek) listeler."""
    # Vaka listesi dondurulur
    return {"cases": list_public_cases()}


@app.post("/v1/cases/evaluate", response_model=CaseEvaluationResponse, tags=["Simulation"])
async def evaluate_case_diagnosis(request: CaseEvaluationRequest):
    """Doktor rolundeki kullanicinin vaka sonunda koydugu teshisi ve ESI kararini degerlendirir."""
    try:
        # Degerlendirme motoru calistirilir
        result = evaluate_doctor_submission(
            case_id=request.case_id,
            submitted_diagnosis=request.submitted_diagnosis,
            submitted_esi=request.submitted_esi,
            question_count=request.question_count,
        )
        # Degerlendirme sonucunu denetim loglarina kaydet
        audit_logger.log_interaction(
            session_id=request.session_id,
            mode="case_evaluation",
            case_id=request.case_id,
            user_message=f"Tanı: {request.submitted_diagnosis} | Seçilen ESI: {request.submitted_esi}",
            assistant_reply=f"Skor: {result['score']}/100 | Doğru Tanı: {result['correct_diagnosis']} | {result['feedback']}",
            urgency_score=result["expected_esi"],
        )
        return CaseEvaluationResponse(**result)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )


@app.get("/v1/logs", tags=["Audit & Logging"])
async def get_audit_logs(
    limit: int = 100,
    mode: Optional[str] = None,
    session_id: Optional[str] = None,
):
    """Kaydedilen tum sohbet ve vaka cozum loglarini filtreleyerek dondurur."""
    # Log kayitlari okunur
    logs = audit_logger.get_logs(limit=limit, mode=mode, session_id=session_id)
    return {"count": len(logs), "logs": logs}


@app.post("/v1/triage/stream", tags=["Triage"])
async def stream_triage_tokens(request: TriageRequest):
    """Server-Sent Events (SSE) protokolunde token bazinda anlik triyaj akisi saglar."""
    try:
        # Cikarim motorunun asenkron generator fonksiyonu cagirilir
        generator = engine.stream_triage(request)
        # StreamingResponse nesnesi text/event-stream tipiyle dondurulur
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",  # Tarayici onbellegini devre disi birak
                "Connection": "keep-alive",  # Baglantiyi acik tut
            },
        )
    except Exception as exc:
        # Hata durumunda 500 kodu dondurulur
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Streaming akisi sirasinda hata olustu: {str(exc)}",
        )
