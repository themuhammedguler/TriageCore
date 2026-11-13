# ==============================================================================
# TriageCore - Faz 4: FastAPI Uretim Seviyesi Web Servisi
# ==============================================================================
# Bu modul, istemcilerden gelen HTTP isteklerini karsilayan, CORS politikalarini
# yoneten, saglik denetimi yapan ve hem senkron hem de Server-Sent Events (SSE)
# streaming endpointlerini sunan ana FastAPI uygulamasidir.
# ==============================================================================

# Asenkron yasam dongusu (lifespan) yonetimi icin contextmanager yuklenir
from contextlib import asynccontextmanager

# FastAPI uygulama sinifi ve streaming yanit nesnesi yuklenir
from fastapi import FastAPI, HTTPException, status

# FastAPI CORS (Cross-Origin Resource Sharing) guvenlik katmani yuklenir
from fastapi.middleware.cors import CORSMiddleware

# SSE streaming destegi icin StreamingResponse yuklenir
from fastapi.responses import StreamingResponse

# Konfigurasyon ayarlari ice aktarilir
from src.api.config import settings

# Cikarim motoru tekil nesnesi ice aktarilir
from src.api.engine import engine

# Istek ve yanit Pydantic semalari ice aktarilir
from src.api.schemas import (
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
    description="Acik agirlikli LLM tabanli, ESI uyumlu Tibbi Triyaj ve Karar Destek API'si.",
    lifespan=lifespan,  # Yasam dongusu yoneticisi
)

# Tum istemcilere (Web/Mobil/Frontend) izin veren CORS ayarlari eklenir
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
