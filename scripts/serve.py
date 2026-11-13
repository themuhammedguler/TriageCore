# ==============================================================================
# TriageCore - FastAPI Canli Sunucu Baslatici (Server Runner)
# ==============================================================================
# Bu script, Uvicorn asenkron sunucusunu baslatir ve Swagger / OpenAPI
# interaktif dokumantasyon arayuzunu yerel tarayicida erisilebilir kilar.
# ==============================================================================

# Asenkron ASGI sunucusu Uvicorn ice aktarilir
import uvicorn

# Proje konfigurasyon ayarlari ice aktarilir
from src.api.config import settings


def start_server():
    """Uvicorn ASGI sunucusunu yapilandirip calistirir."""
    # Bilgilendirme mesaji konsola yazdirilir
    print("=" * 70)
    print(f"  {settings.app_name} v{settings.app_version}")
    print("=" * 70)
    print(f"[*] API Sunucusu: http://{settings.host}:{settings.port}")
    print(f"[*] Swagger UI  : http://{settings.host}:{settings.port}/docs")
    print(f"[*] Redoc UI    : http://{settings.host}:{settings.port}/redoc")
    print(f"[*] Saglik Testi: http://{settings.host}:{settings.port}/health")
    print("-" * 70)
    print("[*] Sunucu baslatiliyor... (Durdurmak icin CTRL+C basin)")

    # Uvicorn sunucusu calistirilir
    uvicorn.run(
        "src.api.main:app",  # FastAPI uygulama modulu
        host=settings.host,  # Dinlenecek host adresi (0.0.0.0)
        port=settings.port,  # Dinlenecek port (8000)
        reload=False,  # Canli ortamda stabil calisma icin
        log_level="info",  # Bilgilendirme log seviyesi
    )


# Script dogrudan calistirildiginda start_server tetiklenir
if __name__ == "__main__":
    start_server()
