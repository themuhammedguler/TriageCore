# ==============================================================================
# TriageCore - Faz 4: API Konfigurasyon ve Ortam Ayarlari
# ==============================================================================
# Bu modul, FastAPI sunucusunun calisma parametrelerini, model dosya yollarini
# ve eszamanli istek (concurrency) limitlerini yonetir.
# ==============================================================================

# Isletim sistemi ortam degiskenlerine erismek icin os modulu yuklenir
import os

# Dosya yollari islemleri icin Path modulu yuklenir
# Pydantic tabanli model konfigurasyon sinifi
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """FastAPI sunucusu icin global yapilandirma parametreleri."""

    # API servis adi
    app_name: str = Field(default="TriageCore Clinical Inference API")
    # API versiyon numarasi
    app_version: str = Field(default="1.0.0")

    # Sunucunun dinleyecegi IP adresi (0.0.0.0 dis erisime acar)
    host: str = Field(default=os.getenv("HOST", "0.0.0.0"))
    # Sunucunun dinleyecegi port numarasi
    port: int = Field(default=int(os.getenv("PORT", "8000")))

    # Calistirilacak GGUF model dosyasinin yolu
    model_path: str = Field(
        default=os.getenv("MODEL_PATH", "models/triagecore-q4_k_m.gguf")
    )

    # Model yuklenemediginde veya yerel testlerde calisacak akilli klinik motor modu
    use_mock_engine: bool = Field(
        default=os.getenv("USE_MOCK_ENGINE", "false").lower() in ("true", "1", "yes")
    )

    # Eszamanli (concurrency) islenebilecek maksimum istek sayisi (GPU boğulmasını onler)
    max_concurrent_requests: int = Field(
        default=int(os.getenv("MAX_CONCURRENT_REQUESTS", "4"))
    )

    # Varsayilan maksimum uretilecek token uzunlugu
    default_max_tokens: int = Field(default=512)
    # Klinik tutarliligi korumak icin varsayilan dusuk sicaklik (temperature)
    default_temperature: float = Field(default=0.2)

    # Yasal klinik feragatname metni
    disclaimer_text: str = Field(
        default=(
            "TriageCore yapay zeka destekli bir on-triyaj karar destek aracidir. "
            "Kesin tibbi tani veya hekim muayenesi yerine gecmez. "
            "Acil hayati tehlike durumlarinda derhal 112 / 911 acil servisi arayiniz."
        )
    )


# Tekil (Singleton) ayarlar nesnesi olusturuluyor
settings = Settings()
