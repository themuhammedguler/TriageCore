# ==============================================================================
# TriageCore - Faz 4: Asenkron Cikarim Motoru (Inference Engine)
# ==============================================================================
# Bu modul, GGUF modelleri (llama.cpp) veya gelistirme/test amacli akilli klinik
# kurallar motorunu yukler; eszamanli istekleri Semaphore ile kuyruga alir ve
# Server-Sent Events (SSE) streaming ciktisi uretir.
# ==============================================================================

# Asenkron I/O islemleri icin asyncio modulu yuklenir
import asyncio

# JSON serilestirme islemleri icin json modulu yuklenir
import json

# Zaman olcumu icin time modulu yuklenir
import time

# Dosya varligi kontrolleri icin Path modulu yuklenir
from pathlib import Path

# Asenkron uretec tip ipuclari icin AsyncGenerator yuklenir
from typing import AsyncGenerator, List, Tuple

# API konfigurasyon ayarlari ice aktarilir
from src.api.config import settings

# Istek ve yanit Pydantic semalari ice aktarilir
from src.api.schemas import (
    EmergencySeverityLevel,
    TriageRequest,
    TriageResponse,
)


class TriageInferenceEngine:
    """Asenkron triyaj cikarimlarini yoneten ve streaming ureten motor sinifi."""

    def __init__(self):
        """Motor bilesenlerini ve eszamanlilik semaforunu ilklendirir."""
        # Eszamanli istek sayisini sinirlandirarak GPU/CPU bellek asimini onleyen semafor
        self.semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
        # Model nesnesi referansi
        self.model = None
        # Kullanilan motor tipi etiketi (llama.cpp veya mock/fallback)
        self.engine_type = "clinical_rules_engine"
        # Donanim tipi etiketi (cpu veya cuda)
        self.device = "cpu"
        # Modelin hazir olma durumu
        self.is_loaded = True

    def load_model(self):
        """Diskteki GGUF model dosyasini llama-cpp-python ile VRAM'e/RAM'e yukler."""
        # Model dosya yolu kontrol edilir
        model_file = Path(settings.model_path)

        # Eger dosya varsa ve mock mod zorlanmadiysa gercek GGUF motorunu baslat
        if model_file.exists() and not settings.use_mock_engine:
            try:
                # llama_cpp kutuphanesi dinamik yuklenir
                from llama_cpp import Llama

                print(f"[*] GGUF Modeli yukleniyor: {model_file.resolve()}")
                # Llama cikarim nesnesi olusturulur (n_gpu_layers=-1 tum katmanlari GPU'ya atar)
                self.model = Llama(
                    model_path=str(model_file.resolve()),
                    n_ctx=2048,  # 2048 token baglam penceresi
                    n_threads=4,  # CPU is parcacigi sayisi
                    n_gpu_layers=-1,  # CUDA varsa tum katmanlari GPU'ya yukler
                    verbose=False,
                )
                self.engine_type = "llama.cpp"
                self.device = "cuda" if self.model.n_gpu_layers > 0 else "cpu"
                self.is_loaded = True
                print(f"[+] Model basariyla yuklendi ({self.engine_type} / {self.device})")
                return
            except Exception as e:
                print(f"[!] llama-cpp yuklenirken hata olustu ({e}). Hibrit klinik motor devreye aliniyor.")

        # Model dosyasi henuz indirilmediyse veya CPU testindeysek akilli klinik motor devrededir
        self.engine_type = "clinical_rules_engine"
        self.device = "cpu"
        self.is_loaded = True
        print("[*] TriageCore Klinik Kurallar ve Test Motoru aktif edildi.")

    def _normalize_text(self, text: str) -> str:
        """Turkce karakterleri ASCII karsiliklarina cevirerek klavye farkliliklarini normalize eder."""
        mapping = {
            "ı": "i", "İ": "i", "ğ": "g", "Ğ": "g",
            "ü": "u", "Ü": "u", "ş": "s", "Ş": "s",
            "ö": "o", "Ö": "o", "ç": "c", "Ç": "c",
        }
        lowered = text.lower()
        for tr_char, en_char in mapping.items():
            lowered = lowered.replace(tr_char, en_char)
        return lowered

    def _analyze_clinical_urgency(self, symptoms: str) -> Tuple[EmergencySeverityLevel, int, bool, List[str], str, str]:
        """Klinik semptom metnini analiz ederek ESI seviyesi ve onerileri cikarir."""
        # Hem orijinal hem normalize edilmis metin hazirlanir
        text = self._normalize_text(symptoms)
        red_flags = []

        # 1. Kategori: Hayati Tehlike (ESI-1 / Resuscitation)
        # Gogus agrisi (unlu dusmesi: gogus -> gogsum / gogs), sol kola yayilim, terleme, nefes darligi, bilinc kaybi
        has_cardiac_site = any(w in text for w in ["gogus", "gogs", "chest", "kalp", "retrosternal"])
        has_cardiac_symptom = any(w in text for w in ["baski", "sol kol", "pressure", "radiat", "terle", "cene", "jaw", "koluma", "agri"])
        if has_cardiac_site and has_cardiac_symptom:
            red_flags.extend(["Akut Retrosternal Baskı Tarzı Göğüs Ağrısı", "Sol Kola/Çeneye Yayılım", "Akut Koroner Sendrom Şüphesi"])
            return (
                EmergencySeverityLevel.LEVEL_1_RESUSCITATION,
                1,
                True,
                red_flags,
                "Belirtiler Akut Miyokard Enfarktüsü (Kalp Krizi) veya instabil anjina ile uyumludur. Dakikalar hayati onem tasir.",
                "Derhal 112 / 911 Acil Servisi arayiniz. Kendi basiniza arac kullanmayiniz. Alerjiniz yoksa bir adet aspirin cigneyiniz ve dinlenme pozisyonunda ambulansi bekleyiniz.",
            )

        # Inme / Felc Belirtileri (FAST kriterleri)
        if any(w in text for w in ["yuzde dusme", "facial droop", "felc", "stroke", "peltek", "slurr", "gucsuzluk", "weakness"]):
            red_flags.extend(["Tek Taraflı Yüz Düşmesi", "Konuşma Bozukluğu (Dizartri)", "Akut İskemik İnme Şüphesi"])
            return (
                EmergencySeverityLevel.LEVEL_1_RESUSCITATION,
                1,
                True,
                red_flags,
                "Belirtiler Akut Serebrovasküler Olay (İnme / Felç) tablosuna isaret etmektedir. Trombolitik tedavi penceresi (< 4.5 saat) acisindan acildir.",
                "Derhal 112 / 911 Acil Ambulans cagiriniz. Hastaya kesinlikle yiyecek, su veya ilac vermeyiniz. Belirtilerin basladigi tam saati not ediniz.",
            )

        # 2. Kategori: Cok Acil (ESI-2 / Emergent)
        # Siddetli karin agrisi (unlu dusmesi: karin -> karnim / karn), yuksek atesli cocukta nobet, agir alerjik reaksiyon
        if any(w in text for w in ["siddetli karin", "siddetli karn", "apandisit", "severe abdominal", "nefes alamiyorum", "anaphylaxis", "alerji"]):
            red_flags.append("Akut Karın veya Şiddetli Solunum Sıkıntısı")
            return (
                EmergencySeverityLevel.LEVEL_2_EMERGENT,
                2,
                True,
                red_flags,
                "Akut Batin (Apandisit, Perforasyon) veya ciddi obstrüktif hava yolu reaksiyonu olasiligi yuksektir.",
                "En yakin hastanenin Acil Servisine dakikalar icinde basvurunuz. Bir sey yiyip icmeyiniz.",
            )

        # 3. Kategori: Orta Derece Acil (ESI-3 / Urgent)
        # Idiyopatik ates, kontrolsuz kusma, derin kesikler
        if any(w in text for w in ["yanma", "idrar", "dysuria", "oksuruk", "ates", "fever", "kusma"]):
            return (
                EmergencySeverityLevel.LEVEL_3_URGENT,
                3,
                False,
                red_flags,
                "Enfeksiyoz surecler (Uriner Sistem Enfeksiyonu veya Alt Solunum Yolu Enfeksiyonu) on plandadir.",
                "24 saat icerisinde bir Aile Hekimi veya Dahiliye / Cocuk Poliklinigine basvurunuz. Bol sivi tuketiniz.",
            )

        # 4. Kategori: Dusuk Aciliyet / Rutin (ESI-4 veya ESI-5)
        return (
            EmergencySeverityLevel.LEVEL_5_NON_URGENT,
            5,
            False,
            red_flags,
            "Hayati tehlike veya kirmizi bayrak bulgusu saptanmamistir. Semptomlar hafif viral veya kas-iskelet kaynaklidir.",
            "Evde istirahat ediniz, bol sivi aliniz. Belirtiler 72 saatten uzun surerse rutin poliklinik randevusu aliniz.",
        )

    async def predict_triage(self, request: TriageRequest) -> TriageResponse:
        """Asenkron olarak semptomlari degerlendirir ve yapilandirilmis yanit dondurur."""
        start_time = time.time()

        # Eszamanlilik semaforu ile kuyruk yonetimi
        async with self.semaphore:
            # Yapay bir I/O beklemesi simule edilir (gercek asenkron akis davranisi icin)
            await asyncio.sleep(0.01)

            # Klinik cikarim fonksiyonu calistirilir
            level, score, is_emerg, flags, assess, action = self._analyze_clinical_urgency(
                request.patient_symptoms
            )

            # Gecen sure hesaplanir
            duration_ms = round((time.time() - start_time) * 1000, 2)

            # Yapilandirilmis Pydantic yaniti olusturulur
            return TriageResponse(
                triage_level=level,
                urgency_score=score,
                is_emergency=is_emerg,
                red_flags=flags,
                clinical_assessment=assess,
                recommended_action=action,
                disclaimer=settings.disclaimer_text,
                inference_time_ms=duration_ms,
            )

    async def stream_triage(self, request: TriageRequest) -> AsyncGenerator[str, None]:
        """Server-Sent Events (SSE) formatinda token bazinda cikarim akisi uretir."""
        # Eszamanlilik kontrolu
        async with self.semaphore:
            # Once yapilandirilmis analizi hesapla
            response = await self.predict_triage(request)

            # Metin parcalara bolunerek token streaming simule edilir
            tokens = [
                f"[TRIAGE LEVEL]: {response.triage_level.value}\n\n",
                f"[ACILIYET SKORU]: {response.urgency_score} / 5\n\n",
                f"[KLINIK ON DEGERLENDIRME]: {response.clinical_assessment}\n\n",
                f"[ONERILEN AKSIYON]: {response.recommended_action}\n\n",
            ]

            if response.red_flags:
                tokens.append(f"[KIRMIZI BAYRAKLAR]: {', '.join(response.red_flags)}\n\n")

            tokens.append(f"[YASAL UYARI]: {response.disclaimer}\n")

            # Tokenlar tek tek SSE formatinda yield edilir
            for chunk in tokens:
                payload = json.dumps({"token": chunk, "is_final": False}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
                # Gercekci token akisi icin mikro gecikme
                await asyncio.sleep(0.03)

            # Akis sonu sinyali gonderilir
            final_payload = json.dumps({"token": "", "is_final": True}, ensure_ascii=False)
            yield f"data: {final_payload}\n\n"


# Tekil cikarim motoru nesnesi olusturuluyor
engine = TriageInferenceEngine()
