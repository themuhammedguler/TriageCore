# ==============================================================================
# TriageCore - Sohbet Kayıt ve Klinik Denetim Loglama Sistemi (Audit Logger)
# ==============================================================================
# Bu modul, hem Klinik Triyaj modunda hem de Doktor Vaka Simulasyonu modunda
# sorulan her soruyu, verilen yaniti, ESI aciliyet seviyesini ve yanit suresini
# JSON Lines (.jsonl) formatinda kalici olarak diske kaydeder ve sorgular.
# ==============================================================================

# Zaman damgasi olusturmak icin datetime modulu yuklenir
# JSON serilestirme islemleri icin json modulu yuklenir
import json
from datetime import datetime, timezone

# Dosya ve klasor yollari icin Path modulu yuklenir
from pathlib import Path

# Tip ipuclari icin Any, Dict, List ve Optional yuklenir
from typing import Any, Dict, List, Optional


class ClinicalAuditLogger:
    """Klinik etkilesimleri JSONL formatinda diske kaydeden ve okuyan sinif."""

    def __init__(self, log_file: Optional[Path] = None):
        """Log dosyasinin yolunu belirler ve klasoru hazirlar."""
        # Varsayilan log dosyasi yolu: logs/clinical_sessions.jsonl
        self.log_file = log_file or Path("logs/clinical_sessions.jsonl")
        # Ust klasor yoksa otomatik olusturulur
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log_interaction(
        self,
        session_id: str,
        mode: str,
        user_message: str,
        assistant_reply: str,
        triage_level: Optional[str] = None,
        urgency_score: Optional[int] = None,
        case_id: Optional[str] = None,
        inference_time_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Tek bir soru-cevap etkilesimini zaman damgasiyla log dosyasina ekler."""
        # Ust klasorun varligini garanti altina al
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Kaydedilecek veri sozlugu olusturulur
        entry = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "session_id": session_id,
            "mode": mode,
            "case_id": case_id,
            "user_message": user_message,
            "assistant_reply": assistant_reply,
            "triage_level": triage_level,
            "urgency_score": urgency_score,
            "inference_time_ms": inference_time_ms,
            "metadata": metadata or {},
        }

        # Dosyanin sonuna (append modu) UTF-8 olarak tek satir JSON yazilir
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Kaydedilen kayit dondurulur
        return entry

    def get_logs(
        self,
        limit: int = 100,
        mode: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Kaydedilen loglari en yeniden en eskiye dogru filtreleyerek dondurur."""
        # Eger log dosyasi henuz olusmamissa bos liste dondur
        if not self.log_file.exists():
            return []

        records: List[Dict[str, Any]] = []
        # Dosya satir satir okunur
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                cleaned = line.strip()
                if not cleaned:
                    continue
                try:
                    item = json.loads(cleaned)
                    # Mod filtresi varsa kontrol et
                    if mode and item.get("mode") != mode:
                        continue
                    # Oturum ID filtresi varsa kontrol et
                    if session_id and item.get("session_id") != session_id:
                        continue
                    records.append(item)
                except json.JSONDecodeError:
                    continue

        # En son yapilan konusmalar en ustte gorunsun diye ters cevirip limit uygula
        records.reverse()
        return records[:limit]


# Tekil (Singleton) denetim loglayici nesnesi olusturuluyor
audit_logger = ClinicalAuditLogger()
