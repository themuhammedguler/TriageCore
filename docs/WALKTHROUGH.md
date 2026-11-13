# TriageCore: End-to-End Open-Source Medical Triage LLM Walkthrough 🩺⚡

> **Kapsam:** TriageCore projesinin 5 ana fazı, hiçbir hazır ticari API kullanılmadan, açık ağırlıklı modeller, QLoRA 4-bit optimizasyonları, GGUF mimarisi, FastAPI streaming SSE ve GitHub Actions CI/CD pipeline'ı ile uçtan uca tamamlanmıştır.

---

## 🏗️ 5 Fazlı Mühendislik Yol Haritası & Sonuçlar

| Faz | Açıklama | Anahtar Teknolojiler | Durum |
| :--- | :--- | :--- | :--- |
| **Faz 1** | **Veri Küratörlüğü & Hazırlığı** | Pydantic, ChatML / Llama-3 Şablonu, PII Sanitization | **TAMAMLANDI ✅** |
| **Faz 2** | **Model Fine-Tuning (QLoRA & PEFT)** | BitsAndBytes NF4, All-linear LoRA ($r=16, \alpha=32$), A100/L4/T4 desteği | **TAMAMLANDI ✅** |
| **Faz 3** | **Checkpoint Birleştirme & Export** | `merge_and_unload()`, Safetensors, llama.cpp GGUF (`Q4_K_M`) | **TAMAMLANDI ✅** |
| **Faz 4** | **Inference API & Streaming** | FastAPI, Server-Sent Events (`yield`), ESI 1-5 Şeması | **TAMAMLANDI ✅** |
| **Faz 5** | **CI/CD Otomasyonu & Test** | GitHub Actions, Astral `uv`, Pytest (16 Test), Yerel CI Runner | **TAMAMLANDI ✅** |

---

## 📂 Oluşturulan Proje Ağacı

```text
TriageCore/
├── .github/
│   └── workflows/
│       └── ci.yml                      # GitHub Actions otomatik CI pipeline'ı
├── data/
│   ├── sample_triage.jsonl             # Altın standart klinik triyaj örnekleri
│   └── processed/                      # (Gitignored) train.jsonl ve val.jsonl
├── docs/
│   ├── ARCHITECTURE_DECISIONS.md       # ADR-001'den ADR-015'e kadar yaşayan mühendislik günlüğü
│   └── WALKTHROUGH.md                  # 5 fazlı projenin detaylı genel bakış ve doğrulama raporu
├── notebooks/
│   └── TriageCore_QLoRA_Colab.ipynb    # Google Colab GPU (A100/L4/T4) eğitim notebook'u
├── scripts/
│   └── run_ci_local.py                 # Tek tıkla yerel CI/CD doğrulama scripti
├── src/
│   ├── data_pipeline/
│   │   ├── schemas.py                  # ChatML ve diyalog şemaları (Pydantic)
│   │   ├── clean.py                    # Regex PII temizliği ve medikal kalite filtreleri
│   │   ├── formatters.py               # Llama-3 sistem istemi ve JSONL yazıcı
│   │   └── download.py                 # HuggingFace indirme ve bölme motoru
│   ├── training/
│   │   └── train_qlora.py              # Satır satır yorumlanmış QLoRA eğitim scripti
│   ├── export/
│   │   └── merge_and_quantize.py       # LoRA merge ve GGUF kuantizasyon motoru
│   └── api/
│       ├── config.py                   # FastAPI global ayarları
│       ├── schemas.py                  # ESI 1-5 ve triyaj istek/yanıt modelleri
│       ├── engine.py                   # SSE streaming ve Türkçe NLP kök motoru
│       └── main.py                     # FastAPI asenkron sunucu uygulaması
├── tests/
│   ├── test_data_pipeline.py           # Veri temizleme ve şema testleri (8 test)
│   ├── test_export.py                  # Model merge ve CLI argüman testleri (2 test)
│   └── test_api.py                     # FastAPI, ESI ve SSE streaming testleri (6 test)
├── pyproject.toml                      # Modern uv ve bağımlılık manifestosu
├── requirements.txt                    # Pip uyumluluk listesi
├── .gitignore                          # Model ağırlıklarını ve büyük verileri koruyan kurallar
└── README.md                           # Kapsamlı proje dokümantasyonu
```

---

## 🧪 Test Doğrulama ve CI/CD Sonuçları

Projedeki tüm birim ve entegrasyon testleri `uv run pytest` ve `python scripts/run_ci_local.py` ile test edilmiştir:

```text
=======================================================
       TriageCore Yerel CI/CD Pipeline Başlatıldı      
=======================================================

[RUNNING] 1. Ruff Linter Denetimi -> `ruff check .`
[SUCCESS] 1. Ruff Linter Denetimi başarıyla tamamlandı (0.14s)

[RUNNING] 2. Pytest Test Süiti (16 Test) -> `pytest --verbose`
[SUCCESS] 2. Pytest Test Süiti (16 Test) başarıyla tamamlandı (1.63s)

[RUNNING] 3. Veri Pipeline Smoke Testi -> `src.data_pipeline.download --seed-only`
[SUCCESS] 3. Veri Pipeline Smoke Testi başarıyla tamamlandı (0.38s)

     TriageCore CI Pipeline Özet Raporu     
┌──────────────────────────────────┬───────┐
│ Pipeline Adımı                   │ Durum │
├──────────────────────────────────┼───────┤
│ 1. Ruff Linter Denetimi          │ GEÇTİ │
│ 2. Pytest Test Süiti (16 Test)   │ GEÇTİ │
│ 3. Veri Pipeline Smoke Testi     │ GEÇTİ │
│ 4. JSONL Veri Dosyaları Kontrolü │ GEÇTİ │
└──────────────────────────────────┴───────┘

[+] TEBRİKLER! Tüm CI/CD testleri ve kalite kontrolleri başarıyla geçti.
Bu kodlar GitHub reposuna gönderilmeye (push) %100 hazırdır.
```

---

## 💻 Git Sürüm Geçmişi (Yalnızca Yerel Depo)

```text
bcb944d feat(ci): complete Phase 5 GitHub Actions workflow and local CI runner
8fb29f1 feat(api): complete Phase 4 FastAPI service with streaming SSE and structured triage
b5f0eb0 feat(export): complete Phase 3 merge and GGUF quantization pipeline with ADR documentation
65a4277 feat(training): add Phase 2 QLoRA training script and Colab notebook with line-by-line comments
8348199 feat(data_pipeline): complete Phase 1 medical triage curation and validation pipeline
```
