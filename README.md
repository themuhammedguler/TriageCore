# TriageCore 🩺⚡

> **End-to-End Open-Source Medical Triage and Clinical Dialogue LLM Engine**  
> Built from scratch with zero proprietary APIs, strictly adhering to production-grade MLOps and top-tier AI engineering practices.

---

## 🎯 Proje Hedefi

TriageCore, acil servis triyaj standartlarını (ESI - Emergency Severity Index) baz alarak hasta semptomlarını değerlendiren, aciliyet skoru üreten ve klinik red-flag (kırmızı bayrak) uyarıları sağlayan açık ağırlıklı (open-weight) bir medikal dil modelidir.

## 🏗️ 5 Aşamalı Mimari

1. **Faz 1: Veri Seti Küratörlüğü ve Hazırlığı** (Tamamlandı ✅)
   - PII (Kişisel Bilgi) anonimizasyonu, gürültü ve web artefaktı temizleme.
   - Pydantic tabanlı ChatML / Llama-3 instruction şablon doğrulaması.
   - Otomatik HuggingFace veri çekme ve train/validation JSONL üretimi.
2. **Faz 2: Google Colab T4 QLoRA & PEFT Fine-Tuning** (Sıradaki Adım 🚀)
   - BitsAndBytes 4-bit NF4 kuantizasyonu ve bellek optimizasyonları (gradient checkpointing, paged AdamW).
3. **Faz 3: Ağırlık Birleştirme (Merge) ve GGUF Export**
   - LoRA adaptörlerini ana modelle birleştirme, llama.cpp ile Q4_K_M kuantizasyonu.
4. **Faz 4: Asenkron Yüksek Başarımlı Inference API**
   - FastAPI + Server-Sent Events (SSE) streaming ve yapılandırılmış ESI triyaj çıktısı.
5. **Faz 5: GitHub Actions CI/CD Pipeline & Otomasyon**
   - Pytest, Ruff linter ve uçtan uca otomatik test süiti.

---

## 🚀 Hızlı Başlangıç (Faz 1)

### 1. Bağımlılıkların Kurulumu (Modern `uv` ile)
```bash
# uv ile testleri ve linter'ı çalıştırma
uv run pytest
uv run ruff check .
```

### 2. Veri Seti İndirme ve Temizleme Pipeline'ı
```bash
# Hugging Face üzerinden medikal veri indirip temizleyerek JSONL üretir
python -m src.data_pipeline.download --max-samples 1500 --val-split 0.1

# Sadece altın standart klinik örneklerle test için:
python -m src.data_pipeline.download --seed-only
```

---

## 🧪 Test Süiti
```bash
pytest tests/
```
