# TriageCore v2: End-to-End Open-Source Medical Triage & Clinical Simulation Walkthrough 🩺⚡

> **Kapsam:** TriageCore projesinin 5 ana fazı ve **Faz 6 (Cloud-to-Local Hibrit Mimari, Kurumsal Klinik Streamlit Arayüzü, Doktor Vaka Simülatörü ve Denetim Loglama Sistemi)** uçtan uca tamamlanmış ve doğrulanmıştır.

---

## 🏗️ Mühendislik Yol Haritası & Tamamlanan Modüller

| Faz | Açıklama | Anahtar Teknolojiler | Durum |
| :--- | :--- | :--- | :--- |
| **Faz 1** | **Veri Küratörlüğü & Hazırlığı** | Pydantic, ChatML / Llama-3 Şablonu, PII Sanitization | **TAMAMLANDI ✅** |
| **Faz 2** | **Model Fine-Tuning (QLoRA & PEFT)** | BitsAndBytes NF4, All-linear LoRA, Google Drive `/MyDrive/AI/TriageCore` Kaydı | **TAMAMLANDI ✅** |
| **Faz 3** | **Checkpoint Birleştirme & Export** | `merge_and_unload()`, Safetensors, llama.cpp GGUF (`Q4_K_M`) | **TAMAMLANDI ✅** |
| **Faz 4** | **Inference API & Streaming** | FastAPI, Server-Sent Events (`yield`), ESI 1-5 Şeması | **TAMAMLANDI ✅** |
| **Faz 5** | **CI/CD Otomasyonu & Test** | GitHub Actions, Astral `uv`, Pytest (20 Test), Yerel CI Runner | **TAMAMLANDI ✅** |
| **Faz 6** | **Colab Ngrok Sunucu, Klinik UI & Doktor Simülatörü** | Colab Keep-Alive, Ngrok/Cloudflared, Streamlit CDSS, OSCE Vaka Bankası, Audit Logger | **TAMAMLANDI ✅** |

---

## 🌟 Faz 6 (v2) İle Gelen Yeni Yetenekler

### 1. Google Drive `/MyDrive/AI/TriageCore` & Colab Keep-Alive ([TriageCore_QLoRA_Colab.ipynb](file:///c:/Users/themu/OneDrive/Masaüstü/Kodex/TriageCore/notebooks/TriageCore_QLoRA_Colab.ipynb))
* **İlk Hücrede Google Drive Bağlantısı:** Notebook'un 1. hücresi doğrudan Google Drive'ı bağlar ve `/content/drive/MyDrive/AI/TriageCore/{checkpoints,lora_adapter,logs}` dizinlerini hazırlar. Eğer daha önce eğitilmiş adaptör varsa tekrar eğitime gerek kalmadan doğrudan son hücredeki sunucuya geçilebilir.
* **Çift Katmanlı Keep-Alive:** 2. hücrede hem tarayıcı DOM ping (60 saniyede bir JS kontrolü) hem de arka plan Python daemon thread'i (2 dakikada bir Drive heartbeat kaydı) çalışarak Colab oturumunun çökmesini önler.
* **Son Hücrede Ngrok / Cloudflared Uzak GPU Sunucusu:** Modeli yerel bilgisayara indirmeye gerek kalmadan Colab GPU'su üzerinde FastAPI başlatır ve dış dünyaya HTTPS adresi üretir.

### 2. Doktor Vaka Simülatörü (OSCE) ve Çok Turlu Diyalog ([cases.py](file:///c:/Users/themu/OneDrive/Masaüstü/Kodex/TriageCore/src/api/cases.py))
* Kullanıcının **Muayene Eden Hekim**, sistemin ise **Hasta** olduğu 5 kapsamlı klinik vaka (*Akut Apandisit, Akut Pulmoner Emboli, Akut Bakteriyel Menenjit, Akut Kolesistit, Diyabetik Ketoasidoz*).
* Hekim hastaya ağrı karakterini, eşlik eden semptomları, özgeçmişi sorabilir; fizik muayene ve laboratuvar/görüntüleme tetkikleri isteyebilir.
* Vaka sonunda hekimin girdiği kesin tanı ve ESI aciliyet kararı 100 üzerinden puanlanır ve klinik ipuçları (*Clinical Pearls*) sunulur.

### 3. "Zero-AI Look" Kurumsal Klinik Konsol ([app.py](file:///c:/Users/themu/OneDrive/Masaüstü/Kodex/TriageCore/src/ui/app.py)) & Denetim Logları ([logger.py](file:///c:/Users/themu/OneDrive/Masaüstü/Kodex/TriageCore/src/api/logger.py))
* Robot ikonlarından arındırılmış, gerçek bir **Hastane Bilgi Yönetim Sistemi (HBYS / CDSS)** tasarım diline sahip 3 modüllü Streamlit arayüzü (`http://localhost:8501`).
* Sorulan her soru, verilen her cevap ve vaka değerlendirmeleri zaman damgalı olarak `logs/clinical_sessions.jsonl` dosyasına kaydedilir ve arayüzden JSON olarak indirilebilir.

---

## 🧪 Doğrulama Sonuçları
* **Birim ve Entegrasyon Testleri:** `20 passed in 1.36s` (Veri hattı, Merge/Export, FastAPI Triyaj, Çok Turlu Sohbet, Doktor Vaka Simülatörü ve Audit Log testleri).
* **Ruff Linter:** Sıfır hata (`All checks passed!`).
* **Canlı Arayüz:** `http://127.0.0.1:8501` üzerinde aktif ve çalışır durumda doğrulandı.
