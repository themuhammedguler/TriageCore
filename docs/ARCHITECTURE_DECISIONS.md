# TriageCore: Mühendislik Kararları ve Mimari Kayıtlar (ADR)
## Architecture Decision Records & Engineering Rationale

> Bu doküman, TriageCore projesinde alınan tüm kritik teknik kararları, "Neden bu teknoloji seçildi?", "Alternatifler neydi?", "Donanım, bellek ve performans kısıtları nelerdi?" sorularının cevaplarıyla birlikte yaşayan bir mühendislik günlüğü olarak belgeler.

---

### [ADR-001] Paket ve Sanal Ortam Yöneticisi Olarak `uv` Seçimi
* **Durum:** Kabul Edildi (Faz 1)
* **Bağlam:** Standart Python `pip` ve `venv` araçları paketleri ardışık indirir, C/wheel paketlerini tek iş parçacığında açar ve diskte her izole ortam için GB'larca mükerrer kopya oluşturur.
* **Karar:** Projede Astral ekibinin Rust ile geliştirdiği `uv` aracına geçildi.
* **Mühendislik Gerekçesi:**
  1. **10x-100x Hız:** Paralel indirme ve PubGrub dependency resolver motoru ile kurulum süreleri dakikalardan saniyelere indi.
  2. **Global Hardlink Cache:** Disk üzerinde aynı kütüphanelerin birden fazla kopyasının depolanmasını engelledi.
  3. **Deterministik Bağımlılık (`pyproject.toml`):** Production ortamında sürümlerin kilitlenmesini ve CI/CD pipeline'larında "bende çalışıyordu" riskini sıfıra indirdi.

---

### [ADR-002] Eğitim Verisinde ChatML / Llama-3 Mesaj Şablonu Seçimi
* **Durum:** Kabul Edildi (Faz 1)
* **Bağlam:** Geleneksel açık kaynak projelerinde Alpaca formatı (`instruction`, `input`, `output`) yaygın olarak kullanılır.
* **Karar:** `messages: [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]` standart ChatML formatı seçildi.
* **Mühendislik Gerekçesi:**
  1. **Özel Kontrol Tokenları (`<|start_header_id|>`, `<|eot_id|>`):** Llama 3 ve modern açık ağırlıklı modeller, rol ayrımını bu tokenlarla öğrenmiştir. Alpaca formatı bu kontrol tokenlarını baypas ettiği için modelde rol karışıklığı (role-confusion) ve istem sızıntısı (prompt injection) riski doğurur.
  2. **Klinik Guardrail Enjeksiyonu:** `system` rolü içerisine kalıcı acil servis (ESI - Emergency Severity Index) kuralları ve tıbbi feragatname şablonu yerleştirilebilmesi sağlandı.

---

### [ADR-003] Veri Temizleme, PII Maskeleme ve Kalite Eşikleri
* **Durum:** Kabul Edildi (Faz 1)
* **Bağlam:** Tıbbi veri setleri hekim adları, iletişim bilgileri, web kazıma kalıntıları (`-->`, `click here`) ve "doktora gidin" gibi tek cümlelik klinik değeri olmayan yanıtlar içerir.
* **Karar:** Pydantic şema doğrulaması ve çok aşamalı Regex sanitization pipeline'ı kuruldu.
* **Mühendislik Gerekçesi:**
  1. **Kişisel Sağlık Bilgisi (PII) Korunması:** Hekim ve hasta isimleri `Attending Clinician`, `[EMAIL]`, `[PHONE]` olarak anonimleştirildi.
  2. **Bellek Güvenliği (OOM Önleme):** Toplam karakteri 3500'ü aşan (>1024 token) diyaloglar elenerek Colab GPU bellek taşma riski ortadan kaldırıldı.
  3. **Bilgi Yoğunluğu Filtresi:** Soru < 25 karakter veya cevap < 50 karakter olan örnekler elenerek modelin "boş konuşması" (canned generic responses) engellendi.

---

### [ADR-004] 4-Bit NormalFloat (NF4) QLoRA ve Çift Kuantizasyon (Double Quant)
* **Durum:** Kabul Edildi (Faz 2)
* **Bağlam:** 8 milyar parametreli bir model 16-bit Float ağırlıklarla ~16.1 GB VRAM kaplar. T4 GPU (15.3 GB) veya L4/A100 GPU'larda tam ağırlık eğitimi (Full Fine-Tuning) yapmak için 40-80 GB VRAM gerekir.
* **Karar:** BitsAndBytes NF4 kuantizasyonu ve çift kuantizasyon (Double Quantization) seçildi.
* **Mühendislik Gerekçesi:**
  1. **NF4 (NormalFloat4):** Derin öğrenme ağırlıkları Gaussian (normal) dağılıma sahiptir. Klasik INT4 lineer aralıklar kullanırken, NF4 ağırlıkların bilgi yoğunluğunu korur ve kuantizasyon hatasını minimize eder.
  2. **VRAM Sıkıştırması:** 16 GB'lik model ~4.1 GB'ye indi.
  3. **Double Quantization:** Kuantizasyon katsayıları da 8-bite kuantize edilerek model başına ekstra ~400 MB VRAM tasarrufu sağlandı.

---

### [ADR-005] Bütün Lineer Katmanlara LoRA Uygulanması (All-Linear LoRA)
* **Durum:** Kabul Edildi (Faz 2)
* **Bağlam:** İlk LoRA makalesi (Hu et al.) adaptörleri yalnızca dikkat katmanlarına (`q_proj`, `v_proj`) eklemeyi öneriyordu.
* **Karar:** Tüm lineer katmanlar hedef alındı: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` ($r=16, \alpha=32$).
* **Mühendislik Gerekçesi:**
  1. **Klinik Alan Adaptasyonu:** Dikkat katmanları token ilişkilerini öğrenirken, MLP (Feed-Forward) katmanları olgusal ve terminolojik bilgiyi depolar. Medikal terimleri ve triyaj mantığını öğretebilmek için MLP katmanlarına adaptör eklemek modelin klinik doğruluğunu kanıtlanmış şekilde artırır.
  2. **Parametre Verimliliği:** Tüm katmanlar hedeflenmesine rağmen eğitilen parametre sayısı toplam modelin %0.3'ü kadardır.

---

### [ADR-006] Donanım Esnekliği (T4 vs L4/A100 ve Bfloat16 / FlashAttention-2)
* **Durum:** Kabul Edildi (Faz 2)
* **Bağlam:** Kullanıcının Colab ortamında T4'ün yanı sıra L4 (24GB) ve A100 (40GB/80GB) erişimi bulunmaktadır.
* **Karar:** Kodumuz GPU mimarisini otomatik algılayacak:
  - **T4'te:** `float16` ve `paged_adamw_8bit`.
  - **L4/A100'de:** Ampere/Ada Lovelace donanım desteğiyle **`bfloat16` (Brain Float 16)** ve **`FlashAttention-2`**.
* **Mühendislik Gerekçesi:**
  1. `bfloat16`, float32 ile aynı dinamik üs (exponent) aralığına sahiptir. `float16`'da yaşanan gradyan taşması (underflow/overflow) ve loss-scaling gereksinimini ortadan kaldırır.
  2. `FlashAttention-2`, dikkat matrislerini SRAM belleğinde tek seferde hesaplayarak bellek karmaşıklığını $O(N^2)$'den $O(N)$'e indirir ve eğitim süresini 2-3 kat hızlandırır.

---

### [ADR-007] Model Birleştirme (LoRA Merge & Unload)
* **Durum:** Planlandı (Faz 3)
* **Bağlam:** LoRA adaptörlerini ana modelden ayrı çalıştırmak, inference sırasında ek bellek referanslarına ve matris çarpım gecikmesine yol açar.
* **Karar:** LoRA matrisleri baz modele kalıcı olarak eklenip (`merge_and_unload`), bağımsız 16-bit Safetensors olarak dışa aktarılacaktır.
* **Mühendislik Gerekçesi:**
  1. $W_{merged} = W_{base} + \frac{\alpha}{r} (B \times A)$ formülüyle tekil ağırlık tensörü elde edilir.
  2. Inference motorunun (vLLM veya llama.cpp) ek adapter yönetim katmanına gerek kalmaz, maksimum çıkarma hızı (tokens/sec) elde edilir.

---

### [ADR-008] GGUF Formatı ve K-Quants (Q4_K_M) Kuantizasyonu
* **Durum:** Planlandı (Faz 3)
* **Bağlam:** Safetensors modelleri CPU veya tüketici GPU'larında çok fazla VRAM tüketir ve yavaş kalır.
* **Karar:** Model GGUF (GPT-Generated Unified Format) standardına ve `Q4_K_M` (k-quants) seviyesine dönüştürülecektir.
* **Mühendislik Gerekçesi:**
  1. **Q4_K_M (Medium K-Quants):** Kritik katmanları (attention q/v) 5-6 bit hassasiyette tutarken, daha az hassas MLP ağırlıklarını 4-bit yapar. Perplexity kaybı neredeyse sıfırdır.
  2. **Evrensel Dağıtım:** GGUF formatındaki model, Mac Metal, Linux CUDA veya sadece CPU içeren standart sunucularda saniyede onlarca token üretebilir.
