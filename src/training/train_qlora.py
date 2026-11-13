# ==============================================================================
# TriageCore - Faz 2: QLoRA & PEFT Fine-Tuning Egitim Scripti
# ==============================================================================
# Bu script, acik agirlikli LLM modellerini (Llama 3 8B / Mistral 7B)
# Google Colab T4 GPU (15GB VRAM) gibi kisitli donanimlarda bellek asimi (OOM)
# yasamadan 4-bit kuantizasyon ve LoRA kullanarak egitmek icin tasarlanmistir.
# ==============================================================================

# Komut satiri argumanlarini (CLI arguments) ayristirmak icin argparse modulu yuklenir
import argparse

# Isletim sistemi duzeyinde dosya yolu ve ortam degiskenlerini yonetmek icin os modulu yuklenir
# Python calisma zamani ve sistem cikislari icin sys modulu yuklenir
import sys


def parse_arguments():
    """Komut satirindan egitim konfigurasyon parametrelerini ayristirir."""
    # Arguman ayristirici nesnesi olusturuluyor
    parser = argparse.ArgumentParser(description="TriageCore QLoRA Training Engine")

    # Model adi veya yerel yolu parametresi ekleniyor
    parser.add_argument(
        "--model_id",  # Parametre adi
        type=str,  # Veri tipi
        default="meta-llama/Meta-Llama-3-8B-Instruct",  # Varsayilan acik agirlikli model
        help="HuggingFace model kimligi veya yerel agirlik dizini",  # Aciklama
    )

    # Egitim verisinin dosya yolu
    parser.add_argument(
        "--train_file",  # Parametre adi
        type=str,  # Veri tipi
        default="data/processed/train.jsonl",  # Faz 1'de urettigimiz egitim verisi
        help="JSONL formatindaki egitim veri setinin yolu",  # Aciklama
    )

    # Dogrulama (validation) verisinin dosya yolu
    parser.add_argument(
        "--val_file",  # Parametre adi
        type=str,  # Veri tipi
        default="data/processed/val.jsonl",  # Faz 1'de urettigimiz dogrulama verisi
        help="JSONL formatindaki dogrulama veri setinin yolu",  # Aciklama
    )

    # Egitilen LoRA agirliklarinin kaydedilecegi cikti dizini
    parser.add_argument(
        "--output_dir",  # Parametre adi
        type=str,  # Veri tipi
        default="./triagecore-lora-adapter",  # Cikti klasoru
        help="LoRA adaptoru ve checkpointlerin yazilacagi dizin",  # Aciklama
    )

    # Egitim epoch sayisi (verinin modelin onunden kac kez gececegi)
    parser.add_argument(
        "--epochs",  # Parametre adi
        type=int,  # Veri tipi
        default=3,  # Medikal triyaj icin 3 epoch dengeli bir baslangictir
        help="Toplam egitim dongusu sayisi (epochs)",  # Aciklama
    )

    # Cihaz basina anlik batch boyutu (T4 GPU'da 2 onerilir, OOM riskini onler)
    parser.add_argument(
        "--batch_size",  # Parametre adi
        type=int,  # Veri tipi
        default=2,  # 15GB VRAM icin ideal anlik batch
        help="Cihaz basina anlik mikro-batch boyutu",  # Aciklama
    )

    # Gradyan biriktirme adim sayisi (Etkin batch size = batch_size * grad_accum_steps)
    parser.add_argument(
        "--grad_accum_steps",  # Parametre adi
        type=int,  # Veri tipi
        default=8,  # 2 * 8 = 16 etkin batch boyutu saglar, stabil ogrenme yaratir
        help="Bellek tuketmeden buyuk batch simule eden gradyan biriktirme adimi",  # Aciklama
    )

    # Ogrenme orani (Learning Rate)
    parser.add_argument(
        "--learning_rate",  # Parametre adi
        type=float,  # Veri tipi
        default=2e-4,  # QLoRA adaptorleri icin standart altin oran (0.0002)
        help="LoRA katmanlari icin baslangic ogrenme orani",  # Aciklama
    )

    # Maksimum token baglam uzunlugu
    parser.add_argument(
        "--max_seq_length",  # Parametre adi
        type=int,  # Veri tipi
        default=1024,  # T4 VRAM'ine rahat sigmasi ve triyaj diyaloglarini kapsamasi icin 1024
        help="Modelin tek seferde isleyecegi maksimum token uzunlugu",  # Aciklama
    )

    # Argumanlar ayristirilarak donduruluyor
    return parser.parse_args()


def configure_quantization(torch):
    """Modeli 4-bit NF4 formatina sokan BitsAndBytes konfigurasyonunu hazirlar."""
    # HuggingFace BitsAndBytesConfig modulu dinamik yukleniyor
    from transformers import BitsAndBytesConfig

    # BitsAndBytesConfig nesnesi olusturuluyor
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Modeli 16-bit yerine 4-bit agirliklarla belleğe yukle (16GB -> ~4GB VRAM)
        bnb_4bit_quant_type="nf4",  # NormalFloat4: Agirliklar normal dagilim gosteriyorsa en az bilgi kaybi veren format
        bnb_4bit_use_double_quant=True,  # Kuantizasyon sabitlerini de kuantize et (ekstra ~0.4 GB VRAM tasarrufu)
        bnb_4bit_compute_dtype=torch.float16,  # T4 GPU mimarisinde matris carpimlari float16 hassasiyetinde calistirilir
    )
    # Konfigurasyon donduruluyor
    return bnb_config


def configure_lora():
    """LoRA (Low-Rank Adaptation) hiperparametrelerini tanimlar."""
    # PEFT LoraConfig sinifi dinamik yukleniyor
    from peft import LoraConfig

    # LoraConfig nesnesi olusturuluyor
    lora_config = LoraConfig(
        r=16,  # Rank boyutu: Adaptor matrislerinin ara boyutu (16, klinik nuanslari yakalamak icin idealdir)
        lora_alpha=32,  # Olcekleme katsayisi: alpha / r = 32 / 16 = 2.0 carpaniyla LoRA etkisi ayarlanir
        target_modules=[  # LoRA matrislerinin eklenecegi transformator katmanlari
            "q_proj",  # Query projeksiyonu (Dikkat mekanizmasi)
            "k_proj",  # Key projeksiyonu (Dikkat mekanizmasi)
            "v_proj",  # Value projeksiyonu (Dikkat mekanizmasi)
            "o_proj",  # Output projeksiyonu (Dikkat mekanizmasi)
            "gate_proj",  # MLP katmani kapi projeksiyonu
            "up_proj",  # MLP katmani yukari projeksiyonu
            "down_proj",  # MLP katmani asagi projeksiyonu
        ],  # Butun lineer katmanlara LoRA eklemek (all-linear), en yuksek basarimi verir
        lora_dropout=0.05,  # Ezberlemeyi (overfitting) onlemek icin %5 oraninda dropout
        bias="none",  # Ekstra bias ogrenilmez, sadece agirlik matrisleri egitilir
        task_type="CAUSAL_LM",  # Gorev tipi: Metin uretimi (Causal Language Modeling)
    )
    # Konfigurasyon donduruluyor
    return lora_config


def train():
    """Ana egitim fonksiyonu: Veriyi yukler, modeli kuantize eder ve egitimi baslatir."""
    # Komut satiri argumanlari okunuyor
    args = parse_arguments()

    # GPU egitimi icin gerekli PyTorch ve Transformers kutuphaneleri dinamik ice aktarilir
    try:
        # PyTorch derin ogrenme kutuphanesi
        import torch

        # HuggingFace Datasets kutuphanesinden veri seti yukleme araci
        from datasets import load_dataset

        # HuggingFace PEFT kutuphanesinden donuk modele LoRA ekleme araclari
        from peft import get_peft_model, prepare_model_for_kbit_training

        # HuggingFace Transformers kutuphanesinden model yukleme bilesenleri
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

        # Supervised Fine-Tuning egitim motoru
        from trl import SFTTrainer
    except ImportError as e:
        # Gerekli GPU kutuphaneleri eksikse aciklayici hata mesaji verilir
        print(f"[!] Eksik Kutuphane: {e}")
        print("[!] Not: QLoRA egitimi icin GPU ortaminda (orn. Google Colab) calistirmalisiniz.")
        print("[!] Gerekli kutuphaneler: torch transformers peft bitsandbytes trl accelerate datasets")
        sys.exit(1)

    # Egitim baslangic bilgisi ekrana yazdiriliyor
    print("[*] TriageCore Egitim Sureci Baslatiliyor...")
    # Kullanilan hedef model konsola yazdirilir
    print(f"[*] Hedef Model: {args.model_id}")
    # CUDA cihaz kontrolu yapilir
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU BULUNAMADI"
    print(f"[*] GPU Cihaz: {device_name}")

    # CUDA erisimi yoksa kullanici uyarilip egitim durdurulur
    if not torch.cuda.is_available():
        print("[!] Hata: QLoRA egitimi icin bir NVIDIA GPU gereklidir!")
        sys.exit(1)

    # 1. ADIM: 4-bit Kuantizasyon ve LoRA Ayarlarini Hazirla
    bnb_config = configure_quantization(torch)
    lora_config = configure_lora()

    # 2. ADIM: Tokenizer Yukleme ve Yapilandirma
    print("[*] Tokenizer yukleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        use_fast=True,
        trust_remote_code=True,
    )

    # Llama-3 modellerinde pad_token varsayilan olarak bos gelebilir; eos_token'a esitlenir
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Causal LM modellerinde padding isleminin sag taraftan yapilmasi zorunludur
    tokenizer.padding_side = "right"

    # 3. ADIM: Kuantize Edilmis Base Modeli Belleğe Yukleme
    print("[*] 4-bit Kuantize Base Model VRAM'e yukleniyor...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Modeli k-bit egitim icin hazirla (LayerNorm katmanlarini float32'ye dondurerek stabilite saglar)
    model = prepare_model_for_kbit_training(model)

    # Aktivasyon bellek tuketimini %60 azaltan Gradient Checkpointing ozelligini aktif et
    model.gradient_checkpointing_enable()

    # Modeli LoRA adapter katmanlari ile sarmala
    model = get_peft_model(model, lora_config)

    # Egitilebilir parametre oranini ekrana yazdir (Genellikle toplam parametrenin %0.1 - %0.3'u kadardir)
    model.print_trainable_parameters()

    # 4. ADIM: Egitim ve Dogrulama Veri Setlerini Yukleme
    print(f"[*] Veri setleri yukleniyor: {args.train_file} & {args.val_file}")
    data_files = {
        "train": args.train_file,
        "validation": args.val_file,
    }

    # JSONL dosyalari HuggingFace Dataset nesnesine donusturuluyor
    dataset = load_dataset("json", data_files=data_files)

    # 5. ADIM: Egitim Hiperparametrelerini (TrainingArguments) Tanimlama
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        optim="paged_adamw_8bit",
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        fp16=True,
        max_grad_norm=0.3,
        report_to="none",
    )

    # 6. ADIM: SFTTrainer (Supervised Fine-Tuning Trainer) Ilklendirme
    print("[*] SFTTrainer hazirlaniyor...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    # 7. ADIM: Egitim Dongusunu Baslatma
    print("[*] Model egitimi baslatiliyor! Bu islem veri boyutuna gore zaman alacaktir...")
    trainer.train()

    # 8. ADIM: Egitilen LoRA Adaptorunu Kaydetme
    print(f"[*] Egitim tamamlandi. LoRA adaptoru kaydediliyor: {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[+] Tebrikler! Faz 2: QLoRA egitimi basariyla tamamlandi.")


# Script dogrudan terminalden calistirildiginda train() fonksiyonu tetiklenir
if __name__ == "__main__":
    train()
