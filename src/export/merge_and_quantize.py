# ==============================================================================
# TriageCore - Faz 3: LoRA Adapter Birlestirme (Merge) ve GGUF Export Scripti
# ==============================================================================
# Bu script, Faz 2'de egitilen kucuk LoRA adapter agirliklarini (safetensors)
# 16-bit temel modelle (Base Model) kalici olarak birlestirir (W_merged = W_base + dW)
# ve ardindan yuksek basarimli inference icin GGUF formatina hazirlar.
# ==============================================================================

# Komut satiri parametrelerini ayristirmak icin argparse modulu yuklenir
import argparse

# Isletim sistemi duzeyi komut ve surec yonetimi icin subprocess modulu yuklenir
# Python calisma zamani ve sistem cikislari icin sys modulu yuklenir
import sys

# Dosya yollari ve klasor islemleri icin pathlib modulu yuklenir
from pathlib import Path


def parse_arguments():
    """Komut satirindan model birlestirme ve GGUF export parametrelerini ayristirir."""
    # Arguman ayristirici nesnesi olusturuluyor
    parser = argparse.ArgumentParser(description="TriageCore LoRA Merge & GGUF Quantizer")

    # Temel model kimligi veya yerel yolu
    parser.add_argument(
        "--base_model_id",  # Parametre adi
        type=str,  # Veri tipi
        default="meta-llama/Meta-Llama-3-8B-Instruct",  # Base model
        help="HuggingFace temel model kimligi veya yerel agirlik klasoru",  # Aciklama
    )

    # Faz 2'de uretilen LoRA adapter klasorunun yolu
    parser.add_argument(
        "--adapter_dir",  # Parametre adi
        type=str,  # Veri tipi
        default="./triagecore-lora-adapter",  # Adapter yolu
        help="Egitilmis LoRA adaptoru dizini (adapter_model.safetensors iceren klasor)",  # Aciklama
    )

    # Birlestirilen 16-bit tam modelin kaydedilecegi cikti dizini
    parser.add_argument(
        "--merged_output_dir",  # Parametre adi
        type=str,  # Veri tipi
        default="./models/triagecore-merged-hf",  # Merged HF model yolu
        help="Birlestirilmis Safetensors modelinin kaydedilecegi klasor",  # Aciklama
    )

    # Donusturulecek nihai GGUF dosyasinin hedef dosya yolu
    parser.add_argument(
        "--gguf_output_path",  # Parametre adi
        type=str,  # Veri tipi
        default="./models/triagecore-q4_k_m.gguf",  # GGUF hedef dosyasi
        help="Uretilecek nihai GGUF model dosyasinin tam yolu",  # Aciklama
    )

    # llama.cpp kuantizasyon tipi (Q4_K_M, Q8_0 veya F16)
    parser.add_argument(
        "--quant_type",  # Parametre adi
        type=str,  # Veri tipi
        default="Q4_K_M",  # Standart K-Quants medium tipi (en iyi perplexity / boyut dengesi)
        help="GGUF kuantizasyon algoritmasi: Q4_K_M, Q5_K_M, Q8_0",  # Aciklama
    )

    # Argumanlar ayristirilarak donduruluyor
    return parser.parse_args()


def merge_lora_into_base(
    base_model_id: str,
    adapter_dir: str,
    merged_output_dir: str,
) -> Path:
    """Temel model ile LoRA adaptoru agirliklarini birlestirip tek bir Safetensors modeli uretir."""
    # PyTorch ve HuggingFace PEFT kutuphaneleri dinamik ice aktarilir
    try:
        # PyTorch derin ogrenme kutuphanesi
        import torch

        # PEFT model sinifi: LoRA adaptoru ile base modeli birbirine baglar
        from peft import PeftModel

        # Model ve Tokenizer yukleme araclari
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        # Kutuphane eksikse kullanici uyarilir
        print(f"[!] Birlestirme icin gerekli kutuphane eksik: {exc}")
        print("[!] Gerekli kutuphaneler: torch transformers peft")
        sys.exit(1)

    # Cikti dizini Path nesnesine donusturulur
    output_path = Path(merged_output_dir)
    # Hedef klasor yoksa olusturulur
    output_path.mkdir(parents=True, exist_ok=True)

    # Durum bilgisi ekrana yazdirilir
    print(f"[*] 1. ADIM: Temel model 16-bit olarak belleğe yukleniyor: {base_model_id}")
    # Donanim bfloat16 destekliyorsa bfloat16, aksi halde float16 hassasiyeti secilir
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    # Base model 16-bit float olarak yuklenir (Burada 4-bit kuantizasyon KULLANILMAZ!)
    # Neden? Cunku LoRA agirliklari orijinal 16-bit uzayinda egitilmistir; kuantize edilmis matrise eklenemez.
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,  # Base model kimligi
        torch_dtype=torch_dtype,  # 16-bit kayan nokta hassasiyeti
        device_map="auto",  # Bellek durumuna gore GPU veya CPU RAM'ine yerlestir
        trust_remote_code=True,  # Guvenli kod calistirma izni
    )

    # Tokenizer ayni sekilde temel modelden yuklenir
    print("[*] 2. ADIM: Tokenizer yukleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    # LoRA adaptor katmanlari temel modelin uzerine monte edilir
    print(f"[*] 3. ADIM: LoRA adaptoru yukleniyor: {adapter_dir}")
    peft_model = PeftModel.from_pretrained(
        base_model,  # Donuk 16-bit base model
        adapter_dir,  # Faz 2'de diske yazilan adapter klasoru
        torch_dtype=torch_dtype,  # Ayni veri tipi
    )

    # LoRA matrisleri kalici olarak ana agirlik tensörlerine eklenir: W_new = W + (B x A) * (alpha / r)
    print("[*] 4. ADIM: LoRA agirliklari kalici olarak birlestiriliyor (merge_and_unload)...")
    merged_model = peft_model.merge_and_unload()

    # Birlestirilen tekil Safetensors modeli hedef klasore kaydedilir
    print(f"[*] 5. ADIM: Birlestirilmis tam model diske kaydediliyor: {output_path.resolve()}")
    merged_model.save_pretrained(
        output_path,  # Hedef cikti yolu
        safe_serialization=True,  # Guvenli Safetensors formati (.bin yerine)
        max_shard_size="4GB",  # Dosyalari 4GB'lik parcalara bolerek depola
    )

    # Tokenizer konfigurasyonu ve ozel kontrol tokenlari cikti dizinine kopyalanir
    tokenizer.save_pretrained(output_path)
    print(f"[+] Model basariyla birlestirildi ve kaydedildi: {output_path}")

    # Cikti yolunu dondur
    return output_path


def print_gguf_conversion_guide(merged_model_dir: Path, gguf_output_path: Path, quant_type: str):
    """llama.cpp kullanarak birlestirilen modeli GGUF formatina donusturme talimatlarini basar."""
    # Bilgilendirme basligi ekrana yazdirilir
    print("\n" + "=" * 80)
    print("  TriageCore GGUF Kuantizasyon Rehberi (llama.cpp)")
    print("=" * 80)
    print(f"[*] Kaynak Model: {merged_model_dir.resolve()}")
    print(f"[*] Hedef GGUF Dosyasi: {gguf_output_path.resolve()}")
    print(f"[*] Hedef Kuantizasyon Algoritmasi: {quant_type}")
    print("-" * 80)
    print("[*] Birlestirilen modeli GGUF'e cevirmek icin su 2 terminal komutunu calistirabilirsiniz:\n")

    # llama.cpp repodan derlenmis convert scripti komutu ornegi
    print("# 1. Asama: Safetensors modelini standart 16-bit GGUF formatina cevirin:")
    print(f"python llama.cpp/convert_hf_to_gguf.py {merged_model_dir} --outfile {merged_model_dir}/model-f16.gguf --outtype f16\n")

    # llama-quantize araci ile 4-bit K-Quants sikistirma komutu ornegi
    print("# 2. Asama: 16-bit GGUF dosyasini ultra-hizli Q4_K_M seviyesine kuantize edin:")
    print(f"llama-quantize {merged_model_dir}/model-f16.gguf {gguf_output_path} {quant_type}\n")
    print("=" * 80 + "\n")


def main():
    """CLI cagrisi ile birlestirme ve export akisini yoneten ana fonksiyon."""
    # Argumanlar okunur
    args = parse_arguments()

    # Birlestirme islemi tetiklenir (GPU/High-RAM ortaminda calistirilir)
    # Eger adapter klasoru yoksa (henuz egitim bitmediyse) kullaniciya rehber gosterilir
    adapter_path = Path(args.adapter_dir)
    if not adapter_path.exists():
        print(f"[!] Uyari: '{args.adapter_dir}' dizini yerel ortamda bulunamadi.")
        print("[!] Colab ortaminda egitilen LoRA adaptorunu zip'ten cikardiktan sonra bu adimi calistirabilirsiniz.")
        # Yine de GGUF rehberini ekrana basar
        print_gguf_conversion_guide(Path(args.merged_output_dir), Path(args.gguf_output_path), args.quant_type)
        return

    # Adapter dizini varsa birlestirme calistirilir
    merged_dir = merge_lora_into_base(
        base_model_id=args.base_model_id,
        adapter_dir=args.adapter_dir,
        merged_output_dir=args.merged_output_dir,
    )

    # GGUF donusturme rehberi ekrana basilir
    print_gguf_conversion_guide(merged_dir, Path(args.gguf_output_path), args.quant_type)


# Script dogrudan calistirildiginda main() fonksiyonu tetiklenir
if __name__ == "__main__":
    main()
