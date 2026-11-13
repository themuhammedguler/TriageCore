# ==============================================================================
# TriageCore - Faz 3 Export ve Merge Birim Testleri
# ==============================================================================

# Dosya yollari yonetimi icin pathlib modulu yuklenir
# Test ortaminda cikti yakalamak icin io modulu yuklenir
import io

# Standart cikti akisini yonlendirmek icin contextlib yuklenir
from contextlib import redirect_stdout
from pathlib import Path

# Test edilecek Faz 3 export fonksiyonlari ice aktarilir
from src.export.merge_and_quantize import (
    parse_arguments,
    print_gguf_conversion_guide,
)


def test_export_arguments_defaults(monkeypatch):
    """CLI varsayilan parametrelerinin dogru tanimlandigini dogrular."""
    # CLI argumanlari bos liste olarak simule edilir
    monkeypatch.setattr("sys.argv", ["merge_and_quantize.py"])
    # Argumanlar okunur
    args = parse_arguments()
    # Varsayilan model adinin Llama-3-8B oldugu dogrulanir
    assert "Meta-Llama-3-8B-Instruct" in args.base_model_id
    # Varsayilan kuantizasyonun Q4_K_M oldugu dogrulanir
    assert args.quant_type == "Q4_K_M"
    # Varsayilan adapter yolunun dogru oldugu dogrulanir
    assert "triagecore-lora-adapter" in args.adapter_dir


def test_print_gguf_conversion_guide():
    """GGUF rehber fonksiyonunun llama.cpp komutlarini dogru formatladigini test eder."""
    # Test klasor ve dosya yollari olusturulur
    merged_dir = Path("dummy/merged")
    gguf_path = Path("dummy/model.gguf")
    quant_type = "Q4_K_M"

    # Konsol ciktisini yakalamak icin buffer nesnesi acilir
    buffer = io.StringIO()
    # Stdout buffer nesnesine yonlendirilir
    with redirect_stdout(buffer):
        print_gguf_conversion_guide(merged_dir, gguf_path, quant_type)

    # Yakalanan metin dizeye donusturulur
    output = buffer.getvalue()
    # Rehber icinde convert_hf_to_gguf komutunun gectigi dogrulanir
    assert "convert_hf_to_gguf.py" in output
    # Rehber icinde llama-quantize komutunun gectigi dogrulanir
    assert "llama-quantize" in output
    # Secilen kuantizasyon tipinin yazildigi dogrulanir
    assert "Q4_K_M" in output
