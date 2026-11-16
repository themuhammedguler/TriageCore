import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/TriageCore_QLoRA_Colab.ipynb")

new_cell2 = """# Isletim sistemi ve dosya yolu islemleri icin os ve pathlib modulleri yuklenir
import os
from pathlib import Path

# Google Colab ortaminda olup olmadigimiz tespit edilir
try:
    from google.colab import drive
    is_colab = True
except ImportError:
    is_colab = False

# ==============================================================================
# GOOGLE DRIVE VEYA YUKSEK HIZLI YEREL COLAB SSD DISKI (Hatasiz & Kesintisiz)
# ==============================================================================
# Eger Google Drive izin penceresi tarayicinizda 'credential propagation' hatasi
# veriyorsa veya ugrasmak istemiyorsaniz, sistem otomatik olarak Colab'in 100 GB'lik
# yuksek hizli SSD diskini (/content/AI/TriageCore) kullanir. Egitim ve Ngrok
# sunucusu hicbir kesinti olmadan calismaya devam eder!

USE_GOOGLE_DRIVE = True   # True: Drive'a baglanmayi dener; False: Dogrudan hizli SSD kullanir
drive_mounted = False

if is_colab and USE_GOOGLE_DRIVE:
    # 1. Zaten bagli mi kontrol et
    if Path("/content/drive/MyDrive").exists():
        drive_mounted = True
        print("[+] Google Drive zaten bagli! '/MyDrive/AI/TriageCore' kullaniliyor.")
    else:
        try:
            print("[*] Google Drive baglantisi deneniyor...")
            drive.mount("/content/drive", force_remount=False)
            if Path("/content/drive/MyDrive").exists():
                drive_mounted = True
                print("[+] Google Drive basariyla baglandi!")
        except Exception as err:
            print(f"[!] Google Drive yetkilendirme uyarisi: {err}")
            print("[*] HIC SORUN DEGIL! Sistem durdurulmuyor; Colab'in kendi yuksek hizli SSD diski (/content/AI/TriageCore) devreye alindi.")
            drive_mounted = False

# Calisma dizinini belirle
if drive_mounted:
    DRIVE_BASE_DIR = Path("/content/drive/MyDrive/AI/TriageCore")
elif is_colab:
    DRIVE_BASE_DIR = Path("/content/AI/TriageCore")
else:
    DRIVE_BASE_DIR = Path("./AI/TriageCore")

# Egitim checkpointlerinin kaydedilecegi alt klasor yolu
CHECKPOINT_DIR = DRIVE_BASE_DIR / "checkpoints"
# Nihai egitilmis LoRA adaptorunun kaydedilecegi alt klasor yolu
ADAPTER_DIR = DRIVE_BASE_DIR / "lora_adapter"
# Uzak sunucu sohbet ve denetim loglarinin kaydedilecegi alt klasor yolu
LOGS_DIR = DRIVE_BASE_DIR / "logs"

# Klasorlerin tamamini (eger yoksa) otomatik olarak olusturur
for folder in [CHECKPOINT_DIR, ADAPTER_DIR, LOGS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

print("=" * 75)
print(f"[+] PROJE CALISMA VE MODEL DIZINI HAZIR: {DRIVE_BASE_DIR.resolve()}")
if drive_mounted:
    print("[*] Modeller ve loglar kalici olarak Google Drive'a kaydedilecek.")
else:
    print("[*] Modeller yuksek hizli Colab SSD'sine kaydedilecek (Egitim daha hizli calisir).")
print("=" * 75)

# Daha onceden egitilmis bir model olup olmadigini kontrol eder
adapter_config_file = ADAPTER_DIR / "adapter_config.json"
if adapter_config_file.exists():
    print("[+] Bilgi: Bu dizinde egitilmis LoRA modeli bulundu. Dogrudan 7. Adima gecebilirsiniz.")
else:
    print("[*] Yeni egitim agirliklari bu dizine yazilacak.")
"""

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

nb["cells"][2]["source"] = [line + "\n" for line in new_cell2.splitlines()]

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("[+] Cell 2 basariyla guncellendi!")
