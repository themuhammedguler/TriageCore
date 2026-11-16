import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/TriageCore_QLoRA_Colab.ipynb")

new_cell2 = """# Isletim sistemi ve dosya yolu islemleri icin os ve pathlib modulleri yuklenir
import os
from pathlib import Path

# Google Colab ortaminda olup olmadigimizi kontrol ederek Drive baglantisini baslatir
try:
    from google.colab import drive

    # Eger Drive zaten bagliysa (ornegin sol paneldeki klasor ikonundan baglandiysa) tekrar sorma
    if not Path("/content/drive/MyDrive").exists():
        try:
            drive.mount("/content/drive", force_remount=False)
        except Exception as e:
            print(f"[!] Standart drive.mount uyarisi: {e}")
            print("[*] 'auth.authenticate_user()' ile alternatif Google kimlik dogrulamasi deneniyor...")
            try:
                from google.colab import auth
                auth.authenticate_user()
                drive.mount("/content/drive", force_remount=True)
            except Exception as auth_err:
                print(f"[!] Otomatik yetkilendirme basarisiz: {auth_err}")
                print("[*] Lutfen Colab sol panelindeki Klasor (📁) simgesine tiklayip uzerinde Drive simgesi olan butona tiklayarak Drive'i manuel baglayiniz.")
                raise auth_err

    # Drive uzerindeki ana 'AI/TriageCore' proje dizini tanimlanir
    DRIVE_BASE_DIR = Path("/content/drive/MyDrive/AI/TriageCore")
    print(f"[+] Google Drive basariyla baglandi: {DRIVE_BASE_DIR}")
except ImportError:
    # Eger Colab disinda yerel bir Jupyter ortaminda calisiliyorsa yerel klasor kullanilir
    DRIVE_BASE_DIR = Path("./AI/TriageCore")
    print(f"[!] Colab disi ortam algilandi. Yerel dizin kullaniliyor: {DRIVE_BASE_DIR.resolve()}")

# Egitim checkpointlerinin kaydedilecegi alt klasor yolu
CHECKPOINT_DIR = DRIVE_BASE_DIR / "checkpoints"
# Nihai egitilmis LoRA adaptorunun kaydedilecegi alt klasor yolu
ADAPTER_DIR = DRIVE_BASE_DIR / "lora_adapter"
# Uzak sunucu sohbet ve denetim loglarinin kaydedilecegi alt klasor yolu
LOGS_DIR = DRIVE_BASE_DIR / "logs"

# Klasorlerin tamamini (eger yoksa) otomatik olarak olusturur
for folder in [CHECKPOINT_DIR, ADAPTER_DIR, LOGS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Drive'da daha onceden egitilmis bir model olup olmadigini kontrol eder
adapter_config_file = ADAPTER_DIR / "adapter_config.json"
if adapter_config_file.exists():
    print("======================================================================")
    print("[+] MUJDE: Google Drive 'AI/TriageCore/lora_adapter' icinde egitilmis model bulundu!")
    print("[*] Isterseniz egitim adimlarini atlayip dogrudan SON HUCREDEKI Ngrok Sunucusunu calistirabilirsiniz.")
    print("======================================================================")
else:
    print("[*] Drive klasor yapisi hazir. Yeni egitim agirliklari bu dizine kaydedilecek.")
"""

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

nb["cells"][2]["source"] = [l + "\n" for l in new_cell2.splitlines()]

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("[+] Cell 2 basariyla guncellendi!")
