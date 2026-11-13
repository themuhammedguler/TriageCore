# ==============================================================================
# TriageCore - Faz 5: Yerel CI/CD Pipeline Doğrulama Scripti
# ==============================================================================
# Bu script, GitHub Actions uzerinde calisacak tum denetimleri (Linter, Pytest,
# Veri Pipeline Smoke Test) yerel ortamda tek bir komutla calistirir.
# Kodlari uzak repoya pushlamadan once %100 basaridan emin olmayi saglar.
# ==============================================================================

# Klasor ve dosya yollari islemleri icin Path modulu yuklenir
# Terminal surecleri calistirmak icin subprocess modulu yuklenir
import subprocess

# Sistem cikislari ve durum kodlari icin sys modulu yuklenir
import sys

# Sure olcumu icin time modulu yuklenir
import time
from pathlib import Path

# Renkli konsol cikti kutuphanesi Rich ice aktarilir
from rich.console import Console

# Tablo gosterimi icin Table bileseni ice aktarilir
from rich.table import Table

# Rich konsol nesnesi olusturuluyor
console = Console()


def run_step(step_name: str, command: list) -> bool:
    """Belirtilen terminal komutunu calistirir ve basari durumunu dondurur."""
    # Adim baslangicini ekrana yazdir
    console.print(f"\n[cyan][RUNNING][/cyan] [bold]{step_name}[/bold] -> `{' '.join(command)}`")
    # Baslangic zamani alinir
    start_time = time.time()

    # Alt surec calistirilir ve ciktilari yakalanir
    process = subprocess.run(command, capture_output=True, text=True)
    # Gecen sure hesaplanir
    elapsed = round(time.time() - start_time, 2)

    # Eger cikis kodu 0 ise basarilidir
    if process.returncode == 0:
        console.print(f"[green][SUCCESS][/green] {step_name} basariyla tamamlandi ({elapsed}s)")
        return True
    else:
        # Hata durumunda konsola kirmizi hata basilir
        console.print(f"[bold red][FAILED][/bold red] {step_name} basarisiz oldu! ({elapsed}s)")
        console.print(f"[red]{process.stderr or process.stdout}[/red]")
        return False


def main():
    """Tum CI pipeline adimlarini sirayla kosturan ana fonksiyon."""
    # Baslik yazdirilir
    console.print("\n[bold magenta]=======================================================[/bold magenta]")
    console.print("[bold magenta]       TriageCore Yerel CI/CD Pipeline Baslatildi      [/bold magenta]")
    console.print("[bold magenta]=======================================================[/bold magenta]")

    # Pipeline adimlari tanimlaniyor
    steps = [
        # 1. Adim: Ruff Kod Standartlari ve Linting Denetimi
        ("1. Ruff Linter Denetimi", [sys.executable, "-m", "ruff", "check", "."]),
        # 2. Adim: Pytest Tum Birim ve API Testleri
        ("2. Pytest Test Süiti (16 Test)", [sys.executable, "-m", "pytest", "--verbose"]),
        # 3. Adim: Faz 1 Veri Pipeline Smoke Testi
        ("3. Veri Pipeline Smoke Testi", [sys.executable, "-m", "src.data_pipeline.download", "--seed-only"]),
    ]

    # Sonuclari saklayacak liste
    results = []
    # Genel basari bayragi
    all_passed = True

    # Adimlar donguye alinir
    for name, cmd in steps:
        # Adim calistirilir
        success = run_step(name, cmd)
        # Sonuc listeye eklenir
        results.append((name, "GECTI" if success else "KALDI", "green" if success else "red"))
        # Eger adim basarisizsa genel bayrak False yapilir
        if not success:
            all_passed = False

    # Uretilen train.jsonl ve val.jsonl dosyalarinin varligi dogrulanir
    train_exists = Path("data/processed/train.jsonl").exists()
    val_exists = Path("data/processed/val.jsonl").exists()
    data_verified = train_exists and val_exists
    results.append(("4. JSONL Veri Dosyalari Kontrolu", "GECTI" if data_verified else "KALDI", "green" if data_verified else "red"))

    # Ozet Tablosu Olusturuluyor
    table = Table(title="\nTriageCore CI Pipeline Ozet Raporu")
    table.add_column("Pipeline Adimi", style="cyan")
    table.add_column("Durum", justify="center")

    for step_title, status_text, color in results:
        table.add_row(step_title, f"[{color}]{status_text}[/{color}]")

    console.print(table)

    # Genel degerlendirme
    if all_passed and data_verified:
        console.print("[bold green]\n[+] TEBRIKLER! Tum CI/CD testleri ve kalite kontrolleri basariyla gecti.[/bold green]")
        console.print("[green]Bu kodlar GitHub reposuna gonderilmeye (push) %100 hazirdir.\n[/green]")
        sys.exit(0)
    else:
        console.print("[bold red]\n[!] DIKKAT: Bazi pipeline adimlari basarisiz oldu. Lutfen hatalari duzeltiniz.\n[/bold red]")
        sys.exit(1)


# Script dogrudan calistirildiginda main() fonksiyonunu tetikler
if __name__ == "__main__":
    main()
