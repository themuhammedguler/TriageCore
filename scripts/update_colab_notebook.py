# -*- coding: utf-8 -*-
"""
TriageCore Colab Notebook Guncelleyici
ChatDoctor sablon spamini temizler, altin standart klinik triyaj veri setini ve
repetition_penalty parametrelerini notebooks/TriageCore_QLoRA_Colab.ipynb icine enjekte eder.
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/TriageCore_QLoRA_Colab.ipynb")

def build_dataset_code() -> str:
    """Altin standart klinik triyaj veri setini ve egitim kodunu Python kodu olarak olusturur."""
    return '''# Veri yukleyici ve egitici siniflarini ice aktar
import inspect
import json
import random
import shutil
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments
from trl import SFTTrainer

# trl >= 0.12 ile gelen SFTConfig veya transformers TrainingArguments dinamik secilir
try:
    from trl import SFTConfig
    UseSFTConfig = True
except ImportError:
    UseSFTConfig = False

# ==============================================================================
# 1. LORA ADAPTOR VE PEFT YAPILANDIRMASI
# ==============================================================================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Eger model henuz PeftModel'e sarilmadiysa sar
if not hasattr(model, "peft_config"):
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, lora_config)
    print("[+] Model LoRA katmanlari ile basariyla sarildi.")
else:
    print("[*] Model zaten LoRA adaptoru ile hazirlandi, mevcut katmanlar korunuyor.")

model.print_trainable_parameters()

# ==============================================================================
# 2. ESKI / BOZUK VERILERIN TEMIZLENMESI (ChatDoctor Sablon Spamini Arindirma)
# ==============================================================================
DRIVE_DATA_DIR = DRIVE_BASE_DIR / "data"
DRIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)

train_path = Path("train.jsonl")
val_path = Path("val.jsonl")

# Eger Google Drive'da eski ChatDoctor sablon spamini iceren train.jsonl varsa temizle
drive_train_file = DRIVE_DATA_DIR / "train.jsonl"
should_clean_old_data = False
if drive_train_file.exists():
    try:
        sample_text = drive_train_file.read_text(encoding="utf-8")[:3000]
        if "Chat Doctor" in sample_text or "Complete blood count" in sample_text or "Serum HDL" in sample_text:
            print("[!] DIKKAT: Google Drive icinde bozuk/tekrarlayan ChatDoctor spam verisi tespit edildi. Otomatik temizleniyor...")
            should_clean_old_data = True
    except Exception:
        pass

if should_clean_old_data:
    if drive_train_file.exists():
        drive_train_file.unlink()
    if (DRIVE_DATA_DIR / "val.jsonl").exists():
        (DRIVE_DATA_DIR / "val.jsonl").unlink()
    # Eski hatali egitim checkpoint ve adaptorlerini de temizle
    if CHECKPOINT_DIR.exists():
        shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if ADAPTER_DIR.exists():
        shutil.rmtree(ADAPTER_DIR, ignore_errors=True)
        ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    print("[+] Eski bozuk veri ve checkpointler basariyla temizlendi. Sifir hatasiz egitim hazirlaniyor.")

# ==============================================================================
# 3. ALTIN STANDART KAPSAMLI KLINIK TRIYAJ VERI SETI (ESI 1 - ESI 5)
# ==============================================================================
SYS_CDSS = (
    "Sen TriageCore Acil Servis ve Klinik Karar Destek Asistanısın (CDSS). "
    "Görevin: Hastanın semptomlarını dikkatle değerlendirmek, kırmızı bayrakları sorgulamak ve kanıta dayalı ESI triyaj derecesi atamaktır. "
    "Yanıtlarını DAİMA ve KESİNLİKLE TÜRKÇE olarak üret. Asla genel test listeleri tekrarlama veya İngilizce şablon cümleleri yazma. "
    "Yanıtını net formatta sun: **Klinik Değerlendirme:**, **Aciliyet:**, **Önerilen Aksiyon Planı:**, **🚨 Kırmızı Bayraklar:**"
)

CURATED_TRIAGE_CASES = [
    # --- ESI-1: RESUSITASYON (HAYATI TEHLIKE / ANINDA MUDAHALE) ---
    {
        "user": "Babam aniden yere yığıldı, sesleniyoruz cevap vermiyor, nefes almıyor gibi!",
        "assistant": """**Klinik Değerlendirme:** Yanıtsızlık ve solunum yokluğu / agonal solunum Kardiyak Arrest (Kalp Durması) tablosunu işaret eder. Saniyeler dahi kritiktir.

**Aciliyet:** ESI Seviye 1: Resüsitasyon (Derhal Müdahale / Hayati Tehlike)

**Önerilen Aksiyon Planı:** 
1. Derhal 112 Acil Servis'i arayın ve telefonu hoparlöre alın.
2. Yakında Otomatik Eksternal Defibrilatör (OED) varsa getirtin.
3. Sert bir zemin üzerinde göğüs kemiğinin ortasına dakikada 100-120 hızında kesintisiz Temel Yaşam Desteği (kalp masajı) başlatın.

**🚨 Kırmızı Bayraklar:** Hasta arrest durumundadır. Sağlık ekipleri gelene kadar kalp masajına asla ara vermeyiniz."""
    },
    {
        "user": "Arkadaşımı arı soktu, 5 dakika içinde dudakları ve dili dev gibi şişti, nefes alamıyor, hırıltı çıkarıp yere yığıldı!",
        "assistant": """**Klinik Değerlendirme:** Arı sokması sonrası dakikalar içinde gelişen üst hava yolu anjiyoödemi, stridor ve hipotansif kollaps Şiddetli Anafilaktik Şok tablosudur. Asfiksi ve kardiyovasküler kollaps riski mevcuttur.

**Aciliyet:** ESI Seviye 1: Resüsitasyon (Anında Müdahale / Kırmızı Alan)

**Önerilen Aksiyon Planı:** 
1. Derhal 112 Acil Servis aranmalıdır.
2. Hastanın yanında Adrenalin Otoenjektörü (EpiPen) varsa tereddüt etmeden uyluğun dış yan kısmından intramüsküler uygulayınız.
3. Hastayı sırtüstü yatırıp bacaklarını 30 cm yükseltiniz (Trendelenburg/şok pozisyonu), solunum güçlüğü varsa yarı oturur pozisyona getiriniz.

**🚨 Kırmızı Bayraklar:** Tam hava yolu obstrüksiyonu gelişmeden acil entübasyon veya krikotiroidotomi gerekebilir."""
    },
    {
        "user": "Kardeşim 8 dakikadır aralıksız sara nöbeti geçiriyor, kaskatı kesildi ve kasılıyor, ağzından köpükler geliyor, morardı!",
        "assistant": """**Klinik Değerlendirme:** 5 dakikadan uzun süren veya bilinç açılmadan tekrarlayan epileptik nöbet Status Epileptikus olarak tanımlanır ve hipoksiye bağlı kalıcı serebral hasar riski taşır.

**Aciliyet:** ESI Seviye 1: Resüsitasyon (Hayati Tehlike / Acil Kırmızı Alan)

**Önerilen Aksiyon Planı:** 
1. Derhal 112 Acil Servis aranmalıdır.
2. Hastanın ağzına asla kaşık, mendil veya parmak sokmayınız; dişlerini açmaya çalışmayınız.
3. Çevredeki sert eşyaları uzaklaştırarak başını yumuşak bir şeyle koruyunuz ve aspirasyonu önlemek için nöbet bitiminde koma/iyileşme (recovery) pozisyonuna alınız.

**🚨 Kırmızı Bayraklar:** Nöbetin 5 dakikayı aşması ve siyanoz (morarma) acil IV/rektal antikonvülsan tedavi gerektirir."""
    },
    {
        "user": "Bacağından cam kesisi oldu, fışkırır tarzda açık kırmızı kan boşanıyor, bez bastırıyoruz ama kan durmuyor!",
        "assistant": """**Klinik Değerlendirme:** Nabızla uyumlu fışkıran açık kırmızı kanama Arteriyel (Atardamar) Kanamasıdır. Dakikalar içinde masif hipovolemik şok ve eksangüinasyona (kan kaybından ölüm) yol açabilir.

**Aciliyet:** ESI Seviye 1: Resüsitasyon (Hayati Tehlike / Anında Müdahale)

**Önerilen Aksiyon Planı:** 
1. Derhal 112'yi arayınız.
2. Temiz bir bez veya tampon ile yaranın tam üzerine tüm vücut ağırlığınızla çok sert bası uygulayınız.
3. Kanama durmuyorsa yaranın 5-7 cm yukarısına (eklem üstüne gelmeyecek şekilde) acil Turnike uygulayınız ve sıkma saatini not ediniz.

**🚨 Kırmızı Bayraklar:** Bilinç bulanıklığı, soğuk soluk cilt ve zayıflayan nabız dekompanse hemorajik şok bulgusudur."""
    },
    {
        "user": "Trafik kazası sonrası araçtan fırlayan yolcu, derin koma halinde, hiçbir uyarana yanıt vermiyor, solunumu hırıltılı ve çok seyrek.",
        "assistant": """**Klinik Değerlendirme:** Ciddi yüksek enerjili künt travma zemininde GCS (Glasgow Koma Skalası) < 8 olan derin koma ve solunum depresyonu tablosudur. Ağır Travmatik Beyin Hasarı ve hava yolu kaybı mevcuttur.

**Aciliyet:** ESI Seviye 1: Resüsitasyon (Kritik Travma / Acil Kırmızı Alan)

**Önerilen Aksiyon Planı:** 
1. Derhal 112 Acil Servis ve itfaiye ekipleri yönlendirilmelidir.
2. Servikal kolar ve omurga tahtası uygulanana kadar boyun aksı kesinlikle bükülmemeli, nötral pozisyonda tutulmalıdır.
3. Çene itme (jaw-thrust) manevrası ile hava yolu açıklığı sağlanmalı, acil endotrakeal entübasyon hazırlığı yapılmalıdır.

**🚨 Kırmızı Bayraklar:** Anizokori (pupil asimetrisi), hipotansiyon ve bradikardi (Cushing triadı) intrakraniyal herniasyon habercisidir."""
    },

    # --- ESI-2: ACIL DURUMLAR (10-15 DAKIKA ICINDE MUDAHALE / YUKSEK RISK) ---
    {
        "user": "58 yaşında erkek hasta. 40 dakikadır göğsümün ortasında mengene gibi sıkıştıran, sol koluma ve çeneme vuran şiddetli baskı var. Soğuk ter boşaldı, midem bulanıyor.",
        "assistant": """**Klinik Değerlendirme:** Retrosternal baskı tarzı göğüs ağrısı, sol kola ve alt çeneye radyasyon, terleme ve bulantı Akut Koroner Sendrom (Akut Miyokard Enfarktüsü / Kalp Krizi) için klasik kardinal klinik bulgulardır.

**Aciliyet:** ESI Seviye 2: Acil (10 Dakika İçinde EKG ve Troponin / Kırmızı Alan)

**Önerilen Aksiyon Planı:** 
1. Derhal 112 Acil Servis aranmalı, hasta kesinlikle kendi aracıyla gitmeye çalışmamalıdır.
2. Hasta tam istirahate alınmalı, efor sarf etmesi önlenmelidir.
3. Bilinen kanama veya alerji kontrendikasyonu yoksa 300 mg çiğneme tableti Asetilsalisilik Asit (Aspirin) çiğnetilmelidir.

**🚨 Kırmızı Bayraklar:** Senkop (bayılma), hipotansiyon, aritmik çarpıntı veya ani nefes darlığı gelişmesi durumunda acil kardiyak arrest protokolüne geçilmelidir."""
    },
    {
        "user": "Dedemin sağ kolu ve bacağı aniden tutmaz oldu, yüzünün sağ tarafı aşağı sarktı, konuşurken dili dolanıyor hiçbir şey anlaşılmıyor. Yaklaşık 25 dakika önce başladı.",
        "assistant": """**Klinik Değerlendirme:** FAST kriterleri (Face-Arm-Speech-Time) pozitif olan ani tek taraflı fasyal asimetri, hemiparezi ve dizartri/afazi Akut İskemik İnme (Serebrovasküler Olay - SVO) tablosunu gösterir. İlk 4.5 saat (trombolitik pencere) kritiktir.

**Aciliyet:** ESI Seviye 2: Acil (Akut İnme Protokolü / Acil BT İncelemesi)

**Önerilen Aksiyon Planı:** 
1. Derhal 112 aranmalı ve ekiplere inme şüphesi olduğu, başlangıç saatinin 25 dakika önce olduğu bildirilmelidir.
2. Hastaya ağızdan kesinlikle su, ilaç veya yiyecek verilmemelidir (yutma refleksi bozulduğu için aspirasyon riski vardır).
3. Tansiyonu çok yüksek olsa bile evde tansiyon düşürücü ilaç verilmemelidir; serebral perfüzyon korunmalıdır.

**🚨 Kırmızı Bayraklar:** Bilinç kaybı, solunum yüzeyelleşmesi veya kusma durumunda baş yan çevrilerek hava yolu açık tutulmalıdır."""
    },
    {
        "user": "Dün akşamdan beri göbek çevrem ağrıyordu, sabaha karşı karnımın sağ alt tarafına indi. Yürürken, öksürürken o nokta bıçak gibi batıyor. Ateşim 38.3 çıktı, karnıma dokununca sıçrıyorum.",
        "assistant": """**Klinik Değerlendirme:** Göbek çevresinden başlayıp sağ alt kadrana (McBurney noktası) lokalize olan ağrı, subfebril ateş ve periton iritasyonu (defans ve rebound hassasiyeti) Akut Apandisit lehinedir. Perforasyon (patlama) ve peritonit riski mevcuttur.

**Aciliyet:** ESI Seviye 2: Acil (Genel Cerrahi Konsültasyonu / Acil Servis)

**Önerilen Aksiyon Planı:** 
1. Vakit kaybetmeden en yakın hastane Acil Servisi'ne başvurulmalıdır.
2. Olası acil cerrahi operasyon ihtimaline karşı ağızdan tüm yiyecek ve içecek alımı (oral alım) derhal kesilmelidir.
3. Muayene bulgularını perdeleyebileceği için doktora görünmeden ağrı kesici veya antibiyotik alınmamalıdır.

**🚨 Kırmızı Bayraklar:** Karında tahta gibi sertlik, yaygınlaşan şiddetli ağrı, sürekli kusma ve 39°C üzerine çıkan ateş durumunda acil müdahale gerekir."""
    },
    {
        "user": "38 yaşında kadınım. Dün gece 12 saatlik otobüs yolculuğundan döndüm. Bugün aniden bıçak saplanır gibi göğüs ağrısı ve şiddetli nefes darlığı başladı. Kalbim 125 atıyor, sağ baldırım şiş ve ağrılı.",
        "assistant": """**Klinik Değerlendirme:** Uzun süreli immobilizasyon ve tek taraflı derin ven trombozu (baldır şişliği) zemininde gelişen ani başlangıçlı plöritik göğüs ağrısı, taşikardi ve dispne Akut Pulmoner Tromboemboli (Akciğer Pıhtısı) için yüksek klinik olasılıktır.

**Aciliyet:** ESI Seviye 2: Acil (Acil Servis / BT Pulmoner Anjiyografi)

**Önerilen Aksiyon Planı:** 
1. 112 Acil Servis ile derhal en yakın tam teşekküllü acil servise nakil sağlanmalıdır.
2. Hasta kesinlikle yürütülmemeli ve bacak masajı yapılmamalıdır (ek pıhtı kopmasını tetikleyebilir).
3. Acil serviste D-Dimer, EKG, arteriyel kan gazı ve kontrastlı Toraks BT Anjiyografi yapılmalıdır.

**🚨 Kırmızı Bayraklar:** Tansiyonun 90 mmHg altına düşmesi, bayılma (senkop) veya dudaklarda morarma masif pulmoner emboli ve kardiyovasküler şok işaretidir."""
    },
    {
        "user": "Hayatımda hiç böyle bir şey yaşamadım! Kafamın içinde bir bomba patladı sanki, ensem kaskatı kesildi, ışığa bakamıyorum ve fışkırır gibi kustum!",
        "assistant": """**Klinik Değerlendirme:** Saniyeler içinde en yüksek şiddete ulaşan ani gök gürültüsü tarzı baş ağrısı (thunderclap headache), meningismus (ense sertliği), fotofobi ve fışkırır kusma Subaraknoid Kanama (Anevrizma Rüptürü) tablosunu düşündürür.

**Aciliyet:** ESI Seviye 2: Acil (Acil Servis / Beyin Cerrahisi Değerlendirmesi)

**Önerilen Aksiyon Planı:** 
1. Derhal 112 Acil Servis çağrılmalı ve acil servise gidilmelidir.
2. Kontrastsız Acil Beyin BT ve gerekirse Lomber Ponksiyon ile subaraknoid kanama ekarte edilmelidir.
3. Kan basıncı sıkı kontrol altına alınmalıdır.

**🚨 Kırmızı Bayraklar:** Uykuya meyil, letarji, çift görme veya vücudun bir yarısında güçsüzlük akut nörolojik kötüleşmeyi gösterir."""
    },
    {
        "user": "Astım hastasıyım, 2 saattir nefes alamıyorum. Mavi fısfısımı 6 kere çektim hiç açılmadı. Kesik kesik tek kelime zor konuşuyorum, tırnaklarım morarmaya başladı.",
        "assistant": """**Klinik Değerlendirme:** Kısa etkili beta-2 agonistlere yanıtsızlık, cümle kuramama, yardımcı solunum kaslarının kullanımı ve siyanoz Hayatı Tehdit Eden Şiddetli Akut Astım Atağı (Status Asthmaticus) tablosudur. Solunum arresti riski yüksektir.

**Aciliyet:** ESI Seviye 2: Acil (Kırmızı Alan / Nebülizasyon ve Sistemik Steroid)

**Önerilen Aksiyon Planı:** 
1. Derhal 112 Acil Servis aranmalıdır.
2. Ambulans gelene kadar aralıklı olarak nebülizatör veya inhaler ile salbutamol kullanımına devam edilmeli, hasta dik oturtulmalıdır.
3. Acil serviste oksijen desteği, yüksek doz nebül bronkodilatör ve IV kortikosteroid tedavisi başlanmalıdır.

**🚨 Kırmızı Bayraklar:** Sessiz akciğer (hırıltının tamamen kesilip solunum sesinin duyulmaması) ve bilinç bulanıklığı solunum kas tükenmişliği ve entübasyon ihtiyacını gösterir."""
    },
    {
        "user": "26 yaşında kadın hasta, 6 haftalık gebelik şüphesi var. 1 saat önce başlayan çok şiddetli tek taraflı alt karın ağrısı, vajinal kanama ve ayağa kalkınca göz kararması bayılma hissi var.",
        "assistant": """**Klinik Değerlendirme:** Amenore (adet gecikmesi), ani başlayan tek taraflı şiddetli pelvik ağrı, vajinal kanama ve ortostatik senkop Rüptüre Ektopik Gebelik (Dış Gebelik Kanaması) tablosudur. Masif intraabdominal hemoraji riski yüksektir.

**Aciliyet:** ESI Seviye 2: Acil (Acil Servis / Acil Kadın Doğum Konsültasyonu)

**Önerilen Aksiyon Planı:** 
1. 112 ile vakit kaybetmeden ameliyathanesi bulunan bir acil servise nakil sağlanmalıdır.
2. Yatak başı transvajinal/abdominal ultrasonografi ile batın içi serbest sıvı ve hematokrit takibi yapılmalıdır.
3. Acil damar yolu açılarak izotonik sıvı replasmanı başlatılmalıdır.

**🚨 Kırmızı Bayraklar:** Kan basıncının düşmesi, taşikardi, soğuk terleme ve bilinç kaybı hemorajik şoka gidişi gösterir."""
    },
    {
        "user": "16 yaşında erkek hasta. Gece yarısı aniden başlayan sol testis torbasında çok şiddetli ağrı, şişlik ve kusma tarifliyor. Testisi yukarı doğru çekilmiş duruyor.",
        "assistant": """**Klinik Değerlendirme:** Ani başlangıçlı şiddetli skrotal ağrı, asimetrik yukarı yerleşimli testis ve kremaster refleks kaybı Testis Torsiyonu için acil alarm bulgusudur. Testisin iskemik nekrozu 6 saat içinde kalıcı organ kaybına yol açar.

**Aciliyet:** ESI Seviye 2: Acil (Acil Üroloji Konsültasyonu / İlk 6 Saat Kritik)

**Önerilen Aksiyon Planı:** 
1. Dakikalar dahi kaybetmeden en yakın hastane Acil Servisi'ne gidilmelidir.
2. Ağızdan yeme-içme derhal kesilmelidir (acil cerrahi eksplorasyon gerekebilir).
3. Acil Renkli Doppler Skrotal USG ile vasküler akım kontrol edilmeli ve cerrahi detorsiyon planlanmalıdır.

**🚨 Kırmızı Bayraklar:** 6 saatin aşılması testisin canlılığını kaybetmesine (orşiektomi) sebep olabileceğinden poliklinik beklenmemelidir."""
    },
    {
        "user": "18 aylık bebeğimin ateşi 39.5 dereceye çıktı. Birden gözleri arkaya kaydı, kolları bacakları titreyip kaskatı kesildi, yaklaşık 1 dakika sürdü şu an uykuya meyilli.",
        "assistant": """**Klinik Değerlendirme:** Yüksek ateş zemininde gelişen jeneralize tonik-klonik nöbet Febril Konvülsiyon tablosudur. İlk nöbet olması durumunda menenjit ve ensefalit gibi intrakraniyal enfeksiyonların ekarte edilmesi gerekir.

**Aciliyet:** ESI Seviye 2: Acil (Acil Servis / Pediatri Değerlendirmesi)

**Önerilen Aksiyon Planı:** 
1. 112 Acil Servis aranarak pediatrik acil servise başvurulmalıdır.
2. Bebeğin üzerindeki kalın giysiler çıkarılmalı, ılık uygulama yapılmalıdır (asla buzlu su veya sirkeli su kullanılmamalıdır).
3. Kusma ihtimaline karşı bebek yan yatırılmalı, ağzına emzik veya parmak sokulmamalıdır.

**🚨 Kırmızı Bayraklar:** Nöbetin 5 dakikadan uzun sürmesi, 24 saatte tekrarlaması, bebeğin uyandırılamaması veya peteşi/purpura (mor döküntü) menenjiti gösterir."""
    },
    {
        "user": "Sol gözümde yarım saat içinde çok fena bir zonklama ve ağrı başladı, ışıkların etrafında gökkuşağı halkaları görüyorum, midem bulanıyor ve gözüm taş gibi sert.",
        "assistant": """**Klinik Değerlendirme:** Ani göz ağrısı, renkli hareler görme, kornea bulanıklığı, midriyatik pupil ve artmış göz içi basıncı Akut Açı Kapanması Glokomu krizidir. Hızlı tedavi edilmezse optik sinir hasarına ve kalıcı körlüğe neden olur.

**Aciliyet:** ESI Seviye 2: Acil (Acil Servis / Göz Hastalıkları Konsültasyonu)

**Önerilen Aksiyon Planı:** 
1. Derhal Göz Hastalıkları branşının bulunduğu bir Acil Servise başvurulmalıdır.
2. Göz içi basıncını düşürmek için acil topikal beta-bloker, alfa-2 agonist ve sistemik karbonik anhidraz inhibitörü (asetazolamid/mannitol) başlanmalıdır.
3. Göze baskı veya masaj yapılmamalıdır.

**🚨 Kırmızı Bayraklar:** Görme keskinliğinde ani belirgin düşüş acil cerrahi veya lazer iridotomi gerektirir."""
    },

    # --- ESI-3: ORTA DERECE ACIL (COKLU TETKIK / 30-60 DAKIKA / VITAL STABIL) ---
    {
        "user": "Sağ yan boşluğumdan (böğrümden) kasığıma doğru dalgalar halinde vuran kıvrandırıcı bir ağrı var. Yerimde duramıyorum, idrarım pembe renkli geldi. Ateşim yok.",
        "assistant": """**Klinik Değerlendirme:** Yan boşluktan kasığa yayılan aralıklı şiddetli kolik ağrı ve hematüri Akut Renal Kolik (Üriner Sistem Taş Hastalığı / Nefrolitiyazis) ile uyumludur. Vital bulgular stabil olduğunda ESI 3 kategorisindedir.

**Aciliyet:** ESI Seviye 3: Orta Derece Acil (Acil Servis / Çoklu Kaynak: İdrar Tetkiki, USG / BT)

**Önerilen Aksiyon Planı:** 
1. Acil Servise başvurarak ağrı kontrolü (IV NSAİİ/spazmolitik) ve hidrasyon tedavisi alınız.
2. İdrar tahlili ve Kontrastsız Düşük Doz Taş BT ile taş boyutu ve obstrüksiyon seviyesi belirlenmelidir.
3. Enfeksiyon parametreleri (lökosit, CRP) kontrol edilmelidir.

**🚨 Kırmızı Bayraklar:** Tabloya 38.5°C üzeri ateş, titreme veya idrar yapamama (anüri) eklenirse enfekte obstrüksiyon (ürosepsis riski) nedeniyle durum ESI-2'ye yükselir."""
    },
    {
        "user": "Dünden beri 7-8 kez sulu ishal oldum, su içsem midem bulanıp kusuyorum. Çok halsizim, ağzım dilim kurudu ama ateşim yok, kanamam yok.",
        "assistant": """**Klinik Değerlendirme:** Sık sulu diyare, kusmaya bağlı yetersiz oral alım ve erken dehidratasyon bulguları Akut Gastroenterit tablosunu gösterir. Sıvı-elektrolit dengesizliği tespiti için intravenöz hidrasyon ve kan tetkikleri gerekebilir.

**Aciliyet:** ESI Seviye 3: Orta Derece Acil (Acil Servis Sarı Alan / IV Sıvı Desteği)

**Önerilen Aksiyon Planı:** 
1. Acil servise başvurarak serum fizyolojik hidrasyonu ve antiemetik desteği alınız.
2. Serum elektrolitleri (sodyum, potasyum), böbrek fonksiyon testleri (üre, kreatinin) incelenmelidir.
3. Yağsız, tuzlu kraker, patates haşlaması ve pirinç lapası gibi diyet uygulanmalı, şekerli içeceklerden kaçınılmalıdır.

**🚨 Kırmızı Bayraklar:** Kanlı veya katran renkli dışkılama, 38.5°C üzeri yüksek ateş, şiddetli karın krampı veya ayağa kalkınca göz kararması (ortostatik senkop) durumunda acil değerlendirme gerekir."""
    },
    {
        "user": "Merdivenden inerken ayağım takıldı, sol el bileğimin üzerine düştüm. Bileğimde şekil bozukluğu var, çok şişti ve dokunamıyorum. Parmaklarımı hareket ettirebiliyorum, his kaybı yok.",
        "assistant": """**Klinik Değerlendirme:** El üzerine düşme mekanizması sonrası oluşan belirgin deformite ve şişlik Kapalı Distal Radius Kırığı (Colles / Smith kırığı) düşündürür. Distal nabızların açık olması ve duyu kusuru bulunmaması acil vasküler tehlikeyi azaltır.

**Aciliyet:** ESI Seviye 3: Orta Derece Acil (Acil Servis / İki Yönlü Grafi ve Ortopedi Konsültasyonu)

**Önerilen Aksiyon Planı:** 
1. Acil Servise başvurunuz; iki yönlü el bileği röntgeni (X-Ray) çekilmelidir.
2. Kolunuzu bir tülbent veya üçgen askı ile göğsünüze sabitleyiniz (hareketsizleştirme).
3. Şişliğin artmasını önlemek için elinizdeki yüzük ve saatleri derhal çıkarınız; havluya sarılı soğuk uygulama yapınız.

**🚨 Kırmızı Bayraklar:** Parmaklarda morarma, soğuma, nabızsızlık veya tam hissizlik gelişmesi durumunda acil redüksiyon gerekir."""
    },
    {
        "user": "Dün akşam ağır bir yağlı et yemeğinden sonra karnımın sağ üst tarafında kaburgamın altına saplanan ve sağ kürek kemiğime vuran bir ağrı başladı. Ateşim ve sarılığım yok.",
        "assistant": """**Klinik Değerlendirme:** Yağlı öğün sonrası tetiklenen sağ üst kadran ağrısı ve skapulaya radyasyon Biliyer Kolik / Akut Kolesistit (Safra Kesesi Taşı/İltihabı) tablosunu destekler.

**Aciliyet:** ESI Seviye 3: Orta Derece Acil (Acil Servis / Batın Ultrasonografisi ve Karaciğer Enzimleri)

**Önerilen Aksiyon Planı:** 
1. Acil Servise başvurunuz; hepatobiliyer USG ile safra kesesi duvar kalınlığı ve taş varlığı incelenmelidir.
2. Karaciğer fonksiyon testleri (AST, ALT, Bilirubin, ALP, GGT) ve amilaz/lipaz düzeylerine bakılmalıdır.
3. Yağlı gıdalardan uzak durunuz.

**🚨 Kırmızı Bayraklar:** Göz aklarında ve ciltte sararma (sarılık), 38.5°C üzeri ateş ve titreme eklenirse Akut Kolanjit (acil ESI-2) riski düşünülmelidir."""
    },
    {
        "user": "4 gündür balgamlı öksürüğüm, halsizliğim ve 38.2 derece ateşim var. Nefes alırken sağ göğsüm hafif batıyor. Oksijen seviyem evde parmaktan baktım %96 çıktı.",
        "assistant": """**Klinik Değerlendirme:** Pürülan balgam, subfebril ateş ve plöritik göğüs ağrısı bulguları Toplum Kökenli Pnömoni (Zatürre) ile uyumludur. Oksijen saturasyonunun %95 üzerinde olması ve takipne olmaması stabil tablodur.

**Aciliyet:** ESI Seviye 3: Orta Derece Acil (Acil Servis veya Göğüs Hastalıkları / PA Akciğer Grafisi ve Kan Tetkiki)

**Önerilen Aksiyon Planı:** 
1. Acil servise veya Göğüs Hastalıkları polikliniğine başvurunuz; PA Akciğer Grafisi (Röntgen) ve kan sayımı (lökositoz, CRP) yapılmalıdır.
2. CRB-65 skorlamasına göre ayaktan veya yatarak antibiyotik tedavisi hekim tarafından planlanmalıdır.
3. Bol sıvı alınız ve istirahat ediniz.

**🚨 Kırmızı Bayraklar:** Oksijen saturasyonunun %92 altına inmesi, solunum sayısının dakikada 30'u aşması veya konfüzyon (bilinç bulanıklığı) durumunda derhal acil müdahale gerekir."""
    },

    # --- ESI-4: YARI ACIL / POLIKLINIK (TEKIL BASIT SIKAYET / ACIL SERVIS GEREKMEZ) ---
    {
        "user": "öksürüyorum",
        "assistant": """**Klinik Değerlendirme:** İzole ve tek başına bildirilen öksürük şikayeti; nefes darlığı, göğüs ağrısı, yüksek ateş, balgamda kan veya siyanoz (morarma) eşlik etmediği sürece acil hayati bir tehdit oluşturmaz. Viral üst solunum yolu enfeksiyonu sekeli veya farengeal tahriş ile uyumludur.

**Aciliyet:** ESI Seviye 4: Yarı Acil / Poliklinik Başvurusu (Acil Servis Başvurusu Gerekmez)

**Önerilen Aksiyon Planı:** 
1. Acil servise başvurmanıza gerek yoktur. Şikayetlerinizin değerlendirilmesi için Aile Hekiminize veya Göğüs Hastalıkları polikliniğine başvurabilirsiniz.
2. Bol ılık sıvı tüketiniz, boğazı tahriş eden kuru hava, toz ve sigara dumanından uzak durunuz.
3. Hekiminize danışmadan gelişigüzel antibiyotik veya öksürük kesici şurup kullanmayınız.

**🚨 Kırmızı Bayraklar:** Nefes darlığı, göğüste baskı hissi, 38.5°C üzeri dirençli ateş, balgamla kan gelmesi veya dudaklarda morarma gelişirse derhal acil servise başvurunuz."""
    },
    {
        "user": "3 gündür kuru öksürüğüm var, boğazım gıcıklanıyor. Ateşim yok, nefes darlığım yok, sadece öksürük rahatsız ediyor. Ne yapmalıyım?",
        "assistant": """**Klinik Değerlendirme:** 3 gündür devam eden, dispne veya sistemik bulguların (ateş, kilo kaybı, hemoptizi) eşlik etmediği kuru öksürük; akut viral bronşit başlangıcı, post-nazal akıntı veya farenks irritasyonunu işaret eder. Acil servis endikasyonu yoktur.

**Aciliyet:** ESI Seviye 4: Yarı Acil / Poliklinik Başvurusu

**Önerilen Aksiyon Planı:** 
1. Aile hekiminize başvurarak muayene olabilirsiniz.
2. Odanızı nemlendiriniz, ılık bitki çayları ve bal tüketebilirsiniz.
3. İstirahat ediniz ve bol su içiniz.

**🚨 Kırmızı Bayraklar:** Hırıltılı solunum, derin nefes alırken göğüste batma, istirahatte nefes yetmezliği veya ateş başlarsa acil servise başvurunuz."""
    },
    {
        "user": "2 haftadır geçmeyen inatçı bir öksürüğüm var. Ateşim hiç çıkmadı, nefes darlığım yok, genel durumum gayet iyi ama öksürük geçmedi.",
        "assistant": """**Klinik Değerlendirme:** 2 haftadır süregelen subakut izole öksürük; geçirilmiş bir üst solunum yolu enfeksiyonu sonrası gelişen bronşiyal hiperreaktivite (post-enfeksiyöz öksürük), alerjik rinit veya gastroözofageal reflü lehinedir. Acil durum değildir.

**Aciliyet:** ESI Seviye 4: Yarı Acil / Poliklinik Başvurusu (Acil Servise Gerek Yoktur)

**Önerilen Aksiyon Planı:** 
1. Göğüs Hastalıkları veya Kulak Burun Boğaz polikliniğinden randevu alarak detaylı muayene ve gereğinde Akciğer Grafisi / Solunum Fonksiyon Testi yaptırınız.
2. Gece reflüsünü önlemek için yatmadan 3 saat önce yemek yemeyi kesiniz ve yastığınızı hafif yükseltiniz.

**🚨 Kırmızı Bayraklar:** Balgamda kan görülmesi, gece terlemesi, istemsiz kilo kaybı veya eforla gelen nefes darlığı durumunda hekim değerlendirmesini öne alınız."""
    },
    {
        "user": "Dünden beri idrarımı yaparken yanma hissediyorum ve sık sık tuvalete koşuyorum. Ateşim yok, sırtımda veya böğrümde ağrı yok.",
        "assistant": """**Klinik Değerlendirme:** Dizüri (idrarda yanma) ve pollaküri (sık idrara çıkma) semptomları; ateş, titreme veya kostovertebral açı hassasiyeti bulunmadığında komplike olmayan Akut Alt Üriner Sistem Enfeksiyonu (Sistit) tablosunu destekler.

**Aciliyet:** ESI Seviye 4: Yarı Acil / Poliklinik Başvurusu (Acil Servise Gerek Yoktur)

**Önerilen Aksiyon Planı:** 
1. Aile hekimliği veya Üroloji polikliniğine başvurarak Tam İdrar Tahlili (TİT) ve İdrar Kültürü yaptırınız.
2. Günde en az 2.5 - 3 litre su tüketerek idrar yollarının yıkanmasını sağlayınız.
3. Kültür sonucu çıkmadan veya hekim reçete etmeden eczaneden rastgele antibiyotik almayınız.

**🚨 Kırmızı Bayraklar:** 38°C üzeri ateş, titreme, bel/böğür ağrısı, bulantı-kusma başlarsa üst üriner sistem enfeksiyonu (Pyelonefrit) şüphesiyle acil servise başvurunuz."""
    },
    {
        "user": "Mutfakta elma soyarken parmağımı bıçakla kestim. Temiz bıçaktı, 2 cm kadar açık duruyor. Temiz bez bastırınca kanaması tamamen durdu ama dikiş gerekir mi?",
        "assistant": """**Klinik Değerlendirme:** Kanama kontrolü sağlanmış, aktif pulsasyon göstermeyen, sinir-tendon kesisi bulgusu (parmak ucunda hissizlik, fleksiyon kaybı) içermeyen yüzeysel lineer laserasyon primer sütürasyon (dikiş) gerektirir.

**Aciliyet:** ESI Seviye 4: Yarı Acil / Ayaktan Müdahale

**Önerilen Aksiyon Planı:** 
1. En yakın acil servis yeşil alanına veya semt polikliniğine başvurarak yara temizliği ve sütür (dikiş) attırınız. (Yara dudaklarının 6-8 saat içinde kapatılması idealdir).
2. Son 5-10 yıl içinde Tetanos aşısı olmadıysanız hekiminize danışarak tetanos profilaksisi yaptırınız.
3. Yarayı temiz bir gazlı bezle sarılı tutunuz.

**🚨 Kırmızı Bayraklar:** Parmak ucunda kalıcı hissizlik, parmağı bükememe veya zonklayıcı pürülan akıntı durumunda hekiminize bilgi veriniz."""
    },
    {
        "user": "Halı sahada top oynarken ayak bileğimi burktum. Şişti ve biraz morardı ama üzerine basabiliyorum, sekerek de olsa 4-5 adım yürüdüm.",
        "assistant": """**Klinik Değerlendirme:** Ottawa Ayak Bileği Kuralları uyarınca hastanın travma sonrası ve muayene anında 4 adım atabilmesi, kemik fraktürü (kırık) olasılığını oldukça düşürür; akut lateral ligaman burkulması/gerilmesi lehinedir.

**Aciliyet:** ESI Seviye 4: Yarı Acil / Ayaktan Poliklinik Başvurusu

**Önerilen Aksiyon Planı:** 
1. Poliklinik veya acil yeşil alana başvurarak fiziki muayene olabilirsiniz.
2. **R.I.C.E. Protokolü:** İstirahat (Rest), günde 3-4 kez havluya sarılı 15 dk buz uygulaması (Ice), elastik bandajla kompresyon (Compression) ve bacağı kalp seviyesinde yüksekte tutma (Elevation).
3. Zorlayıcı spor aktivitelerine ara veriniz.

**🚨 Kırmızı Bayraklar:** Ayağın üzerine kesinlikle basamama, malleol kemik çıkıntılarında şiddetli kemik hassasiyeti veya ayakta tam hissizlik gelişirse röntgen ile kırık ekarte edilmelidir."""
    },
    {
        "user": "2 gündür boğazım fena acıyor, yutkunurken zorlanıyorum. Aynada baktım bademciklerim kızarmış. Ateşim 37.4, nefes almamda sorun yok.",
        "assistant": """**Klinik Değerlendirme:** Yutkunma ile artan boğaz ağrısı ve tonsiller hiperemi; solunum sıkıntısı, trismus (çene kilitlenmesi) veya tek taraflı peritonsiller apse bulguları olmaksızın Akut Tonsillofarenjit ile uyumludur.

**Aciliyet:** ESI Seviye 4: Yarı Acil / Poliklinik Başvurusu

**Önerilen Aksiyon Planı:** 
1. Aile hekiminize başvurarak Centor kriterleri doğrultusunda boğaz muayenesi olunuz (bakteriyel/viral ayrımı).
2. Ilık tuzlu su ile gargara yapınız, yumuşak ve ılık gıdalarla besleniniz.
3. Hekim önerisi olmadan gereksiz antibiyotik kullanmayınız.

**🚨 Kırmızı Bayraklar:** Ağzı açamama, tükürüğünü yutamama, boyunda asimetrik şişlik veya nefes darlığı başlarsa acil servise gidilmelidir."""
    },
    {
        "user": "3 gündür burnum tıkalı, alın ve yanak bölgemde öne eğilince zonklayan bir dolgunluk hissi var. Hafif sarı burun akıntım oluyor. Ateşim yok.",
        "assistant": """**Klinik Değerlendirme:** Öne eğilmekle artan fasiyal dolgunluk ve bası hissi, pürülan nazal akıntı Akut Rinosinüzit tablosunu gösterir. Acil cerrahi veya hayati bir tehlike söz konusu değildir.

**Aciliyet:** ESI Seviye 4: Yarı Acil / Poliklinik Başvurusu

**Önerilen Aksiyon Planı:** 
1. Kulak Burun Boğaz veya Aile Hekimliği polikliniğine başvurunuz.
2. Serum fizyolojik ile burun yıkaması (nazal lavaj) yapınız ve buhar inhalasyonu uygulayınız.
3. Dekonjestan burun spreylerini 4-5 günden uzun süre kullanmayınız (rebound konjesyon riski).

**🚨 Kırmızı Bayraklar:** Göz çevresinde şişlik, kızarıklık, çift görme veya şiddetli ense sertliği eklenirse orbital/intrakraniyal komplikasyon riski nedeniyle acil servise gidilmelidir."""
    },

    # --- ESI-5: ACIL DEGIL / RUTIN TAKIP (SIFIR KAYNAK / POLIKLINIK) ---
    {
        "user": "Rutin tansiyon ilacımın süresi bitti, kutum tükendi. Sağlık ocağına gitmeden acilde yazdırabilir miyim?",
        "assistant": """**Klinik Değerlendirme:** Aktif semptom, organ hasarı bulgusu veya hemodinamik instabilite olmaksızın rutin kronik ilaç reçetesi yenileme talebidir. Acil servisler kronik ilaç yazdırma yeri değildir.

**Aciliyet:** ESI Seviye 5: Acil Değil / Rutin Poliklinik İşlemi

**Önerilen Aksiyon Planı:** 
1. Acil servise başvurmayınız. Kayıtlı olduğunuz Aile Sağlığı Merkezi'ne (Sağlık Ocağı) mesai saatleri içinde giderek ilacınızı yazdırınız.
2. İlacınızı alana kadar tuz kısıtlamasına dikkat ediniz ve evde istirahat halinde tansiyonunuzu takip ediniz.

**🚨 Kırmızı Bayraklar:** Şiddetli baş ağrısı, görme bulanıklığı, göğüs ağrısı veya tansiyon değerinizin 180/110 mmHg üzerine çıkması durumunda acil servise başvurunuz."""
    },
    {
        "user": "10 gün önce koluma atılan 3 dikişi aldırmak istiyorum. Yara tamamen kapanmış, hiç kızarıklık veya akıntı yok.",
        "assistant": """**Klinik Değerlendirme:** İyileşme periyodunu tamamlamış, enfeksiyon bulgusu (eritem, pürülan akıntı, ısı artışı) taşımayan kapalı cerrahi kutanöz dikişlerin alınması işlemidir. Acil servis endikasyonu yoktur.

**Aciliyet:** ESI Seviye 5: Acil Değil / Rutin Ayaktan İşlem

**Önerilen Aksiyon Planı:** 
1. Aile Sağlığı Merkezi'ndeki aile hekiminize veya pansuman polikliniğine başvurarak dikişlerinizi steril şartlarda aldırabilirsiniz.
2. Dikişler alındıktan sonra hekiminizin belirttiği süre boyunca bölgeyi temiz ve kuru tutunuz.

**🚨 Kırmızı Bayraklar:** Yarada açılma, kötü kokulu akıntı veya etrafında yayılan sıcak kızarıklık görülürse hekime gösteriniz."""
    },
    {
        "user": "Ameliyat olalı 1 hafta oldu, pansumanımı değiştirtmek istiyorum. Ağrım sızım yok.",
        "assistant": """**Klinik Değerlendirme:** Rutin cerrahi yara pansumanı değişimi talebidir. Sistemik ateş, yara yeri dehisansı (açılması) veya aktif hemoraji bulunmadığından acil müdahale gerektirmez.

**Aciliyet:** ESI Seviye 5: Acil Değil / Rutin Pansuman

**Önerilen Aksiyon Planı:** 
1. Aile hekiminize veya ameliyatınızı gerçekleştiren cerrahi kliniğin pansuman birimine başvurunuz.
2. Yarayı ıslatmayınız ve steril olmayan materyallerle pansuman yapmayınız.

**🚨 Kırmızı Bayraklar:** Yaranın açılması, aktif sızıntı veya 38°C üzeri ateş gelişirse cerrahi ekibinize başvurunuz."""
    },
    {
        "user": "2 aydır dizimde hafif kireçlenme sızısı var, hava soğuyunca sızlıyor. Acil bir şey değil ama danışmak istedim.",
        "assistant": """**Klinik Değerlendirme:** Aylardır süren, istirahatle azalan, eklemde kilitlenme, sıcaklık artışı veya efüzyon (akut sıvı toplanması) bulunmayan kronik osteoartrit (kireçlenme) tablosudur. Acil servis kapsamında değildir.

**Aciliyet:** ESI Seviye 5: Acil Değil / Elektif Poliklinik Takibi

**Önerilen Aksiyon Planı:** 
1. Fizik Tedavi ve Rehabilitasyon (FTR) veya Ortopedi polikliniğinden randevu alarak kıkırdak yapısını koruyucu egzersizler ve medikal tedavi planlayınız.
2. Dize aşırı yük bindiren merdiven inip çıkma veya ağır kaldırma hareketlerinden kaçınınız.

**🚨 Kırmızı Bayraklar:** Eklemde ani kilitlenme, bacağın üzerine hiç basamama veya eklem üzerinde kızarıklık ve ateş olursa enfeksiyöz artrit açısından incelenmelidir."""
    },
    {
        "user": "Dünden beri burnum akıyor ve ara sıra hapşırıyorum. Boğazımda hafif kaşıntı var, ateşim yok halsizliğim yok.",
        "assistant": """**Klinik Değerlendirme:** Akut seröz rinore ve hapşırma; ateş ve genel durum bozukluğu bulunmadığında hafif Akut Viral Rinit (Basit Nezle / Soğuk Algınlığı) tablosudur.

**Aciliyet:** ESI Seviye 5: Acil Değil / Evde Destek Tedavisi

**Önerilen Aksiyon Planı:** 
1. Acil servise veya polikliniğe gitmenize gerek yoktur.
2. Bol C vitamini ve ılık sıvılar alınız, istirahat ediniz.
3. El hijyenine dikkat ederek enfeksiyonun yayılmasını önleyiniz.

**🚨 Kırmızı Bayraklar:** 3 günden uzun süren 38.5°C üzeri ateş, şiddetli nefes darlığı veya koyu kanlı burun akıntısı olursa hekime başvurunuz."""
    },

    # --- INGILIZCE TRİYAJ VAKALARI (BILINGUAL ESI TRIAGE COMPLIANCE) ---
    {
        "user": "I have been coughing for 2 days, no fever, no shortness of breath, just an irritating dry cough.",
        "assistant": """**Klinik Değerlendirme:** İki gündür süren, ateş, dispne veya göğüs ağrısının eşlik etmediği izole kuru öksürük üst solunum yolu tahrişi veya viral farenjit ile uyumludur. Acil bir tehdit oluşturmaz.

**Aciliyet:** ESI Seviye 4: Yarı Acil / Poliklinik Başvurusu (Acil Servise Gerek Yoktur)

**Önerilen Aksiyon Planı:** 
1. Acil servise başvurmanıza gerek yoktur. Aile Hekiminize başvurabilirsiniz.
2. Bol ılık sıvı tüketiniz ve istirahat ediniz.

**🚨 Kırmızı Bayraklar:** Nefes darlığı, göğüs ağrısı, 38.5°C üzeri ateş veya balgamda kan görülmesi durumunda acil servise başvurunuz."""
    },
    {
        "user": "Severe crushing chest pain radiating to my left shoulder and jaw, profuse sweating and nausea for 30 minutes.",
        "assistant": """**Klinik Değerlendirme:** Retrosternal crushing chest pain radiating to the left shoulder and mandible accompanied by diaphoresis and nausea represents classic signs of Acute Coronary Syndrome (Myocardial Infarction).

**Aciliyet:** ESI Seviye 2: Acil (Immediate Emergency Department / Cardiac Resuscitation Room)

**Önerilen Aksiyon Planı:** 
1. Call 112 / Emergency Medical Services immediately. Do not attempt to drive.
2. Rest completely and avoid any physical exertion.
3. Unless contraindicated, chew 300 mg of Aspirin immediately.

**🚨 Kırmızı Bayraklar:** Syncope, hypotension, or loss of consciousness requires immediate cardiopulmonary resuscitation."""
    },
    {
        "user": "Sudden weakness on the right side of the body, facial drooping, and slurred speech starting 20 minutes ago.",
        "assistant": """**Klinik Değerlendirme:** FAST-positive acute unilateral hemiparesis, facial asymmetry, and dysarthria are pathognomonic for Acute Ischemic Stroke. Immediate door-to-needle time is critical.

**Aciliyet:** ESI Seviye 2: Acil (Acute Stroke Protocol / Urgent Brain Imaging)

**Önerilen Aksiyon Planı:** 
1. Call 112 immediately. Note exact time of onset for thrombolytic eligibility.
2. Keep patient supine with head elevated 30 degrees. Do not give any oral fluids or food.
3. Transport directly to a comprehensive stroke center.

**🚨 Kırmızı Bayraklar:** Airway compromise or altered mental status requires immediate emergency airway management."""
    },
    {
        "user": "I twisted my ankle while jogging. There is mild swelling but I was able to walk 4 steps right after. Is it broken?",
        "assistant": """**Klinik Değerlendirme:** Under the Ottawa Ankle Rules, the ability to bear weight and take 4 steps immediately and in the clinic makes an acute fracture highly unlikely; findings favor an acute lateral ligament sprain.

**Aciliyet:** ESI Seviye 4: Yarı Acil / Outpatient Evaluation

**Önerilen Aksiyon Planı:** 
1. Consult an outpatient clinic or primary care physician.
2. Apply the R.I.C.E. protocol: Rest, Ice (wrapped in towel, 15 min 3x/day), Compression with elastic wrap, and Elevation.
3. Avoid vigorous sports until full painless range of motion is restored.

**🚨 Kırmızı Bayraklar:** Inability to bear weight, severe bone tenderness over malleoli, or neurovascular deficits warrant plain radiography."""
    },
    {
        "user": "Burning sensation during urination and frequent urination since yesterday. No fever, no flank pain.",
        "assistant": """**Klinik Değerlendirme:** Dysuria and pollakiuria in the absence of fever, chills, or costovertebral tenderness point to uncomplicated Acute Cystitis (Lower Urinary Tract Infection).

**Aciliyet:** ESI Seviye 4: Yarı Acil / Poliklinik Başvurusu

**Önerilen Aksiyon Planı:** 
1. Consult a primary care doctor or urologist for urinalysis and culture.
2. Increase daily fluid intake to 2.5-3 liters.
3. Do not take self-prescribed antibiotics before obtaining urine cultures.

**🚨 Kırmızı Bayraklar:** High fever (>38°C), flank pain, or severe rigors suggest ascending infection (Pyelonephritis) requiring urgent care."""
    },
    {
        "user": "I just need a refill for my daily hypertension medicine, my prescription ran out.",
        "assistant": """**Klinik Değerlendirme:** Routine chronic medication refill request with no acute symptoms or organ dysfunction. Emergency Department visit is inappropriate.

**Aciliyet:** ESI Seviye 5: Acil Değil / Rutin Takip

**Önerilen Aksiyon Planı:** 
1. Visit your primary care clinic or family practitioner during normal working hours for prescription renewal.
2. Maintain dietary salt restriction and monitor resting blood pressure at home.

**🚨 Kırmızı Bayraklar:** Severe headache, chest pressure, visual disturbances, or systolic BP exceeding 180 mmHg requires urgent evaluation."""
    }
]

# Veri setini Llama-3 formatinda hazirla
all_conversations = []
for item in CURATED_TRIAGE_CASES:
    all_conversations.append({
        "messages": [
            {"role": "system", "content": SYS_CDSS},
            {"role": "user", "content": item["user"]},
            {"role": "assistant", "content": item["assistant"]}
        ]
    })

# Verileri rastgele karistir
random.seed(42)
random.shuffle(all_conversations)

# %90 Train, %10 Validation olarak bol
split_idx = max(1, int(len(all_conversations) * 0.9))
train_data = all_conversations[:split_idx]
val_data = all_conversations[split_idx:]

with open(train_path, "w", encoding="utf-8") as f:
    for ex in train_data:
        f.write(json.dumps(ex, ensure_ascii=False) + "\\n")
with open(val_path, "w", encoding="utf-8") as f:
    for ex in val_data:
        f.write(json.dumps(ex, ensure_ascii=False) + "\\n")

# Gelecek oturumlar icin Google Drive 'AI/TriageCore/data' dizinine kaydet
shutil.copy(train_path, DRIVE_DATA_DIR / "train.jsonl")
shutil.copy(val_path, DRIVE_DATA_DIR / "val.jsonl")

print(f"[+] TEMIZ VE YUKSEK KALITELI {len(all_conversations)} KLINIK VAKA HAZIRLANDI! (Egitim: {len(train_data)}, Dogrulama: {len(val_data)})")
print("[+] ChatDoctor sablon spamleri tamamen cikarildi; saf klinik triyaj vakalari Google Drive'a yedeklendi.")

# Veri dosyalarini HuggingFace Dataset formatinda yukle
data_files = {"train": str(train_path), "validation": str(val_path)}
dataset = load_dataset("json", data_files=data_files)
print(f"[+] HuggingFace Dataset basariyla hazir! Egitim: {len(dataset['train'])}, Dogrulama: {len(dataset['validation'])}")

# ==============================================================================
# 4. MODEL EGITIM YAPILANDIRMASI VE SFTTRAINER (Hatasiz & Hizli Egitim)
# ==============================================================================
# A100/L4 varsa batch size 4, T4 ise 2 kullan
micro_batch = 4 if is_bfloat16_supported else 2
grad_accum = 4 if is_bfloat16_supported else 4

TargetConfigClass = SFTConfig if UseSFTConfig else TrainingArguments
target_sig = inspect.signature(TargetConfigClass.__init__).parameters

# Egitim konfigurasyon sozluk yapisi olusturulur (3 Epoch ~2-3 dk surer)
config_kwargs = {
    "output_dir": str(CHECKPOINT_DIR),
    "num_train_epochs": 3,                   # 3 Epoch ile model klinik formata tam oturur
    "per_device_train_batch_size": micro_batch,
    "gradient_accumulation_steps": grad_accum,
    "optim": "paged_adamw_8bit",
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 5,
    "logging_steps": 5,
    "save_strategy": "no",                   # Hiz kazanmak ve Drive'i sisirmemek icin sadece egitim sonunda kaydet
    "bf16": is_bfloat16_supported,
    "fp16": not is_bfloat16_supported,
    "max_grad_norm": 0.3,
    "report_to": "none",
}

if "max_seq_length" in target_sig:
    config_kwargs["max_seq_length"] = 512

training_args = TargetConfigClass(**config_kwargs)

trainer_kwargs = {
    "model": model,
    "train_dataset": dataset["train"],
    "eval_dataset": dataset["validation"],
    "peft_config": None if hasattr(model, "peft_config") else lora_config,
    "args": training_args,
}

sft_sig = inspect.signature(SFTTrainer.__init__).parameters
if "max_seq_length" not in config_kwargs and "max_seq_length" in sft_sig:
    trainer_kwargs["max_seq_length"] = 512

if "tokenizer" in sft_sig:
    trainer_kwargs["tokenizer"] = tokenizer
elif "processing_class" in sft_sig:
    trainer_kwargs["processing_class"] = tokenizer

trainer = SFTTrainer(**trainer_kwargs)

# Egitimi baslat (Eski bozuk checkpoint'ler temizlendigi icin sifirdan egitilir)
print("[*] LoRA adaptoru temiz klinik veri seti uzerinde egitiliyor (Yaklasik 2-3 dakika)...")
trainer.train(resume_from_checkpoint=False)

# Nihai LoRA adaptorunu Google Drive '/MyDrive/AI/TriageCore/lora_adapter' icine kaydet!
trainer.model.save_pretrained(str(ADAPTER_DIR))
tokenizer.save_pretrained(str(ADAPTER_DIR))
print(f"\\n[+] EGITIM BASARIYLA TAMAMLANDI!")
print(f"[+] Temiz klinik model kalici olarak Google Drive'a kaydedildi: {ADAPTER_DIR}")
'''

def build_step7_code() -> str:
    """Step 7 kodunu repetition_penalty ve no_repeat_ngram_size ile olusturur."""
    return '''# Asenkron dongu, JSON, zaman, os, torch ve web sunucusu kutuphaneleri yuklenir
import asyncio
import json
import os
import re
import subprocess
import time
from typing import List, Optional

import nest_asyncio
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from peft import PeftModel
from pydantic import BaseModel

# Jupyter icerisinde asenkron Uvicorn calistirabilmek icin nest_asyncio uygulanir
nest_asyncio.apply()

# Eger 'model' degiskeni bellekte yoksa (Colab yeni acildiysa) Drive'dan otomatik yukle
if "model" not in globals():
    print("[*] Model bellekte bulunamadi, Google Drive 'AI/TriageCore/lora_adapter' uzerinden yukleniyor...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR) if (ADAPTER_DIR / "tokenizer_config.json").exists() else MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
    if (ADAPTER_DIR / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
        print("[+] Drive'daki egitilmis LoRA adaptoru basariyla yuklendi!")
    else:
        model = base_model
        print("[!] Drive'da adaptor bulunamadi, base model kullaniliyor.")

# Modeli degerlendirme (inference) moduna al
model.eval()

# Colab uzerinde calisacak FastAPI uygulamasi olusturulur
colab_app = FastAPI(title="TriageCore Remote Colab GPU Engine", version="2.0.0")
colab_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Istek semalari tanimlanir
class ChatMessageItem(BaseModel):
    role: str
    content: str

class RemoteChatRequest(BaseModel):
    session_id: str = "default"
    mode: str = "triage"
    messages: List[ChatMessageItem]
    case_id: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 512

def append_to_drive_log(entry: dict):
    """Her soruyu ve yaniti Google Drive 'AI/TriageCore/logs/colab_sessions.jsonl' dosyasina kaydeder."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "colab_sessions.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\\n")

@colab_app.get("/health")
def health():
    """Streamlit arayuzunun Colab GPU durumunu kontrol ettigi endpoint."""
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Colab-CPU"
    return {
        "status": "healthy",
        "model_loaded": True,
        "engine_type": "colab_gpu_qlora",
        "device": gpu_name,
        "version": "2.0.0",
    }

@colab_app.post("/v1/chat")
def generate_chat(req: RemoteChatRequest):
    """Cok turlu klinik triyaj ve Doktor-Hasta simulasyonu yanitlarini GPU'da uretir."""
    start_t = time.time()
    # Mesaj gecmisini modele uygun listeye cevir
    formatted_msgs = [{"role": m.role, "content": m.content} for m in req.messages]

    # Eger sistem istemi yoksa varsayilan Turkce klinik triyaj istemini en basa ekle
    has_system = any(m.role == "system" for m in req.messages)
    if not has_system:
        default_system = (
            "Sen TriageCore Acil Servis ve Klinik Karar Destek Asistanısın (CDSS). "
            "Görevin: Hastanın ilettiği semptomları değerlendirmek, olası tanıları ve aksiyon planını DAİMA ve KESİNLİKLE TÜRKÇE olarak açıklamaktır. "
            "Asla genel test listeleri tekrarlama veya İngilizce şablon cümleleri yazma. "
            "Yanıtını 'Klinik Değerlendirme', 'Aciliyet', 'Önerilen Aksiyon Planı' ve 'Kırmızı Bayraklar' formatında sun."
        )
        formatted_msgs.insert(0, {"role": "system", "content": default_system})

    # Llama-3 chat sablonunu uygula
    prompt = tokenizer.apply_chat_template(formatted_msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Tekrarlamalari onleyen repetition_penalty=1.15 ve no_repeat_ngram_size=3 ile uretim yap
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=max(0.05, req.temperature),
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Sadece yeni uretilen token'lari al
    input_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_len:]
    reply_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    elapsed_ms = round((time.time() - start_t) * 1000, 2)

    # ESI derecesi ve aciliyet durumunu metinden tespit et
    triage_level = "ESI-3: Urgent (Orta Derece Acil / 1 Saat Icinde)"
    urgency_score = 3
    is_emergency = False

    if any(k in reply_text for k in ["ESI Seviye 1", "ESI-1", "Resüsitasyon", "Hayati Tehlike", "Kardiyak"]):
        urgency_score = 1
        triage_level = "ESI-1: Resuscitation (Hayati Tehlike / Derhal Mudahale)"
        is_emergency = True
    elif any(k in reply_text for k in ["ESI Seviye 2", "ESI-2", "Acil Durum", "Akut Apandisit", "Akut Koroner", "Emboli", "İnme"]):
        urgency_score = 2
        triage_level = "ESI-2: Emergency (Acil Servis / 10-15 Dk)"
        is_emergency = True
    elif any(k in reply_text for k in ["ESI Seviye 4", "ESI-4", "Yarı Acil", "Poliklinik", "Sistit", "Öksürük", "Burkulma"]):
        urgency_score = 4
        triage_level = "ESI-4: Less Urgent (Yarı Acil / Poliklinik Başvurusu)"
        is_emergency = False
    elif any(k in reply_text for k in ["ESI Seviye 5", "ESI-5", "Rutin Takip", "Acil Değil"]):
        urgency_score = 5
        triage_level = "ESI-5: Non-Urgent (Acil Değil / Rutin Takip)"
        is_emergency = False

    # Yaniti Google Drive loglarina kaydet
    log_record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "session_id": req.session_id,
        "mode": req.mode,
        "user_message": req.messages[-1].content if req.messages else "",
        "assistant_reply": reply_text,
        "triage_level": triage_level,
        "urgency_score": urgency_score,
        "inference_time_ms": elapsed_ms,
    }
    append_to_drive_log(log_record)

    return {
        "session_id": req.session_id,
        "mode": req.mode,
        "reply": reply_text,
        "triage_level": triage_level,
        "urgency_score": urgency_score,
        "is_emergency": is_emergency,
        "red_flags": [],
        "inference_time_ms": elapsed_ms,
    }

# ==============================================================================
# TUNEL BASLATICI: Google Colab Gizli Anahtarlar (Secrets) veya Cloudflared
# ==============================================================================
NGROK_AUTH_TOKEN = ""

# 1. Oncelikle Google Colab Secrets (Sol paneldeki anahtar simgesi) kontrol edilir
try:
    from google.colab import userdata
    NGROK_AUTH_TOKEN = userdata.get('NGROK_AUTH_TOKEN')
    if NGROK_AUTH_TOKEN:
        print("[+] 'NGROK_AUTH_TOKEN' Google Colab Gizli Anahtarlarindan (Secrets) basariyla alindi.")
except Exception:
    pass

# 2. Eger Colab Secrets'tan alinamadiysa sistem ortam degiskenleri kontrol edilir
if not NGROK_AUTH_TOKEN:
    NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "")
    if NGROK_AUTH_TOKEN:
        print("[+] 'NGROK_AUTH_TOKEN' sistem ortam degiskenlerinden (os.getenv) basariyla alindi.")

# 3. Token varsa Ngrok tuneli acilir; yoksa ucretsiz Cloudflared tuneline gecilir
if NGROK_AUTH_TOKEN:
    from pyngrok import ngrok
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    public_url = ngrok.connect(8000).public_url
    print("\\n" + "=" * 75)
    print(f"  [NGROK AKTIF] BAGLANTI ADRESINIZ: {public_url}")
    print("  Bu URL'yi kopyalayip yerel Streamlit arayuzunuzdeki Sol Panele yapistirin!")
    print("=" * 75 + "\\n")
else:
    print("[*] Ngrok token bulunamadi, sifir ayar gerektiren Cloudflared HTTPS tuneli aciliyor...")
    !wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    !chmod +x cloudflared-linux-amd64
    tunnel_proc = subprocess.Popen(
        ["./cloudflared-linux-amd64", "tunnel", "--url", "http://127.0.0.1:8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in tunnel_proc.stdout:
        match = re.search(r"https://[a-zA-Z0-9-]+\\.trycloudflare\\.com", line)
        if match:
            public_url = match.group(0)
            print("\\n" + "=" * 75)
            print(f"  [CLOUDFLARE TUNNEL AKTIF] BAGLANTI ADRESINIZ: {public_url}")
            print("  Bu URL'yi kopyalayip yerel Streamlit arayuzunuzdeki Sol Panele yapistirin!")
            print("=" * 75 + "\\n")
            break

# ==============================================================================
# FASTAPI / UVICORN SUNUCUSUNU CALISTIRMA (Python 3.13 & Jupyter Event Loop Uyumlu)
# ==============================================================================
config = uvicorn.Config(app=colab_app, host="0.0.0.0", port=8000, log_level="info")
server = uvicorn.Server(config)

print("[+] FastAPI sunucusu 8000 portunda basariyla dinlemeye basladi!")
print("[*] Sunucu kesintisiz hizmet veriyor. Oturumu durdurmak icin sol ustteki Durdur (Stop) dugmesine basabilirsiniz.\\n")

try:
    await server.serve()
except (KeyboardInterrupt, asyncio.CancelledError):
    print("\\n[*] TriageCore AI sunucusu kullanici tarafindan durduruldu.")
'''

def update_notebook():
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Cell 11 Markdown guncelle
    nb["cells"][11]["source"] = [
        "### 6. Adım: Model Eğitimi ve Google Drive `/MyDrive/AI/TriageCore` Kaydı\n",
        "* **Bozuk ChatDoctor Şablon Spamı Temizlendi**: Eski bozuk veya tekrarlayan veriler temizlenir ve altın standart **ESI 1 - ESI 5 klinik triyaj veri seti** ile eğitilir.\n",
        "* **Hızlı Eğitim**: 3 Epochluk odaklı LoRA eğitimi T4 GPU'da yalnızca **~2-3 dakika** sürer.\n",
        "* Eğitim bittiğinde model ağırlıkları doğrudan Google Drive'daki **`AI/TriageCore/lora_adapter`** klasörüne kalıcı olarak kaydedilir.\n"
    ]

    # Cell 12 Code guncelle
    step6_code = build_dataset_code()
    nb["cells"][12]["source"] = [line + "\n" for line in step6_code.splitlines()]

    # Cell 13 Markdown guncelle
    nb["cells"][13]["source"] = [
        "---\n",
        "### 🚀 7. Adım (SON HÜCRE): Colab Uzak GPU Sunucusu (Ngrok / Cloudflared Endpoint)\n",
        "**Modeli kendi bilgisayarınızda çalıştırmanıza gerek yok!**\n",
        "Bu hücreyi çalıştırdığınızda:\n",
        "1. Eğitilmiş LoRA adaptörünü (`/MyDrive/AI/TriageCore/lora_adapter`) belleğe alır.\n",
        "2. **Tekrar Engelleme Mekanizması**: `repetition_penalty=1.15` ve `no_repeat_ngram_size=3` ile n-gram döngülerini engeller.\n",
        "3. Ngrok veya Cloudflared üzerinden genel bir `https://...` bağlantısı üretir.\n"
    ]

    # Cell 14 Code guncelle
    step7_code = build_step7_code()
    nb["cells"][14]["source"] = [line + "\n" for line in step7_code.splitlines()]

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print("[+] Colab Notebook basariyla guncellendi!")

if __name__ == "__main__":
    update_notebook()
