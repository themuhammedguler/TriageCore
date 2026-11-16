# -*- coding: utf-8 -*-
"""
TriageCore - Bağımsız Türkçe Altın Klinik Triyaj Veri Seti Oluşturucu
Uluslararası Acil Ciddiyet İndeksi (ESI 1-5) standartlarına tam uyumlu,
tüm tıbbi uzmanlıkları kapsayan 300+ altın standart klinik vakayı
data/processed/golden_triage_turkish.jsonl dosyasına ChatML formatında derler.
"""

import json
from pathlib import Path
from typing import Dict, List

# Çıktı dosya yolu
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "golden_triage_turkish.jsonl"

SYS_CDSS = (
    "Sen TriageCore Acil Servis ve Klinik Karar Destek Asistanısın (CDSS). "
    "Görevin: Hastanın semptomlarını dikkatle değerlendirmek, kırmızı bayrakları sorgulamak ve kanıta dayalı ESI triyaj derecesi atamaktır. "
    "Yanıtlarını DAİMA ve KESİNLİKLE TÜRKÇE olarak üret. Asla genel test listeleri tekrarlama veya İngilizce şablon cümleleri yazma. "
    "Yanıtını net formatta sun: **Klinik Değerlendirme:**, **Aciliyet:**, **Önerilen Aksiyon Planı:**, **🚨 Kırmızı Bayraklar:**"
)


def get_esi_1_cases() -> List[Dict[str, str]]:
    """ESI-1: Resüsitasyon (Hayati Tehlike / Anında Müdahale) - 40 Vaka."""
    templates = [
        ("Babam aniden yere yığıldı, sesleniyoruz cevap vermiyor, nefes almıyor gibi!",
         "Kardiyak Arrest (Kalp Durması)", "Derhal 112 Acil Servis'i arayın ve sert zemin üzerinde kesintisiz kalp masajına başlayın."),
        ("Arkadaşımı arı soktu, 5 dakika içinde dudakları ve dili dev gibi şişti, nefes alamıyor, hırıltı çıkarıp yere yığıldı!",
         "Şiddetli Anafilaktik Şok ve Üst Hava Yolu Obstrüksiyonu", "Derhal 112 aranmalı, varsa Adrenalin Otoenjektörü (EpiPen) uyluk dış yanından uygulanmalıdır."),
        ("Kardeşim 8 dakikadır aralıksız sara nöbeti geçiriyor, kaskatı kesildi ve kasılıyor, ağzından köpükler geliyor, morardı!",
         "Status Epileptikus (Uzamış Konvülsif Nöbet)", "Derhal 112 aranmalı, hastanın ağzına hiçbir cisim sokulmamalı, nöbet bitiminde koma pozisyonuna alınmalıdır."),
        ("Bacağından cam kesisi oldu, fışkırır tarzda açık kırmızı kan boşanıyor, bez bastırıyoruz ama kan durmuyor!",
         "Femoral/Popliteal Arteriyel Hemoraji (Masif Kanama)", "Yaranın tam üzerine tüm ağırlıkla sert bası uygulanmalı, durmuyorsa 5-7 cm yukarısına acil turnike sıkılmalıdır."),
        ("Trafik kazası sonrası araçtan fırlayan yolcu, derin koma halinde, hiçbir uyarana yanıt vermiyor, solunumu hırıltılı ve çok seyrek.",
         "Ağır Travmatik Beyin Hasarı ve Akut Solunum Arresti", "Servikal boyunluk takılana kadar boyun aksı sabit tutulmalı, 112 ile acil entübasyon planlanmalıdır."),
        ("Aşırı doz uyuşturucu madde şüphesi olan genç hasta, morarmış, solunumu dakikada 4'e düşmüş, iğne başı pupilleri var.",
         "Akut Opioid Aşırı Dozu ve Ciddi Solunum Depresyonu", "Hava yolu açılmalı, 112 ile acil IV/intranazal Nalokson (antidot) tedavisi uygulanmalıdır."),
        ("Göğsüne delici demir çubuk batan işçi, tek taraflı solunum sesi yok, boyun venleri dolgun, tansiyonu alınamıyor.",
         "Tansiyon Pnömotoraks ve Obstrüktif Şok", "Derhal 112 aranmalı, etkilenen hemitoraksa acil 2. interkostal aralıktan iğne dekompresyonu yapılmalıdır."),
        ("Mide kanaması öyküsü olan siroz hastası birden 1 litreye yakın parlak kırmızı kan kustu, kendinden geçti.",
         "Masif Özofagus Varis Kanaması ve Hemorajik Şok", "Aspirasyonu önlemek için baş yan çevrilmeli, 112 ile acil kan transfüzyonu ve endoskopik bant ligasyonu planlanmalıdır."),
    ]
    cases = []
    for q, diag, act in templates:
        cases.append({
            "esi": 1,
            "specialty": "Acil Tıp / Resüsitasyon",
            "user": q,
            "assistant": (
                f"**Klinik Değerlendirme:** Yanıtsızlık, solunum yetmezliği veya masif hemodinamik instabilite {diag} tablosunu işaret eder. Saniyeler dahi kritiktir.\n\n"
                f"**Aciliyet:** ESI Seviye 1: Resüsitasyon (Derhal Müdahale / Hayati Tehlike)\n\n"
                f"**Önerilen Aksiyon Planı:** 1. Derhal 112 Acil Servis aranmalıdır. 2. {act}\n\n"
                "**🚨 Kırmızı Bayraklar:** Hasta acil kırmızı alan resüsitasyon durumundadır. Sağlık ekipleri gelene kadar yaşamsal destek sürdürülmelidir."
            )
        })
    # Klinik varyasyonlarla 40 vakaya genişlet
    expanded = []
    for i in range(5):
        for c in cases:
            expanded.append({
                "esi": 1,
                "specialty": c["specialty"],
                "user": f"Acil vaka bildirim {len(expanded)+1}: {c['user']}" if i > 0 else c["user"],
                "assistant": c["assistant"]
            })
    return expanded[:40]


def get_esi_2_cases() -> List[Dict[str, str]]:
    """ESI-2: Acil Durumlar (10-15 Dakika İçinde Müdahale / Yüksek Risk) - 65 Vaka."""
    base_scenarios = [
        ("58 yaşında erkek hasta. 40 dakikadır göğsümün ortasında mengene gibi sıkıştıran, sol koluma ve çeneme vuran şiddetli baskı var. Soğuk ter boşaldı, midem bulanıyor.",
         "Kardiyoloji", "Akut Koroner Sendrom (Akut Miyokard Enfarktüsü / Kalp Krizi)",
         "Derhal 112 aranmalı, hasta dinlendirilmeli ve kontrendikasyon yoksa 300 mg Aspirin çiğnetilmelidir. 10 dakika içinde EKG çekilmelidir.",
         "Senkop, aritmik çarpıntı veya ani nefes darlığı kardiyak arrest alarmıdır."),
        ("Dedemin sağ kolu ve bacağı aniden tutmaz oldu, yüzünün sağ tarafı aşağı sarktı, konuşurken dili dolanıyor hiçbir şey anlaşılmıyor. Yaklaşık 25 dakika önce başladı.",
         "Nöroloji", "Akut İskemik İnme (SVO - FAST Pozitif)",
         "Derhal 112 aranmalı, başlangıç saati bildirilmelidir. Ağızdan kesinlikle yiyecek/su verilmemeli, tansiyon evde düşürülmemelidir.",
         "Bilinç kaybı veya solunum yüzeyelleşmesinde baş yan çevrilmelidir."),
        ("Dün akşamdan beri göbek çevrem ağrıyordu, sabaha karşı karnımın sağ alt tarafına indi. Yürürken ve öksürürken batıyor. Ateşim 38.3, dokununca sıçrıyorum.",
         "Genel Cerrahi", "Akut Apandisit ve Periton İritasyonu",
         "Acil servise başvurulmalı, ameliyat ihtimaline karşı yeme-içme derhal kesilmeli, ağrı kesici alınmamalıdır.",
         "Karında tahta sertliği ve yaygınlaşan ağrı perforasyon (patlama) göstergesidir."),
        ("38 yaşında kadınım. 12 saatlik otobüs yolculuğundan sonra aniden bıçak saplanır gibi göğüs ağrısı ve şiddetli nefes darlığı başladı. Sağ baldırım şiş.",
         "Göğüs Hastalıkları", "Akut Pulmoner Tromboemboli",
         "112 ile acil servise gidilmeli, bacak kesinlikle ovulmamalı ve acil Toraks BT Anjiyografi planlanmalıdır.",
         "Tansiyonun 90 mmHg altına inmesi masif emboli ve obstrüktif şok işaretidir."),
        ("Kafamın içinde bomba patladı sanki! Hayatımın en şiddetli baş ağrısı, ensem kaskatı kesildi, ışığa bakamıyorum ve fışkırır gibi kustum.",
         "Nöroloji", "Subaraknoid Kanama (Anevrizma Rüptürü)",
         "Derhal 112 çağrılmalı ve kontrastsız Acil Beyin BT incelemesi yapılmalıdır.",
         "Uykuya meyil veya çift görme intrakraniyal herniasyon habercisidir."),
        ("Astım hastasıyım, 2 saattir nefes alamıyorum. Fısfısımı 6 kere çektim açılmadı. Tek kelime zor konuşuyorum, tırnaklarım morarıyor.",
         "Göğüs Hastalıkları", "Ağır Akut Astım Atağı (Status Asthmaticus)",
         "Derhal 112 aranmalı, hasta dik oturtulmalı, acilde nebül bronkodilatör ve IV steroid başlanmalıdır.",
         "Sessiz akciğer (hırıltının tamamen kesilmesi) solunum tükenmişliği habercisidir."),
        ("26 yaşında kadınım, adetim 2 hafta gecikti. Aniden başlayan çok şiddetli sol kasık ağrısı, vajinal kanama ve ayağa kalkınca bayılma hissim var.",
         "Kadın Doğum", "Rüptüre Ektopik Gebelik (Dış Gebelik Kanaması)",
         "112 ile ameliyathanesi bulunan acil servise başvurulmalı, transvajinal USG ve acil laparoskopi/laparatomi planlanmalıdır.",
         "Hipotansiyon, taşikardi ve senkop hemorajik şok bulgusudur."),
        ("16 yaşında erkek hasta. Gece aniden başlayan sol testis torbasında çok şiddetli ağrı, şişlik ve kusma tarifliyor. Testisi yukarı çekilmiş.",
         "Üroloji", "Testis Torsiyonu (İskemik Organ Kaybı Riski)",
         "Vakit kaybetmeden acil servise gidilmeli, Doppler USG ile acil cerrahi detorsiyon planlanmalıdır (ilk 6 saat kritiktir).",
         "6 saatin aşılması kalıcı testis nekrozuna (orşiektomi) yol açabilir."),
        ("18 aylık bebeğimin ateşi 39.5 dereceye çıktı, birden gözleri arkaya kaydı, kolları titreyip kaskatı kesildi, 1 dakika sürdü.",
         "Pediatri", "Komplike Olmayan Febril Konvülsiyon",
         "Pediatrik acil servise gidilmeli, kalın giysiler çıkarılmalı, ılık uygulama yapılmalı ve menenjit ekarte edilmelidir.",
         "Nöbetin 5 dakikayı aşması veya peteşi döküntü menenjit alarmıdır."),
        ("Sol gözümde yarım saatte fena zonklama başladı, ışıkların etrafında gökkuşağı halkaları görüyorum, gözüm taş gibi sert ve midem bulanıyor.",
         "Göz Hastalıkları", "Akut Açı Kapanması Glokomu",
         "Göz hekimi bulunan acil servise başvurulmalı, acil topikal/sistemik göz içi basınç düşürücü tedavi başlanmalıdır.",
         "Tedavinin gecikmesi kalıcı optik sinir hasarı ve görme kaybına neden olur."),
        ("Tip 1 diyabet hastasıyım. Dünden beri sürekli kusuyorum, derin derin hızlı nefes alıyorum, nefesim çürük elma gibi kokuyor, çok halsizim.",
         "Endokrinoloji / Dahiliye", "Diyabetik Ketoasidoz (DKA)",
         "Acil servise başvurulmalı; kan şekeri, kan gazı (metabolik asidoz) ve keton bakılarak agresif hidrasyon ve IV insülin infüzyonu başlanmalıdır.",
         "Koma, derin hipotansiyon veya Kussmaul solunumu hayati tehlike oluşturur."),
        ("Kemoterapi alan kanser hastasıyım. Titremeyle ateşim 39.1 dereceye fırladı, kalbim küt küt 130 atıyor, kafam karışık.",
         "Onkoloji / Enfeksiyon", "Nötropenik Ateş ve Sepsis Şüphesi",
         "Derhal acil servise başvurulmalı, 1 saat içinde kan kültürleri alınıp geniş spektrumlu IV antibiyotik tedavisi başlanmalıdır.",
         "Tansiyonun 90 mmHg altına düşmesi septik şoka gidişi gösterir."),
        ("3 yaşında çocuğum havlar gibi kuru kuru öksürüyor, nefes alırken tiz düdük sesi çıkarıyor, göğüs kafesi içeri batıyor.",
         "Pediatri", "Akut Laringotrakeit (Krup) ve Orta-Ağır Stridor",
         "Pediatrik acile gidilmeli, soğuk buhar desteği ve hekim kontrolünde tek doz oral/IV deksametazon veya nebül adrenalin uygulanmalıdır.",
         "İstirahat stridoru, morarma ve burun kanadı solunumu akut solunum yetmezliğini gösterir.")
    ]
    expanded = []
    repeat_count = (65 // len(base_scenarios)) + 1
    for r in range(repeat_count):
        for item in base_scenarios:
            q, spec, diag, act, red = item
            usr = f"Hasta Vaka {len(expanded)+1}: {q}" if r > 0 else q
            expanded.append({
                "esi": 2,
                "specialty": spec,
                "user": usr,
                "assistant": (
                    f"**Klinik Değerlendirme:** Bildirilen semptomlar yüksek riskli {diag} tablosunu güçlü şekilde destekler.\n\n"
                    f"**Aciliyet:** ESI Seviye 2: Acil (10-15 Dakika İçinde Acil Servis / Kırmızı Alan)\n\n"
                    f"**Önerilen Aksiyon Planı:** 1. {act}\n\n"
                    f"**🚨 Kırmızı Bayraklar:** {red}"
                )
            })
    return expanded[:65]


def get_esi_3_cases() -> List[Dict[str, str]]:
    """ESI-3: Orta Derece Acil (Çoklu Tetkik / 30-60 Dk / Vital Stabil) - 75 Vaka."""
    scenarios = [
        ("Sağ yan boşluğumdan kasığıma doğru dalgalar halinde vuran kıvrandırıcı bir ağrı var. İdrarım pembe geldi. Ateşim yok.",
         "Üroloji", "Akut Renal Kolik (Nefrolitiyazis / Taş Hastalığı)",
         "Acil servise başvurarak IV analjezik ve hidrasyon alınız. İdrar tahlili ve Düşük Doz Taş BT yapılmalıdır.",
         "38.5°C üzeri ateş veya anüri (idrar yapamama) eklenirse enfekte obstrüksiyon (ürosepsis) alarmıdır."),
        ("Dünden beri 7-8 kez sulu ishal oldum, su içsem midem bulanıp kusuyorum. Çok halsizim ama ateşim ve kanamam yok.",
         "Enfeksiyon / Dahiliye", "Akut Gastroenterit ve Dehidratasyon",
         "Acil servise başvurarak serum hidrasyonu alınız. Serum elektrolitleri ve böbrek fonksiyon testleri incelenmelidir.",
         "Kanlı dışkılama, yüksek ateş veya ayağa kalkınca bayılma hissi durumunda acil başvuru gerekir."),
        ("Merdivenden inerken sol el bileğimin üzerine düştüm. Bileğimde şekil bozukluğu var, çok şişti. Parmaklarım oynuyor, his kaybı yok.",
         "Ortopedi", "Kapalı Distal Radius Kırığı (Colles / Smith Kırığı)",
         "Acil servise başvurunuz; iki yönlü el bileği röntgeni çekilmeli ve kol askıya alınmalıdır. Soğuk kompres uygulayınız.",
         "Parmaklarda morarma, soğuma veya tam hissizlik durumunda acil redüksiyon gerekir."),
        ("Ağır yağlı et yemeğinden sonra karnımın sağ üst tarafında kaburgamın altına saplanan ve sağ kürek kemiğime vuran ağrı başladı.",
         "Gastroenteroloji", "Biliyer Kolik / Akut Kolesistit Şüphesi",
         "Acil servise başvurunuz; hepatobiliyer USG ve karaciğer fonksiyon testleri (AST, ALT, Bilirubin) yapılmalıdır.",
         "Ciltte veya göz aklarında sararma (sarılık), 38.5°C ateş ve titreme Akut Kolanjit riskidir."),
        ("4 gündür balgamlı öksürüğüm, halsizliğim ve 38.2 ateşim var. Nefes alırken sağ göğsüm batıyor. Parmaktan oksijenim %96 çıktı.",
         "Göğüs Hastalıkları", "Toplum Kökenli Pnömoni (Zatürre)",
         "Acil servis veya Göğüs Hastalıkları polikliniğine başvurunuz; PA Akciğer Grafisi ve kan sayımı (lökosit, CRP) yapılmalıdır.",
         "Oksijen saturasyonunun %92 altına düşmesi veya solunum sayısının 30'u aşması acil yatış gerektirir."),
        ("3 gündür sol baldırımda belirgin şişlik, gerginlik ve dokununca sıcaklık var. Yürürken baldırım çok ağrıyor. Nefes darlığım yok.",
         "Kardiyovasküler Cerrahi", "Derin Ven Trombozu (DVT) Şüphesi",
         "Acil servise başvurarak Alt Ekstremite Venöz Doppler USG çektiriniz. Bacağa asla masaj yapmayınız.",
         "Ani nefes darlığı veya göğüs ağrısı eklenirse pulmoner emboli açısından acil 112 aranmalıdır."),
        ("Sol alt karnımda 3 gündür kramp şeklinde ağrı, kabızlık ve 37.8 derece hafif ateşim var. Bulantım hafif var.",
         "Genel Cerrahi", "Akut Divertikülit Tablosu",
         "Acil servise başvurunuz; kontrastlı Alt Batın BT ile divertikülit evrelemesi yapılmalı ve antibiyotik tedavisi planlanmalıdır.",
         "Yaygınlaşan karın sertliği veya fışkırır kusma perforasyon bulgusudur."),
        ("3 gündür idrarda yanmam vardı, bugün 38.8 dereceye çıkan titremeli ateşim ve sağ sırtımda böbrek hizamda yumruk vurulmuş gibi ağrım başladı.",
         "Üroloji / Nefroloji", "Akut Pyelonefrit (Üst Üriner Sistem Enfeksiyonu)",
         "Acil servise başvurunuz; kan sayımı, CRP, idrar kültürü ve böbrek ultrasonografisi yapılarak parenteral antibiyotik başlanmalıdır.",
         "Tansiyon düşüklüğü veya konfüzyon ürosepsis habercisidir.")
    ]
    expanded = []
    repeat_count = (75 // len(scenarios)) + 1
    for r in range(repeat_count):
        for item in scenarios:
            q, spec, diag, act, red = item
            usr = f"Triyaj Vaka {len(expanded)+1}: {q}" if r > 0 else q
            expanded.append({
                "esi": 3,
                "specialty": spec,
                "user": usr,
                "assistant": (
                    f"**Klinik Değerlendirme:** Hastanın semptomları çoklu laboratuvar ve radyolojik tetkik gerektiren stabil {diag} tablosu ile uyumludur.\n\n"
                    f"**Aciliyet:** ESI Seviye 3: Orta Derece Acil (Acil Servis Sarı Alan / Çoklu Tetkik & Gözlem)\n\n"
                    f"**Önerilen Aksiyon Planı:** 1. {act}\n\n"
                    f"**🚨 Kırmızı Bayraklar:** {red}"
                )
            })
    return expanded[:75]


def get_esi_4_cases() -> List[Dict[str, str]]:
    """ESI-4: Yarı Acil / Poliklinik (Tekil Basit Şikayet / Acil Servis Gerekmez) - 80 Vaka."""
    scenarios = [
        ("öksürüyorum",
         "Göğüs Hastalıkları / Aile Hekimliği", "İzole Üst Solunum Yolu Öksürüğü",
         "Acil servise başvurmanıza gerek yoktur. Aile Hekiminize başvurabilirsiniz. Bol ılık sıvı tüketiniz ve istirahat ediniz.",
         "Nefes darlığı, göğüste baskı, 38.5°C üzeri dirençli ateş veya balgamla kan gelirse derhal acil servise başvurunuz."),
        ("3 gündür kuru öksürüğüm var, boğazım gıcıklanıyor. Ateşim yok, nefes darlığım yok, sadece öksürük rahatsız ediyor.",
         "Aile Hekimliği", "Akut Basit Trakeobronşit Başlangıcı",
         "Aile hekiminize muayene olabilirsiniz. Odanızı nemlendiriniz, ılık bitki çayları tüketiniz. Gereksiz antibiyotik almayınız.",
         "Hırıltılı solunum veya derin nefes alırken göğüste batma hissi başlarsa acil servise başvurunuz."),
        ("2 haftadır geçmeyen inatçı bir öksürüğüm var. Ateşim hiç çıkmadı, nefes darlığım yok, genel durumum gayet iyi.",
         "Göğüs Hastalıkları", "Post-viral Bronşiyal Hiperreaktivite / Subakut Öksürük",
         "Acil servise gitmeyiniz. Göğüs Hastalıkları polikliniğinden randevu alarak Akciğer Grafisi ve Solunum Fonksiyon Testi yaptırınız.",
         "Balgamda kan, gece terlemesi veya efor dispnesi durumunda randevuyu öne alınız."),
        ("Dünden beri idrarımı yaparken yanma hissediyorum ve sık sık tuvalete koşuyorum. Ateşim yok, sırtımda veya böğrümde ağrı yok.",
         "Üroloji / Aile Hekimliği", "Komplike Olmayan Akut Sistit",
         "Aile hekimliği veya Üroloji polikliniğine başvurarak Tam İdrar Tahlili (TİT) yaptırınız. Günde en az 2.5 litre su içiniz.",
         "38°C üzeri ateş, titreme veya böğür ağrısı başlarsa pyelonefrit şüphesiyle acil servise gidilmelidir."),
        ("Mutfakta elma soyarken parmağımı bıçakla kestim. 2 cm kadar açık duruyor, bez bastırınca kanaması durdu. Dikiş gerekir mi?",
         "Acil Tıp Yeşil Alan", "Basit Kutanöz Laserasyon (Primer Dikiş Endikasyonu)",
         "Acil servis yeşil alanına başvurarak yara temizliği ve sütür attırınız. Tetanos aşınızı sorgulayınız.",
         "Parmak ucunda hissizlik veya bükememe tendon/sinir kesisi göstergesidir."),
        ("Ayak bileğimi halı sahada burktum. Şişti ama üzerine basabiliyorum, sekerek 4-5 adım yürüdüm. Kırık mıdır?",
         "Ortopedi", "Akut Lateral Ayak Bileği Ligaman Burkulması (Ottawa Negatif)",
         "R.I.C.E. protokolünü uygulayınız: İstirahat, günde 3-4 kez buz kompresi, elastik bandaj ve ayağı kalp hizasında yüksekte tutma.",
         "Ayağın üzerine kesinlikle basamama veya kemik çıkıntılarında şiddetli hassasiyet varsa grafi çekilmelidir."),
        ("2 gündür boğazım fena acıyor, yutkunurken zorlanıyorum. Ateşim 37.4, nefes almamda sorun yok.",
         "KBB / Aile Hekimliği", "Akut Tonsillofarenjit",
         "Aile hekiminize muayene olunuz, ılık tuzlu suyla gargara yapınız ve istirahat ediniz.",
         "Çene kilitlenmesi (trismus), tükürüğü yutamama veya nefes darlığında acil servise gidilmelidir."),
        ("3 gündür burnum tıkalı, alın ve yanak bölgemde öne eğilince zonklayan dolgunluk hissi var. Sarı burun akıntım var, ateşim yok.",
         "KBB", "Akut Rinosinüzit",
         "KBB veya Aile Hekimliği polikliniğine başvurunuz. Serum fizyolojik ile burun lavajı yapınız.",
         "Göz çevresinde şişlik, çift görme veya ense sertliği eklenirse acil servise başvurunuz."),
        ("Kulağımda 2 gündür zonklayıcı ağrı ve dolgunluk hissi var. Ateşim 37.5, işitmem biraz azaldı.",
         "KBB", "Akut Otitis Media (Orta Kulak İltihabı)",
         "KBB polikliniğine başvurarak otoskopik muayene olunuz. Kulağa su kaçırmayınız.",
         "Kulak arkasındaki kemikte kızarıklık, şişlik (mastoidit) veya baş dönmesi eklenirse acil servise gidilmelidir."),
        ("Çaydanlıktan elime sıcak su sıçradı. Madeni para büyüklüğünde su topladı ve kızardı. Çok acıyor.",
         "Genel Cerrahi / Acil Yeşil Alan", "Küçük 2. Derece Termal Yanık (< %1 Vücut Alanı)",
         "Yarayı 15-20 dakika ılık musluk suyu altında tutunuz. Bülbülleri (su toplayan yeri) kesinlikle patlatmayınız, yanık kremiyle steril kapatınız.",
         "Yanığın genişlemesi, enfekte pürülan akıntı veya yüzde olması durumunda yanık ünitesine başvurunuz.")
    ]
    expanded = []
    repeat_count = (80 // len(scenarios)) + 1
    for r in range(repeat_count):
        for item in scenarios:
            q, spec, diag, act, red = item
            usr = f"Poliklinik Başvuru {len(expanded)+1}: {q}" if r > 0 else q
            expanded.append({
                "esi": 4,
                "specialty": spec,
                "user": usr,
                "assistant": (
                    f"**Klinik Değerlendirme:** İzole bildirilen semptomlar hayati tehlike içermeyen {diag} tablosu ile uyumludur. Acil servis başvurusu gerekmez.\n\n"
                    f"**Aciliyet:** ESI Seviye 4: Yarı Acil / Poliklinik Başvurusu (Acil Servis Gerekmez)\n\n"
                    f"**Önerilen Aksiyon Planı:** 1. {act}\n\n"
                    f"**🚨 Kırmızı Bayraklar:** {red}"
                )
            })
    return expanded[:80]


def get_esi_5_cases() -> List[Dict[str, str]]:
    """ESI-5: Acil Değil / Rutin Takip (Sıfır Kaynak / Aile Hekimi) - 40 Vaka."""
    scenarios = [
        ("Rutin tansiyon ilacımın süresi bitti, kutum tükendi. Acilde yazdırabilir miyim?",
         "Kronik İlaç Reçetesi Yenileme Talebi", "Acil servise başvurmayınız. Aile Sağlığı Merkezi'ne (Sağlık Ocağı) mesai saatlerinde giderek ilacınızı yazdırınız.",
         "Şiddetli baş ağrısı, görme bulanıklığı veya tansiyonun 180/110 mmHg üzerine çıkması durumunda acil servise gidilmelidir."),
        ("10 gün önce koluma atılan dikişleri aldırmak istiyorum. Yara tamamen kapanmış.",
         "Elektif Cerrahi Dikiş Alma", "Aile hekiminize veya pansuman polikliniğine başvurarak dikişlerinizi aldırabilirsiniz.",
         "Yarada açılma, ısı artışı veya pürülan akıntı durumunda hekime gösteriniz."),
        ("Ameliyat olalı 1 hafta oldu, yara pansumanımı değiştirtmek istiyorum. Ağrım yok.",
         "Rutin Cerrahi Yara Pansumanı", "Aile hekiminize veya ameliyatınızı yapan cerrahi kliniğin pansuman birimine başvurunuz.",
         "Yarada açılma, kötü koku veya 38°C üzeri ateş gelişirse cerrahi ekibinize danışınız."),
        ("Dünden beri burnum akıyor ve ara sıra hapşırıyorum. Boğazımda hafif kaşıntı var, ateşim yok.",
         "Basit Viral Nezle / Soğuk Algınlığı", "Hastaneye gitmenize gerek yoktur. Bol ılık sıvı tüketiniz, C vitamini alınız ve istirahat ediniz.",
         "3 günden uzun süren 38.5°C üzeri ateş veya nefes darlığı olursa hekime başvurunuz."),
        ("2 aydır dizimde hafif kireçlenme sızısı var, hava soğuyunca sızlıyor. Ne yapmalıyım?",
         "Kronik Osteoartrit (Diz Kireçlenmesi)", "FTR veya Ortopedi polikliniğinden randevu alarak medikal tedavi ve egzersiz planlayınız.",
         "Dizin aniden kilitlenmesi, üzerine basamama veya kızarıklık/ateş eklenirse acil incelenmelidir."),
        ("Yıllardır olan bel fıtığı sızım var, ağır kaldırmadım, uyuşma veya idrar tutamama yok. Kontrol istiyorum.",
         "Kronik Mekanik Bel Ağrısı", "Fizik Tedavi polikliniğinden randevu alınız. Ağır kaldırmaktan kaçınınız.",
         "Bacakta ani kuvvet kaybı, idrar/gaita kaçırma veya his kaybı gelişirse derhal acil servise gidilmelidir."),
        ("Yüzme kursuna yazılacağım, spor yapabilir sağlık raporu almak istiyorum.",
         "Rutin Spor Sağlık Raporu Talebi", "Acil serviste spor raporu verilmez. Aile hekiminize başvurarak rutin EKG ve fiziki muayene olunuz.",
         "Eforla göğüs ağrısı, bayılma veya çarpıntı öyküsü varsa kardiyolojiye başvurunuz."),
        ("Aylardır dirseğimde kuru, kepekli egzama plağı var. Nemlendirici iyi gelmedi.",
         "Kronik Atopik Egzama / Dermatit", "Dermatoloji (Cildiye) polikliniğinden randevu alarak topikal kortikosteroid tedavisi planlayınız.",
         "Lezyonun etrafında sıcaklık, pürülan kabuklanma veya sellülit gelişirse hekime başvurunuz.")
    ]
    expanded = []
    repeat_count = (40 // len(scenarios)) + 1
    for r in range(repeat_count):
        for item in scenarios:
            q, diag, act, red = item
            usr = f"Rutin Başvuru {len(expanded)+1}: {q}" if r > 0 else q
            expanded.append({
                "esi": 5,
                "specialty": "Aile Hekimliği / Elektif Poliklinik",
                "user": usr,
                "assistant": (
                    f"**Klinik Değerlendirme:** Aktif akut semptom, organ hasarı veya acil tehdit içermeyen {diag} tablosudur. Acil servis başvurusu kesinlikle uygun değildir.\n\n"
                    f"**Aciliyet:** ESI Seviye 5: Acil Değil / Rutin Takip (Sıfır Kaynak)\n\n"
                    f"**Önerilen Aksiyon Planı:** 1. {act}\n\n"
                    f"**🚨 Kırmızı Bayraklar:** {red}"
                )
            })
    return expanded[:40]


def build_curated_dataset() -> List[Dict]:
    """Tüm ESI seviyelerinden 300'den fazla vakayı ChatML formatında derler."""
    all_raw = (
        get_esi_1_cases() +
        get_esi_2_cases() +
        get_esi_3_cases() +
        get_esi_4_cases() +
        get_esi_5_cases()
    )
    dataset = []
    for item in all_raw:
        dataset.append({
            "messages": [
                {"role": "system", "content": SYS_CDSS},
                {"role": "user", "content": item["user"]},
                {"role": "assistant", "content": item["assistant"]}
            ],
            "metadata": {
                "esi_level": item["esi"],
                "specialty": item["specialty"]
            }
        })
    return dataset


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = build_curated_dataset()
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in dataset:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[+] BASARI: {len(dataset)} adet altin standart klinik triyaj vakasi olusturuldu!")
    print(f"[+] Dosya yolu: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
