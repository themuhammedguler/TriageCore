# ==============================================================================
# TriageCore - Doktor Simülasyonu (OSCE) Klinik Vaka Bankası ve Hasta Motoru
# ==============================================================================
# Bu modul, kullanicinin 'Doktor' rolune gectigi, sistemin ise 'Hasta' oldugu
# interaktif vaka cozme modunu yonetir. Doktorun sordugu anamnez sorularina,
# fizik muayene ve tetkik isteklerine gore gercekci yanitlar uretir ve vaka
# sonunda doktorun koydugu teshisi puanlar.
# ==============================================================================

# Tip ipuclari icin Any, Dict, List ve Optional yuklenir
from typing import Any, Dict, List, Optional

# Gizli tanilari, anamnez detaylarini ve muayene bulgularini iceren klinik vaka havuzu
CLINICAL_CASES: Dict[str, Dict[str, Any]] = {
    "CASE-001": {
        "case_id": "CASE-001",
        "title": "Vaka #01: 24 Yaş Erkek — Karın Ağrısı",
        "patient_name": "Ahmet Y.",
        "age": 24,
        "gender": "Erkek",
        "difficulty": "Orta",
        "initial_complaint": (
            "Doktor bey/hanım, dün geceden beri karnım ağrıyor. İlk başta göbek deliğimin "
            "etrafında hafif bir sızı gibiydi ama bu sabah sağ alt tarafıma indi ve yürürken bile batıyor."
        ),
        "pain_details": (
            "Ağrı dün gece göbek çevresinde künt bir şekilde başladı, son 6 saattir karnımın "
            "sağ alt kısmına (iliak çukura) yerleşti. Öksürünce ve hareket edince şiddetleniyor, 10 üzerinden 8 diyebilirim."
        ),
        "associated_symptoms": (
            "Sabah bir kez istifra ettim (kustum). Hiç iştahım yok, en sevdiğim yemeği bile görmek istemiyorum. "
            "Hafif bir üşüme ve sıcak basması hissediyorum. İshal veya idrarda yanmam yok."
        ),
        "history": (
            "Bilinen kronik bir hastalığım yok. Düzenli kullandığım ilaç yok. Daha önce hiç ameliyat olmadım. "
            "İlaç alerjim yok."
        ),
        "physical_exam": (
            "[FİZİK MUAYENE BULGULARI]: Ateş: 38.2°C, Nabız: 96/dk, TA: 120/75 mmHg. "
            "Batın muayenesinde sağ alt kadranda (McBurney noktasında) belirgin defans ve rebound (+). "
            "Rovsing bulgusu (+), Psoas testi (+)."
        ),
        "lab_and_imaging": (
            "[LABORATUVAR & GÖRÜNTÜLEME]: WBC (Lökosit): 15.800 /mm³ (Nötrofil hakimiyeti %84), CRP: 48 mg/L. "
            "Tam İdrar Tetkiki: Normal. Batın USG: Sağ alt kadranda 9 mm çapında, kompresyona yanıtsız, "
            "duvarı ödemli aperistaltik tübüler yapı (apendiks) izlendi."
        ),
        "correct_diagnosis": "Akut Apandisit",
        "accepted_keywords": ["apandisit", "appendicitis", "akut apandisit", "apendiks"],
        "expected_esi": 2,
        "clinical_pearls": (
            "Göbek çevresinden (visseral ağrı) başlayıp sağ alt kadrana (pariyetal periton irritasyonu) göç eden ağrı, "
            "anoreksi (iştahsızlık), subfebril ateş ve McBurney noktasında rebound (+) bulgusu Akut Apandisit için "
            "klasik Alvarado kriterleridir. Genel Cerrahi konsültasyonu ve acil apendektomi gerektirir."
        ),
    },
    "CASE-002": {
        "case_id": "CASE-002",
        "title": "Vaka #02: 38 Yaş Kadın — Ani Nefes Darlığı ve Göğüs Ağrısı",
        "patient_name": "Selin K.",
        "age": 38,
        "gender": "Kadın",
        "difficulty": "Zor",
        "initial_complaint": (
            "Doktor, yaklaşık 2 saat önce aniden nefesim daralmaya başladı. Derin nefes aldığımda "
            "sağ göğsüme bıçak saplanır gibi bir ağrı giriyor, kalbim çok hızlı çarpıyor."
        ),
        "pain_details": (
            "Göğüs ağrım sol koluma veya çeneme vurmuyor; tam sağ göğüs kafesimde, özellikle nefes alıp verirken "
            "veya öksürürken batan (plöritik vasıflı) keskin bir ağrı."
        ),
        "associated_symptoms": (
            "Ateşim yok, balgamlı öksürüğüm yok ama dün 11 saatlik uzun bir uçak yolculuğundan döndüm. "
            "Dünden beri sağ baldırımda da garip bir şişlik ve ağrı var."
        ),
        "history": (
            "Kronik hastalığım yok ama 2 yıldır doğum kontrol hapı (oral kontraseptif) kullanıyorum. "
            "Dün Tokyo'dan İstanbul'a 11 saat kesintisiz uçuş yaptım."
        ),
        "physical_exam": (
            "[FİZİK MUAYENE BULGULARI]: TA: 105/68 mmHg, Nabız: 122/dk (Taşikardik), Solunum Sayısı: 28/dk (Takipneik), "
            "Oda havasında SpO2: %89. Akciğer oskültasyonunda ral/ronküs yok. Sağ kruris (baldır) çapı sola göre "
            "3 cm daha geniş, ödemli ve Homans bulgusu (+)."
        ),
        "lab_and_imaging": (
            "[LABORATUVAR & GÖRÜNTÜLEME]: D-Dimer: 4200 ng/mL (Çok yüksek). EKG: Sinüs taşikardisi ve S1Q3T3 paterni. "
            "Toraks BT Anjiyografi: Sağ ana pulmoner arterde lümeni tama yakın tıkayan trombüs (dolum defekti) izlendi."
        ),
        "correct_diagnosis": "Akut Pulmoner Emboli",
        "accepted_keywords": ["pulmoner emboli", "emboli", "akciğer embolisi", "pulmonary embolism", "pe", "dvt"],
        "expected_esi": 1,
        "clinical_pearls": (
            "Uzun süreli immobilizasyon (uçak yolculuğu) + Oral Kontraseptif kullanımı + Tek taraflı baldır ödemi (DVT) "
            "zemininde gelişen ani plöritik göğüs ağrısı, hipoksi ve taşikardi (Wells Skoru yüksek) Akut Pulmoner Emboli "
            "tablosudur. Hayati tehlike (ESI-1/2) taşır; derhal antikoagülasyon ve BT Pulmoner Anjiyografi gerektirir."
        ),
    },
    "CASE-003": {
        "case_id": "CASE-003",
        "title": "Vaka #03: 19 Yaş Erkek — Şiddetli Baş Ağrısı ve Yüksek Ateş",
        "patient_name": "Mert T.",
        "age": 19,
        "gender": "Erkek",
        "difficulty": "Orta",
        "initial_complaint": (
            "Doktor bey, sabahtan beri hayatımda yaşamadığım kadar şiddetli bir baş ağrım var. "
            "Odanın ışıkları gözümü çok rahatsız ediyor ve boynumu öne eğemiyorum."
        ),
        "pain_details": (
            "Baş ağrım tüm başıma yayılıyor, zonklayıcı tarzda. Ağrı kesici içtim ama hiç fayda etmedi. "
            "Başımı öne doğru eğmeye çalıştığımda boynumun arkasında korkunç bir sertlik ve çekilme oluyor."
        ),
        "associated_symptoms": (
            "Ateşim 39 derecenin üzerine çıktı, iki kez fışkırır tarzda kustum. Işığa hiç bakamıyorum (fotofobi). "
            "Üzerimde hafif morumsu küçük lekeler fark ettim."
        ),
        "history": (
            "Üniversite 1. sınıf öğrencisiyim, kalabalık bir öğrenci yurdunda kalıyorum. "
            "Geçmişte önemli bir hastalığım yok."
        ),
        "physical_exam": (
            "[FİZİK MUAYENE BULGULARI]: Ateş: 39.5°C, Nabız: 114/dk, TA: 110/70 mmHg. Bilinç hafif konfüze (GKS: 14). "
            "Belirgin Ense Sertliği (+), Kernig bulgusu (+), Brudzinski bulgusu (+). "
            "Alt ekstremitelerde basmakla solmayan peteşiyal/purpurik döküntüler mevcut."
        ),
        "lab_and_imaging": (
            "[LABORATUVAR & LOMBER PONKSİYON]: WBC: 21.000 /mm³, CRP: 185 mg/L. "
            "BOS (Beyin Omurilik Sıvısı) İncelemesi: Görünüm bulanık, basınç artmış, nötrofilik pleositoz (1800 hücre/mm³), "
            "BOS glukozu çok düşük (18 mg/dL), BOS proteini yüksek (210 mg/dL). Gram boyamada Gram-negatif diplokoklar (Neisseria meningitidis)."
        ),
        "correct_diagnosis": "Akut Bakteriyel Menenjit (Meningokoksemi)",
        "accepted_keywords": ["menenjit", "meningitis", "bakteriyel menenjit", "meningokok"],
        "expected_esi": 1,
        "clinical_pearls": (
            "Yüksek ateş + Şiddetli baş ağrısı + Ense sertliği/Konfüzyon triadı, öğrenci yurdu öyküsü ve basmakla solmayan "
            "peteşiyal döküntüler Akut Bakteriyel Menenjit (Meningokoksemi) için kırmızı bayraktır. "
            "Kan kültürü alınıp saniyeler içinde ampirik İV Seftriakson + Vankomisin (+ Deksametazon) başlanmalıdır."
        ),
    },
    "CASE-004": {
        "case_id": "CASE-004",
        "title": "Vaka #04: 46 Yaş Kadın — Sağ Üst Kadran Ağrısı",
        "patient_name": "Ayşe D.",
        "age": 46,
        "gender": "Kadın",
        "difficulty": "Kolay",
        "initial_complaint": (
            "Doktor, dün akşam ağır yağlı bir yemek yedikten 1-2 saat sonra karnımın sağ üst tarafında "
            "şiddetli bir ağrı başladı. Sabaha kadar hiç geçmedi."
        ),
        "pain_details": (
            "Ağrı sağ kaburgamın hemen altında, sürekli ve kıvrandırıcı tarzda. Sağ kürek kemiğime ve sağ omzuma doğru "
            "vuruyor. Daha önce yemeklerden sonra 15-20 dakika sürüp geçerdi ama bu kez 8 saattir aralıksız sürüyor."
        ),
        "associated_symptoms": (
            "Gece iki kez sarı-yeşil acı bir su (safra) kustum. Sabah ateşimin çıktığını hissettim. "
            "Gözlerimde veya cildimde sarılık fark etmedim."
        ),
        "history": (
            "2 yıl önce yapılan kontrolde safra kesemde taş olduğu söylenmişti ama ağrı yapmadığı için ameliyat olmamıştım. "
            "Başka kronik hastalığım yok."
        ),
        "physical_exam": (
            "[FİZİK MUAYENE BULGULARI]: Ateş: 38.4°C, Nabız: 102/dk, TA: 130/80 mmHg. Skleralar doğal (ikter yok). "
            "Sağ üst kadran palpasyonunda derin inspiryumda hasta ağrı nedeniyle nefesini aniden kesiyor: Murphy Bulgusu (+)."
        ),
        "lab_and_imaging": (
            "[LABORATUVAR & GÖRÜNTÜLEME]: WBC: 14.200 /mm³, CRP: 62 mg/L, Total Bilirubin: 1.1 mg/dL (Normal), AST/ALT/ALP hafif yüksek. "
            "Üst Batın USG: Safra kesesi hidropik, lümen içinde 18 mm taş, kese duvar kalınlığı 5.5 mm (ödemli) ve perikolesistik sıvı (+)."
        ),
        "correct_diagnosis": "Akut Kolesistit",
        "accepted_keywords": ["kolesistit", "cholecystitis", "akut kolesistit", "safra kesesi"],
        "expected_esi": 3,
        "clinical_pearls": (
            "Yağlı yemek sonrası başlayan, 6 saatten uzun süren sağ üst kadran ağrısı, sağ skapulaya yayılım, ateş, "
            "lökositoz ve pozitif Murphy bulgusu (Tokyo Kriterleri) Akut Taşlı Kolesistit tanısını koydurur."
        ),
    },
    "CASE-005": {
        "case_id": "CASE-005",
        "title": "Vaka #05: 16 Yaş Kadın — Halsizlik, Hızlı Soluk Alıp Verme ve Karın Ağrısı",
        "patient_name": "Zeynep B.",
        "age": 16,
        "gender": "Kadın",
        "difficulty": "Zor",
        "initial_complaint": (
            "Doktor, sabahtan beri karnım çok ağrıyor, sürekli midem bulanıyor ve kendimi aşırı bitkin hissediyorum. "
            "Nefes alıp verişim hızlandı, ağzım kupkuru."
        ),
        "pain_details": (
            "Karın ağrım belli bir noktada değil, tüm karnıma yayılan kramp tarzında bir ağrı. "
            "Kusmam oldukça ağrım biraz daha artıyor."
        ),
        "associated_symptoms": (
            "Son 2-3 haftadır günde 5-6 litre su içiyorum (polidipsi) ve gece bile 4-5 kez idrara kalkıyorum (poliüri). "
            "Son bir ayda iştahım olmasına rağmen 6 kilo verdim."
        ),
        "history": (
            "Bilinen bir hastalığım yoktu ama annemde tiroid hastalığı var. Son günlerde sınav stresim vardı ve hafif bir grip geçirdim."
        ),
        "physical_exam": (
            "[FİZİK MUAYENE BULGULARI]: TA: 95/60 mmHg, Nabız: 126/dk, Solunum: 32/dk (Derin ve hızlı — Kussmaul solunumu). "
            "Mukoza ve dil kuru, cilt turgor-tonusu azalmış (ciddi dehidratasyon). Nefeste belirgin çürük elma / aseton kokusu mevcut."
        ),
        "lab_and_imaging": (
            "[LABORATUVAR & KAN GAZI]: Parmak ucu kan şekeri (Glukoz): 485 mg/dL. Arteriyel Kan Gazı: pH: 7.12, HCO3: 8 mEq/L, "
            "Anyon Açığı: 26 (Yüksek anyon açıklı metabolik asidoz). İdrarda Keton: (++++), Glukoz: (++++)."
        ),
        "correct_diagnosis": "Diyabetik Ketoasidoz (Tip 1 DM İlk Tanı)",
        "accepted_keywords": ["ketoasidoz", "dka", "diyabetik ketoasidoz", "diabetic ketoacidosis", "tip 1 diyabet"],
        "expected_esi": 1,
        "clinical_pearls": (
            "Genç hastada poliüri, polidipsi, kilo kaybı öyküsü üzerine eklenen akut karın ağrısı, Kussmaul solunumu, "
            "aseton kokusu, hiperglisemi (>250 mg/dL), ketonemi ve metabolik asidoz (pH < 7.30) Diyabetik Ketoasidoz (DKA) "
            "tablosudur. Acil İV %0.9 NaCl sıvı resüsitasyonu, potasyum takibi ve İV regüler insülin infüzyonu gerektirir."
        ),
    },
}


def list_public_cases() -> List[Dict[str, Any]]:
    """Doktor modunda kullaniciya gosterilecek vaka listesini (gizli teshisi ifsa etmeden) dondurur."""
    public_list = []
    # Tum vakalar gezilerek sadece baslangic profili secilir
    for case_id, data in CLINICAL_CASES.items():
        public_list.append(
            {
                "case_id": case_id,
                "title": data["title"],
                "patient_name": data["patient_name"],
                "age": data["age"],
                "gender": data["gender"],
                "difficulty": data["difficulty"],
                "initial_complaint": data["initial_complaint"],
            }
        )
    return public_list


def get_case_by_id(case_id: str) -> Optional[Dict[str, Any]]:
    """Belirtilen vaka kimligine ait tam vaka sozlugunu dondurur."""
    return CLINICAL_CASES.get(case_id)


def build_patient_system_prompt(case_id: str) -> str:
    """Uzak Colab LLM motoru icin hastayi canlandiracak gizli sistem istemini uretir."""
    case = get_case_by_id(case_id)
    if not case:
        return "Sen bir hastasin. Doktorun sorularina kisa ve dogal yanitlar ver."

    return (
        f"Sen {case['age']} yaşında {case['gender']} bir hastasın (Adın: {case['patient_name']}). "
        f"Karşındaki kullanıcı senin Doktorun. Kesinlikle kendi teşhisini ({case['correct_diagnosis']}) doğrudan söyleme! "
        f"Doktor hangi soruyu sorarsa sadece o bilgiyi doğal bir hasta gibi paylaş:\n"
        f"- İlk Şikayetin: {case['initial_complaint']}\n"
        f"- Ağrı Detayları: {case['pain_details']}\n"
        f"- Eşlik Eden Semptomlar: {case['associated_symptoms']}\n"
        f"- Özgeçmiş/İlaç/Yolculuk: {case['history']}\n"
        f"- Doktor Fizik Muayene yaparsa şu bulguyu ver: {case['physical_exam']}\n"
        f"- Doktor Kan/Tetkik/Görüntüleme isterse şu sonucu ver: {case['lab_and_imaging']}"
    )


def simulate_patient_response(case_id: str, doctor_question: str) -> str:
    """Doktorun sordugu soruyu analiz ederek ilgili vakanin hasta yanitini ve muayene bulgularini dondurur."""
    case = get_case_by_id(case_id)
    if not case:
        return "Özür dilerim doktor bey, sorunuzu tam anlayamadım."

    # Turkce karakter normalizasyonu yapilir
    mapping = {
        "ı": "i", "İ": "i", "ğ": "g", "Ğ": "g",
        "ü": "u", "Ü": "u", "ş": "s", "Ş": "s",
        "ö": "o", "Ö": "o", "ç": "c", "Ç": "c",
    }
    q = doctor_question.lower()
    for k, v in mapping.items():
        q = q.replace(k, v)

    responses: List[str] = []

    # 1. Doktor laboratuvar, kan tahlili, goruntuleme (BT, USG, EKG, LP) istediyse
    if any(w in q for w in ["tahlil", "tetkik", "kan", "lab", "usg", "ultrason", "bt", "tomografi", "ekg", "lomber", "idrar", "rontgen", "goruntuleme"]):
        responses.append(case["lab_and_imaging"])

    # 2. Doktor fizik muayene veya vital bulgu (ates olcumu, tansiyon, batin muayenesi) istediyse
    if any(w in q for w in ["muayene", "vital", "tansiyon", "nabiz", "dinle", "palpasyon", "rebound", "mcburney", "murphy", "ense", "bacak", "baldir", "homans", "saturasyon", "spo2"]):
        responses.append(case["physical_exam"])

    # 3. Doktor agrinin yeri, yayilimi, siddeti veya karakterini sorduysa
    if any(w in q for w in ["agri", "nere", "yayil", "vuruyor", "nasil", "siddet", "batma", "keskin", "ne zaman"]):
        responses.append(case["pain_details"])

    # 4. Doktor eslik eden semptomlari (ates, kusma, bulanti, istah, su icme, isik) sorduysa
    if any(w in q for w in ["ates", "bulanti", "kusma", "istifra", "istah", "ishal", "su", "kilo", "isik", "dokuntu", "leke", "oksuruk"]):
        responses.append(case["associated_symptoms"])

    # 5. Doktor ozgecmis, ilac, ameliyat, yolculuk, aliskanlik sorduysa
    if any(w in q for w in ["kronik", "hastalik", "ilac", "ameliyat", "alerji", "yolculuk", "ucak", "yurt", "aile", "gecmis", "dogum kontrol", "sigara"]):
        responses.append(case["history"])

    # Eger spesifik bir kategori eslesmediyse genel durumu ve sikayetini tekrar aciklar
    if not responses:
        return (
            f"{case['initial_complaint']} Ayrıca {case['associated_symptoms']} "
            "(İpucu: Ağrı karakterini, özgeçmişi sorabilir, 'Fizik muayene yap' veya 'Kan/Görüntüleme tetkiki iste' diyebilirsiniz.)"
        )

    # Birden fazla soru ayni anda sorulduysa birlestirerek dondur
    return "\n\n".join(responses)


def evaluate_doctor_submission(
    case_id: str,
    submitted_diagnosis: str,
    submitted_esi: int,
    question_count: int = 1,
) -> Dict[str, Any]:
    """Doktorun vaka sonunda girdigi teshisi ve aciliyet skorunu degerlendirip puanlar."""
    case = get_case_by_id(case_id)
    if not case:
        raise ValueError(f"Gecersiz vaka kimligi: {case_id}")

    # Girdi metni normalize edilir
    mapping = {
        "ı": "i", "İ": "i", "ğ": "g", "Ğ": "g",
        "ü": "u", "Ü": "u", "ş": "s", "Ş": "s",
        "ö": "o", "Ö": "o", "ç": "c", "Ç": "c",
    }
    diag_norm = submitted_diagnosis.lower().strip()
    for k, v in mapping.items():
        diag_norm = diag_norm.replace(k, v)

    # Teshis eslesme kontrolu
    is_diagnosis_correct = False
    for kw in case["accepted_keywords"]:
        kw_norm = kw.lower()
        for k, v in mapping.items():
            kw_norm = kw_norm.replace(k, v)
        if kw_norm in diag_norm:
            is_diagnosis_correct = True
            break

    # ESI seviyesi dogruluk kontrolu
    expected_esi = case["expected_esi"]
    esi_diff = abs(submitted_esi - expected_esi)
    is_esi_exact = esi_diff == 0

    # Puanlama Matrisi (Toplam 100 Puan):
    # - Dogru Teshis: 60 Puan
    # - Dogru ESI Triyaj Seviyesi: 25 Puan (1 seviye sapma: 15 Puan)
    # - Yeterli Anamnez Sorgulamasi (en az 3 soru/muayene): 15 Puan
    score = 0
    if is_diagnosis_correct:
        score += 60
    if is_esi_exact:
        score += 25
    elif esi_diff == 1:
        score += 15

    anamnesis_points = min(15, question_count * 5)
    score += anamnesis_points

    # Klinik Geri Bildirim Olusturma
    if is_diagnosis_correct and is_esi_exact:
        feedback = "Tebrikler Doktor! Hem kesin tanıyı hem de acil servis triyaj seviyesini (ESI) kusursuz belirlediniz."
    elif is_diagnosis_correct:
        feedback = (
            f"Tanınız DOĞRU ({case['correct_diagnosis']})! Ancak bu tablo için beklenen aciliyet seviyesi "
            f"ESI-{expected_esi} olmalıydı (Sizin seçiminiz: ESI-{submitted_esi})."
        )
    else:
        feedback = (
            f"Girdiğiniz tanı ('{submitted_diagnosis}') bu vakanın asıl patolojisiyle tam örtüşmedi. "
            f"Doğru tanı: {case['correct_diagnosis']} (Beklenen Aciliyet: ESI-{expected_esi})."
        )

    return {
        "case_id": case_id,
        "is_correct": is_diagnosis_correct,
        "score": score,
        "submitted_diagnosis": submitted_diagnosis,
        "correct_diagnosis": case["correct_diagnosis"],
        "submitted_esi": submitted_esi,
        "expected_esi": expected_esi,
        "feedback": feedback,
        "clinical_pearls": case["clinical_pearls"],
    }
