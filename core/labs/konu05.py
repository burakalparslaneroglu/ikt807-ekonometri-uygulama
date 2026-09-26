"""Konu 5 uygulama laboratuvarı: ikili sonuçlarda katsayıdan olasılığa geçmek.

Ders notları §5.15'in (Adım 1–5 ve §5.15.6) birebir karşılığıdır; Adım 4 aynı zamanda
§5.9'daki Tablo 5.1'i yeniden üretir. Veri: Hansen'in cps09mar dosyası, 35 yaş ve altındaki
erkekler (9.137 gözlem). Her ``Check`` notlarda basılı bir sayıdır.

Spesifikasyon: evli = medeni durum kodu 1–3; açıklayıcılar yaş, eğitim yılı, dört gruplu ırk
(beyaz referans; siyah, Asyalı, diğer), Hispanik göstergesi ve bölge (Kuzeydoğu referans).
Irkın ayrıntılı kodlarıyla (bu alt örneklemde 18 kategori) kurulan model yakınsamaz: dört küçük
kategoride (1–2 gözlem) herkes evli veya herkes bekârdır (yarı-tam ayrışma) ve bu katsayıların
MLE'si sonsuza gider.
"""

from __future__ import annotations

from core.hansen_data import CPS09MAR_COLUMNS
from core.labs import expr as E
from core.labs.spec import (
    OLS,
    AverageProfile,
    BinaryChoice,
    Check,
    CoefTarget,
    Derive,
    EffectTable,
    KeepIf,
    LabSpec,
    LabStep,
    LoadHansen,
    MarginalEffects,
    NoteRef,
    Recode,
    ReproClass,
    Scalar,
    ScalarTarget,
    ShowModel,
    StatTarget,
    TableTarget,
)

SECTION = "5.15"
FRAME = "cps"
REGRESSORS = ("age", "education", "race4", "hisp", "region")
CATEGORICAL = ("race4", "hisp", "region")
AGES = tuple(float(age) for age in range(18, 36))

TERMS = (
    ("age", "Yaş"),
    ("education", "Eğitim"),
    ("race4=2", "Siyah"),
    ("race4=3", "Asyalı"),
    ("race4=4", "Diğer ırk"),
    ("hisp=1", "Hispanik"),
    ("region=2", "Orta Batı"),
    ("region=3", "Güney"),
    ("region=4", "Batı"),
)

# Tablo 5.1 (§5.9): (AME, SH) Logit ve Probit için; dayanıklı kovaryans, kuklalarda sonlu fark.
TABLE_51 = {
    "logit": (
        (0.0441, 0.0009), (0.0027, 0.0020), (-0.1617, 0.0184), (0.0001, 0.0210), (-0.0971, 0.0255),
        (-0.0173, 0.0130), (0.0559, 0.0151), (0.0709, 0.0144), (0.0840, 0.0148),
    ),
    "probit": (
        (0.0445, 0.0009), (0.0028, 0.0020), (-0.1582, 0.0181), (0.0019, 0.0212), (-0.0934, 0.0249),
        (-0.0170, 0.0129), (0.0557, 0.0152), (0.0706, 0.0145), (0.0825, 0.0149),
    ),
}


def _effect(model: str, term: str, expected: float, label: str, quantity: str = "coef", decimals: int = 4) -> Check:
    return Check(label, CoefTarget(model, term, quantity), expected, decimals)


def _table_51_checks() -> tuple[Check, ...]:
    checks: list[Check] = []
    for link, title in (("logit", "Logit"), ("probit", "Probit")):
        for (term, label), (estimate, error) in zip(TERMS, TABLE_51[link]):
            checks.append(_effect(f"ame_{link}", term, estimate, f"{title} AME: {label}"))
            checks.append(_effect(f"ame_{link}", term, error, f"{title} SH: {label}", "se"))
    return tuple(checks)


def _effect_rows(model: str) -> tuple[tuple[str, str, str], ...]:
    return tuple((label, model, term) for term, label in TERMS)


STEPS = (
    LabStep(
        number=1,
        title="Aynı araştırma sorusunu üç model ailesiyle sormak",
        note=NoteRef(SECTION, 1, ("Tablo 5.2",)),
        explanation=(
            "Soru: 35 yaş ve altındaki çalışan erkeklerde evli olma olasılığı yaş, eğitim, ırk/etnik köken ve bölgeyle "
            "nasıl ilişkili? Sonuç ikili: $married_i\\in\\{0,1\\}$ (medeni durum kodu 1–3). Aynı soruyu üç modelle "
            "soruyoruz: doğrusal olasılık modeli (LPM), Logit ve Probit.\n\n"
            "LPM'de katsayı doğrudan olasılık farkıdır. Logit ve Probit'te ham katsayı tek indeks ölçeğindedir; "
            "karşılaştırma **ortalama marjinal etki (AME)** üzerinden yapılır. Kovaryans: LPM'de HC1, Logit/Probit'te "
            "sandviç (dayanıklı). Irk dört gruba toplanır (beyaz referans). Ayrıntılı ırk kodlarıyla (bu alt örneklemde 18 "
            "kategori) Logit yakınsamaz: 1–2 gözlemli dört kategoride herkes evli ya da herkes bekârdır, bu kategorilerin "
            "katsayısının MLE'si sonsuza gider (yarı-tam ayrışma)."
        ),
        operations=(
            LoadHansen("cps09mar", "cps09mar.txt", FRAME, CPS09MAR_COLUMNS),
            KeepIf(FRAME, (("age", "<=", 35), ("female", "==", 0)), "Analiz örneklemi: 35 yaş ve altındaki erkekler"),
            Derive(FRAME, "married", E.compare("le", E.var("marital"), 3),
                   "Evli: medeni durum 1–3 (eşiyle, silahlı kuvvetlerdeki eşiyle veya eşi ayrı yaşayan evli)"),
            Recode(FRAME, "race4", "race", (((1,), 1), ((2,), 2), ((4,), 3)), 4,
                   "Irk dört grup: 1 beyaz, 2 siyah, 3 Asyalı, 4 diğer (Hansen'in 21 kodundan)"),
            OLS("lpm", FRAME, "married", REGRESSORS, vcov="HC1", categorical=CATEGORICAL),
            BinaryChoice("logit", FRAME, "married", REGRESSORS, "logit", CATEGORICAL),
            BinaryChoice("probit", FRAME, "married", REGRESSORS, "probit", CATEGORICAL),
            MarginalEffects("logit", "ame_logit", REGRESSORS),
            MarginalEffects("probit", "ame_probit", REGRESSORS),
            EffectTable(
                (
                    ("LPM: yaş", "lpm", "age"),
                    ("Logit AME: yaş", "ame_logit", "age"),
                    ("Probit AME: yaş", "ame_probit", "age"),
                    ("LPM: eğitim", "lpm", "education"),
                    ("Logit AME: eğitim", "ame_logit", "education"),
                    ("Probit AME: eğitim", "ame_probit", "education"),
                ),
                "tablo_52",
                title="Tablo 5.2: LPM ve doğrusal olmayan modellerde olasılık etkileri",
                se_label="SH",
            ),
        ),
        checks=(
            Check("Analiz örneklemi (N)", StatTarget(FRAME, "married", "count"), 9137, decimals=0),
            _effect("lpm", "age", 0.0466, "LPM yaş"),
            _effect("lpm", "age", 0.0010, "LPM yaş HC1 SH", "se"),
            _effect("lpm", "education", 0.00245, "LPM eğitim", decimals=5),
            _effect("lpm", "education", 0.00199, "LPM eğitim HC1 SH", "se", decimals=5),
            _effect("ame_logit", "age", 0.0441, "Logit AME yaş"),
            _effect("ame_logit", "age", 0.0009, "Logit AME yaş SH", "se"),
            _effect("ame_logit", "education", 0.00266, "Logit AME eğitim", decimals=5),
            _effect("ame_logit", "education", 0.00198, "Logit AME eğitim SH", "se", decimals=5),
            _effect("ame_probit", "age", 0.0445, "Probit AME yaş"),
            _effect("ame_probit", "age", 0.0009, "Probit AME yaş SH", "se"),
            _effect("ame_probit", "education", 0.00281, "Probit AME eğitim", decimals=5),
            _effect("ame_probit", "education", 0.00199, "Probit AME eğitim SH", "se", decimals=5),
        ),
        reproducibility=ReproClass.CONVENTION,
        takeaway=(
            "Yaş için üç yöntem aynı mesajı verir: Logit AME'si 0,0441, yani yaşın bir yıl artması evli olma olasılığında "
            "ortalama yaklaşık **4,4 yüzde puanlık** farkla ilişkilidir. Bu, \"olasılık yüzde 4,4 artar\" demek değildir: "
            "başlangıç olasılığı yüzde 50 ise 4,4 yüzde puan yaklaşık yüzde 8,8 göreli artıştır. Eğitimin AME'si yaklaşık "
            "0,0027–0,0028 (0,27–0,28 yüzde puan); standart hatası yaklaşık 0,002 olduğu için bu ilişkide belirsizlik "
            "yaşa göre çok daha büyüktür."
        ),
        code_note=(
            "Logit/Probit dayanıklı kovaryansı üç dilde aynı tanımla hesaplanır: gözlenen Hessian ile sandviç "
            "H⁻¹(Σsᵢsᵢ')H⁻¹. Python `cov_type=\"HC0\"`, R `dayanikli_vcov()`, Stata `vce(robust)`. Stata ayrıca n/(n−1) "
            "ile çarpar (bu örneklemde fark beşinci anlamlı basamakta). R'nin `sandwich::sandwich()` fonksiyonu Probit'te "
            "beklenen bilgi matrisini kullandığı için standart hataları dördüncü basamakta farklı verir; bu yüzden "
            "kullanılmaz. AME'ler Python/R'de kodda tanımlı fonksiyonla, Stata'da `margins` ile hesaplanır."
        ),
    ),
    LabStep(
        number=2,
        title="Logit katsayısını neden doğrudan yorumlamıyoruz?",
        note=NoteRef(SECTION, 2),
        explanation=(
            "Logit modelinde $P(Y_i=1\\mid X_i)=\\Lambda(X_i'\\beta)$ ve sürekli $X_j$ için marjinal etki\n\n"
            "$$\\frac{\\partial P(Y_i=1\\mid X_i)}{\\partial X_{ij}}=\\Lambda(X_i'\\beta)\\{1-\\Lambda(X_i'\\beta)\\}"
            "\\beta_j.$$\n\n"
            "Aynı $\\beta_j$ farklı bireylerde farklı olasılık etkisi üretir; AME bu bireysel etkilerin örneklem "
            "ortalamasıdır: $\\widehat{AME}_j=\\hat\\beta_j\\cdot\\frac1n\\sum_i g(X_i'\\hat\\beta)$. Aşağıda ham katsayı ile "
            "ortalama ölçek çarpanı ayrı ayrı gösteriliyor."
        ),
        operations=(
            Scalar("logit_yas", E.coef("logit", "age"), "Logit yaş katsayısı β̂", percent=False, decimals=4),
            Scalar("olcek_logit", E.div(E.coef("ame_logit", "age"), E.coef("logit", "age")),
                   "Ortalama Λ(1−Λ)", percent=False, decimals=4),
            Scalar("probit_yas", E.coef("probit", "age"), "Probit yaş katsayısı β̂", percent=False, decimals=4),
            Scalar("olcek_probit", E.div(E.coef("ame_probit", "age"), E.coef("probit", "age")),
                   "Ortalama φ", percent=False, decimals=4),
            Scalar("katsayi_orani", E.div(E.coef("logit", "age"), E.coef("probit", "age")),
                   "Logit/Probit katsayı oranı", percent=False, decimals=2),
        ),
        checks=(
            Check("Logit yaş katsayısı", ScalarTarget("logit_yas"), 0.2165),
            Check("Ortalama Λ(1−Λ)", ScalarTarget("olcek_logit"), 0.2036),
            Check("Probit yaş katsayısı", ScalarTarget("probit_yas"), 0.1315),
            Check("Ortalama φ", ScalarTarget("olcek_probit"), 0.3380),
            Check("Logit/Probit katsayı oranı", ScalarTarget("katsayi_orani"), 1.65, decimals=2),
        ),
        reproducibility=ReproClass.EXACT,
        takeaway=(
            "Logit yaş katsayısı 0,2165 ama ortalama ölçek çarpanı Λ(1−Λ) 0,2036; çarpımları AME'yi (0,0441) verir. "
            "Probit'te katsayı 0,1315, ortalama φ 0,3380; çarpım 0,0445. Ham katsayıların oranı yaklaşık 1,65'tir: bu fark "
            "ekonomik değil, iki modelin gizli hata ölçeğini farklı normalize etmesinden gelir (§5.5). Bu nedenle bir "
            "makalede Logit katsayısı 0,40 görmek \"olasılık 40 yüzde puan artar\" anlamına gelmez."
        ),
    ),
    LabStep(
        number=3,
        title="Tahmin edilen olasılık profilini okumak",
        note=NoteRef(SECTION, 3, ("Şekil 5.4",)),
        explanation=(
            "Her yaş için örneklemdeki **herkesin** yaşı o değere eşitlenir; diğer kovaryatlar kendi gözlenen "
            "değerlerinde kalır ve bireysel tahmin edilen olasılıklar ortalanır. Bu, \"ortalama bireyin olasılığı\" ile "
            "aynı işlem değildir: doğrusal olmayan modelde $G(\\mathbb E[X]'\\hat\\beta)\\neq\\mathbb E[G(X'\\hat\\beta)]$."
        ),
        operations=(
            AverageProfile("logit", "profil", "age", AGES, "Yaş", "Ortalama tahmin edilen evlilik olasılığı",
                           "CPS: Logit modelinde yaş profili"),
        ),
        checks=(
            Check("18 yaşında ortalama olasılık", TableTarget("profil", 18, "olasilik"), 0.1067),
            Check("35 yaşında ortalama olasılık", TableTarget("profil", 35, "olasilik"), 0.8186),
        ),
        reproducibility=ReproClass.EXACT,
        takeaway=(
            "Ortalama tahmin edilen evlilik olasılığı 18 yaşında yaklaşık 0,11, 35 yaşında 0,82'dir. Profil S biçimlidir: "
            "eğim orta yaşlarda en büyük, uçlarda daha küçüktür. Doğrusal yaş indeksi 35 yaş altı alt örneklemde makul bir "
            "yerel yaklaşımdır; tam yaş aralığında aynı spesifikasyon yanıltıcı olabilir."
        ),
        code_note="Stata'da aynı hesap `margins, at(age=(18(1)35))` komutudur; grafik `marginsplot` ile çizilir.",
    ),
    LabStep(
        number=4,
        title="Yazılım çıktısından AME tablosuna geçmek",
        note=NoteRef(SECTION, 4, ("§5.9", "Tablo 5.1")),
        explanation=(
            "Yazılımın ilk çıktısı modelin ham Logit katsayılarıdır; ikinci çıktı olasılık ölçeğindeki ortalama marjinal "
            "etkilerdir. Sürekli değişkenlerde (yaş, eğitim) türevin, göstergelerde referans kategoriye göre tahmin edilen "
            "olasılık farkının örneklem ortalaması alınır: ırkta referans beyaz, bölgede Kuzeydoğu. Standart hatalar "
            "dayanıklı kovaryans matrisinden delta yöntemiyle gelir (§5.7.3). Sonuç, §5.9'daki Tablo 5.1'dir."
        ),
        operations=(
            ShowModel("logit"),
            EffectTable(_effect_rows("ame_logit"), "tablo_51_logit", title="Tablo 5.1 (Logit): ortalama marjinal etkiler",
                        se_label="Dayanıklı SH"),
            EffectTable(_effect_rows("ame_probit"), "tablo_51_probit",
                        title="Tablo 5.1 (Probit): ortalama marjinal etkiler", se_label="Dayanıklı SH"),
        ),
        checks=_table_51_checks(),
        reproducibility=ReproClass.CONVENTION,
        takeaway=(
            "Logit ve Probit ham katsayıları farklı ölçekte olsa da AME'ler neredeyse aynıdır. Siyah göstergesi, diğer "
            "değişkenler aynıyken evli olma olasılığında yaklaşık 16 yüzde puan daha düşük bir değerle ilişkilidir; Asyalı "
            "göstergesi sıfıra çok yakın ve belirsizdir. \"Diğer ırk\" grubu da (−0,097) beyazlardan belirgin biçimde "
            "ayrılır: bu grubu referans kategoriye katmak, siyah ve Asyalı katsayılarının anlamını değiştirirdi."
        ),
        code_note=(
            "Stata'da `margins, dydx(age education race4 hisp region)`: `i.` ile girilen değişkenlerde sonlu farkı "
            "kendiliğinden alır. Python'da statsmodels'in `get_margeff(dummy=True)` seçeneği çok düzeyli kategorik "
            "değişkende düzeyleri birbirinden bağımsız 0/1 yapar (olanaksız kombinasyonlar üretir); bu yüzden kodda "
            "tanımlı fonksiyon kullanılır."
        ),
    ),
    LabStep(
        number=5,
        title="Kukla değişken için türev değil sonlu fark",
        note=NoteRef(SECTION, 5),
        explanation=(
            "$D\\in\\{0,1\\}$ bir göstergede türev yerine\n\n"
            "$$\\Delta_D(X)=G(X'\\hat\\beta+\\hat\\beta_D)-G(X'\\hat\\beta)$$\n\n"
            "hesaplanır: her gözlem için $D=1$ ve $D=0$ senaryoları tahmin edilip farkların ortalaması alınır. Türev "
            "tabanlı yaklaşım $\\hat\\beta_D\\cdot\\frac1n\\sum_i g(X_i'\\hat\\beta)$, $D$'yi sürekli bir değişken gibi "
            "küçük bir değişimle ele alır."
        ),
        operations=(
            Scalar("siyah_turev", E.mul(E.coef("logit", "race4=2"), E.div(E.coef("ame_logit", "age"), E.coef("logit", "age"))),
                   "Siyah: türev tabanlı yaklaşım", percent=False, decimals=4),
            Scalar("siyah_fark", E.coef("ame_logit", "race4=2"), "Siyah: sonlu fark (AME)", percent=False, decimals=4),
        ),
        checks=(
            Check("Siyah, türev tabanlı", ScalarTarget("siyah_turev"), -0.1598),
            Check("Siyah, sonlu fark", ScalarTarget("siyah_fark"), -0.1617),
        ),
        reproducibility=ReproClass.EXACT,
        takeaway=(
            "Siyah göstergesi için türev tabanlı yaklaşım −0,1598, doğru tanım olan sonlu fark −0,1617'dir. Bu örnekte fark "
            "küçüktür, çünkü olasılıklar 0,5 civarında ve katsayı orta büyüklüktedir; katsayı büyüdükçe veya olasılıklar "
            "0 ya da 1'e yaklaştıkça fark büyür (Sezgi Deney 3). Yazılımın marjinal etki komutunun göstergeleri nasıl "
            "ele aldığını her zaman kontrol edin."
        ),
    ),
    LabStep(
        number=6,
        title="Bir ikili sonuç makalesindeki tabloyu nasıl okumalıyız?",
        note=NoteRef("5.15.6", 0),
        explanation=(
            "1. Sonuç değişkeninde 1 hangi olayı temsil ediyor?\n"
            "2. Raporlanan sayılar ham Logit/Probit katsayısı mı, odds ratio mu, AME mi?\n"
            "3. Sürekli ve kukla değişkenler için aynı etki tanımı mı kullanılmış?\n"
            "4. Standart hatalar hangi kovaryans yapısına dayanıyor?\n"
            "5. Tahmin edilen olasılıklar $[0,1]$ aralığında mı ve model uç bölgelerde yeterli destek görüyor mu?\n"
            "6. Logit ve Probit benzer olasılık sonuçları veriyor mu?\n"
            "7. Modelin nedensel yorumu için hangi tanımlama varsayımı kullanılıyor?"
        ),
        takeaway=(
            "Tez yazımında örnek sonuç dili: \"Logit modelinden elde edilen ortalama marjinal etki, yaşın bir yıl artmasının "
            "evli olma olasılığında ortalama yaklaşık 4,4 yüzde puanlık pozitif bir farkla ilişkili olduğunu "
            "göstermektedir. Probit tahmini de 4,4 yüzde puan ile neredeyse aynıdır. Sonuçlar gözlemsel CPS verisine "
            "dayandığından katsayılar nedensel yaşam-döngüsü etkisi olarak yorumlanmamıştır.\""
        ),
    ),
)


KONU05_LAB = LabSpec(
    topic_key="konu05",
    title="İkili Sonuçlarda Katsayıdan Olasılığa Geçmek",
    dataset="cps09mar",
    note_section=SECTION,
    steps=STEPS,
    labels=(
        ("(sabit)", "Sabit"),
        ("married", "Evli"),
        ("age", "Yaş"),
        ("education", "Eğitim yılı"),
        ("race4", "Irk (4 grup)"),
        ("hisp", "Hispanik"),
        ("region", "Bölge"),
        ("lpm", "LPM"),
        ("logit", "Logit"),
        ("probit", "Probit"),
        ("ame_logit", "Logit AME"),
        ("ame_probit", "Probit AME"),
        ("race4=2", "Siyah"),
        ("race4=3", "Asyalı"),
        ("race4=4", "Diğer ırk"),
        ("hisp=1", "Hispanik"),
        ("region=2", "Orta Batı"),
        ("region=3", "Güney"),
        ("region=4", "Batı"),
    ),
)
