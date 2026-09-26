"""Konu 2 uygulama laboratuvarı: regresyon tablosunu güvenilir okumak.

Ders notları §2.13'ün (Adım 1–7) birebir karşılığıdır. Her ``Check`` notlarda basılı
bir sayıdır. Veri hazırlığı Konu 1 ile ortaktır (aynı dört türetme).
"""

from __future__ import annotations

from core.labs import expr as E
from core.labs.konu01 import KONU01_LAB
from core.labs.spec import (
    OLS,
    BreuschPagan,
    Check,
    CoefTarget,
    DeltaMethod,
    Derive,
    LabSpec,
    LabStep,
    LinearCombination,
    ModelTarget,
    NoteRef,
    ReproClass,
    Scalar,
    ScalarTarget,
    ShowModel,
    StandardErrorTable,
)

SECTION = "2.13"
FRAME = "cps"
CONTROLS = ("experience", "experience2_100", "female", "region", "race", "hisp")
CATEGORICAL = ("region", "race")

M1 = ("education",)
M2 = ("education", *CONTROLS)
M3 = ("education", "edu_female", *CONTROLS)


def _coef(model: str, term: str, expected: float, label: str, quantity: str = "coef") -> Check:
    return Check(label, CoefTarget(model, term, quantity), expected)


STEPS = (
    LabStep(
        number=1,
        title="Araştırma sorusunu ve hedef katsayıyı açıkça tanımlamak",
        note=NoteRef(SECTION, 1),
        explanation=(
            "Araştırma sorusu: *Eğitim yılı ile saatlik ücret arasındaki koşullu ilişki, deneyim ve temel "
            "demografik özellikler dikkate alındığında ne ölçüdedir?* Bağımlı değişken "
            "$Y_i=\\log(wage_i)$, temel açıklayıcı değişken eğitim yılıdır."
        ),
        takeaway=(
            "Eğitim katsayısını \"nedensel eğitim getirisi\" diye adlandırmıyoruz: yetenek ve aile "
            "geçmişi gibi gözlenmeyen etkenler böyle bir yorum için ek tanımlama varsayımları gerektirir. "
            "Hedef, doğrusal koşullu ilişkinin güvenilir biçimde tahmin ve rapor edilmesidir."
        ),
    ),
    LabStep(
        number=2,
        title="Birden çok spesifikasyon kurmak",
        note=NoteRef(SECTION, 2),
        explanation=(
            "Üç iç içe model: **M1** yalnız eğitim; **M2** deneyim profili, kadın göstergesi, bölge ve "
            "ırk kuklaları ile Hispanik göstergesi eklenmiş; **M3** buna eğitim × kadın etkileşimini "
            "ekler, yani eğitim eğiminin kadın ve erkeklerde farklı olabilmesine izin verir."
        ),
        operations=KONU01_LAB.step(2).operations
        + (
            Derive(FRAME, "edu_female", E.mul(E.var("education"), E.var("female")), "Eğitim × kadın etkileşimi"),
            OLS("m1", FRAME, "lwage", M1),
            OLS("m2", FRAME, "lwage", M2, categorical=CATEGORICAL),
            OLS("m3", FRAME, "lwage", M3, categorical=CATEGORICAL),
        ),
        takeaway=(
            "Spesifikasyonların amacı \"anlamlı katsayı bulmak\" değil, araştırma sorusuna uygun koşullu "
            "ilişkiyi adım adım görünür kılmaktır."
        ),
        code_note="Bölge ve ırk kategorik değişkenlerdir; her düzey için bir kukla (ilk düzey referans) eklenir.",
    ),
    LabStep(
        number=3,
        title="Katsayı ile standart hatayı birbirinden ayırmak",
        note=NoteRef(SECTION, 3, ("Tablo 2.3",)),
        explanation=(
            "Tabloda önce katsayı okunur: M2'de eğitim katsayısı 0,1100. Kesin dönüşümle "
            "$100\\{\\exp(0{,}1100)-1\\}\\approx 11{,}63$: diğer değişkenler sabitken bir ek eğitim yılı "
            "yaklaşık yüzde 11,6 daha yüksek saatlik ücretle ilişkilidir. Sonra standart hata okunur."
        ),
        operations=(
            StandardErrorTable(("m1", "m2", "m3"), "education", "sh_tablosu"),
            Scalar(
                "yuzde_etki",
                E.mul(100, E.sub(E.exp(E.coef("m2", "education")), 1)),
                "M2 eğitim katsayısının kesin yüzde karşılığı",
            ),
        ),
        checks=(
            _coef("m1", "education", 0.1082, "M1 eğitim katsayısı"),
            _coef("m1", "education", 0.0010, "M1 klasik SH", "se"),
            _coef("m1", "education", 0.0011, "M1 HC1 SH", "se_hc1"),
            Check("M1 R²", ModelTarget("m1", "r2"), 0.1930),
            _coef("m2", "education", 0.1100, "M2 eğitim katsayısı"),
            _coef("m2", "education", 0.0010, "M2 klasik SH", "se"),
            _coef("m2", "education", 0.0011, "M2 HC1 SH", "se_hc1"),
            Check("M2 R²", ModelTarget("m2", "r2"), 0.2828),
            _coef("m3", "education", 0.1083, "M3 eğitim katsayısı"),
            _coef("m3", "education", 0.0012, "M3 klasik SH", "se"),
            _coef("m3", "education", 0.0015, "M3 HC1 SH", "se_hc1"),
            Check("M3 R²", ModelTarget("m3", "r2"), 0.2829),
            Check("M2 kesin yüzde etki", ScalarTarget("yuzde_etki"), 11.63, decimals=2),
        ),
        takeaway=(
            "HC1 kullanıldığında **katsayı değişmez**; yalnız tahminin belirsizliği yeniden hesaplanır. "
            "\"Robust regresyon yaptım ve katsayı düzeldi\" ifadesi genel olarak yanlıştır."
        ),
    ),
    LabStep(
        number=4,
        title="Heteroskedastisite tanısını makale diliyle okumak",
        note=NoteRef(SECTION, 4),
        explanation=(
            "M2 artıklarına Breusch–Pagan testi uygulandığında LM istatistiği yaklaşık 175,16, "
            "p-değeri $10^{-23}$ mertebesindedir: homoskedastisite varsayımıyla güçlü biçimde uyuşmaz."
        ),
        operations=(BreuschPagan("m2", "bp"),),
        checks=(Check("Breusch–Pagan LM", ScalarTarget("bp_lm"), 175.16, decimals=2),),
        reproducibility=ReproClass.CONVENTION,
        takeaway=(
            "İki yanlış sonuç çıkarılmamalı: heteroskedastisite OLS katsayısını otomatik olarak yanlı "
            "yapmaz; dayanıklı standart hata da eksik değişken veya içsellik sorununu çözmez. Doğru "
            "raporlama: \"çıkarım heteroskedastisiteye dayanıklı standart hatalarla yapılmıştır.\""
        ),
        code_note=(
            "Stata'nın `estat hettest` komutu varsayılan olarak başka bir sürümü (yalnız uyum değerleri, "
            "normallik varsayımı) hesaplar. Python ve R ile aynı sayı için `rhs iid` seçenekleri gerekir."
        ),
    ),
    LabStep(
        number=5,
        title="Etkileşim katsayısını doğru yorumlamak",
        note=NoteRef(SECTION, 5),
        explanation=(
            "M3'te erkekler referans gruptur: eğitim katsayısı 0,1083 erkeklerin eğimidir; etkileşim "
            "0,0044 ise kadınların eğiminin bundan farkıdır. Kadınlar için eğim bir **doğrusal "
            "birleşimdir**: $0{,}1083+0{,}0044\\approx0{,}1127$. Standart hatası $\\sqrt{a'Va}$ ile, "
            "$a=(1,1)'$ ve $V$ HC1 kovaryans matrisi olmak üzere hesaplanır."
        ),
        operations=(
            OLS("m3_hc1", FRAME, "lwage", M3, vcov="HC1", categorical=CATEGORICAL),
            LinearCombination(
                "kadin_egimi", "m3_hc1", (("education", 1.0), ("edu_female", 1.0)), "Kadınlar için eğitim eğimi"
            ),
        ),
        checks=(
            _coef("m3", "education", 0.1083, "Erkeklerin eğimi (eğitim katsayısı)"),
            _coef("m3", "edu_female", 0.0044, "Etkileşim katsayısı"),
            Check("Kadınların eğimi", ScalarTarget("kadin_egimi"), 0.1127),
        ),
        takeaway=(
            "Etkileşimli modellerde ana etki, diğer etkileşim değişkeninin referans düzeyindeki eğimdir. "
            "İki katsayıyı ayrı ayrı yorumlamak yerine ilgili doğrusal birleşim hesaplanmalıdır."
        ),
    ),
    LabStep(
        number=6,
        title="Yazılımda güvenilir standart hatayı açıkça istemek",
        note=NoteRef(SECTION, 6),
        explanation=(
            "HC1 seçeneği katsayı denklemine yeni bir değişken eklemez; aynı OLS katsayıları üzerinde "
            "heteroskedastisiteye dayanıklı standart hata hesabını etkinleştirir. Aynı çıktıdan yüzde "
            "etkinin standart hatası delta yöntemiyle hesaplanır: $g(\\beta)=100(e^{\\beta}-1)$ için "
            "$\\operatorname{SH}(g)\\approx g'(\\hat\\beta)\\,\\operatorname{SH}(\\hat\\beta)$."
        ),
        operations=(
            OLS("m2_hc1", FRAME, "lwage", M2, vcov="HC1", categorical=CATEGORICAL),
            ShowModel("m2_hc1"),
            DeltaMethod(
                "yuzde_etki_dm",
                "m2_hc1",
                E.mul(100, E.sub(E.exp(E.coef("m2_hc1", "education")), 1)),
                "M2 kesin yüzde etki",
            ),
        ),
        checks=(
            _coef("m2_hc1", "education", 0.1100, "HC1 çıktısında eğitim katsayısı"),
            _coef("m2_hc1", "education", 0.0011, "HC1 çıktısında standart hata", "se"),
        ),
        takeaway=(
            "Bir makalede \"robust standart hatalar\" ifadesini gördüğünüzde hangi robust türünün "
            "kullanıldığını ve veri kümeli ise kümelenme biriminin ne olduğunu ayrıca kontrol edin."
        ),
    ),
    LabStep(
        number=7,
        title="Bir makale tablosunu hangi sırayla okumalıyız?",
        note=NoteRef(SECTION, 7),
        explanation=(
            "Güvenli okuma sırası: (1) bağımlı değişken düzey mi, log mu; (2) temel katsayının birimi; "
            "(3) standart hata türü: klasik, HC, küme-robust, bootstrap; (4) güven aralığı ve p-değeri, "
            "yalnız yıldızlar değil; (5) sütunlar arası spesifikasyon farkı; (6) ekonomik büyüklük; "
            "(7) tanımlama dili: gözlemsel ilişki nedensel etki diye sunuluyor mu?"
        ),
        takeaway=(
            "\"HC1 standart hataları kullanılan temel spesifikasyonda eğitim katsayısı 0,110 olarak "
            "tahmin edilmiştir. Bu değer, deneyim ve gözlenen demografik özellikler sabit tutulduğunda bir "
            "ek eğitim yılının saatlik ücrette yaklaşık yüzde 11,6 daha yüksek bir değerle ilişkili "
            "olduğunu göstermektedir. Tahmin edilen ilişki nedensel eğitim getirisi olarak yorumlanmamalıdır.\""
        ),
    ),
)


KONU02_LAB = LabSpec(
    topic_key="konu02",
    title="Regresyon Tablosunu Güvenilir Okumak",
    dataset="cps09mar",
    note_section=SECTION,
    steps=STEPS,
    labels=KONU01_LAB.labels
    + (
        ("edu_female", "Eğitim × kadın"),
        ("region", "Bölge"),
        ("race", "Irk"),
        ("hisp", "Hispanik"),
        ("m1", "M1"),
        ("m2", "M2"),
        ("m3", "M3"),
    ),
)
