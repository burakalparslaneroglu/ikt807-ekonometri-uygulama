"""Konu 10 uygulama laboratuvarı: aynı katsayı için analitik ve bootstrap belirsizliği.

Ders notları §10.17'nin (Adım 1–5) birebir karşılığıdır. Veri: Hansen'in cps09mar dosyası, Konu 1'deki
analiz örneklemi (50.742 tam zamanlı çalışan). Model Konu 7'deki OLS modeliyle aynıdır:
log(wage) ~ education + experience + experience²/100 + female, HC1.

Bootstrap: 1.000 pairs tekrarı, ``np.random.default_rng(807)``; her tekrarda ``rng.integers(0, n, n)`` ile
satırlar yerine koyarak çekilir ve katsayılar ``numpy.linalg.lstsq`` ile çözülür (notlardaki kodun aynısı).
Uygulama ve üretilen Python betiği notlardaki çekilişi birebir yeniden üretir; R ve Stata aynı dağılımdan
farklı çekiliş yapar ve Monte Carlo toleransıyla denetlenir.
"""

from __future__ import annotations

from core.labs import expr as E
from core.labs.konu01 import KONU01_LAB
from core.labs.spec import (
    BOOT,
    OLS,
    Bootstrap,
    Check,
    CoefTarget,
    Histogram,
    LabSpec,
    LabStep,
    NoteRef,
    ReproClass,
    ScalarTarget,
    StatTarget,
)

SECTION = "10.17"
FRAME = "cps"
REGRESSORS = ("education", "experience", "experience2_100", "female")
REPS = 1000
SEED = 807
ESTIMATE = 0.1148
SE_TOLERANCE = 1.5e-4
"""R ve Stata için Monte Carlo toleransı. Bootstrap SH'sinin Monte Carlo standart hatası yaklaşık SH/√(2B) ≈
2,5·10⁻⁵'tir; R ve Stata tek bir çekiliştir ve notlardaki (Python) çekilişten bu mertebenin birkaç katı
uzaklaşabilir (R: 0,00103). Tolerans yaklaşık altı Monte Carlo standart hatasıdır."""
PERCENTILE_TOLERANCE = 5e-4
"""Yüzde 2,5 ve 97,5 yüzdeliklerinin Monte Carlo standart hatası √(p(1−p)/B)/f(q) ≈ 9·10⁻⁵; tolerans yaklaşık
beş katıdır."""

STEPS = (
    LabStep(
        number=1,
        title="Orijinal tahmin",
        note=NoteRef(SECTION, 1),
        explanation=(
            "Bootstrap'ı soyut bir yeniden örnekleme algoritması olarak değil gerçek bir regresyon katsayısında "
            "uyguluyoruz. Hansen'in CPS verisindeki 50.742 gözlemin tamamında\n\n"
            "$$\\log(wage_i)=\\beta_0+\\beta_1 education_i+\\beta_2 experience_i+\\beta_3 experience_i^2/100"
            "+\\beta_4 female_i+u_i$$\n\n"
            "modelindeki eğitim katsayısına odaklanıyoruz; model Konu 7'deki OLS modeliyle aynıdır. Amaç OLS "
            "katsayısını değiştirmek değil, $\\widehat\\beta_1$'in örnekleme belirsizliğini iki yolla ölçmektir: "
            "heteroskedastisiteye dayanıklı (HC1) analitik standart hata ve bootstrap."
        ),
        operations=KONU01_LAB.step(2).operations + (OLS("ols", FRAME, "lwage", REGRESSORS, vcov="HC1"),),
        checks=(
            Check("Analiz örneklemi (N)", StatTarget(FRAME, "lwage", "count"), 50742, decimals=0),
            Check("Eğitim katsayısı", CoefTarget("ols", "education"), ESTIMATE, decimals=4),
            Check("Eğitim katsayısının HC1 SH'si", CoefTarget("ols", "education", "se"), 0.00107, decimals=5),
        ),
        reproducibility=ReproClass.EXACT,
        takeaway=(
            "Eğitim katsayısı 0,1148, HC1 standart hatası 0,00107. Bu gözlemsel bir ücret ilişkisidir; bootstrap "
            "uygulanması katsayının nedensel yorumunu değiştirmez."
        ),
    ),
    LabStep(
        number=2,
        title="Pairs bootstrap ile 1.000 yeniden örnekleme",
        note=NoteRef(SECTION, 2, ("Tablo 10.3", "Şekil 10.3")),
        explanation=(
            "Her bootstrap tekrarında 50.742 gözlem satırı yerine koyarak yeniden örneklenir ve aynı regresyon baştan "
            "tahmin edilir. $B=1.000$ tekrarda elde edilen eğitim katsayılarının standart sapması bootstrap standart "
            "hatasıdır:\n\n"
            "$$\\widehat{se}_{boot}=\\Bigl\\{\\frac{1}{B-1}\\sum_{b=1}^B\\bigl(\\widehat\\beta_{1b}^*"
            "-\\bar\\beta_1^*\\bigr)^2\\Bigr\\}^{1/2}.$$\n\n"
            "Percentile güven aralığı bootstrap katsayılarının yüzde 2,5 ve 97,5 yüzdelikleridir. Rastgele sayı üreteci "
            "tohumu 807'dir; her tekrarda önce satır numaraları çekilir, sonra katsayılar tasarım matrisiyle çözülür."
        ),
        operations=(
            Bootstrap(
                "ols", FRAME, REPS, SEED, (("egitim", E.coef(BOOT, "education")),), "boot",
                "CPS eğitim katsayısı için 1.000 tekrarlı pairs bootstrap",
            ),
            Histogram(
                "boot", (("egitim", "Pairs bootstrap eğitim katsayısı"),), 0.110, 0.120,
                ((ESTIMATE, "Orijinal OLS tahmini (0,1148)"),), "Bootstrap eğitim katsayısı",
                "CPS: pairs bootstrap dağılımı (B = 1.000)",
            ),
        ),
        checks=(
            Check("Pairs bootstrap SH", ScalarTarget("boot_egitim_se"), 0.00109, decimals=5,
                  mc_tolerance=SE_TOLERANCE),
            Check("Bootstrap percentile alt sınır", ScalarTarget("boot_egitim_lo"), 0.1127, decimals=4,
                  mc_tolerance=PERCENTILE_TOLERANCE),
            Check("Bootstrap percentile üst sınır", ScalarTarget("boot_egitim_hi"), 0.1169, decimals=4,
                  mc_tolerance=PERCENTILE_TOLERANCE),
        ),
        reproducibility=ReproClass.DISTRIBUTIONAL,
        takeaway=(
            "Bootstrap standart hatası 0,00109, HC1 standart hatası 0,00107: bu örnekte analitik dayanıklı standart hata "
            "ile bootstrap neredeyse aynıdır. Percentile %95 aralığı [0,1127; 0,1169]. Bu yakınlık yararlı bir sağlamlık "
            "bulgusudur; ama \"iki yöntem aynı sonucu verdi, model doğrudur\" anlamına gelmez. İki yöntem de aynı "
            "regresyon spesifikasyonunu ve aynı tanımlama varsayımlarını temel alır."
        ),
        code_note=(
            "Regresyon her tekrarda doğrudan tasarım matrisiyle çözülür (`np.linalg.lstsq`, R'de `lm.fit`); 50.742 "
            "gözlemde 1.000 tekrar birkaç saniye sürer. Formül arayüzüyle aynı katsayılar elde edilir, ama her tekrarda "
            "formülü yeniden işlemek hesabı belirgin biçimde yavaşlatır. Python betiği uygulamadaki çekilişin aynısını "
            "yapar ve notlardaki sayıları birebir verir. R (`set.seed(807)`) ve Stata (`set seed 807`) başka rastgele "
            "sayı üreteci kullanır; betik bootstrap SH'sini ±0,00015, percentile sınırlarını ±0,0005 Monte Carlo "
            "toleransıyla denetler (R'de SH 0,00103, sınırlar 0,1127 ve 0,1168 çıkar)."
        ),
    ),
    LabStep(
        number=3,
        title="Bootstrap kodunda yeniden örnekleme birimini açıkça görmek",
        note=NoteRef(SECTION, 3),
        explanation=(
            "Adım 2'nin kodunda yeniden örnekleme birimi tek bir satırdır: Python'da "
            "`i = rng.integers(0, len(y_b), len(y_b))`, R'de `sample.int(nrow(X_b), nrow(X_b), replace = TRUE)`, "
            "Stata'da `bsample`. Bu satır bireysel gözlemleri yeniden örnekler, çünkü CPS uygulamasını bağımsız bir "
            "yatay kesit örneği olarak ele alıyoruz.\n\n"
            "Veri okul, firma, hane veya bölge içinde kümeli olsaydı bireyleri tek tek yeniden örneklemek bağımlılık "
            "yapısını bozabilirdi. O durumda yeniden örnekleme birimi küme olmalıdır: kümeler yerine koyarak çekilir "
            "ve seçilen kümenin bütün gözlemleri birlikte gelir (Stata'da `bsample, cluster(küme)`)."
        ),
        takeaway=(
            "Yeniden örnekleme birimi, verinin hangi bileşenlerinin bağımsız kabul edildiğinin kod karşılığıdır. "
            "Kümeli yapıda gözlem ve küme bootstrap'ının nasıl ayrıştığını Sezgi sekmesinin 3. deneyinde görebilirsiniz."
        ),
    ),
    LabStep(
        number=4,
        title="Bootstrap tablosunu okurken dört soru",
        note=NoteRef(SECTION, 4),
        explanation=(
            "1. **Ne yeniden örneklenmiş?** Satırlar mı, kümeler mi, artıklar mı?\n"
            "2. **Kaç tekrar yapılmış?** Nihai sonuç için Monte Carlo gürültüsü yeterince küçük mü?\n"
            "3. **Hangi güven aralığı?** Normal, percentile, basic, BCa veya percentile-$t$ mi?\n"
            "4. **Bootstrap hangi varsayımı taklit ediyor?** Bağımsız örnekleme mi, heteroskedastisite mi, küme "
            "bağımlılığı mı?"
        ),
        takeaway=(
            "Aynı \"bootstrap\" etiketi çok farklı yeniden örnekleme şemalarını kapsar. Makalede yalnız \"standart "
            "hatalar bootstrap ile hesaplandı\" ifadesi çoğu zaman yetersizdir. Bu laboratuvarın cevapları: satırlar; "
            "B = 1.000; percentile; bağımsız gözlemler (heteroskedastisite serbest)."
        ),
    ),
    LabStep(
        number=5,
        title="Bootstrap'ın çözemediği problemi görünür kılmak",
        note=NoteRef(SECTION, 5),
        explanation=(
            "Eğitim katsayısında gözlenmeyen yetenek nedeniyle içsellik varsa pairs bootstrap içsel regresyonu 1.000 kez "
            "yeniden tahmin eder. Örnekleme dağılımını daha iyi yaklaşıklar; yanlış hedefi doğru hedefe dönüştürmez. "
            "Aynı mantık zayıf araç veya yanlış kurulmuş bir RDD için de geçerlidir: bootstrap belirsizliği yeniden "
            "hesaplar, tanımlama sorununu onarmaz."
        ),
        takeaway=(
            "Tez yazımında örnek raporlama: \"Eğitim katsayısının HC1 standart hatası 0,00107, 1.000 tekrarlı pairs "
            "bootstrap standart hatası 0,00109'dur. Bootstrap percentile yüzde 95 aralığı [0,1127; 0,1169]'dur. İki "
            "çıkarım yaklaşımının benzer sonuç vermesi örnekleme belirsizliği açısından bulgunun istikrarlı olduğunu "
            "göstermektedir; sonuç gözlemsel tanımlama sınırlarını değiştirmemektedir.\""
        ),
    ),
)


KONU10_LAB = LabSpec(
    topic_key="konu10",
    title="Aynı Katsayı İçin Analitik ve Bootstrap Belirsizliği",
    dataset="cps09mar",
    note_section=SECTION,
    steps=STEPS,
    labels=KONU01_LAB.labels + (("female", "Kadın"), ("ols", "OLS (HC1)")),
    consistency_notes=(
        "Notlardaki laboratuvar 5.000 gözlemlik rastgele bir alt örneklem kullanıyordu; alt örneklemi çeken kod ve "
        "tohum verilmediği için Python, R ve Stata aynı örneklemi kuramıyordu. Laboratuvar Hansen'in tam örneklemiyle "
        "(50.742 gözlem) ve açık tohumla (807) yeniden kuruldu; notlar ve sunum güncellendi.",
    ),
)
