"""Konu 7 uygulama laboratuvarı: ortalama katsayıdan dağılımsal profile.

Ders notları §7.16'nın (Adım 1–4 ve §7.16.5–6) birebir karşılığıdır; Adım 1 aynı zamanda
§7.10'daki Tablo 7.1'i yeniden üretir. Veri: Hansen'in cps09mar dosyası, Konu 1'deki analiz
örneklemi (50.742 tam zamanlı çalışan). Her ``Check`` notlarda basılı bir sayıdır.

Kantil regresyon katsayıları doğrusal programlamanın kesin çözümüdür (R ``rq``, Stata ``qreg``
ile aynı). Standart hatalar Hendricks–Koenker (1992) sandviçidir (R ``summary.rq(se = "nid")``).
Kantiller arası fark testi iki tahminin ortak asimptotik kovaryansını kullanır (Koenker 2005, §3.2).
"""

from __future__ import annotations

from core.labs import expr as E
from core.labs.konu01 import KONU01_LAB
from core.labs.spec import (
    OLS,
    Check,
    CoefficientProfile,
    CoefTarget,
    EffectTable,
    LabSpec,
    LabStep,
    NoteRef,
    QuantileDifference,
    QuantileRegression,
    ReproClass,
    Scalar,
    ScalarTarget,
    StatTarget,
)

SECTION = "7.16"
FRAME = "cps"
REGRESSORS = ("education", "female", "experience", "experience2_100")
QUANTILES = ((0.10, "q10"), (0.25, "q25"), (0.50, "q50"), (0.75, "q75"), (0.90, "q90"))

# Tablo 7.2 (§7.16.1): (katsayı, SH) eğitim ve kadın için; kantillerde Hendricks–Koenker SH, OLS'de HC1.
TABLE_72 = {
    "q10": ((0.1036, 0.0017), (-0.1969, 0.0095)),
    "q25": ((0.1095, 0.0011), (-0.2301, 0.0061)),
    "q50": ((0.1154, 0.0009), (-0.2592, 0.0048)),
    "q75": ((0.1193, 0.0010), (-0.2904, 0.0055)),
    "q90": ((0.1226, 0.0014), (-0.3134, 0.0079)),
    "ols": ((0.1148, 0.0011), (-0.2596, 0.0051)),
}
# Tablo 7.1 (§7.10): deneyim ve deneyim²/100 katsayıları.
TABLE_71 = {
    "q10": (0.0368, -0.0650),
    "q25": (0.0338, -0.0559),
    "q50": (0.0332, -0.0513),
    "q75": (0.0350, -0.0512),
    "q90": (0.0378, -0.0542),
    "ols": (0.0356, -0.0563),
}
_MODEL_LABELS = {"q10": "τ = 0,10", "q25": "τ = 0,25", "q50": "τ = 0,50", "q75": "τ = 0,75", "q90": "τ = 0,90",
                 "ols": "OLS"}
_MODELS = ("q10", "q25", "q50", "q75", "q90", "ols")


def _coef(model: str, term: str, expected: float, label: str, quantity: str = "coef") -> Check:
    return Check(label, CoefTarget(model, term, quantity), expected)


def _table_checks() -> tuple[Check, ...]:
    checks: list[Check] = []
    for model in _MODELS:
        (education, education_se), (female, female_se) = TABLE_72[model]
        name = _MODEL_LABELS[model]
        checks += [
            _coef(model, "education", education, f"{name}: eğitim"),
            _coef(model, "education", education_se, f"{name}: eğitim SH", "se"),
            _coef(model, "female", female, f"{name}: kadın"),
            _coef(model, "female", female_se, f"{name}: kadın SH", "se"),
        ]
    for model in _MODELS:
        experience, square = TABLE_71[model]
        name = _MODEL_LABELS[model]
        checks += [
            _coef(model, "experience", experience, f"{name}: deneyim (Tablo 7.1)"),
            _coef(model, "experience2_100", square, f"{name}: deneyim²/100 (Tablo 7.1)"),
        ]
    return tuple(checks)


def _rows(term: str) -> tuple[tuple[str, str, str], ...]:
    return tuple((_MODEL_LABELS[model], model, term) for model in _MODELS)


def _percent(model: str) -> E.Expr:
    return E.mul(100, E.sub(E.exp(E.coef(model, "education")), 1))


STEPS = (
    LabStep(
        number=1,
        title="Kantil katsayılarını tek tek değil profil olarak okumak",
        note=NoteRef(SECTION, 1, ("Tablo 7.2", "Şekil 7.6", "Tablo 7.1")),
        explanation=(
            "Konu 1'deki 50.742 gözlemlik CPS örnekleminde aynı spesifikasyonu beş kantilde tahmin ediyoruz:\n\n"
            "$$Q_\\tau[\\log(wage_i)\\mid X_i]=\\beta_{0\\tau}+\\beta_{1\\tau}education_i+\\beta_{2\\tau}female_i"
            "+\\beta_{3\\tau}experience_i+\\beta_{4\\tau}experience_i^2/100,$$\n\n"
            "$\\tau\\in\\{0{,}10;\\,0{,}25;\\,0{,}50;\\,0{,}75;\\,0{,}90\\}$. Karşılaştırma için aynı spesifikasyonun OLS "
            "tahmini (HC1) de verilir. Kantil tahmini $\\sum_i\\rho_\\tau(Y_i-X_i'b)$ kontrol kaybını en küçük yapan "
            "$b$'dir; bu bir doğrusal programlama problemidir ve kesin çözümü hesaplanır.\n\n"
            "Standart hatalar **Hendricks–Koenker** sandviçidir: $\\hat V_\\tau=\\tau(1-\\tau)H^{-1}X'XH^{-1}$, "
            "$H=\\sum_i\\hat f_iX_iX_i'$. Her gözlemin koşullu yoğunluğu $\\hat f_i$, aynı modelin $\\tau\\pm h$ "
            "kantillerindeki tahmin farkından bulunur: iki kantil doğrusu $X_i$'de birbirine yakınsa, o noktada "
            "yoğunluk yüksektir."
        ),
        operations=KONU01_LAB.step(2).operations
        + tuple(QuantileRegression(name, FRAME, "lwage", REGRESSORS, tau, vcov="nid") for tau, name in QUANTILES)
        + (
            OLS("ols", FRAME, "lwage", REGRESSORS, vcov="HC1"),
            EffectTable(_rows("education"), "egitim", title="Eğitim katsayısı (Tablo 7.2)",
                        se_label="SH (kantilde HK, OLS'de HC1)"),
            EffectTable(_rows("female"), "kadin", title="Kadın göstergesinin katsayısı (Tablo 7.2)",
                        se_label="SH (kantilde HK, OLS'de HC1)"),
            CoefficientProfile(
                QUANTILES, "education", "Kantil τ", "Eğitim katsayısı",
                "CPS: eğitim katsayısının dağılımsal profili", reference="ols",
            ),
            Scalar("yuzde_q10", _percent("q10"), "τ = 0,10: eğitimin kesin yüzde karşılığı", decimals=1),
            Scalar("yuzde_q90", _percent("q90"), "τ = 0,90: eğitimin kesin yüzde karşılığı", decimals=1),
        ),
        checks=(
            Check("Analiz örneklemi (N)", StatTarget(FRAME, "lwage", "count"), 50742, decimals=0),
            *_table_checks(),
            Check("τ = 0,10 kesin yüzde etki", ScalarTarget("yuzde_q10"), 10.9, decimals=1),
            Check("τ = 0,90 kesin yüzde etki", ScalarTarget("yuzde_q90"), 13.0, decimals=1),
        ),
        reproducibility=ReproClass.CONVENTION,
        takeaway=(
            "Eğitim katsayısı alt kantilde 0,1036, üst kantilde 0,1226: tam dönüşümle bir ek eğitim yılı koşullu log "
            "ücret dağılımının 0,10 kantilinde yaklaşık %10,9, 0,90 kantilinde %13,0 daha yüksek ücretle ilişkili. "
            "Kadın katsayısı mutlak değerce büyür (−0,197'den −0,313'e): koşullu cinsiyet farkı üst kantillerde daha "
            "geniş. OLS'in tek eğimi (0,1148) bu profili bir sayıya indirger. Standart hatalar uçlarda büyür: "
            "dağılımın kuyruklarında yoğunluk düşük olduğu için aynı örneklem daha az bilgi taşır."
        ),
        code_note=(
            "Kantil katsayıları doğrusal programlamanın kesin çözümüdür: Python'da `scipy.optimize.linprog` (HiGHS, "
            "dual problem), R'de `quantreg::rq` (Barrodale–Roberts), Stata'da `qreg`. statsmodels `QuantReg` "
            "yinelemeli bir yaklaşım kullanır; bu örneklemde τ = 0,90 eğitim katsayısını dördüncü ondalıkta 0,1225 "
            "verir, kesin çözüm 0,1226'dır. Standart hatalar üç dilde aynı formülle: R `summary(m, se = \"nid\")`, "
            "Python ve Stata'da kodda açıkça yazılı Hendricks–Koenker hesabı. statsmodels'in `vcov=\"robust\"` "
            "seçeneği tek bir ortak yoğunluk kullanır ve farklı standart hata verir; Stata'nın varsayılan `qreg` "
            "standart hatası (iid) de farklıdır."
        ),
    ),
    LabStep(
        number=2,
        title="\"Katsayılar farklı\" ile \"fark istatistiksel olarak kanıtlandı\" ayrımı",
        note=NoteRef(SECTION, 2),
        explanation=(
            "İki kantilde katsayıların ayrı ayrı anlamlı olması, aralarındaki farkın anlamlı olduğunu göstermez. "
            "$H_0:\\beta_{0{,}10}=\\beta_{0{,}90}$ için farkın örnekleme dağılımı gerekir. İki tahmin **aynı veriden** "
            "geldiği için bağımsız değildir; ortak asimptotik kovaryansları\n\n"
            "$$\\mathrm{Cov}(\\hat\\beta_{\\tau_1},\\hat\\beta_{\\tau_2})=\\{\\min(\\tau_1,\\tau_2)-\\tau_1\\tau_2\\}"
            "\\,H_{\\tau_1}^{-1}X'XH_{\\tau_2}^{-1}$$\n\n"
            "ile verilir (Hendricks–Koenker yoğunluklarıyla). Farkın varyansı iki varyansın toplamından bu kovaryansın "
            "iki katı çıkarılarak bulunur ve $z=(\\hat\\beta_{0{,}90}-\\hat\\beta_{0{,}10})/SH$ normal dağılımla "
            "değerlendirilir. Aynı test ortak bootstrap ile de yapılabilir (Konu 10)."
        ),
        operations=(
            QuantileDifference("fark_egitim", "q10", "q90", "education",
                               "Eğitim: β̂(0,90) − β̂(0,10)"),
            QuantileDifference("fark_kadin", "q10", "q90", "female",
                               "Kadın: β̂(0,90) − β̂(0,10)"),
        ),
        checks=(
            Check("Eğitim: kantiller arası fark", ScalarTarget("fark_egitim"), 0.0189),
            Check("Eğitim: farkın SH'si", ScalarTarget("fark_egitim_se"), 0.0021),
            Check("Eğitim: z istatistiği", ScalarTarget("fark_egitim_z"), 8.90, decimals=2),
            Check("Kadın: kantiller arası fark", ScalarTarget("fark_kadin"), -0.1165),
            Check("Kadın: farkın SH'si", ScalarTarget("fark_kadin_se"), 0.0116),
            Check("Kadın: z istatistiği", ScalarTarget("fark_kadin_z"), -10.04, decimals=2),
        ),
        reproducibility=ReproClass.CONVENTION,
        takeaway=(
            "Eğitim eğimi 0,90 ile 0,10 kantilleri arasında 0,0189 artıyor (SH 0,0021, z = 8,90); kadın katsayısı "
            "0,1165 daha negatif (SH 0,0116, z = −10,04). Bu örneklemde fark güçlü biçimde anlamlıdır; ama bu sonuç "
            "profilin görsel eğiminden değil, farkın kendi standart hatasından gelir. Güven bantlarının örtüşüp "
            "örtüşmemesi de eşitlik testiyle aynı şey değildir. Tezde \"heterojenlik vardır\" demeden önce bu testi "
            "raporlayın."
        ),
        code_note=(
            "Ortak kovaryans için her modelin $H^{-1}$ ve $X'X$ matrisleri saklanır: R'de `summary(m, se = \"nid\", "
            "covariance = TRUE)` sonucunun `$Hinv` ve `$J` öğeleri, Python'da `KantilSonucu` nesnesinin `Hinv` ve `J` "
            "alanları, Stata'da `hk_sh` programının oluşturduğu `Hinv_model` ve `J_model` matrisleri."
        ),
    ),
    LabStep(
        number=3,
        title="Koşullu kantil heterojenliğini bireysel etki heterojenliğiyle karıştırmamak",
        note=NoteRef(SECTION, 3),
        explanation=(
            "Kantil regresyon katsayısı $\\beta_\\tau$, koşullu sonuç dağılımının $\\tau$ kantilindeki ilişkiyi özetler. "
            "\"Aynı birey eğitim alsaydı ücreti hangi kantilde nasıl değişirdi?\" gibi nedensel bir heterojenlik "
            "yorumu, bireylerin eğitimden sonra da dağılımdaki sıralarını koruması (sıra değişmezliği) gibi ek "
            "yapısal varsayımlar gerektirir. Gözlemsel CPS uygulamasında sonuçlar betimsel koşullu dağılım "
            "ilişkileridir."
        ),
        takeaway=(
            "\"Eğitim, üst gelir grubundaki bireylere daha çok kazandırıyor\" ifadesi standart koşullu kantil "
            "regresyonundan doğrudan çıkmaz. Model yalnız üst koşullu ücret kantilindeki eğimin daha büyük olduğunu "
            "gösterir; bireysel etki heterojenliği daha güçlü tanımlama koşulları ister."
        ),
    ),
    LabStep(
        number=4,
        title="Kod ve raporlama",
        note=NoteRef(SECTION, 4),
        explanation=(
            "Aynı spesifikasyon bir döngüde bütün kantillerde tahmin edilir ve her kantil için katsayı ile standart "
            "hata birlikte saklanır. Adım 1'in kodu bunu üç dilde yapar. Araştırma sorusu dağılımsal heterojenlik "
            "ise kantil seti sonuçlara bakılmadan önceden gerekçelendirilmeli; bütün profil aynı tabloda ve grafikte "
            "gösterilmelidir. Tek bir medyan regresyonu göstermek mümkündür, ama profili gizler."
        ),
        takeaway=(
            "Raporlama cümlesi örneği: \"Eğitim katsayısı koşullu log ücret dağılımının 0,10 kantilinde 0,104, "
            "medyanda 0,115 ve 0,90 kantilinde 0,123'tür. 0,90 ile 0,10 kantilleri arasındaki fark 0,019'dur "
            "(SH 0,002).\""
        ),
    ),
    LabStep(
        number=5,
        title="Hansen'in DDK örneği ve tezinizde kantil regresyon protokolü",
        note=NoteRef("7.16.5", 0, ("§7.16.6",)),
        explanation=(
            "Hansen, Duflo–Dupas–Kremer tracking verisinde kümeli bootstrap standart hatalarıyla "
            "$\\tau=0{,}1;\\,0{,}3;\\,0{,}5;\\,0{,}7;\\,0{,}9$ kantillerini karşılaştırır. Nokta tahminleri üst "
            "kantillerde daha büyük görünse de güven aralıkları genişler ve genel kanıt zayıftır: dağılımın uçlarında "
            "daha dramatik katsayılar görmek, orada daha çok bilgi olduğu anlamına gelmez.\n\n"
            "**Protokol:**\n\n"
            "1. Ortalama yerine neden kantil estimand'ına ihtiyaç duyduğunuzu açıklayın.\n"
            "2. Kantil setini sonuçlara bakarak değil araştırma sorusuna göre belirleyin.\n"
            "3. Her kantilde katsayı ve standart hatayı birlikte raporlayın.\n"
            "4. Kümeli veri varsa gözlem değil küme yeniden örnekleyen bootstrap kullanın.\n"
            "5. Katsayı profillerini grafikleştirin; kantil kesişmesi olup olmadığını kontrol edin.\n"
            "6. Kantil farklarını yorumlayacaksanız doğrudan fark testi veya ortak bootstrap kullanın.\n"
            "7. Nedensel kantil yorumu için gereken ek tanımlama varsayımlarını açıkça yazın."
        ),
        takeaway=(
            "Adım 2'deki fark testi, protokolün 6. maddesinin analitik karşılığıdır. Kümeli veride aynı test için "
            "kümeleri yeniden örnekleyen ortak bootstrap gerekir (Konu 10)."
        ),
    ),
)


KONU07_LAB = LabSpec(
    topic_key="konu07",
    title="Ortalama Katsayıdan Dağılımsal Profile",
    dataset="cps09mar",
    note_section=SECTION,
    steps=STEPS,
    labels=KONU01_LAB.labels
    + (
        ("female", "Kadın"),
        ("q10", "τ = 0,10"),
        ("q25", "τ = 0,25"),
        ("q50", "τ = 0,50"),
        ("q75", "τ = 0,75"),
        ("q90", "τ = 0,90"),
        ("ols", "OLS"),
    ),
)
