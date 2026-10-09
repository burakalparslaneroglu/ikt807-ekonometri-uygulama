"""Konu 7 genel uygulaması: ortalama katsayıdan dağılımsal profile.

Notlardaki §7.16'nın beş adımı aynı numaralarla ve aynı işlemlerle, verisi değiştirilebilir biçimde yazılır (``build``):
beş kantilde kantil regresyon (Hendricks–Koenker standart hatalarıyla) ve OLS, katsayı profili, kesin yüzde etkiler,
kantiller arası fark testi, yorum sınırları, raporlama ve protokol. Alternatif örnek Card (1995) verisidir (Hansen'in
arşivindeki ``Card1995.dta``): 1976 log saatlik ücreti gözlenen 3.010 erkek (Konu 1'in alternatif örneklemi). "Kendi verini yükle" seçeneğinde aynı
adımlar öğrencinin dosyasıyla kurulur. Notlardaki laboratuvar (``core.labs.konu07``) değişmez.

Rollerin notlardaki karşılıkları: sonuç log saatlik ücret, ana açıklayıcı eğitim yılı, iki kategorili grup kadın
göstergesi, karesiyle eklenen sürekli kontrol potansiyel deneyim; ek kontroller isteğe bağlıdır.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import cache

import numpy as np
import pandas as pd

from core.labs import expr as E
from core.labs import kendi_veri as K
from core.labs import ornek_konu01 as K1
from core.labs.ornek import (
    SCALE_MESSAGE,
    Case,
    CustomLab,
    Option,
    Role,
    TopicVariants,
    full_rank,
    katsayi,
    md,
    sayi,
    sayim,
    stable_design,
    with_app_values,
    with_expected,
)
from core.labs.ornek_veri import ucret_verisi
from core.labs.spec import (
    OLS,
    Check,
    CoefficientProfile,
    CoefTarget,
    Derive,
    EffectTable,
    Indicator,
    LabSpec,
    LabStep,
    NoteRef,
    Operation,
    QuantileDifference,
    QuantileRegression,
    ReproClass,
    Scalar,
    ScalarTarget,
    StatTarget,
)

TOPIC = "konu07"
SECTION = "7.16"
SONUC, ACIKLAYICI, KONTROL, GRUP = "sonuc", "aciklayici", "kontrol", "grup"
LOG = "log"
LOG_Y, Z2, G01 = K1.LOG_Y, K1.Z2, K1.G01
"""Kendi verinde türetilen sütunlar (Konu 1 ile aynı adlar; ``kendi_veri.RESERVED_CODES``)."""
QUANTILES = ((0.10, "q10"), (0.25, "q25"), (0.50, "q50"), (0.75, "q75"), (0.90, "q90"))
MODELS = ("q10", "q25", "q50", "q75", "q90", "ols")
MODEL_LABELS = {"q10": "τ = 0,10", "q25": "τ = 0,25", "q50": "τ = 0,50", "q75": "τ = 0,75", "q90": "τ = 0,90",
                "ols": "OLS"}


@dataclass(frozen=True)
class Plan:
    """Rollerin koddaki sütunları ve metinlerdeki adları.

    ``names``: kontrol etiketlerindeki adlar (``x``, ``group``, ``z``, ``z2``); ``log``: sonuç logaritmik mi (kesin yüzde
    etkiler yalnız o zaman hesaplanır). ``decimals``: katsayı kontrollerinin ondalığı; ``term_decimals`` terim başına
    farklıysa (kendi verin: her terimin tipik katsayı büyüklüğüne göre). ``unstable``: katsayıları dillerde aynı
    çıkmayabilecek kantil modelleri (çözüm gösterim basamağında tek değil); ``unstable_se``: standart hatası aynı
    çıkmayabilecek modeller (τ ± h kantillerinde çözüm tek değil); ``fragile``: aynı nedenle karşılaştırılmayan skalerler
    (kesin yüzde etki, kantiller arası fark ya da z). Bu kontroller tanımdan çıkarılır.
    """

    frame: str
    outcome: str
    x: str
    group: str
    z: str
    z2: str
    controls: tuple[str, ...]
    prepare: tuple[Operation, ...]
    labels: dict
    names: dict
    log: bool
    profile_title: str
    profile_label: str
    decimals: int = 4
    term_decimals: tuple[tuple[str, int], ...] = ()
    unstable: frozenset = field(default_factory=frozenset)
    unstable_se: frozenset = field(default_factory=frozenset)
    fragile: frozenset = field(default_factory=frozenset)

    @property
    def regressors(self) -> tuple[str, ...]:
        return (self.x, self.group, self.z, self.z2, *self.controls)

    def places(self, term: str) -> int:
        """Terimin katsayı ve standart hata kontrollerinin ondalığı."""

        return dict(self.term_decimals).get(term, self.decimals)


def _coef(model: str, term: str, label: str, quantity: str = "coef", decimals: int = 4) -> Check:
    return Check(label, CoefTarget(model, term, quantity), 0.0, decimals)


def _percent(model: str, term: str) -> E.Expr:
    return E.mul(100, E.sub(E.exp(E.coef(model, term)), 1))


def _table_checks(plan: Plan) -> tuple[Check, ...]:
    names, places = plan.names, plan.places
    x, g = plan.x, plan.group
    checks: list[Check] = []
    for model in MODELS:
        title = MODEL_LABELS[model]
        if model not in plan.unstable:
            checks += [_coef(model, x, f"{title}: {names['x']}", decimals=places(x)),
                       _coef(model, g, f"{title}: {names['group']}", decimals=places(g))]
            if model not in plan.unstable_se:
                checks += [_coef(model, x, f"{title}: {names['x']} SH", "se", places(x)),
                           _coef(model, g, f"{title}: {names['group']} SH", "se", places(g))]
    for model in MODELS:
        if model in plan.unstable:
            continue
        title = MODEL_LABELS[model]
        checks += [_coef(model, plan.z, f"{title}: {names['z']} (Tablo 7.1)", decimals=places(plan.z)),
                   _coef(model, plan.z2, f"{title}: {names['z2']} (Tablo 7.1)", decimals=places(plan.z2))]
    return tuple(checks)


def _difference_checks(plan: Plan) -> tuple[Check, ...]:
    if {"q10", "q90"} & (plan.unstable | plan.unstable_se):
        return ()
    names = plan.names
    checks: list[Check] = []
    for name, word, term in (("fark_x", names["x"], plan.x), ("fark_grup", names["group"], plan.group)):
        word = word[:1].upper() + word[1:] if word[:1].isalpha() else word
        candidates = (Check(f"{word}: kantiller arası fark", ScalarTarget(name), 0.0, plan.places(term)),
                      Check(f"{word}: farkın SH'si", ScalarTarget(f"{name}_se"), 0.0, plan.places(term)),
                      Check(f"{word}: z istatistiği", ScalarTarget(f"{name}_z"), 0.0, decimals=2))
        checks += [check for check in candidates if check.target.name not in plan.fragile]
    return tuple(checks)


def _steps(plan: Plan, texts: dict) -> tuple[LabStep, ...]:
    f, names = plan.frame, plan.names
    rows = {term: tuple((MODEL_LABELS[model], model, term) for model in MODELS) for term in (plan.x, plan.group)}
    percent: tuple[Operation, ...] = ()
    percent_checks: tuple[Check, ...] = ()
    if plan.log:
        percent = (Scalar("yuzde_q10", _percent("q10", plan.x), f"τ = 0,10: {names['x']} kesin yüzde karşılığı",
                          decimals=1),
                   Scalar("yuzde_q90", _percent("q90", plan.x), f"τ = 0,90: {names['x']} kesin yüzde karşılığı",
                          decimals=1))
        percent_checks = tuple(Check(f"τ = 0,{tau} kesin yüzde etki", ScalarTarget(f"yuzde_q{tau}"), 0.0, decimals=1)
                               for tau in ("10", "90")
                               if f"q{tau}" not in plan.unstable and f"yuzde_q{tau}" not in plan.fragile)
    return (
        LabStep(
            number=1,
            title="Kantil katsayılarını tek tek değil profil olarak okumak",
            note=NoteRef(SECTION, 1, ("Tablo 7.2", "Şekil 7.6", "Tablo 7.1")),
            explanation=texts[1][0],
            operations=(
                *plan.prepare,
                *(QuantileRegression(name, f, plan.outcome, plan.regressors, tau, vcov="nid") for tau, name in QUANTILES),
                OLS("ols", f, plan.outcome, plan.regressors, vcov="HC1"),
                EffectTable(rows[plan.x], "egim", title=f"{names['title_x']} katsayısı (Tablo 7.2)",
                            se_label="SH (kantilde HK, OLS'de HC1)"),
                EffectTable(rows[plan.group], "grup", title=f"{names['title_group']} katsayısı (Tablo 7.2)",
                            se_label="SH (kantilde HK, OLS'de HC1)"),
                CoefficientProfile(QUANTILES, plan.x, "Kantil τ", plan.profile_label, plan.profile_title,
                                   reference="ols"),
                *percent,
            ),
            checks=(
                Check("Analiz örneklemi (N)", StatTarget(f, plan.outcome, "count"), 0.0, decimals=0),
                *_table_checks(plan),
                *percent_checks,
            ),
            reproducibility=ReproClass.CONVENTION,
            takeaway=texts[1][1],
            code_note=texts.get("kod1", CODE_NOTE_1),
            note_for=texts.get((1, "not")),
        ),
        LabStep(
            number=2,
            title="\"Katsayılar farklı\" ile \"fark istatistiksel olarak kanıtlandı\" ayrımı",
            note=NoteRef(SECTION, 2),
            explanation=texts[2][0],
            operations=(
                QuantileDifference("fark_x", "q10", "q90", plan.x, f"{names['title_x']}: β̂(0,90) − β̂(0,10)"),
                QuantileDifference("fark_grup", "q10", "q90", plan.group, f"{names['title_group']}: β̂(0,90) − β̂(0,10)"),
            ),
            checks=_difference_checks(plan),
            reproducibility=ReproClass.CONVENTION,
            takeaway=texts[2][1],
            code_note=texts.get("kod2", CODE_NOTE_2),
            note_for=texts.get((2, "not")),
        ),
        LabStep(
            number=3,
            title="Koşullu kantil heterojenliğini bireysel tedavi heterojenliğiyle karıştırmamak",
            note=NoteRef(SECTION, 3),
            explanation=texts[3][0],
            takeaway=texts[3][1],
        ),
        LabStep(
            number=4,
            title="Kod ve raporlama",
            note=NoteRef(SECTION, 4),
            explanation=texts.get("rapor", REPORTING),
            takeaway=texts[4][1],
            note_for=texts.get((4, "not")),
        ),
        LabStep(
            number=5,
            title="Hansen'in DDK örneği ve tezinizde kantil regresyon protokolü",
            note=NoteRef("7.16.5", 0, ("§7.16.6",)),
            explanation=PROTOCOL,
            takeaway=texts[5][1],
        ),
    )


def _spec(plan: Plan, texts: dict, *, source: str, title: str, dataset: str) -> LabSpec:
    labels = {**plan.labels, **MODEL_LABELS}
    return LabSpec(topic_key=TOPIC, title=title, dataset=dataset, note_section=SECTION, steps=_steps(plan, texts),
                   labels=tuple(labels.items()), source=source)


# --- Ortak metinler -------------------------------------------------------------------------------------------

ESTIMATOR = (
    "Kantil tahmini $\\sum_i\\rho_\\tau(Y_i-X_i'b)$ kontrol kaybını en küçük yapan $b$'dir; bu bir doğrusal programlama "
    "problemidir ve kesin çözümü hesaplanır.\n\n"
    "Standart hatalar **Hendricks–Koenker** sandviçidir: $\\widehat{\\boldsymbol V}_{\\hat\\beta_\\tau}=\\tau(1-\\tau)\\boldsymbol H_\\tau^{-1}\\boldsymbol X'\\boldsymbol X\\boldsymbol H_\\tau^{-1}$, "
    "$\\boldsymbol H_\\tau=\\sum_i\\hat f_iX_iX_i'$. Her gözlemin koşullu yoğunluğu $\\hat f_i$, aynı modelin $\\tau\\pm h$ kantillerindeki "
    "tahmin farkından bulunur: iki kantil doğrusu $X_i$'de birbirine yakınsa, o noktada yoğunluk yüksektir."
)
DIFFERENCE = (
    "Aynı regresörün katsayısı için $j$ indeksini sabit tutuyoruz (eğitimde $j=1$). İki kantilde katsayıların ayrı ayrı anlamlı olması, aralarındaki farkın anlamlı olduğunu göstermez. "
    "$H_0:\\beta_{j,0{,}10}=\\beta_{j,0{,}90}$ için farkın örnekleme dağılımı gerekir. İki tahmin **aynı veriden** geldiği "
    "için bağımsız değildir; ortak asimptotik kovaryansları\n\n"
    "$$\\widehat{\\operatorname{Cov}}(\\hat\\beta_{\\tau_1},\\hat\\beta_{\\tau_2})=\\{\\min(\\tau_1,\\tau_2)-\\tau_1\\tau_2\\}"
    "\\,\\boldsymbol H_{\\tau_1}^{-1}\\boldsymbol X'\\boldsymbol X\\boldsymbol H_{\\tau_2}^{-1}$$\n\n"
    "ile verilir (Hendricks–Koenker yoğunluklarıyla). Farkın varyansı iki varyansın toplamından bu kovaryansın iki katı "
    "çıkarılarak bulunur ve $z=(\\hat\\beta_{j,0{,}90}-\\hat\\beta_{j,0{,}10})/SH$ normal dağılımla değerlendirilir. Aynı test "
    "ortak bootstrap ile de yapılabilir (Konu 10)."
)
REPORTING = (
    "Aynı spesifikasyon bir döngüde bütün kantillerde tahmin edilir ve her kantil için katsayı ile standart hata birlikte "
    "saklanır. Adım 1'in kodu bunu üç dilde yapar. Araştırma sorusu dağılımsal heterojenlik ise kantil seti sonuçlara "
    "bakılmadan önce gerekçelendirilmeli; bütün profil aynı tabloda ve grafikte gösterilmelidir. Tek bir medyan "
    "regresyonu göstermek mümkündür, ama profili gizler."
)
PROTOCOL = (
    "Hansen, Duflo–Dupas–Kremer tracking verisinde kümeli bootstrap standart hatalarıyla "
    "$\\tau=0{,}1;\\,0{,}3;\\,0{,}5;\\,0{,}7;\\,0{,}9$ kantillerini karşılaştırır. Nokta tahminleri üst kantillerde daha "
    "büyük görünse de güven aralıkları genişler ve genel kanıt zayıftır: dağılımın uçlarında daha dramatik katsayılar "
    "görmek, orada daha çok bilgi olduğu anlamına gelmez.\n\n"
    "**Protokol:**\n\n"
    "1. Ortalama yerine neden kantil estimand'ına ihtiyaç duyduğunuzu açıklayın.\n"
    "2. Kantil setini sonuçlara bakarak değil araştırma sorusuna göre belirleyin.\n"
    "3. Her kantilde katsayı ve standart hatayı birlikte raporlayın.\n"
    "4. Kümeli veri varsa gözlem değil küme yeniden örnekleyen bootstrap kullanın.\n"
    "5. Katsayı profillerini grafikleştirin; kantil kesişmesi olup olmadığını kontrol edin.\n"
    "6. Kantil farklarını yorumlayacaksanız doğrudan fark testi veya ortak bootstrap kullanın.\n"
    "7. Nedensel kantil yorumu için gereken ek tanımlama varsayımlarını açıkça yazın."
)
PROTOCOL_RULE = (
    "Adım 2'deki fark testi, protokolün 6. maddesinin analitik karşılığıdır. Kümeli veride aynı test için kümeleri "
    "yeniden örnekleyen ortak bootstrap gerekir (Konu 10)."
)
CODE_NOTE_1 = (
    "Kantil katsayıları doğrusal programlamanın kesin çözümüdür: Python'da `scipy.optimize.linprog` (HiGHS, dual "
    "problem), R'de `quantreg::rq` (Barrodale–Roberts), Stata'da `qreg`. Standart hatalar üç dilde aynı formülle: R "
    "`summary(m, se = \"nid\")`, Python ve Stata'da kodda açıkça yazılı Hendricks–Koenker hesabı. statsmodels "
    "`QuantReg` yinelemeli bir yaklaşım kullanır ve kesin çözümden dördüncü ondalıkta ayrılabilir; Stata'nın varsayılan "
    "`qreg` standart hatası (iid) de farklıdır."
)
CODE_NOTE_2 = (
    "Ortak kovaryans için her modelin $\\boldsymbol H_\\tau^{-1}$ ve $\\boldsymbol X'\\boldsymbol X$ matrisleri saklanır: R'de `summary(m, se = \"nid\", "
    "covariance = TRUE)` sonucunun `$Hinv` ve `$J` öğeleri, Python'da `KantilSonucu` nesnesinin `Hinv` ve `J` alanları, "
    "Stata'da `hk_sh` programının oluşturduğu `Hinv_model` ve `J_model` matrisleri."
)
CONDITIONAL = (
    "Kantil regresyon katsayısı $\\beta_\\tau$, koşullu sonuç dağılımının $\\tau$ kantilindeki ilişkiyi özetler. \"Aynı "
    "birey bu özelliğe sahip olsaydı sonucu hangi kantilde nasıl değişirdi?\" gibi nedensel bir heterojenlik yorumu, "
    "bireylerin dağılımdaki sıralarını koruması (sıra değişmezliği) gibi ek yapısal varsayımlar gerektirir. Gözlemsel "
    "veride sonuçlar betimsel koşullu dağılım ilişkileridir."
)


# --- Alternatif örnek: Card (1995) ----------------------------------------------------------------------------

ALT_DATA = K1.ALT_DATA
ALT_TITLE = "Card (1995) Verisiyle Ücret Dağılımında Eğitim ve Irk Farkı"


def alternative_plan() -> Plan:
    base = K1.alternative_plan()
    return Plan(
        frame=base.frame, outcome="lwage", x="ed76", group="black", z="experience", z2="experience2_100", controls=(),
        prepare=base.prepare,
        labels={
            "(sabit)": "Sabit",
            "hrwage": "Saatlik ücret (dolar)",
            "lwage": "Log saatlik ücret",
            "ed76": "Eğitim yılı",
            "black": "Siyahi göstergesi",
            "experience": "Potansiyel deneyim",
            "experience2_100": "Deneyim²/100",
        },
        names={"x": "eğitim", "group": "siyahi", "z": "deneyim", "z2": "deneyim²/100", "title_x": "Eğitim",
               "title_group": "Siyahi göstergesinin"},
        log=True,
        profile_title="Card (1995): eğitim katsayısının dağılımsal profili",
        profile_label="Eğitim katsayısı",
    )


ALT_TEXTS = {
    1: (
        "Konu 1'in alternatif örneğindeki Card (1995) verisi: Ulusal Boylamsal Genç Erkekler Araştırması'ndan (NLSYM) "
        "1976 log saatlik ücreti (`lwage76`) gözlenen 3.010 erkek (saatlik ücreti 1 doların altında olan 7 gözlem "
        "dışarıda). Aynı spesifikasyonu beş kantilde tahmin ediyoruz:\n\n"
        "$$Q_\\tau[\\log(wage_i)\\mid X_i]=\\beta_{0,\\tau}+\\beta_{1,\\tau}ed76_i+\\beta_{2,\\tau}black_i"
        "+\\beta_{3,\\tau}experience_i+\\beta_{4,\\tau}experience_i^2/100,$$\n\n"
        "$\\tau\\in\\{0{,}10;\\,0{,}25;\\,0{,}50;\\,0{,}75;\\,0{,}90\\}$; $ed76$ 1976'da tamamlanmış eğitim yılı, "
        "$experience_i=age76_i-ed76_i-6$ potansiyel deneyimdir. Örneklem yalnız erkeklerden oluştuğu için "
        "notlardaki kadın göstergesinin yerinde siyahi göstergesi vardır. Karşılaştırma için aynı spesifikasyonun OLS "
        "tahmini (HC1) de verilir; OLS katsayıları Konu 1'in alternatif örneğindeki Model (3)'ün katsayılarıdır. "
        + ESTIMATOR
    ),
    2: DIFFERENCE,
    3: (
        "Kantil regresyon katsayısı $\\beta_\\tau$, koşullu ücret dağılımının $\\tau$ kantilindeki ilişkiyi özetler. "
        "\"Aynı kişi bir yıl daha fazla eğitim alsaydı ücreti hangi kantilde nasıl değişirdi?\" gibi nedensel bir "
        "heterojenlik yorumu, bireylerin eğitimden sonra da dağılımdaki sıralarını koruması (sıra değişmezliği) gibi ek "
        "yapısal varsayımlar gerektirir. Card (1995) bu veride eğitimin içsel olabileceğini tartışır ve koleje yakınlığı "
        "araç olarak kullanır (Konu 4); burada sonuçlar betimsel koşullu dağılım ilişkileridir."
    ),
}
"""Adım açıklamaları; Adım 4 ve 5'in açıklaması notlarla ortaktır (``REPORTING``, ``PROTOCOL``)."""

ALT_TAKEAWAYS = {
    1: (
        "Eğitim katsayısı 0,10 kantilinde 0,0774, medyanda 0,0853, 0,90 kantilinde 0,0749: profil ters U biçimlidir, "
        "notlardaki CPS örneğindeki gibi kantille artmaz. Kesin dönüşümle bir ek eğitim yılı, koşullu log ücret "
        "dağılımının 0,10 kantilinde yaklaşık %8,0, 0,90 kantilinde %7,8 daha yüksek ücretle ilişkilidir. Siyahi "
        "katsayısı 0,25 kantilinden sonra mutlak değerce küçülür (−0,2757'den 0,90'da −0,1915'e): aynı eğitim ve "
        "deneyimdeki siyahi ve siyahi olmayan erkeklerin koşullu ücret dağılımları üst kantillerde birbirine daha "
        "yakındır. OLS'in tek eğimi (eğitim 0,0818, siyahi −0,2319) bu profili bir sayıya indirger. Standart hatalar "
        "uçlarda büyür (eğitim: medyanda 0,0042, 0,10'da 0,0060, 0,90'da 0,0064): dağılımın kuyruklarında yoğunluk düşük "
        "olduğu için aynı örneklem daha az bilgi taşır."
    ),
    2: (
        "Eğitim eğimi 0,90 ile 0,10 kantilleri arasında yalnız −0,0025 farklı (SH 0,0083, z = −0,30): ters U biçimi iki "
        "ucu birbirine yaklaştırır; bu iki kantil arasındaki fark istatistiksel olarak anlamlı değildir, veri iki eğimi "
        "ayırt etmiyor (bu, eğimlerin eşit olduğunu kanıtlamaz). Siyahi katsayısı 0,90 kantilinde 0,0688 daha az "
        "negatif (SH 0,0367, z = 1,88): nokta tahminleri belirgin biçimde farklı görünse de fark %5 düzeyinde anlamlı "
        "değildir (%10 düzeyinde anlamlıdır). 3.010 gözlemlik bu örneklemde kantiller arası farklar, notlardaki 50.742 "
        "gözlemlik CPS örnekleminden çok daha az kesin ölçülür. Ters U biçimini sınamak için ortadaki kantili uçlarla "
        "karşılaştıran ayrı testler gerekir. Tezde \"heterojenlik vardır\" demeden önce bu testleri raporlayın."
    ),
    3: (
        "\"Siyahi işçilerin dezavantajı yüksek ücretli gruplarda daha küçük\" ifadesi koşullu kantil regresyonundan "
        "doğrudan çıkmaz: model yalnız iki grubun koşullu ücret dağılımlarının üst kantillerde birbirine daha yakın "
        "olduğunu gösterir (ve Adım 2'ye göre bu fark %5 düzeyinde anlamlı değildir). Bireylerin dağılımdaki yeri ve "
        "nedensel yorum ek varsayımlar ister."
    ),
    4: (
        "Raporlama cümlesi örneği: \"Eğitim katsayısı koşullu log ücret dağılımının 0,10 kantilinde 0,077, medyanda "
        "0,085 ve 0,90 kantilinde 0,075'tir. 0,90 ile 0,10 kantilleri arasındaki fark −0,002'dir (SH 0,008) ve "
        "istatistiksel olarak anlamlı değildir.\""
    ),
    5: PROTOCOL_RULE,
}

ALT_EXPECTED = {
    (1, "Analiz örneklemi (N)"): 3010,
    **{(1, f"{MODEL_LABELS[model]}: {word}{suffix}"): value
       for model, values in (
           ("q10", (0.0774, 0.0060, -0.2604, 0.0275)), ("q25", (0.0817, 0.0059, -0.2757, 0.0263)),
           ("q50", (0.0853, 0.0042, -0.2346, 0.0222)), ("q75", (0.0853, 0.0046, -0.2025, 0.0216)),
           ("q90", (0.0749, 0.0064, -0.1915, 0.0275)), ("ols", (0.0818, 0.0037, -0.2319, 0.0174)))
       for (word, suffix), value in zip((("eğitim", ""), ("eğitim", " SH"), ("siyahi", ""), ("siyahi", " SH")), values)},
    **{(1, f"{MODEL_LABELS[model]}: {word} (Tablo 7.1)"): value
       for model, values in (
           ("q10", (0.0914, -0.2670)), ("q25", (0.0896, -0.2509)), ("q50", (0.0911, -0.2600)),
           ("q75", (0.0793, -0.2119)), ("q90", (0.0781, -0.2014)), ("ols", (0.0882, -0.2484)))
       for word, value in zip(("deneyim", "deneyim²/100"), values)},
    (1, "τ = 0,10 kesin yüzde etki"): 8.0,
    (1, "τ = 0,90 kesin yüzde etki"): 7.8,
    (2, "Eğitim: kantiller arası fark"): -0.0025,
    (2, "Eğitim: farkın SH'si"): 0.0083,
    (2, "Eğitim: z istatistiği"): -0.30,
    (2, "Siyahi: kantiller arası fark"): 0.0688,
    (2, "Siyahi: farkın SH'si"): 0.0367,
    (2, "Siyahi: z istatistiği"): 1.88,
}
"""Kontrollerin Card1995 analiz örneklemindeki (3.010 erkek) değerleri, gösterim basamağında: (adım, etiket) → değer.
Metinlerdeki sayılar bunlardır; testler bağımsız bir hesapla ve iki dilde doğrular."""


@cache
def alternative() -> LabSpec:
    """Alternatif örneğin tanımı; beklenen değerler Card1995'in analiz örneklemindeki sayılardır (testle doğrulanır)."""

    texts = {number: (ALT_TEXTS.get(number, ""), ALT_TAKEAWAYS[number]) for number in ALT_TAKEAWAYS}
    spec = _spec(alternative_plan(), texts, source="alternatif", title=ALT_TITLE, dataset=ALT_DATA)
    return with_expected(spec, ALT_EXPECTED)


STORY = (
    "Alternatif örnek Card (1995) verisidir: 1976 log saatlik ücreti gözlenen 3.010 erkek (NLSYM; Hansen'in arşivindeki "
    "Card1995.dta, Konu 1'in alternatif örneğindeki örneklem). Kadın göstergesinin yerinde siyahi göstergesi vardır. "
    "Adımlar notlardaki gibidir: beş kantilde kantil regresyon ve OLS, katsayı profili, kantiller arası fark testi, yorum "
    "sınırları ve raporlama."
)


# --- Kendi verin --------------------------------------------------------------------------------------------

MAX_DECIMALS = 8
"""Katsayı kontrollerinin en çok ondalığı; bir terimin ölçeği daha fazlasını gerektiriyorsa dosya reddedilir."""
MIN_OUTCOME_VALUES = 10
"""Sonuçta en az bu kadar farklı değer olmalı. Az değerli (ör. 0/1, 1–5 ölçekli) bir sonuçta koşullu kantiller basamak
fonksiyonudur: τ ± h kantillerinin doğruları çakışır, Hendricks–Koenker yoğunluğu sıfır çıkar ve standart hatalar
hesaplanamaz."""
NUMERIC_ZERO = 1e-9
"""τ ± h kantillerinde bütün optimal çözümlerin katsayı aralığı bundan dar (katsayının büyüklüğüne göre) ise çözüm
sayısal olarak tektir; değilse standart hata dile göre değişebilir."""
H_CONDITION = 1e12
"""Hendricks–Koenker H = Σ f̂ᵢxᵢxᵢ' matrisinin (köşegeni 1'e ölçeklenmiş) en büyük koşul sayısı; üstünde ters alınamaz."""


def _derived(case: Case) -> pd.DataFrame:
    """Kendi verinde türetilen sütunlarla veri (doğrulama ve tekillik denetimi için; hesap uygulamada ayrıca yapılır)."""

    data = case.data.copy()
    y, z, g = case.roles[SONUC], case.roles[KONTROL], case.roles[GRUP]
    if case.options.get(LOG, True):
        data[LOG_Y] = np.log(data[y].astype(float).where(data[y].astype(float) > 0))
    data[Z2] = data[z].astype(float) ** 2 / 100
    data[G01] = (data[g] == case.levels[GRUP]).astype(float)
    return data


def _decimals(data: pd.DataFrame, outcome: str, term: str) -> int:
    """Bir terimin katsayı ondalığı: tipik katsayı (0,1·sd(Y)/sd(terim)) üç anlamlı basamakla görünsün (en az 4)."""

    ratio = 0.1 * float(data[outcome].std()) / float(data[term].std())
    if not math.isfinite(ratio) or ratio <= 0:
        return 4
    return int(max(4, 2 - math.floor(math.log10(ratio))))


def own_plan(case: Case) -> Plan:
    y, x, z, g = (case.roles[role] for role in (SONUC, ACIKLAYICI, KONTROL, GRUP))
    log = bool(case.options.get(LOG, True))
    pick = case.levels[GRUP]
    name = case.name
    outcome = LOG_Y if log else y
    outcome_name = f"log({name(y)})" if log else name(y)
    prepare: list[Operation] = list(case.load)
    if log:
        prepare.append(Derive(case.frame, LOG_Y, E.log(E.var(y)), "Sonucun doğal logaritması"))
    prepare += [
        Derive(case.frame, Z2, E.div(E.power(E.var(z), 2), 100), "Kontrol değişkeninin karesi / 100"),
        Indicator(case.frame, G01, g, pick, "Grup göstergesi: seçilen kategori 1, diğeri 0"),
    ]
    controls = tuple(case.extras)
    labels = {column: name(column) for column in (y, x, z, g, *controls)}
    labels.update({"(sabit)": "Sabit", Z2: f"{name(z)}²/100", G01: f"{name(g)} = {pick}"})
    if log:
        labels[LOG_Y] = outcome_name
    names = {"x": f"“{name(x)}”", "group": f"“{name(g)}” = “{pick}”", "z": f"“{name(z)}”", "z2": f"“{name(z)}”²/100",
             "title_x": f"“{name(x)}”", "title_group": f"“{name(g)}” = “{pick}” göstergesinin"}
    data = _derived(case)
    places = tuple((term, _decimals(data, outcome, term)) for term in (x, G01, z, Z2))
    return Plan(
        frame=case.frame, outcome=outcome, x=x, group=G01, z=z, z2=Z2, controls=controls, prepare=tuple(prepare),
        labels=labels, names=names, log=log,
        profile_title=f"“{name(x)}” katsayısının dağılımsal profili ({outcome_name})",
        profile_label=f"“{name(x)}” katsayısı",
        decimals=places[0][1], term_decimals=places,
    )


def _inside(value: float, low: float, high: float, decimals: int) -> bool:
    """[en küçük, en büyük] aralığı, değerin gösterim basamağına yuvarlanmışının ± 0,5·10⁻ᵈ penceresinde mi: indirilen
    kod dilin bulduğu değeri bu pencereyle karşılaştırır; aralık pencerenin içindeyse hangi çözüm seçilirse seçilsin
    kontrol geçer."""

    rounded = float(f"{value:.{decimals}f}")
    window = 0.5 * 10 ** (-decimals)
    margin = 1e-9 * max(1.0, abs(rounded))
    return low >= rounded - window + margin and high <= rounded + window - margin


def _invertible(matrix: np.ndarray) -> bool:
    diagonal = np.diag(matrix)
    if not np.isfinite(matrix).all() or np.any(diagonal <= 0):
        return False
    scale = 1.0 / np.sqrt(diagonal)
    return float(np.linalg.cond(matrix * scale[:, None] * scale[None, :])) <= H_CONDITION


@dataclass(frozen=True)
class Stability:
    """Kantil çözümlerinin tekliği (``Plan``'daki aynı adlı alanlar)."""

    unstable: frozenset
    unstable_se: frozenset
    fragile: frozenset


def stability(data: pd.DataFrame, plan: Plan, name: str = "") -> Stability:
    """Kantil çözümlerinin tekliği ve standart hataların hesaplanabilirliği.

    Kantil regresyonun doğrusal programı birden çok optimal çözüme sahip olabilir (ör. çok sayıda eşit değer ya da grup
    büyüklüğü × τ tam sayı); diller farklı köşe çözümü seçebilir. Her kantil modelinde katsayıların bütün optimal
    çözümler üzerindeki aralığı hesaplanır (``quantreg.solution_range``); denetlenen bir katsayının aralığı gösterim
    penceresinin dışına taşarsa modelin kontrolleri çıkarılır (``unstable``). Hendricks–Koenker standart hatası τ ± h
    kantillerindeki çözümleri kullanır; onlar sayısal olarak tek değilse standart hata kontrolleri çıkarılır
    (``unstable_se``). Kesin yüzde etki ve 0,10–0,90 farkı için aynı pencere kuralı uygulanır (``fragile``). Yoğunluk
    matrisi H tekilse standart hatalar hesaplanamaz: ``UploadError`` (``name`` iletideki sonuç adıdır)."""

    from core.labs import quantreg as Q

    columns = list(plan.regressors)
    complete = data[[plan.outcome, *columns]].dropna().astype(float)
    x = np.column_stack([np.ones(len(complete)), complete[columns].to_numpy()])
    y = complete[plan.outcome].to_numpy()
    unit = np.eye(x.shape[1])
    index = {term: j + 1 for j, term in enumerate(columns)}
    checked = (plan.x, plan.group, plan.z, plan.z2)
    unstable, unstable_se, fragile = set(), set(), set()
    fits = {}
    for tau, model in QUANTILES:
        beta, ranges = Q.solution_range(x, y, tau, unit)
        if not all(_inside(beta[index[t]], *ranges[index[t]], plan.places(t)) for t in checked):
            unstable.add(model)
        h = Q.hk_bandwidth(tau, len(y))
        high, high_ranges = Q.solution_range(x, y, tau + h, unit)
        low, low_ranges = Q.solution_range(x, y, tau - h, unit)
        matrix = Q.hk_matrix(x, high, low, h)
        if not _invertible(matrix):
            raise K.UploadError(f"“{name}” için τ = {sayi(tau, 2)} kantilinde Hendricks–Koenker standart hataları "
                                "hesaplanamıyor: τ ± h kantillerinin doğruları çakışıyor (koşullu yoğunluk sıfır). Sonucun "
                                "çok az farklı değeri olabilir ya da değerlerin çoğu birkaç değerde toplanmış olabilir. "
                                "Sürekli bir sonuç seçin.")
        widths = np.concatenate([np.diff(high_ranges, axis=1)[:, 0], np.diff(low_ranges, axis=1)[:, 0]])
        if np.any(widths > NUMERIC_ZERO * np.maximum(1.0, np.abs(np.concatenate([high, low])))):
            unstable_se.add(model)
        fits[model] = (tau, beta, ranges, np.linalg.inv(matrix))
    if plan.log:
        j = index[plan.x]
        for model in ("q10", "q90"):
            _, beta, ranges, _ = fits[model]
            low, high = (100 * (math.exp(value) - 1) for value in ranges[j])
            if model not in unstable and not _inside(100 * (math.exp(beta[j]) - 1), low, high, 1):
                fragile.add(f"yuzde_{model}")
    if not {"q10", "q90"} & (unstable | unstable_se):
        (tau1, beta1, ranges1, hinv1), (tau2, beta2, ranges2, hinv2) = fits["q10"], fits["q90"]
        xtx = x.T @ x
        cov1, cov2 = (tau * (1 - tau) * hinv @ xtx @ hinv for tau, hinv in ((tau1, hinv1), (tau2, hinv2)))
        cross = (min(tau1, tau2) - tau1 * tau2) * hinv1 @ xtx @ hinv2
        for scalar, term in (("fark_x", plan.x), ("fark_grup", plan.group)):
            j = index[term]
            difference = beta2[j] - beta1[j]
            low, high = ranges2[j, 0] - ranges1[j, 1], ranges2[j, 1] - ranges1[j, 0]
            error = math.sqrt(cov1[j, j] + cov2[j, j] - 2 * cross[j, j])
            if not _inside(difference, low, high, plan.places(term)):
                fragile.add(scalar)
            if not _inside(difference / error, low / error, high / error, 2):
                fragile.add(f"{scalar}_z")
    return Stability(frozenset(unstable), frozenset(unstable_se), frozenset(fragile))


def _own_texts(case: Case, plan: Plan) -> dict:
    y, x, z = (md(case.label(role)) for role in (SONUC, ACIKLAYICI, KONTROL))
    group, pick = md(case.label(GRUP)), md(case.levels[GRUP])
    outcome = f"log({y})" if plan.log else y
    controls = [f"“{md(case.name(column))}”" for column in case.extras]
    controls_text = f" Ek kontroller: {', '.join(controls)}." if controls else ""
    unstable = [MODEL_LABELS[name] for name in MODELS if name in plan.unstable]
    unstable_se = [MODEL_LABELS[name] for name in MODELS if name in plan.unstable_se and name not in plan.unstable]
    tails = {"q10", "q90"} & (plan.unstable | plan.unstable_se)

    def direction(value: float) -> str:
        return "yüksek" if value > 0 else "düşük"

    def percents(low: float, high: float) -> str:
        if direction(low) == direction(high):
            return f"yaklaşık %{sayi(abs(low), 1)}, 0,90 kantilinde %{sayi(abs(high), 1)} daha {direction(high)}"
        return (f"yaklaşık %{sayi(abs(low), 1)} daha {direction(low)}, 0,90 kantilinde %{sayi(abs(high), 1)} daha "
                f"{direction(high)}")

    def pattern(errors: list[float]) -> str:
        low, high = errors[0] > errors[2], errors[-1] > errors[2]
        if low and high:
            return ("uçlarda medyandakinden büyük; kuyruklarda koşullu yoğunluk düşük olduğu için aynı örneklem orada "
                    "daha az bilgi taşır.")
        formula = ("Kantil standart hatası yaklaşık $\\sqrt{\\tau(1-\\tau)}/f$ ile orantılıdır: uçlarda $\\tau(1-\\tau)$ "
                   "küçülür, yoğunluk $f$ yeterince düşmezse standart hata büyümez.")
        if not (low or high):
            return "bu veride uçlarda medyandakinden büyük değil. " + formula
        larger, smaller = ("0,10", "0,90") if low else ("0,90", "0,10")
        return f"{larger} kantilinde medyandakinden büyük, {smaller} kantilinde değil. " + formula

    def step1(state) -> str:
        models = state.models
        values = [float(models[name].params[plan.x]) for _, name in QUANTILES]
        errors = [float(models[name].bse[plan.x]) for _, name in QUANTILES]
        group_values = [float(models[name].params[plan.group]) for _, name in QUANTILES]
        ols = float(models["ols"].params[plan.x])
        parts = [f"Analiz örneklemi {sayim(len(state.frames[plan.frame]))} gözlem. “{x}” katsayısı 0,10 kantilinde "
                 f"{katsayi(values[0])}, medyanda {katsayi(values[2])}, 0,90 kantilinde {katsayi(values[-1])}; OLS "
                 f"{katsayi(ols)}. “{group}” = “{pick}” katsayısı 0,10'da {katsayi(group_values[0])}, 0,90'da "
                 f"{katsayi(group_values[-1])}."]
        if plan.log:
            low, high = state.scalars["yuzde_q10"], state.scalars["yuzde_q90"]
            if low != 0 and high != 0:
                parts.append(f"Kesin yüzde dönüşümle “{x}” bir birim yüksek olduğunda koşullu {outcome} dağılımının "
                             f"0,10 kantilinde {percents(low, high)} “{y}” ile ilişkilidir.")
        parts.append(f"“{x}” katsayısının standart hatası 0,10'da {katsayi(errors[0])}, medyanda {katsayi(errors[2])}, "
                     f"0,90'da {katsayi(errors[-1])}: " + pattern(errors))
        if unstable:
            parts.append("Bu veride " + ", ".join(unstable) + " modelinde kantil çözümü gösterim basamağında tek değil "
                         "(doğrusal programın birden çok optimal çözümü var; ör. çok sayıda eşit değer ya da grup "
                         "büyüklüğü × τ tam sayı): diller farklı köşe çözümü seçebileceği için bu modellerin kontrolleri "
                         "indirilen kodda karşılaştırılmaz.")
        if unstable_se:
            parts.append(", ".join(unstable_se) + " modelinde katsayılar tek, ama standart hatanın kullandığı τ ± h "
                         "kantillerinde çözüm tek değil: bu modellerin standart hata kontrolleri karşılaştırılmaz.")
        if {"yuzde_q10", "yuzde_q90"} & plan.fragile:
            parts.append("Kesin yüzde etkilerden en az biri gösterim basamağında tek değil; indirilen kod onu "
                         "karşılaştırmaz.")
        return " ".join(parts)

    def step2(state) -> str:
        s = state.scalars
        parts = []
        for key, word in (("fark_x", f"“{x}”"), ("fark_grup", f"“{group}” = “{pick}”")):
            difference, error = s[key], s[f"{key}_se"]
            verdict = ("%5 düzeyinde anlamlı" if abs(difference / error) > 1.96 else
                       "%5 düzeyinde anlamlı değil: veri iki kantildeki katsayıyı ayırt etmiyor")
            parts.append(f"{word}: β̂(0,90) − β̂(0,10) = {katsayi(difference)} (SH {katsayi(error)}, z = "
                         f"{sayi(difference / error, 2)}; {verdict})")
        text = "; ".join(parts) + ". Sonuç farkın kendi standart hatasından gelir, profilin görsel eğiminden değil."
        if tails:
            text += (" Not: τ = 0,10 ya da 0,90 modelinin çözümü (ya da standart hatası) bu veride tek değil; fark ve "
                     "standart hatası dile göre değişebilir, indirilen kod bunları karşılaştırmaz.")
        elif plan.fragile & {"fark_x", "fark_x_z", "fark_grup", "fark_grup_z"}:
            text += (" Not: farklardan birinin çözümü gösterim basamağında tek değil; indirilen kod o değeri "
                     "karşılaştırmaz.")
        return text

    def step4(state) -> str:
        models = state.models
        values = [float(models[name].params[plan.x]) for _, name in QUANTILES]
        s = state.scalars
        return (f"Raporlama cümlesi örneği: \"“{x}” katsayısı koşullu {outcome} dağılımının 0,10 kantilinde "
                f"{katsayi(values[0])}, medyanda {katsayi(values[2])} ve 0,90 kantilinde {katsayi(values[-1])} olarak "
                f"tahmin edilmiştir. 0,90 ile 0,10 kantilleri arasındaki fark {katsayi(s['fark_x'])} (SH "
                f"{katsayi(s['fark_x_se'])}).\"")

    return {
        1: (
            f"Aynı spesifikasyonu beş kantilde tahmin ediyoruz: sonuç {outcome}; açıklayıcılar “{x}”, “{group}” = "
            f"“{pick}” göstergesi, “{z}” ve “{z}”²/100.{controls_text} $\\tau\\in\\{{0{{,}}10;\\,0{{,}}25;\\,0{{,}}50;"
            f"\\,0{{,}}75;\\,0{{,}}90\\}}$. Karşılaştırma için aynı spesifikasyonun OLS tahmini (HC1) de verilir. "
            + ESTIMATOR,
            "OLS'in tek eğimi kantil profilini bir sayıya indirger. Kantil standart hatası koşullu yoğunluğa bağlıdır: "
            "yoğunluğun düşük olduğu kantillerde aynı örneklem daha az bilgi taşır.",
        ),
        (1, "not"): step1,
        2: (DIFFERENCE, "Güven bantlarının örtüşüp örtüşmemesi eşitlik testiyle aynı şey değildir. Tezde \"heterojenlik "
                        "vardır\" demeden önce bu testi raporlayın."),
        (2, "not"): step2,
        3: (CONDITIONAL, "Üst kantilde eğimin daha büyük olması, \"bu özellik üst gruptaki bireylere daha çok kazandırıyor\" "
                         "demek değildir; bireysel etki heterojenliği daha güçlü tanımlama koşulları ister."),
        4: ("", "Kantil setini önceden belirleyin ve bütün profili raporlayın."),
        (4, "not"): step4,
        5: ("", PROTOCOL_RULE),
        "kod1": CODE_NOTE_1.replace("üç dilde", "iki dilde").replace(", Stata'da `qreg`", "").replace(
            "Python ve Stata'da", "Python'da").replace("; Stata'nın varsayılan `qreg` standart hatası (iid) de farklıdır",
                                                        ""),
        "kod2": CODE_NOTE_2.replace(", Python'da", "; Python'da").replace(
            ", Stata'da `hk_sh` programının oluşturduğu `Hinv_model` ve `J_model` matrisleri", ""),
        "rapor": REPORTING.replace("üç dilde", "iki dilde (Python ve R)"),
    }


def build(case: Case) -> LabSpec:
    """Kendi verinle Konu 7 uygulaması; kontrollerin beklenen değerleri uygulamanın hesabıdır. Kantil çözümü gösterim
    basamağında tek olmayan modellerin ilgili kontrolleri çıkarılır (``stability``)."""

    plan = own_plan(case)
    result = stability(_derived(case), plan, case.name(case.roles[SONUC]))
    plan = replace(plan, unstable=result.unstable, unstable_se=result.unstable_se, fragile=result.fragile)
    spec = _spec(plan, _own_texts(case, plan), source="kendi",
                 title=f"{case.label(SONUC)} ve {case.label(ACIKLAYICI)}", dataset="")
    return with_app_values(spec)


def validate(case: Case) -> None:
    y, x, z, g = (case.roles[role] for role in (SONUC, ACIKLAYICI, KONTROL, GRUP))
    if len({y, x, z, g}) < 4:
        raise K.UploadError("Sonuç, ana açıklayıcı, kontrol ve grup için dört farklı sütun seçin.")
    if any(column in case.extras for column in (y, x, z, g)):
        raise K.UploadError("Ek kontrollerde rollere seçilen sütunlar olmamalı.")
    data = _derived(case)
    outcome = LOG_Y if case.options.get(LOG, True) else y
    distinct = data[y].nunique()
    if distinct < MIN_OUTCOME_VALUES:
        raise K.UploadError(f"“{case.name(y)}” sütununda {distinct} farklı değer var; kantil regresyon için en az "
                            f"{MIN_OUTCOME_VALUES} farklı değer gerekir. Az değerli bir sonuçta (ör. 0/1 ya da 1–5 ölçeği) "
                            "koşullu kantiller basamak fonksiyonudur ve Hendricks–Koenker standart hataları hesaplanamaz; "
                            "böyle bir sonuç için Konu 5'in ikili ya da sıralı modelleri uygundur.")
    if case.options.get(LOG, True) and data[LOG_Y].isna().any():
        raise K.UploadError(f"“{case.name(y)}” sütununda sıfır ya da negatif değer var: logaritma alınamaz. "
                            "Logaritma seçeneğini kapatın.")
    for column, word in ((x, "ana açıklayıcı"), (z, "kontrol")):
        if data[column].nunique() < 3:
            raise K.UploadError(f"“{case.name(column)}” ({word}) sütununda en az 3 farklı değer olmalı.")
    if data[G01].value_counts().min() < 10:
        raise K.UploadError(f"“{case.name(g)}” sütununun iki kategorisinde de en az 10 gözlem olmalı.")
    for column in case.extras:
        if data[column].nunique() < 2:
            raise K.UploadError(f"“{case.name(column)}” sütununda en az iki farklı değer olmalı.")
    regressors = (x, G01, z, Z2, *case.extras)
    if len(data) < 20 * (len(regressors) + 1):
        raise K.UploadError(f"Modelde {len(regressors) + 1} katsayı var; uç kantillerde (0,10 ve 0,90) güvenilir tahmin "
                            f"için en az {20 * (len(regressors) + 1)} gözlem gerekir, dosyada {len(data)} var.")
    if not full_rank(data, regressors):
        raise K.UploadError("Seçilen değişkenler arasında tam doğrusal bağlantı var. Ek kontrolleri değiştirin.")
    if not stable_design(data, outcome, (regressors,)):
        raise K.UploadError(SCALE_MESSAGE)
    for term, label in ((x, case.name(x)), (G01, case.name(g)), (z, case.name(z)), (Z2, f"{case.name(z)}²/100")):
        if _decimals(data, outcome, term) > MAX_DECIMALS:
            raise K.UploadError(f"“{label}” teriminin ölçeği sonuca göre çok büyük: katsayısı {MAX_DECIMALS} ondalıkta "
                                "bile görünmüyor. Dosyada değişkenin birimini değiştirin (ör. gün yerine yıl, TL yerine "
                                "bin TL) ya da sonucun logaritmasını alın.")


def sample() -> pd.DataFrame:
    """Örnek dosya: Konu 1–2'nin kurgusal ücret verisi (``core.labs.ornek_veri``); hata varyansı eğitimle arttığı için
    eğitim katsayısı üst kantillerde daha büyüktür."""

    return ucret_verisi()


ROLES = (
    Role(SONUC, "Sonuç değişkeni", "sayisal", True, (1, 2, 4),
         "Koşullu dağılımı incelenen sayısal değişken (ör. saatlik ücret). Notlardaki gibi logaritması alınabilir."),
    Role(ACIKLAYICI, "Ana açıklayıcı değişken", "sayisal", True, (1, 2, 4),
         "Katsayı profili çizilen değişken (notlarda eğitim)."),
    Role(KONTROL, "Sürekli kontrol değişkeni", "sayisal", True, (1,),
         "Doğrusal ve karesel olarak eklenir (notlarda deneyim)."),
    Role(GRUP, "İki kategorili grup değişkeni", "kategorik", True, (1, 2),
         "Gösterge olarak eklenir (notlarda kadın).", levels=(2, 2), pick="1 ile kodlanan grup"),
)

OPTIONS = (
    Option(LOG, "Sonucun logaritmasını al (log Y)",
           "Notlardaki gibi sonucun doğal logaritması kullanılır; katsayılar yaklaşık yüzde farklar olarak okunur ve kesin "
           "yüzde dönüşüm hesaplanır.",
           default=True, role=SONUC, allowed=K1._positive,
           blocked="Sonuç sütununda sıfır ya da negatif değer olduğu için logaritma alınamaz; sonuç kendi biriminde "
                   "kullanılır."),
)

CUSTOM = CustomLab(
    roles=ROLES,
    build=build,
    sample=sample,
    intro=(
        "Bir Excel (.xlsx) ya da CSV dosyası yükleyin. Sonuç, katsayı profili çizilecek ana açıklayıcı, karesiyle "
        "eklenen sürekli bir kontrol ve iki kategorili bir grup seçilir; ek sayısal kontroller isteğe bağlıdır. Seçilen "
        "sütunların birinde boş hücresi olan satırlar analizden çıkarılır: bütün modeller aynı gözlemlerle tahmin edilir."
    ),
    options=OPTIONS,
    min_rows=100,
    extra_columns=True,
    extra_use="sayisal",
    extra_label="Ek sayısal kontroller (isteğe bağlı, en çok 6)",
    extra_help="Bütün kantil modellerine ve OLS'e eklenir. Boş hücresi olan satırlar çıkarılır.",
    max_extra=6,
    extra_required=True,
    validate=validate,
)

VARIANTS = TopicVariants(alternative=alternative, story=STORY, custom=CUSTOM)
