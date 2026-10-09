"""Konu 5 genel uygulaması: ikili sonuçlarda katsayıdan olasılığa geçmek.

Notlardaki §5.15'in altı adımı aynı numaralarla ve aynı işlemlerle, verisi değiştirilebilir biçimde yazılır (``build``):
LPM, Logit ve Probit ile ortalama marjinal etkiler (AME), Logit katsayısı ile ölçek çarpanı, ortalama tahmin edilen olasılık
profili, AME tablosu, kukla değişkende türev yerine sonlu fark ve okuma soruları. Alternatif örnek Cox, Hansen ve Jimenez
(2004) verisidir (Hansen'in arşivindeki ``CHJ2004.dta``): Filipinler'de bir hanenin yurt dışından havale alıp almadığı.
"Kendi verini yükle" seçeneğinde aynı adımlar öğrencinin dosyasıyla kurulur. Notlardaki laboratuvar (``core.labs.konu05``)
değişmez.

Rollerin notlardaki karşılıkları: ikili sonuç evli olmak, profil değişkeni yaş (olasılık profili onun değerleri boyunca),
iki kategorili gösterge siyah (sonlu fark adımı), ek kontroller eğitim, ırk/etnik köken ve bölge.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd

from core.labs import expr as E
from core.labs import kendi_veri as K
from core.labs.ornek import (
    SCALE_MESSAGE,
    Case,
    CustomLab,
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
from core.labs.ornek_veri import ikili_veri
from core.labs.spec import (
    OLS,
    AverageProfile,
    BinaryChoice,
    Check,
    CoefTarget,
    Derive,
    EffectTable,
    Indicator,
    LabSpec,
    LabStep,
    LoadHansen,
    MarginalEffects,
    NoteRef,
    Operation,
    ReproClass,
    Scalar,
    ScalarTarget,
    ShowModel,
    StatTarget,
    TableTarget,
)

TOPIC = "konu05"
SECTION = "5.15"
SONUC, PROFIL, GOSTERGE = "sonuc", "profil", "gosterge"
Y01, D01 = "sonuc01", "gosterge01"
"""Kendi verinde türetilen göstergeler (``kendi_veri.RESERVED_CODES``: öğrencinin sütunlarına verilmez)."""
MIN_CLASS = 10
"""Sonucun her iki kategorisinde en az bu kadar gözlem olmalı."""


@dataclass(frozen=True)
class Plan:
    """Rollerin koddaki sütunları, tablo satırları ve etiketler.

    ``regressors``: modellerin regresörleri (sıra koddaki sırayla aynı); ``categorical``: bunlardan kategorik olanlar
    (AME'leri referans düzeyine göre olasılık farkı); ``terms``: AME tablosunun (terim, etiket) satırları; ``profile`` ve
    ``indicator``: Adım 1 tablosunun iki değişkeni, Adım 3'ün profil değişkeni ve Adım 5'in göstergesi (terim adı
    ``indicator_term``); ``grid``: Adım 3'ün değerleri.
    """

    frame: str
    outcome: str
    profile: str
    indicator: str
    indicator_term: str
    regressors: tuple[str, ...]
    categorical: tuple[str, ...]
    ame_terms: tuple[str, ...]
    terms: tuple[tuple[str, str], ...]
    grid: tuple[float, ...]
    prepare: tuple[Operation, ...]
    labels: dict
    names: dict
    """Metinlerdeki adlar: ``profile``, ``indicator`` (ör. "yaş", "kadın hane reisi"), ``outcome``."""
    profile_title: str
    profile_label: str
    decimals: int = 4
    discrete: tuple[str, ...] = ()
    """Kategorik tanımlanmamış 0/1 regresörler: AME'leri de türev yerine 1 − 0 farkıdır (kendi verinde gösterge)."""


def _coef(model: str, term: str, label: str, quantity: str = "coef", decimals: int = 4) -> Check:
    return Check(label, CoefTarget(model, term, quantity), 0.0, decimals)


def _cap(text: str) -> str:
    """İlk harfi büyütür, gerisine dokunmaz (``str.capitalize`` öğrencinin sütun adını küçültürdü); Türkçe i → İ."""

    return {"i": "İ", "ı": "I"}.get(text[:1], text[:1].upper()) + text[1:]


def _steps(plan: Plan, texts: dict) -> tuple[LabStep, ...]:
    p, d = plan.profile, plan.indicator_term
    pn, dn = plan.names["profile"], plan.names["indicator"]
    pc, dc = _cap(pn), _cap(dn)
    dec = plan.decimals
    step1_checks = [Check("Analiz örneklemi (N)", StatTarget(plan.frame, plan.outcome, "count"), 0.0, decimals=0)]
    for model, title in (("lpm", "LPM"), ("ame_logit", "Logit AME"), ("ame_probit", "Probit AME")):
        se = "HC1 SH" if model == "lpm" else "SH"
        step1_checks += [
            _coef(model, p, f"{title} {pn}", decimals=dec),
            _coef(model, p, f"{title} {pn} {se}", "se", decimals=dec),
            _coef(model, d, f"{title} {dn}"),
            _coef(model, d, f"{title} {dn} {se}", "se"),
        ]
    rows = tuple((f"{title}: {pn}", model, p) for model, title in
                 (("lpm", "LPM"), ("ame_logit", "Logit AME"), ("ame_probit", "Probit AME"))) + \
        tuple((f"{title}: {dn}", model, d) for model, title in
              (("lpm", "LPM"), ("ame_logit", "Logit AME"), ("ame_probit", "Probit AME")))
    step4_checks: list[Check] = []
    for link, title in (("logit", "Logit"), ("probit", "Probit")):
        for term, label in plan.terms:
            step4_checks += [_coef(f"ame_{link}", term, f"{title} AME: {label}"),
                             _coef(f"ame_{link}", term, f"{title} SH: {label}", "se")]
    return (
        LabStep(
            number=1,
            title="Aynı araştırma sorusunu üç model ailesiyle sormak",
            note=NoteRef(SECTION, 1, ("Tablo 5.2",)),
            explanation=texts[1][0],
            operations=(
                *plan.prepare,
                OLS("lpm", plan.frame, plan.outcome, plan.regressors, vcov="HC1", categorical=plan.categorical),
                BinaryChoice("logit", plan.frame, plan.outcome, plan.regressors, "logit", plan.categorical),
                BinaryChoice("probit", plan.frame, plan.outcome, plan.regressors, "probit", plan.categorical),
                MarginalEffects("logit", "ame_logit", plan.ame_terms, plan.discrete),
                MarginalEffects("probit", "ame_probit", plan.ame_terms, plan.discrete),
                EffectTable(rows, "tablo_52", title="LPM ve doğrusal olmayan modellerde olasılık etkileri", se_label="SH"),
            ),
            checks=tuple(step1_checks),
            reproducibility=ReproClass.CONVENTION,
            takeaway=texts[1][1],
            code_note=CODE_NOTE_1,
            note_for=texts.get((1, "not")),
        ),
        LabStep(
            number=2,
            title="Logit katsayısını neden doğrudan yorumlamıyoruz?",
            note=NoteRef(SECTION, 2),
            explanation=texts[2][0],
            operations=(
                Scalar("logit_profil", E.coef("logit", p), f"Logit {pn} katsayısı β̂", percent=False, decimals=dec),
                Scalar("olcek_logit", E.div(E.coef("ame_logit", p), E.coef("logit", p)), "Ortalama Λ(1−Λ)",
                       percent=False, decimals=4),
                Scalar("probit_profil", E.coef("probit", p), f"Probit {pn} katsayısı β̂", percent=False, decimals=dec),
                Scalar("olcek_probit", E.div(E.coef("ame_probit", p), E.coef("probit", p)), "Ortalama φ",
                       percent=False, decimals=4),
                Scalar("katsayi_orani", E.div(E.coef("logit", p), E.coef("probit", p)), "Logit/Probit katsayı oranı",
                       percent=False, decimals=2),
            ),
            checks=(
                Check(f"Logit {pn} katsayısı", ScalarTarget("logit_profil"), 0.0, dec),
                Check("Ortalama Λ(1−Λ)", ScalarTarget("olcek_logit"), 0.0),
                Check(f"Probit {pn} katsayısı", ScalarTarget("probit_profil"), 0.0, dec),
                Check("Ortalama φ", ScalarTarget("olcek_probit"), 0.0),
                Check("Logit/Probit katsayı oranı", ScalarTarget("katsayi_orani"), 0.0, decimals=2),
            ),
            reproducibility=ReproClass.EXACT,
            takeaway=texts[2][1],
            note_for=texts.get((2, "not")),
        ),
        LabStep(
            number=3,
            title="Tahmin edilen olasılık profilini okumak",
            note=NoteRef(SECTION, 3, ("Şekil 5.4",)),
            explanation=texts[3][0],
            operations=(
                AverageProfile("logit", "profil", p, plan.grid, plan.labels.get(p, p), plan.profile_label,
                               plan.profile_title),
            ),
            checks=(
                Check(f"{pc} = {_value(plan.grid[0])} için ortalama olasılık",
                      TableTarget("profil", plan.grid[0], "olasilik"), 0.0),
                Check(f"{pc} = {_value(plan.grid[-1])} için ortalama olasılık",
                      TableTarget("profil", plan.grid[-1], "olasilik"), 0.0),
            ),
            reproducibility=ReproClass.EXACT,
            takeaway=texts[3][1],
            code_note=texts.get("kod3", ""),
            note_for=texts.get((3, "not")),
        ),
        LabStep(
            number=4,
            title="Yazılım çıktısından AME tablosuna geçmek",
            note=NoteRef(SECTION, 4, ("§5.9", "Tablo 5.1")),
            explanation=texts[4][0],
            operations=(
                ShowModel("logit"),
                EffectTable(tuple((label, "ame_logit", term) for term, label in plan.terms), "tablo_51_logit",
                            title="Logit: ortalama marjinal etkiler", se_label="Dayanıklı SH"),
                EffectTable(tuple((label, "ame_probit", term) for term, label in plan.terms), "tablo_51_probit",
                            title="Probit: ortalama marjinal etkiler", se_label="Dayanıklı SH"),
            ),
            checks=tuple(step4_checks),
            reproducibility=ReproClass.CONVENTION,
            takeaway=texts[4][1],
            code_note=texts.get("kod4", ""),
            note_for=texts.get((4, "not")),
        ),
        LabStep(
            number=5,
            title="Kukla değişken için türev değil sonlu fark",
            note=NoteRef(SECTION, 5),
            explanation=texts[5][0],
            operations=(
                Scalar("gosterge_turev", E.mul(E.coef("logit", d), E.div(E.coef("ame_logit", p), E.coef("logit", p))),
                       f"{dc}: türev tabanlı yaklaşım", percent=False, decimals=4),
                Scalar("gosterge_fark", E.coef("ame_logit", d), f"{dc}: sonlu fark (AME)", percent=False, decimals=4),
            ),
            checks=(
                Check(f"{dc}, türev tabanlı", ScalarTarget("gosterge_turev"), 0.0),
                Check(f"{dc}, sonlu fark", ScalarTarget("gosterge_fark"), 0.0),
            ),
            reproducibility=ReproClass.EXACT,
            takeaway=texts[5][1],
            note_for=texts.get((5, "not")),
        ),
        LabStep(
            number=6,
            title="Bir ikili sonuç makalesindeki tabloyu nasıl okumalıyız?",
            note=NoteRef("5.15.6", 0),
            explanation=QUESTIONS,
            takeaway=texts[6][1],
            note_for=texts.get((6, "not")),
        ),
    )


def _value(value: float) -> str:
    """Izgara değerinin etiketteki yazımı: tam sayılarda binlik ayırıcı, ondalıklarda gereken kadar basamak."""

    value = float(value)
    if value.is_integer():
        return ("−" if value < 0 else "") + sayim(abs(value))
    return np.format_float_positional(value, trim="-").replace("-", "−").replace(".", ",")


def _spec(plan: Plan, texts: dict, *, source: str, title: str, dataset: str) -> LabSpec:
    return LabSpec(topic_key=TOPIC, title=title, dataset=dataset, note_section=SECTION, steps=_steps(plan, texts),
                   labels=tuple(plan.labels.items()), source=source)


# --- Ortak metinler -------------------------------------------------------------------------------------------

QUESTIONS = (
    "1. Sonuç değişkeninde 1 hangi olayı temsil ediyor?\n"
    "2. Raporlanan sayılar ham Logit/Probit katsayısı mı, odds ratio mu, AME mi?\n"
    "3. Sürekli ve kukla değişkenler için aynı etki tanımı mı kullanılmış?\n"
    "4. Standart hatalar hangi kovaryans yapısına dayanıyor?\n"
    "5. Tahmin edilen olasılıklar $[0,1]$ aralığında mı ve model uç bölgelerde yeterli destek görüyor mu?\n"
    "6. Logit ve Probit benzer olasılık sonuçları veriyor mu?\n"
    "7. Modelin nedensel yorumu için hangi tanımlama varsayımı kullanılıyor?"
)
CODE_NOTE_1 = (
    "Logit/Probit dayanıklı kovaryansı üç dilde aynı tanımla hesaplanır: gözlenen Hessian ile sandviç "
    "H⁻¹(Σsᵢsᵢ')H⁻¹. Python `cov_type=\"HC0\"`, R `dayanikli_vcov()`, Stata `vce(robust)`. Stata ayrıca n/(n−1) ile "
    "çarpar (binlerce gözlemde fark beşinci anlamlı basamakta). AME'ler Python/R'de kodda tanımlı fonksiyonla, Stata'da "
    "`margins` ile hesaplanır."
)
CODE_NOTE_1_OWN = (
    "Logit/Probit dayanıklı kovaryansı iki dilde aynı tanımla hesaplanır: gözlenen Hessian ile sandviç H⁻¹(Σsᵢsᵢ')H⁻¹ "
    "(Python `cov_type=\"HC0\"`, R `dayanikli_vcov()`). AME'ler iki dilde kodda tanımlı fonksiyonla hesaplanır."
)
SCALE_RULE = (
    "Ham katsayıların oranı iki modelin gizli hata ölçeğini farklı normalize etmesinden gelir (§5.5); ekonomik bir fark "
    "değildir. Bu nedenle bir Logit katsayısı doğrudan olasılık farkı olarak okunmaz."
)
PROFILE_RULE = (
    "Her değer için örneklemdeki **herkesin** değişkeni o değere eşitlenir; diğer kovaryatlar kendi gözlenen "
    "değerlerinde kalır ve bireysel tahmin edilen olasılıklar ortalanır. Bu, \"ortalama bireyin olasılığı\" ile aynı "
    "işlem değildir: doğrusal olmayan modelde $G(\\mathbb E[X]'\\hat\\beta)\\neq\\mathbb E[G(X'\\hat\\beta)]$."
)
FINITE_RULE = (
    "$D\\in\\{0,1\\}$ bir göstergede türev yerine\n\n"
    "$$\\Delta_D(X)=G(X'\\hat\\beta+\\hat\\beta_D)-G(X'\\hat\\beta)$$\n\n"
    "hesaplanır: her gözlem için $D=1$ ve $D=0$ senaryoları tahmin edilip farkların ortalaması alınır. Türev tabanlı "
    "yaklaşım $\\hat\\beta_D\\cdot\\frac1n\\sum_i g(X_i'\\hat\\beta)$, $D$'yi sürekli bir değişken gibi küçük bir değişimle ele "
    "alır."
)
AME_TABLE = (
    "Yazılımın ilk çıktısı modelin ham Logit katsayılarıdır; ikinci çıktı olasılık ölçeğindeki ortalama marjinal "
    "etkilerdir. Sürekli değişkenlerde türevin, göstergelerde referans kategoriye göre tahmin edilen olasılık farkının "
    "örneklem ortalaması alınır. Standart hatalar dayanıklı kovaryans matrisinden delta yöntemiyle gelir (§5.7.3)."
)


# --- Alternatif örnek: Cox, Hansen ve Jimenez (2004) ------------------------------------------------------------

ALT_DATA = "chj2004"
ALT_FRAME = "chj"
ALT_TITLE = "Filipin Hanelerinde Yurt Dışından Havale: Katsayıdan Olasılığa"
ALT_REGRESSORS = ("age", "egitim", "female", "married", "size", "gelir", "region")
ALT_CATEGORICAL = ("egitim", "female", "married", "region")
ALT_AME = ("age", "egitim", "female", "married", "size", "gelir")
ALT_TERMS = (
    ("age", "Yaş"),
    ("egitim=1", "İlkokul"),
    ("egitim=2", "Ortaöğretim (tamamlanmamış)"),
    ("egitim=3", "Ortaöğretim"),
    ("egitim=4", "Üniversite (tamamlanmamış)"),
    ("egitim=5", "Üniversite"),
    ("female=1", "Kadın hane reisi"),
    ("married=1", "Evli hane reisi"),
    ("size", "Hane büyüklüğü"),
    ("gelir", "Gelir (10 bin peso)"),
)
ALT_AGES = tuple(float(age) for age in range(20, 81))


def alternative_plan() -> Plan:
    frame = ALT_FRAME
    education = E.add(E.add(E.add(E.add(E.var("primary"), E.mul(2, E.var("somesecondary"))),
                                  E.mul(3, E.var("secondary"))), E.mul(4, E.var("someuniversity"))),
                      E.mul(5, E.var("university")))
    return Plan(
        frame=frame, outcome="havale", profile="age", indicator="female", indicator_term="female=1",
        regressors=ALT_REGRESSORS, categorical=ALT_CATEGORICAL, ame_terms=ALT_AME, terms=ALT_TERMS, grid=ALT_AGES,
        prepare=(
            LoadHansen(ALT_DATA, "CHJ2004.dta", frame),
            Derive(frame, "havale", E.compare("gt", E.var("tabroad"), 0),
                   "Sonuç: hane yurt dışından havale alıyor (tabroad > 0)"),
            Derive(frame, "egitim", education,
                   "Hane reisinin eğitimi: 0 ilkokulu bitirmemiş, 1 ilkokul, 2 ortaöğretim (tamamlanmamış), "
                   "3 ortaöğretim, 4 üniversite (tamamlanmamış), 5 üniversite"),
            Derive(frame, "gelir", E.div(E.var("income"), 10000),
                   "Gelir, 10 bin peso (arşivdeki income düzeltilmiş gelirdir: transferler ve emekli maaşı düşülmüş)"),
        ),
        labels={
            "(sabit)": "Sabit",
            "havale": "Yurt dışından havale",
            "age": "Hane reisinin yaşı",
            "egitim": "Eğitim (6 grup)",
            "female": "Kadın hane reisi",
            "married": "Evli hane reisi",
            "size": "Hane büyüklüğü",
            "gelir": "Gelir (10 bin peso)",
            "region": "Bölge",
            "lpm": "LPM",
            "logit": "Logit",
            "probit": "Probit",
            "ame_logit": "Logit AME",
            "ame_probit": "Probit AME",
            **{term: label for term, label in ALT_TERMS if "=" in term},
        },
        names={"profile": "yaş", "indicator": "kadın hane reisi", "outcome": "havale"},
        profile_title="CHJ2004: Logit modelinde hane reisi yaşı profili",
        profile_label="Ortalama tahmin edilen havale olasılığı",
        decimals=5,
    )


ALT_TEXTS = {
    1: (
        "Soru: Filipinler'deki kentsel hanelerde yurt dışından havale alma olasılığı hane reisinin yaşı, cinsiyeti, "
        "eğitimi ve medeni durumu, hane büyüklüğü ve hane geliriyle nasıl ilişkili? Veri Cox, Hansen ve Jimenez (2004) "
        "çalışmasının Filipinler kentsel hane verisidir; Hansen'in arşivindeki `CHJ2004.dta` gelir dağılımının üst %2'sini "
        "ve negatif gelirleri dışlar: 8.684 hane (Konu 6'nın uygulamasındaki veri). Sonuç ikili: "
        "$havale_i=\\mathbf 1\\{tabroad_i>0\\}$ (`tabroad`: yurt dışından alınan transfer, peso); hanelerin %20,4'ü "
        "yurt dışından havale alıyor. Aynı soruyu üç modelle soruyoruz: doğrusal olasılık modeli (LPM), Logit ve "
        "Probit.\n\n"
        "LPM'de katsayı doğrudan olasılık farkıdır. Logit ve Probit'te ham katsayı tek indeks ölçeğindedir; "
        "karşılaştırma **ortalama marjinal etki (AME)** üzerinden yapılır. Kovaryans: LPM'de HC1, Logit/Probit'te "
        "sandviç (dayanıklı). Eğitim altı gruptur (referans: ilkokulu bitirmemiş hane reisi). Gelir, Hansen'in "
        "düzeltilmiş geliridir (transferler ve emekli maaşı düşülmüş) ve 10 bin peso birimindedir. 14 bölgenin 13 "
        "göstergesi kontrol olarak modeldedir. Tabloda yaş ve kadın hane reisi göstergesi karşılaştırılır."
    ),
    2: (
        "Logit modelinde $P(Y_i=1\\mid X_i)=\\Lambda(X_i'\\beta)$ ve sürekli $X_j$ için marjinal etki\n\n"
        "$$\\frac{\\partial P(Y_i=1\\mid X_i)}{\\partial X_{ij}}=\\Lambda(X_i'\\beta)\\{1-\\Lambda(X_i'\\beta)\\}"
        "\\beta_j.$$\n\n"
        "Aynı $\\beta_j$ farklı hanelerde farklı olasılık etkisi üretir; AME bu etkilerin örneklem ortalamasıdır: "
        "$\\widehat{AME}_j=\\hat\\beta_j\\cdot\\frac1n\\sum_i g(X_i'\\hat\\beta)$. Aşağıda hane reisinin yaşı için "
        "ham katsayı ile ortalama ölçek çarpanı ayrı ayrı gösteriliyor."
    ),
    3: PROFILE_RULE.replace("Her değer için örneklemdeki **herkesin** değişkeni", "Her yaş için örneklemdeki "
                            "**bütün hane reislerinin** yaşı"),
    4: (
        "Yazılımın ilk çıktısı modelin ham Logit katsayılarıdır; ikinci çıktı olasılık ölçeğindeki ortalama marjinal "
        "etkilerdir. Sürekli değişkenlerde (yaş, hane büyüklüğü, gelir) türevin, göstergelerde referans kategoriye göre "
        "tahmin edilen olasılık farkının örneklem ortalaması alınır: eğitimde referans ilkokulu bitirmemiş hane reisi, "
        "cinsiyette erkek, medeni durumda evli olmayan hane reisi. Bölge göstergeleri modelde ama tabloda yoktur. "
        "Standart hatalar dayanıklı kovaryans matrisinden delta yöntemiyle gelir (§5.7.3)."
    ),
    5: FINITE_RULE,
    6: "",
}
"""Adım açıklamaları; anlatılan sayılar aşağıdaki kontrollerdir (``ALT_EXPECTED``), ``ALT_TAKEAWAYS`` ile birleşir."""

ALT_TAKEAWAYS = {
    1: (
        "Yaş için üç yöntem aynı mesajı verir: Logit AME'si 0,00336, yani hane reisinin yaşının bir yıl yüksek olması "
        "havale olasılığında ortalama yaklaşık **0,34 yüzde puanlık** farkla ilişkilidir (LPM 0,00325, Probit 0,00340). "
        "Kadın hane reisi göstergesinin Logit AME'si 0,2991: diğer değişkenler aynıyken kadın reisli hanelerde havale "
        "olasılığı ortalama yaklaşık **30 yüzde puan** yüksektir (LPM 0,2925, Probit 0,2824). Ortalama olasılık 0,20 "
        "olduğundan bu göreli olarak çok büyük bir farktır. Ham oranlar farkın nereden geldiğini gösterir: havale alan "
        "hanelerin payı evli kadın hane reislerinde %72,0, evli olmayan kadın reislerde %24,4, evli olmayan erkek "
        "reislerde %22,4, evli erkek reislerde %16,4. Fark neredeyse tamamen evli kadın reislerden gelir: modele evli "
        "kadın hane reisi göstergesi (`marriedf`, kadın × evli) eklendiğinde kadın hane reisi katsayısı 1,68'den 0,11'e "
        "düşer, `marriedf` katsayısı 2,34 olur. Tek bir \"kadın hane reisi\" AME'si bu iki çok farklı grubu ortalar. "
        "Evli bir kadının hane reisi olarak kaydedilmesi eşin hanede olmadığını, örneğin yurt dışında çalıştığını "
        "gösteriyor olabilir; o zaman göç hem hane reisliğini hem havaleyi belirler ve bu fark nedensel bir \"cinsiyet "
        "etkisi\" olarak okunamaz."
    ),
    2: (
        "Logit yaş katsayısı 0,02370 ama ortalama ölçek çarpanı Λ(1−Λ) 0,1416; çarpımları AME'yi (0,00336) verir. "
        "Probit'te katsayı 0,01357, ortalama φ 0,2501; çarpım 0,00340. Ham katsayıların oranı 1,75'tir; notlarda 1,65 idi. "
        "Oran sabit değildir: iki model aynı olasılığı ve aynı marjinal etkiyi verdiğinde katsayılarının oranı "
        "$\\phi(\\Phi^{-1}(p))/\\{p(1-p)\\}$ olur; bu, $p=0{,}5$'te yaklaşık 1,60, $p=0{,}2$'de yaklaşık 1,75'tir. "
        "Bu veride ortalama havale olasılığı 0,20'dir; oranı belirleyen ortalama ölçek çarpanlarının oranıdır "
        "(0,2501/0,1416 ≈ 1,77) ve AME'ler neredeyse aynı olduğundan (0,00336 ve 0,00340) katsayı oranı 1,75 çıkar. "
        + SCALE_RULE
    ),
    3: (
        "Ortalama tahmin edilen havale olasılığı 20 yaşında 0,130, 80 yaşında 0,341'dir. Profil bu aralıkta dışbükeydir: "
        "on yıllık artış 20–30 yaş arasında 0,026, 70–80 yaş arasında 0,044. Olasılıklar 0,5'in altında kaldığı için S "
        "eğrisinin yalnız alt kolunu görüyoruz; notlardaki evlilik profili (0,11'den 0,82'ye) 0,5'i geçtiği için S "
        "biçiminin iki kolunu da gösterir. Aynı yaş katsayısı reisi yaşlı hanelerde daha büyük olasılık farkı üretir."
    ),
    4: (
        "Logit ve Probit ham katsayıları farklı ölçekte olsa da AME'ler birbirine yakındır; en büyük fark kadın hane "
        "reisindedir (0,2991 ve 0,2824), o da bir standart hatanın altındadır. Eğitimde sıralı bir artış var: ilkokulu "
        "bitirmemiş hane reislerine göre fark ilkokul mezunlarında 0,0288, üniversite mezunlarında 0,2062. Gelirin AME'si "
        "−0,0097 (SH 0,0013): diğer değişkenler aynıyken 10 bin peso daha yüksek gelir, havale olasılığında yaklaşık 1 "
        "yüzde puan daha düşük bir değerle ilişkilidir. Konu 6 aynı veride bütün transferlerin gelirle ilişkisini "
        "inceler; buradaki sayı yalnız yurt dışı kanalına ilişkin koşullu bir ilişkidir."
    ),
    5: (
        "Kadın hane reisi göstergesi için türev tabanlı yaklaşım 0,2373, doğru tanım olan sonlu fark 0,2991; fark 0,0618, "
        "yani AME'nin yaklaşık beşte biri. Notlardaki siyah göstergesinde fark yalnız 0,0019'du (−0,1598 ve −0,1617). "
        "Buradaki büyük farkın iki nedeni var: Logit katsayısı büyük (1,68) ve hanelerin yaklaşık %95'inde tahmin edilen "
        "olasılık 0,5'in altında. Gösterge 0'dan 1'e geçince indeks 1,68 birim kayar; bu aralıkta Logit eğrisi çoğunlukla "
        "dışbükeydir ve sonlu fark, ortalama eğime dayanan türev yaklaşımından büyük çıkar (Sezgi Deney 3). Türev "
        "yaklaşımı eğimi gözlenen değerlerde ölçer: hane reislerinin %83'ü erkek olduğundan eğim çoğunlukla aralığın sol "
        "ucunda, eğrinin daha yatık bölgesinde ölçülür. Yazılımın marjinal etki komutunun göstergeleri nasıl ele aldığını "
        "her zaman kontrol edin."
    ),
    6: (
        "Tez yazımında örnek sonuç dili: \"Logit modelinden elde edilen ortalama marjinal etkiye göre, kadın hane reisli "
        "hanelerin yurt dışından havale alma olasılığı, diğer değişkenler aynıyken, erkek hane reisli hanelerinkinden "
        "ortalama yaklaşık 30 yüzde puan yüksektir; Probit tahmini 28 yüzde puandır. Bu ortalama, evli kadın reislerdeki "
        "çok büyük fark ile evli olmayan kadın reislerdeki küçük farkı birleştirir. Fark gözlemsel veriden elde edilmiş "
        "koşullu bir ilişkidir; hane reisliği göç kararının bir sonucu olabileceği için nedensel etki olarak "
        "yorumlanmamıştır.\""
    ),
}

ALT_EXPECTED = {
    (1, "Analiz örneklemi (N)"): 8684,
    (1, "LPM yaş"): 0.00325,
    (1, "LPM yaş HC1 SH"): 0.00034,
    (1, "LPM kadın hane reisi"): 0.2925,
    (1, "LPM kadın hane reisi HC1 SH"): 0.0172,
    (1, "Logit AME yaş"): 0.00336,
    (1, "Logit AME yaş SH"): 0.00032,
    (1, "Logit AME kadın hane reisi"): 0.2991,
    (1, "Logit AME kadın hane reisi SH"): 0.0202,
    (1, "Probit AME yaş"): 0.00340,
    (1, "Probit AME yaş SH"): 0.00032,
    (1, "Probit AME kadın hane reisi"): 0.2824,
    (1, "Probit AME kadın hane reisi SH"): 0.0192,
    (2, "Logit yaş katsayısı"): 0.02370,
    (2, "Ortalama Λ(1−Λ)"): 0.1416,
    (2, "Probit yaş katsayısı"): 0.01357,
    (2, "Ortalama φ"): 0.2501,
    (2, "Logit/Probit katsayı oranı"): 1.75,
    (3, "Yaş = 20 için ortalama olasılık"): 0.1302,
    (3, "Yaş = 80 için ortalama olasılık"): 0.3408,
    **{(4, f"{link} {kind}: {label}"): value for link, rows in (
        ("Logit", ((0.0034, 0.0003), (0.0288, 0.0116), (0.0508, 0.0139), (0.0890, 0.0125), (0.1666, 0.0157),
                   (0.2062, 0.0170), (0.2991, 0.0202), (0.1225, 0.0118), (0.0047, 0.0020), (-0.0097, 0.0013))),
        ("Probit", ((0.0034, 0.0003), (0.0276, 0.0116), (0.0506, 0.0140), (0.0883, 0.0125), (0.1680, 0.0156),
                    (0.2058, 0.0168), (0.2824, 0.0192), (0.1152, 0.0121), (0.0050, 0.0020), (-0.0094, 0.0013))),
    ) for (_, label), pair in zip(ALT_TERMS, rows) for kind, value in zip(("AME", "SH"), pair)},
    (5, "Kadın hane reisi, türev tabanlı"): 0.2373,
    (5, "Kadın hane reisi, sonlu fark"): 0.2991,
}
"""Kontrollerin CHJ2004 tam örneklemindeki (8.684 hane) değerleri, gösterim basamağında: (adım, etiket) → değer.
Metinlerdeki sayılar bunlardır; testler bağımsız bir hesapla ve iki dilde doğrular."""


@cache
def alternative() -> LabSpec:
    """Alternatif örneğin tanımı; beklenen değerler CHJ2004'ün tam verisindeki sayılardır (testle doğrulanır)."""

    texts = {number: (ALT_TEXTS[number], ALT_TAKEAWAYS[number]) for number in ALT_TEXTS}
    texts.update(ALT_CODE_NOTES)
    spec = _spec(alternative_plan(), texts, source="alternatif", title=ALT_TITLE, dataset=ALT_DATA)
    return with_expected(spec, ALT_EXPECTED)


ALT_CODE_NOTES = {
    "kod3": "Stata'da aynı hesap `margins, at(age=(20(1)80))` komutudur; grafik `marginsplot` ile çizilir.",
    "kod4": (
        "Stata'da `margins, dydx(age egitim female married size gelir)`: `i.` ile girilen değişkenlerde (eğitim, "
        "cinsiyet, medeni durum) sonlu farkı kendiliğinden alır. Python'da statsmodels'in `get_margeff(dummy=True)` "
        "seçeneği çok düzeyli kategorik değişkende düzeyleri birbirinden bağımsız 0/1 yapar (olanaksız kombinasyonlar "
        "üretir); bu yüzden kodda tanımlı fonksiyon kullanılır."
    ),
}

STORY = (
    "Alternatif örnek Cox, Hansen ve Jimenez (2004) verisidir: Filipinler'de 8.684 kentsel hane (Hansen'in arşivindeki "
    "CHJ2004.dta; Konu 6'nın uygulamasındaki veri). Sonuç, hanenin yurt dışından havale alıp almadığıdır. Adımlar "
    "notlardaki gibidir: LPM, Logit ve Probit, ortalama marjinal etkiler, yaş profili, AME tablosu ve kadın hane reisi "
    "göstergesinde türev yerine sonlu fark. Yöntem aynı, soru ve veri farklı."
)


# --- Kendi verin --------------------------------------------------------------------------------------------

def _indicators(case: Case) -> pd.DataFrame:
    data = case.data.copy()
    for role, name in ((SONUC, Y01), (GOSTERGE, D01)):
        column = case.roles[role]
        data[name] = np.where(data[column].isna(), np.nan, (data[column] == case.levels[role]).astype(float))
    return data


def _grid(values: pd.Series) -> tuple[float, ...]:
    """Adım 3'ün değerleri: en çok 25 farklı değer varsa hepsi, yoksa %0, %5, …, %100 kantilleri (değişkenin standart
    sapmasına göre dört anlamlı basamak)."""

    data = values.dropna().astype(float)
    distinct = np.sort(data.unique())
    if len(distinct) <= 25:
        return tuple(float(value) for value in distinct)
    digits = int(max(0, 3 - math.floor(math.log10(float(data.std())))))
    points = np.round(np.quantile(data, np.linspace(0, 1, 21)), digits)
    return tuple(float(value) for value in dict.fromkeys(points.tolist()))


def own_plan(case: Case) -> Plan:
    name = case.name
    y, profile, indicator = (case.roles[role] for role in (SONUC, PROFIL, GOSTERGE))
    controls = tuple(case.extras)
    regressors = (profile, D01, *controls)
    pick_d = case.levels[GOSTERGE]
    labels = {column: name(column) for column in (y, profile, indicator, *controls)}
    labels.update({"(sabit)": "Sabit", Y01: f"{name(y)} = {case.levels[SONUC]}", D01: f"{name(indicator)} = {pick_d}",
                   "lpm": "LPM", "logit": "Logit", "probit": "Probit", "ame_logit": "Logit AME",
                   "ame_probit": "Probit AME"})
    terms = ((profile, name(profile)), (D01, f"{name(indicator)} = {pick_d}"),
             *((column, name(column)) for column in controls))
    # Gösterge 0/1 sayısal regresördür; AME'de türev yerine 1 − 0 farkı alınır (``discrete``). Kategorik tanımlansaydı
    # ondalık sütunda katsayının adı dile göre "[T.1.0]" ya da "1" olurdu. Yalnız 0 ve 1 değerli kontroller de sonlu
    # farkla girer (Adım 5'in kuralı).
    return Plan(
        frame=case.frame, outcome=Y01, profile=profile, indicator=D01, indicator_term=D01,
        regressors=regressors, categorical=(), ame_terms=regressors, terms=terms,
        discrete=(D01, *_binary_columns(case.data, controls)),
        grid=_grid(case.data[profile]),
        prepare=(*case.load,
                 Indicator(case.frame, Y01, y, case.levels[SONUC], "Sonuç: seçilen kategori 1, diğeri 0"),
                 Indicator(case.frame, D01, indicator, pick_d, "Gösterge: seçilen kategori 1, diğeri 0")),
        labels=labels,
        names={"profile": f"“{name(profile)}”", "indicator": f"“{name(indicator)}” = “{pick_d}”",
               "outcome": f"“{name(y)}” = “{case.levels[SONUC]}”"},
        profile_title=f"Logit modelinde “{name(profile)}” profili",
        profile_label=f"Ortalama tahmin edilen P({name(y)} = {case.levels[SONUC]})",
        decimals=_profile_decimals(case.data[profile]),
    )


def _binary_columns(data: pd.DataFrame, columns: tuple[str, ...]) -> tuple[str, ...]:
    """Yalnız 0 ve 1 değerlerini alan sütunlar."""

    return tuple(column for column in columns
                 if set(np.unique(data[column].dropna().astype(float))) <= {0.0, 1.0})


MAX_DECIMALS = 8
"""Profil değişkeninin katsayılarına ayrılan en çok ondalık; daha fazlası gerekiyorsa birimi değiştirmek istenir."""


def _profile_decimals(values: pd.Series) -> int:
    """Profil değişkeninin katsayılarının ondalığı: 4 + ⌊log₁₀ sd⌋ (en az 4). Bir standart sapmalık farkın olasılık
    etkisi tipik olarak 0,01–0,2 olduğundan birim başına etki 1/sd ölçeğindedir: yaşta (sd ≈ 14) beş, TL cinsinden
    gelirde (sd ≈ 30.000) sekiz ondalık. Ölçeğe göre seçilir, tahminin büyüklüğüne göre değil: ilişkisiz bir profilde
    katsayı tesadüfen sıfıra yakın olabilir."""

    spread = float(values.astype(float).std())
    if not math.isfinite(spread) or spread <= 0:
        return 4
    return int(4 + max(0, math.floor(math.log10(spread))))


def _own_texts(case: Case, plan: Plan) -> dict:
    y, profile = case.md(SONUC), case.md(PROFIL)
    pick = md(case.levels[SONUC])
    indicator, pick_d = case.md(GOSTERGE), md(case.levels[GOSTERGE])
    controls = [f"“{md(case.name(column))}”" for column in case.extras]

    def step1(state) -> str:
        frame = state.frames[plan.frame]
        share = float(frame[Y01].mean())
        ame = state.models["ame_logit"]
        lpm = state.models["lpm"]
        p_effect, d_effect = float(ame.params[plan.profile]), float(ame.params[plan.indicator_term])
        return (f"Analiz örneklemi {sayim(len(frame))} gözlem; “{y}” = “{pick}” olan gözlemlerin payı "
                f"{sayi(share, 3)}. Logit AME: “{profile}” bir birim yüksekken olasılık ortalama {katsayi(p_effect)} "
                f"(yaklaşık {sayi(100 * p_effect, max(2, plan.decimals - 2))} yüzde puan) farklı; “{indicator}” = "
                f"“{pick_d}” olanlarda, diğer "
                f"değişkenler aynıyken, ortalama {katsayi(d_effect)} farklı. LPM'de aynı iki sayı "
                f"{katsayi(float(lpm.params[plan.profile]))} ve {katsayi(float(lpm.params[plan.indicator_term]))}. Bunlar "
                "koşullu ilişkilerdir; nedensel yorum ayrı bir tanımlama varsayımı ister.")

    def step2(state) -> str:
        s = state.scalars
        return (f"Logit katsayısı {katsayi(s['logit_profil'])}, ortalama ölçek çarpanı Λ(1−Λ) {sayi(s['olcek_logit'], 4)}; "
                f"çarpımları AME'dir. Probit'te katsayı {katsayi(s['probit_profil'])}, ortalama φ "
                f"{sayi(s['olcek_probit'], 4)}; Logit/Probit katsayı oranı {sayi(s['katsayi_orani'], 2)}. " + SCALE_RULE)

    def step3(state) -> str:
        table = state.tables["profil"]["olasilik"]
        first, last = plan.grid[0], plan.grid[-1]
        return (f"“{profile}” = {_value(first)} için ortalama tahmin edilen olasılık {sayi(float(table.loc[first]), 3)}, "
                f"{_value(last)} için {sayi(float(table.loc[last]), 3)}. Profilin eğimi değerler boyunca sabit değildir: "
                "doğrusal olmayan modelde aynı katsayı olasılığın düzeyine göre farklı olasılık değişimi üretir.")

    def step5(state) -> str:
        s = state.scalars
        derivative, finite = s["gosterge_turev"], s["gosterge_fark"]
        return (f"“{indicator}” = “{pick_d}” göstergesi için türev tabanlı yaklaşım {katsayi(derivative)}, doğru "
                f"tanım olan sonlu fark "
                f"{katsayi(finite)}; fark {katsayi(finite - derivative)}. Katsayı büyüdükçe ya da olasılıklar 0 veya "
                "1'e yaklaştıkça iki yaklaşım arasındaki fark büyür.")

    controls_text = f" Ek kontroller: {', '.join(controls)}." if controls else " Ek kontrol seçilmedi."
    return {
        1: (
            f"Soru: “{y}” = “{pick}” olma olasılığı “{profile}” ve “{indicator}” ile nasıl ilişkili? Aynı soruyu üç "
            "modelle soruyoruz: doğrusal olasılık modeli (LPM), Logit ve Probit. LPM'de katsayı doğrudan olasılık "
            "farkıdır; Logit ve Probit'te ham katsayı tek indeks ölçeğindedir, karşılaştırma **ortalama marjinal etki "
            "(AME)** üzerinden yapılır. Kovaryans: LPM'de HC1, Logit/Probit'te sandviç (dayanıklı)." + controls_text,
            "LPM ile iki doğrusal olmayan modelin AME'leri genellikle birbirine yakındır; olasılıklar 0 ya da 1'e "
            "yaklaştıkça LPM'nin doğrusal yaklaşımı zayıflar.",
        ),
        (1, "not"): step1,
        2: (
            "Logit modelinde $P(Y_i=1\\mid X_i)=\\Lambda(X_i'\\beta)$ ve sürekli $X_j$ için marjinal etki "
            "$\\Lambda(X_i'\\beta)\\{1-\\Lambda(X_i'\\beta)\\}\\beta_j$'dir. Aynı $\\beta_j$ farklı gözlemlerde farklı "
            "olasılık etkisi üretir; AME bu etkilerin örneklem ortalamasıdır: "
            "$\\widehat{AME}_j=\\hat\\beta_j\\cdot\\frac1n\\sum_i g(X_i'\\hat\\beta)$.",
            SCALE_RULE,
        ),
        (2, "not"): step2,
        3: (PROFILE_RULE, "Profil S biçimli olabilir: eğim olasılığın 0,5'e yakın olduğu bölgede en büyüktür."),
        (3, "not"): step3,
        4: (AME_TABLE, "Logit ve Probit ham katsayıları farklı ölçekte olsa da AME'ler genellikle birbirine yakındır."),
        5: (FINITE_RULE, "Yazılımın marjinal etki komutunun göstergeleri nasıl ele aldığını her zaman kontrol edin."),
        (5, "not"): step5,
        6: ("", "Tez yazımında AME'yi yüzde puan olarak raporlayın ve gözlemsel veride ilişkiyi nedensel etki gibi "
                "yorumlamayın."),
        "kod4": "AME'ler iki dilde kodda tanımlı fonksiyonla hesaplanır: gösterge ve yalnız 0/1 değer alan kontroller için "
                "1 − 0 farkı (`kesikli`), sürekli değişkenler için türev.",
    }


def build(case: Case) -> LabSpec:
    """Kendi verinle Konu 5 uygulaması; kontrollerin beklenen değerleri uygulamanın hesabıdır."""

    plan = own_plan(case)
    texts = _own_texts(case, plan)
    spec = _spec(plan, texts, source="kendi", title=f"{case.label(SONUC)} ve {case.label(PROFIL)}", dataset="")
    spec = _own_code_notes(spec)
    return with_app_values(spec)


def _own_code_notes(spec: LabSpec) -> LabSpec:
    from dataclasses import replace

    steps = tuple(replace(step, code_note=CODE_NOTE_1_OWN) if step.number == 1 else step for step in spec.steps)
    return replace(spec, steps=steps)


def _separated(data: pd.DataFrame, y: str, regressors: tuple[str, ...]) -> bool:
    """Tam ya da yarı-tam ayrışma (Albert ve Anderson, 1984): her gözlemde sᵢxᵢ'b ≥ 0 olan ve en az bir gözlemde
    sıfırdan büyük olan bir b var mı (sᵢ = 2yᵢ − 1)? Varsa Logit'in MLE'si yoktur: katsayılar sonsuza gider ve diller
    farklı iterasyonda durup farklı sayı verir. Doğrusal programla kesin olarak denetlenir: sᵢxᵢ'b ≥ 0 (her i) ve
    Σᵢ sᵢxᵢ'b = 1 olurlu mu? Sütunlar standartlaştırılır; test birimlerden bağımsızdır."""

    from scipy.optimize import linprog

    complete = data[[y, *regressors]].dropna().astype(float)
    values = complete[list(regressors)].to_numpy()
    spread = values.std(axis=0)
    spread[spread == 0] = 1.0
    design = np.column_stack([np.ones(len(complete)), (values - values.mean(axis=0)) / spread])
    signed = design * (2.0 * complete[y].to_numpy() - 1.0)[:, None]
    result = linprog(np.zeros(design.shape[1]), A_ub=-signed, b_ub=np.zeros(len(signed)),
                     A_eq=signed.sum(axis=0)[None, :], b_eq=[1.0], bounds=[(None, None)] * design.shape[1],
                     method="highs")
    return result.status == 0


def _converges(data: pd.DataFrame, y: str, regressors: tuple[str, ...]) -> bool:
    """Logit Newton–Raphson ile yakınsıyor mu (ayrışma yokken MLE vardır ve tektir)."""

    import statsmodels.api as sm

    complete = data[[y, *regressors]].dropna().astype(float)
    design = sm.add_constant(complete[list(regressors)].to_numpy(), has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = sm.Logit(complete[y].to_numpy(), design).fit(disp=False, maxiter=100)
        except Exception:  # tekil Hessian
            return False
    return bool(fit.mle_retvals.get("converged", False)) and bool(np.isfinite(fit.params).all())


def validate(case: Case) -> None:
    y, profile, indicator = (case.roles[role] for role in (SONUC, PROFIL, GOSTERGE))
    if len({y, profile, indicator}) < 3:
        raise K.UploadError("Sonuç, profil değişkeni ve gösterge için üç farklı sütun seçin.")
    if any(column in case.extras for column in (y, profile, indicator)):
        raise K.UploadError("Ek kontrollerde sonuç, profil değişkeni ya da gösterge olmamalı.")
    data = _indicators(case)
    counts = data[Y01].value_counts()
    if len(counts) < 2 or counts.min() < MIN_CLASS:
        raise K.UploadError(f"“{case.name(y)}” sütununun iki kategorisinde de en az {MIN_CLASS} gözlem olmalı.")
    if data[profile].nunique() < 5:
        raise K.UploadError(f"“{case.name(profile)}” sütununda en az 5 farklı değer olmalı (olasılık profili bu "
                            "değerler boyunca çizilir).")
    if data[D01].value_counts().min() < 5:
        raise K.UploadError(f"“{case.name(indicator)}” sütununun iki kategorisinde de en az 5 gözlem olmalı.")
    for column in case.extras:
        if data[column].nunique() < 2:
            raise K.UploadError(f"“{case.name(column)}” sütununda en az iki farklı değer olmalı.")
    regressors = (profile, D01, *case.extras)
    if len(data) <= 2 * (len(regressors) + 1):
        raise K.UploadError(f"Modelde {len(regressors) + 1} katsayı var; {len(data)} gözlem yetmez.")
    if not full_rank(data, regressors):
        raise K.UploadError("Seçilen değişkenler arasında tam doğrusal bağlantı var. Ek kontrolleri değiştirin.")
    if not stable_design(data, Y01, (regressors,)):
        raise K.UploadError(SCALE_MESSAGE)
    if _profile_decimals(data[profile]) > MAX_DECIMALS:
        raise K.UploadError(f"“{case.name(profile)}” değişkeninin ölçeği çok büyük (standart sapması "
                            f"{sayim(float(data[profile].std()))}): bir birimlik farkın olasılığa etkisi çok küçük olur. "
                            "Dosyada birimini değiştirin (ör. TL yerine bin TL).")
    if _separated(data, Y01, regressors):
        raise K.UploadError("Logit modeli bu seçimlerle tahmin edilemiyor: bir değişken ya da değişkenlerin birleşimi "
                            "sonucu kesin olarak ayırıyor (tam ya da yarı-tam ayrışma; sonlu MLE yoktur, katsayılar ±∞'a ıraksar). Bu değişkeni "
                            "çıkarın ya da kategorileri birleştirin.")
    if not _converges(data, Y01, regressors):
        raise K.UploadError("Logit modeli bu seçimlerle yakınsamıyor. Ek kontrolleri azaltın ya da değişkenlerin "
                            "birimlerini değiştirin.")


def sample() -> pd.DataFrame:
    """Örnek dosya: kurgusal ikili sonuç verisi (``core.labs.ornek_veri``)."""

    return ikili_veri()


ROLES = (
    Role(SONUC, "İkili sonuç değişkeni", "kategorik", True, (1, 2, 3, 4, 5),
         "İki kategorili sonuç (ör. iş bulma: evet/hayır). Olasılığı modellenen kategoriyi seçin.",
         levels=(2, 2), pick="1 ile kodlanan kategori"),
    Role(PROFIL, "Profil değişkeni", "sayisal", True, (1, 2, 3, 4),
         "Olasılık profilinin çizileceği sayısal değişken (notlarda yaş). En az 5 farklı değer."),
    Role(GOSTERGE, "İki kategorili gösterge", "kategorik", True, (1, 4, 5),
         "Sonlu fark adımının göstergesi (notlarda siyah). 1 ile kodlanan kategoriyi seçin.",
         levels=(2, 2), pick="1 ile kodlanan kategori"),
)

CUSTOM = CustomLab(
    roles=ROLES,
    build=build,
    sample=sample,
    intro=(
        "Bir Excel (.xlsx) ya da CSV dosyası yükleyin. İki kategorili sonuç, olasılık profilinin çizileceği sayısal "
        "değişken ve iki kategorili bir gösterge seçilir; ek sayısal kontroller isteğe bağlıdır. Seçilen sütunların "
        "birinde boş hücresi olan satırlar analizden çıkarılır: bütün modeller aynı gözlemlerle tahmin edilir."
    ),
    min_rows=30,
    extra_columns=True,
    extra_use="sayisal",
    extra_label="Ek sayısal kontroller (isteğe bağlı, en çok 8)",
    extra_help="LPM, Logit ve Probit'e eklenir. Boş hücresi olan satırlar çıkarılır.",
    max_extra=8,
    extra_required=True,
    validate=validate,
)

VARIANTS = TopicVariants(alternative=alternative, story=STORY, custom=CUSTOM)
