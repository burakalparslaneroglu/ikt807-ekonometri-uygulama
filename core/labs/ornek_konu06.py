"""Konu 6 genel uygulaması: sıfır yığılmasından sansürlü modele.

Notlardaki §6.16'nın beş adımı aynı numaralarla ve aynı işlemlerle, verisi değiştirilebilir biçimde yazılır (``build``):
analiz örnekleminin kurulması ve sıfır payı, parçalı doğrusal spline ile OLS, LAD ve Tobit profilleri, Tobit'in üç
hedefi ve model kontrolü, sıfırların ekonomik anlamı ve sansürlü sonuç protokolü. Alternatif örnek Card ve Krueger
(1994) verisidir (Hansen'in arşivindeki ``CK1994.dta``): fast-food restoranlarında tam zamanlı çalışan sayısı ile
restoranın günde açık kaldığı saat. "Kendi verini yükle" seçeneğinde aynı adımlar öğrencinin dosyasıyla kurulur.
Notlardaki laboratuvar (``core.labs.konu06``) değişmez.

Rollerin notlardaki karşılıkları: sıfırda yığılan sonuç alınan transfer, ana açıklayıcı (spline ile) gelir, ek
kontroller yaş, eğitim, medeni durum, çocuklar, hane büyüklüğü ve istihdam göstergeleri.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
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
    md,
    sayi,
    sayim,
    stable_design,
    with_app_values,
    with_expected,
)
from core.labs.ornek_veri import bagis_verisi
from core.labs.spec import (
    OLS,
    Check,
    Derive,
    DropMissing,
    LabSpec,
    LabStep,
    LoadHansen,
    NoteRef,
    Operation,
    ProfileCurves,
    QuantileRegression,
    ReproClass,
    StatTarget,
    Summaries,
    TableTarget,
    Tobit,
    TobitFitCheck,
    TobitTargets,
)

TOPIC = "konu06"
SECTION = "6.16"
SONUC, ACIKLAYICI = "sonuc", "aciklayici"
ZERO = "sifir_yuzde"
"""Kendi verinde türetilen sıfır göstergesi (yüzde ölçeğinde) ve spline terimleri ``dugum1``, ``dugum2``, …
(``kendi_veri.RESERVED_CODES``: öğrencinin sütunlarına verilmez)."""
MIN_ZERO, MAX_ZERO = 0.05, 0.50
"""Kendi verinde sonucun sıfır payı bu aralıkta olmalı: az sıfırda sansürlü model gereksizdir; yarıdan fazlası sıfırsa
LAD'nin koşullu medyanı geniş bir bölgede sıfıra yapışır ve çözümü tek olmaz."""
MODELS = (("ols", "OLS"), ("tobit", "Tobit (gizli ortalama)"), ("lad", "LAD (medyan)"))
_CURVE_LABELS = {"ols": "OLS", "tobit": "Tobit gizli ortalama", "lad": "LAD medyan"}
_TARGET_LABELS = {"p_poz": "P(Y>0|x)", "gozlenen": "m(x)", "poz_ort": "m#(x)"}


@dataclass(frozen=True)
class Plan:
    """Rollerin koddaki sütunları, spline düğümleri, ızgara ve etiketler.

    ``point``: kontrol etiketlerinde ızgara değerinin yazımı (ör. "saat {}"); ``summary``: Adım 1 tablosunun (etiket,
    değişken, istatistik, ondalık) satırları; ``decimals``: profil eğrilerinin ve Tobit hedeflerinin ondalığı (olasılık
    hariç; o her zaman üç ondalık).
    """

    frame: str
    outcome: str
    x: str
    knots: tuple[float, ...]
    splines: tuple[str, ...]
    controls: tuple[str, ...]
    prepare: tuple[Operation, ...]
    summary: tuple[tuple[str, str, str, int], ...]
    grid: tuple[float, ...]
    plot_grid: tuple[float, float, int]
    labels: dict
    point: str
    x_label: str
    y_label: str
    title: str
    fit_labels: tuple[str, str, str, str]
    """Adım 3 model kontrolünün dört etiketi: model P(Y>0), veri pozitif payı, model E[Y], veri ortalaması."""
    decimals: int = 2
    lad_spread: float = 0.0
    """Izgara noktalarında, bütün optimal LAD çözümleri üzerinden tahmin edilen medyanın en geniş aralığı."""
    lad_checked: bool = True
    """LAD eğrisi kontrol edilir mi: her ızgara noktasında optimal çözümlerin aralığı, indirilen kodun karşılaştırdığı
    pencerenin (uygulamanın gösterim basamağına yuvarlanmış değeri ± 0,5·10⁻ᵈ) içinde kalmalı. Kalmazsa diller farklı
    köşe çözümü seçip farklı sayı verebilir; eğri gösterilir ama karşılaştırılmaz."""
    tobit_scale: float = 1.0
    """R'de Tobit'in sayısal kararlılık için sonucu böldüğü sayı (``Tobit.scale``)."""

    @property
    def regressors(self) -> tuple[str, ...]:
        return (self.x, *self.splines, *self.controls)

    def derived(self) -> tuple[tuple[str, E.Expr], ...]:
        return tuple((name, E.maximum(E.sub(E.var(self.x), knot), 0)) for name, knot in zip(self.splines, self.knots))


def _value(value: float) -> str:
    """Izgara değerinin etiketteki yazımı: tam sayılarda binlik ayırıcı, ondalıklarda gereken kadar basamak."""

    value = float(value)
    if value.is_integer():
        return ("−" if value < 0 else "") + sayim(abs(value))
    return np.format_float_positional(value, trim="-").replace("-", "−").replace(".", ",")


def _steps(plan: Plan, texts: dict) -> tuple[LabStep, ...]:
    f, y = plan.frame, plan.outcome
    curve_checks = tuple(
        Check(f"{_CURVE_LABELS[model]}, {plan.point.format(_value(value))}", TableTarget("egriler", value, model), 0.0,
              plan.decimals)
        for model, _ in MODELS for value in plan.grid if model != "lad" or plan.lad_checked
    )
    target_checks = tuple(
        Check(f"{_TARGET_LABELS[column]}, {plan.point.format(_value(value))}", TableTarget("hedefler", value, column),
              0.0, 3 if column == "p_poz" else plan.decimals)
        for column in _TARGET_LABELS for value in plan.grid
    )
    p_model, p_data, m_model, m_data = plan.fit_labels
    return (
        LabStep(
            number=1,
            title="Hansen'in veri hazırlama zincirini yeniden üretmek",
            note=NoteRef(SECTION, 1),
            explanation=texts[1][0],
            operations=(
                *plan.prepare,
                Derive(f, ZERO, E.mul(100, E.compare("eq", E.var(y), 0)),
                       "Sonucu sıfır olan gözlem göstergesi, yüzde ölçeğinde (ortalaması sıfır payıdır)"),
                Summaries(f, tuple((label, ZERO if stat == "sifir" else variable, "mean" if stat == "sifir" else stat)
                                   for label, variable, stat, _ in plan.summary), "ozet"),
            ),
            checks=tuple(
                Check(label, StatTarget(f, ZERO if stat == "sifir" else variable, "mean" if stat == "sifir" else stat),
                      0.0, decimals=decimals)
                for label, variable, stat, decimals in plan.summary
            ),
            reproducibility=ReproClass.EXACT,
            takeaway=texts[1][1],
            code_note=texts.get("kod1", ""),
            note_for=texts.get((1, "not")),
        ),
        LabStep(
            number=2,
            title="Gelir etkisini doğrusal olmak zorunda bırakmamak",
            note=NoteRef(SECTION, 2, ("Şekil 6.3", "Tablo 6.4")),
            explanation=texts[2][0],
            operations=(
                *(Derive(f, name, expression, f"Spline terimi {plan.labels[name]}") for name, expression in plan.derived()),
                OLS("ols", f, y, plan.regressors, vcov="HC1"),
                QuantileRegression("lad", f, y, plan.regressors, 0.5),
                Tobit("tobit", f, y, plan.regressors, 0.0, plan.tobit_scale),
                ProfileCurves(
                    models=MODELS,
                    frame=f,
                    variable=plan.x,
                    derived=plan.derived(),
                    values=plan.grid,
                    result="egriler",
                    plot_grid=plan.plot_grid,
                    x_label=plan.x_label,
                    y_label=plan.y_label,
                    title=plan.title,
                ),
            ),
            checks=curve_checks,
            reproducibility=ReproClass.CONVENTION,
            takeaway=texts[2][1],
            code_note=texts.get("kod2", CODE_NOTE_2),
            note_for=texts.get((2, "not")),
        ),
        LabStep(
            number=3,
            title="Tobit katsayısını OLS katsayısı gibi okumamak",
            note=NoteRef(SECTION, 3, ("Tablo 6.5",)),
            explanation=texts[3][0],
            operations=(
                TobitTargets("tobit", "egriler", "hedefler"),
                TobitFitCheck("tobit", "kontrol"),
            ),
            checks=(
                *target_checks,
                Check(p_model, TableTarget("kontrol", "p_poz", "model"), 0.0, 3),
                Check(p_data, TableTarget("kontrol", "p_poz", "veri"), 0.0, 3),
                Check(m_model, TableTarget("kontrol", "ortalama", "model"), 0.0, plan.decimals),
                Check(m_data, TableTarget("kontrol", "ortalama", "veri"), 0.0, plan.decimals),
            ),
            reproducibility=ReproClass.CONVENTION,
            takeaway=texts[3][1],
            note_for=texts.get((3, "not")),
        ),
        LabStep(
            number=4,
            title="Sıfırların ekonomik anlamını sorgulamak",
            note=NoteRef(SECTION, 4),
            explanation=texts[4][0],
            takeaway=texts[4][1],
            note_for=texts.get((4, "not")),
        ),
        LabStep(
            number=5,
            title="CLAD karşılaştırması ve tezinizde sansürlü sonuç protokolü",
            note=NoteRef("6.16.5", 0, ("§6.16.6",)),
            explanation=texts[5][0],
            takeaway=texts[5][1],
        ),
    )


def _spec(plan: Plan, texts: dict, *, source: str, title: str, dataset: str) -> LabSpec:
    return LabSpec(topic_key=TOPIC, title=title, dataset=dataset, note_section=SECTION, steps=_steps(plan, texts),
                   labels=tuple(plan.labels.items()), source=source)


# --- Ortak metinler -------------------------------------------------------------------------------------------

ESTIMATORS = (
    "Aynı spesifikasyon üç tahmin ediciyle: **OLS** gözlenen koşullu ortalamayı, **LAD** (medyan regresyonu) koşullu "
    "medyanı, **Tobit** ise normal-homoskedastik modelde gizli sonucun ortalamasını hedefler. Eğriler, kontroller "
    "örneklem ortalamasındayken $x'\\hat\\beta$'dır."
)
TARGETS = (
    "Tobit'teki $\\beta_j$ gizli sonuç $Y^*$ denkleminin eğimidir. Aynı katsayı setinden üç farklı hedef türetilir "
    "($z=x'\\beta/\\sigma$):\n\n"
    "1. gizli sonuç: $m^*(x)=x'\\beta$,\n"
    "2. sansürlenmeme olasılığı: $P(Y>0\\mid x)=\\Phi(z)$,\n"
    "3. gözlenen ortalama: $m(x)=\\Phi(z)\\,x'\\beta+\\sigma\\phi(z)$; pozitif gözlemlerde ise "
    "$m^{\\#}(x)=x'\\beta+\\sigma\\phi(z)/\\Phi(z)$.\n\n"
    "Aşağıda bu hedefler Adım 2'deki profillerde hesaplanıyor. Son olarak model bir kontrolden geçiyor: Tobit'in ima "
    "ettiği $P(Y>0)$ ve $\\mathbb E[Y]$ örneklem ortalamaları verideki karşılıklarıyla karşılaştırılıyor."
)
PROTOCOL = (
    "**Protokol:**\n\n"
    "1. Sıfır veya sınır değerinin veri üretim mekanizmasını açıklayın.\n"
    "2. Sansürleme ile truncation ve örneklem seçimini birbirinden ayırın.\n"
    "3. Gizli sonuç yorumunun ekonomik olarak anlamlı olup olmadığını tartışın.\n"
    "4. Tobit kullanıyorsanız normalite ve homoskedastisite varsayımlarının güçlü olduğunu açıkça belirtin.\n"
    "5. Mümkünse OLS/LAD/Tobit veya iki parçalı alternatiflerle duyarlılık gösterin.\n"
    "6. Tobit katsayısı yerine hangi marjinal etkinin raporlandığını belirtin.\n"
    "7. Örneklem seçiminde dışlama değişkeninin neden seçim denklemini etkileyip sonuç denklemini doğrudan "
    "etkilemediğini savunun."
)
FIT_RULE = (
    "Adım 3'teki model kontrolü (ima edilen ile gözlenen sıfır payı ve ortalama) her Tobit uygulamasında raporlanabilecek "
    "basit bir tanıdır: büyük bir uyumsuzluk, normal-homoskedastik varsayımın veriyle çeliştiğinin işaretidir."
)
CODE_NOTE_2 = (
    "OLS ve Tobit dillerde aynı sayıyı verir (Tobit: Python'da kodda tanımlı MLE, R `AER::tobit`, Stata `tobit, ll(0)`). "
    "LAD doğrusal programlamanın kesin çözümüdür (Python `scipy.optimize.linprog`, R `quantreg::rq`, Stata `qreg`). "
    "Çözüm tek değilse (R \"nonunique\" uyarısı verir) diller aynı amaç değerini veren farklı köşe çözümleri seçebilir. "
    "Kontrollerin ortalamaları analiz örnekleminden alınır."
)


# --- Alternatif örnek: Card ve Krueger (1994) -------------------------------------------------------------------

ALT_DATA = "ck1994"
ALT_FRAME = "ck"
ALT_TITLE = "Fast-Food Restoranlarında Tam Zamanlı Çalışan: Sıfır Yığılması ve Tobit"
ALT_KNOTS = (12.0, 16.0)
ALT_SPLINES = ("saat12", "saat16")
ALT_CONTROLS = ("kfc", "roys", "wendys", "co_owned", "state", "time")
ALT_GRID = (10.0, 12.0, 14.0, 16.0, 18.0, 24.0)


def alternative_plan() -> Plan:
    frame = ALT_FRAME
    chain = E.var("chain")
    return Plan(
        frame=frame, outcome="empft", x="hoursopen", knots=ALT_KNOTS, splines=ALT_SPLINES, controls=ALT_CONTROLS,
        prepare=(
            LoadHansen(ALT_DATA, "CK1994.dta", frame),
            DropMissing(frame, ("empft", "hoursopen"),
                        "Tam zamanlı çalışan sayısı ya da açık kalma süresi eksik gözlemler (ikinci turda kapanan "
                        "restoranlar dahil)"),
            Derive(frame, "kfc", E.compare("eq", chain, 2), "Zincir göstergesi: KFC (referans Burger King)"),
            Derive(frame, "roys", E.compare("eq", chain, 3), "Zincir göstergesi: Roy Rogers"),
            Derive(frame, "wendys", E.compare("eq", chain, 4), "Zincir göstergesi: Wendy's"),
        ),
        summary=(
            ("Gözlem sayısı (restoran × tur)", "empft", "count", 0),
            ("Sıfır payı (%)", "empft", "sifir", 1),
            ("Ortalama tam zamanlı çalışan", "empft", "mean", 2),
            ("Medyan tam zamanlı çalışan", "empft", "median", 1),
        ),
        grid=ALT_GRID,
        plot_grid=(10.0, 24.0, 141),
        labels={
            "(sabit)": "Sabit",
            "empft": "Tam zamanlı çalışan",
            "hoursopen": "Açık kalma süresi (saat/gün)",
            "kfc": "KFC",
            "roys": "Roy Rogers",
            "wendys": "Wendy's",
            "co_owned": "Şirkete ait",
            "state": "New Jersey",
            "time": "İkinci tur",
            "ols": "OLS",
            "lad": "LAD (medyan)",
            "tobit": "Tobit",
            ZERO: "Sıfır göstergesi (%)",
            **{name: f"(saat − {_value(knot)})₊" for name, knot in zip(ALT_SPLINES, ALT_KNOTS)},
        },
        point="saat {}",
        x_label="Açık kalma süresi (saat/gün)",
        y_label="Tahmin edilen tam zamanlı çalışan",
        title="CK1994: açık kalma süresi–tam zamanlı çalışan profili",
        fit_labels=("Model: ortalama P(Y>0)", "Veri: tam zamanlı çalışanı olanların payı",
                    "Model: ortalama E[Y]", "Veri: ortalama tam zamanlı çalışan"),
    )


ALT_TEXTS = {
    1: (
        "Card ve Krueger (1994), New Jersey'de asgari ücretin 1 Nisan 1992'de saatte 4,25 dolardan 5,05 dolara "
        "çıkarılmasının istihdama etkisini incelemek için New Jersey ve doğu Pennsylvania'daki 410 fast-food restoranında "
        "(Burger King, KFC, Roy Rogers, Wendy's) artıştan önce (Şubat–Mart 1992) ve sonra (Kasım–Aralık 1992) anket "
        "yaptı. Özgün dosyada her restoran tek satırdır; Hansen'in oluşturma dosyası (`CK1994_create.do`) iki turu alt "
        "alta koyar (820 satır; `time` = 0 ilk, 1 ikinci tur) ve iki kez kullanılan 407 numarasını Pennsylvania'daki "
        "restoran için 408 yapar. **Hansen'in veri arşivindeki `CK1994.dta` bu adımların sonucudur.** Bu uygulamanın "
        "sorusu asgari ücretin etkisi değildir: tam zamanlı çalışan sayısı (`empft`) restoranın günde kaç saat açık "
        "olduğuyla (`hoursopen`) nasıl değişiyor?\n\n"
        "İkinci turda kalıcı olarak kapanan 6 restoranın çalışan sayıları 0, açık kalma süresi boş olarak kayıtlıdır "
        "(Card'ın kod kitabında `status2` = 3; Hansen'in dosyasında bu durum değişkeni yoktur). Bu satırlar ve çalışan "
        "sayısı ya da açık kalma süresi eksik diğer satırlar analizden çıkar: 820 satırdan 795 gözlem kalır. Sıfırda "
        "yığılan sonuç $Y=empft\\ge0$'dır; her restoran en çok iki kez gözlenir."
    ),
    2: (
        "Açık kalma süresi parçalı doğrusal spline ile esnekleştirilir: düğümler günde 12 ve 16 saatte (analiz "
        "örnekleminde birinci ve üçüncü çeyrek); her düğüm için $(saat-k)_+=\\max\\{saat-k,0\\}$ terimi eklenir. "
        "Kontroller: zincir göstergeleri (referans Burger King), şirkete ait olma, New Jersey ve ikinci tur.\n\n"
        + ESTIMATORS
    ),
    3: TARGETS,
    4: (
        "Bu veride sıfır, gizli bir sonucun sansürlenmesinden çok bir köşe çözümüne benzer. Tam zamanlı çalışanı olmayan "
        "restoranlar işi daha çok yarı zamanlı çalışanlarla yürütür: bu restoranlarda ortalama 25,1 yarı zamanlı "
        "çalışan var, tam zamanlı çalışanı olanlarda 17,5. Negatif bir \"gizli\" tam zamanlı çalışan talebi ekonomik "
        "bir anlam taşımaz; sıfır, tam ve yarı zamanlı çalışan karması seçiminin bir sonucu olabilir. Tek denklemli "
        "Tobit, \"hiç tam zamanlı çalıştırmama\" kararı ile \"kaç tam zamanlı çalıştırma\" kararını aynı "
        "parametrelerle açıklamaya zorlar; iki parçalı bir model (önce katılım, sonra miktar) daha esnek bir "
        "alternatiftir."
    ),
    5: (
        "Hansen notlardaki örnekte sansürlü LAD (CLAD) tahminini de raporlar ve güçlü çarpıklık ile sansürleme nedeniyle "
        "CLAD'ı dayanıklı bir alternatif olarak tartışır. Bu uygulama CLAD'ı hesaplamaz; amaç, model varsayımları "
        "değiştiğinde estimand'ın ve tahmin edicinin de değişebileceğini görmektir.\n\n" + PROTOCOL
    ),
}
"""Adım açıklamaları; anlatılan sayılar kontrollerdir (``ALT_EXPECTED``) ya da testte bağımsız olarak hesaplanır."""

ALT_TAKEAWAYS = {
    1: (
        "Restoran-turların %18,1'inde hiç tam zamanlı çalışan yok; ortalama 8,29, medyan 6: dağılım sağa çarpıktır. "
        "Kalıcı olarak kapanan restoranların sıfırları başka bir mekanizmadır: orada restoran yoktur. Onları analizde "
        "tutmak, \"tam zamanlı çalıştırmama kararı\" ile \"kapanmayı\" aynı sıfırda birleştirirdi. Ham veriden analiz "
        "örneklemine geçiş yeniden üretilemezse, doğru yazılımla hesaplanmış bir Tobit katsayısı bile araştırmayı "
        "güvenilir kılmaz."
    ),
    2: (
        "OLS eğrisi 10 saatte 3,45, 12 saatte 6,38, 16 saatte 9,19, 24 saatte 23,38 tam zamanlı çalışan verir: 12 saate "
        "kadar her ek saat yaklaşık 1,5 çalışanla, 12–16 saat arasında yaklaşık 0,7, 16 saatten sonra yaklaşık 1,8 "
        "çalışanla ilişkilidir. Üç eğri de saatle artar ve 16 saatten sonra dikleşir; düzeyler farklıdır: LAD medyanı "
        "(10 saatte 2,80, 24 saatte 21,21) 10–24 saat aralığının her noktasında OLS ortalamasının altındadır, çünkü "
        "dağılım sağa çarpıktır. 24 saat değeri yalnız 11 restorana dayanır (19 ile 24 saat arasında gözlem yok). Bu "
        "ilişkiler betimseldir: "
        "uzun saat açık kalan restoranlar konum ve talep bakımından da farklıdır; zincir göstergeleri bu farkların yalnız "
        "bir kısmını tutar."
    ),
    3: (
        "Aynı Tobit tahmini 16 saatte gizli ortalama olarak 8,32, gözlenen ortalama olarak 9,20, tam zamanlı çalışanı "
        "olan restoranlarda 11,21 verir; restoranın en az bir tam zamanlı çalışanı olma olasılığı 10 saatte 0,591, 24 "
        "saatte 0,994. Model kontrolündeki uyumsuzluk notlardakinden çok daha küçüktür: Tobit ortalamada P(Y>0) = 0,778 "
        "öngörürken verideki pay 0,819; ima edilen ortalama 8,56, gözlenen 8,29 (notlarda 0,579'a karşı 0,824 ve "
        "11,46'ya karşı 7,71). Ama iki ortalamanın yakın olması gizli sonuç yorumunu doğrulamaz: sıfırın neyi temsil "
        "ettiği (Adım 4) veriyle değil ekonomik argümanla cevaplanır."
    ),
    4: (
        "$Y$'de çok sayıda sıfır görmek tek başına Tobit seçme gerekçesi değildir. Sıfırın sansürleme, gerçek ekonomik "
        "köşe çözümü, iki aşamalı katılım kararı veya veri kaydı mekanizmasından hangisine karşılık geldiği "
        "açıklanmalıdır. Kalıcı olarak kapanan restoranlar (Adım 1) bu ayrımın uç örneğidir: o sıfırlar ne sansür ne köşe "
        "çözümüdür."
    ),
    5: FIT_RULE,
}

ALT_CODE_NOTES = {
    "kod1": (
        "Card'ın özgün `public.dat` dosyasıyla çalışıyorsanız önce Hansen'in `CK1994_create.do` adımlarını uygulayın; "
        "arşivdeki `CK1994.dta` bu adımların sonucudur."
    ),
    "kod2": (
        "OLS ve Tobit üç dilde aynı sayıyı verir (Tobit: Python'da kodda tanımlı MLE, R `AER::tobit`, Stata "
        "`tobit, ll(0)`). LAD doğrusal programlamanın kesin çözümüdür (Python `scipy.optimize.linprog`, R "
        "`quantreg::rq`, Stata `qreg`); bu örneklemde çözüm tektir, eğriler dillerde aynıdır. Kontrollerin ortalamaları "
        "analiz örnekleminden (795 gözlem) alınır. Adımlar standart hata kullanmaz; çıkarım yapılacaksa her restoran iki "
        "kez gözlendiği için standart hatalar restorana göre kümelenmelidir."
    ),
}

ALT_EXPECTED = {
    (1, "Gözlem sayısı (restoran × tur)"): 795,
    (1, "Sıfır payı (%)"): 18.1,
    (1, "Ortalama tam zamanlı çalışan"): 8.29,
    (1, "Medyan tam zamanlı çalışan"): 6.0,
    **{(2, f"{label}, saat {int(hour)}"): value
       for label, values in (("OLS", (3.45, 6.38, 7.79, 9.19, 12.74, 23.38)),
                             ("Tobit gizli ortalama", (2.08, 5.42, 6.87, 8.32, 11.96, 22.89)),
                             ("LAD medyan", (2.80, 4.47, 5.99, 7.50, 10.93, 21.21)))
       for hour, value in zip(ALT_GRID, values)},
    **{(3, f"{label}, saat {int(hour)}"): value
       for label, values in (("P(Y>0|x)", (0.591, 0.725, 0.776, 0.821, 0.907, 0.994)),
                             ("m(x)", (4.75, 6.95, 8.04, 9.20, 12.36, 22.91)),
                             ("m#(x)", (8.03, 9.58, 10.36, 11.21, 13.63, 23.04)))
       for hour, value in zip(ALT_GRID, values)},
    (3, "Model: ortalama P(Y>0)"): 0.778,
    (3, "Veri: tam zamanlı çalışanı olanların payı"): 0.819,
    (3, "Model: ortalama E[Y]"): 8.56,
    (3, "Veri: ortalama tam zamanlı çalışan"): 8.29,
}
"""Kontrollerin CK1994 analiz örneklemindeki (795 restoran-tur) değerleri, gösterim basamağında: (adım, etiket) → değer.
Metinlerdeki sayılar bunlardır; testler bağımsız bir hesapla ve iki dilde doğrular."""


@cache
def alternative() -> LabSpec:
    """Alternatif örneğin tanımı; beklenen değerler CK1994'ün tam verisindeki sayılardır (testle doğrulanır)."""

    texts = {number: (ALT_TEXTS[number], ALT_TAKEAWAYS[number]) for number in ALT_TEXTS}
    texts.update(ALT_CODE_NOTES)
    spec = _spec(alternative_plan(), texts, source="alternatif", title=ALT_TITLE, dataset=ALT_DATA)
    return with_expected(spec, ALT_EXPECTED)


STORY = (
    "Alternatif örnek Card ve Krueger (1994) verisidir: New Jersey ve Pennsylvania'da 410 fast-food restoranı, iki tur "
    "(Hansen'in arşivindeki CK1994.dta). Sıfırda yığılan sonuç tam zamanlı çalışan sayısı, profil değişkeni restoranın "
    "günde açık kaldığı saattir. Adımlar notlardaki gibidir: analiz örneklemi ve sıfır payı, spline ile OLS, LAD ve Tobit "
    "profilleri, Tobit'in üç hedefi ve model kontrolü, sıfırların ekonomik anlamı."
)


# --- Kendi verin --------------------------------------------------------------------------------------------

def _round_digits(values: pd.Series) -> int:
    """Düğüm ve ızgara değerlerinin ondalığı: değişkenin standart sapmasına göre dört anlamlı basamak (0–0,01 aralığında
    ölçülen bir değişkende de çeyrekler ayrışsın)."""

    spread = float(values.std())
    if not math.isfinite(spread) or spread <= 0:
        return 2
    return int(max(0, 3 - math.floor(math.log10(spread))))


def _quantile_knots(values: pd.Series) -> tuple[float, ...]:
    """Düğümler: çeyrekler (%25, %50, %75), ölçeğe göre yuvarlanmış; tekrarlananlar ve uç değere eşit olanlar çıkar."""

    data = values.dropna().astype(float)
    low, high = float(data.min()), float(data.max())
    points = np.round(np.quantile(data, (0.25, 0.5, 0.75)), _round_digits(data))
    return tuple(float(k) for k in dict.fromkeys(points.tolist()) if low < k < high)


def _grid(values: pd.Series, knots: tuple[float, ...]) -> tuple[float, ...]:
    """Profil değerleri: %5 kantili, düğümler ve %95 kantili (ölçeğe göre yuvarlanmış, artan sırada, tekrarsız)."""

    data = values.dropna().astype(float)
    ends = np.round(np.quantile(data, (0.05, 0.95)), _round_digits(data))
    return tuple(sorted(dict.fromkeys([float(ends[0]), *knots, float(ends[1])])))


def _decimals(scale: float) -> int:
    """Eğri ve ortalamaların ondalığı: ortalaması 1–10 arasında olan sonuçta 2; on kat büyüklükte bir eksik."""

    if not math.isfinite(scale) or scale <= 0:
        return 2
    return int(min(6, max(0, 2 - math.floor(math.log10(scale)))))


def _tobit_scale(values: pd.Series) -> float:
    """R'de Tobit'in sonucu böldüğü sayı: standart sapma 1.000 ve üstündeyse 10'un kuvveti (bölünmüş sonucun standart
    sapması 1–10 arasında); yoksa 1. ``survival::survreg`` standart sapması 10⁶ düzeyindeki sonuçta katsayıları tekil
    sayıp NA verebiliyor."""

    spread = float(values.astype(float).std())
    if not math.isfinite(spread) or spread < 1000:
        return 1.0
    return float(10 ** math.floor(math.log10(spread)))


def own_plan(case: Case) -> Plan:
    name = case.name
    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    data = case.data
    knots = _quantile_knots(data[x])
    splines = tuple(f"dugum{i}" for i in range(1, len(knots) + 1))
    controls = tuple(case.extras)
    decimals = _decimals(float(data[y].astype(float).mean()))
    x_name = name(x)
    labels = {column: name(column) for column in (y, x, *controls)}
    labels.update({"(sabit)": "Sabit", "ols": "OLS", "lad": "LAD (medyan)", "tobit": "Tobit",
                   ZERO: "Sıfır göstergesi (%)",
                   **{spline: f"({x_name} − {_value(knot)})₊" for spline, knot in zip(splines, knots)}})
    values = data[x].dropna().astype(float)
    return Plan(
        frame=case.frame, outcome=y, x=x, knots=knots, splines=splines, controls=controls, prepare=case.load,
        summary=(
            ("Gözlem sayısı", y, "count", 0),
            ("Sıfır payı (%)", y, "sifir", 1),
            ("Ortalama", y, "mean", decimals),
            ("Medyan", y, "median", decimals),
        ),
        grid=_grid(data[x], knots),
        plot_grid=(float(values.min()), float(values.max()), 201),
        labels=labels,
        point="“" + x_name.replace("{", "{{").replace("}", "}}") + "” = {}",  # str.format kalıbı
        x_label=x_name,
        y_label=f"Tahmin edilen {name(y)}",
        title=f"“{x_name}” profili: OLS, Tobit ve LAD",
        fit_labels=("Model: ortalama P(Y>0)", "Veri: pozitif sonuç payı", "Model: ortalama E[Y]", "Veri: ortalama"),
        decimals=decimals,
        tobit_scale=_tobit_scale(data[y]),
    )


def _own_texts(case: Case, plan: Plan) -> dict:
    y, x = case.md(SONUC), case.md(ACIKLAYICI)
    knots = "; ".join(_value(k) for k in plan.knots)
    controls = [f"“{md(case.name(column))}”" for column in case.extras]
    controls_text = f" Ek kontroller: {', '.join(controls)}." if controls else " Ek kontrol seçilmedi."

    def step1(state) -> str:
        frame = state.frames[plan.frame]
        values = frame[plan.outcome].astype(float)
        share = float((values == 0).mean())
        return (f"Analiz örneklemi {sayim(len(frame))} gözlem; “{y}” sıfır olan gözlemlerin payı "
                f"%{sayi(100 * share, 1)}. Ortalama {sayi(float(values.mean()), plan.decimals)}, medyan "
                f"{sayi(float(values.median()), plan.decimals)}: ortalama medyandan ne kadar büyükse sağ kuyruk o kadar "
                "uzundur.")

    def step2(state) -> str:
        table = state.tables["egriler"]
        first, last = plan.grid[0], plan.grid[-1]
        parts = [f"{label}: {sayi(float(table.loc[first, model]), plan.decimals)} → "
                 f"{sayi(float(table.loc[last, model]), plan.decimals)}" for model, label in MODELS]
        unique = ("" if plan.lad_checked else
                  " Bu veride LAD çözümü tek değil: aynı amaç değerini veren çözümler arasında tahmin edilen medyan "
                  f"bir ızgara noktasında {sayi(plan.lad_spread, plan.decimals + 1)} kadar değişebiliyor. Diller farklı "
                  "çözüm seçebileceği için LAD eğrisi indirilen kodda karşılaştırılmaz.")
        return (f"“{x}” = {_value(first)} ile {_value(last)} arasında tahmin edilen eğriler — " + "; ".join(parts)
                + ". Üç tahmin edicinin hedefi farklıdır (ortalama, gizli ortalama, medyan); düzey farkı bu yüzden "
                "\"hangisi doğru\" sorusuna indirgenemez." + unique)

    def step3(state) -> str:
        check = state.tables["kontrol"]
        return (f"Tobit ortalamada P(Y>0) = {sayi(float(check.loc['p_poz', 'model']), 3)} öngörüyor, verideki pozitif "
                f"pay {sayi(float(check.loc['p_poz', 'veri']), 3)}; ima edilen ortalama "
                f"{sayi(float(check.loc['ortalama', 'model']), plan.decimals)}, gözlenen "
                f"{sayi(float(check.loc['ortalama', 'veri']), plan.decimals)}. " + FIT_RULE)

    return {
        1: (
            f"Sonuç “{y}”, negatif olmayan ve bir kısmı sıfır olan bir değişkendir. Seçilen sütunlarda boş hücresi olan "
            "satırlar analizden çıkarılır; aşağıdaki özet analiz örnekleminindir. Sıfır payı ve ortalama–medyan farkı, "
            "sıfır yığılmasının ve sağ kuyruğun ölçüsüdür.",
            "Ham veriden analiz örneklemine geçiş yeniden üretilemezse, doğru yazılımla hesaplanmış bir Tobit katsayısı "
            "bile araştırmayı güvenilir kılmaz.",
        ),
        (1, "not"): step1,
        2: (
            f"“{x}” ile ilişki parçalı doğrusal spline ile esnekleştirilir: düğümler “{x}” değişkeninin çeyreklerinde "
            f"({knots}); her düğüm $k$ için $(x-k)_+=\\max\\{{x-k,0\\}}$ terimi eklenir." + controls_text + "\n\n"
            + ESTIMATORS,
            "Eğrilerin biçimi üç tahmin edicide benzer, düzeyleri farklı olabilir: tahmin hedefleri (ortalama, medyan, "
            "gizli ortalama) farklıdır.",
        ),
        (2, "not"): step2,
        3: (TARGETS, "\"Tobit katsayısı\" tek başına hangi hedefin yorumlandığını söylemez."),
        (3, "not"): step3,
        4: (
            f"“{y}” değişkenindeki sıfır neyi temsil ediyor? Gizli bir sonucun sıfırın altında kalıp sıfır olarak "
            "kaydedilmesi (sansürleme), bir köşe çözümü (ör. hiç harcama yapmama kararı) ya da iki aşamalı bir karar "
            "(önce katılım, sonra miktar) farklı modeller ister.",
            "$Y$'de çok sayıda sıfır görmek tek başına Tobit seçme gerekçesi değildir. Sıfırın sansürleme, gerçek ekonomik "
            "köşe çözümü, iki aşamalı katılım kararı veya veri kaydı mekanizmasından hangisine karşılık geldiği "
            "açıklanmalıdır.",
        ),
        5: (
            "Sansürlü LAD (CLAD) normal dağılım ve homoskedastisite varsayımına dayanmadan koşullu medyanı tahmin eder; "
            "bu uygulama CLAD'ı hesaplamaz. Amaç, model varsayımları değiştiğinde estimand'ın ve tahmin edicinin de "
            "değişebileceğini görmektir.\n\n" + PROTOCOL,
            FIT_RULE,
        ),
    }


def _design(data: pd.DataFrame, plan: Plan) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uygulamanın tasarımı: sabit, açıklayıcı, spline terimleri ve kontroller; ızgarada kontroller analiz örnekleminin
    ortalamasında (``ProfileCurves`` ile aynı). Dönüş: X, y ve ızgara satırları."""

    frame = data.copy()
    for name, expression in plan.derived():
        frame[name] = np.asarray(E.evaluate(expression, frame), dtype=float)
    columns = list(plan.regressors)
    complete = frame[[plan.outcome, *columns]].dropna().astype(float)
    x = np.column_stack([np.ones(len(complete)), complete[columns].to_numpy()])
    grid = pd.DataFrame({plan.x: np.asarray(plan.grid, dtype=float)})
    for name, expression in plan.derived():
        grid[name] = np.asarray(E.evaluate(expression, grid), dtype=float)
    for name in plan.controls:
        grid[name] = float(complete[name].mean())
    points = np.column_stack([np.ones(len(grid)), grid[columns].to_numpy()])
    return x, complete[plan.outcome].to_numpy(), points


def lad_guard(data: pd.DataFrame, plan: Plan) -> tuple[float, bool]:
    """LAD eğrisinin dillerde aynı çıkıp çıkmayacağı. Her ızgara noktasında bütün optimal LAD çözümleri üzerinden
    tahmin edilen medyanın aralığı [en küçük, en büyük] hesaplanır (``quantreg.fitted_value_range``). İndirilen kod
    dilin bulduğu değeri, uygulamanın değerinin gösterim basamağına yuvarlanmışıyla ± 0,5·10⁻ᵈ toleransla karşılaştırır;
    aralık bu pencerenin içindeyse hangi çözümü seçerse seçsin her dil kontrolü geçer. Dönüş: en geniş aralık ve
    kontrol edilip edilemeyeceği."""

    from core.labs import quantreg as Q

    x, y, points = _design(data, plan)
    fitted = points @ Q.rq_fit(x, y, 0.5)
    ranges = Q.fitted_value_range(x, y, 0.5, points)
    window = 0.5 * 10 ** (-plan.decimals)
    rounded = np.array([float(f"{value:.{plan.decimals}f}") for value in fitted])
    margin = 1e-9 * np.maximum(1.0, np.abs(rounded))
    inside = (ranges[:, 0] >= rounded - window + margin) & (ranges[:, 1] <= rounded + window - margin)
    return float((ranges[:, 1] - ranges[:, 0]).max()), bool(inside.all())


def build(case: Case) -> LabSpec:
    """Kendi verinle Konu 6 uygulaması; kontrollerin beklenen değerleri uygulamanın hesabıdır. LAD çözümü gösterim
    basamağında tek değilse LAD eğrisi kontrol edilmez (diller farklı köşe çözümü seçebilir)."""

    plan = own_plan(case)
    spread, checked = lad_guard(case.data, plan)
    plan = replace(plan, lad_spread=spread, lad_checked=checked)
    spec = _spec(plan, _own_texts(case, plan), source="kendi",
                 title=f"{case.label(SONUC)} ve {case.label(ACIKLAYICI)}", dataset="")
    return with_app_values(spec)


def _spline_frame(data: pd.DataFrame, x: str, knots: tuple[float, ...]) -> tuple[pd.DataFrame, list[str]]:
    frame = data.copy()
    names = []
    for i, knot in enumerate(knots, start=1):
        frame[f"dugum{i}"] = np.maximum(frame[x].astype(float) - knot, 0.0)
        names.append(f"dugum{i}")
    return frame, names


MIN_Z = -30.0
"""Profil noktalarında Tobit'in z = x'β/σ değerinin alt sınırı. Daha küçükte Φ(z) ve φ(z) sayısal olarak sıfıra çok
yakındır; pozitiflerde ortalama φ/Φ dillerde hesaplanamayabilir."""


def _tobit_feasible(case: Case, plan: Plan) -> None:
    """Tobit tahmin edilebiliyor mu ve profil noktalarında hedefleri hesaplanabiliyor mu."""

    from core.labs import limited as L

    x, y, points = _design(case.data, plan)
    try:
        beta, sigma, _, _ = L.tobit_newton(y, x, 0.0)
    except (RuntimeError, np.linalg.LinAlgError) as error:
        raise K.UploadError("Tobit modeli bu veriyle tahmin edilemiyor: sonuç açıklayıcıların neredeyse tam bir "
                            "fonksiyonu olabilir. Değişkenleri kontrol edin.") from error
    spread = float(np.std(y))
    if not np.isfinite(beta).all() or not math.isfinite(sigma) or sigma <= 1e-6 * spread:
        raise K.UploadError("Tobit modelinin hata standart sapması sıfıra çok yakın: sonuç açıklayıcıların neredeyse tam "
                            "bir fonksiyonu. Bu veride sansürlü model anlamlı değil.")
    if float((points @ beta / sigma).min()) < MIN_Z:
        raise K.UploadError("Tobit modeli profilin bazı noktalarında sonucun pozitif olma olasılığını sayısal olarak sıfır "
                            "buluyor; hedefler bu noktalarda hesaplanamaz. Açıklayıcı değişkeni ya da kontrolleri "
                            "değiştirin.")


def validate(case: Case) -> None:
    y, x = case.roles[SONUC], case.roles[ACIKLAYICI]
    if y == x:
        raise K.UploadError("Sonuç ve ana açıklayıcı için iki farklı sütun seçin.")
    if any(column in case.extras for column in (y, x)):
        raise K.UploadError("Ek kontrollerde sonuç ya da ana açıklayıcı olmamalı.")
    data = case.data
    values = data[y].astype(float)
    if (values < 0).any():
        raise K.UploadError(f"“{case.name(y)}” sütununda negatif değer var. Bu uygulama sıfırda yığılan, negatif olmayan "
                            "bir sonuç ister (soldan sıfırda sansür).")
    share = float((values == 0).mean())
    if share < MIN_ZERO:
        raise K.UploadError(f"“{case.name(y)}” sütununda sıfırların payı %{sayi(100 * share, 1)}; en az "
                            f"%{sayi(100 * MIN_ZERO, 0)} olmalı. Sıfır yığılması yoksa sansürlü modele gerek yoktur.")
    if share > MAX_ZERO:
        raise K.UploadError(f"“{case.name(y)}” sütununda sıfırların payı %{sayi(100 * share, 1)}; en çok "
                            f"%{sayi(100 * MAX_ZERO, 0)} olmalı. Gözlemlerin yarısından fazlası sıfırsa LAD'nin koşullu "
                            "medyanı sıfıra yapışır ve çözüm tek olmaz.")
    if data[x].nunique() < 10:
        raise K.UploadError(f"“{case.name(x)}” sütununda en az 10 farklı değer olmalı (spline düğümleri çeyreklerde).")
    knots = _quantile_knots(data[x])
    if not knots:
        raise K.UploadError(f"“{case.name(x)}” sütununun çeyrekleri en küçük ya da en büyük değere eşit (değerlerin "
                            "çoğu tek bir değerde toplanmış): spline düğümü kurulamıyor.")
    for column in case.extras:
        if data[column].nunique() < 2:
            raise K.UploadError(f"“{case.name(column)}” sütununda en az iki farklı değer olmalı.")
    frame, splines = _spline_frame(data, x, knots)
    regressors = (x, *splines, *case.extras)
    if len(data) <= 5 * (len(regressors) + 1):
        raise K.UploadError(f"Modelde {len(regressors) + 1} katsayı var; {len(data)} gözlem yetmez (en az "
                            f"{5 * (len(regressors) + 1) + 1}).")
    if not full_rank(frame, regressors):
        raise K.UploadError("Seçilen değişkenler arasında tam doğrusal bağlantı var. Ek kontrolleri değiştirin.")
    if not stable_design(frame, y, (regressors,)):
        raise K.UploadError(SCALE_MESSAGE)
    _tobit_feasible(case, own_plan(case))


def sample() -> pd.DataFrame:
    """Örnek dosya: kurgusal bağış verisi (``core.labs.ornek_veri``)."""

    return bagis_verisi()


ROLES = (
    Role(SONUC, "Sıfırda yığılan sonuç", "sayisal", True, (1, 2, 3, 4),
         "Negatif olmayan, bir kısmı sıfır olan sonuç (notlarda alınan transfer). Sıfır payı %5–%50 olmalı."),
    Role(ACIKLAYICI, "Ana açıklayıcı değişken", "sayisal", True, (2, 3),
         "Profili çizilen sayısal değişken (notlarda gelir); düğümleri çeyreklerinde olan parçalı doğrusal spline ile "
         "girer. En az 10 farklı değer."),
)

CUSTOM = CustomLab(
    roles=ROLES,
    build=build,
    sample=sample,
    intro=(
        "Bir Excel (.xlsx) ya da CSV dosyası yükleyin. Sıfırda yığılan, negatif olmayan bir sonuç ve profili çizilecek "
        "sayısal bir açıklayıcı seçilir; ek sayısal kontroller isteğe bağlıdır. Seçilen sütunların birinde boş hücresi "
        "olan satırlar analizden çıkarılır: OLS, LAD ve Tobit aynı gözlemlerle tahmin edilir."
    ),
    min_rows=50,
    extra_columns=True,
    extra_use="sayisal",
    extra_label="Ek sayısal kontroller (isteğe bağlı, en çok 8)",
    extra_help="Üç modele de eklenir; profillerde örneklem ortalamasında tutulur. Boş hücresi olan satırlar çıkarılır.",
    max_extra=8,
    extra_required=True,
    validate=validate,
)

VARIANTS = TopicVariants(alternative=alternative, story=STORY, custom=CUSTOM)
