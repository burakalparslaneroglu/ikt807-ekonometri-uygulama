"""Konu 9 genel uygulaması: RDD sonuç tablosunu tasarım olarak okumak.

Notlardaki §9.15'in beş adımı aynı numaralarla ve aynı işlemlerle, verisi değiştirilebilir biçimde yazılır (``build``):
tasarım satırları, bant genişliği duyarlılık tablosu, bant genişliği ölçeği (aynı h, üç pencere), eşiğin iki yanında
yerel doğrusal eğriler ve makale okuma listesi. Alternatif örnek Angrist ve Lavy (1999) verisidir (Hansen'in
arşivindeki ``AL1999.dta``): İsrail'de 5. sınıflar ve Maimonides kuralı (bir sınıfta en çok 40 öğrenci). Sınıf
düzeyindeki kayıt 41'e ulaşınca ikinci şube açılır; eşik değişkeni kayıt, eşik 41, sonuç sınıfın okuma puanı. Kural
(eşiği geçme) keskin, sınıf mevcudu bulanıktır: tahmin edilen sıçrama kuralın okuma puanına etkisidir (indirgenmiş
biçim); ilk aşama sıçraması Adım 1'de gösterilir. Aynı okulun sınıfları aynı kaydı paylaştığı için standart hatalar
okul düzeyinde kümelenir (RDD işleminin ``cluster`` seçeneği). "Kendi verini yükle" seçeneğinde aynı adımlar öğrencinin
dosyasıyla kurulur. Notlardaki laboratuvar (``core.labs.konu09``) değişmez.

Rollerin notlardaki karşılıkları: sonuç Head Start ilişkili ölüm oranı, eşik değişkeni 1960 yoksulluk oranı, eşik
59,1984; küme notlarda yoktur (HC1). Kendi verinde eşik ve ana bant genişliği yazılır; tablo ana bandın 0,5–1,5 katıdır.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd

from core.labs import expr as E
from core.labs import kendi_veri as K
from core.labs.ornek import (
    Case,
    CustomLab,
    NumberInput,
    Option,
    Role,
    TopicVariants,
    deger,
    ek,
    exact_fit,
    sayi,
    sayim,
    with_app_values,
    with_expected,
)
from core.labs.ornek_veri import burs_verisi
from core.labs.spec import (
    RDD,
    Check,
    CoefTarget,
    Derive,
    DropMissing,
    EffectTable,
    GroupRank,
    KeepIf,
    LabSpec,
    LabStep,
    LoadHansen,
    ModelTarget,
    NoteRef,
    Operation,
    Plot,
    RDDCurve,
    RDDTable,
    ReproClass,
    Scalar,
    ScalarTarget,
    StatTarget,
    Summaries,
    TableTarget,
    VLine,
)


def _sayi(value: float, decimals: int = 0) -> str:
    """Metinlerde Türkçe sayı, binlik ayırıcıyla (12.345,6)."""

    return sayi(value, decimals, binlik=True)


TOPIC = "konu09"
SECTION = "9.15"
SONUC, ESIK, KUME = "sonuc", "esik_degiskeni", "kume"
TARAF = "sol"
"""Seçenek: tedavi eşiğin solunda (X < c)."""
FACTORS = (0.5, 0.75, 1.0, 1.25, 1.5)
"""Kendi verinde bant genişliği tablosu: ana bandın katları."""
COLUMNS = (("n", "etkin örneklem n_h"), ("tahmin", "τ̂"), ("sh", "SH"), ("alt", "alt %95"), ("ust", "üst %95"))


@dataclass(frozen=True)
class Plan:
    """Rollerin koddaki sütunları, eşik, bant genişlikleri ve etiketler.

    ``bandwidths`` Adım 2'nin tablosudur; ``h_main`` (tablonun ortasındaki h) Adım 3 ve 4'ün bant genişliğidir.
    ``right``: tedavi eşiğin sağında mı (X ≥ c). ``first_stage``: verilirse Adım 1'de bu sütunun eşikteki sıçraması
    (bulanık tasarımın ilk aşaması) tahmin edilir. ``decimals``: sıçrama, SH ve limitlerin ondalığı; ``h_decimals``:
    bant genişliklerinin, ``window_decimals``: pencere genişliklerinin ondalığı.
    """

    frame: str
    x: str
    y: str
    cutoff: float
    cluster: str | None
    prepare: tuple[Operation, ...]
    bandwidths: tuple[float, ...]
    labels: dict
    x_label: str
    y_label: str
    titles: dict
    """``tablo`` (Adım 2) ve ``grafik`` (Adım 4) başlıkları."""
    x_range: tuple[float, float]
    right: bool = True
    first_stage: str | None = None
    decimals: int = 2
    h_decimals: int = 0
    window_decimals: tuple[int, int] = (1, 2)
    summary_label: str = "Gözlem sayısı"

    @property
    def h_main(self) -> float:
        return self.bandwidths[len(self.bandwidths) // 2]


def _value(value: float) -> str:
    """Sayının metindeki yazımı, gereken en az ondalıkla ve binlik ayırıcıyla (8; 4,5; 59,1984; 1.250)."""

    return deger(value)


def _model(plan: Plan, h: float) -> str:
    return "rdd_h" + np.format_float_positional(float(h), trim="-").replace(".", "_")


def _vcov_text(plan: Plan) -> str:
    return "küme SH" if plan.cluster else "HC1 SH"


def _table_checks(plan: Plan) -> tuple[Check, ...]:
    return tuple(
        Check(f"h = {_value(h)}: {label}", TableTarget("bant", h, column), 0.0, 0 if column == "n" else plan.decimals)
        for h in plan.bandwidths
        for column, label in COLUMNS
    )


def _steps(plan: Plan, texts: dict) -> tuple[LabStep, ...]:
    f, x, y, c, h = plan.frame, plan.x, plan.y, plan.cutoff, plan.h_main
    rdd = {"cluster": plan.cluster}
    design = [
        *plan.prepare,
        Derive(f, "esik_sagi", E.compare("ge", E.var(x), c), f"Eşiğin sağı: D = 1{{{plan.x_label} ≥ {_value(c)}}}"),
    ]
    rows = [
        (plan.summary_label, x, "count"),
        ("Eşiğin sağındaki gözlem", "esik_sagi", "sum"),
        (f"{plan.x_label}: en küçük", x, "min"),
        (f"{plan.x_label}: en büyük", x, "max"),
    ]
    checks1 = [
        Check(f"Analiz örneklemi ({plan.summary_label.lower()})", StatTarget(f, x, "count"), 0.0, decimals=0),
        Check("Eşiğin sağındaki gözlem", StatTarget(f, "esik_sagi", "sum"), 0.0, decimals=0),
    ]
    if plan.cluster:
        design.append(GroupRank(f, "kume_sira", plan.cluster, "Küme sıra numarası (küme sayısını saymak için)"))
        rows.append(("Küme sayısı", "kume_sira", "max"))
        checks1.append(Check("Küme sayısı", StatTarget(f, "kume_sira", "max"), 0.0, decimals=0))
    design.append(Summaries(f, tuple(rows), "tasarim"))
    if plan.first_stage:
        design.append(RDD("ilk_asama", f, x, plan.first_stage, c, h, **rdd))
        checks1 += [
            Check(f"İlk aşama: {plan.labels[plan.first_stage].lower()} sıçraması (h = {_value(h)})",
                  CoefTarget("ilk_asama", "D"), 0.0, decimals=plan.decimals),
            Check("İlk aşama: SH", CoefTarget("ilk_asama", "D", "se"), 0.0, decimals=plan.decimals),
        ]
    rect, wide = _value(round(h * math.sqrt(3), plan.window_decimals[1])), _value(round(h * math.sqrt(6),
                                                                                   plan.window_decimals[0]))
    return (
        LabStep(
            number=1,
            title="Katsayıdan önce tasarım satırlarını okuyun",
            note=NoteRef(SECTION, 1),
            explanation=texts[1][0],
            operations=tuple(design),
            checks=tuple(checks1),
            reproducibility=ReproClass.CONVENTION if plan.first_stage and plan.cluster else ReproClass.EXACT,
            takeaway=texts[1][1],
            note_for=texts.get((1, "not")),
        ),
        LabStep(
            number=2,
            title="Bant genişliği tablosunu sonuç avcılığı için değil duyarlılık için kullanmak",
            note=NoteRef(SECTION, 2, ("Tablo 9.2", "Tablo 9.1")),
            explanation=texts[2][0],
            operations=(
                *(RDD(_model(plan, bw), f, x, y, c, bw, **rdd) for bw in plan.bandwidths),
                RDDTable(
                    tuple((bw, _model(plan, bw)) for bw in plan.bandwidths), "bant",
                    "Bant genişliği h (Hansen ölçeği; üçgen çekirdekte pencere ±h√6)", "Eşikte tahmini sıçrama τ̂",
                    plan.titles["tablo"], decimals=plan.decimals,
                ),
            ),
            checks=_table_checks(plan),
            reproducibility=ReproClass.CONVENTION,
            takeaway=texts[2][1],
            code_note=texts.get("kod2", ""),
            note_for=texts.get((2, "not")),
        ),
        LabStep(
            number=3,
            title=texts.get("baslik3", "Bant genişliği ölçeği: aynı h, üç pencere"),
            note=NoteRef(SECTION, 3, ("§9.7",)),
            explanation=texts[3][0],
            operations=(
                Scalar("sol", E.coef(_model(plan, h), E.INTERCEPT), f"Eşiğin solu (h = {_value(h)})", percent=False,
                       decimals=plan.decimals),
                Scalar("sag", E.add(E.coef(_model(plan, h), E.INTERCEPT), E.coef(_model(plan, h), "D")),
                       f"Eşiğin sağı (h = {_value(h)})", percent=False, decimals=plan.decimals),
                Scalar("pencere_ucgen", E.mul(h, E.sqrt(6)), f"Üçgen pencere {_value(h)}√6", percent=False,
                       decimals=plan.window_decimals[0]),
                Scalar("pencere_dikdortgen", E.mul(h, E.sqrt(3)), f"Dikdörtgen pencere {_value(h)}√3", percent=False,
                       decimals=plan.window_decimals[1]),
                RDD("rdd_dikdortgen", f, x, y, c, h, kernel="rectangular", **rdd),
                RDD("rdd_pencere", f, x, y, c, h, scale="window", **rdd),
                EffectTable(
                    (
                        (f"Üçgen, h = {_value(h)} (±{wide})", _model(plan, h), "D"),
                        (f"Dikdörtgen, h = {_value(h)} (±{rect})", "rdd_dikdortgen", "D"),
                        (f"Üçgen, pencere ±{_value(h)}", "rdd_pencere", "D"),
                    ),
                    "olcek", title=f"Aynı \"h = {_value(h)}\", üç pencere: eşikteki sıçrama",
                    se_label="SH (küme)" if plan.cluster else "SH (HC1)",
                ),
            ),
            checks=(
                Check(f"Eşiğin solunda tahmin (h = {_value(h)})", ScalarTarget("sol"), 0.0, decimals=plan.decimals),
                Check(f"Eşiğin sağında tahmin (h = {_value(h)})", ScalarTarget("sag"), 0.0, decimals=plan.decimals),
                Check(f"Üçgen çekirdek penceresi {_value(h)}√6", ScalarTarget("pencere_ucgen"), 0.0,
                      decimals=plan.window_decimals[0]),
                Check(f"Dikdörtgen çekirdek penceresi {_value(h)}√3", ScalarTarget("pencere_dikdortgen"), 0.0,
                      decimals=plan.window_decimals[1]),
                Check("Dikdörtgen çekirdek: τ̂", CoefTarget("rdd_dikdortgen", "D"), 0.0, decimals=plan.decimals),
                Check("Dikdörtgen çekirdek: SH", CoefTarget("rdd_dikdortgen", "D", "se"), 0.0, decimals=plan.decimals),
                Check("Dikdörtgen çekirdek: gözlem sayısı", ModelTarget("rdd_dikdortgen", "nobs"), 0.0, decimals=0),
                Check(f"Pencere ±{_value(h)}: τ̂", CoefTarget("rdd_pencere", "D"), 0.0, decimals=plan.decimals),
                Check(f"Pencere ±{_value(h)}: SH", CoefTarget("rdd_pencere", "D", "se"), 0.0, decimals=plan.decimals),
                Check(f"Pencere ±{_value(h)}: gözlem sayısı", ModelTarget("rdd_pencere", "nobs"), 0.0, decimals=0),
            ),
            reproducibility=ReproClass.CONVENTION,
            takeaway=texts[3][1],
            note_for=texts.get((3, "not")),
        ),
        LabStep(
            number=4,
            title="RDD grafiğini \"güzel sıçrama\" diye değil belirsizlikle birlikte okumak",
            note=NoteRef(SECTION, 4, ("Şekil 9.1",)),
            explanation=texts[4][0],
            operations=(
                Plot(
                    f, x,
                    (
                        RDDCurve(y, c, h, f"Yerel doğrusal tahmin (h = {_value(h)}) ve %95 güven bandı",
                                 cluster=plan.cluster),
                        VLine(c, f"Eşik c = {_value(c)}"),
                    ),
                    plan.x_label, plan.y_label, plan.titles["grafik"], x_range=plan.x_range,
                ),
            ),
            reproducibility=ReproClass.EXACT,
            takeaway=texts[4][1],
            code_note=texts.get("kod4", CODE_NOTE_4),
            note_for=texts.get((4, "not")),
        ),
        LabStep(
            number=5,
            title="Bir RDD makalesini değerlendirmek için kısa kontrol listesi",
            note=NoteRef("9.15.5", 0),
            explanation=CHECKLIST,
            takeaway=texts[5][1],
            note_for=texts.get((5, "not")),
        ),
    )


def _spec(plan: Plan, texts: dict, *, source: str, title: str, dataset: str) -> LabSpec:
    labels = {
        **plan.labels,
        E.INTERCEPT: "Sabit (eşiğin solunda limit)",
        "esik_sagi": "Eşiğin sağı (D)",
        "D": "Eşikteki sıçrama (D)",
        "R": "R = X − c",
        "DR": "D·R",
        **{_model(plan, bw): f"Üçgen çekirdek, h = {_value(bw)}" for bw in plan.bandwidths},
        "rdd_dikdortgen": f"Dikdörtgen çekirdek, h = {_value(plan.h_main)}",
        "rdd_pencere": f"Üçgen çekirdek, pencere ±{_value(plan.h_main)}",
        "ilk_asama": "İlk aşama",
    }
    return LabSpec(topic_key=TOPIC, title=title, dataset=dataset, note_section=SECTION, steps=_steps(plan, texts),
                   labels=tuple(labels.items()), source=source)


# --- Ortak metinler -------------------------------------------------------------------------------------------

DESIGN_ROWS = (
    "Bir RDD tablosunda katsayıdan önce tasarımı tanımlayan satırlar okunur: eşik $c$; eşik değişkeninin (running "
    "variable) tanımı ve yönü; tasarımın keskin (sharp) mı bulanık (fuzzy) mı olduğu; çekirdek türü; yerel polinom "
    "derecesi; ana bant genişliği ve etkin örneklem büyüklüğü; standart hata ve yanlılık düzeltmesi yöntemi.\n\n"
)
CHECKLIST = (
    "1. Eşik kuralı kurumsal olarak açık mı?\n"
    "2. Birimler eşik değişkenini hassas biçimde manipüle edebilir mi?\n"
    "3. Tedavi olasılığı eşikte gerçekten değişiyor mu?\n"
    "4. Önceden belirlenmiş kovaryatlarda ve plasebo sonuçlarda sıçrama var mı?\n"
    "5. Ana bant genişliği nasıl seçildi?\n"
    "6. Alternatif makul bant genişliklerindeki sonuçlar gösterildi mi?\n"
    "7. Global yüksek dereceli polinom yerine yerel yöntem kullanılmış mı?\n"
    "8. Tahmin eşik çevresindeki birimler için yerel etki olarak mı yorumlanıyor?"
)
CODE_NOTE_4 = (
    "Eğriler her tarafta 120 noktada hesaplanır; her nokta ayrı bir ağırlıklı EKK'dir. Python'da NumPy ile vektörel, "
    "R'de `sapply`, Stata'da Mata ile."
)


def _model_text(plan: Plan) -> str:
    """Adım 2'nin model açıklaması (notlardaki gibi), standart hata türüyle."""

    se = ("Standart hata ağırlıklı regresyonun küme-dayanıklı sandviçidir (CR1: kümelerde toplanmış skorlar, "
          "$G/(G-1)\\cdot(n-1)/(n-k)$ çarpanı)" if plan.cluster else
          "Standart hata ağırlıklı regresyonun HC1 standart hatasıdır")
    hs = ";\\,".join(_value(bw).replace(",", "{,}") for bw in plan.bandwidths)
    return (
        "Yerel doğrusal RDD, pencere içindeki gözlemlerde\n\n"
        "$$Y_i=\\alpha+\\tau D_i+\\beta_1R_i+\\beta_2D_iR_i+u_i,\\qquad R_i=X_i-c,$$\n\n"
        "regresyonunu çekirdek ağırlıklarıyla tahmin eder; $\\widehat\\tau$ eşikteki sıçramadır. Hansen çekirdekleri "
        "birim varyansa ölçekler: $h$ çekirdeğin standart sapmasıdır. Birim varyanslı üçgen çekirdek\n\n"
        "$$K(u)=\\frac{1}{\\sqrt6}\\Bigl(1-\\frac{|u|}{\\sqrt6}\\Bigr)\\mathbf 1\\{|u|\\le\\sqrt6\\}$$\n\n"
        "olup gözlem $i$'nin ağırlığı $K(R_i/h)$'dir: ağırlık eşikten $h\\sqrt6$ uzaklıkta sıfıra iner. $n_h$ pozitif "
        f"ağırlık alan gözlem sayısıdır. {se}; güven aralığı $\\widehat\\tau\\pm1{{,}}96\\cdot\\text{{SH}}$'dir. Aynı "
        f"tahmin $h\\in\\{{{hs}\\}}$ için tekrarlanır."
    )


def _scale_text(plan: Plan, what: str) -> str:
    h = _value(plan.h_main)
    return (
        f"Eşikteki iki limit tahmini aynı regresyondan okunur: solda $\\widehat\\alpha$, sağda "
        f"$\\widehat\\alpha+\\widehat\\tau$ ({what}). Aynı bant genişliğinin dikdörtgen çekirdek karşılığı "
        f"$\\pm{h.replace(',', '{,}')}\\sqrt3$ genişliğinde bir penceredir; bu pencerede $D$, $R$ ve $D\\cdot R$ üzerine "
        f"basit EKK ikinci tahmini verir.\n\n"
        f"Aynı \"$h={h.replace(',', '{,}')}$\" pencerenin yarı genişliği olarak okunursa (ölçeklenmemiş üçgen çekirdek) "
        f"pencere $\\pm{h.replace(',', '{,}')}$'{ek(h, 'e')} daralır. Bu bir yöntem farkı değil ölçek farkıdır: yazılımlar ve "
        "makaleler bant genişliğini farklı ölçeklerde raporlayabilir."
    )


# --- Alternatif örnek: Angrist ve Lavy (1999) -----------------------------------------------------------------

ALT_DATA = "al1999"
ALT_FRAME = "al"
ALT_TITLE = "Maimonides Kuralında RDD Sonuç Tablosunu Tasarım Olarak Okumak"
ALT_CUTOFF = 41.0
ALT_BANDWIDTHS = (4.0, 6.0, 8.0, 10.0, 12.0)


def alternative_plan() -> Plan:
    frame = ALT_FRAME
    return Plan(
        frame=frame, x="enrollment", y="avgverb", cutoff=ALT_CUTOFF, cluster="schlcode",
        prepare=(
            LoadHansen(ALT_DATA, "AL1999.dta", frame),
            KeepIf(frame, (("grade", "==", 5),), "Analiz örneklemi: 5. sınıflar"),
            DropMissing(frame, ("enrollment", "avgverb", "classize", "schlcode"),
                        "Kayıt, okuma puanı, sınıf mevcudu veya okul kodu eksik sınıflar"),
        ),
        bandwidths=ALT_BANDWIDTHS,
        labels={
            "enrollment": "Sınıf düzeyinde kayıt (5. sınıf)",
            "avgverb": "Sınıfın ortalama okuma puanı",
            "classize": "Sınıf mevcudu",
            "schlcode": "Okul",
            "kume_sira": "Okul sıra numarası",
        },
        x_label="5. sınıf kaydı (öğrenci)",
        y_label="Sınıfın ortalama okuma puanı",
        titles={"tablo": "AL1999: bant genişliği duyarlılığı (okuma puanı)",
                "grafik": "AL1999: kayıt eşiğinin iki yanında yerel doğrusal tahmin"},
        x_range=(21.0, 61.0),
        first_stage="classize",
        summary_label="Sınıf sayısı",
    )


def _alt_explanations() -> dict:
    plan = alternative_plan()
    return {
        1: (
            DESIGN_ROWS
            + "Angrist ve Lavy (1999) İsrail'de 1991'de 4. ve 5. sınıflarda yapılan ulusal sınavları sınıf düzeyinde "
            "kullanır. **Maimonides kuralı** bir sınıfta en çok 40 öğrenciye izin verir: okulun 5. sınıf kaydı $e$ ise "
            "şube sayısı $\\lfloor(e-1)/40\\rfloor+1$, kuralın öngördüğü sınıf mevcudu "
            "$e/(\\lfloor(e-1)/40\\rfloor+1)$'dir. Kayıt 40'tan 41'e çıkınca kural ikinci şubeyi açtırır ve öngörülen "
            "mevcut 40'tan 20,5'e düşer.\n\n"
            "Birim sınıftır. Eşik değişkeni $X$ okulun 5. sınıf kaydı, eşik $c=41$, gösterge "
            "$D=\\mathbf 1\\{X\\ge41\\}$ (kayıt tam sayıdır). Sonuç sınıfın ortalama okuma (dil) puanıdır (`avgverb`). "
            "Analiz örneklemi Hansen'in `AL1999.dta` dosyasındaki 5. sınıflardır. Aynı okulun sınıfları aynı kaydı "
            "paylaşır ve ortak okul koşullarından etkilenir: standart hatalar okul düzeyinde kümelenir.\n\n"
            "**Keskin mi, bulanık mı?** Kural kayıtla kesin olarak belirlenir: eşiği geçme göstergesi $D$ keskindir. "
            "Gerçek sınıf mevcudu ise kurala tam uymaz; eşikteki mevcut sıçraması bulanık RDD'nin ilk aşamasıdır (§9.12). "
            "Bu laboratuvar notlardaki gibi keskin tasarımın adımlarını izler: tahmin edilen sıçrama eşiği geçmenin "
            "(kuralın) okuma puanına etkisidir, yani indirgenmiş biçimdir; sınıf mevcudunun öğrenci başına etkisi değildir."
        ),
        2: _model_text(plan) + (
            "\n\nBurada eşik değişkeni ve $h$ öğrenci sayısıyla ölçülür: $h=8$ eşiğe 19,6 öğrenciden yakın kayıtlara "
            "pozitif ağırlık verir. En geniş pencere ($h=12$, $\\pm29{,}4$) kuralın üçüncü şubeyi açtırdığı 81 "
            "kayda ulaşmaz."
        ),
        3: _scale_text(plan, "eşikte tahmini okuma puanları"),
        4: (
            "Ana RDD grafiğinde eşiğin iki yanında ayrı yerel doğrusal eğriler ve güven bantları gösterilir. Her $x_0$ "
            "noktasında yalnız o taraftaki sınıflarla üçgen çekirdekli ($h=8$, pencere $\\pm8\\sqrt6$) ağırlıklı EKK'nin "
            "sabit terimi hesaplanır; bant noktasal %95 güven aralığıdır (okul düzeyinde kümelenmiş SH). Grafik 21–61 "
            "kayıt aralığındadır: daha sağda eğrinin penceresi 81'deki ikinci eşiğe ulaşır. Görsel değerlendirmede üç "
            "şey birlikte aranır:\n\n"
            "1. eşikte koşullu ortalamada sıçrama,\n"
            "2. eşik çevresinde yeterli veri desteği,\n"
            "3. güven bantlarının sıçrama büyüklüğüne kıyasla genişliği."
        ),
    }


ALT_TAKEAWAYS = {
    1: (
        "2.018 sınıf 1.001 okuldadır; 1.724'ü eşiğin sağındadır (kayıt ≥ 41): büyük okulların birden çok şubesi olduğu "
        "için sınıfların çoğu yüksek kayıtlı okullardadır, eşiğe yakın kayıtlarda iki tarafta da yeterli sınıf vardır "
        "(Adım 2'deki $n_h$). Kural mevcudu 40'tan 20,5'e indirirken gerçek mevcut eşikte yaklaşık 8,9 öğrenci düşer "
        "(SH 1,94): kural güçlü ama kısmen uygulanır, tasarım sınıf mevcudu için bulanıktır. Kontrol listesinin 3. "
        "sorusu (\"tedavi olasılığı eşikte gerçekten değişiyor mu?\") burada sayıyla cevaplanır."
    ),
    2: (
        "Bant genişliği arttıkça etkin örneklem büyür (315'ten 960 sınıfa), standart hata düşer (3,20'den 1,84'e) ve "
        "tahmin küçülme eğilimindedir (h = 6'da 5,83, h = 12'de 4,26). $h=4$'te güven aralığı sıfırı içerir "
        "[−0,78; 11,77]; $h\\ge6$'da içermez. Tablo \"$h=6$ en büyük etkiyi verdiği için tercih edilir\" ya da "
        "\"$h=4$'te anlamlı değil, etki yok\" diye okunmaz: her satır farklı bir yerellik–varyans dengesi kurar. Ana "
        "sonuç, makul bant aralığında kuralın okuma puanını artırdığı yönündedir (4,3–5,8 puan; sınıfların ortalama "
        "okuma puanlarının standart sapması 7,7). Bu kuralın etkisidir, sınıf mevcudunun değil: bulanık RDD'nin Wald oranı (§9.12) "
        "$h=8$'de 5,20/(−8,90) ≈ −0,58 puan/öğrenci olurdu; onun standart hatası ve ilk aşamanın gücü ayrıca "
        "raporlanmalıdır."
    ),
    3: (
        "Birim varyanslı üçgen çekirdekle $h=8$ eşiğe 19,6 öğrenciden yakın 667 sınıfa pozitif ağırlık verir; sıçrama "
        "5,20 (2,26), eşiğin solunda tahmini okuma puanı 67,88, sağında 73,08. Dikdörtgen çekirdek aynı yerellikte "
        "(±13,86; 460 sınıf) 5,90 (2,47) verir. Pencere ±8'e daraltılırsa 259 sınıf kalır ve tahmin 4,75 (3,47) olur: "
        "nokta tahmini az değişir, belirsizlik belirgin biçimde büyür. Replikasyonda sayılar tutmadığında önce tanım "
        "farkları aranır: analiz örneklemi, eşik değişkeninin merkezlenmesi, eşik, çekirdek, bant genişliğinin "
        "ölçeği, yerel polinom derecesi, kovaryatlar ve standart hata türü (burada okul kümeli)."
    ),
    4: (
        "Eşiğin sağında tahmini koşullu ortalama daha yüksektir: kuralın ikinci şubeyi açtırdığı okullarda okuma "
        "puanı eşikte sıçrar. Eğrilerin eşikteki değerleri Adım 3'teki 67,88 ve 73,08'dir. Bantlar geniştir ve eşikte "
        "daha da genişler: sınırda gözlemler tek taraftadır ve kümeli SH aynı okulun sınıflarını tek bilgi kaynağı "
        "sayar. Belirgin görünen bir sıçrama, bant genişliği ve standart hata dikkate alınmadığında aşırı ikna edici "
        "görünebilir."
    ),
    5: (
        "Bu tasarımın cevapları: kural yasaldır ve açıktır (1); kaydın eşiğe göre ayarlanması bu veride de bir "
        "endişedir: 5. sınıf kaydı 40 olan 7 okula karşı 41 olan 16 okul vardır; Otsu, Xu ve Matsushita (2013) 1991 "
        "verisinde ilk eşikte yığılma bulur; Angrist, Lavy, Leder-Luis ve Shany (2019, *AER: Insights*) 2002–2011 "
        "verisinde eşiklerin hemen üstünde belirgin yığılma bulur ama tahminlerin bu yığılmanın ürünü olmadığını savunur "
        "(2); sınıf mevcudu eşikte yaklaşık 8,9 öğrenci düşer (3); önceden belirlenmiş "
        "değişkenler burada sınanmadı: okulun dezavantajlı öğrenci yüzdesi (`disadvantaged`) için aynı RDD tahmin "
        "edilerek sınanır (4); ana bant notlardaki gibi "
        "önceden sabitlenmiştir ve tablo duyarlılık olarak sunulur (5–6); yöntem yereldir (7); tahmin kaydı 41 "
        "civarındaki okulların sınıfları için yerel bir etkidir (8). Makale dili için örnek: \"Kaydın eşik çevresinde "
        "sonuçla ilişkili biçimde ayarlanmadığı varsayımı altında, yerel doğrusal RDD tahminleri kayıt eşiği "
        "civarındaki okullarda Maimonides kuralının ikinci şubeyi açtırmasının 5. sınıf okuma puanını yaklaşık 4–6 puan "
        "artırdığını göstermektedir; tahmin bant genişliğine duyarlı olmakla birlikte işaret makul bant aralığında "
        "korunmaktadır. Eşikteki kayıt yığılması bu varsayımın ayrıca sınanmasını gerektirir. Bu, kuralın etkisidir; "
        "sınıf mevcudunun etkisi için bulanık RDD gerekir.\""
    ),
}

ALT_CODE_NOTE_2 = (
    "Her bant genişliğinde aynı ağırlıklı regresyon, okul düzeyinde kümelenmiş SH ile: Python'da `statsmodels` "
    "`WLS(...).fit(cov_type=\"cluster\")`, R'de `lm(..., weights = w)` ve `sandwich::vcovCL(type = \"HC1\")`, Stata'da "
    "`regress ... [aw = w], vce(cluster schlcode)`; üç yazılım aynı küme düzeltmesini ($G/(G-1)\\cdot(n-1)/(n-k)$) "
    "kullanır ve aynı sayıları verir. Hazır RDD paketlerinin (ör. rdrobust) varsayılanları farklıdır."
)
ALT_CODE_NOTE_4 = (
    "Eğriler her tarafta 120 noktada hesaplanır; her nokta ayrı bir ağırlıklı EKK'dir ve standart hatası okul "
    "düzeyinde kümelenmiştir (skorlar okullarda toplanır). Python'da NumPy ile vektörel, R'de `sapply` ve `rowsum`, "
    "Stata'da Mata ile."
)


def alternative_texts() -> dict:
    explanations = _alt_explanations()
    texts: dict = {number: (explanations.get(number, CHECKLIST), ALT_TAKEAWAYS[number]) for number in range(1, 6)}
    texts["baslik3"] = "Bant genişliği ölçeği: aynı \"h = 8\", üç pencere"
    texts["kod2"] = ALT_CODE_NOTE_2
    texts["kod4"] = ALT_CODE_NOTE_4
    return texts


ALT_EXPECTED = {
    (1, "Analiz örneklemi (sınıf sayısı)"): 2018,
    (1, "Eşiğin sağındaki gözlem"): 1724,
    (1, "Küme sayısı"): 1001,
    (1, "İlk aşama: sınıf mevcudu sıçraması (h = 8)"): -8.90,
    (1, "İlk aşama: SH"): 1.94,
    **{(2, f"h = {h}: {label}"): value
       for h, row in ((4, (315, 5.50, 3.20, -0.78, 11.77)), (6, (495, 5.83, 2.62, 0.69, 10.96)),
                      (8, (667, 5.20, 2.26, 0.77, 9.63)), (10, (826, 4.59, 1.99, 0.68, 8.50)),
                      (12, (960, 4.26, 1.84, 0.65, 7.87)))
       for (_, label), value in zip(COLUMNS, row)},
    (3, "Eşiğin solunda tahmin (h = 8)"): 67.88,
    (3, "Eşiğin sağında tahmin (h = 8)"): 73.08,
    (3, "Üçgen çekirdek penceresi 8√6"): 19.6,
    (3, "Dikdörtgen çekirdek penceresi 8√3"): 13.86,
    (3, "Dikdörtgen çekirdek: τ̂"): 5.90,
    (3, "Dikdörtgen çekirdek: SH"): 2.47,
    (3, "Dikdörtgen çekirdek: gözlem sayısı"): 460,
    (3, "Pencere ±8: τ̂"): 4.75,
    (3, "Pencere ±8: SH"): 3.47,
    (3, "Pencere ±8: gözlem sayısı"): 259,
}
"""Kontrollerin AL1999 5. sınıf örneklemindeki (2.018 sınıf) değerleri, gösterim basamağında: (adım, etiket) → değer.
Metinlerdeki sayılar bunlardır; testler bağımsız bir hesapla ve iki dilde doğrular."""


@cache
def alternative() -> LabSpec:
    """Alternatif örneğin tanımı; beklenen değerler AL1999'un 5. sınıf örneklemindeki sayılardır (testle doğrulanır)."""

    spec = _spec(alternative_plan(), alternative_texts(), source="alternatif", title=ALT_TITLE, dataset=ALT_DATA)
    return with_expected(spec, ALT_EXPECTED)


STORY = (
    "Alternatif örnek Angrist ve Lavy (1999) verisidir: İsrail'de 1.001 okulda 2.018 beşinci sınıf (Hansen'in arşivindeki "
    "AL1999.dta). Maimonides kuralı bir sınıfta en çok 40 öğrenciye izin verir; okulun 5. sınıf kaydı 41'e ulaşınca "
    "ikinci şube açılır. Eşikte sınıfın ortalama okuma puanındaki sıçrama yerel doğrusal RDD ile, okul düzeyinde "
    "kümelenmiş standart hatayla tahmin edilir. Adımlar notlardaki gibidir: tasarım satırları, bant genişliği "
    "duyarlılığı, bant genişliği ölçeği, RDD grafiği ve kontrol listesi."
)


# --- Kendi verin --------------------------------------------------------------------------------------------

MIN_VALUES = 20
"""Eşik değişkeninde en az farklı değer."""
MIN_SIDE = 10
"""En dar pencerede eşiğin her iki yanında en az gözlem."""
MIN_SIDE_VALUES = 3
"""En dar pencerede eşiğin her iki yanında en az farklı eşik değişkeni değeri (yerel doğrusal tahmin için)."""
MIN_CLUSTERS = 10
"""Küme seçilirse en dar pencerede en az küme."""
MAX_DECIMALS = 6
SHARE = 0.4
"""Varsayılan ana bant: pencere ±h√6 eşiğe en yakın gözlemlerin bu payını içerir."""
CURVE_POINTS = 120


def _significant(value: float, digits: int = 2) -> float:
    """İki anlamlı basamağa yuvarlama (0,0347 → 0,035; 1.234 → 1.200)."""

    if value == 0 or not math.isfinite(value):
        return float(value)
    return float(round(value, digits - 1 - math.floor(math.log10(abs(value)))))


def _spread_digits(values: pd.Series) -> int:
    """Eşik gibi bir değerin ondalığı: değişkenin standart sapmasına göre dört anlamlı basamak."""

    spread = float(values.std())
    if not math.isfinite(spread) or spread <= 0:
        return 2
    return int(max(0, 3 - math.floor(math.log10(spread))))


def default_cutoff(values: pd.Series) -> float | None:
    """Eşik alanının ilk değeri: değişkenin medyanı (öğrenci kendi eşiğini yazmalıdır)."""

    data = values.dropna().astype(float)
    if data.empty:
        return None
    return float(round(float(data.median()), _spread_digits(data)))


def default_bandwidth(values: pd.Series, cutoff: float) -> float:
    """Önceden tanımlı kural: üçgen pencere ±h√6, eşiğe en yakın gözlemlerin %40'ını içerir. h, 4·10ᵏ'nın katına
    yuvarlanır (ör. 3,3 → 3,2): tablonun 0,5–1,5 katları aynı ondalıkla yazılır (1,6; 2,4; 3,2; 4; 4,8)."""

    distance = np.abs(values.dropna().astype(float).to_numpy() - cutoff)
    raw = float(np.quantile(distance, SHARE)) / math.sqrt(6.0)
    if raw <= 0 or not math.isfinite(raw):
        return 0.0
    unit = 4 * 10.0 ** (math.floor(math.log10(raw)) - 1)
    return float(round(max(unit, round(raw / unit) * unit), 12))


def _places(value: float) -> int:
    """Sayının yazımındaki ondalık basamak sayısı."""

    text = np.format_float_positional(float(value), precision=10, trim="-")
    return len(text.split(".")[1]) if "." in text else 0


def bandwidth_grid(h: float) -> tuple[float, ...]:
    """Tablo: ana bandın 0,5; 0,75; 1; 1,25 ve 1,5 katı (on ondalıkta yuvarlanmış)."""

    return tuple(float(round(h * factor, 10)) for factor in FACTORS)


def _decimals(values: pd.Series) -> int:
    """Sıçrama ve SH'nin ondalığı: sonucun standart sapmasına göre üç anlamlı basamak (en az 2)."""

    spread = float(values.astype(float).std())
    if not math.isfinite(spread) or spread <= 0:
        return 2
    return int(max(2, 2 - math.floor(math.log10(spread))))


def _numbers(case: Case) -> tuple[float, float | None]:
    numbers = case.extra.get("sayilar", {})
    return float(numbers["esik"]), numbers.get("h")


def own_plan(case: Case) -> Plan:
    y, x = case.roles[SONUC], case.roles[ESIK]
    cluster = case.roles.get(KUME)
    data = case.data
    cutoff, chosen = _numbers(case)
    h = float(chosen) if chosen else default_bandwidth(data[x], cutoff)
    bandwidths = bandwidth_grid(h)
    values = data[x].astype(float)
    reach = 2 * h * math.sqrt(6.0)
    window = int(max(2, 2 - math.floor(math.log10(h))))  # pencerelerde üç anlamlı basamak, en az iki ondalık
    labels = {column: case.name(column) for column in (y, x) + ((cluster,) if cluster else ())}
    labels["kume_sira"] = "Küme sıra numarası"
    return Plan(
        frame=case.frame, x=x, y=y, cutoff=cutoff, cluster=cluster, prepare=case.load, bandwidths=bandwidths,
        labels=labels, x_label=case.name(x), y_label=case.name(y),
        titles={"tablo": "Kendi veriniz: bant genişliği duyarlılığı",
                "grafik": "Kendi veriniz: eşiğin iki yanında yerel doğrusal tahmin"},
        x_range=(max(float(values.min()), cutoff - reach), min(float(values.max()), cutoff + reach)),
        right=not case.options.get(TARAF, False), decimals=_decimals(data[y]),
        h_decimals=max(_places(bw) for bw in bandwidths), window_decimals=(window, window),
    )


def _own_texts(case: Case, plan: Plan) -> dict:
    y, x = case.md(SONUC), case.md(ESIK)
    cluster = case.md(KUME) if plan.cluster else None
    c, h = _value(plan.cutoff), _value(plan.h_main)
    d = plan.decimals
    _, chosen = _numbers(case)
    side = "sağında ($X\\ge c$)" if plan.right else "solunda ($X<c$)"
    sign = 1.0 if plan.right else -1.0
    effect = ("Tedavi eşiğin sağında olduğu için tedavinin eşikteki etkisi $\\widehat\\tau$'dur." if plan.right else
              "Tedavi eşiğin solunda olduğu için tedavinin eşikteki etkisi $-\\widehat\\tau$'dur: tablo sağ limitten sol "
              "limiti çıkarır.")
    rule = (f"Ana bant genişliği sizin yazdığınız h = {h}'{ek(h, 'dir')}." if chosen else
            f"Ana bant genişliği önceden tanımlı bir kuralla h = {h} alındı: üçgen pencere $\\pm h\\sqrt6$ eşiğe en "
            "yakın gözlemlerin yaklaşık %40'ını içerir. Bu veri-temelli optimal bir bant seçimi değildir; kendi bandınızı "
            "yazabilirsiniz.")
    clustering = (f" Gözlemler “{cluster}” kümeleri içinde bağımlı olabileceği için standart hatalar küme düzeyinde "
                  "kümelenmiştir." if plan.cluster else " Küme seçilmedi; standart hatalar HC1'dir.")

    def step1(state) -> str:
        frame = state.frames[plan.frame]
        n, right = len(frame), int(frame["esik_sagi"].sum())
        groups = f"; {sayim(frame[plan.cluster].nunique())} küme" if plan.cluster else ""
        side_text = ("Tedavinin eşiğin sağındaki birimlere uygulandığı kabul edildi; kural tersse (ör. eşiğin altındakilere "
                     "yardım) “Tedavi eşiğin solunda” seçeneğini açın." if plan.right else
                     "Tedavinin eşiğin solundaki birimlere uygulandığı kabul edildi (“Tedavi eşiğin solunda” seçeneği).")
        return (f"Analiz örneklemi {sayim(n)} gözlem{groups}. {sayim(right)} gözlem eşiğin sağında (X ≥ {c}), "
                f"{sayim(n - right)} gözlem solunda. {side_text} Hangi tarafın tedavi aldığı veriden değil kurumsal "
                "kuraldan bilinir.")

    def step2(state) -> str:
        table = state.tables["bant"]
        effects = sign * table["tahmin"]
        low, high = sign * table["alt"], sign * table["ust"]
        lows, highs = np.minimum(low, high), np.maximum(low, high)
        covers = [h_value for h_value, a, b in zip(table.index, lows, highs) if a <= 0 <= b]
        same_sign = bool((effects > 0).all() or (effects < 0).all())
        text = (f"Bant genişliği büyüdükçe (h = {_value(table.index[0])} → {_value(table.index[-1])}) etkin örneklem "
                f"{sayim(table['n'].iloc[0])} → {sayim(table['n'].iloc[-1])}, standart hata "
                f"{_sayi(float(table['sh'].iloc[0]), d)} → {_sayi(float(table['sh'].iloc[-1]), d)} olur. Tedavinin "
                f"tahmini etkisi {_sayi(float(effects.min()), d)} ile {_sayi(float(effects.max()), d)} arasındadır")
        text += "; işaret bütün bantlarda aynıdır." if same_sign else "; işaret bantlar arasında değişir."
        if covers:
            text += " Güven aralığı şu bantlarda sıfırı içerir: h = " + "; ".join(_value(v) for v in covers) + "."
        else:
            text += " Güven aralıklarının hiçbiri sıfırı içermez."
        return text

    def step3(state) -> str:
        s, models = state.scalars, state.models
        rect, narrow = models["rdd_dikdortgen"], models["rdd_pencere"]
        jump = ("Sıçrama $\\widehat\\tau$ (sağ limit − sol limit" + ("" if plan.right else "; tedavinin etkisi $-\\widehat\\tau$")
                + ")")
        return (f"h = {h} iken eşiğin solunda tahmin {_sayi(float(s['sol']), d)}, sağında {_sayi(float(s['sag']), d)}. "
                f"{jump}: aynı yerellikte dikdörtgen çekirdek "
                f"(±{_value(round(plan.h_main * math.sqrt(3), plan.window_decimals[1]))}; "
                f"{sayim(rect.nobs)} gözlem) {_sayi(float(rect.params['D']), d)} ({_sayi(float(rect.bse['D']), d)}); "
                f"pencere ±{h}'{ek(h, 'e')} daraltılınca ({sayim(narrow.nobs)} gözlem) {_sayi(float(narrow.params['D']), d)} "
                f"({_sayi(float(narrow.bse['D']), d)}).")

    def step4(state) -> str:
        s = state.scalars
        higher = "daha yüksek" if s["sag"] > s["sol"] else "daha düşük"
        return (f"Eşiğin sağında tahmini koşullu ortalama {higher}; eğrilerin eşikteki değerleri Adım 3'teki "
                f"tahminlerdir (sol {_sayi(float(s['sol']), d)}, sağ {_sayi(float(s['sag']), d)}). Grafik eşiğin iki "
                "yanında iki pencere genişliğine kadar uzanır ("
                f"±{_value(round(2 * plan.h_main * math.sqrt(6), plan.window_decimals[0]))}; verinin aralığıyla "
                "sınırlı).")

    return {
        1: (
            DESIGN_ROWS + f"Eşik değişkeni $X$ “{x}”, eşik $c={c.replace(',', '{,}')}$, gösterge "
            f"$D=\\mathbf 1\\{{X\\ge c\\}}$ (eşiğin sağı). Sonuç “{y}”. Tedavi eşiğin {side} kalan birimlere "
            "uygulanır. " + effect + clustering + "\n\nBu laboratuvar keskin tasarım varsayar. Tedavi eşikte yalnız "
            "olasılıksal olarak değişiyorsa (bulanık RDD, §9.12) tahmin edilen sıçrama eşiği geçmenin etkisidir "
            "(indirgenmiş biçim); tedavinin etkisi için ilk aşama sıçraması da gerekir.",
            "\"Eşiğin sağındaki katsayı pozitif/negatiftir\" demek tek başına yeterli değildir: sağ tarafın hangi tedavi "
            "durumunu temsil ettiği kurumsal kuraldan okunur.",
        ),
        (1, "not"): step1,
        2: (_model_text(plan) + "\n\nTablodaki bantlar ana bandın 0,5; 0,75; 1; 1,25 ve 1,5 katıdır. " + rule,
            "Tablo \"en büyük etkiyi veren\" ya da \"en küçük standart hatayı veren\" bandı seçmek için okunmaz: her satır "
            "farklı bir yerellik–varyans dengesi kurar. Ana bant önceden tanımlı bir kuralla seçilir, tablo duyarlılık "
            "analizi olarak sunulur."),
        (2, "not"): step2,
        3: (_scale_text(plan, f"eşikte tahmini “{y}” değerleri"),
            "Replikasyonda sayılar tutmadığında önce tanım farkları aranır: analiz örneklemi, eşik değişkeninin "
            "merkezlenmesi, eşik, çekirdek, bant genişliğinin ölçeği, yerel polinom derecesi, kovaryatlar ve standart "
            "hata türü."),
        (3, "not"): step3,
        4: (
            "Ana RDD grafiğinde eşiğin iki yanında ayrı yerel doğrusal eğriler ve güven bantları gösterilir. Her $x_0$ "
            f"noktasında yalnız o taraftaki gözlemlerle üçgen çekirdekli ($h={h.replace(',', '{,}')}$, pencere "
            f"$\\pm h\\sqrt6$) ağırlıklı EKK'nin sabit terimi hesaplanır; bant noktasal %95 güven aralığıdır"
            + (" (küme SH)." if plan.cluster else " (HC1).") + " Görsel değerlendirmede üç şey birlikte aranır:\n\n"
            "1. eşikte koşullu ortalamada sıçrama,\n"
            "2. eşik çevresinde yeterli veri desteği,\n"
            "3. güven bantlarının sıçrama büyüklüğüne kıyasla genişliği.",
            "Belirgin görünen bir sıçrama, bant genişliği ve standart hata dikkate alınmadığında aşırı ikna edici "
            "görünebilir; tersine, nokta bulutunun gürültülü olması geçerli yerel tahminin bilgi taşımadığı anlamına "
            "gelmez.",
        ),
        (4, "not"): step4,
        5: ("", "Bu dosyayla cevaplanamayan sorular da raporun parçasıdır: eşik değişkeninin yoğunluğu (manipülasyon), "
                "önceden belirlenmiş kovaryatlarda ve plasebo sonuçlarda sıçrama, tedavinin eşikte gerçekten değişip "
                "değişmediği. Tahmin eşik çevresindeki birimler için yerel bir etkidir."),
        "kod2": "Her bant genişliğinde aynı ağırlıklı regresyon: Python'da `statsmodels` `WLS`, R'de `lm(..., weights = w)` "
                + ("ve `sandwich::vcovCL(type = \"HC1\")`; küme düzeltmesi iki dilde aynıdır." if plan.cluster else
                   "ve `sandwich::vcovHC(type = \"HC1\")`; iki dil aynı sayıları verir."),
        "kod4": "Eğriler her tarafta 120 noktada hesaplanır; her nokta ayrı bir ağırlıklı EKK'dir. Python'da NumPy ile "
                "vektörel, R'de `sapply` ile.",
    }


def build(case: Case) -> LabSpec:
    """Kendi verinle Konu 9 uygulaması; kontrollerin beklenen değerleri uygulamanın hesabıdır."""

    plan = own_plan(case)
    spec = _spec(plan, _own_texts(case, plan), source="kendi",
                 title=f"{case.label(SONUC)}: eşikte sıçrama", dataset="")
    return with_app_values(spec)


def _side_conditioning(x: np.ndarray, points: np.ndarray, width: float) -> np.ndarray:
    """Üçgen çekirdekli tek taraflı yerel doğrusal tahminin noktalardaki ρ = (S₀S₂ − S₁²)/(S₀S₂) değeri; noktanın
    penceresinde üçten az gözlem varsa 0."""

    d = x[None, :] - points[:, None]
    w = np.maximum(1.0 - np.abs(d) / width, 0.0)
    s0, s1, s2 = w.sum(axis=1), (w * d).sum(axis=1), (w * d * d).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = (s0 * s2 - s1**2) / (s0 * s2)
    rho = np.where(np.isfinite(rho), rho, 0.0)
    return np.where((w > 0).sum(axis=1) > 2, rho, 0.0)


def validate(case: Case) -> None:
    y, x = case.roles[SONUC], case.roles[ESIK]
    cluster = case.roles.get(KUME)
    if y == x or cluster in (y, x):
        raise K.UploadError("Sonuç, eşik değişkeni ve küme için farklı sütunlar seçin.")
    data = case.data
    values = data[x].astype(float)
    cutoff, chosen = _numbers(case)
    if chosen is not None and chosen <= 0:
        raise K.UploadError("Bant genişliği pozitif bir sayı olmalı.")
    if values.nunique() < MIN_VALUES:
        raise K.UploadError(f"“{case.name(x)}” sütununda en az {MIN_VALUES} farklı değer olmalı (eşiğin iki yanında "
                            "yerel doğrusal tahmin için).")
    if not (values.min() < cutoff <= values.max()):
        raise K.UploadError(f"Eşik ({_value(cutoff)}) “{case.name(x)}” değişkeninin gözlenen aralığının "
                            f"({_value(values.min())}–{_value(values.max())}) içinde olmalı.")
    if data[y].nunique() < 2:
        raise K.UploadError(f"“{case.name(y)}” sütununda en az iki farklı değer olmalı.")
    if _decimals(data[y]) > MAX_DECIMALS:
        raise K.UploadError(f"“{case.name(y)}” sütununun ölçeği çok küçük: tahminler {MAX_DECIMALS} ondalıkta bile "
                            "ayırt edilemiyor. Dosyada birimini değiştirin (ör. ton yerine kilogram).")
    h = float(chosen) if chosen else default_bandwidth(values, cutoff)
    if h <= 0:
        raise K.UploadError("Eşik çevresindeki gözlemlerin çoğu eşikle aynı değerde; bant genişliği hesaplanamıyor. "
                            "Bir bant genişliği yazın.")
    narrow = min(bandwidth_grid(h)) * math.sqrt(6.0)
    r = values.to_numpy() - cutoff
    inside = np.abs(r) < narrow
    for name, mask in (("sol", (r < 0) & inside), ("sağ", (r >= 0) & inside)):
        count, distinct = int(mask.sum()), int(np.unique(r[mask]).size)
        if count < MIN_SIDE or distinct < MIN_SIDE_VALUES:
            raise K.UploadError(
                f"En dar pencerede (h = {_value(min(bandwidth_grid(h)))}, ±{_value(round(narrow, 4))}) eşiğin {name} "
                f"tarafında {count} gözlem ve {distinct} farklı değer var; her iki yanda en az {MIN_SIDE} gözlem ve "
                f"{MIN_SIDE_VALUES} farklı değer gerekir. Daha büyük bir bant genişliği yazın ya da eşiği kontrol edin.")
    if cluster:
        groups = data.loc[inside, cluster].nunique()
        if groups < MIN_CLUSTERS:
            raise K.UploadError(f"En dar pencerede “{case.name(cluster)}” sütununda {groups} küme var; kümelenmiş standart "
                                f"hata için en az {MIN_CLUSTERS} küme gerekir. Daha büyük bir bant genişliği yazın ya da "
                                "küme seçmeyin.")
        if data[cluster].nunique() == len(data):
            raise K.UploadError(f"“{case.name(cluster)}” sütununda her gözlem ayrı bir küme: küme standart hatası HC1 ile "
                                "aynı bilgiyi verir. Küme seçmeyin ya da doğru küme sütununu seçin.")
    window = np.abs(r) <= h * math.sqrt(6.0)
    side = (r >= 0).astype(float)
    frame = pd.DataFrame({"y": data[y].astype(float).to_numpy(), "d": side, "r": r, "dr": side * r})[window]
    if exact_fit(frame, "y", ("d", "r", "dr")):
        raise K.UploadError(f"Ana pencerede “{case.name(y)}”, eşik değişkeninin iki yanında (neredeyse) tam doğrusal "
                            "(R² ≈ 1): standart hatalar yuvarlama hatasına duyarlıdır ve yazılımlar arasında aynı "
                            "çıkmaz.")
    plan = own_plan(case)
    low, high = plan.x_range
    width = h * math.sqrt(6.0)
    points = (np.linspace(low, cutoff, CURVE_POINTS), np.linspace(cutoff, high, CURVE_POINTS))
    for name, mask, grid in (("sol", r < 0, points[0]), ("sağ", r >= 0, points[1])):
        if np.any(_side_conditioning(values.to_numpy()[mask], grid, width) < RD_RELIABLE):
            raise K.UploadError(f"Eşiğin {name} tarafında “{case.name(x)}” değerleri arasında büyük boşluklar var: RDD "
                                "grafiğinin eğrisi bazı noktalarda sayısal olarak hesaplanamıyor (pencerede ağırlığı "
                                "anlamlı tek bir değer kalıyor). Daha büyük bir bant genişliği yazın.")


RD_RELIABLE = 1e-6
"""Grafik eğrisinin her noktasında en küçük göreli belirleyici (``core.labs.smoothing.RELIABLE`` ile aynı ölçüt)."""


def suggest(table: K.UploadedTable) -> dict[str, str]:
    """Küme rolü için adında okul, küme, firma, sınıf, il, köy ya da kurum sözcüğü geçen ve 10 ile gözlem sayısının
    yarısı arasında farklı değer alan sütun önerilir."""

    words = {"okul", "küme", "kume", "firma", "sınıf", "sinif", "school", "cluster", "köy", "koy", "kurum", "il"}
    rows = len(table.frame)
    for column in table.columns:
        if words & set(K.name_words(column)):
            count = table.frame[column].map(K.clean_text).dropna().nunique()
            if MIN_CLUSTERS <= count <= rows / 2:
                return {KUME: column}
    return {}


def sample() -> pd.DataFrame:
    """Örnek dosya: kurgusal bursluluk sınavı verisi (``core.labs.ornek_veri``); eşik 70."""

    return burs_verisi()


ROLES = (
    Role(SONUC, "Sonuç değişkeni", "sayisal", True, (1, 2, 3, 4),
         "Eşikte sıçraması tahmin edilen sayısal değişken (notlarda Head Start ile ilişkili ölüm oranı)."),
    Role(ESIK, "Eşik değişkeni (running variable)", "sayisal", True, (1, 2, 3, 4),
         "Tedaviyi eşik kuralıyla belirleyen değişken (notlarda 1960 yoksulluk oranı); en az 20 farklı değer."),
    Role(KUME, "Küme (ör. okul)", "serbest", False, (1, 2, 3, 4),
         "Gözlemlerin bağımlı olduğu birim. Seçilirse standart hatalar kümelenir; seçilmezse HC1. Boş hücre olamaz.",
         complete=True),
)

NUMBERS = (
    NumberInput("esik", "Eşik değeri c",
                "Tedaviyi belirleyen eşik (notlarda 59,1984); D = 1{X ≥ c}. İlk değer değişkenin medyanıdır, kendi "
                "eşiğinizi yazın. Ondalık için virgül ya da nokta kullanılabilir; binlik ayırıcı kullanmayın (“5.000” iki türlü "
                "okunabildiği için kabul edilmez).", role=ESIK, default=default_cutoff),
    NumberInput("h", "Ana bant genişliği h (isteğe bağlı)",
                "Hansen ölçeği: üçgen çekirdekte pencere ±h√6. Boş bırakılırsa pencere eşiğe en yakın gözlemlerin "
                "yaklaşık %40'ını içerecek biçimde seçilir. Tablo ana bandın 0,5–1,5 katıdır. Eşik değişkeni "
                "değişince alan boşalır.", role=ESIK,
                required=False, placeholder="Boş: veriden (%40 kuralı)"),
)

OPTIONS = (
    Option(TARAF, "Tedavi eşiğin solunda (X < c)",
           "Açıksa tedavi eşiğin altındaki birimlere uygulanır (ör. gelir eşiğinin altındakilere yardım); tablodaki "
           "sıçrama yine sağ eksi soldur, tedavinin etkisi −τ̂ olarak yorumlanır.", default=False),
)

CUSTOM = CustomLab(
    roles=ROLES,
    build=build,
    sample=sample,
    intro=(
        "Bir Excel (.xlsx) ya da CSV dosyası yükleyin. Sonuç ve eşik değişkeni seçilir, eşik değeri yazılır; gözlemler "
        "kümeler (ör. okul) içinde bağımlıysa küme sütunu da seçilir. Seçilen sütunlarda boş hücresi olan satırlar "
        "analizden çıkarılır. Örnek dosyada eşik 70'tir (bursluluk sınavı)."
    ),
    options=OPTIONS,
    min_rows=50,
    validate=validate,
    suggest=suggest,
    numbers=NUMBERS,
)

VARIANTS = TopicVariants(alternative=alternative, story=STORY, custom=CUSTOM)
