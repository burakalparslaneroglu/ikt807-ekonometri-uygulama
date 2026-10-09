"""Konu 2 genel uygulaması: regresyon tablosunu güvenilir okumak.

Notlardaki §2.13'ün yedi adımı aynı numaralarla ve aynı işlemlerle, verisi değiştirilebilir biçimde yazılır
(``build``): üç iç içe model, klasik ve HC1 standart hata, Breusch–Pagan testi, etkileşim ve doğrusal birleşim, delta
yöntemi. Alternatif örnek Card (1995) verisidir (Hansen'in arşivindeki ``Card1995.dta``); "Kendi verini yükle"
seçeneğinde aynı adımlar öğrencinin dosyasıyla kurulur. Notlardaki laboratuvar (``core.labs.konu02``) değişmez.

Rollerin notlardaki karşılıkları: sonuç log saatlik ücret, temel açıklayıcı değişken eğitim yılı, grup kadın göstergesi
(M3'te eğitimle etkileşimi), kontroller deneyim profili ve demografik göstergeler.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd

from core.labs import expr as E
from core.labs import kendi_veri as K
from core.labs import ornek_konu01 as K1
from core.labs.ornek import (
    EXACT_FIT_NOTE,
    SCALE_MESSAGE,
    Case,
    CustomLab,
    Option,
    Role,
    TopicVariants,
    exact_fit,
    full_rank,
    katsayi,
    liste,
    md,
    sayi,
    stable_checks,
    stable_design,
    with_app_values,
    with_expected,
    yuzde,
)
from core.labs.ornek_veri import ucret_verisi
from core.labs.spec import (
    OLS,
    BreuschPagan,
    Check,
    CoefTarget,
    DeltaMethod,
    Derive,
    Indicator,
    LabSpec,
    LabStep,
    LinearCombination,
    ModelTarget,
    NoteRef,
    Operation,
    ReproClass,
    Scalar,
    ScalarTarget,
    ShowModel,
    StandardErrorTable,
)

TOPIC = "konu02"
SECTION = "2.13"
SONUC, ACIKLAYICI, GRUP = "sonuc", "aciklayici", "grup"
LOG = "log"
LOG_Y, G01, XG = "log_sonuc", "grup01", "etkilesim"
"""Kendi verinde türetilen sütunlar (``kendi_veri.RESERVED_CODES``: öğrencinin sütunlarına verilmez)."""
INTERCEPT = E.INTERCEPT
BP_NAME = "bp"


@dataclass(frozen=True)
class Plan:
    """Rollerin koddaki sütunları ve kontrol etiketlerindeki adlar.

    ``controls``: M2'ye eklenen değişkenler (grup göstergesi dahil, notlardaki gibi); ``interaction``: M3'teki
    açıklayıcı değişken × grup terimi. ``slope``: (skaler adı, açıklama) grup 1'in eğimi için doğrusal birleşim.
    """

    frame: str
    outcome: str
    log: bool
    x: str
    group: str
    interaction: str
    controls: tuple[str, ...]
    prepare: tuple[Operation, ...]
    x_word: str
    reference_slope: str
    group_slope: str
    slope: tuple[str, str]
    labels: dict
    code_note: str


def _coef(model: str, term: str, label: str, quantity: str = "coef") -> Check:
    return Check(label, CoefTarget(model, term, quantity), 0.0)


def _steps(plan: Plan, texts: dict, exact: bool) -> tuple[LabStep, ...]:
    m1 = (plan.x,)
    m2 = (plan.x, *plan.controls)
    m3 = (plan.x, plan.interaction, *plan.controls)
    word = plan.x_word
    step3_ops: list[Operation] = [StandardErrorTable(("m1", "m2", "m3"), plan.x, "sh_tablosu")]
    step3_checks = []
    for model in ("m1", "m2", "m3"):
        name = model.upper()
        step3_checks += [
            _coef(model, plan.x, f"{name} {word} katsayısı"),
            _coef(model, plan.x, f"{name} klasik SH", "se"),
            _coef(model, plan.x, f"{name} HC1 SH", "se_hc1"),
            Check(f"{name} R²", ModelTarget(model, "r2"), 0.0),
        ]
    if plan.log:
        step3_ops.append(Scalar("yuzde_etki", E.mul(100, E.sub(E.exp(E.coef("m2", plan.x)), 1)),
                                f"M2 {word} katsayısının kesin yüzde karşılığı" if not texts.get("own") else
                                "M2 temel katsayının kesin yüzde karşılığı"))
        step3_checks.append(Check("M2 kesin yüzde etki", ScalarTarget("yuzde_etki"), 0.0, decimals=2))
    slope_name, slope_comment = plan.slope
    step6_ops: list[Operation] = [OLS("m2_hc1", plan.frame, plan.outcome, m2, vcov="HC1"), ShowModel("m2_hc1")]
    if plan.log:
        step6_ops.append(DeltaMethod("yuzde_etki_dm", "m2_hc1",
                                     E.mul(100, E.sub(E.exp(E.coef("m2_hc1", plan.x)), 1)), "M2 kesin yüzde etki"))
    return (
        LabStep(number=1, title="Araştırma sorusunu ve hedef katsayıyı açıkça tanımlamak", note=NoteRef(SECTION, 1),
                explanation=texts[1][0], takeaway=texts[1][1]),
        LabStep(
            number=2,
            title="Birden çok spesifikasyon kurmak",
            note=NoteRef(SECTION, 2),
            explanation=texts[2][0],
            operations=plan.prepare + (
                Derive(plan.frame, plan.interaction, E.mul(E.var(plan.x), E.var(plan.group)), texts["interaction"]),
                OLS("m1", plan.frame, plan.outcome, m1),
                OLS("m2", plan.frame, plan.outcome, m2),
                OLS("m3", plan.frame, plan.outcome, m3),
            ),
            takeaway=texts[2][1],
            code_note=plan.code_note,
        ),
        LabStep(
            number=3,
            title="Katsayı ile standart hatayı birbirinden ayırmak",
            note=NoteRef(SECTION, 3),
            explanation=texts[3][0],
            operations=tuple(step3_ops),
            checks=stable_checks(step3_checks, exact),
            takeaway=texts[3][1],
            note_for=texts.get((3, "not")),
        ),
        LabStep(
            number=4,
            title="Heteroskedastisite tanısını makale diliyle okumak",
            note=NoteRef(SECTION, 4),
            explanation=texts[4][0],
            operations=(BreuschPagan("m2", BP_NAME),),
            checks=stable_checks((Check("Breusch–Pagan LM", ScalarTarget(f"{BP_NAME}_lm"), 0.0, decimals=2),),
                                 exact, (f"{BP_NAME}_lm",)),
            reproducibility=ReproClass.CONVENTION,
            takeaway=texts[4][1],
            code_note=(
                "Stata'nın `estat hettest` komutu varsayılan olarak başka bir sürümü (yalnız uyum değerleri, "
                "normallik varsayımı) hesaplar. Python ve R ile aynı sayı için `rhs iid` seçenekleri gerekir."
            ),
            note_for=texts.get((4, "not")),
        ),
        LabStep(
            number=5,
            title="Etkileşim katsayısını doğru yorumlamak",
            note=NoteRef(SECTION, 5),
            explanation=texts[5][0],
            operations=(
                OLS("m3_hc1", plan.frame, plan.outcome, m3, vcov="HC1"),
                LinearCombination(slope_name, "m3_hc1", ((plan.x, 1.0), (plan.interaction, 1.0)), slope_comment),
            ),
            checks=(
                _coef("m3", plan.x, plan.reference_slope),
                _coef("m3", plan.interaction, "Etkileşim katsayısı"),
                Check(plan.group_slope, ScalarTarget(slope_name), 0.0),
            ),
            takeaway=texts[5][1],
            note_for=texts.get((5, "not")),
        ),
        LabStep(
            number=6,
            title="Yazılımda güvenilir standart hatayı açıkça istemek",
            note=NoteRef(SECTION, 6),
            explanation=texts[6][0],
            operations=tuple(step6_ops),
            checks=stable_checks((
                _coef("m2_hc1", plan.x, f"HC1 çıktısında {word} katsayısı"),
                _coef("m2_hc1", plan.x, "HC1 çıktısında standart hata", "se"),
            ), exact),
            takeaway=texts[6][1],
            note_for=texts.get((6, "not")),
        ),
        LabStep(number=7, title="Bir makale tablosunu hangi sırayla okumalıyız?", note=NoteRef(SECTION, 7),
                explanation=texts[7][0], takeaway=texts[7][1], note_for=texts.get((7, "not"))),
    )


def _spec(plan: Plan, texts: dict, *, source: str, title: str, dataset: str, exact: bool = False) -> LabSpec:
    return LabSpec(topic_key=TOPIC, title=title, dataset=dataset, note_section=SECTION,
                   steps=_steps(plan, texts, exact), labels=tuple(plan.labels.items()), source=source)


# --- Ortak metinler -------------------------------------------------------------------------------------------

SPECIFICATIONS = (
    "Spesifikasyonların amacı \"anlamlı katsayı bulmak\" değil, araştırma sorusuna uygun koşullu ilişkiyi adım adım "
    "görünür kılmaktır."
)
HC1_SAME_COEF = (
    "HC1 kullanıldığında **katsayı değişmez**; yalnız tahminin belirsizliği yeniden hesaplanır. \"Robust regresyon "
    "yaptım ve katsayı düzeldi\" ifadesi genel olarak yanlıştır."
)
HC1_EXPLAIN = (
    "HC1 seçeneği katsayı denklemine yeni bir değişken eklemez; aynı OLS katsayıları üzerinde heteroskedastisiteye "
    "dayanıklı standart hata hesabını etkinleştirir."
)
DELTA_EXPLAIN = (
    " Aynı çıktıdan yüzde etkinin standart hatası delta yöntemiyle hesaplanır: $g(\\beta)=100(e^{\\beta}-1)$ için "
    "$\\operatorname{SH}(g)\\approx g'(\\hat\\beta)\\,\\operatorname{SH}(\\hat\\beta)$."
)
ROBUST_KIND = (
    "Bir makalede \"robust standart hatalar\" ifadesini gördüğünüzde hangi robust türünün kullanıldığını ve veri "
    "kümeli ise kümelenme biriminin ne olduğunu ayrıca kontrol edin."
)
READING_ORDER = (
    "Güvenli okuma sırası: (1) bağımlı değişken düzey mi, log mu; (2) temel katsayının birimi; (3) standart hata "
    "türü: klasik, HC, küme-robust, bootstrap; (4) güven aralığı ve p-değeri, yalnız yıldızlar değil; (5) sütunlar "
    "arası spesifikasyon farkı; (6) ekonomik büyüklük; (7) tanımlama dili: gözlemsel ilişki nedensel etki diye "
    "sunuluyor mu?"
)
INTERACTION_RULE = (
    "Etkileşimli modellerde ana etki, diğer etkileşim değişkeninin referans düzeyindeki eğimdir. İki katsayıyı ayrı "
    "ayrı yorumlamak yerine ilgili doğrusal birleşim hesaplanmalıdır."
)


# --- Alternatif örnek: Card (1995) ----------------------------------------------------------------------------

ALT_DATA = "card1995"
ALT_TITLE = "Card (1995) Verisiyle Regresyon Tablosunu Güvenilir Okumak"


def alternative_plan() -> Plan:
    base = K1.alternative_plan()
    labels = dict(base.labels)
    labels.update({
        "ed_black": "Eğitim × siyahi",
        "reg76r": "Güney (1976)",
        "smsa76r": "Metropol (1976)",
        "m1": "M1",
        "m2": "M2",
        "m3": "M3",
        "m2_hc1": "M2 (HC1)",
        "m3_hc1": "M3 (HC1)",
    })
    return Plan(
        frame=base.frame, outcome="lwage", log=True, x="ed76", group="black", interaction="ed_black",
        controls=("experience", "experience2_100", "black", "reg76r", "smsa76r"),
        prepare=base.prepare, x_word="eğitim",
        reference_slope="Siyahi olmayanların eğimi (eğitim katsayısı)", group_slope="Siyahilerin eğimi",
        slope=("siyahi_egimi", "Siyahiler için eğitim eğimi"), labels=labels,
        code_note=("Kontroller: potansiyel deneyim ve karesi (/100) ile siyahi, Güney (reg76r) ve metropol alan "
                   "(smsa76r) göstergeleri; bölge göstergeleri 1976 içindir."),
    )


ALT_TEXTS = {
    "interaction": "Eğitim × siyahi etkileşimi",
    1: (
        "Araştırma sorusu: *Eğitim yılı ile saatlik ücret arasındaki koşullu ilişki, deneyim, ırk ve yaşanılan bölge "
        "dikkate alındığında ne ölçüdedir?* Veri Card (1995)'in 1976 NLSYM örneklemidir (3.010 genç erkek). Bağımlı "
        "değişken $Y_i=\\log(wage_i)$, temel açıklayıcı değişken eğitim yılıdır ($ed76$).",
        "Eğitim katsayısını \"nedensel eğitim getirisi\" diye adlandırmıyoruz: yetenek ve aile geçmişi gibi "
        "gözlenmeyen etkenler böyle bir yorum için ek tanımlama varsayımları gerektirir. Hedef, doğrusal koşullu "
        "ilişkinin güvenilir biçimde tahmin ve rapor edilmesidir.",
    ),
    2: (
        "Üç iç içe model: **M1** yalnız eğitim; **M2** deneyim profili ile siyahi, Güney ve metropol alan (SMSA) "
        "göstergeleri eklenmiş; **M3** buna eğitim × siyahi etkileşimini ekler, yani eğitim eğiminin siyahi ve siyahi "
        "olmayan erkeklerde farklı olabilmesine izin verir.",
        SPECIFICATIONS,
    ),
    3: (
        "Tabloda önce katsayı okunur: M2'de eğitim katsayısı 0,0740. Kesin dönüşümle "
        "$100\\{\\exp(0{,}0740)-1\\}\\approx 7{,}68$: diğer değişkenler sabitken bir ek eğitim yılı yaklaşık yüzde 7,7 "
        "daha yüksek saatlik ücretle ilişkilidir. Sonra standart hata okunur.",
        HC1_SAME_COEF + " Bu veride klasik ve HC1 standart hataları da birbirine çok yakındır (M2'de 0,0035 ve 0,0036).",
    ),
    4: (
        "M2 artıklarına Breusch–Pagan testi uygulandığında LM istatistiği yaklaşık 6,51, p-değeri yaklaşık 0,37'dir "
        "(6 serbestlik derecesi): bu veride homoskedastisite varsayımı reddedilmez. Notlardaki CPS örneğinde test "
        "güçlü biçimde reddediyordu.",
        "Testin reddetmemesi homoskedastisitenin doğru olduğunu kanıtlamaz; yalnız bu veride aleyhine güçlü kanıt "
        "olmadığını söyler. HC1'i varsayılan olarak raporlamanın maliyeti küçüktür: hata varyansı sabitse klasik ve "
        "HC1 standart hataları büyük örneklemde birbirine yaklaşır. Heteroskedastisite OLS katsayısını otomatik olarak "
        "yanlı yapmaz; dayanıklı standart hata da eksik değişken veya içsellik sorununu çözmez.",
    ),
    5: (
        "M3'te siyahi olmayanlar referans gruptur: eğitim katsayısı 0,0701 siyahi olmayanların eğimidir; etkileşim "
        "0,0182 ise siyahilerin eğiminin bundan farkıdır. Siyahiler için eğim bir **doğrusal birleşimdir**: "
        "$0{,}0701+0{,}0182\\approx0{,}088$. Standart hatası $\\sqrt{\\boldsymbol a'\\widehat{\\boldsymbol V}_{\\hat\\beta}\\boldsymbol a}$ ile, $\\boldsymbol a=(1,1)'$ (iki katsayıya ait HC1 kovaryans alt matrisiyle) ve $\\widehat{\\boldsymbol V}_{\\hat\\beta}$ ilgili HC1 kovaryans alt "
        "matrisi olmak üzere hesaplanır.",
        INTERACTION_RULE + " Etkileşimli modelde siyahi göstergesinin katsayısı (−0,4146) eğitim sıfırken iki grubun "
        "farkıdır; örneklemde eğitimi sıfır olan kimse yoktur, bu yüzden tek başına yorumlanmaz.",
    ),
    6: (HC1_EXPLAIN + DELTA_EXPLAIN + " Bu veride M2'nin yüzde etkisi 7,68, delta yöntemiyle standart hatası 0,39'dur.",
        ROBUST_KIND),
    7: (
        READING_ORDER,
        "\"HC1 standart hataları kullanılan temel spesifikasyonda eğitim katsayısı 0,074 olarak tahmin edilmiştir. Bu "
        "değer, deneyim, ırk ve yaşanılan bölge sabit tutulduğunda bir ek eğitim yılının saatlik ücrette yaklaşık "
        "yüzde 7,7 daha yüksek bir değerle ilişkili olduğunu göstermektedir. Tahmin edilen ilişki nedensel eğitim "
        "getirisi olarak yorumlanmamalıdır.\"",
    ),
}

ALT_EXPECTED = {
    (3, "M1 eğitim katsayısı"): 0.0521,
    (3, "M1 klasik SH"): 0.0029,
    (3, "M1 HC1 SH"): 0.0029,
    (3, "M1 R²"): 0.0987,
    (3, "M2 eğitim katsayısı"): 0.0740,
    (3, "M2 klasik SH"): 0.0035,
    (3, "M2 HC1 SH"): 0.0036,
    (3, "M2 R²"): 0.2905,
    (3, "M3 eğitim katsayısı"): 0.0701,
    (3, "M3 klasik SH"): 0.0038,
    (3, "M3 HC1 SH"): 0.0039,
    (3, "M3 R²"): 0.2925,
    (3, "M2 kesin yüzde etki"): 7.68,
    (4, "Breusch–Pagan LM"): 6.51,
    (5, "Siyahi olmayanların eğimi (eğitim katsayısı)"): 0.0701,
    (5, "Etkileşim katsayısı"): 0.0182,
    (5, "Siyahilerin eğimi"): 0.0882,
    (6, "HC1 çıktısında eğitim katsayısı"): 0.0740,
    (6, "HC1 çıktısında standart hata"): 0.0036,
}
"""Kontrollerin Card1995 tam örneklemindeki (3.010 erkek) değerleri, gösterim basamağında: (adım, etiket) → değer.
Metinlerdeki sayılar bunlardır; testler bağımsız bir hesapla ve üç dilde doğrular."""


@cache
def alternative() -> LabSpec:
    """Alternatif örneğin tanımı; beklenen değerler Card1995'in tam örneklemindeki sayılardır (testle doğrulanır)."""

    spec = _spec(alternative_plan(), ALT_TEXTS, source="alternatif", title=ALT_TITLE, dataset=ALT_DATA)
    return with_expected(spec, ALT_EXPECTED)


STORY = (
    "Alternatif örnek Card (1995) verisidir: 1976'da 24–34 yaşındaki 3.010 genç erkek (NLSYM; Hansen'in arşivindeki "
    "Card1995.dta). Adımlar notlardaki gibidir: üç iç içe model, klasik ve HC1 standart hata, Breusch–Pagan testi, "
    "eğitim × siyahi etkileşimi ve delta yöntemi."
)


# --- Kendi verin --------------------------------------------------------------------------------------------

def _derived_data(case: Case) -> pd.DataFrame:
    data = case.data.copy()
    y, x, g = case.roles[SONUC], case.roles[ACIKLAYICI], case.roles[GRUP]
    if case.options.get(LOG, True):
        data[LOG_Y] = np.log(data[y].astype(float))
    data[G01] = (data[g] == case.levels[GRUP]).astype(float)
    data[XG] = data[x].astype(float) * data[G01]
    return data


def own_plan(case: Case) -> Plan:
    y, x, g = (case.roles[role] for role in (SONUC, ACIKLAYICI, GRUP))
    log = bool(case.options.get(LOG, True))
    pick = case.levels[GRUP]
    other = next(item for item in case.orders[g] if item != pick)
    name = case.name
    outcome = LOG_Y if log else y
    prepare: list[Operation] = list(case.load)
    if log:
        prepare.append(Derive(case.frame, LOG_Y, E.log(E.var(y)), "Sonucun doğal logaritması"))
    prepare.append(Indicator(case.frame, G01, g, pick, "Grup göstergesi: seçilen kategori 1, diğeri 0"))
    labels = {column: name(column) for column in (y, x, g, *case.extras)}
    labels.update({
        G01: f"{name(g)} = {pick}",
        XG: f"{name(x)} × {pick}",
        INTERCEPT: "Sabit",
        "m1": "M1",
        "m2": "M2",
        "m3": "M3",
        "m2_hc1": "M2 (HC1)",
        "m3_hc1": "M3 (HC1)",
    })
    if log:
        labels[LOG_Y] = f"log({name(y)})"
    return Plan(
        frame=case.frame, outcome=outcome, log=log, x=x, group=G01, interaction=XG,
        controls=(G01, *case.extras), prepare=tuple(prepare), x_word=name(x),
        reference_slope=f"“{other}” grubunun eğimi ({name(x)} katsayısı)", group_slope=f"“{pick}” grubunun eğimi",
        slope=("grup_egimi", "1 ile kodlanan grubun eğimi"), labels=labels,
        code_note="Grup göstergesi, seçtiğiniz kategoride 1, diğerinde 0'dır; etkileşim terimi açıklayıcı değişken × "
                  "gösterge çarpımıdır.",
    )


def _own_texts(case: Case, plan: Plan, exact: bool) -> dict:
    y, x, group = (md(case.label(role)) for role in (SONUC, ACIKLAYICI, GRUP))
    pick = md(case.levels[GRUP])
    other = md(next(item for item in case.orders[case.roles[GRUP]] if item != case.levels[GRUP]))
    outcome = f"log({y})" if plan.log else y
    extras = [f"“{md(case.name(column))}”" for column in case.extras]
    controls = f"“{pick}” grup göstergesi" + (f" ve ek kontroller ({liste(extras)})" if extras else "")
    exact_note = (" " + EXACT_FIT_NOTE) if exact else ""

    def step3(state) -> str:
        table = state.tables["sh_tablosu"]
        b2 = table.loc["m2", "katsayi"]
        text = (f"M2'de “{x}” katsayısı {katsayi(b2)}; klasik SH {katsayi(table.loc['m2', 'klasik_sh'])}, HC1 SH "
                f"{katsayi(table.loc['m2', 'hc1_sh'])}.")
        if plan.log:
            text += (f" Kesin dönüşümle bu katsayı, diğer değişkenler sabitken sonuçta "
                     f"{yuzde(state.scalars['yuzde_etki'], 2)} farka karşılık gelir.")
        return text + " " + HC1_SAME_COEF + exact_note

    def step4(state) -> str:
        if exact:
            return EXACT_FIT_NOTE
        lm, p, df = (state.scalars[f"{BP_NAME}_{key}"] for key in ("lm", "p", "df"))
        verdict = ("%5 düzeyinde homoskedastisite varsayımı reddedilir: hata varyansı açıklayıcı değişkenlerle "
                   "değişiyor görünür." if p < 0.05 else
                   "%5 düzeyinde homoskedastisite varsayımı reddedilmez; bu, varsayımın doğru olduğunu kanıtlamaz.")
        return (f"Breusch–Pagan LM = {sayi(lm, 2)}, serbestlik derecesi {int(df)}, p-değeri {sayi(p, 4)}: {verdict} "
                "Heteroskedastisite OLS katsayısını otomatik olarak yanlı yapmaz; dayanıklı standart hata da eksik "
                "değişken veya içsellik sorununu çözmez.")

    def step5(state) -> str:
        result = state.models["m3"]
        b, d = float(result.params[plan.x]), float(result.params[plan.interaction])
        slope_name = plan.slope[0]
        return (f"M3'te “{other}” referans gruptur: “{x}” katsayısı {katsayi(b)} bu grubun eğimidir; etkileşim "
                f"{katsayi(d)} ise “{pick}” grubunun eğiminin bundan farkıdır. “{pick}” grubunun eğimi "
                f"{katsayi(state.scalars[slope_name])} (HC1 SH {katsayi(state.scalars[slope_name + '_se'])}). "
                + INTERACTION_RULE)

    def step6(state) -> str:
        text = ROBUST_KIND
        if plan.log:
            text = (f"M2'nin yüzde etkisi {sayi(state.scalars['yuzde_etki_dm'], 2)}, delta yöntemiyle standart hatası "
                    f"{sayi(state.scalars['yuzde_etki_dm_se'], 3)}. " + text)
        return text + exact_note

    def step7(state) -> str:
        result = state.models["m2_hc1"]
        b = float(result.params[plan.x])
        text = (f"\"HC1 standart hataları kullanılan temel spesifikasyonda “{x}” katsayısı {katsayi(b)} olarak tahmin "
                "edilmiştir.")
        if plan.log:
            percent = state.scalars["yuzde_etki"]
            direction = "yüksek" if percent >= 0 else "düşük"
            text += (f" Bu değer, {controls} sabit tutulduğunda “{x}” bir birim daha yüksek olan gözlemlerde "
                     f"sonucun yaklaşık {yuzde(abs(percent), 1)} daha {direction} bir değerle ilişkili olduğunu "
                     "göstermektedir.")
        else:
            text += (f" Bu değer, {controls} sabit tutulduğunda “{x}” bir birim daha yüksek olan gözlemlerin "
                     f"sonucunun ortalama {katsayi(b)} birim farklı olduğunu göstermektedir.")
        return text + " Tahmin edilen ilişki nedensel etki olarak yorumlanmamalıdır.\""

    return {
        "own": True,
        "interaction": "Etkileşim: açıklayıcı değişken × grup göstergesi",
        1: (
            f"Araştırma sorusu: *“{x}” ile “{y}” arasındaki koşullu ilişki, kontroller dikkate alındığında ne "
            f"ölçüdedir?* Bağımlı değişken {outcome}, temel açıklayıcı değişken “{x}”.",
            "Katsayıyı \"nedensel etki\" diye adlandırmıyoruz: gözlenmeyen etkenler böyle bir yorum için ek tanımlama "
            "varsayımları gerektirir. Hedef, doğrusal koşullu ilişkinin güvenilir biçimde tahmin ve rapor edilmesidir.",
        ),
        2: (
            f"Üç iç içe model: **M1** yalnız “{x}”; **M2** {controls} eklenmiş; **M3** buna “{x}” × gösterge "
            f"etkileşimini ekler, yani “{x}” eğiminin “{pick}” ve “{other}” gruplarında farklı olabilmesine izin verir.",
            SPECIFICATIONS,
        ),
        3: ("Tabloda önce katsayı okunur, sonra standart hata. Aynı katsayı için klasik ve HC1 standart hatası yan "
            "yana gösterilir.", HC1_SAME_COEF),
        (3, "not"): step3,
        4: ("M2 artıklarına Breusch–Pagan testi (Koenker, n·R²) uygulanır: artık kareleri modelin bütün "
            "regresörleriyle açıklanabiliyor mu?", ""),
        (4, "not"): step4,
        5: (f"M3'te “{other}” referans gruptur. “{pick}” grubunun eğimi bir **doğrusal birleşimdir**: ana etki + "
            "etkileşim. Standart hatası $\\sqrt{\\boldsymbol a'\\widehat{\\boldsymbol V}_{\\hat\\beta}\\boldsymbol a}$ ile, $\\boldsymbol a=(1,1)'$ (iki katsayıya ait HC1 kovaryans alt matrisiyle) ve $\\widehat{\\boldsymbol V}_{\\hat\\beta}$ ilgili HC1 kovaryans alt matrisi olmak üzere "
            "hesaplanır.", INTERACTION_RULE),
        (5, "not"): step5,
        6: (HC1_EXPLAIN + (DELTA_EXPLAIN if plan.log else
                           " Sonuç logaritmik olmadığı için yüzde dönüşüm ve delta yöntemi gerekmez."), ROBUST_KIND),
        (6, "not"): step6,
        7: (READING_ORDER, ""),
        (7, "not"): step7,
    }


def build(case: Case) -> LabSpec:
    """Kendi verinle Konu 2 uygulaması; kontrollerin beklenen değerleri uygulamanın hesabıdır."""

    plan = own_plan(case)
    exact = exact_fit(_derived_data(case), plan.outcome, (plan.x, plan.interaction, *plan.controls))
    texts = _own_texts(case, plan, exact)
    spec = _spec(plan, texts, source="kendi", title=f"{case.label(SONUC)} ve {case.label(ACIKLAYICI)}",
                 dataset="", exact=exact)
    return with_app_values(spec)


def validate(case: Case) -> None:
    y, x, g = (case.roles[role] for role in (SONUC, ACIKLAYICI, GRUP))
    if y == x:
        raise K.UploadError("Sonuç ve açıklayıcı değişken için farklı sütunlar seçin.")
    if y in case.extras or x in case.extras:
        raise K.UploadError("Ek kontrollerde sonuç ya da açıklayıcı değişken olmamalı.")
    data = case.data
    for column in (y, x, *case.extras):
        if data[column].nunique() < 2:
            raise K.UploadError(f"“{case.name(column)}” sütununda en az iki farklı değer olmalı.")
    if case.options.get(LOG, True) and (data[y] <= 0).any():
        raise K.UploadError(f"“{case.name(y)}” sütununda sıfır ya da negatif değer var; logaritma alınamaz. "
                            "“Sonucun logaritmasını al” seçeneğini kapatın ya da başka bir sütun seçin.")
    derived = _derived_data(case)
    if derived[G01].value_counts().min() < 2:
        raise K.UploadError(f"“{case.name(g)}” sütununun iki kategorisinde de en az iki gözlem olmalı.")
    columns = (x, XG, G01, *case.extras)
    if len(derived) <= len(columns) + 2:
        raise K.UploadError(f"M3'te {len(columns) + 1} katsayı var; en az {len(columns) + 3} gözlem gerekir.")
    if not full_rank(derived, columns):
        raise K.UploadError("Seçilen değişkenler arasında tam doğrusal bağlantı var (ör. bir kontrol diğerlerinin "
                            "doğrusal birleşimi ya da açıklayıcı değişken bir grupta sabit). M3 tahmin edilemez; ek "
                            "kontrolleri değiştirin.")
    outcome = LOG_Y if case.options.get(LOG, True) else y
    controls = (G01, *case.extras)
    if not stable_design(derived, outcome, ((x,), (x, *controls), (x, XG, *controls))):
        raise K.UploadError(SCALE_MESSAGE)


def sample() -> pd.DataFrame:
    """Örnek dosya: kurgusal ücret verisi (``core.labs.ornek_veri``)."""

    return ucret_verisi()


ROLES = (
    Role(SONUC, "Sonuç değişkeni", "sayisal", True, (2, 3, 4, 5, 6, 7),
         "Açıklanan sayısal değişken (ör. saatlik ücret). Notlardaki gibi logaritması alınabilir."),
    Role(ACIKLAYICI, "Temel açıklayıcı değişken", "sayisal", True, (2, 3, 5, 6, 7),
         "Katsayısı raporlanan değişken (ör. eğitim yılı)."),
    Role(GRUP, "İki kategorili grup değişkeni", "kategorik", True, (2, 5),
         "M2'de gösterge olarak, M3'te açıklayıcı değişkenle etkileşimiyle eklenir (ör. cinsiyet).",
         levels=(2, 2), pick="1 ile kodlanan grup"),
)

OPTIONS = (
    Option(LOG, "Sonucun logaritmasını al (log Y)",
           "Notlardaki gibi sonucun doğal logaritması kullanılır; katsayılar yaklaşık yüzde farklar olarak okunur.",
           default=True, role=SONUC, allowed=K1._positive,
           blocked="Sonuç sütununda sıfır ya da negatif değer olduğu için logaritma alınamaz; sonuç kendi biriminde "
                   "kullanılır."),
)

CUSTOM = CustomLab(
    roles=ROLES,
    build=build,
    sample=sample,
    intro=(
        "Bir Excel (.xlsx) ya da CSV dosyası yükleyin. Sonuç, temel açıklayıcı değişken ve iki kategorili bir grup "
        "değişkeni seçilir; ek sayısal kontroller M2 ve M3'e eklenir. Seçilen sütunların birinde boş hücresi olan "
        "satırlar analizden çıkarılır: üç model aynı gözlemlerle tahmin edilir."
    ),
    options=OPTIONS,
    min_rows=10,
    extra_columns=True,
    extra_use="sayisal",
    extra_label="Ek sayısal kontroller (isteğe bağlı, en çok 8)",
    extra_help="M2 ve M3'e eklenir (ör. deneyim, medeni durum göstergesi). Boş hücresi olan satırlar çıkarılır.",
    max_extra=8,
    extra_required=True,
    validate=validate,
)

VARIANTS = TopicVariants(alternative=alternative, story=STORY, custom=CUSTOM)
