"""Konu 12 genel uygulaması: DML'i tanımlama stratejisinin yerine koymamak.

Notlardaki §12.15'in beş adımı aynı numaralarla ve aynı işlemlerle, verisi değiştirilebilir biçimde yazılır (``build``):
aynı estimand için üç tahmin yolu (kontrolsüz regresyon, doğrusal kontrollü OLS, Lasso yardımcı modelli çapraz
uyarlamalı DML ve 11 kat kuralının medyanı), çapraz uyarlamanın bağımlılık birimine göre kurulması, bölme duyarlılığı,
benzer OLS ve DML sonuçlarının yorumu ve tez protokolü.

Alternatif örnek Card (1995) verisidir (Hansen'in arşivindeki ``Card1995.dta``): eğitim yılının 1976 log ücretine etkisi,
yalnız tedavi öncesi kontrollerle (yaş, ırk, 1966'daki güney ve metropol göstergeleri, anne ve baba eğitimi, 14 yaşında
aile yapısı ve kütüphane kartı). Tasarım gözlemseldir: tanımlama koşullu bağımsızlık varsayımına dayanır; notlardaki
rastgele deneyin (DDK2011) tersine D için Lasso birçok terim tutar. Deneyim (yaş − eğitim − 6) eğitimin mekanik bir
fonksiyonu olduğu için kontrol değildir: θ aynı yaştaki kişileri karşılaştırır ve kaybedilen deneyimi içerir. Konu 11'in
alternatifindeki IQ, KWW, 1976 konumu ve okula kayıt da eğitimden etkilenebildiği için kullanılmaz. "Kendi verini yükle"
seçeneğinde aynı adımlar öğrencinin dosyasıyla kurulur. Notlardaki laboratuvar (``core.labs.konu12``) değişmez.

Rollerin notlardaki karşılıkları: sonuç toplam test puanı, tedavi tracking, kontroller altı başlangıç değişkeni, küme
okul (kat ve standart hata birimi).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd

from core.labs import expr as E
from core.labs import kendi_veri as K
from core.labs import penalized as P
from core.labs.ornek import (
    Case,
    CustomLab,
    Role,
    TopicVariants,
    deger,
    exact_fit,
    full_rank,
    md,
    sayi,
    sayim,
    with_app_values,
    with_expected,
)
from core.labs.ornek_veri import kurs_verisi
from core.labs.penalized import GOLDEN
from core.labs.spec import (
    OLS,
    THETA,
    Check,
    CoefTarget,
    CrossFitDML,
    Derive,
    Dictionary,
    DMLSplits,
    DropMissing,
    EffectTable,
    EstimatePlot,
    GroupRank,
    GroupSummary,
    LabSpec,
    LabStep,
    LoadHansen,
    NoteRef,
    Operation,
    ReproClass,
    RowNumber,
    ScalarTable,
    ScalarTarget,
    StatTarget,
    dictionary_terms,
    dml_fold_key,
)


def _sayi(value: float, decimals: int = 0) -> str:
    """Metinlerde Türkçe sayı, binlik ayırıcıyla (12.345,6)."""

    return sayi(value, decimals, binlik=True)


TOPIC = "konu12"
SECTION = "12.15"
FOLDS = 5
SQRT2 = math.sqrt(2.0)
RULES = (("phi", "φ", GOLDEN),) + tuple((f"kok{p}", f"√{p}", math.sqrt(float(p)))
                                         for p in (3, 5, 7, 11, 13, 17, 19, 23, 29, 31))
SONUC, TEDAVI, KUME = "sonuc", "tedavi", "kume"


@dataclass(frozen=True)
class Plan:
    """Rollerin koddaki sütunları, sözlük ve gösterim ayarları.

    ``rank``: katların kurulduğu sıra numarası (küme varsa küme sırası, yoksa gözlem sırası); ``linear``: kontrollü
    OLS'in kontrolleri; ``terms``: yardımcı Lasso modellerinin terimleri (sözlük); ``cluster``: kat ve standart hata
    birimi (yoksa gözlem, HC1). ``decimals``: tahmin ve SH'lerin ondalığı.
    """

    frame: str
    y: str
    d: str
    prepare: tuple[Operation, ...]
    rank: str
    linear: tuple[str, ...]
    terms: tuple[str, ...]
    cluster: str | None
    grid: tuple[float, float, int]
    labels: dict
    rows: tuple[tuple[str, str, str], ...]
    titles: dict
    """``tablo`` (karşılaştırma) ve ``grafik`` (tahminler) başlıkları; ``eksen`` grafiğin x ekseni."""
    decimals: int = 4
    count_label: str = "Analiz örneklemi (N)"
    unit_label: str | None = None
    fold_columns: tuple[tuple[str, str, str], ...] = ()


def _fold(rank: str, multiplier: float) -> E.Expr:
    """⌊5{r·c}⌋ + 1."""

    scaled = E.mul(E.var(rank), multiplier)
    return E.add(E.floor(E.mul(FOLDS, E.sub(scaled, E.floor(scaled)))), 1)


def _steps(plan: Plan, texts: dict) -> tuple[LabStep, ...]:
    f, y, d, dd = plan.frame, plan.y, plan.d, plan.decimals
    vcov = {"vcov": "cluster", "cluster": plan.cluster} if plan.cluster else {"vcov": "HC1"}

    def coef(model: str, term: str, label: str, quantity: str = "coef") -> Check:
        return Check(label, CoefTarget(model, term, quantity), 0.0, decimals=dd)

    se_kind = "küme" if plan.cluster else "HC1"
    names = ("Kontrolsüz regresyon", "Doğrusal kontrollü OLS")
    unit_checks = ((Check(plan.unit_label, StatTarget(f, plan.rank, "max"), 0.0, decimals=0),)
                   if plan.unit_label else ())
    return (
        LabStep(
            number=1,
            title="Aynı estimand için üç tahmin yolu",
            note=NoteRef(SECTION, 1, ("Tablo 12.2", "Şekil 12.4")),
            explanation=texts[1][0],
            operations=(
                *plan.prepare,
                Derive(f, "dis_kat", _fold(plan.rank, GOLDEN),
                       "DML dış katı: ⌊5{rφ}⌋ + 1, φ = (√5 − 1)/2" + ("; bütün küme aynı katta" if plan.cluster else "")),
                Derive(f, "ic_kat", _fold(plan.rank, SQRT2), "Lasso cezası için iç kat: ⌊5{r√2}⌋ + 1"),
                OLS("ham", f, y, (d,), **vcov),
                OLS("ayarli", f, y, (d, *plan.linear), **vcov),
                CrossFitDML("dml", f, y, d, plan.terms, "dis_kat", "ic_kat", plan.grid, "min", plan.cluster),
                DMLSplits("bolmeler", "dml", plan.rank, RULES, FOLDS, "bolme_tablosu",
                          unit="kümelerin" if plan.cluster else "gözlemlerin", decimals=dd),
                EffectTable(plan.rows, "karsilastirma", plan.titles["tablo"]),
                EstimatePlot(plan.rows, "tahminler", plan.titles["eksen"], plan.titles["grafik"],
                             splits="bolme_tablosu", splits_row=3, decimals=dd),
            ),
            checks=(
                Check(plan.count_label, StatTarget(f, y, "count"), 0.0, decimals=0),
                *unit_checks,
                coef("ham", d, f"{names[0]}: katsayı"),
                coef("ham", d, f"{names[0]}: SH ({se_kind})", "se"),
                coef("ayarli", d, f"{names[1]}: katsayı"),
                coef("ayarli", d, f"{names[1]}: SH ({se_kind})", "se"),
                coef("dml", THETA, "DML (Lasso, φ kuralı)"),
                coef("dml", THETA, "DML: SH", "se"),
                coef("bolmeler", THETA, "DML, 11 bölmenin medyanı"),
                coef("bolmeler", THETA, "DML medyan SH", "se"),
                Check("D için Lasso: sıfırdan farklı katsayı (en çok)", ScalarTarget(dml_fold_key("dml", "d", "max")),
                      0.0, decimals=0),
                Check("Y için Lasso: sıfırdan farklı katsayı (en az)", ScalarTarget(dml_fold_key("dml", "y", "min")),
                      0.0, decimals=0),
                Check("Y için Lasso: sıfırdan farklı katsayı (en çok)", ScalarTarget(dml_fold_key("dml", "y", "max")),
                      0.0, decimals=0),
            ),
            reproducibility=ReproClass.CONVENTION,
            takeaway=texts[1][1],
            code_note=texts.get("kod1", ""),
            note_for=texts.get((1, "not")),
        ),
        LabStep(
            number=2,
            title=texts.get("baslik2", "Çapraz uyarlamayı bağımlılık birimine göre kurmak"),
            note=NoteRef(SECTION, 2),
            explanation=texts[2][0],
            operations=(GroupSummary(f, "dis_kat", plan.fold_columns, "dis_katlar"),),
            reproducibility=ReproClass.EXACT,
            takeaway=texts[2][1],
            note_for=texts.get((2, "not")),
        ),
        LabStep(
            number=3,
            title="DML makalesindeki sonuç tablosunu nasıl okumalıyız?",
            note=NoteRef(SECTION, 3),
            explanation=texts[3][0],
            operations=(
                ScalarTable(
                    (
                        ("En küçük θ̂ (11 bölme)", E.ref("bolmeler_min")),
                        ("En büyük θ̂ (11 bölme)", E.ref("bolmeler_max")),
                        ("Medyan θ̂", E.coef("bolmeler", THETA)),
                        ("Medyan SH", E.se("bolmeler", THETA)),
                    ),
                    "bolme_ozeti", decimals=dd,
                ),
            ),
            checks=(
                Check("11 bölmede en küçük tahmin", ScalarTarget("bolmeler_min"), 0.0, decimals=dd),
                Check("11 bölmede en büyük tahmin", ScalarTarget("bolmeler_max"), 0.0, decimals=dd),
            ),
            reproducibility=ReproClass.CONVENTION,
            takeaway=texts[3][1],
            note_for=texts.get((3, "not")),
        ),
        LabStep(
            number=4,
            title="Benzer OLS ve DML sonuçlarını nasıl yorumlamalıyız?",
            note=NoteRef(SECTION, 4),
            explanation=texts[4][0],
            takeaway=texts[4][1],
            note_for=texts.get((4, "not")),
        ),
        LabStep(
            number=5,
            title="Kendi tezinizde DML için uygulama protokolü",
            note=NoteRef("12.15.5", 0),
            explanation=PROTOCOL,
            takeaway=texts[5][1],
            note_for=texts.get((5, "not")),
        ),
    )


def _spec(plan: Plan, texts: dict, *, source: str, title: str, dataset: str) -> LabSpec:
    labels = {**plan.labels, "dis_kat": "Dış kat", "ic_kat": "İç kat", "ham": plan.rows[0][0],
              "ayarli": plan.rows[1][0], "dml": "DML (φ kuralı)", "bolmeler": "DML, 11 bölmenin medyanı",
              THETA: f"θ ({plan.labels.get(plan.d, plan.d)})"}
    return LabSpec(topic_key=TOPIC, title=title, dataset=dataset, note_section=SECTION, steps=_steps(plan, texts),
                   labels=tuple(labels.items()), source=source)


# --- Ortak metinler -------------------------------------------------------------------------------------------

PROTOCOL = (
    "1. Önce estimand ve tanımlama varsayımını düz yazıyla açıklayın.\n"
    "2. Hangi değişkenlerin treatment sonrası olduğunu kontrol edin; post-treatment kontrolleri nuisance setine "
    "mekanik biçimde eklemeyin.\n"
    "3. Fold yapısını bağımlılık birimine göre kurun.\n"
    "4. Ön işleme ve tuning'i her eğitim katı içinde yapın.\n"
    "5. En az iki makul yardımcı model öğrenicisi ile duyarlılık düşünün.\n"
    "6. Hedef katsayı için veri yapısına uygun robust/küme standart hata kullanın.\n"
    "7. DML sonucunu basit benchmark modellerle yan yana raporlayın.\n"
    "8. Replikasyon paketinde kat kuralını veya seed'i, öğrenici ayarlarını ve paket sürümlerini saklayın; birkaç "
    "bölme altında duyarlılığı raporlayın."
)
DML_FORMULA = (
    "Hedef katsayı artıkların artıklar üzerine sabitsiz regresyonudur:\n\n"
    "$$\\widehat\\theta=\\frac{\\sum_i\\widehat V_i\\widehat U_i}{\\sum_i\\widehat V_i^2},\\qquad "
    "\\widehat U_i=Y_i-\\widehat m_Y^{(-k)}(X_i),\\quad \\widehat V_i=D_i-\\widehat m_D^{(-k)}(X_i).$$"
)
SPLITS_TEXT = (
    "DML çıktısında yalnız hedef katsayı ve p-değerini görmek yeterli değildir. En az şu ayrıntılar raporlanmalıdır:\n\n"
    "1. hedef parametre ve tanımlama varsayımı,\n"
    "2. treatment ve outcome tanımları,\n"
    "3. kullanılan yardımcı model öğrenicileri,\n"
    "4. tuning prosedürü,\n"
    "5. kat sayısı ve bölme birimi,\n"
    "6. ön işlemenin kat içinde yapılıp yapılmadığı,\n"
    "7. hedef katsayı için standart hata türü,\n"
    "8. farklı öğrenici, kat ve seed seçimlerinde duyarlılık.\n\n"
    "Aynı hesabı dış katlarda $\\varphi$ yerine 3 ile 31 arasındaki on asal sayının kareköküyle kurulan on farklı kuralla "
    "tekrarlıyoruz; yalnız gözlemlerin katlara dağılımı değişir, veri, öğrenici ve kat sayısı aynıdır. On bir bölme "
    "Chernozhukov vd. (2018) gibi birleştirilir:\n\n"
    "$$\\widehat\\theta_{med}=\\operatorname{medyan}_s\\widehat\\theta_s,\\qquad "
    "\\widehat V_{\\widehat\\theta,med}=\\operatorname{medyan}_s\\{\\widehat V_{\\widehat\\theta,s}+(\\widehat\\theta_s-"
    "\\widehat\\theta_{med})^2\\}.$$"
)


def _grid_text(grid: tuple[float, float, int]) -> str:
    high, low, count = grid

    def power(value: float) -> str:
        return _sayi(value, 1 if value % 1 else 0).replace(",", "{,}")

    return f"$\\lambda_y\\in[10^{{{power(low)}}},10^{{{power(high)}}}]$ aralığında {count} noktalı logaritmik ızgarada"


# --- Alternatif örnek: Card (1995) ----------------------------------------------------------------------------

ALT_DATA = "card1995"
ALT_FRAME = "card"
ALT_TITLE = "Card Verisinde DML'i Tanımlama Stratejisinin Yerine Koymamak"
ALT_CONTROLS = ("age76", "black", "south66", "smsa66r", "momed", "daded", "momdad14", "libcrd14")
ALT_CENTERS = (("yas", "age76", 28), ("anne", "momed", 10), ("baba", "daded", 10))
ALT_DICTIONARY = Dictionary(
    ALT_FRAME, ("yas", "black", "south66", "smsa66r", "anne", "baba", "momdad14", "libcrd14"), ("yas", "anne", "baba"), 3,
    True, "Aday terim sözlüğü: sekiz değişken, üç sürekli değişkenin 2. ve 3. kuvvetleri, 28 ikili etkileşim",
)
ALT_GRID = (1.0, -4.0, 51)
ALT_LABELS = {
    "lwage76": "Log saatlik ücret (1976)", "ed76": "Eğitim yılı (1976)", "age76": "Yaş (1976)", "black": "Siyahi",
    "south66": "Güney (1966)", "smsa66r": "Metropol (1966)", "momed": "Anne eğitimi", "daded": "Baba eğitimi",
    "momdad14": "14 yaşında anne ve babayla", "libcrd14": "14 yaşında kütüphane kartı", "yas": "Yaş − 28",
    "anne": "Anne eğitimi − 10", "baba": "Baba eğitimi − 10", "sira": "Gözlem sıra numarası", "kisi": "Kişi",
    "egitim_ort": "Ortalama eğitim yılı",
}
ALT_ROWS = (
    ("Kontrolsüz regresyon, HC1", "ham", "ed76"),
    ("Doğrusal kontrollü OLS, HC1", "ayarli", "ed76"),
    ("DML, Lasso, bireysel çapraz uyarlama", "dml", THETA),
    ("DML, 11 kat kuralının medyanı", "bolmeler", THETA),
)


def alternative_plan() -> Plan:
    frame = ALT_FRAME
    prepare = (
        LoadHansen(ALT_DATA, "Card1995.dta", frame),
        DropMissing(frame, ("lwage76", "ed76", *ALT_CONTROLS),
                    "Analiz örneklemi: 1976 log ücreti, eğitimi ve tedavi öncesi kontrolleri gözlenen erkekler"),
        *(Derive(frame, name, E.sub(E.var(source), center), f"{ALT_LABELS[source]} − {center}")
          for name, source, center in ALT_CENTERS),
        ALT_DICTIONARY,
        RowNumber(frame, "sira", "Gözlem sıra numarası r = 1, …, n (hane kimliği olmadığı için her kişi ayrı birim)"),
    )
    return Plan(
        frame=frame, y="lwage76", d="ed76", prepare=prepare, rank="sira", linear=ALT_CONTROLS,
        terms=dictionary_terms(ALT_DICTIONARY), cluster=None, grid=ALT_GRID, labels=ALT_LABELS, rows=ALT_ROWS,
        titles={"tablo": "Card1995: eğitimin log ücrete etkisi için OLS ve DML karşılaştırması",
                "grafik": "Card1995: aynı araştırma sorusu, farklı tahmin yolları",
                "eksen": "Eğitim yılının katsayısı (log ücret)"},
        decimals=4, fold_columns=(("kisi", "lwage76", "count"), ("egitim_ort", "ed76", "mean")),
    )


ALT_TEXTS = {
    1: (
        "Bu kez tasarım rastgele değil gözlemseldir: Card'ın (1995) NLSYM verisinde eğitim yılının ($D$) 1976 log ücretine "
        "($Y$) etkisi. Tanımlama varsayımı koşullu bağımsızlıktır: aynı tedavi öncesi özelliklere ($X$) sahip kişiler "
        "arasında eğitim, potansiyel ücretlerle ilişkisiz kabul edilir. DML bu varsayımı sağlamaz; yalnız $X$'in "
        "fonksiyonel biçimini esnek ayarlar.\n\n"
        "$X$ yalnız eğitimden önce belirlenmiş değişkenlerdir: yaş, siyahi, 1966'daki güney ve metropol göstergeleri, anne "
        "ve baba eğitimi, 14 yaşında anne ve babayla yaşama ve kütüphane kartı. Deneyim (yaş − eğitim − 6) eğitimin "
        "mekanik bir fonksiyonu olduğu için kontrol edilmez; bu yüzden $\\theta$ aynı yaştaki kişileri karşılaştırır ve "
        "bir yıl fazla eğitimin bir yıl kaybedilen deneyimini de içerir (Konu 1–2'deki deneyimi sabit tutan katsayıdan "
        "küçüktür). 2.997 erkekle yardımcı modeller $m_Y(X)=E[Y\\mid X]$ ve $m_D(X)=E[D\\mid X]$ Lasso'dur ve 42 terimli "
        "bir sözlük kullanır: sekiz değişkenin kendisi (yaş, anne ve baba eğitimi yuvarlak bir sabitten farkları olarak), "
        "üç sürekli değişkenin ikinci ve üçüncü kuvvetleri ve 28 ikili etkileşim. " + DML_FORMULA + "\n\nStandart hatalar "
        "HC1'dir (her kişi ayrı birim kabul edilir; Adım 2). Son satır, dış katları 11 farklı kuralla kuran bölmelerin "
        "medyanıdır (Adım 3)."
    ),
    2: (
        "NLSYM hane temelli bir örneklemdir: başlangıçta genç erkek katılımcıların yaklaşık %80'i başka bir NLS "
        "katılımcısıyla (ör. anne, baba ya da kardeş) aynı hanede yaşıyordu; yaklaşık üçte biri bu dosyanın kohortundan "
        "başka bir genç erkekle (ör. erkek kardeşiyle) aynı hanedeydi. Hansen'in dosyasında hane kimliği yoktur; bu "
        "yüzden burada her kişi ayrı birim kabul edilir. Katlar kişilere açık bir kuralla atanır: gözlem sıra numarası $r=1,\\ldots,2.997$ olmak "
        "üzere\n\n"
        "$$\\text{kat}=\\lfloor5\\{r\\varphi\\}\\rfloor+1,\\qquad \\varphi=\\frac{\\sqrt5-1}{2}.$$\n\n"
        "Her eğitim katında Lasso cezası da aynı birimde katlarla seçilir: iç kat $\\lfloor5\\{r\\sqrt2\\}\\rfloor+1$'dir. "
        "Aşağıdaki tablo dış katların büyüklüğünü ve her kattaki ortalama eğitimi gösterir.",
        "Kümeli bir veride (notlardaki okul) aynı kümenin gözlemleri aynı katta kalmalıdır. Burada hane kimliği "
        "olmadığı için her kişi kendi kümesi sayılır: aynı hanedeki kardeşler farklı katlara düşebilir ve HC1 standart "
        "hatası hane içi bağımlılığı hesaba katmaz. Kat yapısı ve standart hata türü aynı bağımlılık birimine göre "
        "seçilir; birim veride yoksa bu bir sınırlama olarak raporlanır.",
    ),
    3: (SPLITS_TEXT, ""),
    4: (
        "Bu örnekte doğrusal kontrollü OLS ile DML nokta tahminleri birbirine yakındır. Bu, kontrollerin doğrusal biçiminin "
        "bu veri için yeterli olabileceğini gösterir; gözlenmeyen karıştırıcı olmadığını göstermez. Notlardaki deneyin "
        "tersine burada tanımlamayı destekleyen bir rastgele atama yoktur: yetenek gibi gözlenmeyen bir özellik hem "
        "eğitimi hem ücreti etkiliyorsa ikisi de aynı yanlılığı taşır.\n\n"
        "Konu 4'te aynı veride koleje yakınlık aracıyla 2SLS katsayısı OLS'inkinin yaklaşık 1,8 katıydı (log puan "
        "olarak 0,1315'e karşı 0,0747; deneyim kontrolleriyle, yani başka bir estimand). DML bu farkı açıklamaz ve gidermez: gözlenen $X$'e koşullanan "
        "her yöntem gözlenmeyen karıştırıcı problemini ortak taşır.",
        "",
    ),
    5: (PROTOCOL, ""),
}

ALT_TAKEAWAYS = {
    1: (
        "Kontrolsüz regresyon 0,0522; tedavi öncesi kontrollerle doğrusal OLS 0,0348, DML 0,0330 (SH 0,0031): bir yıl "
        "fazla eğitim aynı yaştaki ve aynı aile geçmişindeki kişiler arasında log ücrette yaklaşık 0,033 farkla ilişkilidir. "
        "Kontroller tahmini üçte bir küçültür; esnek ayarlama (DML) doğrusal OLS'ten yalnız 0,0018 farklıdır, standart "
        "hatadan (0,0031) küçük bir fark. Tasarımın izi yardımcı modellerde görünür: eğitim aile geçmişi ve bölgeyle güçlü biçimde "
        "öngörüldüğü için D için Lasso katlarda en çok 33 terim tutar (notlardaki rastgele deneyde hiç). Y için Lasso 42 "
        "terimden 29 ile 37 arasını tutar."
    ),
    3: (
        "Tahmin 0,0320 ile 0,0335 arasında değişir; medyan 0,0329, medyan standart hatası 0,0032. En büyük ile en küçük "
        "bölme arasındaki fark (0,0015) standart hatanın yarısından azdır. Tek bir bölmenin sonucunu \"en iyi\" bölme olarak seçmek yerine bölme kuralı "
        "önceden kaydedilmeli ve bölmeler arası değişim raporlanmalıdır."
    ),
    4: (
        "DML, karmaşık fonksiyonel biçimi ve yüksek boyutlu gözlenen kontrolleri yönetmek için güçlüdür; fakat geçerli araç, "
        "randomization, unconfoundedness veya başka bir tanımlama kaynağının yerine geçmez. Örnek sonuç paragrafı: "
        "\"Tedavi öncesi kontrollerle eğitimin log ücrete katsayısı doğrusal OLS'te 0,035, Lasso-DML'de 0,033 olarak tahmin "
        "edilmiştir; on bir kat kuralında DML tahminleri 0,032 ile 0,034 arasında değişmiştir. Nedensel yorum, tedavi "
        "öncesi kontroller verildiğinde eğitimin yetenek gibi gözlenmeyen ücret belirleyicileriyle ilişkisiz olduğu "
        "(koşullu bağımsızlık) varsayımına dayanır; bu varsayım DML ile sınanmamıştır.\""
    ),
    5: (
        "Bu laboratuvarın seçimleri kodda açıktır: tedavi öncesi kontroller, sözlük, ızgara, iç ve dış kat kuralları, "
        "HC1 standart hatası ve 11 bölmelik duyarlılık. Madde 2 bu veride somuttur: deneyim, IQ, KWW ve 1976'daki konum "
        "ücreti iyi öngörür (Konu 11'in alternatifi) ama eğitimden etkilenebilir; nuisance setine eklenmez."
    ),
}

ALT_CODE_NOTE = (
    "Lasso yazılım ölçeğinde, " + _grid_text(ALT_GRID) + " iç CV hatasını en küçük yapan cezadır (notlardaki ızgaradan bir "
    "on yıl aşağı uzanır: φ kuralında Y için seçilen ceza bir katta 10⁻³'ün altındadır); özellikler her eğitim katında "
    "ölçeklenir. "
    "Python `enet_path`, R `glmnet`, Stata kodda yazılı Mata koordinat inişi aynı optimuma yakınsar. 11 bölmenin her "
    "biri 5 × 2 yardımcı model ve her birinde 5 katlı iç CV demektir; hesap birkaç saniye sürer."
)


def _alt_texts() -> dict:
    texts = {1: (ALT_TEXTS[1], ALT_TAKEAWAYS[1]), 2: ALT_TEXTS[2], 3: (ALT_TEXTS[3][0], ALT_TAKEAWAYS[3]),
             4: (ALT_TEXTS[4][0], ALT_TAKEAWAYS[4]), 5: (PROTOCOL, ALT_TAKEAWAYS[5])}
    texts["kod1"] = ALT_CODE_NOTE
    texts["baslik2"] = "Çapraz uyarlamayı bağımlılık birimine göre kurmak"
    return texts


ALT_EXPECTED = {
    (1, "Analiz örneklemi (N)"): 2997,
    (1, "Kontrolsüz regresyon: katsayı"): 0.0522,
    (1, "Kontrolsüz regresyon: SH (HC1)"): 0.0029,
    (1, "Doğrusal kontrollü OLS: katsayı"): 0.0348,
    (1, "Doğrusal kontrollü OLS: SH (HC1)"): 0.0031,
    (1, "DML (Lasso, φ kuralı)"): 0.0330,
    (1, "DML: SH"): 0.0031,
    (1, "DML, 11 bölmenin medyanı"): 0.0329,
    (1, "DML medyan SH"): 0.0032,
    (1, "D için Lasso: sıfırdan farklı katsayı (en çok)"): 33,
    (1, "Y için Lasso: sıfırdan farklı katsayı (en az)"): 29,
    (1, "Y için Lasso: sıfırdan farklı katsayı (en çok)"): 37,
    (3, "11 bölmede en küçük tahmin"): 0.0320,
    (3, "11 bölmede en büyük tahmin"): 0.0335,
}
"""Kontrollerin Card1995 örneklemindeki (2.997 erkek) değerleri; testler bağımsız bir hesapla doğrular."""


@cache
def alternative() -> LabSpec:
    """Alternatif örneğin tanımı; beklenen değerler Card1995 örneklemindeki sayılardır (testle doğrulanır)."""

    spec = _spec(alternative_plan(), _alt_texts(), source="alternatif", title=ALT_TITLE, dataset=ALT_DATA)
    return with_expected(spec, ALT_EXPECTED)


STORY = (
    "Alternatif örnek Card (1995) verisidir: 2.997 genç erkekte eğitim yılının 1976 log ücretine etkisi (Hansen'in "
    "arşivindeki Card1995.dta), yalnız tedavi öncesi kontrollerle. Kontrolsüz regresyon, doğrusal kontrollü OLS ve Lasso "
    "yardımcı modelli DML karşılaştırılır. Tasarım gözlemseldir: DML esnek ayarlama yapar, gözlenmeyen yetenek sorununu "
    "çözmez."
)


# --- Kendi verin --------------------------------------------------------------------------------------------

MAX_ROWS = 5000
"""Kendi verinde en çok gözlem: 11 bölme × 5 dış kat × 2 yardımcı model × 5 katlı iç CV (indirilen kodda da)."""
MIN_ROWS = 100
MIN_CONTROLS = 2
MAX_TERMS = 100
MIN_CLUSTERS = 20
MIN_GROUP = 20
"""Tedavi iki değerliyse her değerde en az gözlem."""


def _significant(value: float, digits: int = 2) -> float:
    if value == 0 or not math.isfinite(value):
        return 0.0
    return float(round(value, digits - 1 - math.floor(math.log10(abs(value)))))


def _value(value: float) -> str:
    return deger(value)


def _new_name(base: str, suffix: str, taken: set[str]) -> str:
    name = f"{base}_{suffix}"
    while name in taken:
        name += "_"
    taken.add(name)
    return name


def _ranks(case: Case) -> np.ndarray:
    """Katların sıra numarası: küme varsa küme sırası (küme değerlerinin sıralı düzeni), yoksa gözlem sırası."""

    data = case.data
    if case.has(KUME):
        return P.dense_rank(data[case.roles[KUME]].to_numpy()).astype(float)
    return np.arange(1, len(data) + 1, dtype=float)


def _subsets(ranks: np.ndarray) -> list[np.ndarray]:
    """Ölçeklerin öğrenildiği bütün satır kümeleri: 11 kat kuralının her dış eğitim katı ve onun iç CV eğitim kısımları."""

    inner = P.multiplier_folds(ranks, SQRT2, FOLDS)
    subsets = []
    for _, _, multiplier in RULES:
        outer = P.multiplier_folds(ranks, multiplier, FOLDS)
        for k in range(1, FOLDS + 1):
            train = outer != k
            subsets.append(train)
            subsets += [train & (inner != j) for j in range(1, FOLDS + 1)]
    return subsets


@dataclass(frozen=True)
class OwnDesign:
    derive: tuple[Operation, ...]
    dictionary: Dictionary
    excluded: tuple[str, ...]
    centered: tuple[tuple[str, str, float], ...]
    frame: pd.DataFrame


def own_design(case: Case) -> OwnDesign:
    data, frame_name = case.data, case.frame
    taken = set(data.columns)
    frame = data.copy()
    derive: list[Operation] = []
    centered: list[tuple[str, str, float]] = []
    names: list[str] = []
    for column in case.extras:
        values = frame[column].astype(float)
        if values.nunique() >= 3:
            center = _significant(float(values.mean()))
            name = _new_name(column, "m", taken)
            derive.append(Derive(frame_name, name, E.sub(E.var(column), center),
                                 f"{case.name(column)} − {_value(center)}"))
            frame[name] = values - center
            centered.append((name, column, center))
            names.append(name)
        else:
            names.append(column)
    candidate = Dictionary(frame_name, tuple(names), tuple(name for name, _, _ in centered), 2, True,
                           "Aday terim sözlüğü: kontroller, sürekli olanların kareleri ve bütün ikili çarpımlar")
    created = {}
    for name in candidate.powers:
        created[f"{name}_2"] = frame[name].to_numpy(dtype=float) ** 2
    for index, first in enumerate(candidate.base):
        for second in candidate.base[index + 1:]:
            created[f"{first}_x_{second}"] = frame[first].to_numpy(dtype=float) * frame[second].to_numpy(dtype=float)
    frame = pd.concat([frame, pd.DataFrame(created, index=frame.index)], axis=1)
    terms = dictionary_terms(candidate)
    values = frame[list(terms)].to_numpy(dtype=float)
    constant = np.zeros(len(terms), dtype=bool)
    for mask in _subsets(_ranks(case)):
        part = values[mask]
        constant |= part.max(axis=0) == part.min(axis=0)
    excluded = tuple(term for term, flag in zip(terms, constant) if flag)
    dictionary = Dictionary(frame_name, candidate.base, candidate.powers, 2, True, candidate.comment, exclude=excluded)
    return OwnDesign(tuple(derive), dictionary, excluded, tuple(centered), frame)


def _lasso_max(frame: pd.DataFrame, target: str, terms) -> float:
    x = frame[list(terms)].to_numpy(dtype=float)
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    y = frame[target].to_numpy(dtype=float)
    return float(np.max(np.abs(x.T @ (y - y.mean()))) / len(y))


def own_grid(frame: pd.DataFrame, y: str, d: str, terms) -> tuple[float, float, int]:
    """Y ve D için ortak Lasso ızgarası: büyük λ_max'tan küçük λ_max'ın dört on yıl altına, on yılda on nokta (en çok 81;
    daha geniş aralıkta nokta sayısı 81'de kalır). Üsler bir ondalığa yuvarlanır."""

    values = [_lasso_max(frame, target, terms) for target in (y, d)]
    top = math.ceil(math.log10(max(values)) * 10) / 10
    bottom = math.floor(math.log10(min(values)) * 10) / 10 - 4
    return (top, round(bottom, 1), int(min(81, round((top - bottom) * 10) + 1)))


def own_plan(case: Case) -> Plan:
    import statsmodels.api as sm

    data = case.data
    y, d = case.roles[SONUC], case.roles[TEDAVI]
    cluster = case.roles.get(KUME)
    design = own_design(case)
    terms = dictionary_terms(design.dictionary)
    linear = case.extras
    fit = sm.OLS(data[y].astype(float), sm.add_constant(data[[d, *linear]].astype(float))).fit(cov_type="HC1")
    se = float(fit.bse[d])
    decimals = int(min(8, max(2, 2 - math.floor(math.log10(se))))) if se > 0 else 4
    rank = "kume_sira" if cluster else "sira"
    rank_op = (GroupRank(case.frame, rank, cluster, "Küme sıra numarası r = 1, …, G (küme değerlerinin sırasıyla)")
               if cluster else RowNumber(case.frame, rank, "Gözlem sıra numarası r = 1, …, n"))
    labels = {column: case.name(column) for column in data.columns}
    labels.update({name: f"{case.name(source)} − {_value(center)}" for name, source, center in design.centered})
    labels.update({rank: "Küme sıra numarası" if cluster else "Gözlem sıra numarası", "gozlem": "Gözlem",
                   "tedavi_ort": f"Ortalama “{case.name(d)}”"})
    se_kind = "küme SH" if cluster else "HC1"
    return Plan(
        frame=case.frame, y=y, d=d, prepare=case.load + design.derive + (design.dictionary, rank_op), rank=rank,
        linear=linear, terms=terms, cluster=cluster, grid=own_grid(design.frame, y, d, terms), labels=labels,
        rows=((f"Kontrolsüz regresyon, {se_kind}", "ham", d), (f"Doğrusal kontrollü OLS, {se_kind}", "ayarli", d),
              ("DML, Lasso, çapraz uyarlama", "dml", THETA), ("DML, 11 kat kuralının medyanı", "bolmeler", THETA)),
        titles={"tablo": "Kendi veriniz: OLS ve DML karşılaştırması",
                "grafik": "Kendi veriniz: aynı araştırma sorusu, farklı tahmin yolları",
                "eksen": f"“{case.name(d)}” katsayısı"},
        decimals=decimals, unit_label="Küme sayısı" if cluster else None,
        fold_columns=(("gozlem", y, "count"), ("tedavi_ort", d, "mean")),
    )


def _span(scalars, target: str) -> str:
    """Katlardaki sıfırdan farklı katsayı sayısı: "3–6" ya da (hepsi aynıysa) "5"."""

    low, high = int(scalars[dml_fold_key("dml", target, "min")]), int(scalars[dml_fold_key("dml", target, "max")])
    return str(low) if low == high else f"{low}–{high}"


def _own_texts(case: Case, plan: Plan, design: OwnDesign) -> dict:
    y, d = case.md(SONUC), case.md(TEDAVI)
    cluster = case.md(KUME) if plan.cluster else None
    dd = plan.decimals
    controls = ", ".join(f"“{md(case.name(column))}”" for column in case.extras)
    dictionary = (f"{len(plan.terms)} terimli bir sözlük kullanır: kontrollerin kendileri (sürekli olanlar yuvarlak bir "
                  "sabitten farkları olarak), sürekli kontrollerin kareleri ve bütün ikili çarpımlar")
    if design.excluded:
        dictionary += (f" ({len(design.excluded)} çarpım ya da kare bir eğitim katında sabit kaldığı için çıkarıldı)")
    unit = (f"“{cluster}” kümeleri" if plan.cluster else "gözlemler")

    def step1(state) -> str:
        models = state.models
        raw, linear = float(models["ham"].params[plan.d]), float(models["ayarli"].params[plan.d])
        dml, dml_se = float(models["dml"].params[THETA]), float(models["dml"].bse[THETA])
        s = state.scalars
        text = (f"Kontrolsüz regresyon {_sayi(raw, dd)}, doğrusal kontrollü OLS {_sayi(linear, dd)}, DML {_sayi(dml, dd)} "
                f"(SH {_sayi(dml_se, dd)}). DML ile doğrusal OLS arasındaki fark {_sayi(abs(dml - linear), dd)}: "
                + ("standart hatadan küçük." if abs(dml - linear) < dml_se else "standart hatadan büyük; kontrollerin "
                   "doğrusal olmayan etkileri tahmini değiştiriyor olabilir.")
                + f" D için Lasso katlarda {_span(s, 'd')}, Y için Lasso {_span(s, 'y')} terim tutar.")
        if int(s[dml_fold_key("dml", "d", "max")]) == 0:
            text += (" D için Lasso hiçbir terim tutmuyor: tedavi kontrollerle öngörülemiyor (rastgele atamada olduğu "
                     "gibi).")
        return text

    def step3(state) -> str:
        s = state.scalars
        low, high = float(s["bolmeler_min"]), float(s["bolmeler_max"])
        se = float(state.models["bolmeler"].bse[THETA])
        return (f"Tahmin {_sayi(low, dd)} ile {_sayi(high, dd)} arasında değişir; medyan "
                f"{_sayi(float(state.models['bolmeler'].params[THETA]), dd)}, medyan SH {_sayi(se, dd)}. En büyük ile en "
                f"küçük bölme arasındaki fark ({_sayi(high - low, dd)}) standart hatanın "
                + ("yarısından azdır." if high - low < se / 2 else "yarısından büyüktür: sonuç bölmeye duyarlıdır."))

    se_text = (f"Standart hatalar “{cluster}” düzeyinde kümelenmiştir." if plan.cluster else
               "Standart hatalar HC1'dir (küme seçilmedi).")
    return {
        1: (f"Soru: “{d}” değişkeninin “{y}” üzerindeki etkisi. Kontroller ({controls}) yalnız tedaviden önce belirlenmiş "
            "olmalıdır: tedaviden etkilenen bir değişkeni kontrol etmek tahmini yanlı kılar (Adım 5, madde 2). Tanımlama "
            "varsayımı tasarımdan gelir (rastgele atama ya da koşullu bağımsızlık); DML yalnız kontrollerin fonksiyonel "
            f"biçimini esnek ayarlar. Yardımcı modeller Lasso'dur ve {dictionary}. " + DML_FORMULA + "\n\n"
            + se_text + " Son satır, dış katları 11 farklı kuralla kuran bölmelerin medyanıdır (Adım 3).",
            "OLS ile DML'in yakın çıkması gözlenmeyen karıştırıcı olmadığını göstermez."),
        (1, "not"): step1,
        2: (f"Katlar {unit} üzerinden açık bir kuralla kurulur: sıra numarası $r$ için "
            "$\\text{kat}=\\lfloor5\\{r\\varphi\\}\\rfloor+1$; Lasso cezası her eğitim katında $\\lfloor5\\{r\\sqrt2\\}"
            "\\rfloor+1$ iç katlarıyla seçilir." + (" Bütün küme aynı katta kalır: yardımcı model aynı kümenin "
                                                    "değerlendirme gözlemlerini eğitimde görmez." if plan.cluster else
                                                    " Veri kümeliyse küme sütununu seçin."),
            "Kat yapısı ve standart hata türü aynı bağımlılık birimine göre seçilir."),
        3: (SPLITS_TEXT, "Bölme kuralı önceden kaydedilmeli ve bölmeler arası değişim raporlanmalıdır."),
        (3, "not"): step3,
        4: ("Doğrusal kontrollü OLS ile DML yakınsa, doğrusal kontrol biçimi bu veri için yeterli olabilir. Bu yakınlık "
            "gözlenmeyen karıştırıcı olmadığına ilişkin bir test değildir: ikisi de yalnız gözlenen kontrollere koşullanır.",
            "DML geçerli araç, rastgele atama, koşullu bağımsızlık veya başka bir tanımlama kaynağının yerine geçmez."),
        5: (PROTOCOL, "Kontrollerin tedaviden önce ölçüldüğünü, kat yapısının bağımlılık birimine uyduğunu ve bölme "
                      "duyarlılığını raporlayın."),
        "kod1": ("Lasso yazılım ölçeğinde, " + _grid_text(plan.grid) + " iç CV hatasını en küçük yapan cezadır; ızgara veriden "
                 "kurulur (Y ve D için Lasso'nun bütün katsayıları sıfırladığı en küçük cezalardan dört on yıl aşağı). "
                 "Python `enet_path` ve R `glmnet` aynı optimuma yakınsar."),
        "baslik2": "Çapraz uyarlamayı bağımlılık birimine göre kurmak",
    }


def build(case: Case) -> LabSpec:
    """Kendi verinle Konu 12 uygulaması; kontrollerin beklenen değerleri uygulamanın hesabıdır."""

    design = own_design(case)
    plan = own_plan(case)
    spec = _spec(plan, _own_texts(case, plan, design), source="kendi",
                 title=f"{case.label(TEDAVI)}: DML ve tanımlama", dataset="")
    return with_app_values(spec)


def validate(case: Case) -> None:
    from core.labs.ornek import stable_design, SCALE_MESSAGE

    data = case.data
    y, d = case.roles[SONUC], case.roles[TEDAVI]
    cluster = case.roles.get(KUME)
    if y == d or cluster in (y, d) or y in case.extras or d in case.extras:
        raise K.UploadError("Sonuç, tedavi, küme ve kontroller için farklı sütunlar seçin.")
    if len(data) < MIN_ROWS:
        raise K.UploadError(f"DML için en az {MIN_ROWS} gözlem gerekir; seçilen sütunlarda {len(data)} gözlem var.")
    if len(data) > MAX_ROWS:
        raise K.UploadError(f"Analizde {sayim(len(data))} gözlem var; 11 bölme ve iç çapraz doğrulama süresi nedeniyle en çok "
                            f"{sayim(MAX_ROWS)} gözlem kullanılabilir.")
    if len(case.extras) < MIN_CONTROLS:
        raise K.UploadError(f"En az {MIN_CONTROLS} kontrol seçin (DML'in yardımcı modelleri kontrollerden kurulur).")
    for column in (y, d, *case.extras):
        if data[column].nunique() < 2:
            raise K.UploadError(f"“{case.name(column)}” sütunu sabit.")
    values = data[d].astype(float)
    if values.nunique() == 2 and values.value_counts().min() < MIN_GROUP:
        raise K.UploadError(f"“{case.name(d)}” iki değerli ve değerlerinden biri {MIN_GROUP}'den az gözlemde: çapraz "
                            "uyarlama katlarında tedavi sabit kalabilir.")
    columns = (d, *case.extras)
    if not full_rank(data, columns):
        raise K.UploadError("Tedavi ve kontroller arasında tam doğrusal bağlantı var; kontrollerden birini çıkarın.")
    if exact_fit(data, y, columns) or exact_fit(data, d, case.extras):
        raise K.UploadError("Sonuç ya da tedavi kontrollerin (neredeyse) tam doğrusal bir fonksiyonu (R² ≈ 1): artıklar "
                            "yuvarlama hatasından oluşur.")
    if not stable_design(data, y, (columns,)):
        raise K.UploadError(SCALE_MESSAGE)
    if cluster:
        count = data[cluster].nunique()
        if count < MIN_CLUSTERS:
            raise K.UploadError(f"“{case.name(cluster)}” sütununda {count} küme var; kümeli çapraz uyarlama ve küme standart "
                                f"hatası için en az {MIN_CLUSTERS} küme gerekir.")
        if count == len(data):
            raise K.UploadError(f"“{case.name(cluster)}” sütununda her gözlem ayrı bir küme. Küme seçmeyin ya da doğru küme "
                                "sütununu seçin.")
    design = own_design(case)
    terms = dictionary_terms(design.dictionary)
    if len(terms) > MAX_TERMS:
        raise K.UploadError(f"Sözlükte {len(terms)} terim var; en çok {MAX_TERMS}. Daha az kontrol seçin.")
    if len(terms) < MIN_CONTROLS or len(design.excluded) > len(terms):
        raise K.UploadError("Kontrollerin kareleri ve çarpımlarının çoğu eğitim katlarında sabit kalıyor (ör. seyrek "
                            "göstergeler). Başka kontroller seçin.")
    base_dropped = [name for name in design.dictionary.base if name in design.excluded]
    if base_dropped:
        raise K.UploadError("Bazı kontroller bir eğitim katında sabit kalıyor (ör. çok seyrek bir gösterge); bu "
                            "kontrolleri çıkarın.")


def suggest(table: K.UploadedTable) -> dict[str, str]:
    """Küme rolü için adında okul, küme, firma, sınıf, il, köy ya da kurum sözcüğü geçen ve 20 ile gözlem sayısının
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
    """Örnek dosya: kurgusal mesleki kurs verisi (``core.labs.ornek_veri``)."""

    return kurs_verisi()


ROLES = (
    Role(SONUC, "Sonuç değişkeni", "sayisal", True, (1, 3, 4), "Etkisi ölçülen sonuç (notlarda toplam test puanı)."),
    Role(TEDAVI, "Tedavi", "sayisal", True, (1, 2, 3, 4),
         "Etkisi tahmin edilen değişken; iki değerli (0/1) ya da sürekli (notlarda tracking)."),
    Role(KUME, "Küme (ör. okul, firma)", "serbest", False, (1, 2),
         "Gözlemlerin bağımlı olduğu birim. Seçilirse katlar kümelere göre kurulur ve standart hatalar kümelenir; "
         "seçilmezse gözlemler bağımsız sayılır (HC1). Boş hücre olamaz; en az 20 küme.", complete=True),
)

CUSTOM = CustomLab(
    roles=ROLES,
    build=build,
    sample=sample,
    intro=(
        "Bir Excel (.xlsx) ya da CSV dosyası yükleyin. Sonuç, tedavi ve tedaviden önce belirlenmiş kontroller seçilir; "
        "gözlemler kümeler (ör. okul, firma) içinde bağımlıysa küme sütunu da seçilir. Seçilen sütunlarda boş hücresi olan "
        f"satırlar analizden çıkarılır. Hesap 11 bölme ve iç çapraz doğrulama içerdiği için en çok {sayim(MAX_ROWS)} gözlem "
        "kullanılır; birkaç saniye sürebilir."
    ),
    min_rows=MIN_ROWS,
    extra_columns=True,
    extra_label="Kontroller (tedaviden önce belirlenmiş)",
    extra_help="Yardımcı modellerin değişkenleri (notlarda başlangıç puanı, cinsiyet, yaş, okul yönetimi, ek öğretmen, "
               "yüzdelik); en az 2, en çok 8. Tedaviden etkilenen değişkenleri seçmeyin.",
    max_extra=8,
    extra_required=True,
    validate=validate,
    suggest=suggest,
)

VARIANTS = TopicVariants(alternative=alternative, story=STORY, custom=CUSTOM)
