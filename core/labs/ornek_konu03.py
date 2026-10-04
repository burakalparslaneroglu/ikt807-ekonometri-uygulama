"""Konu 3 genel uygulaması: atanmış bir tedaviyi seçilmiş alt gruptan ayırmak.

Notlardaki §3.15'in beş adımı aynı numaralarla ve aynı işlemlerle, verisi değiştirilebilir biçimde yazılır (``build``):
denge tablosu, ham ve kovaryat ayarlı fark, seçilmiş grubun başlangıç farkı ve tanımlama soruları. Alternatif örnek
Di Tella ve Schargrodsky (2004) verisidir (Hansen'in arşivindeki ``DS2004.dta``): 1994 Buenos Aires saldırısından sonra
Yahudi kurumu bulunan bloklara verilen polis koruması. Atama rastgele değildir; metinler koşullu bağımsızlık
varsayımını açıkça yazar ve fark-farkı (DiD) gibi notlarda işlenmeyen bir tasarıma geçmez. "Kendi verini yükle"
seçeneğinde aynı adımlar öğrencinin dosyasıyla kurulur. Notlardaki laboratuvar (``core.labs.konu03``) değişmez.

Rollerin notlardaki karşılıkları: sonuç toplam test puanı, tedavi tracking, atama birimi okul (kümelenmiş standart
hata), başlangıç değişkenleri denge tablosu ve kovaryat ayarlı model, seçilmiş grup düşük akış (lowstream) ve dayandığı
başlangıç standart puanı.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd

from core.labs import expr as E
from core.labs import kendi_veri as K
from core.labs.ornek import (
    EXACT_FIT_NOTE,
    SCALE_MESSAGE,
    Case,
    CustomLab,
    Role,
    TopicVariants,
    exact_fit,
    full_rank,
    katsayi,
    liste,
    md,
    sayi,
    sayim,
    stable_checks,
    stable_design,
    with_app_values,
    with_expected,
)
from core.labs.ornek_veri import deney_verisi
from core.labs.spec import (
    OLS,
    Check,
    CoefTarget,
    Derive,
    EffectTable,
    GroupMean,
    Indicator,
    KeepIf,
    LabSpec,
    LabStep,
    LoadHansen,
    ModelTarget,
    NoteRef,
    Operation,
    ReproClass,
    StatTarget,
)

TOPIC = "konu03"
SECTION = "3.15"
SONUC, TEDAVI, ATAMA, SECILMIS, SECIM = "sonuc", "tedavi", "atama", "secilmis", "secim"
T01, S01 = "tedavi01", "secilmis01"
"""Kendi verinde türetilen göstergeler (``kendi_veri.RESERVED_CODES``: öğrencinin sütunlarına verilmez)."""
MIN_CLUSTERS = 4


@dataclass(frozen=True)
class Plan:
    """Rollerin koddaki sütunları, tablo başlıkları ve kontrol etiketlerindeki adlar.

    ``balance``: denge tablosunun (değişken, tablodaki ad) satırları; ``controls``: kovaryat ayarlı modelin ek
    değişkenleri. ``selected``/``selected_base``: seçilmiş grup göstergesi ve grubun dayandığı başlangıç değişkeni;
    ``selected_where``: Adım 4'ün alt örneklemi (notlarda yalnız tracking okulları). ``reversion``: Adım 4'te sonucun
    da aynı göstergeyle karşılaştırıldığı alt örneklem (alternatif örnek: korunmayan bloklarda ortalamaya dönüş).
    """

    frame: str
    outcome: str
    treat: str
    cluster: str | None
    balance: tuple[tuple[str, str], ...]
    controls: tuple[str, ...]
    selected: str | None
    selected_base: str | None
    selected_where: tuple[str, float] | None
    reversion: tuple[str, float] | None
    prepare: tuple[Operation, ...]
    balance_prepare: tuple[Operation, ...]
    selected_prepare: tuple[Operation, ...]
    labels: dict
    titles: dict
    treated_label: str

    @property
    def vcov(self) -> str:
        return "cluster" if self.cluster else "HC1"

    @property
    def se_word(self) -> str:
        """Kontrol etiketlerinde standart hatanın adı (notlardaki gibi "küme SH")."""

        return "küme SH" if self.cluster else "HC1 SH"

    @property
    def se_label(self) -> str:
        """Tabloların ve tek başına duran kontrol etiketinin başlığı (notlardaki gibi "Küme SH")."""

        return "Küme SH" if self.cluster else "HC1 SH"


def _ols(plan: Plan, name: str, outcome: str, regressors: tuple[str, ...],
         where: tuple[str, float] | None = None) -> OLS:
    return OLS(name, plan.frame, outcome, regressors, vcov=plan.vcov, cluster=plan.cluster, where=where)


def _coef(model: str, term: str, label: str, quantity: str = "coef", decimals: int = 4) -> Check:
    return Check(label, CoefTarget(model, term, quantity), 0.0, decimals)


def _steps(plan: Plan, texts: dict, exact: bool) -> tuple[LabStep, ...]:
    se_label = plan.se_label
    step1_checks = (
        Check("Analiz örneklemi (N)", StatTarget(plan.frame, plan.outcome, "count"), 0.0, decimals=0),
        Check(plan.treated_label, StatTarget(plan.frame, plan.treat, "sum"), 0.0, decimals=0),
    )

    step2_ops: list[Operation] = []
    step2_checks: list[Check] = []
    if plan.balance:
        step2_ops += list(plan.balance_prepare)
        step2_ops += [_ols(plan, f"d_{variable}", variable, (plan.treat,)) for variable, _ in plan.balance]
        step2_ops.append(EffectTable(tuple((label, f"d_{variable}", plan.treat) for variable, label in plan.balance),
                                     "denge_tablosu", title=plan.titles["denge"], se_label=se_label))
        for variable, label in plan.balance:
            step2_checks += [
                _coef(f"d_{variable}", plan.treat, f"{label}: fark"),
                _coef(f"d_{variable}", plan.treat, f"{label}: {plan.se_word}", "se"),
                _coef(f"d_{variable}", plan.treat, f"{label}: p-değeri", "p", 3),
            ]

    step3_ops: list[Operation] = [_ols(plan, "ham_fark", plan.outcome, (plan.treat,))]
    rows = [("Ham fark", "ham_fark", plan.treat)]
    step3_checks = [
        _coef("ham_fark", plan.treat, "Ham fark: τ̂"),
        _coef("ham_fark", plan.treat, f"Ham fark: {plan.se_word}", "se"),
        _coef("ham_fark", plan.treat, "Ham fark: p-değeri", "p", 3),
        Check("Ham fark: n", ModelTarget("ham_fark", "nobs"), 0.0, decimals=0),
    ]
    if plan.controls:
        step3_ops.append(_ols(plan, "ayarli", plan.outcome, (plan.treat, *plan.controls)))
        rows.append(("Kovaryat ayarlı", "ayarli", plan.treat))
        step3_checks += [
            _coef("ayarli", plan.treat, "Kovaryat ayarlı: τ̂"),
            _coef("ayarli", plan.treat, f"Kovaryat ayarlı: {plan.se_word}", "se"),
            _coef("ayarli", plan.treat, "Kovaryat ayarlı: p-değeri", "p", 3),
            Check("Kovaryat ayarlı: n", ModelTarget("ayarli", "nobs"), 0.0, decimals=0),
        ]
    step3_ops.append(EffectTable(tuple(rows), "etki_tablosu", title=plan.titles["etki"], se_label=se_label))

    step4_ops: list[Operation] = []
    step4_checks: list[Check] = []
    if plan.selected and plan.selected_base:
        step4_ops += list(plan.selected_prepare)
        step4_ops.append(_ols(plan, "secilmis", plan.selected_base, (plan.selected,), plan.selected_where))
        rows4 = [(plan.titles["secilmis_satir"], "secilmis", plan.selected)]
        step4_checks += [
            _coef("secilmis", plan.selected, "δ̂ (seçilmiş grup)"),
            _coef("secilmis", plan.selected, plan.se_label, "se"),
        ]
        if plan.reversion is not None:
            step4_ops.append(_ols(plan, "donus", plan.outcome, (plan.selected,), plan.reversion))
            rows4.append((plan.titles["donus_satir"], "donus", plan.selected))
            step4_checks += [
                _coef("donus", plan.selected, "Sonuç farkı (seçilmiş grup)"),
                _coef("donus", plan.selected, f"Sonuç farkı: {plan.se_word}", "se"),
            ]
        step4_ops.append(EffectTable(tuple(rows4), "secilmis_tablosu", title=plan.titles["secilmis"],
                                     se_label=se_label))

    return (
        LabStep(number=1, title="Araştırma sorusu ve atama birimi", note=NoteRef(SECTION, 1),
                explanation=texts[1][0], operations=plan.prepare, checks=step1_checks, takeaway=texts[1][1],
                code_note=texts.get("kod1", ""), note_for=texts.get((1, "not"))),
        LabStep(number=2, title="Denge tablosunu mekanik bir sınav gibi okumamak",
                note=NoteRef(SECTION, 2, ("Tablo 3.1",)), explanation=texts[2][0], operations=tuple(step2_ops),
                checks=stable_checks(step2_checks, exact), reproducibility=ReproClass.CONVENTION,
                takeaway=texts[2][1], code_note=texts["p"] if step2_ops else "", note_for=texts.get((2, "not"))),
        LabStep(number=3, title="Ham fark ile kovaryat ayarlı tahmini karşılaştırmak",
                note=NoteRef(SECTION, 3, ("Tablo 3.2",)), explanation=texts[3][0], operations=tuple(step3_ops),
                checks=stable_checks(step3_checks, exact), reproducibility=ReproClass.CONVENTION,
                takeaway=texts[3][1], code_note=texts["p"], note_for=texts.get((3, "not"))),
        LabStep(number=4, title="Aynı veride seçilmiş bir grubun neden nedensel karşılaştırma olmadığını görmek",
                note=NoteRef(SECTION, 4), explanation=texts[4][0], operations=tuple(step4_ops),
                checks=tuple(step4_checks), takeaway=texts[4][1], code_note=texts.get("kod4", ""),
                note_for=texts.get((4, "not"))),
        LabStep(number=5, title="Makale çıktısını okurken sorulacak tanımlama soruları", note=NoteRef("3.15.5", 0),
                explanation=QUESTIONS, takeaway=texts[5][1], note_for=texts.get((5, "not"))),
    )


def _spec(plan: Plan, texts: dict, *, source: str, title: str, dataset: str, exact: bool = False) -> LabSpec:
    return LabSpec(topic_key=TOPIC, title=title, dataset=dataset, note_section=SECTION,
                   steps=_steps(plan, texts, exact), labels=tuple(plan.labels.items()), source=source)


# --- Ortak metinler -------------------------------------------------------------------------------------------

QUESTIONS = (
    "1. Tedavi nasıl atanmıştır: rastgele mi, politika kuralıyla mı, bireysel seçimle mi?\n"
    "2. Atama birimi ile gözlem birimi aynı mıdır?\n"
    "3. Standart hatalar hangi düzeyde kümelenmiştir?\n"
    "4. Kovaryatlar tedaviden önce mi ölçülmüştür?\n"
    "5. Kovaryat eklenince örneklem değişiyor mu?\n"
    "6. Sonuç \"program etkisi\" mi, yoksa seçilmiş gruplar arası koşullu fark mı?\n"
    "7. Yalnız p-değerinin eşik değiştirmesine bakılarak ekonomik sonuç abartılıyor mu?"
)
P_VALUE_NOTE = (
    "p-değerleri notlardaki gibi normal yaklaşımla hesaplanır: $2\\Phi(-|t|)$. Stata `estimates table` küme-dayanıklı "
    "SH için t(G−1) dağılımını (G = küme sayısı), R `coeftest` ise t(n−k) dağılımını kullanır; bu yüzden yazılım "
    "tablolarındaki p-değerleri üçüncü basamakta farklı görünebilir. Kontrol satırları normal yaklaşımı kullanır. Küme "
    "düzeltmesi $G/(G-1)\\cdot(N-1)/(N-K)$ dillerde aynıdır; standart hatalar birebir tutar."
)
P_VALUE_NOTE_OWN = (
    "p-değerleri notlardaki gibi normal yaklaşımla hesaplanır: $2\\Phi(-|t|)$. R `coeftest` t(n−k) dağılımını kullanır; "
    "bu yüzden yazılım tablolarındaki p-değerleri üçüncü basamakta farklı görünebilir. Küme düzeltmesi "
    "$G/(G-1)\\cdot(N-1)/(N-K)$ iki dilde aynıdır; standart hatalar birebir tutar."
)
P_VALUE_NOTE_HC1 = (
    "p-değerleri notlardaki gibi normal yaklaşımla hesaplanır: $2\\Phi(-|t|)$. R `coeftest` t(n−k) dağılımını kullanır; "
    "bu yüzden yazılım tablolarındaki p-değerleri üçüncü basamakta farklı görünebilir. HC1 standart hataları iki dilde "
    "birebir tutar."
)
BALANCE_RULE = (
    "Denge tablosu bir geçerlik sınavı değildir: örneklem kompozisyonunu anlamak ve önceden belirlenmiş kovaryat "
    "ayarlamasının yararını değerlendirmek için kullanılır."
)
READING_RULE = (
    "Tanımlama, regresyon komutundan önce gelir. Aynı OLS komutu, atama mekanizması savunulabilen bir tedavi katsayısı "
    "için nedensel bir estimand'ı tahmin edebilirken, seçilmiş bir grup göstergesi için yalnız koşullu ilişkiyi "
    "özetleyebilir. Yöntemin adı değil, değişkenin veri üretim sürecindeki rolü nedensel yorumu belirler."
)


# --- Alternatif örnek: Di Tella ve Schargrodsky (2004) -------------------------------------------------------

ALT_DATA = "ds2004"
ALT_FRAME = "ds"
ALT_TITLE = "Saldırı Sonrası Polis Korumasıyla Atanmış Tedaviyi Seçilmiş Gruptan Ayırmak"
ALT_BALANCE = (
    ("onceki", "Saldırı öncesi aylık hırsızlık"),
    ("public", "Kamu binası"),
    ("gasstation", "Benzin istasyonu"),
    ("bank", "Banka"),
    ("once", "Once mahallesi"),
    ("crespo", "Villa Crespo mahallesi"),
)


def alternative_plan() -> Plan:
    frame = ALT_FRAME
    labels = {
        E.INTERCEPT: "Sabit",
        "thefts": "Aylık araç hırsızlığı",
        "sameblock": "Korunan blok (blokta Yahudi kurumu)",
        "block": "Blok",
        "month": "Ay",
        "barrio": "Mahalle",
        "onceki": "Saldırı öncesi aylık hırsızlık",
        "public": "Kamu binası",
        "gasstation": "Benzin istasyonu",
        "bank": "Banka",
        "once": "Once mahallesi",
        "crespo": "Villa Crespo mahallesi",
        "yuksek": "Saldırı öncesi hırsızlık kaydı olan blok",
        "ham_fark": "Ham fark",
        "ayarli": "Kovaryat ayarlı",
        "secilmis": "Seçilmiş grup",
        "donus": "Ortalamaya dönüş",
    }
    return Plan(
        frame=frame, outcome="thefts", treat="sameblock", cluster="block", balance=ALT_BALANCE,
        controls=tuple(variable for variable, _ in ALT_BALANCE), selected="yuksek", selected_base="onceki",
        selected_where=None, reversion=("sameblock", 0.0),
        prepare=(
            LoadHansen(ALT_DATA, "DS2004.dta", frame),
            GroupMean(frame, "onceki", "thefts", "block",
                      "Her bloğun saldırı öncesi (Nisan–Haziran) aylık ortalama araç hırsızlığı", ("month", "<=", 6)),
            KeepIf(frame, (("month", ">=", 8),), "Analiz örneklemi: saldırı sonrası tam aylar (Ağustos–Aralık)"),
        ),
        balance_prepare=(
            Indicator(frame, "once", "barrio", "Once", "Once mahallesi göstergesi (Belgrano referans)"),
            Indicator(frame, "crespo", "barrio", "V. Crespo", "Villa Crespo mahallesi göstergesi (Belgrano referans)"),
        ),
        selected_prepare=(
            Derive(frame, "yuksek", E.compare("gt", E.var("onceki"), 0),
                   "Seçilmiş grup: saldırı öncesi hırsızlık kaydı sıfırdan büyük olan blok"),
        ),
        labels=labels,
        titles={
            "denge": "Korunan ve korunmayan bloklar arasında saldırı öncesi farklar",
            "etki": "Polis koruması için iki temel tahmin",
            "secilmis": "Seçilmiş grup: saldırı öncesi hırsızlık kaydı olan bloklar",
            "secilmis_satir": "Saldırı öncesi aylık hırsızlık (seçimi tanımlayan değişken)",
            "donus_satir": "Saldırı sonrası aylık hırsızlık, korunmayan bloklar",
        },
        treated_label="Korunan blok-ay gözlemi",
    )


ALT_TEXTS = {
    1: (
        "Soru: polis varlığı araç hırsızlığını azaltır mı? 18 Temmuz 1994'te Buenos Aires'teki AMIA saldırısından sonra "
        "federal polis bütün Yahudi kurumlarının önüne gece gündüz koruma yerleştirdi. Veri Di Tella ve Schargrodsky "
        "(2004)'ün üç mahallesindeki (Belgrano, Once, Villa Crespo) 876 bloktur; her blok için Nisan–Aralık 1994 aylık "
        "araç hırsızlığı sayısı vardır. Tedavi göstergesi `sameblock`: blokta Yahudi kurumu var, yani blok korunuyor "
        "(37 blok). Müdahale **blok düzeyinde** atanmıştır ve her blok her ay gözlenir: aynı bloğun aylarındaki hata "
        "terimleri bağımsız kabul edilmemelidir; standart hatalar blok düzeyinde kümelenir. Analiz örneklemi "
        "saldırıdan sonraki tam aylardır (Ağustos–Aralık): 876 × 5 = 4.380 blok-ay, bunların 185'i korunan bloklarda.",
        "Estimand, korunan bloklarda polis korumasının aylık araç hırsızlığına ortalama etkisidir. Atama **rastgele "
        "değildir**: koruma, yeri saldırıdan önce belirlenmiş kurumlara verildi. Nedensel yorum, gözlenen blok "
        "özellikleri sabitken kurumların konumunun, koruma olmasaydı gözlenecek (potansiyel) saldırı sonrası "
        "hırsızlıkla ilişkisiz olduğu varsayımına (koşullu bağımsızlık) dayanır. Bu yüzden kovaryatlar burada yalnız "
        "hassasiyet için değil, bu varsayımı desteklemek için de kullanılır.",
    ),
    "kod1": (
        "Saldırı öncesi ortalama Nisan–Haziran aylarından alınır; Temmuz verisi yalnız 1–17 günlerini kapsadığı için "
        "dışarıda kalır. Ortalama bütün aylar okunduktan sonra her blok için hesaplanır, ardından yalnız saldırı sonrası "
        "aylar tutulur. Fark-farkı (DiD) tasarımı notlarda işlenmediği için bu uygulama saldırı sonrası aylarda "
        "karşılaştırma yapar."
    ),
    2: (
        "Her saldırı öncesi değişken için $x_b=\\alpha+\\delta\\,sameblock_b+e_b$ regresyonu blok düzeyinde kümelenmiş "
        "standart hatayla tahmin edilir; $\\hat\\delta$ korunan ve korunmayan bloklar arasındaki farktır. Değişkenler: "
        "saldırı öncesi aylık ortalama hırsızlık, kamu binası, benzin istasyonu ve banka göstergeleri ile iki mahalle "
        "göstergesi (Belgrano referans). Analiz örnekleminde her blok beş kez yer aldığı için standart hata blok "
        "düzeyinde kümelenir.",
        "Saldırı öncesi hırsızlıkta ve blok özelliklerinde iki grup arasında belirgin fark yok (p-değerleri 0,50–0,80). "
        "Mahalle ise belirgin biçimde farklı: korunan blokların Once'de olma olasılığı 0,30 daha yüksek (p < 0,001). "
        "Rastgele atanmış bir deneyde böyle bir fark tesadüftür; burada tasarımdan gelir, çünkü Yahudi kurumları "
        "Once'de yoğunlaşır. " + BALANCE_RULE + " Kovaryat ayarlı model mahalle göstergelerini bu yüzden içerir: "
        "ayarlama burada hassasiyetin yanında koşullu bağımsızlık varsayımını desteklemek için de yapılır.",
    ),
    3: (
        "**Ham fark:** $thefts_{bt}=\\alpha+\\tau\\,sameblock_b+e_{bt}$, saldırı sonrası aylar. **Kovaryat ayarlı** "
        "model bloğun saldırı öncesi ortalamasını, kamu binası, benzin istasyonu ve banka göstergelerini ve iki mahalle "
        "göstergesini ekler. İki modelde de standart hatalar blok düzeyinde kümelenir; değişkenlerde eksik değer "
        "olmadığı için iki sütunun örneklemi aynıdır (4.380 blok-ay).",
        "Ham fark −0,0696 (küme SH 0,0115): korunan bloklarda ayda ortalama 0,035, korunmayanlarda 0,105 araç "
        "çalınmış. Kovaryat ayarlı tahmin −0,0569 (0,0124): korunmayan blokların ortalamasının yarısından fazlası. "
        "Kontroller eklenince fark yaklaşık beşte bir küçülür: korunan bloklar hırsızlığın daha az olduğu Once'de "
        "yoğunlaştığı için ham fark mahalle farkını da içerir. Rastgele atamada kovaryat ayarlaması nokta tahminini "
        "sistematik biçimde değiştirmez; burada değiştirmesi atamanın rastgele olmadığının işaretidir. İki p-değeri de "
        "0,001'in altındadır; ekonomik büyüklük yorumu p-değerinden değil, tahminin kendisinden gelir.",
    ),
    4: (
        "Saldırı öncesi hırsızlık kaydı sıfırdan büyük olan bloklar (blokların %47,8'i) seçilmiş bir gruptur: grup, "
        "sonuç değişkeninin kendi geçmişine göre mekanik olarak oluşur. Kayıtlar kesirlidir: kavşakta bildirilen bir "
        "hırsızlık, kavşaktaki dört bloğun her birine 0,25 olarak yazılır (Di Tella ve Schargrodsky, 2004). Bunu "
        "görmek için saldırı öncesi ortalamayı bu göstergeye regres ederiz; ayrıca korunmayan bloklarda saldırı sonrası "
        "hırsızlığı aynı göstergeyle karşılaştırırız.",
        "Seçilmiş blokların saldırı öncesi ortalaması 0,2005 daha yüksek; saldırıdan sonra korunmayan bloklarda fark "
        "0,0603'e iner. Bu blokların aylık hırsızlığı 0,200'den 0,136'ya düşmüş, ötekilerinki 0'dan 0,076'ya "
        "çıkmıştır. Bu bir polis etkisi değildir (karşılaştırılan blokların hiçbiri korunmuyor): sonucun geçmiş "
        "değerine göre seçilen gruplar ortalamaya döner. Aynı veride `sameblock` için koşullu bağımsızlık argümanı "
        "kurulabilir; geçmiş hırsızlığa göre seçilen grup için kurulamaz.",
    ),
    "kod4": (
        "Kodda kritik satır yalnız regresyon formülü değildir: küme değişkeni (`block`) atama biriminin blok olduğunu "
        "çıkarım aşamasına taşır. Tedavi okul, firma, köy veya blok düzeyinde atanmışsa standart hata stratejisi de bu "
        "tasarımla uyumlu olmalıdır."
    ),
    5: ("", READING_RULE + " Bu veride `sameblock` rastgele atanmamıştır: koruma, dışsal bir olayın (saldırı) ardından "
            "kurumların önceden belirlenmiş konumuna göre verilmiştir. Nedensel yorum, bu atama mekanizmasına ilişkin "
            "açıkça yazılmış varsayıma dayanır."),
    "p": P_VALUE_NOTE.replace("(G = küme sayısı)", "(G = 876 blok)"),
}

ALT_EXPECTED = {
    (1, "Analiz örneklemi (N)"): 4380,
    (1, "Korunan blok-ay gözlemi"): 185,
    (2, "Saldırı öncesi aylık hırsızlık: fark"): 0.0175,
    (2, "Saldırı öncesi aylık hırsızlık: küme SH"): 0.0323,
    (2, "Saldırı öncesi aylık hırsızlık: p-değeri"): 0.589,
    (2, "Kamu binası: fark"): 0.0231,
    (2, "Kamu binası: küme SH"): 0.0377,
    (2, "Kamu binası: p-değeri"): 0.540,
    (2, "Benzin istasyonu: fark"): 0.0068,
    (2, "Benzin istasyonu: küme SH"): 0.0271,
    (2, "Benzin istasyonu: p-değeri"): 0.803,
    (2, "Banka: fark"): -0.0258,
    (2, "Banka: küme SH"): 0.0384,
    (2, "Banka: p-değeri"): 0.501,
    (2, "Once mahallesi: fark"): 0.2974,
    (2, "Once mahallesi: küme SH"): 0.0830,
    (2, "Once mahallesi: p-değeri"): 0.000,
    (2, "Villa Crespo mahallesi: fark"): 0.0570,
    (2, "Villa Crespo mahallesi: küme SH"): 0.0801,
    (2, "Villa Crespo mahallesi: p-değeri"): 0.477,
    (3, "Ham fark: τ̂"): -0.0696,
    (3, "Ham fark: küme SH"): 0.0115,
    (3, "Ham fark: p-değeri"): 0.000,
    (3, "Ham fark: n"): 4380,
    (3, "Kovaryat ayarlı: τ̂"): -0.0569,
    (3, "Kovaryat ayarlı: küme SH"): 0.0124,
    (3, "Kovaryat ayarlı: p-değeri"): 0.000,
    (3, "Kovaryat ayarlı: n"): 4380,
    (4, "δ̂ (seçilmiş grup)"): 0.2005,
    (4, "Küme SH"): 0.0078,
    (4, "Sonuç farkı (seçilmiş grup)"): 0.0603,
    (4, "Sonuç farkı: küme SH"): 0.0094,
}
"""Kontrollerin DS2004 tam verisindeki değerleri, gösterim basamağında: (adım, etiket) → değer. Metinlerdeki sayılar
bunlardır; testler bağımsız bir hesapla ve üç dilde doğrular."""


@cache
def alternative() -> LabSpec:
    """Alternatif örneğin tanımı; beklenen değerler DS2004'ün tam verisindeki sayılardır (testle doğrulanır)."""

    spec = _spec(alternative_plan(), ALT_TEXTS, source="alternatif", title=ALT_TITLE, dataset=ALT_DATA)
    return with_expected(spec, ALT_EXPECTED)


STORY = (
    "Alternatif örnek Di Tella ve Schargrodsky (2004) verisidir: 1994 Buenos Aires saldırısından sonra Yahudi kurumu "
    "bulunan 37 bloğa polis koruması verildi; üç mahallede 876 blok, saldırı sonrası aylarda 4.380 blok-ay (Hansen'in "
    "arşivindeki DS2004.dta). Adımlar notlardaki gibidir: denge tablosu, ham ve kovaryat ayarlı fark, seçilmiş grup. "
    "Fark şu: atama rastgele değildir; koruma, dışsal bir olayın ardından kurumların konumuna göre verildi."
)


# --- Kendi verin --------------------------------------------------------------------------------------------

def _indicators(case: Case) -> pd.DataFrame:
    data = case.data.copy()
    treat = case.roles[TEDAVI]
    data[T01] = np.where(data[treat].isna(), np.nan, (data[treat] == case.levels[TEDAVI]).astype(float))
    if case.has(SECILMIS):
        selected = case.roles[SECILMIS]
        data[S01] = np.where(data[selected].isna(), np.nan, (data[selected] == case.levels[SECILMIS]).astype(float))
    return data


def _selected_where(case: Case, data: pd.DataFrame) -> tuple[str, float] | None:
    """Adım 4'ün alt örneklemi: seçilmiş grup tedavi grubunda iki değer de alıyorsa notlardaki gibi yalnız tedavi
    grubu; almıyorsa bütün örneklem."""

    if not (case.has(SECILMIS) and case.has(SECIM)):
        return None
    inside = data.loc[data[T01] == 1.0, S01].dropna()
    return (T01, 1.0) if inside.nunique() == 2 else None


def own_plan(case: Case) -> Plan:
    name = case.name
    y = case.roles[SONUC]
    treat = case.roles[TEDAVI]
    pick = case.levels[TEDAVI]
    cluster = case.roles.get(ATAMA)
    covariates = tuple(column for column in case.extras if column not in (y, treat))
    data = _indicators(case)
    selected = case.has(SECILMIS) and case.has(SECIM)
    prepare: list[Operation] = list(case.load)
    prepare.append(Indicator(case.frame, T01, treat, pick, "Tedavi göstergesi: seçilen kategori 1, diğeri 0"))
    selected_prepare: tuple[Operation, ...] = ()
    if selected:
        selected_prepare = (Indicator(case.frame, S01, case.roles[SECILMIS], case.levels[SECILMIS],
                                      "Seçilmiş grup göstergesi: seçilen kategori 1, diğeri 0"),)
    labels = {column: name(column) for column in (y, treat, *covariates)}
    if cluster:
        labels[cluster] = name(cluster)
    labels.update({E.INTERCEPT: "Sabit", T01: f"{name(treat)} = {pick}", "ham_fark": "Ham fark",
                   "ayarli": "Kovaryat ayarlı", "secilmis": "Seçilmiş grup"})
    base = None
    if selected:
        base = case.roles[SECIM]
        labels[S01] = f"{name(case.roles[SECILMIS])} = {case.levels[SECILMIS]}"
        labels.setdefault(base, name(base))
    return Plan(
        frame=case.frame, outcome=y, treat=T01, cluster=cluster,
        balance=tuple((column, name(column)) for column in covariates), controls=covariates,
        selected=S01 if selected else None, selected_base=base,
        selected_where=_selected_where(case, data), reversion=None,
        prepare=tuple(prepare), balance_prepare=(), selected_prepare=selected_prepare, labels=labels,
        titles={
            "denge": "Tedavi ve karşılaştırma grupları arasında başlangıç değişkeni farkları",
            "etki": "Tedavi etkisi için iki temel tahmin",
            "secilmis": "Seçilmiş grubun başlangıç farkı",
            "secilmis_satir": f"Seçilmiş grup: {name(case.roles[SECILMIS])} = {case.levels[SECILMIS]}"
            if selected else "",
        },
        treated_label="Tedavi grubundaki gözlem",
    )


def _own_texts(case: Case, plan: Plan, exact: bool) -> dict:
    y, treat = (case.md(role) for role in (SONUC, TEDAVI))
    pick = md(case.levels[TEDAVI])
    other = md(next(item for item in case.orders[case.roles[TEDAVI]] if item != case.levels[TEDAVI]))
    cluster = case.md(ATAMA) if case.has(ATAMA) else None
    covariates = [f"“{md(case.name(column))}”" for column in plan.controls]
    exact_note = (" " + EXACT_FIT_NOTE) if exact else ""
    se_word = plan.se_word

    def step1(state) -> str:
        frame = state.frames[plan.frame]
        n, treated = len(frame), int(frame[T01].sum())
        text = (f"Analiz örneklemi {sayim(n)} gözlem: tedavi grubunda (“{treat}” = “{pick}”) {sayim(treated)}, "
                f"karşılaştırma grubunda (“{other}”) {sayim(n - treated)} gözlem.")
        if cluster:
            groups = int(frame[case.roles[ATAMA]].nunique())
            text += (f" Atama birimi “{cluster}”: {sayim(groups)} küme. Standart hatalar bu düzeyde kümelenir; aynı "
                     "kümedeki gözlemlerin hata terimleri bağımsız kabul edilmez.")
            if groups < 30:
                text += (f" Küme sayısı az ({groups}); kümelenmiş standart hata az kümede kendisi de belirsizdir, "
                         "p-değerleri temkinli okunmalıdır.")
        else:
            text += (" Atama birimi seçilmedi: tedavinin gözlem düzeyinde atandığı varsayılır ve HC1 standart hatası "
                     "kullanılır. Tedavi okul, köy ya da firma gibi bir birim düzeyinde atandıysa o birimi seçin.")
        return text

    def step2(state) -> str:
        if not plan.balance:
            return ("Denge tablosu için tedaviden önce ölçülmüş en az bir başlangıç değişkeni seçin (ör. başlangıç "
                    "puanı, yaş).")
        table = state.tables["denge_tablosu"]
        flagged = [f"“{md(label)}”" for label, row in table.iterrows() if row["p"] < 0.10]
        text = f"Denge tablosunda {len(table)} değişken var; "
        if flagged:
            text += (f"p-değeri 0,10'un altında olanlar: {liste(flagged)}. Rastgele atamada bazı farkların yalnız "
                     "örnekleme değişkenliğiyle belirgin görünmesi olağandır; atama rastgele değilse fark tasarımdan "
                     "gelebilir ve o değişken karşılaştırmada tutulmalıdır.")
        else:
            text += "hiçbirinde p-değeri 0,10'un altında değil."
        return text + " " + BALANCE_RULE + exact_note

    def step3(state) -> str:
        table = state.tables["etki_tablosu"]
        raw = table.loc["Ham fark"]
        text = (f"Ham fark {katsayi(raw['tahmin'])} ({se_word} {katsayi(raw['sh'])}, p = {sayi(raw['p'], 3)}, "
                f"n = {sayim(raw['n'])}).")
        if "Kovaryat ayarlı" in table.index:
            adjusted = table.loc["Kovaryat ayarlı"]
            text += (f" Kovaryat ayarlı tahmin {katsayi(adjusted['tahmin'])} ({se_word} "
                     f"{katsayi(adjusted['sh'])}, p = {sayi(adjusted['p'], 3)}, n = {sayim(adjusted['n'])}).")
            if int(adjusted["n"]) != int(raw["n"]):
                text += (f" Kovaryat ayarlı modelde yalnız bütün başlangıç değişkenleri gözlenen satırlar kalır "
                         f"({sayim(raw['n'])} → {sayim(adjusted['n'])}): sütunlar arası fark yalnız \"kontroller "
                         "eklendi\" diye yorumlanmamalıdır.")
            text += (" Rastgele atamada iki nokta tahmini birbirine yakın olmalıdır; kovaryatlar tanımlamayı değil "
                     "hassasiyeti artırır. Büyük bir fark, atamanın rastgele olmadığının ya da örneklemin değiştiğinin "
                     "işareti olabilir.")
        else:
            text += " Kovaryat ayarlı model için başlangıç değişkeni seçin."
        return text + " \"Anlamlı/anlamsız\" ikili dili yerine tahminin büyüklüğü yorumlanır." + exact_note

    def step4(state) -> str:
        if plan.selected is None:
            return ("Bu adım için “Seçilmiş grup göstergesi” ve grubun dayandığı başlangıç değişkenini seçin (ör. "
                    "başlangıç puanına göre oluşturulmuş düşük başarı grubu ve başlangıç puanı).")
        result = state.models["secilmis"]
        delta, se = float(result.params[S01]), float(result.bse[S01])
        group = md(case.name(case.roles[SECILMIS]))
        level = md(case.levels[SECILMIS])
        base = md(case.name(case.roles[SECIM]))
        where = ("yalnız tedavi grubunda, notlardaki gibi" if plan.selected_where is not None
                 else "bütün örneklemde (gösterge tedavi grubunda iki kategoriyi birden almadığı için)")
        direction = "yüksek" if delta > 0 else "düşük"
        size = (f"ortalama {katsayi(abs(delta))} daha {direction}" if delta != 0 else "ortalamada aynı")
        return (f"Regresyon {where} tahmin edildi: “{group}” = “{level}” olan gözlemlerin “{base}” değeri {size} "
                f"({se_word} {katsayi(se)}). Grup başlangıç değerine göre oluştuysa bu fark seçimin kendisidir: bu "
                "grupları sonuç değişkeninde doğrudan karşılaştırmak grubun nedensel etkisini vermez.")

    return {
        1: (
            f"Soru: “{treat}” (“{pick}” = 1) sonuç değişkeni “{y}” üzerinde ortalama ne kadar fark yaratır? Önce "
            "tedavinin nasıl ve hangi birim düzeyinde atandığını yazarız: atama birimi çıkarımın (standart hatanın) "
            "düzeyini belirler. Analiz örneklemi sonucu ve tedavi durumu gözlenen satırlardır.",
            "Estimand tedavinin ortalama etkisidir. Rastgele atamada tedavi göstergesi ile potansiyel sonuçlar "
            "arasındaki bağımsızlık tasarımla desteklenir; atama rastgele değilse nedensel yorum için atama "
            "mekanizmasına ilişkin varsayım açıkça yazılmalıdır. Regresyon kontrolleri tanımlamayı kendileri "
            "yaratmaz.",
        ),
        (1, "not"): step1,
        2: (
            "Her başlangıç değişkeni için $x_i=\\alpha+\\delta\\,T_i+e_i$ regresyonu tahmin edilir ($T_i$ tedavi "
            f"göstergesi); $\\hat\\delta$ iki grup arasındaki farktır. Seçilen değişkenler: "
            f"{liste(covariates) if covariates else 'yok'}. Her regresyon o değişkenin gözlendiği satırlarla yapılır, "
            "bu yüzden N satırdan satıra değişebilir.",
            BALANCE_RULE,
        ),
        (2, "not"): step2,
        3: (
            f"**Ham fark:** “{y}” = α + τ·T + e. **Kovaryat ayarlı** model başlangıç değişkenlerini ekler; yalnız bütün "
            f"başlangıç değişkenleri gözlenen satırlar kalır. Standart hata: {se_word}.",
            "Bir p-değerinin 0,05'in iki yanında kalması farklı bir ekonomik hikâye değildir.",
        ),
        (3, "not"): step3,
        4: (
            "Seçilmiş grup, rastgele atanmamış ve çoğu zaman başlangıç değerine göre mekanik olarak oluşan bir "
            "gruptur (notlarda tracking okullarındaki düşük akış). Bunu görmek için grubun dayandığı başlangıç "
            "değişkenini seçilmiş grup göstergesine regres ederiz.",
            "Seçim değişkenin tanımına gömülüyse aynı OLS komutu nedensel bir etki değil, seçimin kendisini ölçer.",
        ),
        (4, "not"): step4,
        5: ("", READING_RULE),
        "p": P_VALUE_NOTE_OWN if case.has(ATAMA) else P_VALUE_NOTE_HC1,
        "kod1": "Tedavi göstergesi, seçtiğiniz kategoride 1, diğerinde 0'dır.",
    }


def build(case: Case) -> LabSpec:
    """Kendi verinle Konu 3 uygulaması; kontrollerin beklenen değerleri uygulamanın hesabıdır."""

    plan = own_plan(case)
    data = _indicators(case)
    exact = exact_fit(data, plan.outcome, (T01, *plan.controls))
    texts = _own_texts(case, plan, exact)
    spec = _spec(plan, texts, source="kendi", title=f"{case.label(TEDAVI)} ve {case.label(SONUC)}", dataset="",
                 exact=exact)
    return with_app_values(spec)


def _cluster_samples(case: Case, data: pd.DataFrame,
                     covariates: tuple[str, ...]) -> list[tuple[str, pd.DataFrame, str]]:
    """Kümelenmiş standart hatanın kullanıldığı tahmin örneklemleri (ham fark örneklemi bütün veridir) ve her biri
    için önerilen çözüm."""

    samples = [(f"“{case.name(column)}” denge satırı", data.dropna(subset=[column, T01]),
                f"“{case.name(column)}” değişkenini başlangıç değişkenlerinden çıkarın.") for column in covariates]
    if covariates:
        samples.append(("kovaryat ayarlı model", data.dropna(subset=[case.roles[SONUC], T01, *covariates]),
                        "Boş hücresi çok olan başlangıç değişkenlerini çıkarın."))
    if case.has(SECILMIS) and case.has(SECIM):
        sample = data if _selected_where(case, data) is None else data[data[T01] == 1.0]
        samples.append(("Adım 4 (seçilmiş grup)", sample.dropna(subset=[case.roles[SECIM], S01]),
                        "Seçilmiş grup göstergesi ile dayandığı değişkeni kaldırın; Adım 1–3 bu dosyayla çalışır."))
    return samples


def validate(case: Case) -> None:
    y, treat = case.roles[SONUC], case.roles[TEDAVI]
    if y == treat:
        raise K.UploadError("Sonuç ve tedavi için farklı sütunlar seçin.")
    if case.has(SECILMIS) and case.roles[SECILMIS] == treat:
        raise K.UploadError("Seçilmiş grup göstergesi için tedavi göstergesinden farklı bir sütun seçin.")
    if case.has(SECIM) and case.roles[SECIM] == y:
        raise K.UploadError("Seçimin dayandığı değişken tedaviden önce ölçülmüş olmalı; sonuç değişkenini seçmeyin.")
    if y in case.extras or treat in case.extras:
        raise K.UploadError("Başlangıç değişkenlerinde sonuç ya da tedavi sütunu olmamalı.")
    data = _indicators(case)
    if data[y].nunique() < 2:
        raise K.UploadError(f"“{case.name(y)}” sütununda en az iki farklı değer olmalı.")
    if data[T01].value_counts().min() < 2:
        raise K.UploadError(f"“{case.name(treat)}” sütununun iki kategorisinde de en az iki gözlem olmalı.")
    if case.has(ATAMA):
        cluster = case.roles[ATAMA]
        if cluster in (y, treat) or cluster in case.extras:
            raise K.UploadError("Atama birimi için sonuç, tedavi ya da başlangıç değişkeninden farklı bir sütun seçin.")
        groups = data[cluster].nunique()
        if groups < MIN_CLUSTERS:
            raise K.UploadError(f"“{case.name(cluster)}” sütununda {groups} küme var; kümelenmiş standart hata için en "
                                f"az {MIN_CLUSTERS} küme gerekir. Tedavi bu birim düzeyinde atandıysa daha çok küme "
                                "içeren bir veri gerekir; atama birimini kaldırmak (HC1) yalnız tedavi gözlem düzeyinde "
                                "atandıysa doğrudur.")
    covariates = tuple(column for column in case.extras if column not in (y, treat))
    for column in covariates:
        if data[column].dropna().nunique() < 2:
            raise K.UploadError(f"“{case.name(column)}” sütununda en az iki farklı değer olmalı.")
        if data.loc[data[column].notna(), T01].nunique() < 2:
            raise K.UploadError(f"“{case.name(column)}” yalnız bir grupta gözleniyor; denge tablosunda kullanılamaz.")
    complete = data.dropna(subset=[y, T01, *covariates])
    if covariates and len(complete) <= len(covariates) + 3:
        raise K.UploadError(f"Kovaryat ayarlı modelde {len(covariates) + 2} katsayı var; bütün başlangıç değişkenleri "
                            f"gözlenen {len(complete)} satır yetmez.")
    if not full_rank(data, (T01, *covariates)):
        raise K.UploadError("Tedavi ve başlangıç değişkenleri arasında tam doğrusal bağlantı var (ör. bir değişken "
                            "yalnız bir grupta değişiyor ya da diğerlerinin doğrusal birleşimi). Başlangıç "
                            "değişkenlerini değiştirin.")
    designs = [(T01,), (T01, *covariates)] if covariates else [(T01,)]
    if not stable_design(data, y, designs):
        raise K.UploadError(SCALE_MESSAGE)
    for column in covariates:
        if not stable_design(data, column, [(T01,)]):
            raise K.UploadError(SCALE_MESSAGE)
    if case.has(SECILMIS) and case.has(SECIM):
        base = case.roles[SECIM]
        where = _selected_where(case, data)
        sample = data if where is None else data[data[T01] == 1.0]
        sample = sample.dropna(subset=[base, S01])
        if sample[S01].nunique() < 2 or sample[S01].value_counts().min() < 2:
            raise K.UploadError(f"“{case.name(case.roles[SECILMIS])}” sütununun iki kategorisinde de en az iki "
                                f"gözlem (ve “{case.name(base)}” değeri) olmalı.")
        if not stable_design(sample, base, [(S01,)]):
            raise K.UploadError(SCALE_MESSAGE)
    if case.has(ATAMA):
        cluster = case.roles[ATAMA]
        for label, sample, remedy in _cluster_samples(case, data, covariates):
            groups = sample[cluster].nunique()
            if groups < MIN_CLUSTERS:
                raise K.UploadError(f"“{case.name(cluster)}” sütununda {label} örnekleminde {groups} küme var; "
                                    f"kümelenmiş standart hata için her tahmin örnekleminde en az {MIN_CLUSTERS} küme "
                                    f"gerekir. {remedy}")


def suggest(table: K.UploadedTable) -> dict[str, str]:
    """İlk açılışta önerilen roller (öğrenci değiştirebilir). Adında tedavi sözcüğü (program, deney, tedavi …) geçen,
    tedavi öncesi bir değişkene benzemeyen ve iki kategorisinde de en az iki gözlem olan sütun tedavi göstergesi; adı
    birime benzeyen, boş hücresi olmayan ve en az ``MIN_CLUSTERS``, en çok gözlem sayısının yarısı kadar farklı değeri
    olan sütun atama birimi; adında tedavi sözcüğü geçmeyen, akış/seçilmiş ya da grup ile başlangıç/düzey sözcüğü geçen
    iki değerli sütun seçilmiş grup; adında "başlangıç" geçen sayısal sütun seçimin dayandığı değişken."""

    hints: dict[str, str] = {}
    n = len(table.frame)
    for column in table.columns:
        words = set(K.name_words(column))
        cells = table.frame[column].map(K.clean_text)
        values = table.frame[column][cells.notna()]
        counts = cells.dropna().value_counts()
        levels = len(counts)
        named = bool(words & TREATMENT_WORDS)
        treatment = named and not words & (BASELINE_WORDS | PRE_WORDS)
        if TEDAVI not in hints and treatment and levels == 2 and counts.min() >= 2:
            hints[TEDAVI] = column
        elif ATAMA not in hints and words & CLUSTER_WORDS and MIN_CLUSTERS <= levels <= n // 2 \
                and cells.notna().all():
            hints[ATAMA] = column
        elif SECILMIS not in hints and not named and levels == 2 and (
                words & SELECTED_WORDS or (words & GROUP_WORDS and words & (BASELINE_WORDS | LEVEL_WORDS))):
            hints[SECILMIS] = column
        elif SECIM not in hints and words & BASELINE_WORDS and pd.api.types.is_numeric_dtype(values) and levels > 2:
            hints[SECIM] = column
    if not (SECILMIS in hints and SECIM in hints):  # iki rol birlikte çalışır
        hints.pop(SECILMIS, None)
        hints.pop(SECIM, None)
    return hints


TREATMENT_WORDS = frozenset(("program", "deney", "tedavi", "müdahale", "mudahale", "treatment", "treated", "treat",
                             "katılım", "katilim"))
"""Tedavi önerisi; "kontrol" bilerek yok: "Kontrol grubu" sütununda 1 ile kodlanan kategori karşılaştırma grubudur."""
PRE_WORDS = frozenset(("öncesi", "oncesi", "before", "pre"))
CLUSTER_WORDS = frozenset(("okul", "school", "köy", "koy", "village", "küme", "kume", "cluster", "sınıf", "sinif",
                           "class", "blok", "block", "il", "ilçe", "ilce", "mahalle", "firma", "şube", "sube"))
SELECTED_WORDS = frozenset(("akış", "akis", "akışı", "akisi", "stream", "lowstream", "seçilmiş", "secilmis",
                            "selected"))
GROUP_WORDS = frozenset(("grup", "grubu", "group"))
LEVEL_WORDS = frozenset(("düşük", "dusuk", "yüksek", "yuksek", "zayıf", "zayif", "başarı", "basari", "alt", "üst",
                         "ust"))
BASELINE_WORDS = frozenset(("başlangıç", "baslangic", "baseline", "önceki", "onceki", "ilk"))


def sample() -> pd.DataFrame:
    """Örnek dosya: kurgusal okul deneyi (``core.labs.ornek_veri``)."""

    return deney_verisi()


ROLES = (
    Role(SONUC, "Sonuç değişkeni", "sayisal", True, (1, 3),
         "Tedaviden sonra ölçülen sayısal sonuç (ör. dönem sonu test puanı)."),
    Role(TEDAVI, "Tedavi göstergesi", "kategorik", True, (1, 2, 3, 4),
         "İki kategorili değişken: tedavi (ör. programa alınan okul) ve karşılaştırma grubu.",
         levels=(2, 2), pick="Tedavi grubu (1 ile kodlanan)"),
    Role(ATAMA, "Atama birimi (küme)", "serbest", False, (1, 2, 3, 4),
         "Tedavinin atandığı birim (ör. okul, köy). Seçilirse standart hatalar bu birim düzeyinde kümelenir; "
         "seçilmezse HC1 kullanılır. Boş hücre olamaz.", complete=True),
    Role(SECILMIS, "Seçilmiş grup göstergesi", "kategorik", False, (4,),
         "Rastgele atanmamış, başlangıç değerine göre oluşmuş iki kategorili grup (ör. düşük başarı akışı). Boş "
         "hücreler (ör. grubun tanımlı olmadığı karşılaştırma okulları) Adım 4'te dışarıda kalır.",
         levels=(2, 2), pick="Seçilmiş grup (1 ile kodlanan)", group="secim"),
    Role(SECIM, "Seçimin dayandığı başlangıç değişkeni", "sayisal", False, (4,),
         "Seçilmiş grubun oluşturulduğu tedavi öncesi değişken (ör. başlangıç puanı). Başlangıç değişkenlerinden biri "
         "de olabilir.", group="secim", shared=True),
)

CUSTOM = CustomLab(
    roles=ROLES,
    build=build,
    sample=sample,
    intro=(
        "Bir Excel (.xlsx) ya da CSV dosyası yükleyin. Sonuç ve iki kategorili tedavi göstergesi seçilir; atama birimi "
        "(küme) ve tedaviden önce ölçülmüş başlangıç değişkenleri isteğe bağlıdır. Sonucu ya da tedavisi boş olan "
        "satırlar analizden çıkarılır; başlangıç değişkenlerinde boş hücre olabilir: denge tablosunun her satırı o "
        "değişkenin gözlendiği satırlarla, kovaryat ayarlı model bütün başlangıç değişkenleri gözlenen satırlarla "
        "tahmin edilir. Adım 4 için seçilmiş grup göstergesi ve dayandığı başlangıç değişkeni birlikte seçilir."
    ),
    min_rows=10,
    extra_columns=True,
    extra_use="sayisal",
    extra_label="Başlangıç (tedavi öncesi) değişkenleri (isteğe bağlı, en çok 6)",
    extra_help="Denge tablosunda ve kovaryat ayarlı modelde kullanılır. Tedaviden sonra ölçülen değişken seçmeyin.",
    max_extra=6,
    extra_required=False,
    validate=validate,
    suggest=suggest,
)

VARIANTS = TopicVariants(alternative=alternative, story=STORY, custom=CUSTOM)
