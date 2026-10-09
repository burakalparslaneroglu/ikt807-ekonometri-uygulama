"""Konu 4 genel uygulaması: ilk aşamadan 2SLS'ye.

Notlardaki §4.17'nin beş adımı aynı numaralarla ve aynı işlemlerle, verisi değiştirilebilir biçimde yazılır (``build``):
OLS, ilk aşama, indirgenmiş biçim ve 2SLS (HC1), ilk aşama F'si, dolaylı EKK oranı, kesin yüzde dönüşüm ve elle ikinci
aşama. Alternatif örnek Angrist ve Krueger (1991) verisidir (Hansen'in arşivindeki ``AK1991.dta``): araç, yılın ilk
çeyreğinde doğmuş olmak. "Kendi verini yükle" seçeneğinde aynı adımlar öğrencinin dosyasıyla kurulur. Notlardaki
laboratuvar (``core.labs.konu04``) değişmez.

Rollerin notlardaki karşılıkları: sonuç log saatlik ücret, endojen açıklayıcı eğitim yılı, araç koleje yakınlık, dışsal
kontroller deneyim profili, ırk, metropol ve bölge göstergeleri.
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
    md,
    sayi,
    sayim,
    stable_checks,
    stable_design,
    with_app_values,
    with_expected,
    yuzde,
)
from core.labs.ornek_veri import arac_verisi
from core.labs.spec import (
    IV,
    OLS,
    Check,
    CoefTarget,
    Derive,
    DropMissing,
    EffectTable,
    LabSpec,
    LabStep,
    LoadHansen,
    NoteRef,
    Operation,
    Predict,
    ReproClass,
    Scalar,
    ScalarTarget,
    StatTarget,
)

TOPIC = "konu04"
SECTION = "4.17"
SONUC, ICSEL, ARAC = "sonuc", "icsel", "arac"
LOG = "log"
LOG_Y, HAT = "log_sonuc", "icsel_hat"
"""Kendi verinde türetilen sütunlar (``kendi_veri.RESERVED_CODES``: öğrencinin sütunlarına verilmez)."""
WEAK_F = 10.0


@dataclass(frozen=True)
class Plan:
    """Rollerin koddaki sütunları ve etiketlerdeki adlar.

    ``word``: endojen değişkenin kontrol etiketlerindeki adı (notlarda "eğitim"; kendi verinde boş, etiketler genel);
    ``rows``: Tablo 4.1 karşılığının satır adları; ``hat``: ilk aşama uyum değerlerinin sütunu.
    """

    frame: str
    outcome: str
    endogenous: str
    instrument: str
    controls: tuple[str, ...]
    categorical: tuple[str, ...]
    log: bool
    prepare: tuple[Operation, ...]
    hat: str
    word: str
    rows: dict
    labels: dict


def _coef(model: str, term: str, label: str, quantity: str = "coef", decimals: int = 4) -> Check:
    return Check(label, CoefTarget(model, term, quantity), 0.0, decimals)


def _percent(model: str, term: str) -> E.Expr:
    return E.mul(100, E.sub(E.exp(E.coef(model, term)), 1))


def _steps(plan: Plan, texts: dict, exact: bool) -> tuple[LabStep, ...]:
    x, z, w = plan.endogenous, plan.instrument, plan.controls
    word = f" {plan.word}" if plan.word else ""

    def ols(name: str, outcome: str, regressors: tuple[str, ...]) -> OLS:
        return OLS(name, plan.frame, outcome, regressors, vcov="HC1", categorical=plan.categorical)

    step2_ops: list[Operation] = [
        ols("ols", plan.outcome, (x, *w)),
        ols("ilk", x, (z, *w)),
        ols("indirgenmis", plan.outcome, (z, *w)),
        IV("iv", plan.frame, plan.outcome, (x,), (z,), w, categorical=plan.categorical),
        EffectTable(
            (
                (plan.rows["ols"], "ols", x),
                (plan.rows["ilk"], "ilk", z),
                (plan.rows["indirgenmis"], "indirgenmis", z),
                (plan.rows["iv"], "iv", x),
            ),
            "tablo_iv",
            title=plan.rows["baslik"],
            se_label="HC1 SH",
        ),
        Scalar("ilk_asama_F", E.power(E.div(E.coef("ilk", z), E.se("ilk", z)), 2), "İlk aşama F (robust t²)",
               percent=False),
        Scalar("oran", E.div(E.coef("indirgenmis", z), E.coef("ilk", z)), "Oran λ̂/π̂ (dolaylı EKK)", percent=False,
               decimals=4),
    ]
    step2_checks = [
        _coef("ols", x, f"OLS{word} katsayısı"),
        _coef("ols", x, "OLS HC1 SH", "se"),
        _coef("ilk", z, "İlk aşama katsayısı"),
        _coef("ilk", z, "İlk aşama HC1 SH", "se"),
        _coef("indirgenmis", z, "İndirgenmiş biçim katsayısı"),
        _coef("indirgenmis", z, "İndirgenmiş biçim HC1 SH", "se"),
        _coef("iv", x, f"2SLS{word} katsayısı"),
        _coef("iv", x, "2SLS HC1 SH", "se"),
        Check("İlk aşama F", ScalarTarget("ilk_asama_F"), 0.0, decimals=2),
        Check("Oran λ̂/π̂", ScalarTarget("oran"), 0.0),
    ]
    if plan.log:
        step2_ops += [
            Scalar("ols_yuzde", _percent("ols", x), "OLS kesin yüzde etki", decimals=1),
            Scalar("iv_yuzde", _percent("iv", x), "2SLS kesin yüzde etki", decimals=1),
        ]
        step2_checks += [
            Check("OLS yüzde etki", ScalarTarget("ols_yuzde"), 0.0, decimals=1),
            Check("2SLS yüzde etki", ScalarTarget("iv_yuzde"), 0.0, decimals=1),
        ]
    return (
        LabStep(number=1, title="Yapısal denklem, ilk aşama ve indirgenmiş biçim", note=NoteRef(SECTION, 1),
                explanation=texts[1][0], operations=plan.prepare,
                checks=(Check("Analiz örneklemi (N)", StatTarget(plan.frame, plan.outcome, "count"), 0.0,
                              decimals=0),),
                takeaway=texts[1][1], note_for=texts.get((1, "not"))),
        LabStep(number=2, title="Sonuçları tek bir tanımlama zinciri olarak okumak",
                note=NoteRef(SECTION, 2, ("Tablo 4.1",)), explanation=texts[2][0], operations=tuple(step2_ops),
                checks=stable_checks(step2_checks, exact, ("ilk_asama_F",)), reproducibility=ReproClass.CONVENTION,
                takeaway=texts[2][1], code_note=CODE_NOTE_2SLS, note_for=texts.get((2, "not"))),
        LabStep(number=3, title="Araç geçerliliğini p-değerinden türetmemek", note=NoteRef(SECTION, 3),
                explanation=texts[3][0], takeaway=texts[3][1], note_for=texts.get((3, "not"))),
        LabStep(
            number=4,
            title="2SLS'i yazılım komutunun arkasındaki iki projeksiyon olarak görmek",
            note=NoteRef(SECTION, 4),
            explanation=texts[4][0],
            operations=(
                Predict("ilk", plan.frame, plan.hat, "fitted"),
                ols("elle", plan.outcome, (plan.hat, *w)),
                EffectTable(
                    ((plan.rows["iv_komut"], "iv", x), (plan.rows["elle"], "elle", plan.hat)),
                    "iki_asama",
                    title="Aynı katsayı, farklı standart hata",
                    se_label="HC1 SH",
                ),
            ),
            checks=(_coef("elle", plan.hat, "Elle ikinci aşama katsayısı"),),
            reproducibility=ReproClass.CONVENTION,
            code_note=(
                "Tablodaki 2SLS satırı Adım 2'deki ayarlarla hesaplanır (Python `debiased=True`, R "
                "`vcovHC(type = \"HC1\")`, Stata `vce(robust) small`); elle ikinci aşama sıradan HC1 OLS'dir."
            ),
            takeaway=texts[4][1],
            note_for=texts.get((4, "not")),
        ),
        LabStep(number=5, title="Bir IV makalesindeki tabloyu nasıl okumalıyız?", note=NoteRef("4.17.5", 0),
                explanation=CHECKLIST, takeaway=texts[5][1], note_for=texts.get((5, "not"))),
    )


def _spec(plan: Plan, texts: dict, *, source: str, title: str, dataset: str, exact: bool = False) -> LabSpec:
    return LabSpec(topic_key=TOPIC, title=title, dataset=dataset, note_section=SECTION,
                   steps=_steps(plan, texts, exact), labels=tuple(plan.labels.items()), source=source)


# --- Ortak metinler -------------------------------------------------------------------------------------------

CODE_NOTE_2SLS = (
    "2SLS standart hatası dillerde aynı ayarla hesaplanır: Python `debiased=True`, R `vcovHC(type = \"HC1\")`, Stata "
    "`vce(robust) small`. Stata'da `small` yazılmazsa dayanıklı kovaryans n/(n−k) ile ölçeklenmez. Yazılım "
    "tablolarındaki p-değerleri t(n−k) dağılımıyla, uygulamadaki tablo normal yaklaşımla hesaplanır."
)
CHAIN = (
    "Dört tahmin aynı kontrollerle ve HC1 standart hatasıyla: OLS, ilk aşama, indirgenmiş biçim ve 2SLS. İlk aşama "
    "gücü robust $t$ istatistiğinin karesiyle ($F=t^2$) özetlenir. Tek endojen değişken ve tek dışlanmış araçta "
    "$\\hat\\lambda/\\hat\\pi$ oranı (dolaylı en küçük kareler) 2SLS katsayısını verir."
)
REPORTING = (
    "Bu üç denklem ayrı raporlama nesneleridir. İyi bir IV makalesi yalnız ikinci aşama katsayısını gösterip ilk "
    "aşamayı gizlememelidir."
)
VALIDITY_RULE = (
    "İlk soru veriden doğrudan araştırılabilir; ikinci soru büyük ölçüde ekonomik ve kurumsal argümana dayanır. İlk "
    "aşamanın güçlü olması dışlama kısıtını kanıtlamaz. Tam tanımlı modelde tek araç için aşırı tanımlama testi de "
    "yoktur."
)
PROJECTION = (
    "2SLS, endojen değişkenin **araçların açıkladığı bileşenini** kullanır. Bunu elle görmek için ilk aşamanın uyum "
    "değerlerini alır, sonucu bunlar ve aynı kontroller üzerine regres ederiz. Katsayı 2SLS ile birebir aynıdır; "
    "standart hata ise aynı değildir."
)
SECOND_STAGE = (
    "Elle ikinci aşamanın standart hatası 2SLS standart hatası değildir: uyum değerleri aynı örneklemden tahmin "
    "edilmiştir ve ikinci aşama artığı yapısal hata için doğru artık değildir; doğru artık $\\boldsymbol Y-\\boldsymbol X\\hat\\beta_{2SLS}$'dir "
    "(§4.8). Çıkarım için 2SLS rutininin IV-robust kovaryansı kullanılır. Elle ikinci aşamanın p-değeri indirgenmiş "
    "biçiminkiyle aynıdır, çünkü tam tanımlı modelde tahmin edilen endojen değişken araç ve kontrollerin doğrusal bir "
    "fonksiyonudur."
)
CHECKLIST = (
    "Bir IV sonuç tablosunda şunları arayın:\n\n"
    "1. OLS karşılaştırması,\n"
    "2. ilk aşama katsayıları ve zayıf araç tanıları,\n"
    "3. hangi değişkenlerin araç, hangilerinin dahil edilen dışsal kontrol olduğu,\n"
    "4. 2SLS katsayısı ve uygun robust/küme standart hatası,\n"
    "5. birden fazla araç varsa aşırı tanımlama sonuçları,\n"
    "6. IV estimand'ının hangi popülasyona ilişkin olduğunun tartışılması,\n"
    "7. aracın dışlama kısıtını destekleyen kurumsal gerekçe."
)
THESIS_RULE = (
    "Tez yazımında kaçınılacak ifade: \"İçsellik testi anlamlı olduğu için IV modeli doğrudur\" veya \"ilk aşama F "
    "istatistiği 10'u geçtiği için araç geçerlidir\". IV tasarımının güvenilirliği, araç ilgililiği ile "
    "dışlama/dışsallık argümanının birlikte savunulmasına bağlıdır."
)


def _system(y: str, x: str, z: str) -> str:
    """Notların §4.3 gösterimiyle üç denklem (ad sembolleri KaTeX'te düz metin)."""

    return (
        "$$\\begin{aligned}"
        f"\\text{{Yapısal:}}\\quad {y}_i &= \\beta\\, {x}_i + W_i'\\gamma + e_i\\\\"
        f"\\text{{İlk aşama:}}\\quad {x}_i &= \\pi\\, {z}_i + W_i'\\delta + v_i\\\\"
        f"\\text{{İndirgenmiş biçim:}}\\quad {y}_i &= \\lambda\\, {z}_i + W_i'\\rho + \\eta_i"
        "\\end{aligned}$$"
    )


# --- Alternatif örnek: Angrist ve Krueger (1991) --------------------------------------------------------------

ALT_DATA = "ak1991"
ALT_FRAME = "ak"
ALT_TITLE = "Angrist–Krueger Verisiyle İlk Aşamadan 2SLS'ye"
ALT_CONTROLS = ("yob", "black", "smsa", "region")


def alternative_plan() -> Plan:
    frame = ALT_FRAME
    return Plan(
        frame=frame, outcome="logwage", endogenous="edu", instrument="q1", controls=ALT_CONTROLS,
        categorical=("yob", "region"), log=True,
        prepare=(
            LoadHansen(ALT_DATA, "AK1991.dta", frame),
            Derive(frame, "q1", E.compare("eq", E.var("qob"), 1),
                   "Araç: yılın ilk çeyreğinde doğmuş (1), diğer çeyreklerde doğmuş (0)"),
            DropMissing(frame, ("logwage", "edu", "q1", *ALT_CONTROLS),
                        "Analiz örneklemi: kazanç, eğitim, araç ve bütün kontrolleri gözlenen erkekler"),
        ),
        hat="edu_hat", word="eğitim",
        rows={
            "baslik": "OLS, ilk aşama, indirgenmiş biçim ve 2SLS (Angrist–Krueger)",
            "ols": "OLS eğitim katsayısı",
            "ilk": "İlk aşama: q1 → eğitim",
            "indirgenmis": "İndirgenmiş biçim: q1 → log kazanç",
            "iv": "2SLS eğitim katsayısı",
            "iv_komut": "2SLS (yazılımın IV komutu)",
            "elle": "Elle ikinci aşama: tahmin edilen eğitim",
        },
        labels={
            E.INTERCEPT: "Sabit",
            "logwage": "Log haftalık kazanç (1979)",
            "edu": "Eğitim yılı",
            "q1": "İlk çeyrekte doğmuş",
            "qob": "Doğum çeyreği",
            "yob": "Doğum yılı",
            "black": "Siyahi",
            "smsa": "Metropol",
            "region": "Bölge",
            "edu_hat": "Tahmin edilen eğitim",
            "ols": "OLS",
            "ilk": "İlk aşama",
            "indirgenmis": "İndirgenmiş biçim",
            "iv": "2SLS",
            "elle": "Elle ikinci aşama",
        },
    )


ALT_TEXTS = {
    1: (
        "Soru: eğitimin kazanç üzerindeki etkisi. Notlardaki gibi eğitim tercihi yetenek ve aile geçmişi gibi "
        "gözlenmeyen etkenlerle ilişkili olabileceği için OLS nedensel getiriyi tanımlamayabilir. Araç: yılın ilk "
        "çeyreğinde doğmuş olmak (`q1`). ABD'de çocuklar okula doğdukları takvim yılına göre başlar, zorunlu eğitim ise "
        "okulu bırakma yaşına göre tanımlanır: ilk çeyrekte doğanlar okula daha büyük yaşta başlar ve yasal bırakma "
        "yaşına daha az eğitimle ulaşır (Angrist ve Krueger, 1991). Veri 1980 nüfus sayımından 1930–1939 doğumlu "
        "329.509 erkektir; sonuç 1979 haftalık kazancının logaritmasıdır. Üç denklem (notların §4.3 gösterimiyle):\n\n"
        + _system("logwage", "edu", "q1")
        + "\n\n$W_i$: sabit terim ile doğum yılı göstergeleri (1930 referans), siyahi, metropol ve bölge göstergeleri. "
        "Veride eksik değer yoktur; analiz örneklemi 329.509 erkektir.",
        REPORTING,
    ),
    2: (
        CHAIN,
        "İlk aşama: ilk çeyrekte doğanların eğitimi, diğer kontroller sabitken yaklaşık 0,098 yıl daha az; robust "
        "$F\\simeq57{,}73$. Etki küçüktür ama 329.509 gözlemde kesin ölçülür: aracın gücü katsayının tek başına "
        "büyüklüğünden değil, standart hatasına oranından okunur. Dolaylı en küçük kareler oranı "
        "$\\hat\\lambda/\\hat\\pi=(-0{,}00936)/(-0{,}09815)\\approx0{,}0953$, 2SLS katsayısının kendisidir. OLS'nin kesin "
        "yüzde dönüşümü yaklaşık %6,6, 2SLS'ninki %10,0; 2SLS'nin standart hatası (0,0261) OLS'ninkinin (0,00038) "
        "yaklaşık 70 katıdır. Card verisindeki gibi burada da \"IV daha büyük, o halde OLS aşağı yanlı\" sonucu otomatik "
        "değildir: araç varsayımları, ölçüm hatası, heterojen eğitim getirileri ve IV'nin hangi alt popülasyonun yerel "
        "etkisini tanımladığı birlikte düşünülmelidir. Dışsallık, dışlama ve monotonluk varsayımları altında 2SLS, "
        "doğum çeyreğinin eğitimini değiştirdiği erkeklerde (uyumlular, §4.12) eğitim getirilerinin ağırlıklı "
        "ortalamasıdır; ağırlıklar, aracın her eğitim eşiğini geçme olasılığına etkisiyle orantılıdır (Angrist ve "
        "Imbens, 1995; kontroller doymuş olmadığı için bu yorum yaklaşıktır). Bu veride ilk aşama katsayısının "
        "yaklaşık %54'ü 12 yıl ve altındaki eşiklerden, %46'sı lise sonrası eşiklerden gelir: zorunlu eğitim yasası "
        "yalnız ilk gruptaki eşikleri etkileyebileceği için tahmin, yalnız bu yasa yüzünden okulda kalan erkeklerin "
        "getirisi değildir. Card'ın koleje yakınlık aracı başka eşikleri ve başka erkekleri etkileyebilir.",
    ),
    3: (
        "IV için iki ayrı soru vardır:\n\n"
        "1. **İlgililik (relevance):** `q1` eğitimle ilişkili mi? İlk aşama bunu inceler.\n"
        "2. **Dışsallık/dışlama:** doğum çeyreği kazancı eğitim dışında başka bir kanal üzerinden etkiliyor mu ve "
        "yapısal hata ile ortogonal mi?",
        VALIDITY_RULE + " Doğum çeyreği rastgeleye yakın görünse de doğum mevsimi annenin özelliklerine göre değişebilir "
        "(Buckles ve Hungerman, 2013): bu, dışsallık ve dışlama varsayımlarının veriyle değil argümanla savunulduğunu "
        "hatırlatır.",
    ),
    4: (
        PROJECTION.replace("endojen değişkenin", "endojen eğitim değişkeninin"),
        SECOND_STAGE + " Bu veride elle ikinci aşamanın standart hatası (0,0271) 2SLS'ninkinden (0,0261) biraz "
        "büyüktür; genel olarak fark büyük olabilir ve yönü önceden bilinemez.",
    ),
    5: ("", THESIS_RULE + " Angrist ve Krueger'in özgün çalışması doğum çeyreği ile doğum yılı ve eyalet "
            "etkileşimlerinden çok sayıda araç kullanır; çok sayıda zayıf araç 2SLS'i OLS'ye doğru yanlı yapabilir "
            "(Bound, Jaeger ve Baker, 1995). Bu uygulama tek araç kullanır."),
}

ALT_EXPECTED = {
    (1, "Analiz örneklemi (N)"): 329509,
    (2, "OLS eğitim katsayısı"): 0.0639,
    (2, "OLS HC1 SH"): 0.0004,
    (2, "İlk aşama katsayısı"): -0.0981,
    (2, "İlk aşama HC1 SH"): 0.0129,
    (2, "İndirgenmiş biçim katsayısı"): -0.0094,
    (2, "İndirgenmiş biçim HC1 SH"): 0.0027,
    (2, "2SLS eğitim katsayısı"): 0.0953,
    (2, "2SLS HC1 SH"): 0.0261,
    (2, "İlk aşama F"): 57.73,
    (2, "Oran λ̂/π̂"): 0.0953,
    (2, "OLS yüzde etki"): 6.6,
    (2, "2SLS yüzde etki"): 10.0,
    (4, "Elle ikinci aşama katsayısı"): 0.0953,
}
"""Kontrollerin AK1991 tam örneklemindeki (329.509 erkek) değerleri, gösterim basamağında: (adım, etiket) → değer.
Metinlerdeki sayılar bunlardır; testler bağımsız bir hesapla ve üç dilde doğrular."""


@cache
def alternative() -> LabSpec:
    """Alternatif örneğin tanımı; beklenen değerler AK1991'in tam örneklemindeki sayılardır (testle doğrulanır)."""

    spec = _spec(alternative_plan(), ALT_TEXTS, source="alternatif", title=ALT_TITLE, dataset=ALT_DATA)
    return with_expected(spec, ALT_EXPECTED)


STORY = (
    "Alternatif örnek Angrist ve Krueger (1991) verisidir: 1980 nüfus sayımından 1930–1939 doğumlu 329.509 erkek "
    "(Hansen'in arşivindeki AK1991.dta). Araç yılın ilk çeyreğinde doğmuş olmaktır. Adımlar notlardaki gibidir: OLS, "
    "ilk aşama, indirgenmiş biçim, 2SLS ve elle ikinci aşama. Soru aynı (eğitimin getirisi), araç ve veri farklı."
)


# --- Kendi verin --------------------------------------------------------------------------------------------

def _derived_data(case: Case) -> pd.DataFrame:
    data = case.data.copy()
    if case.options.get(LOG, True):
        values = data[case.roles[SONUC]].astype(float)
        data[LOG_Y] = np.log(values.where(values > 0))  # pozitif olmayan değer: eksik (``validate`` reddeder)
    return data


def own_plan(case: Case) -> Plan:
    y, x, z = (case.roles[role] for role in (SONUC, ICSEL, ARAC))
    log = bool(case.options.get(LOG, True))
    name = case.name
    outcome = LOG_Y if log else y
    prepare: list[Operation] = list(case.load)
    if log:
        prepare.append(Derive(case.frame, LOG_Y, E.log(E.var(y)), "Sonucun doğal logaritması"))
    labels = {column: name(column) for column in (y, x, z, *case.extras)}
    labels.update({E.INTERCEPT: "Sabit", HAT: f"Tahmin edilen {name(x)}", "ols": "OLS", "ilk": "İlk aşama",
                   "indirgenmis": "İndirgenmiş biçim", "iv": "2SLS", "elle": "Elle ikinci aşama"})
    if log:
        labels[LOG_Y] = f"log({name(y)})"
    return Plan(
        frame=case.frame, outcome=outcome, endogenous=x, instrument=z, controls=tuple(case.extras), categorical=(),
        log=log, prepare=tuple(prepare), hat=HAT, word="",
        rows={
            "baslik": "OLS, ilk aşama, indirgenmiş biçim ve 2SLS",
            "ols": f"OLS: {name(x)}",
            "ilk": f"İlk aşama: {name(z)} → {name(x)}",
            "indirgenmis": f"İndirgenmiş biçim: {name(z)} → sonuç",
            "iv": f"2SLS: {name(x)}",
            "iv_komut": "2SLS (yazılımın IV komutu)",
            "elle": f"Elle ikinci aşama: tahmin edilen {name(x)}",
        },
        labels=labels,
    )


def _own_texts(case: Case, plan: Plan, exact: bool) -> dict:
    y, x, z = (case.md(role) for role in (SONUC, ICSEL, ARAC))
    outcome = f"log({y})" if plan.log else y
    controls = [f"“{md(case.name(column))}”" for column in plan.controls]
    exact_note = (" " + EXACT_FIT_NOTE) if exact else ""

    def step1(state) -> str:
        frame = state.frames[plan.frame]
        return (f"Analiz örneklemi {sayim(len(frame))} gözlem: sonuç, endojen değişken, araç ve bütün kontroller "
                "gözlenen satırlar.")

    def step2(state) -> str:
        first, reduced = state.models["ilk"], state.models["indirgenmis"]
        pi, lam = float(first.params[plan.instrument]), float(reduced.params[plan.instrument])
        f = state.scalars["ilk_asama_F"]
        b_ols, b_iv = float(state.models["ols"].params[plan.endogenous]), float(state.models["iv"].params[
            plan.endogenous])
        text = (f"İlk aşama: “{z}” bir birim yüksekken “{x}”, kontroller sabitken {katsayi(pi)} birim farklı; robust "
                f"$F\\simeq{sayi(f, 2).replace(',', '{,}')}$. ")
        if f < WEAK_F:
            text += ("F 10'un altında: araç zayıf olabilir. Zayıf araçta 2SLS OLS'ye doğru yanlıdır ve normal "
                     "yaklaşımlı güven aralıkları güvenilmez; \"F > 10\" yine de evrensel bir eşik değildir. ")
        else:
            text += "Bu, aracın endojen değişkenle kayda değer ilişkisini gösterir; F > 10 yine de evrensel bir kural değildir. "
        text += (f"Dolaylı EKK oranı λ̂/π̂ = {katsayi(lam)} / {katsayi(pi)} ≈ {katsayi(state.scalars['oran'])}, "
                 f"2SLS katsayısının kendisidir. OLS katsayısı {katsayi(b_ols)}, 2SLS {katsayi(b_iv)}.")
        if plan.log:
            text += (f" Kesin yüzde dönüşümle OLS {yuzde(state.scalars['ols_yuzde'], 1)}, 2SLS "
                     f"{yuzde(state.scalars['iv_yuzde'], 1)}.")
        text += (" İki katsayının farkı kendiliğinden OLS'nin yanlı olduğunu kanıtlamaz: araç varsayımları ve IV'nin "
                 "hangi alt grubun yerel etkisini ölçtüğü birlikte düşünülmelidir.")
        return text + exact_note

    def step4(state) -> str:
        table = state.tables["iki_asama"]
        iv_se, hand_se = table.iloc[0]["sh"], table.iloc[1]["sh"]
        if katsayi(hand_se) == katsayi(iv_se):
            comparison = f"2SLS'ninkiyle ({katsayi(iv_se)}) dört ondalıkta aynıdır"
        else:
            comparison = f"2SLS'ninkinden ({katsayi(iv_se)}) {'küçüktür' if hand_se < iv_se else 'büyüktür'}"
        return (f"Katsayılar aynı ({katsayi(table.iloc[0]['tahmin'])}); elle ikinci aşamanın standart hatası "
                f"({katsayi(hand_se)}) {comparison}. " + SECOND_STAGE + exact_note)

    return {
        1: (
            f"Soru: “{x}” değişkeninin “{y}” üzerindeki etkisi. “{x}” gözlenmeyen etkenlerle ilişkili olabileceği için "
            f"OLS bu etkiyi tanımlamayabilir. Araç: “{z}”. Üç denklem (notların §4.3 gösterimiyle): yapısal denklem "
            f"{outcome} = β·“{x}” + W′γ + e; ilk aşama “{x}” = π·“{z}” + W′δ + v; indirgenmiş biçim {outcome} = "
            f"λ·“{z}” + W′ρ + η. W: sabit terim" + (f" ile {', '.join(controls)}" if controls else "") + ".",
            REPORTING,
        ),
        (1, "not"): step1,
        2: (CHAIN, ""),
        (2, "not"): step2,
        3: (
            "IV için iki ayrı soru vardır:\n\n"
            f"1. **İlgililik (relevance):** “{z}” “{x}” ile ilişkili mi? İlk aşama bunu inceler.\n"
            f"2. **Dışsallık/dışlama:** “{z}” sonucu “{x}” dışında başka bir kanal üzerinden etkiliyor mu ve yapısal "
            "hata ile ortogonal mi?",
            VALIDITY_RULE,
        ),
        4: (PROJECTION, ""),
        (4, "not"): step4,
        5: ("", THESIS_RULE),
    }


def build(case: Case) -> LabSpec:
    """Kendi verinle Konu 4 uygulaması; kontrollerin beklenen değerleri uygulamanın hesabıdır."""

    plan = own_plan(case)
    data = _derived_data(case)
    exact = (exact_fit(data, plan.outcome, (plan.endogenous, *plan.controls))
             or exact_fit(data, plan.endogenous, (plan.instrument, *plan.controls))
             or exact_fit(data, plan.outcome, (plan.instrument, *plan.controls)))
    texts = _own_texts(case, plan, exact)
    spec = _spec(plan, texts, source="kendi", title=f"{case.label(SONUC)} ve {case.label(ICSEL)}", dataset="",
                 exact=exact)
    return with_app_values(spec)


def _first_stage_zero(data: pd.DataFrame, x: str, z: str, controls: tuple[str, ...]) -> bool:
    """Kontroller sabitken araç endojen değişkenle ilişkisiz mi (ilk aşama katsayısı sayısal olarak sıfır): 2SLS oranı
    tanımsızdır ve diller aynı sayıyı veremez."""

    complete = data[[x, z, *controls]].dropna().astype(float)
    design = np.column_stack([np.ones(len(complete)), complete[[*controls]].to_numpy()]) if controls else \
        np.ones((len(complete), 1))
    residual_z = complete[z].to_numpy() - design @ np.linalg.lstsq(design, complete[z].to_numpy(), rcond=None)[0]
    residual_x = complete[x].to_numpy() - design @ np.linalg.lstsq(design, complete[x].to_numpy(), rcond=None)[0]
    scale = float(np.linalg.norm(residual_z) * np.linalg.norm(residual_x))
    return scale == 0.0 or abs(float(residual_z @ residual_x)) <= 1e-8 * scale


def validate(case: Case) -> None:
    y, x, z = (case.roles[role] for role in (SONUC, ICSEL, ARAC))
    if len({y, x, z}) < 3:
        raise K.UploadError("Sonuç, endojen değişken ve araç için üç farklı sütun seçin.")
    if any(column in case.extras for column in (y, x, z)):
        raise K.UploadError("Kontrollerde sonuç, endojen değişken ya da araç olmamalı.")
    data = _derived_data(case)
    for column in (y, x, z, *case.extras):
        if data[column].nunique() < 2:
            raise K.UploadError(f"“{case.name(column)}” sütununda en az iki farklı değer olmalı.")
    if case.options.get(LOG, True) and (data[y] <= 0).any():
        raise K.UploadError(f"“{case.name(y)}” sütununda sıfır ya da negatif değer var; logaritma alınamaz. "
                            "“Sonucun logaritmasını al” seçeneğini kapatın ya da başka bir sütun seçin.")
    controls = tuple(case.extras)
    if len(data) <= len(controls) + 4:
        raise K.UploadError(f"Modellerde {len(controls) + 2} katsayı var; en az {len(controls) + 5} gözlem gerekir.")
    for columns in ((x, *controls), (z, *controls)):
        if not full_rank(data, columns):
            raise K.UploadError("Seçilen değişkenler arasında tam doğrusal bağlantı var (ör. araç ya da endojen değişken "
                                "kontrollerin doğrusal birleşimi). Kontrolleri değiştirin.")
    if _first_stage_zero(data, x, z, controls):
        raise K.UploadError(f"Kontroller sabitken “{case.name(z)}” ile “{case.name(x)}” ilişkisiz: ilk aşama katsayısı "
                            "sıfır olduğu için 2SLS tanımlı değildir. Başka bir araç seçin.")
    if exact_fit(data, x, (z, *controls)):
        raise K.UploadError(f"“{case.name(x)}”, araç ve kontrollerin (neredeyse) tam doğrusal bir fonksiyonu: ilk aşama "
                            "tam uyumludur (R² ≈ 1), ilk aşama F sınırsız büyür ve 2SLS, OLS ile aynı olur. Başka bir "
                            "endojen değişken ya da araç seçin.")
    outcome = LOG_Y if case.options.get(LOG, True) else y
    if not (stable_design(data, outcome, ((x, *controls), (z, *controls)))
            and stable_design(data, x, ((z, *controls),))):
        raise K.UploadError(SCALE_MESSAGE)


def sample() -> pd.DataFrame:
    """Örnek dosya: kurgusal araç değişkeni verisi (``core.labs.ornek_veri``)."""

    return arac_verisi()


ROLES = (
    Role(SONUC, "Sonuç değişkeni", "sayisal", True, (1, 2, 4),
         "Açıklanan sayısal değişken (ör. saatlik ücret). Notlardaki gibi logaritması alınabilir."),
    Role(ICSEL, "Endojen açıklayıcı değişken", "sayisal", True, (1, 2, 4),
         "Etkisi araştırılan, gözlenmeyen etkenlerle ilişkili olabilecek değişken (ör. eğitim yılı)."),
    Role(ARAC, "Araç değişkeni", "sayisal", True, (1, 2, 3, 4),
         "Dışlanmış araç: endojen değişkenle ilişkili olan, sonucu yalnız onun üzerinden etkilediği varsayılan değişken "
         "(ör. koleje yakınlık). 0/1 ya da sayısal olabilir."),
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
        "Bir Excel (.xlsx) ya da CSV dosyası yükleyin. Sonuç, endojen açıklayıcı değişken ve tek bir araç seçilir; "
        "dışsal kontroller isteğe bağlıdır ve bütün denklemlere girer. Seçilen sütunların birinde boş hücresi olan "
        "satırlar analizden çıkarılır: bütün modeller aynı gözlemlerle tahmin edilir."
    ),
    options=OPTIONS,
    min_rows=10,
    extra_columns=True,
    extra_use="sayisal",
    extra_label="Dışsal kontroller (isteğe bağlı, en çok 8)",
    extra_help="OLS, ilk aşama, indirgenmiş biçim ve 2SLS'e eklenir (ör. deneyim, cinsiyet göstergesi). Boş hücresi "
               "olan satırlar çıkarılır.",
    max_extra=8,
    extra_required=True,
    validate=validate,
)

VARIANTS = TopicVariants(alternative=alternative, story=STORY, custom=CUSTOM)
