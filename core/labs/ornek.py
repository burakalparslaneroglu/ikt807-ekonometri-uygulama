"""Uygulama sekmesinin ek veri kaynakları: alternatif örnek ve öğrencinin kendi verisi.

Ders notlarının laboratuvarları (``core.labs.konuNN``) değişmez. Her konu için ek olarak bir **genel uygulama** yazılır
(``core.labs.ornek_konuNN``): notlardaki adımlar aynı numaralarla ve aynı işlemlerle, fakat veri bir ``Case``'ten gelir.

* **Alternatif örnek**, genel uygulamanın Hansen'in veri arşivindeki başka bir gerçek veriyle kurulmuş hâlidir. Veri
  depoda tutulmaz; notlardaki gibi çalışma anında indirilir ya da yüklenir. Kontrollerin beklenen değerleri bu verinin
  tam örneklemiyle hesaplanmış sayılardır (``with_expected``): metinlerdeki sayılar da onlardır ve testlerde bağımsız bir
  hesapla doğrulanır. Böylece tanım veriden önce kurulur, kod ve indirilecek betik veri yüklenmeden de görünür.
* **Kendi verini yükle** seçeneğinde aynı genel uygulama öğrencinin Excel ya da CSV dosyasıyla kurulur. Kontrollerin
  beklenen değerleri uygulamanın aynı veriyle hesabıdır (``with_app_values``); indirilen Python ve R kodu bu değerleri
  yeniden üretmelidir. Yorumlar sonuçlardan yazılır (``LabStep.note_for``).

Öğrencinin sütun ve kategori adları metinlere ``md`` ile girer: Markdown ve KaTeX işaretleri kaçırılır, böylece bir ad
(ör. "Fiyat ($)", "a|b", "1. sınıf") sayfanın biçimini bozmaz. Ad hiçbir zaman matematik ifadesinin içine yazılmaz.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from core.labs.runner import run_lab
from core.labs.spec import CoefTarget, LabSpec, Operation, ScalarTarget

SOURCE_LABELS = {
    "notlar": "Notlardaki örnek",
    "alternatif": "Alternatif örnek",
    "kendi": "Kendi verini yükle",
}
POSITIVE_WORDS = ("1", "evet", "var", "kadın", "kadin", "female", "geçti", "başarılı", "tuttu", "memnun", "doğru",
                  "olumlu", "kabul", "program", "tedavi", "deney", "katıldı", "yes", "true")
"""İki kategorili bir değişkende varsayılan "1" kategorisi (ilk eşleşen; büyük-küçük harf Türkçe kuralıyla)."""


# --- Sayı yazımı (metinler için) --------------------------------------------------------

def sayi(value: float, decimals: int = 0) -> str:
    """Türkçe sayı: ondalık virgül, tipografik eksi (0,625; −1,5). Yuvarlanınca sıfır olan değer "−0" yazılmaz."""

    if abs(value) < 0.5 * 10 ** (-decimals):
        value = 0.0
    return f"{value:.{decimals}f}".replace(".", ",").replace("-", "−")


def yuzde(value: float, decimals: int = 1) -> str:
    """Yüzde işareti sayıdan önce (%62,5)."""

    return "%" + sayi(value, decimals)


def sayim(value: float) -> str:
    """Sayım: binlik ayırıcı nokta (4.360)."""

    return f"{int(round(value)):,}".replace(",", ".")


def liste(items: list[str]) -> str:
    """Türkçe sıralama: "A", "A ve B", "A, B ve C"."""

    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " ve " + items[-1]


def katsayi(value: float) -> str:
    """Katsayının metindeki yazımı: dört ondalık; çok küçük değerlerde anlamlı basamaklar korunur (0,000034 sıfır
    yazılmaz)."""

    if value != 0 and abs(value) < 0.00005:
        digits = min(12, 3 - int(math.floor(math.log10(abs(value)))))
        return sayi(value, digits)
    return sayi(value, 4)


_MARKDOWN = re.compile(r"([\\`*_\[\]<>#|$~&])")


def md(text: object) -> str:
    """Markdown metnine girecek kullanıcı metni (sütun ya da kategori adı): biçim işaretleri kaçırılır, satır sonları
    boşluk olur, baştaki "1." numaralı liste sanılmaz."""

    value = re.sub(r"\s*[\r\n]+\s*", " ", str(text))
    value = _MARKDOWN.sub(r"\\\1", value)
    return re.sub(r"^(\d+)([.)])", r"\1\\\2", value)


def default_pick(categories: Iterable[str]) -> str:
    """İki kategorili değişkende 1 ile kodlanacak varsayılan kategori: {0, 1} kodunda 1, aksi hâlde Evet, Kadın, Program
    gibi bir kategori; yoksa sıradaki son kategori (alfabetik sırada ikincisi)."""

    from core.labs.kendi_veri import fold

    items = list(categories)
    folded = {fold(item): item for item in items}
    for word in POSITIVE_WORDS:
        if word in folded:
            return folded[word]
    return items[-1]


# --- Örnek (vaka) ---------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Case:
    """Genel uygulamanın verisi ve değişkenlerin rolleri.

    ``load``: veriyi kuran işlemler (alternatif örnekte Hansen veri seti ve türetmeler, kendi verinde ``ReadFile``).
    ``data``: kendi verinde aynı verinin temizlenmiş hâli (seçenekleri, metinleri ve doğrulamaları kurmak için);
    alternatif örnekte ``None`` (tanım veriden önce kurulur). ``roles``: rol → sütun (koddaki ad); ``extras``: ek sayısal
    sütunlar (koddaki adlar); ``labels``: sütun → ekranda görünen ad; ``levels``: rol → 1 ile kodlanan kategori;
    ``orders``: sütun → kategori sırası; ``options``: açık/kapalı seçenekler (ör. sonucun logaritması); ``extra``:
    konuya özgü ayarlar ve metinler.
    """

    source: str
    load: tuple[Operation, ...]
    frame: str
    data: pd.DataFrame | None
    roles: Mapping[str, str]
    labels: Mapping[str, str]
    extras: tuple[str, ...] = ()
    levels: Mapping[str, str] = field(default_factory=dict)
    orders: Mapping[str, tuple] = field(default_factory=dict)
    options: Mapping[str, bool] = field(default_factory=dict)
    extra: Mapping[str, object] = field(default_factory=dict)

    def column(self, role: str) -> str:
        return self.roles[role]

    def label(self, role: str) -> str:
        return self.name(self.roles[role])

    def name(self, column: str) -> str:
        """Sütunun ekranda görünen adı."""

        return self.labels.get(column, column)

    def md(self, role: str) -> str:
        """Rolün sütun adı, Markdown metnine girecek biçimde."""

        return md(self.label(role))

    def has(self, role: str) -> bool:
        return role in self.roles

    @property
    def own(self) -> bool:
        return self.source == "kendi"


def app_values(spec: LabSpec, sources: Mapping[str, pd.DataFrame] | None = None) -> dict[tuple[int, str], float]:
    """Kontrollerin uygulamadaki değerleri: (adım, etiket) → değer. Gösterim basamağında sıfır olan değer (ör. 1e-17)
    kodda "-0.000" yazılmasın diye sıfır alınır; hesaplanamayan değer bir hatadır (doğrulama bunu önlemeliydi)."""

    run = run_lab(spec, dict(sources or {}))
    values: dict[tuple[int, str], float] = {}
    for step in spec.steps:
        for result in run.step_checks(step.number):
            if not math.isfinite(result.value):
                raise ValueError(f"{result.check.label}: değer hesaplanamadı.")
            small = abs(result.value) < 0.5 * 10 ** (-result.check.decimals)
            values[(step.number, result.check.label)] = 0.0 if small else float(result.value)
    return values


def with_expected(spec: LabSpec, values: Mapping[tuple[int, str], float]) -> LabSpec:
    """Kontrollerin beklenen değerlerini verilen sayılarla doldurur; her kontrolün tam bir değeri olmalıdır."""

    labels = {(step.number, check.label) for step in spec.steps for check in step.checks}
    if len(labels) != sum(len(step.checks) for step in spec.steps):
        raise ValueError(f"{spec.topic_key}: bir adımda aynı etiketli iki kontrol var.")
    missing = sorted(labels - set(values))
    extra = sorted(set(values) - labels)
    if missing or extra:
        raise ValueError(f"{spec.topic_key}: beklenen değerler kontrollerle eşleşmiyor (eksik: {missing}, fazla: {extra}).")
    steps = tuple(
        replace(step, checks=tuple(replace(check, expected=float(values[(step.number, check.label)]))
                                   for check in step.checks))
        for step in spec.steps
    )
    return replace(spec, steps=steps)


def with_app_values(spec: LabSpec, sources: Mapping[str, pd.DataFrame] | None = None) -> LabSpec:
    """Kontrollerin beklenen değerlerini uygulamanın kendi hesabıyla doldurur (kendi verin)."""

    return with_expected(spec, app_values(spec, sources))


# --- Kendi verin: denetimler ------------------------------------------------------------

EXACT_FIT = 1e-9
"""Artık kareler toplamı, toplam kareler toplamının bu oranından küçükse uyum (neredeyse) tamdır."""


def exact_fit(data: pd.DataFrame, y: str, regressors: Iterable[str]) -> bool:
    """Sonuç, regresörlerin (neredeyse) tam bir doğrusal birleşimi mi (R² ≈ 1). Böyle bir veride standart hata sıfıra
    çok yakındır; standart hata, t, p-değeri ve Breusch–Pagan testi yuvarlama hatasına duyarlıdır, Python ile R'de aynı
    çıkmaz."""

    columns = list(dict.fromkeys(regressors))
    complete = data[[y, *columns]].dropna().astype(float)
    design = np.column_stack([np.ones(len(complete)), complete[columns].to_numpy()])
    target = complete[y].to_numpy()
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    total = float(((target - target.mean()) ** 2).sum())
    return float((residual ** 2).sum()) <= EXACT_FIT * total


def full_rank(data: pd.DataFrame, columns: Iterable[str]) -> bool:
    """Sabit terim ve ``columns`` sütunlarından kurulan tasarım matrisi tam sütun ranklı mı (tam doğrusal bağlantı yok)."""

    names = list(dict.fromkeys(columns))
    complete = data[names].dropna().astype(float)
    if len(complete) <= len(names):
        return False
    design = np.column_stack([np.ones(len(complete)), complete.to_numpy()])
    scale = np.abs(design).max(axis=0)
    scale[scale == 0] = 1.0
    singular = np.linalg.svd(design / scale, compute_uv=False)
    return bool(singular[-1] > 1e-8 * singular[0])


NUMERIC_LIMIT = 1e-9
"""Uygulamanın OLS katsayıları (statsmodels, sözde ters) ile sütunları ortalanıp ölçeklenmiş tasarımdaki çözüm arasında
izin verilen en büyük fark (büyük katsayılarda göreli). Kötü ölçekte (ör. milyonlarla ölçülen bir kontrolün karesi)
sözde ters hassasiyet kaybeder; R'nin QR çözümü etkilenmez, iki dil ayrışır."""
CONDITION_LIMIT = 3e3
"""Sütunları birim uzunluğa ölçeklenmiş tasarımın (sabit terim dahil) en büyük koşul sayısı. Üstünde değişkenler
neredeyse doğrusal bağlantılıdır (ör. dar aralıkta yıl ve karesi): QR ve sözde ters çözümleri dördüncü ondalıkta
ayrışabilir. Sıradan verilerde değer 10–1.000 aralığındadır (ör. yaş ve karesi 79, boy (cm) ve karesi 876); yıl ve
karesi 10⁶ düzeyindedir."""


def stable_design(data: pd.DataFrame, outcome: str, designs: Iterable[Iterable[str]]) -> bool:
    """Her tasarımda (sabit terim + sütunlar) katsayılar yazılımlar arasında dördüncü ondalıkta aynı hesaplanabilir mi:
    ölçeklenmiş koşul sayısı ``CONDITION_LIMIT`` altında ve uygulamanın çözümü, sütunları ortalanıp ölçeklenmiş
    tasarımdaki çözümle ``NUMERIC_LIMIT`` içinde aynı. Sıradan verilerde fark 1e-14 düzeyindedir."""

    import statsmodels.api as sm

    for columns in designs:
        names = list(dict.fromkeys(columns))
        complete = data[[outcome, *names]].dropna().astype(float)
        target = complete[outcome].to_numpy()
        values = complete[names].to_numpy()
        design = np.column_stack([np.ones(len(complete)), values])
        if not np.isfinite(design).all() or not np.isfinite(target).all():
            return False
        norms = np.linalg.norm(design, axis=0)
        norms[norms == 0] = 1.0
        if float(np.linalg.cond(design / norms)) > CONDITION_LIMIT:
            return False
        app = np.asarray(sm.OLS(target, design).fit().params, dtype=float)
        means, scales = values.mean(axis=0), values.std(axis=0)
        scales[scales == 0] = 1.0
        standard = np.column_stack([np.ones(len(complete)), (values - means) / scales])
        solution, *_ = np.linalg.lstsq(standard, target, rcond=None)
        slopes = solution[1:] / scales
        reference = np.concatenate([[solution[0] - slopes @ means], slopes])
        if np.any(np.abs(app - reference) > NUMERIC_LIMIT * np.maximum(1.0, np.abs(reference))):
            return False
    return True


SCALE_MESSAGE = ("Seçilen değişkenlerle katsayılar yazılımlar arasında aynı hassasiyetle hesaplanamıyor. İki yaygın "
                 "neden var: değişkenlerin ölçekleri çok farklıdır (ör. bir değişken milyonlarla ölçülüyor ya da bir "
                 "kontrolün karesi çok büyük) ya da bir değişken dar bir aralıkta büyük değerler alıyor (ör. yıl, doğum "
                 "yılı). Dosyada büyük değişkenlerin birimini değiştirin (ör. TL yerine bin TL) ya da yıl yerine bir "
                 "başlangıç yılından bu yana geçen süreyi (ör. yıl − 2015) veya yaşı kullanın.")


_FRAGILE = ("se", "se_hc1", "p")


def stable_checks(checks, exact: bool, fragile_scalars: Iterable[str] = ()):
    """Uyum tamsa standart hataya bağlı kontroller (standart hata, p-değeri ve verilen skalerler, ör. Breusch–Pagan LM)
    çıkarılır."""

    if not exact:
        return tuple(checks)
    scalars = set(fragile_scalars)

    def fragile(check) -> bool:
        target = check.target
        if isinstance(target, CoefTarget):
            return target.quantity in _FRAGILE
        return isinstance(target, ScalarTarget) and target.name in scalars

    return tuple(check for check in checks if not fragile(check))


EXACT_FIT_NOTE = ("Sonuç, açıklayıcı değişkenlerin neredeyse tam bir doğrusal birleşimidir (R² ≈ 1). Böyle bir veride "
                  "standart hatalar sıfıra çok yakındır ve yuvarlama hatasına duyarlıdır; indirilen kod bu değerleri "
                  "karşılaştırmaz.")


# --- Kendi verin: roller ----------------------------------------------------------------

@dataclass(frozen=True)
class Role:
    """Kendi verinde öğrencinin bir sütun seçtiği rol.

    ``use``: ``kategorik``, ``sayisal`` ya da ``serbest`` (olduğu gibi; ör. kimlik sütunu). ``required``: rol zorunludur
    ve bu sütunda değeri olmayan satırlar analizden çıkarılır. ``levels``: kategorik rolün kategori sayısı aralığı.
    ``pick``: verilirse öğrenci bu rolün kategorilerinden birini de seçer (ör. 1 ile kodlanan grup).
    """

    key: str
    label: str
    use: str
    required: bool
    steps: tuple[int, ...]
    help: str
    levels: tuple[int, int] = (2, 15)
    pick: str | None = None
    group: str | None = None
    suggest: bool = False
    """İsteğe bağlı rol için de dosyadan bir sütun önerilir (öğrenci kaldırabilir)."""
    complete: bool = False
    """Seçilirse sütunda boş hücre olamaz."""
    unique: bool = False
    """Seçilirse sütundaki değerler birbirinden farklı olmalı (kimlik sütunu)."""
    shared: bool = False
    """Bu rolün sütunu ek sütunlardan biri olarak da seçilebilir (ör. seçilmiş grubun dayandığı başlangıç puanı, denge
    tablosunda da yer alır)."""


@dataclass(frozen=True)
class Option:
    """Kendi verinde konuya özgü açık/kapalı seçenek (ör. sonucun logaritması).

    ``role``: seçeneğin bağlı olduğu rol; ``allowed``: o rolün sütunu (dosyadaki değerler) için seçenek kullanılabilir mi;
    kullanılamazsa seçenek kapalı kalır ve ``blocked`` açıklaması gösterilir.
    """

    key: str
    label: str
    help: str
    default: bool = True
    role: str | None = None
    allowed: Callable[[pd.Series], bool] | None = None
    blocked: str = ""


@dataclass(frozen=True)
class CustomLab:
    """Bir konunun "Kendi verini yükle" tanımı: roller, genel uygulamayı kuran fonksiyon ve örnek dosya.

    ``extra_columns``: öğrenci rollerin dışında ek sütunlar da seçebilir (``extra_use`` kullanımıyla; ör. ek sayısal
    kontroller), en çok ``max_extra`` sütun. ``extra_required``: ek sütunlarda boş hücresi olan satırlar da analizden
    çıkarılır (bütün modeller aynı gözlemlerle kurulsun diye).
    """

    roles: tuple[Role, ...]
    build: Callable[[Case], LabSpec]
    sample: Callable[[], pd.DataFrame]
    intro: str
    options: tuple[Option, ...] = ()
    order_roles: tuple[str, ...] = ()
    """Kategori sırası seçeneğinin uygulandığı roller."""
    min_rows: int = 5
    extra_columns: bool = False
    extra_use: str = "sayisal"
    extra_label: str = "Ek sayısal değişkenler (isteğe bağlı)"
    extra_help: str = ""
    max_extra: int = 6
    extra_required: bool = False
    validate: Callable[[Case], None] | None = None
    """Konuya özgü ek denetim; kullanılamıyorsa ``UploadError``."""
    suggest: Callable[[object], Mapping[str, str]] | None = None
    """Dosyanın ilk açılışında rollere önerilen sütunlar (``UploadedTable`` → rol → sütun)."""


@dataclass(frozen=True)
class TopicVariants:
    """Bir konunun ek veri kaynakları."""

    alternative: Callable[[], LabSpec]
    story: str
    """Alternatif örneğin kısa tanımı (sekmenin üstünde gösterilir)."""
    custom: CustomLab | None = None


# --- Kendi verin: seçimlerden örneğe -----------------------------------------------------

ORDER_TEXT = {
    "alfabetik": "alfabetik sırayla",
    "dosya": "dosyadaki ilk görülme sırasıyla",
    "frekans": "frekansa göre (çoktan aza)",
}


@dataclass(frozen=True)
class CustomChoices:
    """Öğrencinin veri panelindeki seçimleri.

    ``roles``: rol → dosyadaki sütun adı (seçilmediyse ``None``); ``extra``: ek sütunlar (dosyadaki adlar); ``order``:
    kategori sırası kuralı (``kendi_veri.ORDER_RULES``); ``picks``: rol → seçilen kategori; ``options``: seçenek →
    açık/kapalı.
    """

    roles: Mapping[str, str | None]
    extra: tuple[str, ...] = ()
    order: str = "alfabetik"
    picks: Mapping[str, str] = field(default_factory=dict)
    options: Mapping[str, bool] = field(default_factory=dict)


def _selections(custom: CustomLab, table, choices: CustomChoices):
    from core.labs import kendi_veri as K

    for role in custom.roles:
        if role.required and not choices.roles.get(role.key):
            raise K.UploadError(f"“{role.label}” için bir sütun seçin.")
    groups: dict[str, list[Role]] = {}
    for role in custom.roles:
        if role.group:
            groups.setdefault(role.group, []).append(role)
    for members in groups.values():
        chosen = [role for role in members if choices.roles.get(role.key)]
        if chosen and len(chosen) < len(members):
            missing = [f"“{role.label}”" for role in members if not choices.roles.get(role.key)]
            raise K.UploadError("Bu roller birlikte çalışır; şunlar için de sütun seçin: " + liste(missing) + ".")
    if len(choices.extra) > custom.max_extra:
        raise K.UploadError(f"En çok {custom.max_extra} ek sütun seçilebilir.")
    uses: dict[str, str] = {}
    required: dict[str, bool] = {}
    shared: dict[str, bool] = {}
    for role in custom.roles:
        original = choices.roles.get(role.key)
        if not original:
            continue
        shared[original] = shared.get(original, True) and role.shared
        if original not in table.columns:
            raise K.UploadError(f"“{original}” sütunu dosyada yok.")
        previous = uses.get(original)
        if previous is not None and previous != role.use:
            raise K.UploadError(f"“{original}” sütunu farklı türde iki rol için seçildi; her rol için uygun bir sütun "
                                "seçin.")
        uses[original] = role.use
        required[original] = required.get(original, False) or role.required
    for original in choices.extra:
        if original not in table.columns:
            raise K.UploadError(f"“{original}” sütunu dosyada yok.")
        if original in uses:
            if shared.get(original) and uses[original] == custom.extra_use:
                required[original] = required[original] or custom.extra_required
                continue
            raise K.UploadError(f"“{original}” sütunu hem bir rol için hem ek sütun olarak seçildi.")
        uses[original] = custom.extra_use
        required[original] = custom.extra_required
    # Kod adları dosyadaki bütün sütunlar için dosya sırasıyla verilir: bir sütunun adı seçimlere bağlı değildir
    # ("Gelir (TL)" ile "Gelir TL" hangi rollere seçilirse seçilsin gelir_tl ve gelir_tl_2 olur).
    taken: set[str] = set()
    codes: dict[str, str] = {}
    for original in table.columns:
        codes[original] = K.code_name(original, taken)
        taken.add(codes[original])
    return [K.Selection(codes[original], original, use, required=required.get(original, False))
            for original, use in uses.items()]


def custom_case(custom: CustomLab, table, choices: CustomChoices) -> tuple[Case, tuple[str, ...]]:
    """Öğrencinin dosyası ve seçimlerinden genel uygulamanın örneğini kurar; kullanılamıyorsa ``UploadError``."""

    from core.labs import kendi_veri as K

    selections = _selections(custom, table, choices)
    prepared = K.prepare(table, selections, frame="veri", comment=f"Yüklediğiniz veri dosyası: {table.file_name}")
    data = prepared.frame
    if len(data) < custom.min_rows:
        raise K.UploadError(f"Analiz için en az {custom.min_rows} gözlem gerekir; seçilen sütunlarda {len(data)} "
                            "gözlem var.")
    names = {item.original: item.name for item in selections}
    roles = {role.key: names[choices.roles[role.key]] for role in custom.roles if choices.roles.get(role.key)}
    extras = tuple(names[original] for original in choices.extra)
    labels = {item.name: item.original for item in selections}
    for role in custom.roles:
        if role.key not in roles:
            continue
        column = roles[role.key]
        values = data[column]
        blanks = int(values.isna().sum())
        if role.complete and blanks:
            raise K.UploadError(f"“{labels[column]}” sütununda {blanks} boş hücre var. “{role.label}” rolündeki "
                                "sütunda her gözlemin değeri olmalı; boş hücreleri doldurun ya da başka bir sütun "
                                "seçin.")
        if role.unique and values.duplicated().any():
            repeated = values[values.duplicated()].iloc[0]
            raise K.UploadError(f"“{labels[column]}” sütununda tekrar eden değerler var (ör. “{repeated}”). "
                                f"“{role.label}” rolündeki sütunda her gözlemin değeri farklı olmalı.")
    orders: dict[str, tuple] = {}
    levels: dict[str, str] = {}
    for role in custom.roles:
        if role.key not in roles or role.use != "kategorik":
            continue
        column = roles[role.key]
        K.check_levels(data[column], labels[column], *role.levels)
        orders[column] = K.category_order(data[column], choices.order)
        if role.pick:
            pick = choices.picks.get(role.key)
            levels[role.key] = pick if pick in orders[column] else default_pick(orders[column])
    options = {option.key: bool(choices.options.get(option.key, option.default)) for option in custom.options}
    case = Case(
        source="kendi",
        load=(prepared.read,),
        frame="veri",
        data=data,
        roles=roles,
        labels=labels,
        extras=extras,
        levels=levels,
        orders=orders,
        options=options,
        extra={"order_text": ORDER_TEXT[choices.order], "file_name": table.file_name},
    )
    if custom.validate is not None:
        custom.validate(case)
    return case, prepared.notes
