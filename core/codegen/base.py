"""Laboratuvar tanımından Python, R ve Stata kodu üretmenin ortak çatısı."""

from __future__ import annotations

from dataclasses import dataclass

from core.labs import expr as E
from core.labs.spec import (
    IV,
    OLS,
    BinMeans,
    LocalCurve,
    AverageProfile,
    BinaryChoice,
    Check,
    Derive,
    MarginalEffects,
    ProfileCurves,
    QuantileRegression,
    Tobit,
    CoefTarget,
    Curve,
    LabSpec,
    MeanPoints,
    ModelLine,
    MonteCarlo,
    Scatter,
    ZeroLine,
    LabStep,
    LoadHansen,
    Operation,
    Scalar,
)

LANGUAGES = ("Python", "R", "Stata")

HANSEN_ARCHIVE_URL = "https://users.ssc.wisc.edu/~behansen/econometrics/Econometrics%20Data.zip"
HANSEN_PAGE_URL = "https://users.ssc.wisc.edu/~behansen/econometrics/"


@dataclass(frozen=True)
class LayerStyle:
    color: str
    rgb: str
    dashed: bool = False
    dotted: bool = False


_LINE_COLORS = (("#B3392F", "179 57 47"), ("#2F9E6B", "47 158 107"), ("#6B4C9A", "107 76 154"))
_POINT_COLORS = (("#107C89", "16 124 137"), ("#C98A1B", "201 138 27"))
_CURVE_COLORS = (("#07373D", "7 55 61"), ("#6B4C9A", "107 76 154"), ("#C98A1B", "201 138 27"))
_HISTOGRAM_COLORS = (("#107C89", "16 124 137"), ("#B3392F", "179 57 47"), ("#C98A1B", "201 138 27"))


def layer_styles(layers) -> list[LayerStyle]:
    """Grafik katmanlarının üç dilde ve uygulamada aynı renkleri.

    Aynı türden ikinci katman (ör. ikinci ortalama serisi) farklı renk alır; ilk
    katmanların renkleri bütün konularda sabittir.
    """

    styles: list[LayerStyle] = []
    counts = {"lines": 0, "points": 0, "curves": 0}
    for layer in layers:
        if isinstance(layer, MeanPoints):
            color, rgb = _POINT_COLORS[counts["points"] % len(_POINT_COLORS)]
            styles.append(LayerStyle(color, rgb))
            counts["points"] += 1
        elif isinstance(layer, Scatter):
            styles.append(LayerStyle("#9AA5A6", "154 165 166"))
        elif isinstance(layer, Curve):
            position = counts["curves"] if layer.color is None else layer.color
            color, rgb = _CURVE_COLORS[position % len(_CURVE_COLORS)]
            styles.append(LayerStyle(color, rgb, dashed=layer.dashed))
            counts["curves"] += 1
        elif isinstance(layer, ModelLine):
            color, rgb = _LINE_COLORS[counts["lines"] % len(_LINE_COLORS)]
            styles.append(LayerStyle(color, rgb, dashed=layer.dashed))
            counts["lines"] += 1
        elif isinstance(layer, ZeroLine):
            styles.append(LayerStyle("#07373D", "7 55 61", dotted=True))
        elif isinstance(layer, LocalCurve):
            position = counts["curves"] if layer.color is None else layer.color
            color, rgb = _CURVE_COLORS[position % len(_CURVE_COLORS)]
            styles.append(LayerStyle(color, rgb, dashed=layer.dashed))
            counts["curves"] += 1
        elif isinstance(layer, BinMeans):
            color, rgb = _POINT_COLORS[counts["points"] % len(_POINT_COLORS)]
            styles.append(LayerStyle(color, rgb))
            counts["points"] += 1
        else:
            raise TypeError(f"Tanınmayan grafik katmanı: {type(layer).__name__}")
    return styles


def histogram_styles(count: int) -> list[LayerStyle]:
    """Histogram serilerinin üç dilde ve uygulamada aynı renkleri."""

    return [LayerStyle(*_HISTOGRAM_COLORS[index % len(_HISTOGRAM_COLORS)]) for index in range(count)]


def flatten(operations) -> list[Operation]:
    """İşlemler ve Monte Carlo döngülerinin içindeki işlemler, sırasıyla."""

    found: list[Operation] = []
    for op in operations:
        found.append(op)
        if isinstance(op, MonteCarlo):
            found.extend(flatten(op.body))
    return found


@dataclass(frozen=True)
class LanguageInfo:
    name: str
    extension: str
    highlight: str
    mime: str


LANGUAGE_INFO = {
    "Python": LanguageInfo("Python", "py", "python", "text/x-python"),
    "R": LanguageInfo("R", "R", "r", "text/plain"),
    "Stata": LanguageInfo("Stata", "do", "stata", "text/plain"),
}


ESTIMATORS = (OLS, IV, BinaryChoice, Tobit, QuantileRegression)


def model_settings(spec: LabSpec) -> dict[str, object]:
    """Her model adının son tahmin ayarları (tür, kovaryans, küme); döngü içindekiler dahil.

    Marjinal etki sonuçları da katsayıları etkiler olan bir model gibi kaydedilir.
    """

    settings: dict[str, object] = {}
    for step in spec.steps:
        for op in flatten(step.operations):
            if isinstance(op, ESTIMATORS):
                settings[op.name] = op
            elif isinstance(op, MarginalEffects):
                settings[op.name] = op
    return settings


def expressions(operations) -> list[E.Expr]:
    """İşlemlerdeki bütün ifadeler (türetilmiş değişkenler, skalerler, grafik eğrileri, döngü çıktıları)."""

    from core.labs.spec import Curve, Plot, Scalar as ScalarOp

    found: list[E.Expr] = []
    for op in flatten(operations):
        if isinstance(op, Derive):
            found.append(op.expr)
        elif isinstance(op, ScalarOp):
            found.append(op.expr)
        elif isinstance(op, Plot):
            found.extend(layer.expr for layer in op.layers if isinstance(layer, Curve))
        elif isinstance(op, ProfileCurves):
            found.extend(expression for _, expression in op.derived)
        elif isinstance(op, MonteCarlo):
            found.extend(expression for _, expression in op.collect)
    return found


def functions_used(operations) -> set[str]:
    """İfadelerde geçen fonksiyon adları (ör. ``normcdf``): içe aktarmaları belirler."""

    names: set[str] = set()

    def visit(node: E.Expr) -> None:
        if isinstance(node, E.Call):
            names.add(node.fn)
            for argument in node.args:
                visit(argument)
        elif isinstance(node, E.BinOp):
            visit(node.left)
            visit(node.right)

    for expression in expressions(operations):
        visit(expression)
    return names


def link_of(settings) -> str:
    """Marjinal etkinin bağlantısı: Logit/Probit kendi bağlantısı, OLS doğrusal olasılık modeli."""

    if isinstance(settings, BinaryChoice):
        return settings.link
    return "dogrusal"


def profile_others(op: ProfileCurves, models: dict[str, object]) -> list[str]:
    """Profil tasarımında örneklem ortalamasında tutulan regresörler (sıra korunur)."""

    derived = {name for name, _ in op.derived}
    others: list[str] = []
    for model, _ in op.models:
        for name in models[model].regressors:
            if name != op.variable and name not in derived and name not in others:
                others.append(name)
    return others


def table_row_text(row) -> str:
    """Tablo satır adının metin biçimi (R satır adları, Stata skaler adları için)."""

    if isinstance(row, str):
        return row
    return E.format_number(row)


def coefficient_models(expression: E.Expr) -> list[str]:
    """Bir ifadenin başvurduğu model adları (sırası korunarak)."""

    found: list[str] = []
    if isinstance(expression, (E.Coef, E.StdErr)):
        found.append(expression.model)
    elif isinstance(expression, E.BinOp):
        found.extend(coefficient_models(expression.left))
        found.extend(coefficient_models(expression.right))
    elif isinstance(expression, E.Call):
        for argument in expression.args:
            found.extend(coefficient_models(argument))
    unique: list[str] = []
    for name in found:
        if name not in unique:
            unique.append(name)
    return unique


class Generator:
    """Tek bir hedef dil için kod üreticisi. Alt sınıflar dili tanımlar."""

    language = ""
    comment = "#"

    def __init__(self, spec: LabSpec) -> None:
        self.spec = spec
        self.models = model_settings(spec)
        self.has_checks = any(step.checks for step in spec.steps)
        self.real_data = spec.kind != "sezgi"
        self.p_checks = any(
            isinstance(check.target, CoefTarget) and check.target.quantity == "p"
            for step in spec.steps
            for check in step.checks
        )
        self.quiet = False
        """Monte Carlo döngüsü içinde ekrana yazdırma satırları üretilmez."""

    def is_iv(self, model: str) -> bool:
        return isinstance(self.models.get(model), IV)

    def is_effects(self, model: str) -> bool:
        return isinstance(self.models.get(model), MarginalEffects)

    def needs_sample(self, op: OLS) -> bool:
        """Tahmin örneklemini açıkça oluşturmak gerekir mi?

        Alt grup (``where``) her zaman; küme-dayanıklı SH gerçek veride (eksik değer
        olabilir) gerekir: küme değişkeni, tahmin örneklemiyle satır satır eşleşmelidir.
        """

        return op.where is not None or (op.cluster is not None and self.real_data)

    @staticmethod
    def used_variables(op: OLS) -> list[str]:
        used = [op.outcome, *op.regressors] + ([op.cluster] if op.cluster else [])
        return list(dict.fromkeys(used))

    @staticmethod
    def shown_terms(op: OLS | IV) -> list[str] | None:
        """Kısa çıktı: çok regresörlü modelde yalnız ilk (ilgilenilen) terim; ``None`` hepsi."""

        if isinstance(op, IV):
            return list(op.endogenous)
        if len(op.regressors) > 5:
            return [op.regressors[0]]
        return None

    # --- Alt sınıfların doldurduğu parçalar -------------------------------
    def imports(self, operations: tuple[Operation, ...], *, script: bool = False) -> list[str]:
        return []

    def output_setup(self) -> list[str]:
        """Tam betikte içe aktarmalardan hemen sonra gelen çıktı ayarı (dile özgü; çoğu dilde boş)."""

        return []

    def helpers(self, operations: tuple[Operation, ...], *, with_checks: bool) -> list[str]:
        return []

    def operation(self, op: Operation) -> list[str]:
        raise NotImplementedError

    def check_lines(self, checks: tuple[Check, ...]) -> list[str]:
        raise NotImplementedError

    def closing(self) -> list[str]:
        return []

    def helper_names(self, operations: tuple[Operation, ...]) -> list[str]:
        """İşlemlerin gerektirdiği yardımcı fonksiyonlar (ör. marjinal etkiler, Tobit)."""

        return []

    def helper_code(self, name: str) -> list[str]:
        return []

    def function_helpers(self, operations: tuple[Operation, ...], before: tuple[Operation, ...] = ()) -> list[str]:
        """Yardımcı fonksiyon tanımları; ``before`` işlemlerinde zaten tanımlananlar tekrar yazılmaz."""

        defined = set(self.helper_names(before))
        lines: list[str] = []
        for name in self.helper_names(operations):
            if name not in defined:
                lines.extend(self.helper_code(name))
        return lines

    # --- Ortak yapı -------------------------------------------------------
    def banner(self, text: str) -> list[str]:
        rule = self.comment + " " + "=" * 74
        return [rule, f"{self.comment} {text}", rule]

    def header(self) -> list[str]:
        c = self.comment
        if self.spec.kind == "sezgi":
            return [
                f"{c} IKT 807 Ekonometrik Modelleme ve Uygulamaları",
                f"{c} Konu {self.spec.topic_key[-2:]} sezgi deneyi: {self.spec.title}",
                f"{c} Ders notları §{self.spec.note_section} ile ilişkili kontrollü simülasyon.",
                f"{c}",
                f"{c} Veri bu betikte üretilir; veri üretim süreci (DGP) aşağıda açıkça yazılıdır.",
                f"{c} Rastgele sayı üreteçleri diller arasında farklıdır: Python sürümü uygulamadaki",
                f"{c} sayıların aynısını verir, R ve Stata aynı dağılımdan farklı çekiliş yapar.",
                "",
            ]
        return [
            f"{c} IKT 807 Ekonometrik Modelleme ve Uygulamaları",
            f"{c} Konu {self.spec.topic_key[-2:]} uygulama laboratuvarı: {self.spec.title}",
            f"{c} Ders notları §{self.spec.note_section} ile birebir aynı adımlar.",
            f"{c}",
            f"{c} Veri: Bruce E. Hansen, Econometrics veri arşivi",
            f"{c}       {HANSEN_PAGE_URL}",
            f"{c} Betik sonunda sonuçlar ders notlarındaki basılı değerlerle karşılaştırılır.",
            "",
        ]

    def step_title(self, step: LabStep) -> str:
        word = "Deney" if self.spec.kind == "sezgi" else "Adım"
        return f"{word} {step.number}: {step.title}   ({step.note.label()})"

    def render_operations(self, operations: tuple[Operation, ...]) -> list[str]:
        lines: list[str] = []
        for op in operations:
            block = self.operation(op)
            if block:
                lines.extend(block)
                lines.append("")
        return lines

    def step_snippet(self, number: int) -> str:
        """Uygulama ekranında gösterilen, tek adımlık kod parçası."""

        step = self.spec.step(number)
        lines: list[str] = []
        needs_loader = any(isinstance(op, LoadHansen) for op in step.operations)
        if not step.operations:
            return ""
        lines.extend(self.imports(step.operations))
        if needs_loader:
            lines.extend(self.helpers(step.operations, with_checks=False))
        elif number > 1 and self.spec.kind != "sezgi":
            lines.append(f"{self.comment} Önceki adımlar çalıştırılmış olmalıdır (veri ve modeller hazır).")
            lines.append("")
        lines.extend(self.function_helpers(step.operations, self.spec.operations_through(number - 1)))
        lines.extend(self.render_operations(step.operations))
        return "\n".join(lines).rstrip() + "\n"

    def script(self) -> str:
        """Bütün laboratuvarı tek başına çalışan bir dosya olarak üretir."""

        operations: tuple[Operation, ...] = tuple(
            op for step in self.spec.steps for op in step.operations
        )
        lines = self.header()
        lines.extend(self.imports(operations, script=True))
        lines.extend(self.output_setup())
        lines.extend(self.helpers(operations, with_checks=self.has_checks))
        lines.extend(self.function_helpers(operations))
        for step in self.spec.steps:
            if not step.operations and not step.checks:
                continue
            lines.extend(self.banner(self.step_title(step)))
            lines.append("")
            lines.extend(self.render_operations(step.operations))
            if step.checks:
                lines.extend(self.check_lines(step.checks))
                lines.append("")
        if self.has_checks:
            lines.extend(self.closing())
        return "\n".join(lines).rstrip() + "\n"


def scalar_model(op: Scalar) -> str | None:
    """Tek modelin katsayılarını kullanan ifadede o model; birden çok modelde ``None``."""

    models = coefficient_models(op.expr)
    return models[0] if len(models) == 1 else None


def generator(spec: LabSpec, language: str) -> Generator:
    from core.codegen.python_gen import PythonGenerator
    from core.codegen.r_gen import RGenerator
    from core.codegen.stata_gen import StataGenerator

    classes = {"Python": PythonGenerator, "R": RGenerator, "Stata": StataGenerator}
    if language not in classes:
        raise ValueError(f"Desteklenmeyen dil: {language}")
    return classes[language](spec)


def render_script(spec: LabSpec, language: str) -> str:
    return generator(spec, language).script()


def render_step(spec: LabSpec, number: int, language: str) -> str:
    return generator(spec, language).step_snippet(number)


def script_filename(spec: LabSpec, language: str) -> str:
    return f"ikt807_{spec.topic_key}_uygulama.{LANGUAGE_INFO[language].extension}"

def categorical_comment(op, comment: str) -> str:
    """Kategorik regresörler için doğru açıklama: doymuş model mi, kukla kontroller mi."""

    if op.categorical and all(r in op.categorical for r in op.regressors):
        return f"{comment} Doymuş model: {', '.join(op.categorical)} değişkeninin her değeri için ayrı bir kukla"
    names = ", ".join(op.categorical)
    return f"{comment} {names} kategorik: her düzey için bir kukla, ilk düzey referans"


def continuous_terms(op) -> list[str]:
    return [r for r in op.regressors if r not in op.categorical]
