"""Laboratuvar tanımından Python, R ve Stata kodu üretmenin ortak çatısı."""

from __future__ import annotations

from dataclasses import dataclass

from core.labs import expr as E
from core.labs.spec import (
    OLS,
    Check,
    Curve,
    LabSpec,
    MeanPoints,
    ModelLine,
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


def layer_styles(layers) -> list[LayerStyle]:
    """Grafik katmanlarının üç dilde ve uygulamada aynı renkleri."""

    styles: list[LayerStyle] = []
    lines = 0
    for layer in layers:
        if isinstance(layer, MeanPoints):
            styles.append(LayerStyle("#107C89", "16 124 137"))
        elif isinstance(layer, Scatter):
            styles.append(LayerStyle("#9AA5A6", "154 165 166"))
        elif isinstance(layer, Curve):
            styles.append(LayerStyle("#07373D", "7 55 61"))
        elif isinstance(layer, ModelLine):
            color, rgb = _LINE_COLORS[lines % len(_LINE_COLORS)]
            styles.append(LayerStyle(color, rgb, dashed=layer.dashed))
            lines += 1
        elif isinstance(layer, ZeroLine):
            styles.append(LayerStyle("#07373D", "7 55 61", dotted=True))
        else:
            raise TypeError(f"Tanınmayan grafik katmanı: {type(layer).__name__}")
    return styles


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


def model_settings(spec: LabSpec) -> dict[str, OLS]:
    """Her model adının son tahmin ayarları (kovaryans türü, küme)."""

    settings: dict[str, OLS] = {}
    for step in spec.steps:
        for op in step.operations:
            if isinstance(op, OLS):
                settings[op.name] = op
    return settings


def coefficient_models(expression: E.Expr) -> list[str]:
    """Bir ifadenin başvurduğu model adları (sırası korunarak)."""

    found: list[str] = []
    if isinstance(expression, E.Coef):
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

    # --- Alt sınıfların doldurduğu parçalar -------------------------------
    def imports(self, operations: tuple[Operation, ...]) -> list[str]:
        return []

    def helpers(self, operations: tuple[Operation, ...], *, with_checks: bool) -> list[str]:
        return []

    def operation(self, op: Operation) -> list[str]:
        raise NotImplementedError

    def check_lines(self, checks: tuple[Check, ...]) -> list[str]:
        raise NotImplementedError

    def closing(self) -> list[str]:
        return []

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
        lines.extend(self.render_operations(step.operations))
        return "\n".join(lines).rstrip() + "\n"

    def script(self) -> str:
        """Bütün laboratuvarı tek başına çalışan bir dosya olarak üretir."""

        operations: tuple[Operation, ...] = tuple(
            op for step in self.spec.steps for op in step.operations
        )
        lines = self.header()
        lines.extend(self.imports(operations))
        lines.extend(self.helpers(operations, with_checks=self.has_checks))
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
    models = coefficient_models(op.expr)
    if len(models) > 1:
        raise ValueError("Bir skaler ifade şimdilik yalnız tek modelin katsayılarını kullanabilir.")
    return models[0] if models else None


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
