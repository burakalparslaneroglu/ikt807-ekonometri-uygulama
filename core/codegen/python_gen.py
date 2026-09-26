"""Laboratuvar tanımından Python (pandas + statsmodels) kodu üretir."""

from __future__ import annotations

from core.codegen.base import HANSEN_ARCHIVE_URL, Generator, categorical_comment, continuous_terms, layer_styles
from core.labs import expr as E
from core.labs.spec import (
    OLS,
    BreuschPagan,
    ClusterDraw,
    DeltaMethod,
    LinearCombination,
    StandardErrorTable,
    Check,
    CoefTarget,
    Curve,
    Derive,
    Describe,
    Draw,
    MeanPoints,
    ModelLine,
    NewSample,
    Plot,
    Predict,
    Scatter,
    Summaries,
    ZeroLine,
    GroupMeanPlot,
    GroupSummary,
    LoadHansen,
    ModelTarget,
    Operation,
    ProjectionPlot,
    RegressionTable,
    Scalar,
    ScalarTarget,
    ShowModel,
    StatTarget,
)

_STAT = {"count": "count", "sum": "sum", "mean": "mean", "sd": "std", "median": "median", "min": "min", "max": "max"}


def _term(term: str) -> str:
    return "Intercept" if term == E.INTERCEPT else term


def _wrapped_list(opening: str, items: list[str], closing: str, width: int = 88) -> list[str]:
    """Uzun listeleri okunur satırlara böler."""

    lines: list[str] = []
    current = opening
    indent = " " * len(opening)
    for index, item in enumerate(items):
        piece = item + (", " if index < len(items) - 1 else "")
        if len(current) + len(piece.rstrip()) > width and current.strip() not in (opening.strip(), ""):
            lines.append(current.rstrip())
            current = indent
        current += piece
    lines.append(current.rstrip() + closing)
    return lines


class PythonGenerator(Generator):
    language = "Python"
    comment = "#"

    def dialect(self, frame: str) -> E.Dialect:
        return E.Dialect(
            variable=lambda name: f'{frame}["{name}"]',
            coefficient=lambda model, term: f'{model}.params["{_term(term)}"]',
            functions={"log": "np.log", "exp": "np.exp", "sqrt": "np.sqrt", "maximum": "np.maximum", "minimum": "np.minimum", "round": "np.rint", "floor": "np.floor"},
            power="**",
        )

    # --- Başlık ve yardımcılar ------------------------------------------
    def imports(self, operations: tuple[Operation, ...]) -> list[str]:
        lines: list[str] = []
        if any(isinstance(op, LoadHansen) for op in operations):
            lines += ["import io", "import urllib.request", "import zipfile", ""]
        if any(isinstance(op, (GroupMeanPlot, ProjectionPlot, Plot)) for op in operations):
            lines.append("import matplotlib.pyplot as plt")
        lines.append("import numpy as np")
        lines.append("import pandas as pd")
        if any(isinstance(op, OLS) for op in operations):
            lines.append("import statsmodels.formula.api as smf")
        if any(isinstance(op, RegressionTable) for op in operations):
            lines.append("from statsmodels.iolib.summary2 import summary_col")
        if any(isinstance(op, BreuschPagan) for op in operations):
            lines.append("from statsmodels.stats.diagnostic import het_breuschpagan")
        lines.append("")
        return lines

    def helpers(self, operations: tuple[Operation, ...], *, with_checks: bool) -> list[str]:
        lines: list[str] = []
        if any(isinstance(op, LoadHansen) for op in operations):
            lines += [
                f'HANSEN_ARSIV = "{HANSEN_ARCHIVE_URL}"',
                "",
                "# Dosyayı kendiniz indirdiyseniz yolunu yazın (ör. \"cps09mar.txt\").",
                "# None bırakırsanız veri Hansen'in sayfasından otomatik indirilir.",
                "YEREL_DOSYA = None",
                "",
                "",
                "def hansen_verisi(dosya_adi, sutunlar, yerel_dosya=None):",
                '    """Hansen\'in veri arşivinden bir dosyayı okur.',
                "",
                "    Hansen'in .txt dosyalarında başlık satırı yoktur ve değerler boşlukla",
                "    ayrılır; değişken adları açıklama belgesindeki sırayla verilir.",
                '    """',
                "    if yerel_dosya:",
                '        if str(yerel_dosya).lower().endswith(".dta"):',
                "            return pd.read_stata(yerel_dosya)",
                '        return pd.read_csv(yerel_dosya, sep=r"\\s+", header=None, names=sutunlar)',
                '    istek = urllib.request.Request(HANSEN_ARSIV, headers={"User-Agent": "Mozilla/5.0"})',
                "    with urllib.request.urlopen(istek, timeout=300) as yanit:",
                "        arsiv = zipfile.ZipFile(io.BytesIO(yanit.read()))",
                "    for ad in arsiv.namelist():",
                '        if ad.lower().split("/")[-1] == dosya_adi:',
                "            with arsiv.open(ad) as dosya:",
                '                return pd.read_csv(dosya, sep=r"\\s+", header=None, names=sutunlar)',
                '    raise FileNotFoundError(f"{dosya_adi} Hansen arşivinde bulunamadı.")',
                "",
                "",
            ]
        if with_checks:
            lines += [
                "def kontrol_et(etiket, deger, beklenen, ondalik=4):",
                '    """Hesaplanan değeri ders notlarındaki basılı değerle karşılaştırır."""',
                "    tolerans = 0.5 * 10 ** (-ondalik) + 1e-12",
                '    durum = "OK  " if abs(deger - beklenen) <= tolerans else "HATA"',
                '    print(f"  {durum} {etiket}: {deger:.{ondalik}f}  (notlar: {beklenen})")',
                '    assert abs(deger - beklenen) <= tolerans, f"{etiket} notlarla uyuşmuyor."',
                "",
                "",
            ]
        return lines

    # --- İşlemler --------------------------------------------------------
    def operation(self, op: Operation) -> list[str]:
        if isinstance(op, StandardErrorTable):
            models = ", ".join(op.models)
            names = ", ".join(f'"{m}"' for m in op.models)
            t = op.term
            return [
                "# Aynı katsayı, iki farklı belirsizlik ölçüsü: klasik ve HC1 standart hata",
                f"{op.result} = pd.DataFrame({{",
                f'    "katsayı": [m.params["{t}"] for m in ({models})],',
                f'    "klasik SH": [m.bse["{t}"] for m in ({models})],',
                f'    "HC1 SH": [m.HC1_se["{t}"] for m in ({models})],',
                f'    "R2": [m.rsquared for m in ({models})],',
                f"}}, index=[{names}])",
                f"print({op.result}.round(4))",
            ]
        if isinstance(op, BreuschPagan):
            return [
                "# Breusch–Pagan (Koenker, n·R²): artık kareleri modelin bütün regresörleriyle",
                f"{op.name}_lm, {op.name}_p, _, _ = het_breuschpagan({op.model}.resid, {op.model}.model.exog)",
                f'print(f"LM = {{{op.name}_lm:.2f}}, p = {{{op.name}_p:.2e}}")',
            ]
        if isinstance(op, LinearCombination):
            weights = ", ".join(f'"{_term(t)}": {E.format_number(w)}' for t, w in op.weights)
            return [
                f"# {op.comment}: a'β ve √(a'Va)",
                f"a = pd.Series({{{weights}}}, dtype=float)",
                f"{op.name} = float(a @ {op.model}.params[a.index])",
                f"{op.name}_se = float(np.sqrt(a @ {op.model}.cov_params().loc[a.index, a.index] @ a))",
                f'print(f"{op.comment}: {{{op.name}:.4f}} (SH {{{op.name}_se:.4f}})")',
            ]
        if isinstance(op, DeltaMethod):
            dialect = self.dialect("")
            coefs = E.coefficients(op.expr)
            gradient = ", ".join(E.render(E.derivative(op.expr, c), dialect) for c in coefs)
            terms = ", ".join(f'"{_term(c.term)}"' for c in coefs)
            return [
                f"# {op.comment}",
                f"{op.name} = {E.render(op.expr, dialect)}",
                "# Delta yöntemi: SH = √(g'(β)' V g'(β))",
                f"turev = np.array([{gradient}])",
                f"V = {op.model}.cov_params().loc[[{terms}], [{terms}]].to_numpy()",
                f"{op.name}_se = float(np.sqrt(turev @ V @ turev))",
                f'print(f"{op.comment}: {{{op.name}:.2f}} (delta SH {{{op.name}_se:.3f}})")',
            ]
        if isinstance(op, NewSample):
            return [
                "# Sabit tohum: betik her çalıştırmada aynı veriyi üretir",
                f"rng = np.random.default_rng({op.seed})",
                f'{op.frame} = pd.DataFrame({{"id": np.arange(1, {op.nobs} + 1)}})',
            ]
        if isinstance(op, ClusterDraw):
            a, b = E.format_number(op.first), E.format_number(op.second)
            return [
                f"# {op.comment} (her küme için bir çekiliş)",
                f"kume_soku = rng.normal({a}, {b}, size={op.groups})",
                f'{op.frame}["{op.name}"] = kume_soku[{op.frame}["{op.cluster}"].astype(int) - 1]',
            ]
        if isinstance(op, Draw):
            a, b = E.format_number(op.first), E.format_number(op.second)
            call = f"rng.normal({a}, {b}, size=len({op.frame}))" if op.distribution == "normal" \
                else f"rng.uniform({a}, {b}, size=len({op.frame}))"
            return [f"# {op.comment}", f'{op.frame}["{op.name}"] = {call}']
        if isinstance(op, Predict):
            attribute = "fittedvalues" if op.kind == "fitted" else "resid"
            return [f'{op.frame}["{op.name}"] = {op.model}.{attribute}']
        if isinstance(op, Summaries):
            lines = [f"{op.result} = pd.Series({{"]
            for label, variable, stat in op.rows:
                lines.append(f'    "{label}": {op.frame}["{variable}"].{_STAT[stat]}(),')
            lines += ["})", f"print({op.result}.round(10))"]
            return lines
        if isinstance(op, Plot):
            return self._plot(op)
        if isinstance(op, LoadHansen):
            return [
                f"# Değişken sırası: Hansen'in {op.dataset} açıklama belgesi",
                *_wrapped_list("sutunlar = [", [f'"{c}"' for c in op.columns], "]"),
                f'{op.frame} = hansen_verisi("{op.member}", sutunlar, YEREL_DOSYA)',
                f'print({op.frame}.shape)  # (gözlem, değişken)',
            ]
        if isinstance(op, Derive):
            rhs = E.render(op.expr, self.dialect(op.frame))
            return [f"# {op.comment}", f'{op.frame}["{op.name}"] = {rhs}']
        if isinstance(op, Describe):
            columns = ", ".join(f'"{name}"' for name in op.variables)
            stats = ", ".join(f'"{_STAT[s]}"' for s in op.stats)
            return [
                f"{op.result} = {op.frame}[[{columns}]].agg([{stats}]).T",
                f"print({op.result}.round(4))",
            ]
        if isinstance(op, GroupSummary):
            lines = [f'{op.result} = {op.frame}.groupby("{op.by}").agg(']
            for name, variable, stat in op.columns:
                lines.append(f'    {name}=("{variable}", "{_STAT[stat]}"),')
            lines += [")", f"print({op.result}.round(4))"]
            return lines
        if isinstance(op, OLS):
            terms = [f"C({r})" if r in op.categorical else r for r in op.regressors]
            formula = f"{op.outcome} ~ " + " + ".join(terms)
            if op.vcov == "classic":
                fit = ".fit()"
            elif op.vcov == "HC1":
                fit = '.fit(cov_type="HC1")'
            else:
                fit = f'.fit(cov_type="cluster", cov_kwds={{"groups": {op.frame}["{op.cluster}"]}})'
            lines = [f'{op.name} = smf.ols("{formula}", data={op.frame}){fit}']
            if op.categorical:
                lines.insert(0, categorical_comment(op, "#"))
                shown = continuous_terms(op)
                if shown:
                    names = ", ".join(f'"{t}"' for t in shown)
                    lines.append(f"print({op.name}.params[[{names}]].round(4))")
            else:
                lines.append(f"print({op.name}.params.round(4))")
            return lines
        if isinstance(op, ShowModel):
            return [f"print({op.model}.summary())"]
        if isinstance(op, RegressionTable):
            models = ", ".join(op.models)
            names = ", ".join(f'"({i})"' for i in range(1, len(op.models) + 1))
            order = ", ".join(f'"{_term(t)}"' for t in op.terms)
            return [
                f"{op.result} = summary_col(",
                f"    [{models}],",
                f"    model_names=[{names}],",
                f"    regressor_order=[{order}],",
                '    float_format="%.4f",',
                '    info_dict={"N": lambda m: f"{int(m.nobs)}"},',
                ")",
                f"print({op.result})",
            ]
        if isinstance(op, Scalar):
            rhs = E.render(op.expr, self.dialect(""))
            return [
                f"# {op.comment}",
                f"{op.name} = {rhs}",
                f'print(f"{op.comment}: {{{op.name}:.2f}}")',
            ]
        if isinstance(op, GroupMeanPlot):
            pairs = ", ".join(f'({value}, "{label}")' for value, label in sorted(op.group_labels))
            return [
                "fig, ax = plt.subplots(figsize=(8, 5))",
                f"for deger, etiket in [{pairs}]:",
                f'    ortalama = {op.frame}[{op.frame}["{op.group}"] == deger].groupby("{op.x}")["{op.y}"].mean()',
                '    ax.plot(ortalama.index, ortalama.values, marker="o", label=etiket)',
                f'ax.set_xlabel("{op.x_label}")',
                f'ax.set_ylabel("{op.y_label}")',
                f'ax.set_title("{op.title}")',
                "ax.legend()",
                "ax.grid(alpha=0.3)",
                "plt.show()",
            ]
        if isinstance(op, ProjectionPlot):
            return [
                f'ortalama = {op.frame}.groupby("{op.x}")["{op.y}"].agg(["mean", "size"])',
                "izgara = np.linspace(ortalama.index.min(), ortalama.index.max(), 100)",
                "fig, ax = plt.subplots(figsize=(8, 5))",
                'ax.scatter(ortalama.index, ortalama["mean"], s=np.sqrt(ortalama["size"]) * 2,',
                '           alpha=0.7, label="Koşullu ortalama")',
                f'ax.plot(izgara, {op.model}.params["Intercept"] + {op.model}.params["{op.x}"] * izgara,',
                '        linewidth=2, label="OLS doğrusal projeksiyonu")',
                f'ax.set_xlabel("{op.x_label}")',
                f'ax.set_ylabel("{op.y_label}")',
                f'ax.set_title("{op.title}")',
                "ax.legend()",
                "ax.grid(alpha=0.3)",
                "plt.show()",
            ]
        raise TypeError(f"Python üreticisi bu işlemi tanımıyor: {type(op).__name__}")

    def _plot(self, op: Plot) -> list[str]:
        frame, x = op.frame, op.x
        grid = E.Dialect(
            variable=lambda name: "izgara",
            coefficient=lambda model, term: f'{model}.params["{_term(term)}"]',
            functions=self.dialect(frame).functions,
            power="**",
        )
        lines = [
            "fig, ax = plt.subplots(figsize=(8, 5))",
            f'izgara = np.linspace({frame}["{x}"].min(), {frame}["{x}"].max(), 200)',
        ]
        for layer, style in zip(op.layers, layer_styles(op.layers)):
            if isinstance(layer, MeanPoints):
                name = f"ort_{layer.y}"
                lines += [
                    f'{name} = {frame}.groupby("{x}")["{layer.y}"].agg(["mean", "size"])',
                    f'ax.scatter({name}.index, {name}["mean"], s=np.sqrt({name}["size"]) * 3,',
                    f'           color="{style.color}", zorder=3, label="{layer.label}")',
                ]
            elif isinstance(layer, Scatter):
                lines.append(
                    f'ax.scatter({frame}["{x}"], {frame}["{layer.y}"], s=6, alpha=0.25, '
                    f'color="{style.color}", label="{layer.label}")'
                )
            elif isinstance(layer, Curve):
                lines.append(
                    f'ax.plot(izgara, {E.render(layer.expr, grid)}, color="{style.color}", '
                    f'linewidth=2, label="{layer.label}")'
                )
            elif isinstance(layer, ModelLine):
                pattern = '"--"' if style.dashed else '"-"'
                lines += [
                    f'ax.plot(izgara, {layer.model}.params["Intercept"] + {layer.model}.params["{x}"] * izgara,',
                    f'        color="{style.color}", linewidth=2, linestyle={pattern}, label="{layer.label}")',
                ]
            elif isinstance(layer, ZeroLine):
                lines.append(
                    f'ax.axhline(0, color="{style.color}", linewidth=1, linestyle=":", label="{layer.label}")'
                )
        lines += [
            f'ax.set_xlabel("{op.x_label}")',
            f'ax.set_ylabel("{op.y_label}")',
            f'ax.set_title("{op.title}")',
            "ax.legend()",
            "ax.grid(alpha=0.3)",
            "plt.show()",
        ]
        return lines

    # --- Notlarla karşılaştırma -----------------------------------------
    def target(self, target) -> str:
        if isinstance(target, StatTarget):
            if target.where is None:
                series = f'{target.frame}["{target.variable}"]'
            else:
                variable, value = target.where
                series = (
                    f'{target.frame}.loc[{target.frame}["{variable}"] == '
                    f'{E.format_number(value)}, "{target.variable}"]'
                )
            return f"{series}.{_STAT[target.stat]}()"
        if isinstance(target, CoefTarget):
            attribute = {"coef": "params", "se": "bse", "se_hc1": "HC1_se"}[target.quantity]
            return f'{target.model}.{attribute}["{_term(target.term)}"]'
        if isinstance(target, ModelTarget):
            return f"{target.model}.rsquared" if target.quantity == "r2" else f"{target.model}.nobs"
        if isinstance(target, ScalarTarget):
            return target.name
        raise TypeError(f"Tanınmayan hedef: {type(target).__name__}")

    def check_lines(self, checks: tuple[Check, ...]) -> list[str]:
        lines = ['print("Notlarla karşılaştırma:")']
        for check in checks:
            expected = f"{check.expected:.{check.decimals}f}"
            lines.append(
                f'kontrol_et("{check.label}", {self.target(check.target)}, {expected}, {check.decimals})'
            )
        return lines

    def closing(self) -> list[str]:
        return ['print("\\nBütün değerler ders notlarıyla uyuşuyor.")']
