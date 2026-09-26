"""Laboratuvar tanımından Stata do-dosyası üretir.

Bilinçli seçimler:

* ``version 14``: öğrencilerdeki sürüm karışık; bu konular Stata 14 ve üstünde çalışır.
* ``generate double``: Stata'nın varsayılan ``float`` türü yaklaşık 7 anlamlı basamak
  taşır; türetilmiş değişkenler çift hassasiyetle saklanır.
* Kontroller ``kontrol_et`` programıyla yapılır; bir değer tutmazsa do-dosyası durur.
"""

from __future__ import annotations

from core.codegen.base import HANSEN_ARCHIVE_URL, Generator, categorical_comment, continuous_terms, layer_styles, scalar_model
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

_TABSTAT = {"count": "n", "sum": "sum", "mean": "mean", "sd": "sd", "median": "p50", "min": "min", "max": "max"}
_COLLAPSE = {"count": "count", "sum": "sum", "mean": "mean", "sd": "sd", "median": "median", "min": "min", "max": "max"}
_RESULT = {"count": "r(N)", "sum": "r(sum)", "mean": "r(mean)", "sd": "r(sd)", "median": "r(p50)", "min": "r(min)", "max": "r(max)"}
_SYMBOLS = ("O", "S", "T", "D")


def _term(term: str) -> str:
    return "_cons" if term == E.INTERCEPT else term


def _vce(op: OLS) -> str:
    if op.vcov == "classic":
        return ""
    if op.vcov == "HC1":
        return ", vce(robust)"
    return f", vce(cluster {op.cluster})"


class StataGenerator(Generator):
    language = "Stata"
    comment = "*"

    def dialect(self) -> E.Dialect:
        return E.Dialect(
            variable=lambda name: name,
            coefficient=lambda model, term: f"_b[{_term(term)}]",
            functions={"log": "ln", "exp": "exp", "sqrt": "sqrt", "maximum": "max", "minimum": "min", "round": "round", "floor": "floor"},
            power="^",
        )

    def header(self) -> list[str]:
        lines = super().header()
        return lines + [
            "version 14",
            "clear all",
            "set more off",
            "set type double  // yeni değişkenler çift hassasiyetle saklanır",
            "",
        ]

    # --- Yardımcılar -----------------------------------------------------
    def helpers(self, operations: tuple[Operation, ...], *, with_checks: bool) -> list[str]:
        lines: list[str] = []
        if with_checks:
            lines += [
                "capture program drop kontrol_et",
                "program define kontrol_et",
                "    args deger beklenen ondalik etiket",
                "    local tolerans = 0.5 * 10^(-`ondalik') + 1e-12",
                "    if abs(`deger' - `beklenen') > `tolerans' {",
                "        display as error \"  HATA `etiket': \" %12.`ondalik'f `deger' \"  (notlar: `beklenen')\"",
                "        exit 9",
                "    }",
                "    display as text \"  OK   `etiket': \" %12.`ondalik'f `deger' \"  (notlar: `beklenen')\"",
                "end",
                "",
            ]
        return lines

    def _load(self, op: LoadHansen) -> list[str]:
        member = op.member
        found = "if !_rc & `\"`yerel_dosya'\"' == \"\" local yerel_dosya"
        variables = " ".join(op.columns)
        return [
            f"* Dosyayı kendiniz indirdiyseniz yolunu yazın (ör. \"{member}\" veya \"{op.dataset}.dta\");",
            "* boş bırakırsanız veri Hansen'in sayfasından bir kez indirilir ve hansen_veri",
            "* klasörüne açılır. Sonraki çalıştırmalarda aynı klasör yeniden kullanılır.",
            'local yerel_dosya ""',
            "",
            "if `\"`yerel_dosya'\"' == \"\" {",
            "    forvalues deneme = 1/2 {",
            f"        * {member} dosyasını hansen_veri altında iki klasör derinliğe kadar ara",
            f'        capture confirm file "hansen_veri/{member}"',
            f'        {found} "hansen_veri/{member}"',
            '        local ust ""',
            '        capture local ust : dir "hansen_veri" dirs "*"',
            "        foreach k1 of local ust {",
            f'            capture confirm file "hansen_veri/`k1\'/{member}"',
            f'            {found} "hansen_veri/`k1\'/{member}"',
            '            local alt ""',
            "            capture local alt : dir \"hansen_veri/`k1'\" dirs \"*\"",
            "            foreach k2 of local alt {",
            f'                capture confirm file "hansen_veri/`k1\'/`k2\'/{member}"',
            f'                {found} "hansen_veri/`k1\'/`k2\'/{member}"',
            "            }",
            "        }",
            "        if `\"`yerel_dosya'\"' == \"\" & `deneme' == 1 {",
            f'            copy "{HANSEN_ARCHIVE_URL}" ///',
            '                "hansen_veri.zip", replace',
            '            capture mkdir "hansen_veri"',
            '            quietly cd "hansen_veri"',
            '            unzipfile "../hansen_veri.zip", replace',
            '            quietly cd ".."',
            '            erase "hansen_veri.zip"',
            "        }",
            "    }",
            "    if `\"`yerel_dosya'\"' == \"\" {",
            f'        display as error "{member} arşivde bulunamadı. hansen_veri klasöründe dosyayı bulup"',
            '        display as error "yerel_dosya satırına yolunu yazın."',
            "        exit 601",
            "    }",
            "}",
            "",
            f"* Hansen'in .txt dosyasında başlık yoktur; değişkenler açıklama belgesindeki sırayla okunur.",
            "if lower(substr(`\"`yerel_dosya'\"', -4, .)) == \".dta\" {",
            "    use \"`yerel_dosya'\", clear",
            "}",
            "else {",
            f"    infile {variables} ///",
            "        using \"`yerel_dosya'\", clear",
            "}",
            "describe, short",
        ]

    # --- İşlemler --------------------------------------------------------
    def operation(self, op: Operation) -> list[str]:
        if isinstance(op, StandardErrorTable):
            lines = ["* Aynı modeller HC1 ile: katsayılar değişmez, yalnız standart hatalar değişir"]
            names: list[str] = []
            for model in op.models:
                settings = self.models[model]
                regressors = " ".join(f"i.{r}" if r in settings.categorical else r for r in settings.regressors)
                lines += [f"quietly regress {settings.outcome} {regressors}, vce(robust)", f"estimates store {model}_r"]
                names += [model, f"{model}_r"]
            lines.append(f"estimates table {' '.join(names)}, keep({op.term}) b(%9.4f) se(%9.4f) stats(r2)")
            return lines
        if isinstance(op, BreuschPagan):
            return [
                "* Breusch–Pagan (Koenker, n·R²): Python/R ile aynı olması için rhs ve iid seçenekleri",
                f"quietly estimates restore {op.model}",
                "estat hettest, rhs iid",
                f"scalar {op.name}_lm = r(chi2)",
                f"scalar {op.name}_p = r(p)",
            ]
        if isinstance(op, LinearCombination):
            parts = []
            for term, weight in op.weights:
                name = _term(term)
                piece = name if weight == 1 else f"{E.format_number(weight)}*{name}"
                parts.append(piece if not parts else f"+ {piece}")
            return [
                f"* {op.comment}: a'β ve √(a'Va)",
                f"quietly estimates restore {op.model}",
                f"lincom {' '.join(parts)}",
                f"scalar {op.name} = r(estimate)",
                f"scalar {op.name}_se = r(se)",
            ]
        if isinstance(op, DeltaMethod):
            return [
                f"* {op.comment}; nlcom delta yöntemini kendisi uygular",
                f"quietly estimates restore {op.model}",
                f"nlcom ({op.name}: {E.render(op.expr, self.dialect())})",
                "matrix nl_b = r(b)",
                "matrix nl_V = r(V)",
                f"scalar {op.name} = nl_b[1,1]",
                f"scalar {op.name}_se = sqrt(nl_V[1,1])",
            ]
        if isinstance(op, NewSample):
            return [
                "* Sabit tohum: do-dosyası her çalıştırmada aynı veriyi üretir",
                "clear",
                f"set obs {op.nobs}",
                f"set seed {op.seed}",
                "generate long id = _n",
            ]
        if isinstance(op, ClusterDraw):
            a, b = E.format_number(op.first), E.format_number(op.second)
            return [
                f"* {op.comment} (her küme için bir çekiliş)",
                f"sort {op.cluster} id",
                f"by {op.cluster}: generate double {op.name} = rnormal({a}, {b}) if _n == 1",
                f"by {op.cluster}: replace {op.name} = {op.name}[1]",
                "sort id",
            ]
        if isinstance(op, Draw):
            a, b = E.format_number(op.first), E.format_number(op.second)
            call = f"rnormal({a}, {b})" if op.distribution == "normal" else f"runiform({a}, {b})"
            return [f"* {op.comment}", f"generate double {op.name} = {call}"]
        if isinstance(op, Predict):
            option = "xb" if op.kind == "fitted" else "residuals"
            return [f"quietly estimates restore {op.model}", f"predict double {op.name}, {option}"]
        if isinstance(op, Summaries):
            lines = []
            for label, variable, stat in op.rows:
                lines += [
                    f"quietly summarize {variable}",
                    f'display as text "{label}: " as result %18.10f {_RESULT[stat]}',
                ]
            return lines
        if isinstance(op, Plot):
            return self._plot(op)
        if isinstance(op, LoadHansen):
            return self._load(op)
        if isinstance(op, Derive):
            rhs = E.render(op.expr, self.dialect())
            return [f"* {op.comment}", f"generate double {op.name} = {rhs}"]
        if isinstance(op, Describe):
            stats = " ".join(_TABSTAT[s] for s in op.stats)
            return [
                f"tabstat {' '.join(op.variables)}, statistics({stats}) ///",
                "    columns(statistics) format(%9.4f)",
            ]
        if isinstance(op, GroupSummary):
            parts = " ".join(
                f"({_COLLAPSE[stat]}) {name}" if name == variable else f"({_COLLAPSE[stat]}) {name} = {variable}"
                for name, variable, stat in op.columns
            )
            names = " ".join(name for name, _, stat in op.columns if stat != "count")
            lines = [
                "preserve",
                f"collapse {parts}, by({op.by})",
            ]
            if names:
                lines.append(f"format {names} %9.4f")
            lines += ["list, noobs abbreviate(16) separator(0)", "restore"]
            return lines
        if isinstance(op, OLS):
            regressors = " ".join(f"i.{r}" if r in op.categorical else r for r in op.regressors)
            lines = [
                f"quietly regress {op.outcome} {regressors}{_vce(op)}",
                f"estimates store {op.name}",
            ]
            if op.categorical:
                lines.insert(0, categorical_comment(op, "*"))
                shown = continuous_terms(op)
                if shown:
                    lines.append(f"estimates table {op.name}, keep({' '.join(shown)}) b(%9.4f)")
            else:
                lines.append(f"estimates table {op.name}, b(%9.4f)")
            return lines
        if isinstance(op, ShowModel):
            return [f"estimates replay {op.model}"]
        if isinstance(op, RegressionTable):
            keep = " ".join(_term(t) for t in op.terms)
            return [
                f"estimates table {' '.join(op.models)}, b(%9.4f) se(%9.4f) ///",
                f"    stats(N r2) keep({keep})",
            ]
        if isinstance(op, Scalar):
            lines = [f"* {op.comment}"]
            model = scalar_model(op)
            if model:
                lines.append(f"quietly estimates restore {model}")
            rhs = E.render(op.expr, self.dialect())
            lines += [
                f"scalar {op.name} = {rhs}",
                f'display "{op.comment}: " %6.2f scalar({op.name})',
            ]
            return lines
        if isinstance(op, GroupMeanPlot):
            ordered = sorted(op.group_labels)
            layers = " ///\n       ".join(
                f"(connected {op.y} {op.x} if {op.group} == {value}, msymbol({_SYMBOLS[i % 4]}))"
                for i, (value, _) in enumerate(ordered)
            )
            legend = " ".join(f'{i + 1} "{label}"' for i, (_, label) in enumerate(ordered))
            return [
                "preserve",
                f"collapse (mean) {op.y}, by({op.x} {op.group})",
                f"twoway {layers}, ///",
                f"       legend(order({legend})) xtitle(\"{op.x_label}\") ///",
                f"       ytitle(\"{op.y_label}\") title(\"{op.title}\", size(medium))",
                "restore",
            ]
        if isinstance(op, ProjectionPlot):
            return [
                f"quietly estimates restore {op.model}",
                "local b0 = _b[_cons]",
                f"local b1 = _b[{op.x}]",
                "preserve",
                f"collapse (mean) ortalama = {op.y} (count) n = {op.y}, by({op.x})",
                f"twoway (scatter ortalama {op.x} [aweight = n], msymbol(Oh)) ///",
                f"       (function y = `b0' + `b1' * x, range({op.x})), ///",
                '       legend(order(1 "Koşullu ortalama" 2 "OLS doğrusal projeksiyonu")) ///',
                f"       xtitle(\"{op.x_label}\") ytitle(\"{op.y_label}\") ///",
                f"       title(\"{op.title}\", size(medium))",
                "restore",
            ]
        raise TypeError(f"Stata üreticisi bu işlemi tanımıyor: {type(op).__name__}")

    def _plot(self, op: Plot) -> list[str]:
        x = op.x
        grid = E.Dialect(
            variable=lambda name: "x",
            coefficient=lambda model, term: f"_b[{_term(term)}]",
            functions=self.dialect().functions,
            power="^",
        )
        setup: list[str] = []
        layers: list[str] = []
        legend: list[str] = []
        for index, (layer, style) in enumerate(zip(op.layers, layer_styles(op.layers)), start=1):
            color = f'"{style.rgb}"'
            if isinstance(layer, MeanPoints):
                mean, count, first = f"ort_{layer.y}", f"n_{layer.y}", f"ilk_{x}"
                setup += [
                    f"capture drop {mean} {count} {first}",
                    f"bysort {x}: egen double {mean} = mean({layer.y})",
                    f"bysort {x}: egen double {count} = count({layer.y})",
                    f"egen byte {first} = tag({x})",
                ]
                layers.append(f"(scatter {mean} {x} if {first} [aweight = {count}], msymbol(O) mcolor({color}))")
            elif isinstance(layer, Scatter):
                layers.append(f"(scatter {layer.y} {x}, msymbol(p) mcolor(gs10))")
            elif isinstance(layer, Curve):
                layers.append(
                    f"(function y = {E.render(layer.expr, grid)}, range({x}) lcolor({color}) lwidth(medthick))"
                )
            elif isinstance(layer, ModelLine):
                a, b = f"a_{layer.model}", f"b_{layer.model}"
                setup += [
                    f"quietly estimates restore {layer.model}",
                    f"local {a} = _b[_cons]",
                    f"local {b} = _b[{x}]",
                ]
                pattern = " lpattern(dash)" if style.dashed else ""
                layers.append(
                    f"(function y = `{a}' + `{b}' * x, range({x}) lcolor({color}) lwidth(medthick){pattern})"
                )
            else:  # ZeroLine
                layers.append(f"(function y = 0, range({x}) lcolor({color}) lpattern(dot))")
            legend.append(f'{index} "{layer.label}"')
        body = " ///\n       ".join(layers)
        return setup + [
            f"twoway {body}, ///",
            f"       legend(order({' '.join(legend)}) cols(1) position(11) ring(0)) ///",
            f'       xtitle("{op.x_label}") ytitle("{op.y_label}") ///',
            f'       title("{op.title}", size(medium))',
        ]

    # --- Notlarla karşılaştırma -----------------------------------------
    def check_lines(self, checks: tuple[Check, ...]) -> list[str]:
        lines = ['display as text "Notlarla karşılaştırma:"']
        restored: str | None = None
        for check in checks:
            target = check.target
            expected = f"{check.expected:.{check.decimals}f}"
            if isinstance(target, StatTarget):
                condition = ""
                if target.where is not None:
                    variable, value = target.where
                    condition = f" if {variable} == {E.format_number(value)}"
                lines.append(f"quietly summarize {target.variable}{condition}, detail")
                value_expr = _RESULT[target.stat]
                restored = None
            elif isinstance(target, CoefTarget):
                stored = f"{target.model}_r" if target.quantity == "se_hc1" else target.model
                if restored != stored:
                    lines.append(f"quietly estimates restore {stored}")
                    restored = stored
                prefix = "_b" if target.quantity == "coef" else "_se"
                value_expr = f"{prefix}[{_term(target.term)}]"
            elif isinstance(target, ModelTarget):
                if restored != target.model:
                    lines.append(f"quietly estimates restore {target.model}")
                    restored = target.model
                value_expr = "e(r2)" if target.quantity == "r2" else "e(N)"
            elif isinstance(target, ScalarTarget):
                value_expr = f"scalar({target.name})"
            else:
                raise TypeError(f"Tanınmayan hedef: {type(target).__name__}")
            lines.append(
                f'kontrol_et `={value_expr}\' {expected} {check.decimals} "{check.label}"'
            )
        return lines

    def closing(self) -> list[str]:
        return ['display as result _newline "Bütün değerler ders notlarıyla uyuşuyor."']
