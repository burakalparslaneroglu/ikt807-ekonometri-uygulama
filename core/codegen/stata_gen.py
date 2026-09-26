"""Laboratuvar tanımından Stata do-dosyası üretir.

Bilinçli seçimler:

* ``version 14``: öğrencilerdeki sürüm karışık; bu konular Stata 14 ve üstünde çalışır.
* ``generate double``: Stata'nın varsayılan ``float`` türü yaklaşık 7 anlamlı basamak
  taşır; türetilmiş değişkenler çift hassasiyetle saklanır.
* ``ivregress ..., vce(robust) small``: ``small`` olmadan dayanıklı kovaryans n/(n−k)
  ile ölçeklenmez; Python (``debiased=True``) ve R (``vcovHC(type = "HC1")``) ile aynı
  standart hata için gereklidir.
* Monte Carlo sonuçları ``tempfile`` ile toplanır: do-dosyası bitince dosya kendiliğinden
  silinir, çalışma klasöründe (ör. OneDrive) iz bırakmaz.
* Kontroller ``kontrol_et`` programıyla yapılır; bir değer tutmazsa do-dosyası durur.
"""

from __future__ import annotations

from core.codegen.base import (
    HANSEN_ARCHIVE_URL,
    Generator,
    categorical_comment,
    coefficient_models,
    continuous_terms,
    histogram_styles,
    layer_styles,
    profile_others,
    scalar_model,
    table_row_text,
)
from core.labs import expr as E
from core.labs.runner import CI_MULTIPLIER, coverage_key
from core.labs.spec import (
    IV,
    OLS,
    AverageProfile,
    BinaryChoice,
    KeepIf,
    MarginalEffects,
    ProfileCurves,
    QuantileRegression,
    Recode,
    TableTarget,
    Tobit,
    TobitFitCheck,
    TobitTargets,
    BreuschPagan,
    Check,
    ClusterDraw,
    CoefTarget,
    Curve,
    DeltaMethod,
    Derive,
    Describe,
    Draw,
    DropMissing,
    EffectTable,
    GroupMeanPlot,
    GroupSummary,
    Histogram,
    LinearCombination,
    LoadHansen,
    MeanPoints,
    ModelLine,
    ModelTarget,
    MonteCarlo,
    NewSample,
    Operation,
    Plot,
    Predict,
    ProjectionPlot,
    RegressionTable,
    Scalar,
    ScalarTarget,
    Scatter,
    ShowModel,
    StandardErrorTable,
    StatTarget,
    Summaries,
    ZeroLine,
)

_TABSTAT = {"count": "n", "sum": "sum", "mean": "mean", "sd": "sd", "median": "p50", "min": "min", "max": "max"}
_COLLAPSE = {"count": "count", "sum": "sum", "mean": "mean", "sd": "sd", "median": "median", "min": "min", "max": "max"}
_RESULT = {"count": "r(N)", "sum": "r(sum)", "mean": "r(mean)", "sd": "r(sd)", "median": "r(p50)", "min": "r(min)", "max": "r(max)"}
_SYMBOLS = ("O", "S", "T", "D")
_REFERENCE_STYLES = (("7 55 61", "dash", "kesikli"), ("107 76 154", "dot", "noktalı"))
_QUIET_PREFIXES = ("estimates table", "display", "describe", "summarize", "count")


def _term(term: str) -> str:
    """Dilden bağımsız terim adının Stata karşılığı: sabit ve ``değişken=düzey`` (→ ``düzey.değişken``)."""

    if term == E.INTERCEPT:
        return "_cons"
    if "=" in term:
        variable, level = term.split("=", 1)
        return f"{level}.{variable}"
    return term


def _numlist(values) -> str:
    """Stata sayı listesi; ardışık tam sayılar ``a(1)b`` olarak."""

    numbers = [float(v) for v in values]
    if len(numbers) > 2 and all(v.is_integer() for v in numbers) and all(
        b - a == 1 for a, b in zip(numbers, numbers[1:])
    ):
        return f"{int(numbers[0])}(1){int(numbers[-1])}"
    return " ".join(E.format_number(v) for v in numbers)


def _cell(table: str, column: str, row) -> str:
    """Tablo hücresini tutan Stata skalerinin adı (en çok 32 karakter)."""

    text = table_row_text(row)
    if not text.replace("_", "").isalnum() or text.startswith("-"):
        raise ValueError(f"Stata skaler adına uygun olmayan satır: {row}")
    name = f"{table}_{column}_{text}"
    if len(name) > 32:
        raise ValueError(f"Stata skaler adı çok uzun: {name}")
    return name


_FUNCTIONS = {
    "log": "ln", "exp": "exp", "sqrt": "sqrt", "maximum": "max", "minimum": "min",
    "round": "round", "floor": "floor", "positive": "({0} > 0)",
    "logistic": "invlogit", "normcdf": "normal", "normpdf": "normalden",
    **{name: f"({{0}} {symbol} {{1}})" for name, symbol in E.COMPARISONS.items()},
}
_LINE_COLORS = ("16 124 137", "179 57 47", "47 158 107", "7 55 61")


def _vce(op: OLS) -> str:
    if op.vcov == "classic":
        return ""
    if op.vcov == "HC1":
        return ", vce(robust)"
    return f", vce(cluster {op.cluster})"


def _command(text: str, limit: int = 130, width: int = 90) -> list[str]:
    """``limit``ten uzun Stata komutunu ``///`` ile ``width`` genişliğinde satırlara böler."""

    if len(text) <= limit:
        return [text]
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        if current and len(current) + len(word) + 1 > width:
            lines.append(current + " ///")
            current = "    " + word
        else:
            current = f"{current} {word}" if current else word
    lines.append(current)
    return lines


def _scalar_name(prefix: str, model: str, term: str) -> str:
    return f"{prefix}_{model}_{_term(term)}"


def _without_repeated_restores(lines: list[str]) -> list[str]:
    """Arka arkaya aynı modeli yeniden yükleyen ``estimates restore`` satırlarını çıkarır."""

    kept: list[str] = []
    active: str | None = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("quietly estimates restore "):
            model = stripped.rsplit(" ", 1)[-1]
            if model == active:
                continue
            active = model
        elif stripped.startswith((
            "quietly regress", "regress", "quietly ivregress", "ivregress", "logit", "probit", "quietly logit",
            "quietly probit", "tobit", "quietly tobit", "qreg", "quietly qreg", "margins", "quietly margins",
            "nlcom", "quietly nlcom",
        )):
            active = None
        kept.append(line)
    return kept


class StataGenerator(Generator):
    language = "Stata"
    comment = "*"

    def dialect(self) -> E.Dialect:
        return E.Dialect(
            variable=lambda name: name,
            coefficient=lambda model, term: f"_b[{_term(term)}]",
            functions=_FUNCTIONS,
            power="^",
            standard_error=lambda model, term: f"_se[{_term(term)}]",
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

    def _scalar_lines(self, name: str, expression: E.Expr) -> list[str]:
        """``scalar name = ifade``; ifade birden çok modelin katsayısını kullanabilir."""

        models = coefficient_models(expression)
        if len(models) <= 1:
            lines = [f"quietly estimates restore {models[0]}"] if models else []
            return lines + [f"scalar {name} = {E.render(expression, self.dialect())}"]
        lines: list[str] = []
        stored: list[tuple[str, str, str]] = []

        def collect(node: E.Expr) -> None:
            if isinstance(node, (E.Coef, E.StdErr)):
                kind = "b" if isinstance(node, E.Coef) else "se"
                item = (kind, node.model, node.term)
                if item not in stored:
                    stored.append(item)
            elif isinstance(node, E.BinOp):
                collect(node.left)
                collect(node.right)
            elif isinstance(node, E.Call):
                for argument in node.args:
                    collect(argument)

        collect(expression)
        for model in models:
            lines.append(f"quietly estimates restore {model}")
            for kind, owner, term in stored:
                if owner == model:
                    lines.append(f"scalar {_scalar_name(kind, owner, term)} = _{kind}[{_term(term)}]")
        dialect = E.Dialect(
            variable=lambda variable: variable,
            coefficient=lambda model, term: f"scalar({_scalar_name('b', model, term)})",
            functions=self.dialect().functions,
            power="^",
            standard_error=lambda model, term: f"scalar({_scalar_name('se', model, term)})",
        )
        return lines + [f"scalar {name} = {E.render(expression, dialect)}"]

    def _load(self, op: LoadHansen) -> list[str]:
        member = op.member
        found = "if !_rc & `\"`yerel_dosya'\"' == \"\" local yerel_dosya"
        example = f'"{member}"' if member.lower().endswith(".dta") else f'"{member}" veya "{op.dataset}.dta"'
        lines = [
            f"* Dosyayı kendiniz indirdiyseniz yolunu yazın (ör. {example});",
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
        ]
        if member.lower().endswith(".dta"):
            return lines + [
                "* Değişken adları dosyada kayıtlıdır; Python ve R ile aynı olsun diye küçük harfe çevrilir.",
                "* Ders notlarının öğretim CSV'si de yerel dosya olarak verilebilir.",
                "if lower(substr(`\"`yerel_dosya'\"', -4, .)) == \".csv\" {",
                "    import delimited \"`yerel_dosya'\", clear",
                "}",
                "else {",
                "    use \"`yerel_dosya'\", clear",
                "}",
                "capture rename *, lower",
                "describe, short",
            ]
        variables = " ".join(op.columns)
        return lines + [
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
        limited = self._limited_operation(op)
        if limited is not None:
            if self.quiet:
                limited = [line for line in limited if not line.lstrip().startswith(_QUIET_PREFIXES)]
            return limited
        lines = self._operation(op)
        if self.quiet:
            lines = [line for line in lines if not line.lstrip().startswith(_QUIET_PREFIXES)]
        return lines

    def _operation(self, op: Operation) -> list[str]:
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
            if op.seed is None:
                return ["clear", f"set obs {op.nobs}", "generate long id = _n"]
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
            option = "xb" if op.kind in ("fitted", "index") else "residuals"
            lines = ["* Doğrusal indeks x'β (olasılık değil)"] if op.kind == "index" else []
            return lines + [f"quietly estimates restore {op.model}", f"predict double {op.name}, {option}"]
        if isinstance(op, Summaries):
            lines = []
            for label, variable, stat in op.rows:
                detail = ", detail" if stat == "median" else ""
                lines += [
                    f"quietly summarize {variable}{detail}",
                    f'display as text "{label}: " as result %18.10f {_RESULT[stat]}',
                ]
            return lines
        if isinstance(op, Plot):
            return self._plot(op)
        if isinstance(op, LoadHansen):
            return self._load(op)
        if isinstance(op, DropMissing):
            return [
                f"* {op.comment}",
                *_command(f"drop if missing({', '.join(op.variables)})"),
                "count",
            ]
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
            return self._ols(op)
        if isinstance(op, IV):
            return self._iv(op)
        if isinstance(op, EffectTable):
            return self._effect_table(op)
        if isinstance(op, MonteCarlo):
            return self._monte_carlo(op)
        if isinstance(op, Histogram):
            return self._histogram(op)
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
            if scalar_model(op) is not None or not coefficient_models(op.expr):
                model = scalar_model(op)
                if model:
                    lines.append(f"quietly estimates restore {model}")
                lines.append(f"scalar {op.name} = {E.render(op.expr, self.dialect())}")
            else:
                lines += self._scalar_lines(op.name, op.expr)
            width = max(6, op.decimals + 4)
            lines.append(f'display "{op.comment}: " %{width}.{op.decimals}f scalar({op.name})')
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

    # --- Sınırlı bağımlı değişken işlemleri -------------------------------
    def _limited_operation(self, op: Operation) -> list[str] | None:
        if isinstance(op, KeepIf):
            parts = " & ".join(
                f"{variable} {operator} {E.format_number(value)}" for variable, operator, value in op.conditions
            )
            return [f"* {op.comment}", f"keep if {parts}", "count"]
        if isinstance(op, Recode):
            rules = " ".join(
                f"({' '.join(E.format_number(v) for v in values)}={code})" for values, code in op.mapping
            )
            return [
                f"* {op.comment}",
                f"recode {op.source} {rules} (else={op.other}), generate({op.name})",
                f"tabulate {op.name}",
            ]
        if isinstance(op, BinaryChoice):
            regressors = " ".join(f"i.{r}" if r in op.categorical else r for r in op.regressors)
            command = "logit" if op.link == "logit" else "probit"
            vce = ", vce(robust)" if op.vcov == "robust" else ""
            lines = []
            if op.vcov == "robust":
                lines.append("* vce(robust): gözlenen Hessian ile sandviç; Stata ayrıca n/(n−1) ile ölçekler")
            lines += _command(f"quietly {command} {op.outcome} {regressors}{vce}")
            lines.append(f"estimates store {op.name}")
            shown = continuous_terms(op) if op.categorical else list(op.regressors)
            if shown:
                lines.append(f"estimates table {op.name}, keep({' '.join(shown)}) b(%9.4f) se(%9.4f)")
            return lines
        if isinstance(op, MarginalEffects):
            return self._marginal_effects(op)
        if isinstance(op, AverageProfile):
            numbers = _numlist(op.values)
            matrix = f"{op.name}_m"
            return [
                f"* {op.variable} bütün gözlemlerde sırayla aynı değere eşitlenir; diğer değişkenler gözlenen",
                "* değerlerinde kalır (margins, at). Her değerde tahmin edilen olasılıkların ortalaması.",
                f"quietly estimates restore {op.model}",
                f"margins, at({op.variable}=({numbers}))",
                f"matrix {matrix} = r(b)",
                f'marginsplot, noci xtitle("{op.x_label}") ytitle("{op.y_label}") ///',
                f'    title("{op.title}", size(medium))',
                "local j = 0",
                f"foreach deger of numlist {numbers} {{",
                "    local ++j",
                f"    scalar {op.name}_olasilik_`deger' = el({matrix}, 1, `j')",
                "}",
            ]
        if isinstance(op, Tobit):
            regressors = " ".join(op.regressors)
            left = E.format_number(op.left)
            return [
                f"* Tobit: {op.outcome} soldan {left} noktasında sansürlü; MLE",
                *_command(f"quietly tobit {op.outcome} {regressors}, ll({left})"),
                f"estimates store {op.name}",
                f"estimates table {op.name}, keep({op.regressors[0]}) b(%9.4f) se(%9.4f)",
                "* σ: Stata 14 /sigma, Stata 15 ve sonrası var(e.bağımlı) olarak raporlar",
                f"capture scalar {op.name}_sigma = _b[/sigma]",
                f"if _rc != 0 scalar {op.name}_sigma = sqrt(_b[/var(e.{op.outcome})])",
                f'display "sigma: " %9.4f scalar({op.name}_sigma)',
            ]
        if isinstance(op, QuantileRegression):
            regressors = " ".join(op.regressors)
            quantile = E.format_number(op.q)
            return [
                f"* Kantil regresyon, q = {quantile}" + (" (medyan / LAD)" if op.q == 0.5 else ""),
                *_command(f"quietly qreg {op.outcome} {regressors}, quantile({quantile})"),
                f"estimates store {op.name}",
                f"estimates table {op.name}, keep({op.regressors[0]}) b(%9.4f)",
            ]
        if isinstance(op, ProfileCurves):
            return self._profile_curves(op)
        if isinstance(op, TobitTargets):
            sigma = f"scalar({op.model}_sigma)"
            numbers = _numlist(self._profile_values(op.curves))
            return [
                "* Tobit'in üç hedefi: z = x'β/σ; P(Y>0|x) = Φ(z), m(x) = Φ(z)x'β + σφ(z), m#(x) = x'β + σφ(z)/Φ(z)",
                f"foreach g of numlist {numbers} {{",
                f"    scalar t_xb = scalar({op.curves}_{op.model}_`g')",
                f"    scalar t_z = scalar(t_xb) / {sigma}",
                f"    scalar {op.result}_gizli_`g' = scalar(t_xb)",
                f"    scalar {op.result}_p_poz_`g' = normal(scalar(t_z))",
                f"    scalar {op.result}_gozlenen_`g' = normal(scalar(t_z))*scalar(t_xb) + {sigma}*normalden(scalar(t_z))",
                f"    scalar {op.result}_poz_ort_`g' = scalar(t_xb) + {sigma}*normalden(scalar(t_z))/normal(scalar(t_z))",
                f"    display %6.0f `g' %9.3f scalar({op.result}_gizli_`g') %9.3f scalar({op.result}_p_poz_`g') ///",
                f"        %9.3f scalar({op.result}_gozlenen_`g') %9.3f scalar({op.result}_poz_ort_`g')",
                "}",
            ]
        if isinstance(op, TobitFitCheck):
            settings = self.models[op.model]
            left = E.format_number(settings.left)
            sigma = f"scalar({op.model}_sigma)"
            return [
                "* Model kontrolü: Tobit'in ima ettiği P(Y>0) ve E[Y], örneklem üzerinde ortalanır",
                f"quietly estimates restore {op.model}",
                "predict double t_xb, xb",
                f"generate double t_p = normal((t_xb - {left}) / {sigma})",
                f"generate double t_m = {left} + t_p*(t_xb - {left}) + {sigma}*normalden((t_xb - {left}) / {sigma})",
                f"generate double t_poz = {settings.outcome} > {left}",
                "quietly summarize t_p",
                f"scalar {op.result}_model_p_poz = r(mean)",
                "quietly summarize t_m",
                f"scalar {op.result}_model_ortalama = r(mean)",
                "quietly summarize t_poz",
                f"scalar {op.result}_veri_p_poz = r(mean)",
                f"quietly summarize {settings.outcome}",
                f"scalar {op.result}_veri_ortalama = r(mean)",
                f'display "P(Y>0): model " %6.4f scalar({op.result}_model_p_poz) "  veri " %6.4f scalar({op.result}_veri_p_poz)',
                f'display "E[Y]:   model " %8.4f scalar({op.result}_model_ortalama) "  veri " %8.4f scalar({op.result}_veri_ortalama)',
                "drop t_xb t_p t_m t_poz",
            ]
        return None

    def _profile_values(self, name: str):
        for step in self.spec.steps:
            for op in step.operations:
                if isinstance(op, ProfileCurves) and op.result == name:
                    return op.values
        raise ValueError(f"Profil tablosu bulunamadı: {name}")

    def _marginal_effects(self, op: MarginalEffects) -> list[str]:
        settings = self.models[op.model]
        categorical = getattr(settings, "categorical", ())
        lines = [
            "* Ortalama marjinal etkiler (margins): sürekli değişkende türev, kategorik (i.) değişkende referans",
            "* düzeyine göre olasılık farkı; standart hatalar delta yöntemiyle",
            f"quietly estimates restore {op.model}",
        ]
        if op.discrete:
            if any(term not in op.discrete or term in categorical for term in op.terms):
                raise ValueError("Stata: kesikli etkiler ayrı bir MarginalEffects işleminde verilmelidir.")
            for term in op.terms:
                lines += [
                    f"* {term}: türev yerine 1 − 0 farkı (herkes için {term}=1 ve {term}=0 senaryoları)",
                    f"quietly margins, at({term}=0) at({term}=1) post",
                    f"nlcom ({term}: _b[2._at] - _b[1._at]), post",
                ]
            lines.append(f"estimates store {op.name}")
            return lines
        lines += [
            f"margins, dydx({' '.join(op.terms)}) post",
            f"estimates store {op.name}",
        ]
        return lines

    def _profile_curves(self, op: ProfileCurves) -> list[str]:
        others = profile_others(op, self.models)
        numbers = _numlist(op.values)
        models = " ".join(model for model, _ in op.models)

        def terms(variable_text: str) -> str:
            dialect = E.Dialect(
                variable=lambda name: variable_text if name == op.variable else name,
                coefficient=lambda model, term: f"_b[{_term(term)}]",
                functions=_FUNCTIONS,
                power="^",
            )
            pieces = [f"_b[{op.variable}]*{variable_text}"]
            pieces += [f"_b[{name}]*{E.render(expression, dialect)}" for name, expression in op.derived]
            return " + ".join(pieces)

        constant = " + ".join(["_b[_cons]"] + [f"_b[{name}]*scalar(ort_{name})" for name in others])
        low, high, count = op.plot_grid
        step = E.format_number((high - low) / (int(count) - 1))
        lines = [
            f"* Profil: {op.variable} ızgarasında x'β; türetilen terimler ızgaradan, diğer regresörler",
            "* örneklem ortalamasında. OLS/LAD'de x'β tahmin edilen ortalama/medyan, Tobit'te gizli ortalama.",
            *_command(f"foreach v of varlist {' '.join(others)} {{"),
            "    quietly summarize `v'",
            "    scalar ort_`v' = r(mean)",
            "}",
            f"foreach m in {models} {{",
            "    quietly estimates restore `m'",
            *[f"    {line}" for line in _command(f"scalar sabit_`m' = {constant}")],
            f"    foreach g of numlist {numbers} {{",
            *[f"        {line}" for line in _command(f"scalar {op.result}_`m'_`g' = scalar(sabit_`m') + " + terms("`g'"))],
            "    }",
            "}",
            f"foreach g of numlist {numbers} {{",
            "    display %6.0f `g' " + " ".join(f"%9.2f scalar({op.result}_{model}_`g')" for model, _ in op.models),
            "}",
            "* Eğriler: ince ızgarada aynı hesap",
            "preserve",
            "clear",
            f"quietly set obs {int(count)}",
            f"generate double {op.variable} = {E.format_number(low)} + (_n - 1)*{step}",
            f"foreach m in {models} {{",
            "    quietly estimates restore `m'",
            *[f"    {line}" for line in _command(f"generate double egri_`m' = scalar(sabit_`m') + {terms(op.variable)}")],
            "}",
        ]
        layers = [
            f'(line egri_{model} {op.variable}, lcolor("{color}") lwidth(medthick))'
            for (model, _), color in zip(op.models, _LINE_COLORS)
        ]
        legend = " ".join(f'{index} "{label}"' for index, (_, label) in enumerate(op.models, start=1))
        lines += [
            f"twoway {layers[0]} ///",
            *[f"       {layer} ///" for layer in layers[1:]],
            f"       , legend(order({legend})) ///",
            f'       xtitle("{op.x_label}") ytitle("{op.y_label}") title("{op.title}", size(medium))',
            "restore",
        ]
        return lines

    # --- Tahminler -------------------------------------------------------
    def _ols(self, op: OLS) -> list[str]:
        regressors = " ".join(f"i.{r}" if r in op.categorical else r for r in op.regressors)
        condition = ""
        lines: list[str] = []
        if op.where is not None:
            variable, value = op.where
            condition = f" if {variable} == {E.format_number(value)}"
            lines.append(f"* Tahmin örneklemi: {variable} = {E.format_number(value)} olan gözlemler "
                         "(eksik değerli gözlemleri Stata kendisi dışarıda bırakır)")
        lines += _command(f"quietly regress {op.outcome} {regressors}{condition}{_vce(op)}")
        lines.append(f"estimates store {op.name}")
        if op.categorical:
            lines.insert(0, categorical_comment(op, "*"))
            shown = continuous_terms(op)
            if shown:
                lines.append(f"estimates table {op.name}, keep({' '.join(shown)}) b(%9.4f)")
            return lines
        shown = self.shown_terms(op)
        if shown:
            lines.append(f"estimates table {op.name}, keep({' '.join(shown)}) b(%9.4f) se(%9.4f)")
        else:
            lines.append(f"estimates table {op.name}, b(%9.4f)")
        return lines

    def _iv(self, op: IV) -> list[str]:
        exogenous = " ".join(op.exogenous)
        instrumented = f"({' '.join(op.endogenous)} = {' '.join(op.instruments)})"
        command = f"quietly ivregress 2sls {op.outcome} {exogenous} {instrumented}, vce(robust) small"
        command = command.replace("  ", " ")
        return [
            f"* 2SLS: {', '.join(op.endogenous)} içsel, araç {', '.join(op.instruments)}",
            "* small: HC1 ölçeği n/(n−k) → Python (debiased=True) ve R (vcovHC HC1) ile aynı standart hata",
            *_command(command),
            f"estimates store {op.name}",
            f"estimates table {op.name}, keep({' '.join(self.shown_terms(op))}) b(%9.4f) se(%9.4f)",
        ]

    def _effect_table(self, op: EffectTable) -> list[str]:
        models = list(dict.fromkeys(model for _, model, _ in op.rows))
        terms = list(dict.fromkeys(_term(term) for _, _, term in op.rows))
        clustered = any(isinstance(self.models.get(m), OLS) and self.models[m].vcov == "cluster" for m in models)
        distribution = "küme SH'de t(G−1)" if clustered else "t(n−k)"
        normal = [m for m in models if isinstance(self.models.get(m), (MarginalEffects, BinaryChoice, Tobit))]
        lines = [f"* {op.title}" if op.title else "* Tahminler yan yana"]
        lines += [f"*   {model}: {label}" for label, model, _ in op.rows]
        if normal and len(normal) == len(models):
            lines.append("* p-değerleri normal dağılımla (z); notlarla aynı yaklaşım")
        elif normal:
            lines.append(f"* p-değerleri: OLS satırlarında Stata'nın {distribution} dağılımı, marjinal etkilerde normal "
                         "dağılım; notlar normal yaklaşım kullanır")
        else:
            lines.append(f"* p-değerleri Stata'nın kendi dağılımıyla ({distribution}); notlar normal yaklaşım kullanır")
        lines += _command(
            f"estimates table {' '.join(models)}, keep({' '.join(terms)}) b(%9.4f) se(%9.4f) p(%9.3f) stats(N)"
        )
        return lines

    def _monte_carlo(self, op: MonteCarlo) -> list[str]:
        names = [name for name, _ in op.collect]
        lines = [
            f"* {op.comment}",
            f"* {op.reps} tekrar; tohum döngüden önce bir kez ayarlanır. Sonuçlar geçici dosyada",
            "* toplanır (tempfile): do-dosyası bitince silinir, klasörde dosya bırakmaz.",
            f"set seed {op.seed}",
            "tempname sonuc",
            "tempfile mc_dosya",
            f"postfile `sonuc' {' '.join(names)} using \"`mc_dosya'\", replace",
            f"forvalues tekrar = 1/{op.reps} {{",
            "    quietly {",
        ]
        self.quiet = True
        try:
            for inner in op.body:
                lines += [f"        {line}" if line else "" for line in self.operation(inner)]
        finally:
            self.quiet = False
        collected: list[str] = []
        for name, expression in op.collect:
            collected += self._scalar_lines(f"c_{name}", expression)
        lines += [f"        {line}" for line in _without_repeated_restores(collected)]
        lines += [
            f"        post `sonuc' {' '.join(f'(scalar(c_{name}))' for name in names)}",
            "    }",
            "}",
            "postclose `sonuc'",
            "use \"`mc_dosya'\", clear",
            f"summarize {' '.join(names)}",
        ]
        if op.coverage:
            lines.append(f"* %95 güven aralığı: tahmin ± {CI_MULTIPLIER}·SH; gerçek değeri kapsayan tekrarların payı")
        for estimate, standard_error, truth in op.coverage:
            key = coverage_key(op.result, estimate)
            lines += [
                f"generate byte kapsar = abs({estimate} - {E.format_number(truth)}) <= {CI_MULTIPLIER} * {standard_error}",
                "quietly summarize kapsar",
                f"scalar {key} = r(mean)",
                "drop kapsar",
                f'display "Kapsama oranı, {estimate} (gerçek değer {E.format_number(truth)}): " %5.3f scalar({key})',
            ]
        return lines

    def _histogram(self, op: Histogram) -> list[str]:
        lower, upper = E.format_number(op.lower), E.format_number(op.upper)
        width = E.format_number(round((op.upper - op.lower) / op.bins, 10))
        layers = []
        for (column, _), style in zip(op.columns, histogram_styles(len(op.columns))):
            layers.append(
                f"(histogram {column} if inrange({column}, {lower}, {upper}), start({lower}) width({width}) "
                f'frequency fcolor(none) lcolor("{style.rgb}") lwidth(medthick))'
            )
        legend = " ".join(f'{index} "{label}"' for index, (_, label) in enumerate(op.columns, start=1))
        references = []
        notes = []
        for index, (value, label) in enumerate(op.references):
            rgb, pattern, word = _REFERENCE_STYLES[index % len(_REFERENCE_STYLES)]
            references.append(f'xline({E.format_number(value)}, lpattern({pattern}) lcolor("{rgb}"))')
            notes.append(f"{word} çizgi: {label}")
        lines = [
            f"* [{lower}, {upper}] dışındaki değerler çizilmez; kaç tane olduğu aşağıda sayılır",
            "* Stata 14 uyumu için çubuklar içi boş (saydamlık Stata 15 ile gelir)",
            f"twoway {layers[0]} ///",
            *[f"       {layer} ///" for layer in layers[1:]],
            f"       , {' '.join(references)} ///",
            f'       legend(order({legend})) note("{"; ".join(notes)}") ///',
            f'       xtitle("{op.x_label}") ytitle("Tekrar sayısı") title("{op.title}", size(medium))',
        ]
        for column, _ in op.columns:
            lines.append(f"count if !inrange({column}, {lower}, {upper})")
        return lines

    def _plot(self, op: Plot) -> list[str]:
        x = op.x
        stored: list[tuple[str, str]] = []

        def local(model: str, term: str) -> str:
            if (model, term) not in stored:
                stored.append((model, term))
            return f"`k_{model}_{_term(term).replace('.', '_')}'"

        grid = E.Dialect(
            variable=lambda name: "x",
            coefficient=local,
            functions=self.dialect().functions,
            power="^",
        )
        for layer in op.layers:
            if isinstance(layer, Curve):
                E.render(layer.expr, grid)
        setup: list[str] = []
        for model in dict.fromkeys(model for model, _ in stored):
            setup.append(f"quietly estimates restore {model}")
            setup += [
                f"local k_{owner}_{_term(term).replace('.', '_')} = _b[{_term(term)}]"
                for owner, term in stored if owner == model
            ]
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
                pattern = " lpattern(dash)" if style.dashed else ""
                layers.append(
                    f"(function y = {E.render(layer.expr, grid)}, range({x}) lcolor({color}) lwidth(medthick){pattern})"
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
                term = _term(target.term)
                if target.quantity == "coef":
                    value_expr = f"_b[{term}]"
                elif target.quantity == "p":
                    value_expr = f"2*normal(-abs(_b[{term}]/_se[{term}]))"
                else:
                    value_expr = f"_se[{term}]"
            elif isinstance(target, ModelTarget):
                if restored != target.model:
                    lines.append(f"quietly estimates restore {target.model}")
                    restored = target.model
                value_expr = "e(r2)" if target.quantity == "r2" else "e(N)"
            elif isinstance(target, ScalarTarget):
                value_expr = f"scalar({target.name})"
            elif isinstance(target, TableTarget):
                value_expr = f"scalar({_cell(target.table, target.column, target.row)})"
            else:
                raise TypeError(f"Tanınmayan hedef: {type(target).__name__}")
            lines.append(
                f'kontrol_et `={value_expr}\' {expected} {check.decimals} "{check.label}"'
            )
        return lines

    def closing(self) -> list[str]:
        return ['display as result _newline "Bütün değerler ders notlarıyla uyuşuyor."']
