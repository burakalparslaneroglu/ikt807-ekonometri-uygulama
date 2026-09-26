"""Laboratuvar tanımından Python (pandas + statsmodels + linearmodels) kodu üretir."""

from __future__ import annotations

from core.codegen.base import (
    HANSEN_ARCHIVE_URL,
    Generator,
    categorical_comment,
    continuous_terms,
    flatten,
    histogram_styles,
    layer_styles,
)
from core.labs import expr as E
from core.labs.runner import CI_MULTIPLIER, coverage_key
from core.labs.spec import (
    IV,
    OLS,
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

_STAT = {"count": "count", "sum": "sum", "mean": "mean", "sd": "std", "median": "median", "min": "min", "max": "max"}
_REFERENCE_STYLES = (("#07373D", '"--"'), ("#6B4C9A", '":"'))


def _term(term: str) -> str:
    return "Intercept" if term == E.INTERCEPT else term


def _quoted(names) -> list[str]:
    return [f'"{name}"' for name in names]


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


def _formula_pieces(formula: str, width: int = 76) -> list[str]:
    """Uzun formülü `` + `` sınırlarından bölünmüş dize parçalarına ayırır."""

    if len(formula) <= width:
        return [f'"{formula}"']
    head, _, tail = formula.partition(" ~ ")
    terms = tail.split(" + ")
    pieces: list[str] = []
    current = f"{head} ~ {terms[0]}"
    for term in terms[1:]:
        if len(current) + len(term) + 3 > width:
            pieces.append(current)
            current = ""
        current += f" + {term}"
    pieces.append(current)
    return [f'"{piece}"' for piece in pieces]


def _fit_call(name: str, constructor: str, formula: str, data: str, fit: str) -> list[str]:
    single = f'{name} = {constructor}("{formula}", data={data}){fit}'
    limit = 100 if "cov_kwds" in fit else 150
    if len(formula) <= 100 and len(single) <= limit:
        return [single]
    pieces = _formula_pieces(formula)
    return [f"{name} = {constructor}(", *[f"    {piece}" for piece in pieces[:-1]], f"    {pieces[-1]},",
            f"    data={data},", f"){fit}"]


class PythonGenerator(Generator):
    language = "Python"
    comment = "#"

    def dialect(self, frame: str) -> E.Dialect:
        return E.Dialect(
            variable=lambda name: f'{frame}["{name}"]',
            coefficient=lambda model, term: f'{model}.params["{_term(term)}"]',
            functions={
                "log": "np.log", "exp": "np.exp", "sqrt": "np.sqrt", "maximum": "np.maximum",
                "minimum": "np.minimum", "round": "np.rint", "floor": "np.floor",
                "positive": "np.where({0} > 0, 1.0, 0.0)",
            },
            power="**",
            standard_error=self._standard_error,
        )

    def _standard_error(self, model: str, term: str) -> str:
        attribute = "std_errors" if self.is_iv(model) else "bse"
        return f'{model}.{attribute}["{_term(term)}"]'

    # --- Başlık ve yardımcılar ------------------------------------------
    def imports(self, operations: tuple[Operation, ...]) -> list[str]:
        ops = flatten(operations)
        lines: list[str] = []
        if any(isinstance(op, LoadHansen) for op in ops):
            lines += ["import io", "import urllib.request", "import zipfile", ""]
        if any(isinstance(op, (GroupMeanPlot, ProjectionPlot, Plot, Histogram)) for op in ops):
            lines.append("import matplotlib.pyplot as plt")
        lines.append("import numpy as np")
        lines.append("import pandas as pd")
        if any(isinstance(op, OLS) for op in ops):
            lines.append("import statsmodels.formula.api as smf")
        if any(isinstance(op, IV) for op in ops):
            lines.append("from linearmodels.iv import IV2SLS")
        if any(isinstance(op, EffectTable) for op in ops) or self.p_checks:
            lines.append("from scipy import stats")
        if any(isinstance(op, RegressionTable) for op in ops):
            lines.append("from statsmodels.iolib.summary2 import summary_col")
        if any(isinstance(op, BreuschPagan) for op in ops):
            lines.append("from statsmodels.stats.diagnostic import het_breuschpagan")
        lines.append("")
        return lines

    def _loader_text(self) -> list[str]:
        return [
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
        ]

    def _stata_loader_text(self) -> list[str]:
        return [
            "def hansen_verisi(dosya_adi, yerel_dosya=None):",
            '    """Hansen\'in veri arşivinden bir Stata (.dta) dosyasını okur.',
            "",
            "    Değişken adları dosyada kayıtlıdır; Python, R ve Stata'da aynı olsun diye",
            "    küçük harfe çevrilir. Yerel dosya olarak ders notlarının öğretim CSV'si de",
            "    verilebilir.",
            '    """',
            '    if yerel_dosya and str(yerel_dosya).lower().endswith(".csv"):',
            "        veri = pd.read_csv(yerel_dosya)",
            "    elif yerel_dosya:",
            "        veri = pd.read_stata(yerel_dosya)",
            "    else:",
            '        istek = urllib.request.Request(HANSEN_ARSIV, headers={"User-Agent": "Mozilla/5.0"})',
            "        with urllib.request.urlopen(istek, timeout=300) as yanit:",
            "            arsiv = zipfile.ZipFile(io.BytesIO(yanit.read()))",
            '        adlar = [ad for ad in arsiv.namelist() if ad.lower().split("/")[-1] == dosya_adi.lower()]',
            "        if not adlar:",
            '            raise FileNotFoundError(f"{dosya_adi} Hansen arşivinde bulunamadı.")',
            "        veri = pd.read_stata(io.BytesIO(arsiv.read(adlar[0])))",
            "    veri.columns = [ad.lower() for ad in veri.columns]",
            "    return veri",
        ]

    def helpers(self, operations: tuple[Operation, ...], *, with_checks: bool) -> list[str]:
        lines: list[str] = []
        loads = [op for op in operations if isinstance(op, LoadHansen)]
        if loads:
            stata_file = loads[0].member.lower().endswith(".dta")
            example = loads[0].member if stata_file else "cps09mar.txt"
            lines += [
                f'HANSEN_ARSIV = "{HANSEN_ARCHIVE_URL}"',
                "",
                f"# Dosyayı kendiniz indirdiyseniz yolunu yazın (ör. \"{example}\").",
                "# None bırakırsanız veri Hansen'in sayfasından otomatik indirilir.",
                "YEREL_DOSYA = None",
                "",
                "",
                *(self._stata_loader_text() if stata_file else self._loader_text()),
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
        lines = self._operation(op)
        if self.quiet:
            lines = [line for line in lines if not line.lstrip().startswith("print(")]
        return lines

    def _operation(self, op: Operation) -> list[str]:
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
            frame = f'{op.frame} = pd.DataFrame({{"id": np.arange(1, {op.nobs} + 1)}})'
            if op.seed is None:
                return [frame]
            return [
                "# Sabit tohum: betik her çalıştırmada aynı veriyi üretir",
                f"rng = np.random.default_rng({op.seed})",
                frame,
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
            if op.member.lower().endswith(".dta"):
                return [
                    f"# Hansen'in {op.member.rsplit('.', 1)[0]} veri seti; değişken adları dosyada kayıtlı",
                    f'{op.frame} = hansen_verisi("{op.member}", YEREL_DOSYA)',
                    f"print({op.frame}.shape)  # (gözlem, değişken)",
                ]
            return [
                f"# Değişken sırası: Hansen'in {op.dataset} açıklama belgesi",
                *_wrapped_list("sutunlar = [", [f'"{c}"' for c in op.columns], "]"),
                f'{op.frame} = hansen_verisi("{op.member}", sutunlar, YEREL_DOSYA)',
                f'print({op.frame}.shape)  # (gözlem, değişken)',
            ]
        if isinstance(op, DropMissing):
            return [
                f"# {op.comment}",
                *_wrapped_list(f"{op.frame} = {op.frame}.dropna(subset=[", _quoted(op.variables), "]).copy()"),
                f"print(len({op.frame}))  # analiz örneklemi",
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
                f'print(f"{op.comment}: {{{op.name}:.{op.decimals}f}}")',
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

    # --- Tahminler -------------------------------------------------------
    def _ols(self, op: OLS) -> list[str]:
        terms = [f"C({r})" if r in op.categorical else r for r in op.regressors]
        formula = f"{op.outcome} ~ " + " + ".join(terms)
        lines: list[str] = []
        data = op.frame
        if self.needs_sample(op):
            data = f"veri_{op.name}"
            source = op.frame
            if op.where is not None:
                variable, value = op.where
                source = f'{op.frame}[{op.frame}["{variable}"] == {E.format_number(value)}]'
                lines.append(f"# Tahmin örneklemi: {variable} = {E.format_number(value)} olan, eksiksiz gözlemler")
            else:
                lines.append("# Tahmin örneklemi: model ve küme değişkeni eksiksiz gözlemler")
            lines += _wrapped_list(f"{data} = {source}.dropna(subset=[", _quoted(self.used_variables(op)), "])")
        if op.vcov == "classic":
            fit = ".fit()"
        elif op.vcov == "HC1":
            fit = '.fit(cov_type="HC1")'
        else:
            fit = f'.fit(cov_type="cluster", cov_kwds={{"groups": {data}["{op.cluster}"]}})'
        lines += _fit_call(op.name, "smf.ols", formula, data, fit)
        if op.categorical:
            lines.insert(0, categorical_comment(op, "#"))
            shown = continuous_terms(op)
            if shown:
                names = ", ".join(f'"{t}"' for t in shown)
                lines.append(f"print({op.name}.params[[{names}]].round(4))")
            return lines
        shown = self.shown_terms(op)
        if shown:
            lines.append(f"print({op.name}.params[[{', '.join(_quoted(shown))}]].round(4))")
        else:
            lines.append(f"print({op.name}.params.round(4))")
        return lines

    def _iv(self, op: IV) -> list[str]:
        exogenous = " + ".join(("1", *op.exogenous))
        formula = f"{op.outcome} ~ {exogenous} + [{' + '.join(op.endogenous)} ~ {' + '.join(op.instruments)}]"
        comment = (
            f"# 2SLS: {', '.join(op.endogenous)} içsel, araç {', '.join(op.instruments)}; "
            + ("dışsal kontroller iki aşamada da yer alır" if op.exogenous else "yalnız sabit terim dışsal")
        )
        lines = [comment, "# debiased=True: HC1 ölçeği n/(n−k); R ve Stata ile aynı standart hata"]
        lines += _fit_call(op.name, "IV2SLS.from_formula", formula, op.frame, '.fit(cov_type="robust", debiased=True)')
        lines.append(f"print({op.name}.params[[{', '.join(_quoted(self.shown_terms(op)))}]].round(4))")
        return lines

    def _effect_table(self, op: EffectTable) -> list[str]:
        lines = [f"# {op.title}" if op.title else "# Tahminler yan yana", "satirlar = ["]
        for label, model, term in op.rows:
            lines.append(
                f'    ("{label}", {model}.params["{_term(term)}"], {self._standard_error(model, term)}, '
                f"int({model}.nobs)),"
            )
        lines += [
            "]",
            f'{op.result} = pd.DataFrame(satirlar, columns=["etiket", "tahmin", "SH", "N"]).set_index("etiket")',
            "# p-değeri: normal yaklaşımla iki yönlü, 2Φ(−|tahmin/SH|)",
            f'{op.result}.insert(2, "p", 2 * stats.norm.sf(({op.result}["tahmin"] / {op.result}["SH"]).abs()))',
            f"print({op.result}.round(4))",
        ]
        return lines

    def _monte_carlo(self, op: MonteCarlo) -> list[str]:
        lines = [
            f"# {op.comment}",
            f"# {op.reps} tekrar; rastgele sayı üreteci döngüden önce bir kez tohumlanır",
            f"rng = np.random.default_rng({op.seed})",
            "sonuclar = []",
            f"for tekrar in range({op.reps}):",
        ]
        self.quiet = True
        try:
            for inner in op.body:
                lines += [f"    {line}" if line else "" for line in self.operation(inner)]
        finally:
            self.quiet = False
        dialect = self.dialect("")
        lines.append("    sonuclar.append({")
        for name, expression in op.collect:
            lines.append(f'        "{name}": {E.render(expression, dialect)},')
        lines += [
            "    })",
            f"{op.result} = pd.DataFrame(sonuclar)",
            f"print({op.result}.describe().round(4))",
        ]
        if op.coverage:
            lines.append(f"# %95 güven aralığı: tahmin ± {CI_MULTIPLIER}·SH; gerçek değeri kapsayan tekrarların payı")
        for estimate, standard_error, truth in op.coverage:
            key = coverage_key(op.result, estimate)
            lines += [
                f'{key} = (({op.result}["{estimate}"] - {E.format_number(truth)}).abs() '
                f'<= {CI_MULTIPLIER} * {op.result}["{standard_error}"]).mean()',
                f'print(f"Kapsama oranı, {estimate} (gerçek değer {E.format_number(truth)}): {{{key}:.3f}}")',
            ]
        return lines

    def _histogram(self, op: Histogram) -> list[str]:
        lower, upper = E.format_number(op.lower), E.format_number(op.upper)
        columns = [column for column, _ in op.columns]
        lines = [
            f"# [{lower}, {upper}] dışındaki değerler çizilmez; kaç tane olduğu aşağıda yazdırılır",
            f"kutular = np.linspace({lower}, {upper}, {op.bins} + 1)",
            "fig, ax = plt.subplots(figsize=(8, 5))",
        ]
        for (column, label), style in zip(op.columns, histogram_styles(len(op.columns))):
            lines.append(
                f'ax.hist({op.table}["{column}"], bins=kutular, alpha=0.55, color="{style.color}", label="{label}")'
            )
        for index, (value, label) in enumerate(op.references):
            color, pattern = _REFERENCE_STYLES[index % len(_REFERENCE_STYLES)]
            lines.append(
                f'ax.axvline({E.format_number(value)}, color="{color}", linestyle={pattern}, linewidth=2, '
                f'label="{label}")'
            )
        selected = f"{op.table}[[{', '.join(_quoted(columns))}]]"
        lines += [
            f'ax.set_xlabel("{op.x_label}")',
            'ax.set_ylabel("Tekrar sayısı")',
            f'ax.set_title("{op.title}")',
            "ax.legend()",
            "ax.grid(alpha=0.3)",
            "plt.show()",
            f"disarida = (({selected} < {lower}) | ({selected} > {upper})).sum()",
            'print("Aralık dışında kalan değer sayısı:", disarida.to_dict())',
        ]
        return lines

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
                pattern = ', linestyle="--"' if style.dashed else ""
                values = E.render(layer.expr, grid)
                if not E.variables(layer.expr):
                    values = f"np.full_like(izgara, {values})"
                lines.append(
                    f'ax.plot(izgara, {values}, color="{style.color}", '
                    f'linewidth=2{pattern}, label="{layer.label}")'
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
            coefficient = f'{target.model}.params["{_term(target.term)}"]'
            if target.quantity == "coef":
                return coefficient
            if target.quantity == "se":
                return self._standard_error(target.model, target.term)
            if target.quantity == "se_hc1":
                return f'{target.model}.HC1_se["{_term(target.term)}"]'
            if target.quantity == "p":
                return f"2 * stats.norm.sf(abs({coefficient} / {self._standard_error(target.model, target.term)}))"
            raise ValueError(f"Desteklenmeyen katsayı niceliği: {target.quantity}")
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
