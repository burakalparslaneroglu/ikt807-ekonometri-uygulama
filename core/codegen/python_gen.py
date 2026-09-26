"""Laboratuvar tanımından Python (pandas + statsmodels + linearmodels) kodu üretir."""

from __future__ import annotations

from core.codegen.base import (
    HANSEN_ARCHIVE_URL,
    Generator,
    categorical_comment,
    continuous_terms,
    flatten,
    functions_used,
    histogram_styles,
    layer_styles,
    link_of,
    profile_others,
)
from core.codegen import python_np as NP
from core.codegen import python_rdd_boot as RB
from core.labs import expr as E
from core.labs.runner import CI_MULTIPLIER, coverage_key, uses_replicate_se
from core.labs.spec import (
    IV,
    OLS,
    RDD,
    Bootstrap,
    RDDCurve,
    RDDTable,
    VLine,
    BandwidthCV,
    BinMeans,
    CoefficientProfile,
    LocalCurve,
    LocalLinear,
    LocalResidual,
    QuantileDifference,
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

_STAT = {"count": "count", "sum": "sum", "mean": "mean", "sd": "std", "median": "median", "min": "min", "max": "max"}
_FUNCTIONS = {
    "log": "np.log", "exp": "np.exp", "sqrt": "np.sqrt", "abs": "np.abs", "maximum": "np.maximum",
    "minimum": "np.minimum", "round": "np.rint", "floor": "np.floor",
    "positive": "np.where({0} > 0, 1.0, 0.0)",
    "logistic": "expit", "normcdf": "stats.norm.cdf", "normpdf": "stats.norm.pdf",
    "sin": "np.sin", "cos": "np.cos",
    **{name: f"np.where({{0}} {symbol} {{1}}, 1.0, 0.0)" for name, symbol in E.COMPARISONS.items()},
}
_LINK_NAMES = {"logit": "Logit", "probit": "Probit"}
_REFERENCE_STYLES = (("#07373D", '"--"'), ("#6B4C9A", '":"'))


STATA_NUMERIC_TYPES = (
    "# Stata sayıları byte/int/long/float türlerinde saklayabilir; pandas bunları",
    "# int8/int16/int32/float32 okur. Küçük tamsayılar kare gibi işlemlerde uyarı",
    "# vermeden taşar (int8'de 12**2 = -112), float32 hassasiyet kaybettirir.",
    "# Bu yüzden tamsayılar int64'e, float32 sütunlar float64'e çevrilir.",
    'tamsayi = veri.select_dtypes("integer").columns',
    'veri[tamsayi] = veri[tamsayi].astype("int64")',
    'kisa = veri.select_dtypes("float32").columns',
    'veri[kisa] = veri[kisa].astype("float64")',
)
"""Üretilen koddaki .dta okuyucusunun sayı türü düzeltmesi (uygulama da hesabı float64 ile yapar)."""


def _term(term: str) -> str:
    """Dilden bağımsız terim adının statsmodels karşılığı: sabit ve ``değişken=düzey`` kuklaları."""

    if term == E.INTERCEPT:
        return "Intercept"
    if "=" in term:
        variable, level = term.split("=", 1)
        return f"C({variable})[T.{level}]"
    return term


def _row(row) -> str:
    return f'"{row}"' if isinstance(row, str) else E.format_number(row)


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


_MARGINAL_EFFECTS_HELPER = [
    "class Etkiler:",
    '    """Ortalama marjinal etkiler: tahminler (params), delta yöntemi SH\'leri (bse), gözlem sayısı."""',
    "",
    "    def __init__(self, params, bse, nobs):",
    "        self.params, self.bse, self.nobs = params, bse, nobs",
    "",
    "",
    "def ortalama_marjinal_etkiler(model, baglanti, terimler, kesikli=()):",
    '    """AME: sürekli terimde türevin, kategorik terimde referans düzeyine göre olasılık farkının',
    "    örneklem ortalaması. Standart hata delta yöntemiyle, modelin kendi kovaryansıyla: √(∇'V∇).",
    "",
    '    baglanti: "logit", "probit" veya "dogrusal" (doğrusal olasılık modeli: etki = katsayı).',
    '    kesikli: kategorik tanımlanmamış 0/1 değişkenler; bunlarda da türev yerine 1 − 0 farkı alınır.',
    '    """',
    "    X = pd.DataFrame(model.model.exog, columns=model.model.exog_names)",
    "    b = model.params[X.columns].to_numpy()",
    "    V = model.cov_params().loc[X.columns, X.columns].to_numpy()",
    '    if baglanti == "logit":',
    "        G = expit",
    "        g = lambda v: expit(v) * (1 - expit(v))",
    "        g1 = lambda v: g(v) * (1 - 2 * expit(v))",
    '    elif baglanti == "probit":',
    "        G, g = stats.norm.cdf, stats.norm.pdf",
    "        g1 = lambda v: -v * stats.norm.pdf(v)",
    "    else:",
    "        G, g, g1 = (lambda v: v), np.ones_like, np.zeros_like",
    "    M = X.to_numpy()",
    "    tahmin, sh = {}, {}",
    "",
    "    def fark(ad, birler, sifirlar):",
    "        X1, X0 = M.copy(), M.copy()",
    "        X1[:, sifirlar] = 0.0",
    "        X0[:, sifirlar] = 0.0",
    "        X1[:, birler] = 1.0",
    "        z1, z0 = X1 @ b, X0 @ b",
    "        tahmin[ad] = np.mean(G(z1) - G(z0))",
    "        turev = (g(z1)[:, None] * X1 - g(z0)[:, None] * X0).mean(axis=0)",
    "        sh[ad] = np.sqrt(turev @ V @ turev)",
    "",
    "    for terim in terimler:",
    "        duzeyler = [(j, ad.split(\"[T.\")[1].rstrip(\"]\").removesuffix(\".0\"))",
    "                    for j, ad in enumerate(X.columns) if ad.startswith(f\"C({terim})[T.\")]",
    "        if duzeyler:  # kategorik: her düzey, referans düzeyine göre",
    "            grup = [j for j, _ in duzeyler]",
    "            for j, duzey in duzeyler:",
    "                fark(f\"{terim}={duzey}\", [j], grup)",
    "        elif terim in kesikli:  # 0/1 değişken: türev yerine 1 − 0 farkı",
    "            j = list(X.columns).index(terim)",
    "            fark(terim, [j], [j])",
    "        else:  # sürekli: türevin örneklem ortalaması",
    "            j = list(X.columns).index(terim)",
    "            z = M @ b",
    "            tahmin[terim] = np.mean(g(z)) * b[j]",
    "            turev = np.mean(g(z)) * np.eye(len(b))[j] + b[j] * (g1(z)[:, None] * M).mean(axis=0)",
    "            sh[terim] = np.sqrt(turev @ V @ turev)",
    "    return Etkiler(pd.Series(tahmin), pd.Series(sh), int(model.nobs))",
]

_TOBIT_HELPER = [
    "class TobitSonucu:",
    '    """Tobit tahmini: params (β), bse, sigma, sigma_se, nobs, llf; statsmodels benzeri alanlar."""',
    "",
    "    def __init__(self, **alanlar):",
    "        self.__dict__.update(alanlar)",
    "",
    "",
    "def tobit_mle(y, X, sol=0.0):",
    '    """Soldan `sol` noktasında sansürlü Tobit, maksimum olabilirlik.',
    "",
    "    Newton–Raphson, Olsen (1978) parametrelemesiyle: γ = β/σ, θ = 1/σ; bu parametrelemede",
    "    log-olabilirlik içbükeydir. Standart hatalar ters gözlenen bilgi matrisinden (R AER::tobit",
    "    ve Stata tobit ile aynı).",
    '    """',
    "    adlar = list(X.columns)",
    "    x = X.to_numpy(dtype=float)",
    "    yk = np.asarray(y, dtype=float) - sol",
    "    poz = yk > 0",
    "    xp, xs, yp = x[poz], x[~poz], yk[poz]",
    "    b0 = np.linalg.lstsq(x, yk, rcond=None)[0]",
    "    s0 = float(np.std(yk - x @ b0))",
    "    g, t = b0 / s0, 1.0 / s0",
    "",
    "    def ll(g, t):",
    "        return float(np.sum(np.log(t) + stats.norm.logpdf(t * yp - xp @ g)) + np.sum(stats.norm.logcdf(-(xs @ g))))",
    "",
    "    def turevler(g, t):",
    "        r = t * yp - xp @ g",
    "        zs = xs @ g",
    "        oran = np.exp(stats.norm.logpdf(zs) - stats.norm.logcdf(-zs))",
    "        grad = np.concatenate([xp.T @ r - xs.T @ oran, [poz.sum() / t - r @ yp]])",
    "        h_gg = -(xp.T @ xp) - (xs * (oran * (oran - zs))[:, None]).T @ xs",
    "        h_gt = xp.T @ yp",
    "        h_tt = -poz.sum() / t**2 - yp @ yp",
    "        return grad, np.block([[h_gg, h_gt[:, None]], [h_gt[None, :], np.array([[h_tt]])]])",
    "",
    "    deger = ll(g, t)",
    "    for _ in range(200):",
    "        grad, H = turevler(g, t)",
    "        adim = np.linalg.solve(H, -grad)",
    "        boy = 1.0",
    "        while True:  # adım yarılama: log-olabilirlik azalmasın",
    "            g1, t1 = g + boy * adim[:-1], t + boy * adim[-1]",
    "            if t1 > 0:",
    "                yeni = ll(g1, t1)",
    "                if yeni >= deger - 1e-12:",
    "                    break",
    "            boy /= 2.0",
    "        g, t, deger = g1, t1, yeni",
    "        if np.max(np.abs(boy * adim)) < 1e-10:",
    "            break",
    "    _, H = turevler(g, t)",
    "    k = len(adlar)",
    "    J = np.zeros((k + 1, k + 1))  # (γ, θ) → (β, σ) dönüşümünün Jacobian'ı",
    "    J[:k, :k] = np.eye(k) / t",
    "    J[:k, k] = -g / t**2",
    "    J[k, k] = -1.0 / t**2",
    "    V = J @ np.linalg.inv(-H) @ J.T",
    "    sh = np.sqrt(np.diag(V))",
    "    beta = g / t",
    "    beta[0] += sol",
    "    params = pd.Series(beta, index=adlar)",
    "    return TobitSonucu(",
    "        params=params, bse=pd.Series(sh[:k], index=adlar), sigma=1.0 / t, sigma_se=float(sh[k]),",
    "        nobs=len(yk), llf=deger, exog=X, endog=np.asarray(y, dtype=float), sol=sol,",
    "        fittedvalues=pd.Series(x @ beta, index=X.index),",
    "    )",
]


class PythonGenerator(Generator):
    language = "Python"
    comment = "#"

    def dialect(self, frame: str) -> E.Dialect:
        return E.Dialect(
            variable=lambda name: f'{frame}["{name}"]',
            coefficient=lambda model, term: f'{model}.params["{self._key(model, term)}"]',
            functions=_FUNCTIONS,
            power="**",
            standard_error=self._standard_error,
        )

    def _standard_error(self, model: str, term: str) -> str:
        attribute = "std_errors" if self.is_iv(model) else "bse"
        return f'{model}.{attribute}["{self._key(model, term)}"]'

    def _key(self, model: str, term: str) -> str:
        """Etki sonuçlarında terim adı olduğu gibi (``race4=2``); modellerde statsmodels adı."""

        return term if self.is_effects(model) else _term(term)

    # --- Başlık ve yardımcılar ------------------------------------------
    def imports(self, operations: tuple[Operation, ...], *, script: bool = False) -> list[str]:
        ops = flatten(operations)
        functions = functions_used(operations)
        lines: list[str] = []
        standard = ["import sys"] if script else []
        if any(isinstance(op, LoadHansen) for op in ops):
            standard += ["import io", "import urllib.request", "import zipfile"]
        if standard:
            lines += sorted(standard) + [""]
        if any(isinstance(op, (GroupMeanPlot, ProjectionPlot, Plot, Histogram, AverageProfile, ProfileCurves,
                               CoefficientProfile, BandwidthCV, RDDTable))
               for op in ops):
            lines.append("import matplotlib.pyplot as plt")
        lines.append("import numpy as np")
        lines.append("import pandas as pd")
        if any(isinstance(op, RDD) for op in ops):
            lines.append("import statsmodels.api as sm")
        if any(isinstance(op, (OLS, BinaryChoice)) for op in ops):
            lines.append("import statsmodels.formula.api as smf")
        if any(isinstance(op, IV) for op in ops):
            lines.append("from linearmodels.iv import IV2SLS")
        needs_stats = (
            any(isinstance(op, (EffectTable, MarginalEffects, Tobit, TobitTargets, TobitFitCheck, QuantileRegression,
                                QuantileDifference)) for op in ops)
            or self.p_checks or bool(functions & {"normcdf", "normpdf"})
        )
        if needs_stats:
            lines.append("from scipy import stats")
        if any(isinstance(op, QuantileRegression) for op in ops):
            lines.append("from scipy.optimize import linprog")
        if "logistic" in functions or any(isinstance(op, MarginalEffects) for op in ops):
            lines.append("from scipy.special import expit")
        if any(isinstance(op, RegressionTable) for op in ops):
            lines.append("from statsmodels.iolib.summary2 import summary_col")
        if any(isinstance(op, BreuschPagan) for op in ops):
            lines.append("from statsmodels.stats.diagnostic import het_breuschpagan")
        lines.append("")
        return lines

    def output_setup(self) -> list[str]:
        return [
            "# Windows'ta çıktı bir dosyaya ya da başka bir programa yönlendirildiğinde Python yerel kod",
            "# sayfasını (ör. cp1254) kullanır ve τ, β̂ gibi karakterleri yazamaz; çıktı UTF-8 olsun.",
            'if hasattr(sys.stdout, "reconfigure"):',
            '    sys.stdout.reconfigure(encoding="utf-8")',
            "",
        ]

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
            "            veri = pd.read_stata(yerel_dosya, convert_categoricals=False)",
            *["            " + line for line in STATA_NUMERIC_TYPES],
            "            return veri",
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
            "    küçük harfe çevrilir. Değer etiketleri kategoriye dönüştürülmez. Yerel dosya",
            "    olarak ders notlarının öğretim CSV'si de verilebilir.",
            '    """',
            '    if yerel_dosya and str(yerel_dosya).lower().endswith(".csv"):',
            "        veri = pd.read_csv(yerel_dosya)",
            "    elif yerel_dosya:",
            "        veri = pd.read_stata(yerel_dosya, convert_categoricals=False)",
            "    else:",
            '        istek = urllib.request.Request(HANSEN_ARSIV, headers={"User-Agent": "Mozilla/5.0"})',
            "        with urllib.request.urlopen(istek, timeout=300) as yanit:",
            "            arsiv = zipfile.ZipFile(io.BytesIO(yanit.read()))",
            '        adlar = [ad for ad in arsiv.namelist() if ad.lower().split("/")[-1] == dosya_adi.lower()]',
            "        if not adlar:",
            '            raise FileNotFoundError(f"{dosya_adi} Hansen arşivinde bulunamadı.")',
            "        veri = pd.read_stata(io.BytesIO(arsiv.read(adlar[0])), convert_categoricals=False)",
            "    veri.columns = [ad.lower() for ad in veri.columns]",
            *["    " + line for line in STATA_NUMERIC_TYPES],
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

    def helper_names(self, operations: tuple[Operation, ...]) -> list[str]:
        ops = flatten(operations)
        layers = [layer for op in ops if isinstance(op, Plot) for layer in op.layers]
        names: list[str] = []
        if any(isinstance(op, MarginalEffects) for op in ops):
            names.append("marjinal_etkiler")
        if any(isinstance(op, Tobit) for op in ops):
            names.append("tobit")
        if any(isinstance(op, QuantileRegression) for op in ops):
            names.append("kantil")
        if any(isinstance(op, QuantileDifference) for op in ops):
            names.append("kantil_farki")
        if any(isinstance(op, (LocalLinear, LocalResidual)) for op in ops) or any(
            isinstance(layer, LocalCurve) for layer in layers
        ):
            names.append("yerel")
        if any(isinstance(op, BandwidthCV) for op in ops):
            names.append("cv")
        if any(isinstance(layer, BinMeans) for layer in layers):
            names.append("aralik")
        if any(isinstance(op, RDD) for op in ops):
            names.append("rdd")
        if any(isinstance(layer, RDDCurve) for layer in layers):
            names.append("rdd_egri")
        if any(isinstance(op, Bootstrap) and any(uses_replicate_se(e) for _, e in op.collect) for op in ops):
            names.append("hc1")
        return names

    def helper_code(self, name: str) -> list[str]:
        texts = {
            "marjinal_etkiler": _MARGINAL_EFFECTS_HELPER,
            "tobit": _TOBIT_HELPER,
            "kantil": NP.QUANTILE_HELPER,
            "kantil_farki": NP.DIFFERENCE_HELPER,
            "yerel": NP.LOCAL_HELPER,
            "cv": NP.CV_HELPER,
            "aralik": NP.BINS_HELPER,
            "rdd": RB.RDD_HELPER,
            "rdd_egri": RB.CURVE_HELPER,
            "hc1": RB.HC1_HELPER,
        }
        return texts[name] + ["", ""] if name in texts else []

    # --- İşlemler --------------------------------------------------------
    def operation(self, op: Operation) -> list[str]:
        lines = self._operation(op)
        if self.quiet:
            lines = [line for line in lines if not line.lstrip().startswith("print(")]
        return lines

    def _operation(self, op: Operation) -> list[str]:
        nonparametric = NP.operation(self, op)
        if nonparametric is not None:
            return nonparametric
        resampling = RB.operation(self, op)
        if resampling is not None:
            return resampling
        limited = self._limited_operation(op)
        if limited is not None:
            return limited
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
            if op.kind == "index":
                return [f"# Doğrusal indeks x'β (olasılık değil)", f'{op.frame}["{op.name}"] = {op.model}.fittedvalues']
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

    # --- Sınırlı bağımlı değişken işlemleri -------------------------------
    def _limited_operation(self, op: Operation) -> list[str] | None:
        if isinstance(op, KeepIf):
            parts = " & ".join(
                f'({op.frame}["{variable}"] {operator} {E.format_number(value)})'
                for variable, operator, value in op.conditions
            )
            return [
                f"# {op.comment}",
                f"{op.frame} = {op.frame}[{parts}].copy()",
                f"print(len({op.frame}))  # analiz örneklemi",
            ]
        if isinstance(op, Recode):
            conditions = []
            for values, _ in op.mapping:
                if len(values) == 1:
                    conditions.append(f'{op.frame}["{op.source}"] == {E.format_number(values[0])}')
                else:
                    listed = ", ".join(E.format_number(v) for v in values)
                    conditions.append(f'{op.frame}["{op.source}"].isin([{listed}])')
            codes = ", ".join(str(code) for _, code in op.mapping)
            return [
                f"# {op.comment}",
                *_wrapped_list(f'{op.frame}["{op.name}"] = np.select([', conditions, "],"),
                f"    [{codes}], default={op.other})",
                f'print({op.frame}["{op.name}"].value_counts().sort_index())',
            ]
        if isinstance(op, BinaryChoice):
            return self._binary(op)
        if isinstance(op, MarginalEffects):
            settings = self.models[op.model]
            terms = ", ".join(f'"{t}"' for t in op.terms)
            call = f'{op.name} = ortalama_marjinal_etkiler({op.model}, "{link_of(settings)}", [{terms}]'
            if op.discrete:
                call += f", kesikli=[{', '.join(_quoted(op.discrete))}]"
            lines = [
                "# Ortalama marjinal etkiler: sürekli değişkende türev, kategorik değişkende referans düzeyine",
                "# göre olasılık farkı; standart hatalar delta yöntemiyle, modelin kovaryansıyla",
                call + ")",
                f'print(pd.DataFrame({{"AME": {op.name}.params, "SH": {op.name}.bse}}).round(4))',
            ]
            return lines
        if isinstance(op, AverageProfile):
            values = ", ".join(E.format_number(v) for v in op.values)
            data = self.models[op.model].frame
            return [
                f"# {op.variable} bütün gözlemlerde sırayla aynı değere eşitlenir; diğer değişkenler gözlenen",
                "# değerlerinde kalır. Her değerde tahmin edilen olasılıkların ortalaması alınır.",
                f"degerler = [{values}]",
                "olasiliklar = []",
                "for deger in degerler:",
                f"    kopya = {data}.copy()",
                f'    kopya["{op.variable}"] = deger',
                f"    olasiliklar.append({op.model}.predict(kopya).mean())",
                f'{op.name} = pd.DataFrame({{"olasilik": olasiliklar}}, index=degerler)',
                f"print({op.name}.round(4))",
                "fig, ax = plt.subplots(figsize=(8, 5))",
                f'ax.plot({op.name}.index, {op.name}["olasilik"], marker="o", color="#107C89")',
                f'ax.set_xlabel("{op.x_label}")',
                f'ax.set_ylabel("{op.y_label}")',
                f'ax.set_title("{op.title}")',
                "ax.grid(alpha=0.3)",
                "plt.show()",
            ]
        if isinstance(op, Tobit):
            regressors = f"regresorler_{op.name}"
            return [
                f"# Tobit: {op.outcome} soldan {E.format_number(op.left)} noktasında sansürlü; MLE (tobit_mle)",
                *_wrapped_list(f"{regressors} = [", _quoted(op.regressors), "]"),
                f"tasarim = {op.frame}[{regressors}].copy()",
                'tasarim.insert(0, "Intercept", 1.0)',
                f'{op.name} = tobit_mle({op.frame}["{op.outcome}"], tasarim, sol={E.format_number(op.left)})',
                f'print({op.name}.params[{self._shown(op)}].round(4), "sigma:", round({op.name}.sigma, 4))',
            ]
        if isinstance(op, ProfileCurves):
            return self._profile_curves(op)
        if isinstance(op, TobitTargets):
            latent = f'{op.curves}["{op.model}"]'
            sigma = f"{op.model}.sigma"
            return [
                "# Tobit'in üç hedefi: z = x'β/σ; P(Y>0|x) = Φ(z), m(x) = Φ(z)x'β + σφ(z), m#(x) = x'β + σφ(z)/Φ(z)",
                f"z = {latent} / {sigma}",
                f"{op.result} = pd.DataFrame({{",
                f'    "gizli": {latent},',
                '    "p_poz": stats.norm.cdf(z),',
                f'    "gozlenen": stats.norm.cdf(z) * {latent} + {sigma} * stats.norm.pdf(z),',
                f'    "poz_ort": {latent} + {sigma} * stats.norm.pdf(z) / stats.norm.cdf(z),',
                "})",
                f"print({op.result}.round(3))",
            ]
        if isinstance(op, TobitFitCheck):
            model = op.model
            return [
                "# Model kontrolü: Tobit'in ima ettiği P(Y>0) ve E[Y], örneklem üzerinde ortalanır",
                f"xb = {model}.exog.to_numpy() @ {model}.params.to_numpy()",
                f"z = (xb - {model}.sol) / {model}.sigma",
                f"{op.result} = pd.DataFrame({{",
                f'    "model": [stats.norm.cdf(z).mean(), np.mean({model}.sol + stats.norm.cdf(z) * (xb - {model}.sol)'
                f" + {model}.sigma * stats.norm.pdf(z))],",
                f'    "veri": [np.mean({model}.endog > {model}.sol), np.mean({model}.endog)],',
                '}, index=["p_poz", "ortalama"])',
                f"print({op.result}.round(4))",
            ]
        return None

    def _shown(self, op) -> str:
        shown = [op.regressors[0]] if len(op.regressors) > 5 else list(op.regressors)
        return "[" + ", ".join(f'"{name}"' for name in ["Intercept", *shown]) + "]"

    def _binary(self, op: BinaryChoice) -> list[str]:
        terms = [f"C({r})" if r in op.categorical else r for r in op.regressors]
        formula = f"{op.outcome} ~ " + " + ".join(terms)
        if op.vcov == "robust":
            comment = (f"# {_LINK_NAMES[op.link]} (MLE); dayanıklı kovaryans: HC0 = gözlenen Hessian ile "
                       "sandviç H⁻¹(Σsᵢsᵢ')H⁻¹")
            fit = '.fit(disp=False, cov_type="HC0")'
        else:
            comment = f"# {_LINK_NAMES[op.link]} (MLE); klasik kovaryans: ters gözlenen bilgi matrisi"
            fit = ".fit(disp=False)"
        builder = "smf.logit" if op.link == "logit" else "smf.probit"
        lines = [comment, *_fit_call(op.name, builder, formula, op.frame, fit)]
        shown = continuous_terms(op) if op.categorical else list(op.regressors)
        if shown:
            lines.append(f"print({op.name}.params[[{', '.join(_quoted(shown))}]].round(4))")
        return lines

    def _profile_curves(self, op: ProfileCurves) -> list[str]:
        others = profile_others(op, self.models)
        dialect = self.dialect("P")
        values = ", ".join(E.format_number(v) for v in op.values)
        pairs = ", ".join(f'("{model}", {model})' for model, _ in op.models)
        lines = [
            f"# Profil: {op.variable} ızgarasında x'β; türetilen terimler ızgaradan, diğer regresörler",
            "# örneklem ortalamasında. OLS/LAD'de x'β tahmin edilen ortalama/medyan, Tobit'te gizli ortalama.",
            "def profil_tasarimi(degerler):",
            f'    P = pd.DataFrame({{"{op.variable}": np.asarray(degerler, dtype=float)}})',
        ]
        for name, expression in op.derived:
            lines.append(f'    P["{name}"] = {E.render(expression, dialect)}')
        lines += _wrapped_list("    for ad in [", _quoted(others), "]:")
        lines += [
            f"        P[ad] = {op.frame}[ad].mean()",
            '    P.insert(0, "Intercept", 1.0)',
            "    return P",
            "",
            "",
            "def dogrusal_indeks(model, P):",
            "    return P[model.params.index].to_numpy() @ model.params.to_numpy()",
            "",
            "",
            f"degerler = [{values}]",
            "izgara = profil_tasarimi(degerler)",
            f"{op.result} = pd.DataFrame({{ad: dogrusal_indeks(m, izgara) for ad, m in [{pairs}]}}, index=degerler)",
            f"print({op.result}.round(2))",
            f"ince = profil_tasarimi(np.linspace({E.format_number(op.plot_grid[0])}, "
            f"{E.format_number(op.plot_grid[1])}, {int(op.plot_grid[2])}))",
            "fig, ax = plt.subplots(figsize=(8, 5))",
        ]
        colors = ("#107C89", "#B3392F", "#2F9E6B", "#07373D")
        for (model, label), color in zip(op.models, colors):
            lines.append(
                f'ax.plot(ince["{op.variable}"], dogrusal_indeks({model}, ince), color="{color}", linewidth=2, '
                f'label="{label}")'
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
                f'    ("{label}", {model}.params["{self._key(model, term)}"], {self._standard_error(model, term)}, '
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
            coefficient=lambda model, term: f'{model}.params["{self._key(model, term)}"]',
            functions=self.dialect(frame).functions,
            power="**",
        )
        if op.x_range is None:
            low, high = f'{frame}["{x}"].min()', f'{frame}["{x}"].max()'
        else:
            low, high = (E.format_number(value) for value in op.x_range)
        lines = ["fig, ax = plt.subplots(figsize=(8, 5))"]
        if any(isinstance(layer, (Curve, ModelLine, LocalCurve)) for layer in op.layers):
            lines.append(f"izgara = np.linspace({low}, {high}, 200)")
        curves = 0
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
            elif isinstance(layer, LocalCurve):
                pattern = ', linestyle="--"' if style.dashed else ""
                degree = "" if layer.degree == 1 else f", derece={layer.degree}"
                lines += [
                    f'ax.plot(izgara, yerel_dogrusal({frame}["{x}"], {frame}["{layer.y}"], izgara, '
                    f'{E.format_number(layer.bandwidth)}{degree}),',
                    f'        color="{style.color}", linewidth=2{pattern}, label="{layer.label}")',
                ]
            elif isinstance(layer, BinMeans):
                name = f"aralik_{layer.y}"
                lines += [
                    f'{name} = aralik_ortalamalari({frame}["{x}"], {frame}["{layer.y}"], {layer.bins})',
                    f'ax.scatter({name}["x"], {name}["y"], s=28, color="{style.color}", zorder=3,',
                    f'           label="{layer.label}")',
                ]
            elif isinstance(layer, RDDCurve):
                curves += 1
                name = f"egri_{curves}"
                points = "" if layer.points == 120 else f", nokta={layer.points}"
                lines += [
                    "# Eşiğin iki yanında ayrı yerel doğrusal tahmin (üçgen çekirdek, pencere ±h√6) ve %95 bant",
                    f'{name} = rdd_egrisi({frame}["{x}"], {frame}["{layer.y}"], {E.format_number(layer.cutoff)}, '
                    f"{E.format_number(layer.bandwidth)}, {low}, {high}{points})",
                    f'for taraf, parca in {name}.groupby("taraf", sort=False):',
                    f'    ax.fill_between(parca["x"], parca["alt"], parca["ust"], color="{style.color}", alpha=0.18,',
                    "                    linewidth=0)",
                    f'    ax.plot(parca["x"], parca["tahmin"], color="{style.color}", linewidth=2,',
                    f'            label="{layer.label}" if taraf == "sol" else None)',
                ]
            elif isinstance(layer, VLine):
                lines.append(
                    f'ax.axvline({E.format_number(layer.x)}, color="{style.color}", linestyle="-.", linewidth=1.5, '
                    f'label="{layer.label}")'
                )
        if op.x_range is not None:
            lines.append(f"ax.set_xlim({low}, {high})")
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
            coefficient = f'{target.model}.params["{self._key(target.model, target.term)}"]'
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
        if isinstance(target, TableTarget):
            return f'{target.table}.loc[{_row(target.row)}, "{target.column}"]'
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
