"""Python kodu: regresyon süreksizliği ve bootstrap işlemleri (Konu 9–10).

Yardımcı fonksiyonlar uygulamanın hesabıyla (``core.labs.rdd``, ``core.labs.resample``) aynı formülleri ve
aynı rastgele çekiliş sırasını kullanır: üretilen betik uygulamanın sayılarını verir. Eşitlik testlerle
denetlenir.
"""

from __future__ import annotations

from core.labs import expr as E
from core.labs.runner import CI_MULTIPLIER, bootstrap_key, uses_replicate_se
from core.labs.spec import BOOT, RDD, Bootstrap, RDDTable, ScalarTable

KERNEL_NAMES = {"triangular": "ucgen", "rectangular": "dikdortgen"}
SCALE_NAMES = {"hansen": "hansen", "window": "pencere"}

RDD_HELPER = [
    'def rdd_yerel_dogrusal(veri, x, y, esik, h, cekirdek="ucgen", olcek="hansen"):',
    '    """Keskin RDD: eşikte yerel doğrusal sıçrama (D katsayısı), ağırlıklı EKK ve HC1 standart hatası.',
    "",
    "    Pozitif ağırlıklı gözlemlerde Y'nin D = 1{X ≥ c}, R = X − c ve D·R üzerine çekirdek ağırlıklı EKK'si.",
    "    Hansen ölçeği: çekirdek birim varyanslıdır ve h çekirdeğin standart sapmasıdır; üçgen çekirdekte ağırlık",
    '    eşikten h√6, dikdörtgende h√3 uzaklıkta sıfırlanır. olcek="pencere": h pencerenin yarı genişliğidir.',
    '    """',
    '    pencere = h * np.sqrt(6 if cekirdek == "ucgen" else 3) if olcek == "hansen" else h',
    "    ornek = veri[[x, y]].dropna()",
    "    r = ornek[x].to_numpy(dtype=float) - esik",
    '    w = np.maximum(1 - np.abs(r) / pencere, 0) if cekirdek == "ucgen" else (np.abs(r) <= pencere) * 1.0',
    "    m = w > 0",
    "    d = (r[m] >= 0) * 1.0",
    '    Z = pd.DataFrame({"Intercept": 1.0, "D": d, "R": r[m], "DR": d * r[m]})',
    '    return sm.WLS(ornek[y].to_numpy(dtype=float)[m], Z, weights=w[m]).fit(cov_type="HC1")',
]

CURVE_HELPER = [
    "def rdd_egrisi(x, y, esik, h, alt, ust, nokta=120):",
    '    """Eşiğin iki yanında ayrı yerel doğrusal tahmin ve noktasal %95 güven bandı.',
    "",
    "    Her x0 noktasında yalnız o taraftaki gözlemlerle, üçgen çekirdekli (pencere ±h√6) ağırlıklı EKK'nin sabit",
    "    terimi: (S2·T0 − S1·T1)/(S0·S2 − S1²). SH o yerel regresyonun HC1 sandviçidir (k/(k−2) çarpanıyla).",
    '    """',
    "    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)",
    "    tamam = ~(np.isnan(x) | np.isnan(y))",
    "    x, y = x[tamam], y[tamam]",
    "    parcalar = []",
    '    for taraf, noktalar, secim in (("sol", np.linspace(alt, esik, nokta), x < esik),',
    '                                   ("sag", np.linspace(esik, ust, nokta), x >= esik)):',
    "        xs, ys = x[secim], y[secim]",
    "        d = xs[None, :] - noktalar[:, None]",
    "        w = np.maximum(1 - np.abs(d) / (h * np.sqrt(6)), 0)",
    "        k = (w > 0).sum(axis=1)",
    "        s0, s1, s2 = w.sum(axis=1), (w * d).sum(axis=1), (w * d * d).sum(axis=1)",
    "        t0, t1 = w @ ys, (w * d) @ ys",
    "        with np.errstate(invalid=\"ignore\", divide=\"ignore\"):",
    "            det = s0 * s2 - s1**2",
    "            a = (s2 * t0 - s1 * t1) / det",
    "            b = (s0 * t1 - s1 * t0) / det",
    "            u = (w * (ys[None, :] - a[:, None] - b[:, None] * d)) ** 2",
    "            v = (s2**2 * u.sum(axis=1) - 2 * s1 * s2 * (u * d).sum(axis=1)",
    "                 + s1**2 * (u * d * d).sum(axis=1)) / det**2 * k / (k - 2)",
    "        a, sh = np.where(k > 2, a, np.nan), np.where(k > 2, np.sqrt(v), np.nan)",
    f'        parcalar.append(pd.DataFrame({{"x": noktalar, "tahmin": a, "alt": a - {CI_MULTIPLIER} * sh,',
    f'                                      "ust": a + {CI_MULTIPLIER} * sh, "taraf": taraf}}))',
    "    return pd.concat(parcalar, ignore_index=True)",
]

HC1_HELPER = [
    "def hc1_sh(X, y, b):",
    '    """HC1 standart hataları: n/(n−k)·(X\'X)⁻¹(Σ eᵢ²xᵢxᵢ\')(X\'X)⁻¹ (percentile-t için her tekrarda)."""',
    "    n, k = X.shape",
    "    ters = np.linalg.inv(X.T @ X)",
    "    skor = X * (y - X @ b)[:, None]",
    "    return np.sqrt(np.diag(n / (n - k) * ters @ (skor.T @ skor) @ ters))",
]

_METHOD_COMMENTS = {
    "pairs": "Pairs bootstrap: gözlem satırları yerine koyarak çekilir; model her tekrarda baştan tahmin edilir",
    "wild": "Wild bootstrap: regresörler sabit; Y* = Xβ̂ + ê·ξ, ξ Rademacher (±1, eşit olasılıkla)",
    "cluster": "Küme bootstrap'ı: kümeler yerine koyarak çekilir; seçilen kümenin bütün gözlemleri birlikte gelir",
}


def _number(value: float) -> str:
    return E.format_number(value)


def rdd_call(op: RDD) -> str:
    arguments = [op.frame, f'"{op.x}"', f'"{op.y}"', _number(op.cutoff), _number(op.bandwidth)]
    if op.kernel != "triangular":
        arguments.append(f'cekirdek="{KERNEL_NAMES[op.kernel]}"')
    if op.scale != "hansen":
        arguments.append(f'olcek="{SCALE_NAMES[op.scale]}"')
    return f"{op.name} = rdd_yerel_dogrusal({', '.join(arguments)})"


def rdd_comment(op: RDD) -> str:
    h = _number(op.bandwidth)
    if op.scale == "window":
        window = f"pencere ±{h} (h pencerenin yarı genişliği)"
    else:
        root = "6" if op.kernel == "triangular" else "3"
        window = f"Hansen ölçeği, pencere ±h√{root}"
    kernel = "üçgen" if op.kernel == "triangular" else "dikdörtgen"
    return f"Keskin RDD, {kernel} çekirdek, h = {h} ({window}); sıçrama D katsayısıdır"


def operation(gen, op) -> list[str] | None:
    """RDD ve bootstrap işlemlerinin Python kodu; tanımadığı işlemde ``None``."""

    if isinstance(op, RDD):
        return [
            f"# {rdd_comment(op)}",
            rdd_call(op),
            f"print(f\"τ̂ = {{{op.name}.params['D']:.4f}} (HC1 SH {{{op.name}.bse['D']:.4f}}), "
            f"n = {{int({op.name}.nobs)}}\")",
        ]
    if isinstance(op, RDDTable):
        return _rdd_table(op)
    if isinstance(op, Bootstrap):
        return _bootstrap(gen, op)
    if isinstance(op, ScalarTable):
        dialect = gen.dialect("")
        lines = [f"{op.result} = pd.Series({{"]
        for label, expression in op.rows:
            lines.append(f'    "{label}": {E.render(expression, dialect)},')
        lines += ['}).to_frame("deger")', f"print({op.result}.round({op.decimals}))"]
        return lines
    return None


def _rdd_table(op: RDDTable) -> list[str]:
    models = ", ".join(model for _, model in op.rows)
    rows = ", ".join(_number(h) for h, _ in op.rows)
    table = op.result
    return [
        f"# Bant genişliği duyarlılığı: her h için sıçrama, HC1 SH ve %95 güven aralığı (tahmin ± {CI_MULTIPLIER}·SH)",
        f"{table} = pd.DataFrame(",
        f'    [(m.nobs, m.params["D"], m.bse["D"]) for m in ({models},)],',
        f'    columns=["n", "tahmin", "sh"], index=pd.Index([{rows}], name="h"),',
        ")",
        f'{table}["alt"] = {table}["tahmin"] - {CI_MULTIPLIER} * {table}["sh"]',
        f'{table}["ust"] = {table}["tahmin"] + {CI_MULTIPLIER} * {table}["sh"]',
        f"print({table}.round(2))",
        "fig, ax = plt.subplots(figsize=(8, 5))",
        f'ax.errorbar({table}.index, {table}["tahmin"], yerr={CI_MULTIPLIER} * {table}["sh"], fmt="o", color="#107C89",',
        '            capsize=4, linewidth=2, label="Tahmin ve %95 güven aralığı")',
        'ax.axhline(0, color="#07373D", linewidth=1, linestyle=":")',
        f'ax.set_xlabel("{op.x_label}")',
        f'ax.set_ylabel("{op.y_label}")',
        f'ax.set_title("{op.title}")',
        "ax.legend()",
        "ax.grid(alpha=0.3)",
        "plt.show()",
    ]


def _dialect(gen) -> E.Dialect:
    """Tekrarın katsayısı ``b[...]``, HC1 SH'si ``s[...]``; diğer modeller orijinal tahmin."""

    base = gen.dialect("")

    def coefficient(model: str, term: str) -> str:
        if model == BOOT:
            return f'b["{gen._key(model, term)}"]'
        return base.coefficient(model, term)

    def standard_error(model: str, term: str) -> str:
        if model == BOOT:
            return f's["{gen._key(model, term)}"]'
        return base.standard_error(model, term)

    return E.Dialect(variable=base.variable, coefficient=coefficient, functions=base.functions, power=base.power,
                     standard_error=standard_error, scalar=base.scalar)


def _bootstrap(gen, op: Bootstrap) -> list[str]:
    model = op.model
    need_se = any(uses_replicate_se(expression) for _, expression in op.collect)
    lines = [
        f"# {op.comment}",
        f"# {_METHOD_COMMENTS[op.method]}",
        f"X_b = {model}.model.exog  # tahmin örnekleminin tasarım matrisi (sabit dahil)",
        f"y_b = {model}.model.endog",
        f"adlar = {model}.model.exog_names",
    ]
    if op.method == "wild":
        lines += [f"uyum = {model}.fittedvalues.to_numpy()", f"artik = {model}.resid.to_numpy()"]
    if op.method == "cluster":
        lines += [
            f'kume_b = {op.frame}.loc[{model}.model.data.row_labels, "{op.cluster}"].to_numpy()',
            "kume_adlari, kume_kodu = np.unique(kume_b, return_inverse=True)",
            "uyeler = [np.flatnonzero(kume_kodu == g) for g in range(len(kume_adlari))]",
        ]
    if op.seed is not None:
        lines.append(f"rng = np.random.default_rng({op.seed})")
    else:
        lines.append("# Aynı rastgele sayı üreteci (rng) veri çekilişlerinin ardından kaldığı yerden devam eder")
    lines += ["tekrarlar = []", f"for tekrar in range({op.reps}):"]
    if op.method == "pairs":
        lines += [
            "    i = rng.integers(0, len(y_b), len(y_b))  # satırları yerine koyarak çek",
            "    X_t, y_t = X_b[i], y_b[i]",
        ]
    elif op.method == "wild":
        lines += ["    X_t, y_t = X_b, uyum + artik * rng.choice([-1.0, 1.0], len(y_b))"]
    else:
        lines += [
            "    secilen = rng.integers(0, len(uyeler), len(uyeler))  # kümeleri yerine koyarak çek",
            "    i = np.concatenate([uyeler[g] for g in secilen])",
            "    X_t, y_t = X_b[i], y_b[i]",
        ]
    lines.append("    b = pd.Series(np.linalg.lstsq(X_t, y_t, rcond=None)[0], index=adlar)")
    if need_se:
        lines.append("    s = pd.Series(hc1_sh(X_t, y_t, b.to_numpy()), index=adlar)")
    dialect = _dialect(gen)
    entries = ", ".join(f'"{name}": {E.render(expression, dialect)}' for name, expression in op.collect)
    lines += [f"    tekrarlar.append({{{entries}}})", f"{op.result} = pd.DataFrame(tekrarlar)"]
    lines.append("# Bootstrap standart hatası (ddof = 1) ve percentile sınırları (yüzde 2,5 ve 97,5)")
    for name, _ in op.collect:
        se, lo, hi = (bootstrap_key(op.result, name, statistic) for statistic in ("se", "lo", "hi"))
        lines += [
            f'{se} = {op.result}["{name}"].std(ddof=1)',
            f'{lo}, {hi} = np.quantile({op.result}["{name}"], [0.025, 0.975])',
            f'print(f"{name}: bootstrap SH = {{{se}:.5f}}, percentile %95 GA [{{{lo}:.4f}}; {{{hi}:.4f}}]")',
        ]
    return lines
