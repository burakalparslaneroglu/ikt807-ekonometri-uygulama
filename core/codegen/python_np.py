"""Python kodu: kantil regresyon ve parametrik olmayan regresyon işlemleri (Konu 7–8).

Yardımcı fonksiyonlar uygulamanın hesabıyla (``core.labs.quantreg``, ``core.labs.smoothing``)
aynı formülleri kullanır; eşitlik testlerle denetlenir.
"""

from __future__ import annotations

from core.labs import expr as E
from core.labs.runner import CI_MULTIPLIER
from core.labs.smoothing import bandwidth_grid
from core.labs.spec import (
    BandwidthCV,
    CoefficientProfile,
    LocalLinear,
    LocalResidual,
    QuantileDifference,
    QuantileRegression,
)

QUANTILE_HELPER = [
    "class KantilSonucu:",
    '    """Kantil regresyon sonucu: params, bse, q, nobs; Hendricks–Koenker parçaları Hinv ve J."""',
    "",
    "    def __init__(self, **alanlar):",
    "        self.__dict__.update(alanlar)",
    "",
    "",
    "def kantil_regresyonu(veri, bagimli, regresorler, tau, sh=True):",
    '    """Kantil regresyon: doğrusal programlamanın kesin çözümü (R rq ve Stata qreg ile aynı).',
    "",
    "    Dual problem: en büyük y'a, X'a = (1 − τ)X'1, 0 ≤ a ≤ 1; β eşitlik kısıtlarının gölge",
    "    fiyatlarıdır. statsmodels QuantReg yinelemeli bir yaklaşım kullanır ve büyük örneklemde",
    "    beşinci ondalıkta farklılaşabilir.",
    "    sh=True: Hendricks–Koenker standart hataları (R summary.rq, se = \"nid\"): koşullu yoğunluk",
    "    her gözlemde τ ± h kantil doğrularının farkından tahmin edilir; h Hall–Sheather bandıdır.",
    '    """',
    "    ornek = veri[[bagimli, *regresorler]].dropna()",
    "    X = ornek[list(regresorler)].astype(float)",
    '    X.insert(0, "Intercept", 1.0)',
    "    x, y = X.to_numpy(), ornek[bagimli].to_numpy(dtype=float)",
    "",
    "    def coz(t):",
    '        r = linprog(-y, A_eq=x.T, b_eq=(1 - t) * x.sum(axis=0), bounds=(0, 1), method="highs")',
    "        return -r.eqlin.marginals",
    "",
    "    sonuc = KantilSonucu(",
    "        params=pd.Series(coz(tau), index=X.columns), bse=pd.Series(np.nan, index=X.columns),",
    "        q=tau, nobs=len(y), exog=X, endog=y,",
    "    )",
    "    if sh:",
    "        # Hall–Sheather bant genişliği (τ ölçeğinde); τ ± h aralığı (0, 1) dışına taşarsa yarıya indirilir",
    "        z = stats.norm.ppf(tau)",
    "        h = (len(y) ** (-1 / 3) * stats.norm.ppf(0.975) ** (2 / 3)",
    "             * (1.5 * stats.norm.pdf(z) ** 2 / (2 * z**2 + 1)) ** (1 / 3))",
    "        while tau - h < 0 or tau + h > 1:",
    "            h /= 2",
    "        fark = x @ (coz(tau + h) - coz(tau - h))",
    "        f = np.maximum(0, 2 * h / (fark - np.finfo(float).eps ** 0.5))",
    "        sonuc.Hinv = np.linalg.inv(x.T @ (f[:, None] * x))",
    "        sonuc.J = x.T @ x",
    "        V = tau * (1 - tau) * sonuc.Hinv @ sonuc.J @ sonuc.Hinv",
    "        sonuc.cov = pd.DataFrame(V, index=X.columns, columns=X.columns)",
    "        sonuc.bse = pd.Series(np.sqrt(np.diag(V)), index=X.columns)",
    "    return sonuc",
]

DIFFERENCE_HELPER = [
    "def kantil_farki(dusuk, yuksek, terim):",
    '    """β̂(τ₂) − β̂(τ₁) ve standart hatası. İki tahmin aynı veriden geldiği için bağımsız değildir:',
    "    Cov(β̂τ₁, β̂τ₂) = (min(τ₁, τ₂) − τ₁τ₂) H₁⁻¹ X'X H₂⁻¹ (Hendricks–Koenker parçalarıyla).",
    '    """',
    "    j = list(dusuk.params.index).index(terim)",
    "    C = (min(dusuk.q, yuksek.q) - dusuk.q * yuksek.q) * dusuk.Hinv @ dusuk.J @ yuksek.Hinv",
    "    fark = yuksek.params.iloc[j] - dusuk.params.iloc[j]",
    "    sh = np.sqrt(dusuk.cov.iloc[j, j] + yuksek.cov.iloc[j, j] - 2 * C[j, j])",
    "    return fark, sh",
]

LOCAL_HELPER = [
    "def yerel_dogrusal(x, y, noktalar, h, derece=1):",
    '    """Gauss çekirdekli yerel doğrusal (derece=1) veya yerel sabit / Nadaraya–Watson (derece=0) tahmin.',
    "",
    "    Ağırlık w = exp(−u²/2), u = (Xᵢ − x)/h: h çekirdeğin standart sapmasıdır. Yerel doğrusal tahmin",
    "    ağırlıklı en küçük karelerin sabit terimidir: m̂(x) = (S₂T₀ − S₁T₁)/(S₀S₂ − S₁²).",
    '    """',
    "    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)",
    "    tamam = ~(np.isnan(x) | np.isnan(y))",
    "    x, y = x[tamam], y[tamam]",
    "    p = np.atleast_1d(np.asarray(noktalar, dtype=float))",
    "    d = x[None, :] - p[:, None]",
    "    w = np.exp(-0.5 * (d / h) ** 2)",
    "    s0, s1, s2 = w.sum(axis=1), (w * d).sum(axis=1), (w * d * d).sum(axis=1)",
    "    t0, t1 = w @ y, (w * d) @ y",
    "    return t0 / s0 if derece == 0 else (s2 * t0 - s1 * t1) / (s0 * s2 - s1**2)",
]

CV_HELPER = [
    "def cv_olcutu(x, y, h_degerleri, kume=None):",
    '    """CV(h): dışarıda bırakılan tahmin hatalarının kareler ortalaması.',
    "",
    "    Birini dışarıda bırakmada gözlem kendi tahmininden çıkarılır; küme verilirse gözlemin bütün",
    "    kümesi (ör. okulu) çıkarılır: küme-silmeli CV.",
    '    """',
    "    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)",
    "    D = x[None, :] - x[:, None]  # D[i, j] = x_j − x_i",
    "    ayni = None if kume is None else (np.asarray(kume)[:, None] == np.asarray(kume)[None, :])",
    "",
    "    def hata(W):",
    "        WD = W * D",
    "        s0, s1, s2 = W.sum(axis=1), WD.sum(axis=1), (WD * D).sum(axis=1)",
    "        m = (s2 * (W @ y) - s1 * (WD @ y)) / (s0 * s2 - s1**2)",
    "        return np.mean((y - m) ** 2)",
    "",
    "    satirlar = []",
    "    for h in h_degerleri:",
    "        W = np.exp(-0.5 * (D / h) ** 2)",
    "        np.fill_diagonal(W, 0.0)",
    '        satir = {"h": h, "cv": hata(W)}',
    "        if ayni is not None:",
    "            W[ayni] = 0.0",
    '            satir["cv_kume"] = hata(W)',
    "        satirlar.append(satir)",
    '    return pd.DataFrame(satirlar).set_index("h")',
]

BINS_HELPER = [
    "def aralik_ortalamalari(x, y, k):",
    '    """x\'in eşit genişlikli k aralığında x ve y ortalamaları (verinin özeti)."""',
    "    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)",
    "    tamam = ~(np.isnan(x) | np.isnan(y))",
    "    x, y = x[tamam], y[tamam]",
    "    genislik = (x.max() - x.min()) / k",
    "    aralik = np.minimum(np.floor((x - x.min()) / genislik), k - 1)",
    '    return pd.DataFrame({"aralik": aralik, "x": x, "y": y}).groupby("aralik")[["x", "y"]].mean()',
]


def _wrap_list(opening: str, items: list[str], closing: str, width: int = 88) -> list[str]:
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


def _names(items) -> str:
    return "[" + ", ".join(f'"{item}"' for item in items) + "]"


def grid_text(low: float, high: float, step: float) -> str:
    count = len(bandwidth_grid(low, high, step))
    return f"np.round({E.format_number(low)} + {E.format_number(step)} * np.arange({count}), 10)"


def operation(gen, op) -> list[str] | None:
    """Kantil ve parametrik olmayan işlemlerin Python kodu; tanımadığı işlemde ``None``."""

    if isinstance(op, QuantileRegression):
        quantile = E.format_number(op.q)
        label = " (medyan / LAD)" if op.q == 0.5 else ""
        regressors = _names(op.regressors)
        lines = [f"# Kantil regresyon, τ = {quantile}{label}: kesin çözüm"]
        if op.vcov == "nid":
            lines[0] += "; Hendricks–Koenker standart hataları"
        tail = ")" if op.vcov == "nid" else ", sh=False)"
        call = f'{op.name} = kantil_regresyonu({op.frame}, "{op.outcome}", {regressors}, {quantile}{tail}'
        if len(call) > 110:
            lines += _wrap_list(f"regresorler_{op.name} = [", [f'"{r}"' for r in op.regressors], "]")
            lines.append(f'{op.name} = kantil_regresyonu({op.frame}, "{op.outcome}", regresorler_{op.name}, '
                         f"{quantile}{tail}")
        else:
            lines.append(call)
        shown = gen._shown(op)
        selection = f".loc[{shown}]" if len(op.regressors) > 5 else ""
        if op.vcov == "nid":
            lines.append(
                f'print(pd.DataFrame({{"katsayı": {op.name}.params, "SH": {op.name}.bse}}){selection}.round(4))'
            )
        else:
            lines.append(f"print({op.name}.params{selection}.round(4))")
        return lines
    if isinstance(op, QuantileDifference):
        low, high = gen.models[op.low], gen.models[op.high]
        term = gen._key(op.low, op.term)
        return [
            f"# {op.comment}: β̂(τ = {E.format_number(high.q)}) − β̂(τ = {E.format_number(low.q)})",
            "# İki tahmin aynı veriden gelir; ortak kovaryans hesaba katılır",
            f'{op.name}, {op.name}_se = kantil_farki({op.low}, {op.high}, "{term}")',
            f"{op.name}_z = {op.name} / {op.name}_se",
            f"{op.name}_p = 2 * stats.norm.sf(abs({op.name}_z))",
            f'print(f"Fark = {{{op.name}:.4f}} (SH {{{op.name}_se:.4f}}), z = {{{op.name}_z:.2f}}, '
            f'p = {{{op.name}_p:.2e}}")',
        ]
    if isinstance(op, CoefficientProfile):
        return _coefficient_profile(gen, op)
    if isinstance(op, LocalLinear):
        values = ", ".join(E.format_number(v) for v in op.values)
        pairs = ", ".join(f'("{column}", {E.format_number(h)})' for column, h in op.bandwidths)
        return [
            f"# Gauss çekirdekli yerel doğrusal tahmin: {op.y} ~ m({op.x}); h çekirdeğin standart sapmasıdır",
            f"degerler = [{values}]",
            f"{op.result} = pd.DataFrame({{",
            f'    ad: yerel_dogrusal({op.frame}["{op.x}"], {op.frame}["{op.y}"], degerler, h)',
            f"    for ad, h in [{pairs}]",
            "}, index=degerler)",
            f"print({op.result}.round(2))",
        ]
    if isinstance(op, BandwidthCV):
        return _bandwidth_cv(op)
    if isinstance(op, LocalResidual):
        h = E.format_number(op.bandwidth)
        frame = op.frame
        return [
            f"# {op.comment}",
            f'{frame}["{op.name}"] = {frame}["{op.variable}"] - yerel_dogrusal({frame}["{op.x}"], '
            f'{frame}["{op.variable}"], {frame}["{op.x}"], {h})',
        ]
    return None


def _coefficient_profile(gen, op: CoefficientProfile) -> list[str]:
    taus = ", ".join(E.format_number(tau) for tau, _ in op.models)
    models = ", ".join(model for _, model in op.models)
    term = gen._key(op.models[0][1], op.term)
    lines = [
        f"# {op.y_label}: kantiller boyunca profil ve noktasal %95 güven bandı (± {CI_MULTIPLIER}·SH)",
        "profil = pd.DataFrame({",
        f'    "tau": [{taus}],',
        f'    "katsayi": [m.params["{term}"] for m in ({models})],',
        f'    "sh": [m.bse["{term}"] for m in ({models})],',
        "})",
        f'alt, ust = profil["katsayi"] - {CI_MULTIPLIER} * profil["sh"], profil["katsayi"] + {CI_MULTIPLIER} * profil["sh"]',
        "print(profil.round(4))",
        "fig, ax = plt.subplots(figsize=(8, 5))",
        'ax.fill_between(profil["tau"], alt, ust, color="#107C89", alpha=0.18, label="Noktasal %95 güven bandı")',
        'ax.plot(profil["tau"], profil["katsayi"], marker="o", color="#107C89", linewidth=2, label="Kantil regresyon")',
    ]
    if op.reference:
        reference = gen._key(op.reference, op.term)
        lines.append(
            f'ax.axhline({op.reference}.params["{reference}"], color="#B3392F", linestyle="--", linewidth=2, '
            f'label="{op.reference_label}")'
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


def _bandwidth_cv(op: BandwidthCV) -> list[str]:
    low, high, step = op.grid
    frame = op.frame
    lines = [
        "# Çapraz doğrulama ölçütü CV(h), h ızgarasında" + (": birini ve kümeyi dışarıda bırakarak" if op.cluster else ""),
        f"h_izgara = {grid_text(low, high, step)}",
    ]
    used = [op.x, op.y] + ([op.cluster] if op.cluster else [])
    lines.append(f"ornek = {frame}[{_names(used)}].dropna()")
    lines.append(f'{op.result} = cv_olcutu(ornek["{op.x}"], ornek["{op.y}"], h_izgara'
                 + (f', kume=ornek["{op.cluster}"])' if op.cluster else ")"))
    lines.append(f'{op.name}_h = {op.result}["cv"].idxmin()  # eşitlikte küçük h')
    if op.cluster:
        lines.append(f'{op.name}_h_kume = {op.result}["cv_kume"].idxmin()')
        lines.append(f'print("CV ile seçilen h:", {op.name}_h, "| küme-silmeli CV ile:", {op.name}_h_kume)')
    else:
        lines.append(f'print("CV ile seçilen h:", {op.name}_h)')
    lines += [
        "fig, ax = plt.subplots(figsize=(8, 5))",
        f'ax.plot({op.result}.index, {op.result}["cv"] - {op.result}["cv"].min(), color="#107C89", linewidth=2,',
        '        label="Birini dışarıda bırak")',
        f'ax.axvline({op.name}_h, color="#107C89", linestyle=":")',
    ]
    if op.cluster:
        lines += [
            f'ax.plot({op.result}.index, {op.result}["cv_kume"] - {op.result}["cv_kume"].min(), color="#B3392F",',
            '        linewidth=2, label="Küme-silmeli")',
            f'ax.axvline({op.name}_h_kume, color="#B3392F", linestyle=":")',
        ]
    lines += [
        f'ax.set_xlabel("{op.x_label}")',
        'ax.set_ylabel("CV(h) − en küçük CV")',
        f'ax.set_title("{op.title}")',
        "ax.legend()",
        "ax.grid(alpha=0.3)",
        "plt.show()",
    ]
    return lines
