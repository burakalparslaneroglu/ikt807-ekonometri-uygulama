"""Stata kodu: kantil regresyon ve parametrik olmayan regresyon işlemleri (Konu 7–8).

Kantil regresyon ``qreg`` ile (doğrusal programlamanın kesin çözümü). Hendricks–Koenker
standart hataları R ``summary.rq(se = "nid")`` hesabıyla birebir aynı olsun diye ``hk_sh``
programında açıkça yazılır. Yerel doğrusal tahmin ve çapraz doğrulama Mata fonksiyonlarıyla;
işlemler tek satırlık ``mata:`` çağrılarıdır, böylece Monte Carlo döngülerinde de çalışır.
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

HK_HELPER = [
    "* Hendricks–Koenker (1992) standart hataları; R summary.rq(se = \"nid\") ile aynı hesap.",
    "* Koşullu yoğunluk her gözlemde τ ± h kantil doğrularının farkından tahmin edilir:",
    "*   f_i = max(0, 2h / (x_i'(b(τ+h) − b(τ−h)) − 1.49e-08)), h Hall–Sheather bandı.",
    "* V = τ(1−τ) H⁻¹ X'X H⁻¹, H = X'FX. Hinv_model ve J_model (X'X) kantiller arası fark için saklanır.",
    "capture program drop hk_sh",
    "program define hk_sh, eclass",
    "    syntax varlist(min=2 numeric), tau(real) model(name)",
    "    gettoken bagimli regresorler : varlist",
    "    tempvar ornek yh yl f",
    "    quietly estimates restore `model'",
    "    quietly generate byte `ornek' = e(sample)",
    "    scalar hk_z = invnormal(`tau')",
    "    scalar hk_h = e(N)^(-1/3) * invnormal(0.975)^(2/3) * ///",
    "        (1.5*normalden(scalar(hk_z))^2 / (2*scalar(hk_z)^2 + 1))^(1/3)",
    "    while `tau' - scalar(hk_h) < 0 | `tau' + scalar(hk_h) > 1 {",
    "        scalar hk_h = scalar(hk_h) / 2",
    "    }",
    "    quietly qreg `bagimli' `regresorler' if `ornek', quantile(`=`tau' + scalar(hk_h)')",
    "    quietly predict double `yh' if `ornek', xb",
    "    quietly qreg `bagimli' `regresorler' if `ornek', quantile(`=`tau' - scalar(hk_h)')",
    "    quietly predict double `yl' if `ornek', xb",
    "    quietly generate double `f' = max(0, 2*scalar(hk_h) / (`yh' - `yl' - 1.4901161193847656e-08)) if `ornek'",
    "    quietly matrix accum XFX = `regresorler' if `ornek' [iweight = `f']",
    "    quietly matrix accum XX = `regresorler' if `ornek'",
    "    matrix Hinv_`model' = invsym(XFX)",
    "    matrix J_`model' = XX",
    "    local w = `tau' * (1 - `tau')",
    "    matrix V_hk = `w' * Hinv_`model' * J_`model' * Hinv_`model'",
    "    matrix V_hk = (V_hk + V_hk') / 2",
    "    quietly estimates restore `model'",
    "    local adlar : colfullnames e(b)",
    "    matrix rownames V_hk = `adlar'",
    "    matrix colnames V_hk = `adlar'",
    "    ereturn repost V = V_hk",
    "    quietly estimates store `model'",
    "end",
]

LOCAL_HELPER = [
    "* Gauss çekirdekli yerel doğrusal (derece = 1) veya yerel sabit / Nadaraya–Watson (derece = 0) tahmin.",
    "* Ağırlık w = exp(−u²/2), u = (Xi − x)/h: h çekirdeğin standart sapmasıdır. Yerel doğrusal tahmin",
    "* ağırlıklı en küçük karelerin sabit terimidir: (S2·T0 − S1·T1)/(S0·S2 − S1²).",
    "* yerel_uyum: örneklem noktalarında tahmin (artik = 1 ise y − tahmin) değişkene yazılır.",
    "* yerel_skalerler: seçilen noktalardaki tahminler Stata skalerlerine yazılır.",
    "capture mata: mata drop yerel_dogrusal()",
    "capture mata: mata drop yerel_uyum()",
    "capture mata: mata drop yerel_skalerler()",
    "mata:",
    "real colvector yerel_dogrusal(real colvector x, real colvector y, real colvector p, real scalar h, real scalar derece)",
    "{",
    "    real colvector m, d, w",
    "    real scalar i, s0, s1, s2, t0, t1",
    "    m = J(rows(p), 1, .)",
    "    for (i = 1; i <= rows(p); i++) {",
    "        d = x :- p[i]",
    "        w = exp(-0.5 :* (d :/ h):^2)",
    "        s0 = sum(w)",
    "        s1 = sum(w :* d)",
    "        s2 = sum(w :* d :* d)",
    "        t0 = sum(w :* y)",
    "        t1 = sum(w :* d :* y)",
    "        m[i] = (derece == 0 ? t0 / s0 : (s2 * t0 - s1 * t1) / (s0 * s2 - s1^2))",
    "    }",
    "    return(m)",
    "}",
    "",
    "void yerel_uyum(string scalar xad, string scalar yad, string scalar yeni, real scalar h, real scalar derece, real scalar artik)",
    "{",
    "    real matrix A",
    "    real colvector tamam, m",
    "    A = st_data(., (xad, yad))",
    "    tamam = selectindex(rowmissing(A) :== 0)",
    "    m = yerel_dogrusal(A[tamam, 1], A[tamam, 2], A[tamam, 1], h, derece)",
    "    st_store(tamam, yeni, (artik ? A[tamam, 2] - m : m))",
    "}",
    "",
    "void yerel_skalerler(string scalar xad, string scalar yad, real rowvector p, real scalar h, string rowvector adlar)",
    "{",
    "    real matrix A",
    "    real colvector m",
    "    real scalar i",
    "    A = st_data(., (xad, yad))",
    "    A = select(A, rowmissing(A) :== 0)",
    "    m = yerel_dogrusal(A[., 1], A[., 2], p', h, 1)",
    "    for (i = 1; i <= cols(p); i++) {",
    "        st_numscalar(adlar[i], m[i])",
    "    }",
    "}",
    "end",
]

CV_HELPER = [
    "* CV(h): dışarıda bırakılan tahmin hatalarının kareler ortalaması. Birini dışarıda bırakmada gözlem",
    "* kendi tahmininden, küme-silmeli CV'de gözlemin bütün kümesi (ör. okulu) çıkarılır.",
    "* cv_sec: h ızgarasında CV tablosunu döndürür; en küçük CV'yi veren h (eşitlikte küçük olan)",
    "* ad_h ve ad_h_kume skalerlerine yazılır. tablo_veri: tabloyu boş veri setine değişken olarak yazar.",
    "capture mata: mata drop cv_hata()",
    "capture mata: mata drop ilk_en_kucuk()",
    "capture mata: mata drop cv_sec()",
    "capture mata: mata drop tablo_veri()",
    "mata:",
    "real scalar cv_hata(real matrix W, real matrix D, real colvector y)",
    "{",
    "    real matrix WD",
    "    real colvector s0, s1, s2, m",
    "    WD = W :* D",
    "    s0 = rowsum(W)",
    "    s1 = rowsum(WD)",
    "    s2 = rowsum(WD :* D)",
    "    m = (s2 :* (W * y) - s1 :* (WD * y)) :/ (s0 :* s2 - s1:^2)",
    "    return(mean((y - m):^2))",
    "}",
    "",
    "real scalar ilk_en_kucuk(real colvector v)",
    "{",
    "    real scalar k, en",
    "    en = 1",
    "    for (k = 2; k <= rows(v); k++) {",
    "        if (v[k] < v[en]) en = k",
    "    }",
    "    return(en)",
    "}",
    "",
    "real matrix cv_sec(string scalar xad, string scalar yad, string scalar kad, real scalar alt, real scalar adim, real scalar sayi, string scalar ad)",
    "{",
    "    real matrix A, D, D2, W, disari, sonuc",
    "    real colvector hs, y",
    "    real scalar k, n",
    "    A = st_data(., (kad == \"\" ? (xad, yad) : (xad, yad, kad)))",
    "    A = select(A, rowmissing(A) :== 0)",
    "    n = rows(A)",
    "    y = A[., 2]",
    "    D = J(n, 1, 1) * A[., 1]' - A[., 1] * J(1, n, 1)",
    "    D2 = D :* D",
    "    if (kad != \"\") disari = 1 :- (A[., 3] :== A[., 3]')",
    "    hs = round(alt :+ adim :* (0::(sayi - 1)), 1e-10)",
    "    sonuc = J(sayi, 2, .)",
    "    for (k = 1; k <= sayi; k++) {",
    "        W = exp(-0.5 :* D2 :/ (hs[k] * hs[k]))",
    "        _diag(W, 0)",
    "        sonuc[k, 1] = cv_hata(W, D, y)",
    "        if (kad != \"\") {",
    "            sonuc[k, 2] = cv_hata(W :* disari, D, y)",
    "        }",
    "    }",
    "    st_numscalar(ad + \"_h\", hs[ilk_en_kucuk(sonuc[., 1])])",
    "    if (kad != \"\") st_numscalar(ad + \"_h_kume\", hs[ilk_en_kucuk(sonuc[., 2])])",
    "    return((hs, sonuc))",
    "}",
    "",
    "void tablo_veri(real matrix T, string rowvector adlar)",
    "{",
    "    st_addobs(rows(T))",
    "    (void) st_addvar(\"double\", adlar)",
    "    st_store(., adlar, T)",
    "}",
    "end",
]

_TEAL = "16 124 137"
_TEAL_LIGHT = "195 222 226"
"""Güven bandı: Stata 14'te saydamlık yok; %25 örtücülükteki rengin beyaz üzerindeki karşılığı."""
_RED = "179 57 47"


def _term(term: str) -> str:
    return "_cons" if term == E.INTERCEPT else term


def _strings(items) -> str:
    return "(" + ", ".join(f'"{item}"' for item in items) + ")"


def helper_names(ops, layers) -> list[str]:
    from core.labs.spec import LocalCurve

    names: list[str] = []
    if any(isinstance(op, QuantileRegression) and op.vcov == "nid" for op in ops):
        names.append("hk_sh")
    if any(isinstance(op, (LocalLinear, LocalResidual)) for op in ops) or any(
        isinstance(layer, LocalCurve) for layer in layers
    ):
        names.append("yerel")
    if any(isinstance(op, BandwidthCV) for op in ops):
        names.append("cv")
    return names


HELPERS = {"hk_sh": HK_HELPER, "yerel": LOCAL_HELPER, "cv": CV_HELPER}


def operation(gen, op, command) -> list[str] | None:
    """Kantil ve parametrik olmayan işlemlerin Stata kodu; tanımadığı işlemde ``None``.

    ``command`` uzun komutları ``///`` ile bölen fonksiyondur (``stata_gen._command``).
    """

    if isinstance(op, QuantileRegression):
        regressors = " ".join(op.regressors)
        quantile = E.format_number(op.q)
        label = " (medyan / LAD)" if op.q == 0.5 else ""
        lines = [f"* Kantil regresyon, τ = {quantile}{label}: doğrusal programlamanın kesin çözümü (qreg)"]
        lines += command(f"quietly qreg {op.outcome} {regressors}, quantile({quantile})")
        lines.append(f"estimates store {op.name}")
        if op.vcov == "nid":
            lines.append("* Standart hatalar Hendricks–Koenker: qreg'in varsayılan (iid) kovaryansının yerine yazılır")
            lines += command(f"hk_sh {op.outcome} {regressors}, tau({quantile}) model({op.name})")
            keep = f"keep({op.regressors[0]}) " if len(op.regressors) > 5 else ""
            lines.append(f"estimates table {op.name}, {keep}b(%9.4f) se(%9.4f)")
        else:
            lines.append(f"estimates table {op.name}, keep({op.regressors[0]}) b(%9.4f)")
        return lines
    if isinstance(op, QuantileDifference):
        return _difference(gen, op)
    if isinstance(op, CoefficientProfile):
        return _coefficient_profile(gen, op)
    if isinstance(op, LocalLinear):
        return _local_linear(gen, op, command)
    if isinstance(op, BandwidthCV):
        return _bandwidth_cv(op, command)
    if isinstance(op, LocalResidual):
        h = E.format_number(op.bandwidth)
        return [
            f"* {op.comment}",
            f"generate double {op.name} = .",
            f'mata: yerel_uyum("{op.x}", "{op.variable}", "{op.name}", {h}, 1, 1)',
        ]
    return None


def _difference(gen, op: QuantileDifference) -> list[str]:
    low, high = gen.models[op.low], gen.models[op.high]
    t1, t2 = E.format_number(low.q), E.format_number(high.q)
    term = _term(op.term)
    name = op.name
    return [
        f"* {op.comment}: b(τ = {t2}) − b(τ = {t1})",
        "* İki tahmin aynı veriden gelir: Cov = (min(τ1, τ2) − τ1·τ2) H1⁻¹ X'X H2⁻¹ (hk_sh matrisleri)",
        f"local w = min({t1}, {t2}) - {t1} * {t2}",
        f"matrix C = `w' * Hinv_{op.low} * J_{op.low} * Hinv_{op.high}",
        f'local j = colnumb(C, "{term}")',
        f"quietly estimates restore {op.low}",
        f"scalar {name}_dusuk = _b[{term}]",
        f"scalar {name}_v = _se[{term}]^2",
        f"quietly estimates restore {op.high}",
        f"scalar {name} = _b[{term}] - scalar({name}_dusuk)",
        f"scalar {name}_se = sqrt(scalar({name}_v) + _se[{term}]^2 - 2*el(C, `j', `j'))",
        f"scalar {name}_z = scalar({name}) / scalar({name}_se)",
        f"scalar {name}_p = 2*normal(-abs(scalar({name}_z)))",
        f'display "Fark = " %7.4f scalar({name}) " (SH " %6.4f scalar({name}_se) ")" ///',
        f'    ", z = " %6.2f scalar({name}_z) ", p = " %9.2e scalar({name}_p)',
    ]


def _coefficient_profile(gen, op: CoefficientProfile) -> list[str]:
    term = _term(op.term)
    count = len(op.models)
    lines = [
        f"* {op.y_label}: kantiller boyunca profil ve noktasal %95 güven bandı (± {CI_MULTIPLIER}·SH)",
        f"matrix profil = J({count}, 3, .)",
        "matrix colnames profil = tau katsayi sh",
    ]
    for row, (tau, model) in enumerate(op.models, start=1):
        lines += [
            f"quietly estimates restore {model}",
            f"matrix profil[{row}, 1] = {E.format_number(tau)}",
            f"matrix profil[{row}, 2] = _b[{term}]",
            f"matrix profil[{row}, 3] = _se[{term}]",
        ]
    lines.append("matrix list profil, format(%9.4f)")
    if op.reference:
        lines += [f"quietly estimates restore {op.reference}", f"scalar profil_ref = _b[{term}]"]
    lines += [
        "preserve",
        "clear",
        "quietly svmat double profil, names(col)",
        f"generate double alt = katsayi - {CI_MULTIPLIER}*sh",
        f"generate double ust = katsayi + {CI_MULTIPLIER}*sh",
        f'twoway (rarea alt ust tau, fcolor("{_TEAL_LIGHT}") lcolor("{_TEAL_LIGHT}")) ///',
        f'       (connected katsayi tau, lcolor("{_TEAL}") mcolor("{_TEAL}") lwidth(medthick)) ///',
    ]
    legend = '2 "Kantil regresyon" 1 "%95 güven bandı"'
    if op.reference:
        lines.append(
            f'       (function y = scalar(profil_ref), range(tau) lcolor("{_RED}") lpattern(dash) lwidth(medthick)) ///'
        )
        legend += f' 3 "{op.reference_label}"'
    lines += [
        f"       , legend(order({legend})) ///",
        f'       xtitle("{op.x_label}") ytitle("{op.y_label}") title("{op.title}", size(medium))',
        "restore",
    ]
    return lines


def _local_linear(gen, op: LocalLinear, command) -> list[str]:
    from core.codegen.stata_gen import _cell

    points = "(" + ", ".join(E.format_number(v) for v in op.values) + ")"
    lines = [
        f"* Gauss çekirdekli yerel doğrusal tahmin: {op.y} = m({op.x}); h çekirdeğin standart sapmasıdır.",
        f"* Tahminler {op.result}_<sütun>_<nokta> skalerlerinde saklanır.",
    ]
    for column, h in op.bandwidths:
        names = _strings(_cell(op.result, column, value) for value in op.values)
        lines += command(
            f'mata: yerel_skalerler("{op.x}", "{op.y}", {points}, {E.format_number(h)}, {names})'
        )
    header = f'display "{op.x:>10}"' + "".join(f' "{column:>10}"' for column, _ in op.bandwidths)
    lines.append(header)
    for value in op.values:
        cells = " ".join(f"%10.2f scalar({_cell(op.result, column, value)})" for column, _ in op.bandwidths)
        lines.append(f'display "{E.format_number(value):>10}" {cells}')
    return lines


def _bandwidth_cv(op: BandwidthCV, command) -> list[str]:
    low, high, step = op.grid
    count = len(bandwidth_grid(low, high, step))
    cluster = op.cluster or ""
    lines = [
        "* Çapraz doğrulama ölçütü CV(h), h ızgarasında"
        + (": birini ve kümeyi dışarıda bırakarak" if op.cluster else ": birini dışarıda bırakarak"),
        f"* h = {E.format_number(low)}, {E.format_number(low + step)}, ..., {E.format_number(high)} "
        f"({count} değer); n×n ağırlık matrisiyle hesaplanır",
        *command(
            f'mata: {op.result} = cv_sec("{op.x}", "{op.y}", "{cluster}", {E.format_number(low)}, '
            f'{E.format_number(step)}, {count}, "{op.name}")'
        ),
    ]
    if op.cluster:
        lines.append(
            f'display "CV ile seçilen h: " scalar({op.name}_h) "   küme-silmeli CV ile: " scalar({op.name}_h_kume)'
        )
    else:
        lines.append(f'display "CV ile seçilen h: " scalar({op.name}_h)')
    lines += [
        "preserve",
        "clear",
        f'mata: tablo_veri({op.result}, ("h", "cv", "cv_kume"))',
        "quietly summarize cv",
        "generate double cv_fark = cv - r(min)",
    ]
    layers = [f'(line cv_fark h, lcolor("{_TEAL}") lwidth(medthick))']
    marks = [f'xline(`=scalar({op.name}_h)\', lcolor("{_TEAL}") lpattern(dot))']
    legend = '1 "Birini dışarıda bırak"'
    if op.cluster:
        lines += ["quietly summarize cv_kume", "generate double cv_kume_fark = cv_kume - r(min)"]
        layers.append(f'(line cv_kume_fark h, lcolor("{_RED}") lwidth(medthick))')
        marks.append(f'xline(`=scalar({op.name}_h_kume)\', lcolor("{_RED}") lpattern(dot))')
        legend += ' 2 "Küme-silmeli"'
    lines += [
        f"twoway {layers[0]} ///",
        *[f"       {layer} ///" for layer in layers[1:]],
        f"       , {marks[0]} ///",
        *[f"       {mark} ///" for mark in marks[1:]],
        f"       legend(order({legend})) ///",
        f'       xtitle("{op.x_label}") ytitle("CV(h) − en küçük CV") title("{op.title}", size(medium))',
        "restore",
    ]
    return lines


def plot_layer(layer, index: int, x: str, color: str, dashed: bool) -> tuple[list[str], str, list[str]]:
    """Grafik katmanı için (hazırlık satırları, twoway katmanı, sonra silinecek değişkenler)."""

    from core.labs.spec import BinMeans, LocalCurve

    if isinstance(layer, LocalCurve):
        name = f"egri_{index}"
        pattern = " lpattern(dash)" if dashed else ""
        setup = [
            f"capture drop {name}",
            f"generate double {name} = .",
            f'mata: yerel_uyum("{x}", "{layer.y}", "{name}", {E.format_number(layer.bandwidth)}, {layer.degree}, 0)',
        ]
        return setup, f"(line {name} {x}, sort lcolor({color}) lwidth(medthick){pattern})", [name]
    if isinstance(layer, BinMeans):
        y, k = layer.y, layer.bins
        group, mx, my, tag = f"ar_{y}", f"ax_{y}", f"ay_{y}", f"at_{y}"
        setup = [
            f"capture drop {group} {mx} {my} {tag}",
            f"* {x}'in eşit genişlikli {k} aralığında {x} ve {y} ortalamaları",
            f"quietly summarize {x} if !missing({y})",
            f"generate double {group} = min(floor(({x} - r(min)) / ((r(max) - r(min)) / {k})), {k} - 1) ///",
            f"    if !missing({x}, {y})",
            f"bysort {group}: egen double {mx} = mean({x})",
            f"bysort {group}: egen double {my} = mean({y})",
            f"egen byte {tag} = tag({group})",
        ]
        return setup, f"(scatter {my} {mx} if {tag}, msymbol(O) mcolor({color}))", [group, mx, my, tag]
    raise TypeError(f"Tanınmayan grafik katmanı: {type(layer).__name__}")
