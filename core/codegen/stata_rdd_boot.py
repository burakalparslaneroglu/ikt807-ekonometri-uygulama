"""Stata kodu: regresyon süreksizliği ve bootstrap işlemleri (Konu 9–10).

RDD ``rdd_yd`` programıyla tahmin edilir: ``regress y D R DR [aweight = w] if w > 0, vce(robust)``;
aweight ölçeği sandviçte sadeleştiği için bu, Python (statsmodels WLS, HC1) ve R (``vcovHC`` HC1) ile aynı
standart hatadır. RDD grafiğinin eğrileri Mata ile hesaplanır. Bootstrap tekrarları açık bir döngüdür
(``bsample``; wild'da Rademacher çarpanı); sonuçlar ``tempfile`` ile toplanır, klasörde dosya bırakmaz.
Stata'nın rastgele sayı üreteci farklı olduğu için bootstrap sonuçları dağılımda aynıdır.
"""

from __future__ import annotations

from core.codegen.python_rdd_boot import KERNEL_NAMES, SCALE_NAMES, rdd_comment
from core.labs import expr as E
from core.labs.runner import CI_MULTIPLIER, bootstrap_key, uses_replicate_se
from core.labs.spec import BOOT, OLS, RDD, Bootstrap, RDDTable, ScalarTable

RDD_HELPER = [
    "* Keskin RDD: eşikte yerel doğrusal sıçrama (D katsayısı), ağırlıklı EKK ve HC1 standart hatası.",
    "* Pozitif ağırlıklı gözlemlerde y'nin D = 1{x >= c}, R = x - c ve DR = D·R üzerine regresyonu:",
    "* regress y D R DR [aweight = w] if w > 0, vce(robust). Hansen ölçeği: çekirdek birim varyanslıdır",
    "* ve h çekirdeğin standart sapmasıdır; üçgen çekirdekte ağırlık eşikten h√6, dikdörtgende h√3",
    "* uzaklıkta sıfırlanır. olcek(pencere): h pencerenin yarı genişliğidir. D, R, DR ve rdd_w her",
    "* çağrıda yeniden oluşturulur.",
    "capture program drop rdd_yd",
    "program define rdd_yd",
    "    syntax varlist(min=2 max=2 numeric), esik(real) h(real) [cekirdek(string) olcek(string)]",
    "    gettoken y x : varlist",
    "    local pencere = `h'",
    "    if \"`olcek'\" != \"pencere\" {",
    "        local pencere = `h' * sqrt(cond(\"`cekirdek'\" == \"dikdortgen\", 3, 6))",
    "    }",
    "    capture drop D R DR rdd_w",
    "    quietly generate double R = `x' - `esik' if !missing(`x', `y')",
    "    quietly generate double D = (R >= 0) if !missing(R)",
    "    quietly generate double DR = D * R",
    "    if \"`cekirdek'\" == \"dikdortgen\" {",
    "        quietly generate double rdd_w = (abs(R) <= `pencere') if !missing(R)",
    "    }",
    "    else {",
    "        quietly generate double rdd_w = max(1 - abs(R) / `pencere', 0) if !missing(R)",
    "    }",
    "    quietly regress `y' D R DR [aweight = rdd_w] if rdd_w > 0 & !missing(rdd_w), vce(robust)",
    "end",
]

CURVE_HELPER = [
    "* RDD grafiği: eşiğin iki yanında ayrı yerel doğrusal tahmin ve noktasal %95 güven bandı. Her nokta x0",
    "* için yalnız o taraftaki gözlemlerle üçgen çekirdekli (pencere ±h√6) ağırlıklı EKK'nin sabit terimi:",
    "* (S2·T0 − S1·T1)/(S0·S2 − S1²); SH o yerel regresyonun HC1 sandviçi (k/(k−2) çarpanıyla).",
    "* rdd_egrisi() ilk n0 satırdaki veriyi kullanır ve sonuçları sonraki 2×nokta satıra yazar:",
    "* <önek>_x, <önek>_m (tahmin), <önek>_alt, <önek>_ust ve <önek>_sag (0: eşiğin solu, 1: sağı).",
    "capture mata: mata drop rdd_taraf()",
    "capture mata: mata drop rdd_egrisi()",
    "mata:",
    "real matrix rdd_taraf(real colvector x, real colvector y, real colvector p, real scalar pencere)",
    "{",
    "    real matrix sonuc",
    "    real colvector d, w, u",
    "    real scalar i, k, s0, s1, s2, t0, t1, det, a, b",
    "    sonuc = J(rows(p), 2, .)",
    "    for (i = 1; i <= rows(p); i++) {",
    "        d = x :- p[i]",
    "        w = 1 :- abs(d) :/ pencere",
    "        w = w :* (w :> 0)",
    "        k = sum(w :> 0)",
    "        if (k > 2) {",
    "            s0 = sum(w)",
    "            s1 = sum(w :* d)",
    "            s2 = sum(w :* d :* d)",
    "            t0 = sum(w :* y)",
    "            t1 = sum(w :* d :* y)",
    "            det = s0 * s2 - s1^2",
    "            a = (s2 * t0 - s1 * t1) / det",
    "            b = (s0 * t1 - s1 * t0) / det",
    "            u = (w :* (y :- a :- b :* d)):^2",
    "            sonuc[i, 1] = a",
    "            sonuc[i, 2] = sqrt((s2^2 * sum(u) - 2 * s1 * s2 * sum(u :* d) + s1^2 * sum(u :* d :* d)) / det^2 * k / (k - 2))",
    "        }",
    "    }",
    "    return(sonuc)",
    "}",
    "",
    "void rdd_egrisi(string scalar xad, string scalar yad, real scalar n0, real scalar esik, real scalar pencere, real scalar alt, real scalar ust, real scalar nokta, string scalar onek)",
    "{",
    "    real matrix A, L, S",
    "    real colvector sol, sag, pl, pr",
    "    string rowvector adlar",
    "    A = st_data((1, n0), (xad, yad))",
    "    A = select(A, rowmissing(A) :== 0)",
    "    if (missing(alt)) alt = min(A[., 1])",
    "    if (missing(ust)) ust = max(A[., 1])",
    "    sol = selectindex(A[., 1] :< esik)",
    "    sag = selectindex(A[., 1] :>= esik)",
    "    pl = rangen(alt, esik, nokta)",
    "    pr = rangen(esik, ust, nokta)",
    "    L = rdd_taraf(A[sol, 1], A[sol, 2], pl, pencere) \\ rdd_taraf(A[sag, 1], A[sag, 2], pr, pencere)",
    f"    S = ((pl \\ pr), L[., 1], (L[., 1] - {CI_MULTIPLIER} :* L[., 2]), (L[., 1] + {CI_MULTIPLIER} :* L[., 2]), (J(nokta, 1, 0) \\ J(nokta, 1, 1)))",
    "    adlar = (onek + \"_x\", onek + \"_m\", onek + \"_alt\", onek + \"_ust\", onek + \"_sag\")",
    "    (void) st_addvar(\"double\", adlar)",
    "    st_store(((n0 + 1), (n0 + 2 * nokta)), adlar, S)",
    "}",
    "end",
]

_METHOD_COMMENTS = {
    "pairs": "Pairs bootstrap: gözlem satırları yerine koyarak çekilir (bsample); model her tekrarda baştan tahmin edilir.",
    "wild": "Wild bootstrap: regresörler sabit; y* = xb + e·ξ, ξ Rademacher (±1, eşit olasılıkla).",
    "cluster": "Küme bootstrap'ı: kümeler yerine koyarak çekilir (bsample, cluster()); seçilen kümenin bütün "
               "gözlemleri birlikte gelir.",
}
BAND_LIGHT = {"16 124 137": "195 222 226", "201 138 27": "240 226 198"}
"""Güven bandı rengi: Stata 14'te saydamlık yok; çizgi renginin %25 örtücülükle beyaz üzerindeki karşılığı."""


def _number(value: float) -> str:
    return E.format_number(value)


def _term(term: str) -> str:
    return "_cons" if term == E.INTERCEPT else term


def result_file(result: str) -> str:
    """Bootstrap tekrarlarının toplandığı tempfile'ın yerel makro adı."""

    return f"{result}_dosya"


def operation(gen, op, command) -> list[str] | None:
    """RDD ve bootstrap işlemlerinin Stata kodu; tanımadığı işlemde ``None``."""

    if isinstance(op, RDD):
        options = f"esik({_number(op.cutoff)}) h({_number(op.bandwidth)})"
        if op.kernel != "triangular":
            options += f" cekirdek({KERNEL_NAMES[op.kernel]})"
        if op.scale != "hansen":
            options += f" olcek({SCALE_NAMES[op.scale]})"
        return [
            f"* {rdd_comment(op)}",
            *command(f"rdd_yd {op.y} {op.x}, {options}"),
            f"estimates store {op.name}",
            f"estimates table {op.name}, keep(D) b(%9.4f) se(%9.4f) stats(N)",
        ]
    if isinstance(op, RDDTable):
        return _rdd_table(op)
    if isinstance(op, Bootstrap):
        return _bootstrap(gen, op, command)
    if isinstance(op, ScalarTable):
        lines: list[str] = []
        width = max(len(label) for label, _ in op.rows) + 3
        for index, (label, expression) in enumerate(op.rows, start=1):
            name = f"{op.result}_{index}"
            lines += gen._scalar_lines(name, expression)
            lines.append(
                f'display as text "{label}" _col({width}) as result %10.{op.decimals}f scalar({name})'
            )
        return lines
    return None


def _rdd_table(op: RDDTable) -> list[str]:
    count = len(op.rows)
    table = op.result
    models = " ".join(model for _, model in op.rows)
    values = " ".join(_number(h) for h, _ in op.rows)
    multiplier = CI_MULTIPLIER
    return [
        f"* Bant genişliği duyarlılığı: her h için sıçrama, HC1 SH ve %95 güven aralığı (tahmin ± {multiplier}·SH)",
        f"local modeller {models}",
        f"local h_degerleri {values}",
        f"matrix {table} = J({count}, 6, .)",
        f"matrix colnames {table} = h n tahmin sh alt ust",
        f"forvalues i = 1/{count} {{",
        "    local m : word `i' of `modeller'",
        "    local h : word `i' of `h_degerleri'",
        "    quietly estimates restore `m'",
        f"    matrix {table}[`i', 1] = `h'",
        f"    matrix {table}[`i', 2] = e(N)",
        f"    matrix {table}[`i', 3] = _b[D]",
        f"    matrix {table}[`i', 4] = _se[D]",
        f"    matrix {table}[`i', 5] = _b[D] - {multiplier}*_se[D]",
        f"    matrix {table}[`i', 6] = _b[D] + {multiplier}*_se[D]",
        f"    scalar {table}_n_`h' = e(N)",
        f"    scalar {table}_tahmin_`h' = _b[D]",
        f"    scalar {table}_sh_`h' = _se[D]",
        f"    scalar {table}_alt_`h' = _b[D] - {multiplier}*_se[D]",
        f"    scalar {table}_ust_`h' = _b[D] + {multiplier}*_se[D]",
        "}",
        f"matrix list {table}, format(%9.2f)",
        "preserve",
        "clear",
        f"quietly svmat double {table}, names(col)",
        'twoway (rcap alt ust h, lcolor("16 124 137") lwidth(medthick)) ///',
        '       (scatter tahmin h, mcolor("16 124 137") msymbol(O)) ///',
        '       , yline(0, lpattern(dot) lcolor("7 55 61")) legend(order(2 "Tahmin" 1 "%95 güven aralığı")) ///',
        f'       xtitle("{op.x_label}") ytitle("{op.y_label}") title("{op.title}", size(medium))',
        "restore",
    ]


def _scalar_name(kind: str, model: str, term: str) -> str:
    return f"{kind}_{model}_{_term(term)}"


def _originals(op: Bootstrap) -> list[tuple[str, str, str]]:
    """``collect`` ifadelerinde geçen orijinal model katsayıları ve SH'leri (tür, model, terim)."""

    found: list[tuple[str, str, str]] = []

    def visit(node) -> None:
        if isinstance(node, (E.Coef, E.StdErr)) and node.model != BOOT:
            item = ("b" if isinstance(node, E.Coef) else "se", node.model, node.term)
            if item not in found:
                found.append(item)
        elif isinstance(node, E.BinOp):
            visit(node.left)
            visit(node.right)
        elif isinstance(node, E.Call):
            for argument in node.args:
                visit(argument)

    for _, expression in op.collect:
        visit(expression)
    return found


def _dialect(gen) -> E.Dialect:
    base = gen.dialect()

    def coefficient(model: str, term: str) -> str:
        return f"_b[{_term(term)}]" if model == BOOT else f"scalar({_scalar_name('b', model, term)})"

    def standard_error(model: str, term: str) -> str:
        return f"_se[{_term(term)}]" if model == BOOT else f"scalar({_scalar_name('se', model, term)})"

    return E.Dialect(variable=base.variable, coefficient=coefficient, functions=base.functions, power=base.power,
                     standard_error=standard_error, scalar=base.scalar)


def _bootstrap(gen, op: Bootstrap, command) -> list[str]:
    settings = gen.models[op.model]
    if not isinstance(settings, OLS):
        raise ValueError("Stata: bootstrap yalnız OLS modelleri için üretilir.")
    need_se = any(uses_replicate_se(expression) for _, expression in op.collect)
    regressors = " ".join(f"i.{r}" if r in settings.categorical else r for r in settings.regressors)
    robust = ", vce(robust)" if need_se else ""
    names = [name for name, _ in op.collect]
    handle, sample, results = f"{op.result}_sonuc", f"{op.result}_orneklem", result_file(op.result)
    lines = [
        f"* {op.comment}",
        f"* {_METHOD_COMMENTS[op.method]}",
        "* Tekrarlar geçici dosyada toplanır (tempfile): do-dosyası bitince silinir, klasörde dosya bırakmaz.",
        f"quietly estimates restore {op.model}",
    ]
    for kind, model, term in _originals(op):
        if model != op.model:
            lines.append(f"quietly estimates restore {model}")
        lines.append(f"scalar {_scalar_name(kind, model, term)} = _{kind}[{_term(term)}]")
        if model != op.model:
            lines.append(f"quietly estimates restore {op.model}")
    kept = list(dict.fromkeys([settings.outcome, *settings.regressors] + ([op.cluster] if op.cluster else [])))
    lines += ["preserve", "quietly keep if e(sample)"]
    if op.method == "wild":
        lines += ["quietly predict double boot_uyum, xb", "quietly predict double boot_artik, residuals"]
    else:
        lines += [*command(f"keep {' '.join(kept)}"), f"tempfile {sample}", f'quietly save "`{sample}\'"']
    if op.seed is not None:
        lines.append(f"set seed {op.seed}")
    else:
        lines.append("* Rastgele sayı üreteci veri çekilişlerinin ardından kaldığı yerden devam eder")
    lines += [
        f"tempname {handle}",
        f"tempfile {results}",
        f"postfile `{handle}' {' '.join(names)} using \"`{results}'\", replace",
        f"forvalues tekrar = 1/{op.reps} {{",
    ]
    if op.method == "wild":
        lines += [
            "    capture drop boot_y",
            "    quietly generate double boot_y = boot_uyum + boot_artik * cond(runiform() < 0.5, -1, 1)",
            *[f"    {line}" for line in command(f"quietly regress boot_y {regressors}{robust}")],
        ]
    else:
        resample = "bsample" if op.method == "pairs" else f"bsample, cluster({op.cluster})"
        lines += [
            f"    quietly use \"`{sample}'\", clear",
            f"    {resample}",
            *[f"    {line}" for line in command(f"quietly regress {settings.outcome} {regressors}{robust}")],
        ]
    dialect = _dialect(gen)
    posted = " ".join(f"({E.render(expression, dialect)})" for _, expression in op.collect)
    lines += [
        *[f"    {line}" for line in command(f"post `{handle}' {posted}")],
        "}",
        f"postclose `{handle}'",
        f"quietly use \"`{results}'\", clear",
        "* Bootstrap standart hatası ve percentile sınırları (yüzde 2,5 ve 97,5)",
    ]
    for name in names:
        se, lo, hi = (bootstrap_key(op.result, name, statistic) for statistic in ("se", "lo", "hi"))
        lines += [
            f"quietly summarize {name}",
            f"scalar {se} = r(sd)",
            f"_pctile {name}, percentiles(2.5 97.5)",
            f"scalar {lo} = r(r1)",
            f"scalar {hi} = r(r2)",
            f'display "{name}: bootstrap SH = " %8.5f scalar({se}) ", percentile %95 GA [" %7.4f scalar({lo}) '
            f'"; " %7.4f scalar({hi}) "]"',
        ]
    lines.append("restore")
    return lines
