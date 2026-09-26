"""R kodu: regresyon süreksizliği ve bootstrap işlemleri (Konu 9–10).

RDD ağırlıklı ``lm`` ile tahmin edilir, standart hata ``sandwich::vcovHC(type = "HC1")``: Python
(statsmodels WLS, HC1) ve Stata (``regress [aw = w], vce(robust)``) ile aynı sayı. Bootstrap tekrarları
``lm.fit`` ile çözülür; R'nin rastgele sayı üreteci farklı olduğu için sonuçlar dağılımda aynıdır.
"""

from __future__ import annotations

from core.codegen.python_rdd_boot import KERNEL_NAMES, SCALE_NAMES, rdd_comment
from core.labs import expr as E
from core.labs.runner import CI_MULTIPLIER, bootstrap_key, uses_replicate_se
from core.labs.spec import BOOT, RDD, Bootstrap, RDDTable, ScalarTable

RDD_HELPER = [
    "# Keskin RDD: eşikte yerel doğrusal sıçrama (D katsayısı), ağırlıklı EKK; SH: vcovHC(type = \"HC1\").",
    "# Pozitif ağırlıklı gözlemlerde Y'nin D = 1{X >= c}, R = X - c ve D·R üzerine çekirdek ağırlıklı EKK'si.",
    "# Hansen ölçeği: çekirdek birim varyanslıdır ve h çekirdeğin standart sapmasıdır; üçgen çekirdekte ağırlık",
    "# eşikten h√6, dikdörtgende h√3 uzaklıkta sıfırlanır. olcek = \"pencere\": h pencerenin yarı genişliğidir.",
    'rdd_yerel_dogrusal <- function(veri, x, y, esik, h, cekirdek = "ucgen", olcek = "hansen") {',
    '  pencere <- if (olcek == "hansen") h * sqrt(if (cekirdek == "ucgen") 6 else 3) else h',
    "  ornek <- data.frame(Y = veri[[y]], R = veri[[x]] - esik)",
    "  ornek <- ornek[complete.cases(ornek), ]",
    '  ornek$w <- if (cekirdek == "ucgen") pmax(1 - abs(ornek$R) / pencere, 0) else as.numeric(abs(ornek$R) <= pencere)',
    "  ornek <- ornek[ornek$w > 0, ]",
    "  ornek$D <- as.numeric(ornek$R >= 0)",
    "  ornek$DR <- ornek$D * ornek$R",
    "  lm(Y ~ D + R + DR, data = ornek, weights = w)",
    "}",
]

CURVE_HELPER = [
    "# Eşiğin iki yanında ayrı yerel doğrusal tahmin ve noktasal %95 güven bandı. Her x0 noktasında yalnız o",
    "# taraftaki gözlemlerle üçgen çekirdekli (pencere ±h√6) ağırlıklı EKK'nin sabit terimi:",
    "# (S2·T0 − S1·T1)/(S0·S2 − S1²); SH o yerel regresyonun HC1 sandviçi (k/(k−2) çarpanıyla).",
    "rdd_taraf <- function(x, y, noktalar, pencere) {",
    "  t(sapply(noktalar, function(p) {",
    "    d <- x - p",
    "    w <- pmax(1 - abs(d) / pencere, 0)",
    "    k <- sum(w > 0)",
    "    if (k <= 2) return(c(NA, NA))",
    "    s0 <- sum(w); s1 <- sum(w * d); s2 <- sum(w * d^2)",
    "    t0 <- sum(w * y); t1 <- sum(w * d * y)",
    "    det <- s0 * s2 - s1^2",
    "    a <- (s2 * t0 - s1 * t1) / det",
    "    b <- (s0 * t1 - s1 * t0) / det",
    "    u <- (w * (y - a - b * d))^2",
    "    v <- (s2^2 * sum(u) - 2 * s1 * s2 * sum(u * d) + s1^2 * sum(u * d^2)) / det^2 * k / (k - 2)",
    "    c(a, sqrt(v))",
    "  }))",
    "}",
    "rdd_egrisi <- function(x, y, esik, h, alt, ust, nokta = 120) {",
    "  tamam <- !(is.na(x) | is.na(y))",
    "  x <- x[tamam]",
    "  y <- y[tamam]",
    "  sol <- seq(alt, esik, length.out = nokta)",
    "  sag <- seq(esik, ust, length.out = nokta)",
    "  L <- rbind(rdd_taraf(x[x < esik], y[x < esik], sol, h * sqrt(6)),",
    "             rdd_taraf(x[x >= esik], y[x >= esik], sag, h * sqrt(6)))",
    f"  data.frame(x = c(sol, sag), tahmin = L[, 1], alt = L[, 1] - {CI_MULTIPLIER} * L[, 2],",
    f"             ust = L[, 1] + {CI_MULTIPLIER} * L[, 2], sag = rep(c(FALSE, TRUE), each = nokta))",
    "}",
]

HC1_HELPER = [
    "# HC1 standart hataları: n/(n−k)·(X'X)⁻¹(Σ eᵢ²xᵢxᵢ')(X'X)⁻¹ (percentile-t için her tekrarda)",
    "hc1_sh <- function(X, y, b) {",
    "  ters <- solve(crossprod(X))",
    "  skor <- X * drop(y - X %*% b)",
    "  sqrt(diag(nrow(X) / (nrow(X) - ncol(X)) * ters %*% crossprod(skor) %*% ters))",
    "}",
]

_METHOD_COMMENTS = {
    "pairs": "Pairs bootstrap: gözlem satırları yerine koyarak çekilir; model her tekrarda baştan tahmin edilir",
    "wild": "Wild bootstrap: regresörler sabit; Y* = Xβ̂ + ê·ξ, ξ Rademacher (±1, eşit olasılıkla)",
    "cluster": "Küme bootstrap'ı: kümeler yerine koyarak çekilir; seçilen kümenin bütün gözlemleri birlikte gelir",
}


def _number(value: float) -> str:
    return E.format_number(value)


def operation(gen, op) -> list[str] | None:
    """RDD ve bootstrap işlemlerinin R kodu; tanımadığı işlemde ``None``."""

    if isinstance(op, RDD):
        arguments = [op.frame, f'"{op.x}"', f'"{op.y}"', _number(op.cutoff), _number(op.bandwidth)]
        if op.kernel != "triangular":
            arguments.append(f'cekirdek = "{KERNEL_NAMES[op.kernel]}"')
        if op.scale != "hansen":
            arguments.append(f'olcek = "{SCALE_NAMES[op.scale]}"')
        return [
            f"# {rdd_comment(op)}",
            f"{op.name} <- rdd_yerel_dogrusal({', '.join(arguments)})",
            f'print(coeftest({op.name}, vcov. = {gen.vcov(op.name)})["D", , drop = FALSE])',
        ]
    if isinstance(op, RDDTable):
        return _rdd_table(gen, op)
    if isinstance(op, Bootstrap):
        return _bootstrap(gen, op)
    if isinstance(op, ScalarTable):
        dialect = gen.dialect("")
        entries = [f'  "{label}" = {E.render(expression, dialect)}' for label, expression in op.rows]
        return [
            f"{op.result} <- data.frame(deger = c(",
            ",\n".join(entries),
            "))",
            f"print(round({op.result}, {op.decimals}))",
        ]
    return None


def _rdd_table(gen, op: RDDTable) -> list[str]:
    models = ", ".join(model for _, model in op.rows)
    rows = "c(" + ", ".join(_number(h) for h, _ in op.rows) + ")"
    table = op.result
    return [
        f"# Bant genişliği duyarlılığı: her h için sıçrama, HC1 SH ve %95 güven aralığı (tahmin ± {CI_MULTIPLIER}·SH)",
        f"{table} <- data.frame(t(sapply(list({models}), function(m) c(",
        '  n = nobs(m), tahmin = coef(m)[["D"]], sh = sqrt(diag(sandwich::vcovHC(m, type = "HC1")))[["D"]]',
        f"))), row.names = {rows})",
        f"{table}$alt <- {table}$tahmin - {CI_MULTIPLIER} * {table}$sh",
        f"{table}$ust <- {table}$tahmin + {CI_MULTIPLIER} * {table}$sh",
        f"print(round({table}, 2))",
        f"h_degerleri <- {rows}",
        f"plot(h_degerleri, {table}$tahmin, pch = 16, col = \"#107C89\", ylim = range(c({table}$alt, {table}$ust, 0)),",
        f'     xlab = "{op.x_label}", ylab = "{op.y_label}", main = "{op.title}")',
        f'arrows(h_degerleri, {table}$alt, h_degerleri, {table}$ust, angle = 90, code = 3, length = 0.05,',
        '       col = "#107C89", lwd = 2)',
        'abline(h = 0, col = "#07373D", lty = 3)',
        'legend("bottomright", legend = "Tahmin ve %95 güven aralığı", col = "#107C89", pch = 16, lty = 1, bty = "n")',
    ]


def _dialect(gen) -> E.Dialect:
    """Tekrarın katsayısı ``b[[...]]``, HC1 SH'si ``s[[...]]``; diğer modeller orijinal tahmin."""

    base = gen.dialect("")

    def coefficient(model: str, term: str) -> str:
        if model == BOOT:
            return f'b[["{gen._key(model, term)}"]]'
        return base.coefficient(model, term)

    def standard_error(model: str, term: str) -> str:
        if model == BOOT:
            return f's[["{gen._key(model, term)}"]]'
        return base.standard_error(model, term)

    return E.Dialect(variable=base.variable, coefficient=coefficient, functions=base.functions, power=base.power,
                     standard_error=standard_error, scalar=base.scalar)


def _bootstrap(gen, op: Bootstrap) -> list[str]:
    model = op.model
    need_se = any(uses_replicate_se(expression) for _, expression in op.collect)
    lines = [
        f"# {op.comment}",
        f"# {_METHOD_COMMENTS[op.method]}",
        f"X_b <- model.matrix({model})  # tahmin örnekleminin tasarım matrisi (sabit dahil)",
        f"y_b <- model.response(model.frame({model}))",
    ]
    if op.method == "wild":
        lines += [f"uyum <- fitted({model})", f"artik <- resid({model})"]
    if op.method == "cluster":
        lines += [
            f'kume_b <- {op.frame}[rownames(X_b), "{op.cluster}"]',
            "uyeler <- split(seq_len(nrow(X_b)), kume_b)",
        ]
    if op.seed is not None:
        lines.append(f"set.seed({op.seed})")
    else:
        lines.append("# Rastgele sayı üreteci veri çekilişlerinin ardından kaldığı yerden devam eder")
    lines += [f'tekrarlar <- vector("list", {op.reps})', f"for (tekrar in seq_len({op.reps})) {{"]
    if op.method == "pairs":
        lines += [
            "  i <- sample.int(nrow(X_b), nrow(X_b), replace = TRUE)  # satırları yerine koyarak çek",
            "  X_t <- X_b[i, , drop = FALSE]",
            "  y_t <- y_b[i]",
        ]
    elif op.method == "wild":
        lines += ["  X_t <- X_b", "  y_t <- uyum + artik * sample(c(-1, 1), length(artik), replace = TRUE)"]
    else:
        lines += [
            "  secilen <- sample.int(length(uyeler), length(uyeler), replace = TRUE)  # kümeleri yerine koyarak çek",
            "  i <- unlist(uyeler[secilen], use.names = FALSE)",
            "  X_t <- X_b[i, , drop = FALSE]",
            "  y_t <- y_b[i]",
        ]
    lines.append("  b <- coef(lm.fit(X_t, y_t))")
    if need_se:
        lines.append("  s <- hc1_sh(X_t, y_t, b)")
    dialect = _dialect(gen)
    entries = ", ".join(f"{name} = {E.render(expression, dialect)}" for name, expression in op.collect)
    lines += [
        f"  tekrarlar[[tekrar]] <- c({entries})",
        "}",
        f"{op.result} <- as.data.frame(do.call(rbind, tekrarlar))",
        "# Bootstrap standart hatası ve percentile sınırları (yüzde 2,5 ve 97,5; quantile tür 7 = NumPy varsayılanı)",
    ]
    for name, _ in op.collect:
        se, lo, hi = (bootstrap_key(op.result, name, statistic) for statistic in ("se", "lo", "hi"))
        lines += [
            f"{se} <- sd({op.result}${name})",
            f"{lo} <- unname(quantile({op.result}${name}, 0.025))",
            f"{hi} <- unname(quantile({op.result}${name}, 0.975))",
            f'cat(sprintf("{name}: bootstrap SH = %.5f, percentile %%95 GA [%.4f; %.4f]\\n", {se}, {lo}, {hi}))',
        ]
    return lines
