"""R kodu: kantil regresyon ve parametrik olmayan regresyon işlemleri (Konu 7–8).

Kantil regresyon ``quantreg::rq`` (kesin çözüm, Barrodale–Roberts) ve
``summary.rq(se = "nid", covariance = TRUE)`` ile; yerel doğrusal tahmin ve çapraz doğrulama
uygulamanın hesabıyla aynı formüllerle yazılmış kısa fonksiyonlarla.
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

NOBS_HELPER = [
    "# quantreg'de nobs() yöntemi yok: gözlem sayısı artık sayısıdır",
    "nobs.rq <- function(object, ...) length(object$residuals)",
]

LOCAL_HELPER = [
    "# Gauss çekirdekli yerel doğrusal (derece = 1) veya yerel sabit / Nadaraya–Watson (derece = 0) tahmin.",
    "# Ağırlık w = exp(−u²/2), u = (Xᵢ − x)/h: h çekirdeğin standart sapmasıdır.",
    "# Yerel doğrusal tahmin ağırlıklı en küçük karelerin sabit terimidir: (S₂T₀ − S₁T₁)/(S₀S₂ − S₁²).",
    "yerel_dogrusal <- function(x, y, noktalar, h, derece = 1) {",
    "  tamam <- !(is.na(x) | is.na(y))",
    "  x <- x[tamam]",
    "  y <- y[tamam]",
    "  sapply(noktalar, function(p) {",
    "    d <- x - p",
    "    w <- exp(-0.5 * (d / h)^2)",
    "    s0 <- sum(w); s1 <- sum(w * d); s2 <- sum(w * d^2)",
    "    t0 <- sum(w * y); t1 <- sum(w * d * y)",
    "    if (derece == 0) t0 / s0 else (s2 * t0 - s1 * t1) / (s0 * s2 - s1^2)",
    "  })",
    "}",
]

CV_HELPER = [
    "# CV(h): dışarıda bırakılan tahmin hatalarının kareler ortalaması. Birini dışarıda bırakmada gözlem",
    "# kendi tahmininden, küme-silmeli CV'de gözlemin bütün kümesi (ör. okulu) çıkarılır.",
    "cv_olcutu <- function(x, y, h_degerleri, kume = NULL) {",
    "  D <- outer(x, x, function(a, b) b - a)  # D[i, j] = x_j − x_i",
    "  ayni <- if (is.null(kume)) NULL else outer(kume, kume, \"==\")",
    "  hata <- function(W) {",
    "    WD <- W * D",
    "    s0 <- rowSums(W); s1 <- rowSums(WD); s2 <- rowSums(WD * D)",
    "    m <- (s2 * drop(W %*% y) - s1 * drop(WD %*% y)) / (s0 * s2 - s1^2)",
    "    mean((y - m)^2)",
    "  }",
    "  sonuc <- t(sapply(h_degerleri, function(h) {",
    "    W <- exp(-0.5 * (D / h)^2)",
    "    diag(W) <- 0",
    "    cv <- hata(W)",
    "    if (is.null(ayni)) return(c(cv = cv, cv_kume = NA))",
    "    W[ayni] <- 0",
    "    c(cv = cv, cv_kume = hata(W))",
    "  }))",
    "  data.frame(sonuc, row.names = h_degerleri)",
    "}",
]

BINS_HELPER = [
    "# x'in eşit genişlikli k aralığında x ve y ortalamaları (verinin özeti)",
    "aralik_ortalamalari <- function(x, y, k) {",
    "  tamam <- !(is.na(x) | is.na(y))",
    "  x <- x[tamam]",
    "  y <- y[tamam]",
    "  aralik <- pmin(floor((x - min(x)) / ((max(x) - min(x)) / k)), k - 1)",
    "  data.frame(x = tapply(x, aralik, mean), y = tapply(y, aralik, mean))",
    "}",
]


def grid_text(low: float, high: float, step: float) -> str:
    count = len(bandwidth_grid(low, high, step))
    return f"round({E.format_number(low)} + {E.format_number(step)} * (0:{count - 1}), 10)"


def summary_name(model: str) -> str:
    return f"{model}_ozet"


def operation(gen, op) -> list[str] | None:
    """Kantil ve parametrik olmayan işlemlerin R kodu; tanımadığı işlemde ``None``."""

    if isinstance(op, QuantileRegression):
        formula = f"{op.outcome} ~ " + " + ".join(op.regressors)
        quantile = E.format_number(op.q)
        label = " (medyan / LAD)" if op.q == 0.5 else ""
        lines = [f"# Kantil regresyon, τ = {quantile}{label}: kesin çözüm (rq, Barrodale–Roberts);",
                 "# çözüm tek değilse rq bunu uyarıyla bildirir"]
        lines += gen._call(op.name, "quantreg::rq", formula, op.frame, f"tau = {quantile}")
        shown = ["(Intercept)", op.regressors[0]] if len(op.regressors) > 5 else ["(Intercept)", *op.regressors]
        if op.vcov == "nid":
            summary = summary_name(op.name)
            lines += [
                "# Hendricks–Koenker standart hataları (se = \"nid\"); Hinv ve J kantiller arası fark için saklanır",
                f'{summary} <- summary({op.name}, se = "nid", covariance = TRUE)',
                f"dimnames({summary}$cov) <- list(names(coef({op.name})), names(coef({op.name})))",
                f"print(round({summary}$coefficients[c({', '.join(chr(34) + s + chr(34) for s in shown)}), 1:2], 4))",
            ]
        else:
            lines.append(f"print(round(coef({op.name})[c({', '.join(chr(34) + s + chr(34) for s in shown)})], 4))")
        return lines
    if isinstance(op, QuantileDifference):
        low, high = gen.models[op.low], gen.models[op.high]
        term = gen._key(op.low, op.term)
        a, b = summary_name(op.low), summary_name(op.high)
        weight = f"(min({E.format_number(low.q)}, {E.format_number(high.q)}) - {E.format_number(low.q)} * " \
                 f"{E.format_number(high.q)})"
        return [
            f"# {op.comment}: β̂(τ = {E.format_number(high.q)}) − β̂(τ = {E.format_number(low.q)})",
            "# İki tahmin aynı veriden gelir: Cov = (min(τ₁, τ₂) − τ₁τ₂) H₁⁻¹ X'X H₂⁻¹",
            f'j <- which(names(coef({op.low})) == "{term}")',
            f"C <- {weight} * {a}$Hinv %*% {a}$J %*% {b}$Hinv",
            f"{op.name} <- unname(coef({op.high})[j] - coef({op.low})[j])",
            f"{op.name}_se <- sqrt({a}$cov[j, j] + {b}$cov[j, j] - 2 * C[j, j])",
            f"{op.name}_z <- {op.name} / {op.name}_se",
            f"{op.name}_p <- 2 * pnorm(-abs({op.name}_z))",
            f'cat(sprintf("Fark = %.4f (SH %.4f), z = %.2f, p = %.2e\\n", {op.name}, {op.name}_se, '
            f"{op.name}_z, {op.name}_p))",
        ]
    if isinstance(op, CoefficientProfile):
        return _coefficient_profile(gen, op)
    if isinstance(op, LocalLinear):
        values = "c(" + ", ".join(E.format_number(v) for v in op.values) + ")"
        lines = [
            f"# Gauss çekirdekli yerel doğrusal tahmin: {op.y} ~ m({op.x}); h çekirdeğin standart sapmasıdır",
            f"degerler <- {values}",
            f"{op.result} <- data.frame(",
        ]
        for column, h in op.bandwidths:
            lines.append(f"  {column} = yerel_dogrusal({op.frame}${op.x}, {op.frame}${op.y}, degerler, "
                         f"{E.format_number(h)}),")
        lines += ["  row.names = degerler", ")", f"print(round({op.result}, 2))"]
        return lines
    if isinstance(op, BandwidthCV):
        return _bandwidth_cv(op)
    if isinstance(op, LocalResidual):
        frame = op.frame
        return [
            f"# {op.comment}",
            f"{frame}${op.name} <- {frame}${op.variable} - yerel_dogrusal({frame}${op.x}, {frame}${op.variable}, "
            f"{frame}${op.x}, {E.format_number(op.bandwidth)})",
        ]
    return None


def _coefficient_profile(gen, op: CoefficientProfile) -> list[str]:
    taus = ", ".join(E.format_number(tau) for tau, _ in op.models)
    term = gen._key(op.models[0][1], op.term)
    models = ", ".join(model for _, model in op.models)
    summaries = ", ".join(summary_name(model) for _, model in op.models)
    lines = [
        f"# {op.y_label}: kantiller boyunca profil ve noktasal %95 güven bandı (± {CI_MULTIPLIER}·SH)",
        f"profil <- data.frame(",
        f"  tau = c({taus}),",
        f'  katsayi = sapply(list({models}), function(m) coef(m)[["{term}"]]),',
        f'  sh = sapply(list({summaries}), function(s) sqrt(diag(s$cov))[["{term}"]])',
        ")",
        f"alt <- profil$katsayi - {CI_MULTIPLIER} * profil$sh",
        f"ust <- profil$katsayi + {CI_MULTIPLIER} * profil$sh",
        "print(round(profil, 4))",
    ]
    reference = ""
    if op.reference:
        reference_value = f'coef({op.reference})[["{gen._key(op.reference, op.term)}"]]'
        reference = f", {reference_value}"
    lines += [
        f"plot(profil$tau, profil$katsayi, type = \"n\", ylim = range(c(alt, ust{reference})),",
        f'     xlab = "{op.x_label}", ylab = "{op.y_label}", main = "{op.title}")',
        'polygon(c(profil$tau, rev(profil$tau)), c(alt, rev(ust)), col = adjustcolor("#107C89", 0.18), border = NA)',
        'lines(profil$tau, profil$katsayi, type = "b", pch = 16, col = "#107C89", lwd = 2)',
    ]
    legend = ['"Kantil regresyon"'], ['"#107C89"'], ["1"]
    if op.reference:
        lines.append(f'abline(h = {reference_value}, col = "#B3392F", lty = 2, lwd = 2)')
        legend[0].append(f'"{op.reference_label}"')
        legend[1].append('"#B3392F"')
        legend[2].append("2")
    lines += [
        f'legend("topleft", legend = c({", ".join(legend[0])}), col = c({", ".join(legend[1])}),',
        f'       lty = c({", ".join(legend[2])}), lwd = 2, bty = "n")',
    ]
    return lines


def _bandwidth_cv(op: BandwidthCV) -> list[str]:
    low, high, step = op.grid
    frame = op.frame
    used = [op.x, op.y] + ([op.cluster] if op.cluster else [])
    kume = ", kume = ornek$" + op.cluster if op.cluster else ""
    lines = [
        "# Çapraz doğrulama ölçütü CV(h), h ızgarasında" + (": birini ve kümeyi dışarıda bırakarak" if op.cluster else ""),
        f"h_izgara <- {grid_text(low, high, step)}",
        f"ornek <- na.omit({frame}[, c({', '.join(chr(34) + name + chr(34) for name in used)})])",
        f"{op.result} <- cv_olcutu(ornek${op.x}, ornek${op.y}, h_izgara{kume})",
        f"{op.name}_h <- h_izgara[which.min({op.result}$cv)]  # eşitlikte küçük h",
    ]
    if op.cluster:
        lines += [
            f"{op.name}_h_kume <- h_izgara[which.min({op.result}$cv_kume)]",
            f'cat("CV ile seçilen h:", {op.name}_h, "| küme-silmeli CV ile:", {op.name}_h_kume, "\\n")',
        ]
    else:
        lines.append(f'cat("CV ile seçilen h:", {op.name}_h, "\\n")')
    fark = f"{op.result}$cv - min({op.result}$cv)"
    lines += [
        f"plot(h_izgara, {fark}, type = \"l\", col = \"#107C89\", lwd = 2,",
        f"     ylim = range(c({fark}"
        + (f", {op.result}$cv_kume - min({op.result}$cv_kume)" if op.cluster else "") + ")),",
        f'     xlab = "{op.x_label}", ylab = "CV(h) − en küçük CV", main = "{op.title}")',
        f'abline(v = {op.name}_h, col = "#107C89", lty = 3)',
    ]
    if op.cluster:
        lines += [
            f'lines(h_izgara, {op.result}$cv_kume - min({op.result}$cv_kume), col = "#B3392F", lwd = 2)',
            f'abline(v = {op.name}_h_kume, col = "#B3392F", lty = 3)',
            'legend("topright", legend = c("Birini dışarıda bırak", "Küme-silmeli"), col = c("#107C89", "#B3392F"),',
            '       lwd = 2, bty = "n")',
        ]
    return lines
