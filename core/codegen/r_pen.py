"""R kodu: model seçimi, düzenlileştirme ve DML işlemleri (Konu 11–12).

Lasso ve Elastic Net ``glmnet`` ile (``standardize = FALSE``, açık ceza ızgarası, ``thresh = 1e-16``; yol erken
kesilmesin diye ``glmnet.control(fdev = 0, devmax = 1)``), Ridge kapalı biçimle (özdeğer ayrışımı), cezasız EKK QR ile
çözülür. Katlar ve ızgara kodda açıkça verildiği için sayılar Python ile aynıdır.
"""

from __future__ import annotations

from core.codegen.base import DARK, GRAY, SERIES_COLORS
from core.codegen.python_pen import SCALE_NOTES
from core.labs import expr as E
from core.labs.runner import CI_MULTIPLIER
from core.labs.spec import (
    CoefPath,
    ComplexityCurve,
    CrossFitDML,
    CVCurve,
    Dictionary,
    DMLSplits,
    DotPlot,
    DoubleSelection,
    DrawColumns,
    EstimatePlot,
    GroupRank,
    LinePlot,
    ModelMetrics,
    Penalized,
    PostSelection,
    RowNumber,
    dictionary_terms,
    dml_fold_key,
    penalized_key,
)

PENALIZED_HELPER = [
    "# glmnet yolu erken kesmesin: ızgaranın bütün cezaları hesaplansın",
    "invisible(glmnet::glmnet.control(fdev = 0, devmax = 1))",
    "",
    "# 10^ust'ten 10^alt'a logaritmik eşit aralıklı ceza ızgarası, büyükten küçüğe",
    "ceza_izgarasi <- function(ust, alt, nokta) 10^seq(ust, alt, length.out = nokta)",
    "",
    "# Eğitim verisinden öğrenilen ölçek ve kategori düzeyleriyle iki tasarım matrisi. Sürekli değişkenler eğitim",
    "# ortalaması ve (n'e bölünen) standart sapmasıyla ölçeklenir; kategori göstergeleri eğitimde görülen",
    "# düzeylerden kurulur, ilk düzey referanstır.",
    "tasarim <- function(egitim, diger, sayisal, kategorik = character(0), olcekle = TRUE) {",
    "  xa <- as.matrix(egitim[, sayisal, drop = FALSE])",
    "  xb <- as.matrix(diger[, sayisal, drop = FALSE])",
    "  if (olcekle) {",
    "    ortalama <- colMeans(xa)",
    "    ss <- sqrt(colMeans(sweep(xa, 2, ortalama)^2))",
    '    xa <- sweep(sweep(xa, 2, ortalama), 2, ss, "/")',
    '    xb <- sweep(sweep(xb, 2, ortalama), 2, ss, "/")',
    "  }",
    "  for (ad in kategorik) {",
    "    for (duzey in sort(unique(egitim[[ad]]))[-1]) {",
    "      xa <- cbind(xa, as.numeric(egitim[[ad]] == duzey))",
    "      xb <- cbind(xb, as.numeric(diger[[ad]] == duzey))",
    '      colnames(xa)[ncol(xa)] <- colnames(xb)[ncol(xb)] <- paste0(ad, "=", duzey)',
    "    }",
    "  }",
    "  list(a = xa, b = xb)",
    "}",
    "",
    "# Izgaradaki her ceza için sabit terim, katsayılar ve xb tahminleri; sabit terim cezasızdır (veri merkezlenir).",
    "# ceza: \"ols\"; \"ridge\" (SSE ölçeği, (Y − Xβ)'(Y − Xβ) + λβ'β); \"lasso\" veya \"enet\" (glmnet ölçeği,",
    "# (1/(2n))‖Y − Xβ‖² + λ[r‖β‖₁ + (1 − r)/2·‖β‖²], r = l1 = glmnet alpha).",
    'ceza_yolu <- function(xa, ya, xb, izgara, ceza, l1 = 1) {',
    "  xo <- colMeans(xa)",
    "  yo <- mean(ya)",
    "  xc <- sweep(xa, 2, xo)",
    "  yc <- ya - yo",
    '  if (ceza == "ols") {',
    "    katsayi <- matrix(qr.solve(xc, yc), ncol(xa), length(izgara))",
    '  } else if (ceza == "ridge") {',
    "    e <- eigen(crossprod(xc), symmetric = TRUE)",
    "    r <- drop(crossprod(e$vectors, crossprod(xc, yc)))",
    "    katsayi <- matrix(sapply(izgara, function(l) e$vectors %*% (r / (e$values + l))), nrow = ncol(xa))",
    "  } else {",
    '    oran <- if (ceza == "lasso") 1 else l1',
    "    f <- glmnet::glmnet(xc, yc, alpha = oran, lambda = izgara, standardize = FALSE, intercept = FALSE,",
    "                        thresh = 1e-16, maxit = 1e7)",
    "    katsayi <- matrix(as.matrix(f$beta), nrow = ncol(xa))",
    "  }",
    "  sabit <- yo - drop(xo %*% katsayi)",
    '  list(sabit = sabit, katsayi = katsayi, tahmin = sweep(xb %*% katsayi, 2, sabit, "+"))',
    "}",
    "",
    "# K-katlı CV: ölçek ve göstergeler her katta o katın eğitim verisinden öğrenilir. Ölçüt kat ortalama karesel",
    "# hatalarının ağırlıksız ortalaması, SH katlar arası standart sapma/√K. \"min\": en küçük CV (eşitlikte en büyük",
    "# ceza); \"1se\": en küçüğün bir SH'si içindeki en büyük ceza.",
    'ceza_cv <- function(veri, y, sayisal, kategorik, kat, izgara, ceza, kural = "min", l1_oranlari = 1,',
    "                    olcekle = TRUE) {",
    "  anahtarlar <- sort(unique(kat))",
    '  oranlar <- if (ceza == "enet") l1_oranlari else 1',
    "  hata <- array(NA_real_, c(length(oranlar), length(anahtarlar), length(izgara)))",
    "  for (j in seq_along(anahtarlar)) {",
    "    e <- kat != anahtarlar[j]",
    "    x <- tasarim(veri[e, , drop = FALSE], veri[!e, , drop = FALSE], sayisal, kategorik, olcekle)",
    "    for (r in seq_along(oranlar)) {",
    "      yol <- ceza_yolu(x$a, veri[[y]][e], x$b, izgara, ceza, oranlar[r])",
    "      hata[r, j, ] <- colMeans((veri[[y]][!e] - yol$tahmin)^2)",
    "    }",
    "  }",
    "  ortalama <- apply(hata, c(1, 3), mean)",
    "  sh <- apply(hata, c(1, 3), sd) / sqrt(length(anahtarlar))",
    "  i <- which.min(t(ortalama)) - 1",
    "  r <- i %/% length(izgara) + 1",
    "  g <- i %% length(izgara) + 1",
    '  if (kural == "1se") g <- which(ortalama[r, ] <= ortalama[r, g] + sh[r, g])[1]',
    "  list(l1 = oranlar[r], lam = izgara[g],",
    "       tablo = data.frame(lambda = izgara, cv_ort = ortalama[r, ], cv_sh = sh[r, ]))",
    "}",
    "",
    "cezali_sonuc <- function(sabit, b, lam, oran, yv, tahmin, e, cv = NULL, yol = NULL) {",
    "  artik <- yv - tahmin",
    '  list(params = c("(Intercept)" = sabit, b), lam = lam, l1 = oran,',
    "       test_mse = if (any(!e)) mean(artik[!e]^2) else NA, egitim_mse = mean(artik[e]^2),",
    "       sifirdan = sum(b != 0), norm = sqrt(sum(b^2)), cv = cv, yol = yol, tahmin = tahmin,",
    "       secilen = names(b)[b != 0], nobs = sum(e))",
    "}",
    "",
    "# Eğitim satırlarında (egitim = 1) tahmin; ceza eğitim katlarıyla (kat) CV'de seçilir; test: egitim = 0.",
    'ceza_modeli <- function(veri, y, sayisal, kategorik = character(0), ceza = "ols", izgara = NULL, egitim = NULL,',
    '                        kat = NULL, kural = "min", l1_oranlari = 1, olcekle = TRUE, yol = FALSE) {',
    "  e <- if (is.null(egitim)) rep(TRUE, nrow(veri)) else egitim == 1",
    "  egt <- veri[e, , drop = FALSE]",
    '  oran <- if (ceza == "enet") l1_oranlari[1] else 1',
    "  lam <- 0",
    "  cv <- NULL",
    '  if (ceza != "ols") {',
    "    s <- ceza_cv(egt, y, sayisal, kategorik, kat[e], izgara, ceza, kural, l1_oranlari, olcekle)",
    "    oran <- s$l1",
    "    lam <- s$lam",
    "    cv <- s$tablo",
    "  }",
    "  x <- tasarim(egt, veri, sayisal, kategorik, olcekle)",
    "  yv <- veri[[y]]",
    "  f <- ceza_yolu(x$a, yv[e], x$b, lam, ceza, oran)",
    "  yol_tablosu <- NULL",
    "  if (yol) {  # bütün eğitim örnekleminde ızgara boyunca katsayılar",
    "    k <- ceza_yolu(x$a, yv[e], x$a[1, , drop = FALSE], izgara, ceza, oran)$katsayi",
    "    yol_tablosu <- list(lambda = izgara, katsayi = matrix(t(k), ncol = ncol(x$a),",
    "                                                           dimnames = list(NULL, colnames(x$a))))",
    "  }",
    "  cezali_sonuc(f$sabit[1], setNames(drop(f$katsayi), colnames(x$a)), lam, oran, yv, drop(f$tahmin), e, cv,",
    "               yol_tablosu)",
    "}",
]

POST_HELPER = [
    "# Post-Lasso: kaynak modelin sıfırdan farklı katsayılı terimleri üzerinde cezasız EKK (aynı örneklem ve ölçek)",
    "post_secim <- function(veri, y, kaynak, sayisal, kategorik = character(0), egitim = NULL, olcekle = TRUE) {",
    "  e <- if (is.null(egitim)) rep(TRUE, nrow(veri)) else egitim == 1",
    "  x <- tasarim(veri[e, , drop = FALSE], veri, sayisal, kategorik, olcekle)",
    "  sec <- match(kaynak$secilen, colnames(x$a))",
    "  yv <- veri[[y]]",
    '  f <- ceza_yolu(x$a[, sec, drop = FALSE], yv[e], x$b[, sec, drop = FALSE], 0, "ols")',
    "  b <- setNames(rep(0, ncol(x$a)), colnames(x$a))",
    "  b[sec] <- f$katsayi[, 1]",
    "  cezali_sonuc(f$sabit[1], b, 0, 1, yv, drop(f$tahmin), e)",
    "}",
]

DML_HELPER = [
    "# θ̂ = Σv̂û/Σv̂² (sabitsiz); SH: HC1 (n/(n − 1)) ya da küme-dayanıklı (skorlar küme içinde toplanır, G/(G − 1))",
    "artik_regresyonu <- function(u, v, kume = NULL) {",
    "  teta <- sum(v * u) / sum(v^2)",
    "  skor <- v * (u - teta * v)",
    "  if (is.null(kume)) {",
    "    orta <- sum(skor^2) * length(u) / (length(u) - 1)",
    "  } else {",
    "    toplam <- tapply(skor, kume, sum)",
    "    orta <- sum(toplam^2) * length(toplam) / (length(toplam) - 1)",
    "  }",
    "  c(theta = teta, se = sqrt(orta) / sum(v^2))",
    "}",
    "",
    "# DML2: her dış kat k için m_Y(X) = E[Y|X] ve m_D(X) = E[D|X] k dışındaki gözlemlerde öğrenilir, k'de artık",
    "# alınır. dis_kat NULL: çapraz uyarlama yok (artıklaştırma). ogrenici \"lasso\": ceza iç katlarla (ic_kat)",
    "# CV'de seçilir; \"ols\": cezasız EKK. Sonuç coef(), vcov() ve nobs() ile okunur (terim \"theta\").",
    'dml_capraz <- function(veri, y, d, ozellikler, dis_kat = NULL, ic_kat = NULL, izgara = NULL, kural = "min",',
    '                       kume = NULL, ogrenici = "lasso") {',
    "  n <- nrow(veri)",
    "  dis <- if (is.null(dis_kat)) rep(1, n) else dis_kat",
    "  sapka <- list(y = numeric(n), d = numeric(n))",
    "  katlar <- list()",
    "  for (k in sort(unique(dis))) {",
    "    degerlendirme <- dis == k",
    "    e <- if (is.null(dis_kat)) rep(TRUE, n) else !degerlendirme",
    "    egt <- veri[e, , drop = FALSE]",
    "    kayit <- c(kat = k)",
    '    for (ek in c("y", "d")) {',
    '      hedef <- if (ek == "y") y else d',
    '      if (ogrenici == "ols") {',
    "        lam <- 0",
    '        ceza <- "ols"',
    "      } else {",
    '        lam <- ceza_cv(egt, hedef, ozellikler, character(0), ic_kat[e], izgara, "lasso", kural)$lam',
    '        ceza <- "lasso"',
    "      }",
    "      x <- tasarim(egt, veri[degerlendirme, , drop = FALSE], ozellikler)",
    "      f <- ceza_yolu(x$a, egt[[hedef]], x$b, lam, ceza)",
    "      sapka[[ek]][degerlendirme] <- drop(f$tahmin)",
    '      kayit[paste0("lambda_", ek)] <- lam',
    '      kayit[paste0("sifirdan_", ek)] <- sum(f$katsayi != 0)',
    "    }",
    "    katlar[[length(katlar) + 1]] <- kayit",
    "  }",
    "  u <- veri[[y]] - sapka$y",
    "  v <- veri[[d]] - sapka$d",
    "  s <- artik_regresyonu(u, v, kume)",
    '  structure(list(theta = s[["theta"]], se = s[["se"]], n = n, u = u, v = v,',
    '                 katlar = as.data.frame(do.call(rbind, katlar))), class = "dml_sonuc")',
    "}",
    'coef.dml_sonuc <- function(object, ...) c(theta = object$theta)',
    'vcov.dml_sonuc <- function(object, ...) matrix(object$se^2, 1, 1, dimnames = list("theta", "theta"))',
    "nobs.dml_sonuc <- function(object, ...) object$n",
]

DOUBLE_SELECTION_HELPER = [
    "# S_Y: Y ve her X, [1, D] üzerinde artıklaştırıldıktan sonra (D cezalanmasın diye) Lasso; S_D: D'nin X",
    "# üzerine Lasso'su. Yalnız-sonuç Post-Lasso Y ~ D + S_Y, double selection Y ~ D + (S_Y ∪ S_D); SH: HC1.",
    'cift_secim <- function(veri, y, d, kontroller, kat, izgara, kural = "1se") {',
    "  Z <- cbind(1, veri[[d]])",
    "  blok <- as.matrix(veri[, c(kontroller, y)])",
    "  artik <- as.data.frame(blok - Z %*% qr.solve(Z, blok))",
    '  s_y <- ceza_modeli(artik, y, kontroller, character(0), "lasso", izgara, kat = kat, kural = kural)$secilen',
    '  s_d <- ceza_modeli(veri, d, kontroller, character(0), "lasso", izgara, kat = kat, kural = kural)$secilen',
    "  birlesim <- kontroller[kontroller %in% c(s_y, s_d)]",
    "  ols <- function(adlar) lm(reformulate(c(d, adlar), response = y), data = veri)",
    "  list(sonuc = ols(s_y), cift = ols(birlesim), kumeler = list(sonuc = s_y, tedavi = s_d, birlesim = birlesim))",
    "}",
]

COMPLEXITY_HELPER = [
    "# Polinom derecesi 1, …, D: eğitim MSE (σ̂² = SSR/n), AIC = n + n·log(2πσ̂²) + 2K, BIC = n + n·log(2πσ̂²) + K·log n",
    "# (K = d + 2: katsayılar ve σ²), LOOCV (kaldıraçla kesin) ve test MSE.",
    "karmasiklik <- function(veri, x, y, en_buyuk, egitim) {",
    "  e <- egitim == 1",
    "  satirlar <- lapply(seq_len(en_buyuk), function(d) {",
    '    X <- outer(veri[[x]], 0:d, "^")',
    "    q <- qr(X[e, , drop = FALSE])",
    "    b <- qr.coef(q, veri[[y]][e])",
    "    artik <- veri[[y]][e] - drop(X[e, , drop = FALSE] %*% b)",
    "    h <- rowSums(qr.Q(q)^2)  # kaldıraç değerleri",
    "    n <- sum(e)",
    "    K <- d + 2",
    "    s2 <- sum(artik^2) / n",
    "    temel <- n + n * log(2 * pi * s2)",
    "    c(derece = d, egitim_mse = s2, loocv = mean((artik / (1 - h))^2),",
    "      test_mse = mean((veri[[y]][!e] - drop(X[!e, , drop = FALSE] %*% b))^2), aic = temel + 2 * K,",
    "      bic = temel + K * log(n))",
    "  })",
    "  tablo <- as.data.frame(do.call(rbind, satirlar))",
    "  rownames(tablo) <- tablo$derece",
    "  tablo",
    "}",
]

HELPERS = {
    "ceza": PENALIZED_HELPER,
    "post": POST_HELPER,
    "dml": DML_HELPER,
    "cift": DOUBLE_SELECTION_HELPER,
    "karmasiklik": COMPLEXITY_HELPER,
}


def _number(value: float) -> str:
    return E.format_number(value)


def _quoted(names) -> str:
    return ", ".join(f'"{name}"' for name in names)


def _wrapped(opening: str, items: list[str], closing: str, width: int = 100) -> list[str]:
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


def list_definition(name: str, values) -> list[str]:
    return _wrapped(f"{name} <- c(", [f'"{value}"' for value in values], ")")


def _list(gen, values) -> tuple[list[str], str]:
    lines, name = gen.use_list(values)
    if name is not None:
        return lines, name
    return [], f"c({_quoted(values)})"


def _grid(spec) -> str:
    high, low, count = spec
    return f"ceza_izgarasi({_number(high)}, {_number(low)}, {int(count)})"


def packages(ops) -> list[str]:
    if any(
        (isinstance(op, Penalized) and op.penalty in ("lasso", "enet"))
        or (isinstance(op, CrossFitDML) and op.learner == "lasso")
        or isinstance(op, (DoubleSelection, DMLSplits))
        for op in ops
    ):
        return ["glmnet"]
    return []


def operation(gen, op) -> list[str] | None:
    """Konu 11–12 işlemlerinin R kodu; tanımadığı işlemde ``None``."""

    if isinstance(op, RowNumber):
        return [f"# {op.comment}", f"{op.frame}${op.name} <- seq_len(nrow({op.frame}))"]
    if isinstance(op, GroupRank):
        return [
            f"# {op.comment}",
            f"{op.frame}${op.name} <- match({op.frame}${op.source}, sort(unique({op.frame}${op.source})))",
        ]
    if isinstance(op, Dictionary):
        return _dictionary(gen, op)
    if isinstance(op, DrawColumns):
        return _draw_columns(op)
    if isinstance(op, Penalized):
        return _penalized(gen, op)
    if isinstance(op, PostSelection):
        return _post_selection(gen, op)
    if isinstance(op, ModelMetrics):
        return _metrics(op)
    if isinstance(op, DotPlot):
        return _dot_plot(op)
    if isinstance(op, CVCurve):
        return _cv_curve(op)
    if isinstance(op, CoefPath):
        return _coef_path(op)
    if isinstance(op, CrossFitDML):
        return _crossfit(gen, op)
    if isinstance(op, DMLSplits):
        return _splits(gen, op)
    if isinstance(op, DoubleSelection):
        return _double_selection(gen, op)
    if isinstance(op, EstimatePlot):
        return _estimate_plot(gen, op)
    if isinstance(op, ComplexityCurve):
        return [
            "# Polinom derecesi boyunca model seçim ölçütleri; en küçük değerlerin dereceleri yazdırılır",
            f'{op.result} <- karmasiklik({op.frame}, "{op.x}", "{op.y}", {op.max_degree}, {op.frame}${op.sample})',
            f"print(round({op.result}, 4))",
            *[f"{op.name}_d_{column} <- {op.result}$derece[which.min({op.result}${column})]"
              for column in ("egitim_mse", "loocv", "test_mse", "aic", "bic")],
            f'cat(sprintf("En küçük ölçütün derecesi: LOOCV %d, AIC %d, BIC %d, test %d\\n", {op.name}_d_loocv, '
            f"{op.name}_d_aic, {op.name}_d_bic, {op.name}_d_test_mse))",
        ]
    if isinstance(op, LinePlot):
        return _line_plot(op)
    return None


def _dictionary(gen, op: Dictionary) -> list[str]:
    frame = op.frame
    lines = [f"# {op.comment}"]
    if op.powers:
        lines += [
            f"for (ad in c({_quoted(op.powers)})) {{",
            f"  for (kuvvet in 2:{op.degree}) {frame}[[paste0(ad, \"_\", kuvvet)]] <- {frame}[[ad]]^kuvvet",
            "}",
        ]
    if op.interactions:
        lines += [
            *_wrapped("taban <- c(", [f'"{name}"' for name in op.base], ")"),
            "for (i in seq_along(taban)) {",
            "  for (j in seq_along(taban)) {",
            f"    if (i < j) {frame}[[paste0(taban[i], \"_x_\", taban[j])]] <- {frame}[[taban[i]]] * {frame}[[taban[j]]]",
            "  }",
            "}",
        ]
    definition, name = gen.use_list(dictionary_terms(op))
    lines += definition
    if name is not None:
        lines.append(f'cat(length({name}), "terim\\n")')
    return lines


def _draw_columns(op: DrawColumns) -> list[str]:
    frame, count, prefix = op.frame, op.count, op.prefix
    if op.rho == 0:
        body = [f"for (j in seq_len({count})) sutunlar[, j] <- rnorm(nrow({frame}))"]
    else:
        rho = _number(op.rho)
        body = [
            f"# x1 = z1, xj = {rho}·x(j−1) + √(1 − {rho}²)·zj: Corr(xj, xk) = {rho}^|j − k|, her sütunun varyansı 1",
            f"for (j in seq_len({count})) {{",
            f"  z <- rnorm(nrow({frame}))",
            f"  sutunlar[, j] <- if (j == 1) z else {rho} * sutunlar[, j - 1] + sqrt(1 - {rho}^2) * z",
            "}",
        ]
    return [
        f"# {op.comment}",
        f"sutunlar <- matrix(0, nrow({frame}), {count}, dimnames = list(NULL, paste0(\"{prefix}\", seq_len({count}))))",
        *body,
        f"{frame} <- cbind({frame}, sutunlar)",
    ]


_ATTRIBUTES = {"test_mse": "test_mse", "egitim_mse": "egitim_mse", "sifirdan": "sifirdan", "norm": "norm",
               "lambda": "lam", "l1": "l1"}


def _scalar_lines(gen, model: str) -> list[str]:
    return [
        f"{penalized_key(model, quantity)} <- {model}${attribute}"
        for quantity, attribute in _ATTRIBUTES.items()
        if penalized_key(model, quantity) in gen.scalar_refs
    ]


def _penalized(gen, op: Penalized) -> list[str]:
    lines, numeric = _list(gen, op.numeric)
    categorical = "character(0)"
    if op.categorical:
        extra, categorical = _list(gen, op.categorical)
        lines += extra
    arguments = [op.frame, f'"{op.outcome}"', numeric, categorical, f'"{op.penalty}"']
    if op.penalty != "ols":
        arguments.append(_grid(op.grid))
    if op.sample:
        arguments.append(f"egitim = {op.frame}${op.sample}")
    if op.penalty != "ols":
        arguments.append(f"kat = {op.frame}${op.folds}")
        if op.rule != "min":
            arguments.append(f'kural = "{op.rule}"')
        if op.penalty == "enet":
            arguments.append(f"l1_oranlari = c({', '.join(_number(r) for r in op.l1_ratios)})")
    if not op.standardize:
        arguments.append("olcekle = FALSE")
    if op.path:
        arguments.append("yol = TRUE")
    if op.penalty == "ols":
        lines.append(f"# {op.name}: cezasız EKK" + (" (eğitim örnekleminde)" if op.sample else ""))
    else:
        high, low, count = op.grid
        rule = "en küçük CV" if op.rule == "min" else "bir standart hata kuralı"
        lines.append(f"# {SCALE_NOTES[op.penalty]}; ceza {count} noktalı ızgarada (10^{_number(high)} … "
                     f"10^{_number(low)}) CV ile, {rule}")
    lines += _wrapped(f"{op.name} <- ceza_modeli(", arguments, ")")
    lines += _scalar_lines(gen, op.name)
    if op.penalty == "ols":
        lines.append(f'cat(sprintf("{op.name}: test MSE = %.4f, katsayı = %d\\n", {op.name}$test_mse, {op.name}$sifirdan))')
    else:
        lines.append(
            f'cat(sprintf("{op.name}: λ = %.6g, test MSE = %.4f, sıfırdan farklı katsayı = %d\\n", {op.name}$lam, '
            f"{op.name}$test_mse, {op.name}$sifirdan))"
        )
    return lines


def _post_selection(gen, op: PostSelection) -> list[str]:
    source: Penalized = gen.models[op.source]
    lines, numeric = _list(gen, source.numeric)
    categorical = "character(0)"
    if source.categorical:
        extra, categorical = _list(gen, source.categorical)
        lines += extra
    arguments = [source.frame, f'"{source.outcome}"', op.source, numeric, categorical]
    if source.sample:
        arguments.append(f"egitim = {source.frame}${source.sample}")
    if not source.standardize:
        arguments.append("olcekle = FALSE")
    lines.append(f"# Post-Lasso: {op.source} modelinin seçtiği terimlerle cezasız EKK")
    lines += _wrapped(f"{op.name} <- post_secim(", arguments, ")")
    lines += _scalar_lines(gen, op.name)
    lines.append(
        f'cat(sprintf("{op.name}: test MSE = %.4f, seçilen terim = %d\\n", {op.name}$test_mse, {op.name}$sifirdan))'
    )
    return lines


def _metrics(op: ModelMetrics) -> list[str]:
    models = ", ".join(f"{model} = {model}" for model, _ in op.rows)
    labels = ", ".join(f'"{label}"' for _, label in op.rows)
    return [
        "# Dış-örneklem karşılaştırması: test MSE, sıfırdan farklı katsayı, ‖β̂‖₂ ve seçilen ceza",
        f"modeller <- list({models})",
        f"{op.result} <- data.frame(",
        f"  etiket = c({labels}),",
        "  test_mse = sapply(modeller, function(m) m$test_mse),",
        "  sifirdan = sapply(modeller, function(m) m$sifirdan),",
        "  norm = sapply(modeller, function(m) m$norm),",
        "  lambda = sapply(modeller, function(m) if (m$lam > 0) m$lam else NA),",
        "  row.names = names(modeller)",
        ")",
        f"print({op.result}, digits = 6)",
    ]


def _dot_plot(op: DotPlot) -> list[str]:
    color = SERIES_COLORS[0][0]
    pairs = ", ".join(key + ' = "' + label + '"' for key, label in op.labels)
    return [
        f"degerler <- {op.table}${op.column}",
        f"etiketler <- c({pairs})[rownames({op.table})]",
        "konum <- rev(seq_along(degerler))",
        "eski <- par(mar = c(5, 9, 4, 2))",
        f'plot(degerler, konum, pch = 16, cex = 1.6, col = "{color}", yaxt = "n", ylab = "",',
        "     xlim = range(degerler) + c(-0.25, 0.45) * diff(range(degerler)),",
        f'     xlab = "{op.x_label}", main = "{op.title}")',
        "axis(2, at = konum, labels = etiketler, las = 1)",
        f'text(degerler, konum, sprintf("%.{op.decimals}f", degerler), pos = 4)',
        f'legend("bottomright", legend = "{op.x_label}", col = "{color}", pch = 16, bty = "n")',
        "par(eski)",
    ]


def _cv_curve(op: CVCurve) -> list[str]:
    color, dark = SERIES_COLORS[0][0], DARK[0]
    model = op.model
    return [
        f"cv <- {model}$cv",
        "plot(cv$lambda, cv$cv_ort, log = \"x\", type = \"n\",",
        "     ylim = range(c(cv$cv_ort - cv$cv_sh, cv$cv_ort + cv$cv_sh)),",
        f'     xlab = "{op.x_label}", ylab = "CV ortalama karesel hatası", main = "{op.title}")',
        "polygon(c(cv$lambda, rev(cv$lambda)), c(cv$cv_ort - cv$cv_sh, rev(cv$cv_ort + cv$cv_sh)),",
        f'        col = adjustcolor("{color}", 0.18), border = NA)',
        f'lines(cv$lambda, cv$cv_ort, col = "{color}", lwd = 2)',
        f'abline(v = {model}$lam, col = "{dark}", lty = 2, lwd = 1.5)',
        'legend("topleft", legend = c("CV ortalama karesel hatası", "±1 SH (katlar arası)",',
        f'                            sprintf("Seçilen λ = %.4g", {model}$lam)),',
        f'       col = c("{color}", NA, "{dark}"), fill = c(NA, adjustcolor("{color}", 0.18), NA), border = NA,',
        "       lty = c(1, NA, 2), lwd = c(2, NA, 1.5), bty = \"n\")",
    ]


def _coef_path(op: CoefPath) -> list[str]:
    color, gray, dark = SERIES_COLORS[0][0], GRAY[0], DARK[0]
    model = op.model
    return [
        f"yol <- {model}$yol",
        f"vurgulu <- c({_quoted(op.highlight)})",
        "plot(NA, xlim = rev(range(yol$lambda)), ylim = range(yol$katsayi), log = \"x\",",
        f'     xlab = "{op.x_label}", ylab = "Katsayı", main = "{op.title}")',
        "for (ad in setdiff(colnames(yol$katsayi), vurgulu)) {",
        f'  lines(yol$lambda, yol$katsayi[, ad], col = "{gray}", lwd = 0.8)',
        "}",
        f'for (ad in vurgulu) lines(yol$lambda, yol$katsayi[, ad], col = "{color}", lwd = 2)',
        f'abline(v = {model}$lam, col = "{dark}", lty = 2, lwd = 1.5)',
        f'legend("bottomleft", legend = c("{op.highlight_label}", "{op.other_label}",',
        f'                               sprintf("CV ile seçilen λ = %.3g", {model}$lam)),',
        f'       col = c("{color}", "{gray}", "{dark}"), lty = c(1, 1, 2), lwd = c(2, 0.8, 1.5), bty = "n")',
    ]


def _crossfit(gen, op: CrossFitDML) -> list[str]:
    lines, features = _list(gen, op.features)
    arguments = [op.frame, f'"{op.outcome}"', f'"{op.treatment}"', features]
    arguments.append(f"{op.frame}${op.outer}" if op.outer else "NULL")
    if op.learner == "lasso":
        arguments += [f"{op.frame}${op.inner}", _grid(op.grid)]
        if op.rule != "min":
            arguments.append(f'"{op.rule}"')
    if op.cluster:
        arguments.append(f"kume = {op.frame}${op.cluster}")
    if op.learner != "lasso":
        arguments.append(f'ogrenici = "{op.learner}"')
    learner = "Lasso (ceza iç katlarla CV'de)" if op.learner == "lasso" else "cezasız EKK"
    if op.outer:
        lines.append(f"# DML2: yardımcı modeller {learner}; her dış katın artığı o katı görmemiş modelden")
    else:
        lines.append(f"# Artıklaştırma: yardımcı modeller {learner}, bütün örneklemde (çapraz uyarlama yok)")
    lines += _wrapped(f"{op.name} <- dml_capraz(", arguments, ")")
    se = "okul/küme SH" if op.cluster else "HC1 SH"
    lines.append(f'cat(sprintf("θ̂ = %.4f ({se} %.4f), n = %d\\n", {op.name}$theta, {op.name}$se, {op.name}$n))')
    if op.outer and op.learner == "lasso":
        lines.append(f"print({op.name}$katlar)")
    if op.learner == "lasso":
        for part in ("y", "d"):
            for statistic in ("min", "max"):
                key = dml_fold_key(op.name, part, statistic)
                if key in gen.scalar_refs:
                    lines.append(f"{key} <- {statistic}({op.name}$katlar$sifirdan_{part})")
    if op.residuals is not None:
        u, v = op.residuals
        lines += [f"{op.frame}${u} <- {op.name}$u", f"{op.frame}${v} <- {op.name}$v"]
    return lines


def _splits(gen, op: DMLSplits) -> list[str]:
    source: CrossFitDML = gen.models[op.dml]
    lines, features = _list(gen, source.features)
    arguments = [source.frame, f'"{source.outcome}"', f'"{source.treatment}"', features, "kat"]
    if source.learner == "lasso":
        arguments += [f"{source.frame}${source.inner}", _grid(source.grid)]
        if source.rule != "min":
            arguments.append(f'"{source.rule}"')
    if source.cluster:
        arguments.append(f"kume = {source.frame}${source.cluster}")
    if source.learner != "lasso":
        arguments.append(f'ogrenici = "{source.learner}"')
    keys = ", ".join(f'"{key}"' for key, _, _ in op.rules)
    labels = ", ".join(f'"{label}"' for _, label, _ in op.rules)
    multipliers = ", ".join(_number(multiplier) for _, _, multiplier in op.rules)
    lines += [
        f"# Bölme duyarlılığı: dış katlar ⌊{op.folds}{{r·c}}⌋ + 1; aynı veri, öğrenici ve kat sayısı, yalnız okulların",
        "# katlara dağılımı değişir. Medyan birleştirme: θ̂_med = medyan θ̂_s, σ̂²_med = medyan{σ̂²_s + (θ̂_s − θ̂_med)²}",
        *_wrapped("carpanlar <- c(", multipliers.split(", "), ")"),
        f"{op.result} <- data.frame(etiket = c({labels}), theta = NA_real_, sh = NA_real_,",
        f"{' ' * (len(op.result) + 15)}row.names = c({keys}))",
        "for (s in seq_along(carpanlar)) {",
        f"  x <- {source.frame}${op.rank} * carpanlar[s]",
        f"  kat <- floor({op.folds} * (x - floor(x))) + 1",
        *[f"  {line}" for line in _wrapped("b <- dml_capraz(", arguments, ")")],
        f"  {op.result}$theta[s] <- b$theta",
        f"  {op.result}$sh[s] <- b$se",
        "}",
        f"medyan <- median({op.result}$theta)",
        f"medyan_sh <- sqrt(median({op.result}$sh^2 + ({op.result}$theta - medyan)^2))",
        f'{op.name} <- structure(list(theta = medyan, se = medyan_sh, n = nrow({source.frame})), class = "dml_sonuc")',
        f"{op.name}_min <- min({op.result}$theta)",
        f"{op.name}_max <- max({op.result}$theta)",
        f"print({op.result}, digits = 4)",
        f'cat(sprintf("Medyan θ̂ = %.3f (SH %.3f); aralık %.3f – %.3f\\n", medyan, medyan_sh, {op.name}_min, '
        f"{op.name}_max))",
    ]
    return lines


def _double_selection(gen, op: DoubleSelection) -> list[str]:
    lines, controls = _list(gen, op.controls)
    arguments = [op.frame, f'"{op.outcome}"', f'"{op.treatment}"', controls, f"{op.frame}${op.folds}", _grid(op.grid)]
    if op.rule != "1se":
        arguments.append(f'"{op.rule}"')
    rule = "bir standart hata kuralı" if op.rule == "1se" else "en küçük CV"
    lines += [
        f"# Yalnız-sonuç Post-Lasso ve double selection; Lasso cezaları CV ile, {rule}",
        *_wrapped(f"{op.name}_kume <- cift_secim(", arguments, ")"),
        f"{op.name}_sonuc <- {op.name}_kume$sonuc",
        f"{op.name} <- {op.name}_kume$cift",
    ]
    tracked = _quoted(op.track)
    for suffix, key in (("ny", "sonuc"), ("nd", "tedavi"), ("n", "birlesim")):
        lines.append(f"{op.name}_{suffix} <- length({op.name}_kume$kumeler${key})")
        if op.track:
            lines.append(f"{op.name}_{suffix}_iz <- sum({op.name}_kume$kumeler${key} %in% c({tracked}))")
    lines.append(f"print(sapply({op.name}_kume$kumeler, length))  # seçilen kontrol sayıları")
    return lines


def _estimate_plot(gen, op: EstimatePlot) -> list[str]:
    color, gray, red = SERIES_COLORS[0][0], GRAY[0], SERIES_COLORS[1][0]
    estimates = [f'coef({model})[["{gen._key(model, term)}"]]' for _, model, term in op.rows]
    errors = [gen._standard_error(model, term) for _, model, term in op.rows]
    labels = ", ".join(f'"{label}"' for label, _, _ in op.rows)
    lines = [
        f"{op.result} <- data.frame(",
        *_wrapped("  tahmin = c(", estimates, "),"),
        *_wrapped("  sh = c(", errors, "),"),
        f"  row.names = c({labels})",
        ")",
        f"{op.result}$alt <- {op.result}$tahmin - {CI_MULTIPLIER} * {op.result}$sh",
        f"{op.result}$ust <- {op.result}$tahmin + {CI_MULTIPLIER} * {op.result}$sh",
        f"print(round({op.result}, 3))",
        f"konum <- rev(seq_len(nrow({op.result})))",
    ]
    extent = [f"{op.result}$alt", f"{op.result}$ust"]
    if op.splits:
        extent.append(f"{op.splits}$theta")
    if op.truth is not None:
        extent.append(_number(op.truth))
    lines += [
        "eski <- par(mar = c(5, 17, 4, 2))",
        f'plot({op.result}$tahmin, konum, xlim = range(c({", ".join(extent)})), ylim = c(0.5, length(konum) + 0.5),',
        f'     pch = 16, col = "{color}", yaxt = "n", ylab = "", xlab = "{op.x_label}", main = "{op.title}")',
        f'arrows({op.result}$alt, konum, {op.result}$ust, konum, angle = 90, code = 3, length = 0.05, col = "{color}", lwd = 2)',
        f"axis(2, at = konum, labels = rownames({op.result}), las = 1, cex.axis = 0.8)",
    ]
    legend = {"label": ['"Tahmin ve %95 güven aralığı"'], "col": [f'"{color}"'], "pch": ["16"], "lty": ["1"]}
    if op.splits:
        lines.append(
            f'points({op.splits}$theta, rep(konum[{op.splits_row + 1}] - 0.25, nrow({op.splits})), pch = 16, cex = 0.7, '
            f'col = "{gray}")'
        )
        legend["label"].append('"Farklı kat kurallarıyla DML tahminleri"')
        legend["col"].append(f'"{gray}"')
        legend["pch"].append("16")
        legend["lty"].append("NA")
    if op.truth is not None:
        lines.append(f'abline(v = {_number(op.truth)}, col = "{red}", lty = 2, lwd = 1.5)')
        legend["label"].append(f'"{op.truth_label}"')
        legend["col"].append(f'"{red}"')
        legend["pch"].append("NA")
        legend["lty"].append("2")
    lines += [
        f'legend("topright", legend = c({", ".join(legend["label"])}), col = c({", ".join(legend["col"])}),',
        f'       pch = c({", ".join(legend["pch"])}), lty = c({", ".join(legend["lty"])}), bty = "n", cex = 0.8)',
        "par(eski)",
    ]
    return lines


def _line_plot(op: LinePlot) -> list[str]:
    columns = [column for column, _ in op.columns]
    lines = [f"cizim <- {op.table}[, c({_quoted(columns)}), drop = FALSE]"]
    if op.relative:
        lines.append("cizim <- sweep(cizim, 2, apply(cizim, 2, min))  # her ölçütten kendi en küçük değeri çıkarılır")
    lines += [
        "x <- as.numeric(rownames(cizim))",
        "plot(NA, xlim = range(x), ylim = range(as.matrix(cizim)),",
        f'     xlab = "{op.x_label}", ylab = "{op.y_label}", main = "{op.title}")',
    ]
    colors = []
    for (column, _), (color, _) in zip(op.columns, SERIES_COLORS):
        colors.append(f'"{color}"')
        lines.append(f'lines(x, cizim${column}, type = "b", pch = 16, col = "{color}", lwd = 2)')
    labels = ", ".join(f'"{label}"' for _, label in op.columns)
    lines.append(f'legend("topright", legend = c({labels}), col = c({", ".join(colors)}), pch = 16, lty = 1, bty = "n")')
    return lines
