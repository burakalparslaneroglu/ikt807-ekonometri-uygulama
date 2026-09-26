"""Laboratuvar tanımından R kodu üretir.

Varsayılan yığın temel R'dır. Dayanıklı çıkarım gereken konularda ``sandwich`` ve
``lmtest`` kullanılır; kovaryans türü kodda açıkça yazılır (ör. ``type = "HC1"``).
Hansen'in ``.dta`` dosyaları ``haven``, 2SLS ``AER::ivreg`` ile okunur/tahmin edilir.
"""

from __future__ import annotations

from core.codegen.base import (
    HANSEN_ARCHIVE_URL,
    Generator,
    categorical_comment,
    continuous_terms,
    flatten,
    histogram_styles,
    layer_styles,
    link_of,
    profile_others,
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

_STAT = {"count": "length", "sum": "sum", "mean": "mean", "sd": "sd", "median": "median", "min": "min", "max": "max"}
_COLORS = ('"#107C89"', '"#B3392F"', '"#2F9E6B"', '"#07373D"')
_REFERENCE_STYLES = (('"#07373D"', "2"), ('"#6B4C9A"', "3"))


def _term(term: str) -> str:
    """Dilden bağımsız terim adının R karşılığı: sabit ve ``değişken=düzey`` kuklaları."""

    if term == E.INTERCEPT:
        return "(Intercept)"
    if "=" in term:
        variable, level = term.split("=", 1)
        return f"factor({variable}){level}"
    return term


_FUNCTIONS = {
    "log": "log", "exp": "exp", "sqrt": "sqrt", "maximum": "pmax", "minimum": "pmin",
    "round": "round", "floor": "floor", "positive": "as.numeric({0} > 0)",
    "logistic": "plogis", "normcdf": "pnorm", "normpdf": "dnorm",
    **{name: f"as.numeric({{0}} {symbol} {{1}})" for name, symbol in E.COMPARISONS.items()},
}
_COLORS_HEX = ("#107C89", "#B3392F", "#2F9E6B", "#07373D")


def _quoted(names) -> str:
    return ", ".join(f'"{name}"' for name in names)


def _wrapped(opening: str, items: list[str], closing: str, width: int = 88) -> list[str]:
    """Uzun ``c(...)`` listelerini okunur satırlara böler."""

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


def _formula_lines(formula: str, width: int = 78) -> list[str]:
    """Uzun R formülünü ``+`` ve ``|`` işaretlerinden sonra bölünmüş satırlara ayırır."""

    tokens: list[str] = []
    for part_index, part in enumerate(formula.split(" | ")):
        terms = part.split(" + ")
        for index, term in enumerate(terms):
            separator = " +" if index < len(terms) - 1 else (" |" if part_index < formula.count(" | ") else "")
            tokens.append(term + separator)
    lines: list[str] = []
    current = ""
    for token in tokens:
        if current and len(current) + len(token) + 1 > width:
            lines.append(current)
            current = "  " + token
        else:
            current = f"{current} {token}" if current else token
    lines.append(current)
    return lines


def _r_values(values) -> str:
    """Sayı listesinin R yazımı; ardışık tam sayılar ``a:b`` olarak."""

    numbers = [float(v) for v in values]
    if len(numbers) > 2 and all(v.is_integer() for v in numbers) and all(
        b - a == 1 for a, b in zip(numbers, numbers[1:])
    ):
        return f"{int(numbers[0])}:{int(numbers[-1])}"
    return "c(" + ", ".join(E.format_number(v) for v in numbers) + ")"


_ROBUST_HELPER = [
    "# Dayanıklı (sandviç) kovaryans, gözlenen Hessian ile: H⁻¹ (Σ sᵢsᵢ') H⁻¹, serbestlik düzeltmesiz.",
    "# statsmodels (HC0) ve Stata (vce(robust); ayrıca n/(n−1) çarpanı) ile aynı tanım. Probit'te",
    "# sandwich::sandwich() beklenen bilgi matrisini kullandığı için standart hatalar biraz farklı çıkar.",
    "dayanikli_vcov <- function(model) {",
    "  X <- model.matrix(model)",
    "  y <- model$y",
    "  z <- drop(X %*% coef(model))",
    '  if (model$family$link == "logit") {',
    "    p <- plogis(z)",
    "    skor <- (y - p) * X",
    "    w <- p * (1 - p)",
    "  } else {",
    "    q <- 2 * y - 1",
    "    oran <- q * dnorm(q * z) / pnorm(q * z)",
    "    skor <- oran * X",
    "    w <- oran * (oran + z)",
    "  }",
    "  ekmek <- solve(crossprod(X, w * X))",
    "  ekmek %*% crossprod(skor) %*% ekmek",
    "}",
]

_MARGINAL_EFFECTS_HELPER = [
    "# Ortalama marjinal etkiler (AME): sürekli terimde türevin, kategorik terimde referans düzeyine göre",
    "# olasılık farkının örneklem ortalaması. SH delta yöntemiyle: sqrt(∇' V ∇).",
    '# baglanti: "logit", "probit" veya "dogrusal" (doğrusal olasılık modeli: etki = katsayı).',
    "# kesikli: kategorik tanımlanmamış 0/1 değişkenler; bunlarda da türev yerine 1 − 0 farkı alınır.",
    "ortalama_marjinal_etkiler <- function(model, V, baglanti, terimler, kesikli = character()) {",
    "  X <- model.matrix(model)",
    "  b <- coef(model)[colnames(X)]",
    "  V <- V[colnames(X), colnames(X)]",
    '  if (baglanti == "logit") {',
    "    G <- plogis; g <- dlogis; g1 <- function(v) dlogis(v) * (1 - 2 * plogis(v))",
    '  } else if (baglanti == "probit") {',
    "    G <- pnorm; g <- dnorm; g1 <- function(v) -v * dnorm(v)",
    "  } else {",
    "    G <- function(v) v; g <- function(v) rep(1, length(v)); g1 <- function(v) rep(0, length(v))",
    "  }",
    "  fark <- function(birler, sifirlar) {",
    "    X1 <- X; X0 <- X",
    "    X1[, sifirlar] <- 0; X0[, sifirlar] <- 0; X1[, birler] <- 1",
    "    z1 <- drop(X1 %*% b); z0 <- drop(X0 %*% b)",
    "    turev <- colMeans(g(z1) * X1 - g(z0) * X0)",
    "    c(mean(G(z1) - G(z0)), sqrt(drop(t(turev) %*% V %*% turev)))",
    "  }",
    "  tahmin <- c(); sh <- c()",
    "  for (terim in terimler) {",
    '    onek <- paste0("factor(", terim, ")")',
    "    gruplar <- colnames(X)[startsWith(colnames(X), onek)]",
    "    if (length(gruplar) > 0) {",
    "      for (sutun in gruplar) {",
    '        ad <- paste0(terim, "=", substring(sutun, nchar(onek) + 1))',
    "        sonuc <- fark(sutun, gruplar)",
    "        tahmin[ad] <- sonuc[1]; sh[ad] <- sonuc[2]",
    "      }",
    "    } else if (terim %in% kesikli) {",
    "      sonuc <- fark(terim, terim)",
    "      tahmin[terim] <- sonuc[1]; sh[terim] <- sonuc[2]",
    "    } else {",
    "      z <- drop(X %*% b)",
    "      tahmin[terim] <- mean(g(z)) * b[[terim]]",
    "      turev <- mean(g(z)) * (colnames(X) == terim) + b[[terim]] * colMeans(g1(z) * X)",
    "      sh[terim] <- sqrt(drop(t(turev) %*% V %*% turev))",
    "    }",
    "  }",
    '  structure(list(coefficients = tahmin, se = sh, n = nrow(X)), class = "ame_sonucu")',
    "}",
    "coef.ame_sonucu <- function(object, ...) object$coefficients",
    "vcov.ame_sonucu <- function(object, ...) {",
    "  V <- diag(object$se^2, nrow = length(object$se))",
    "  dimnames(V) <- list(names(object$se), names(object$se))",
    "  V",
    "}",
    "nobs.ame_sonucu <- function(object, ...) object$n",
]


class RGenerator(Generator):
    language = "R"
    comment = "#"

    def dialect(self, frame: str) -> E.Dialect:
        return E.Dialect(
            variable=lambda name: f"{frame}${name}",
            coefficient=lambda model, term: f'coef({model})[["{self._key(model, term)}"]]',
            functions=_FUNCTIONS,
            power="^",
            standard_error=self._standard_error,
        )

    def _key(self, model: str, term: str) -> str:
        """Etki sonuçlarında terim adı olduğu gibi (``race4=2``); modellerde R adı."""

        return term if self.is_effects(model) else _term(term)

    def _standard_error(self, model: str, term: str) -> str:
        return f'sqrt(diag({self.vcov(model)}))[["{self._key(model, term)}"]]'

    def vcov(self, model: str, name: str | None = None) -> str:
        """Modelin kovaryans matrisi ifadesi; ``name`` verilirse o değişken adıyla yazılır."""

        symbol = name or model
        settings = self.models.get(model)
        if isinstance(settings, IV):
            return f'sandwich::vcovHC({symbol}, type = "HC1")'
        if isinstance(settings, BinaryChoice):
            return f"dayanikli_vcov({symbol})" if settings.vcov == "robust" else f"vcov({symbol})"
        if isinstance(settings, (Tobit, QuantileRegression, MarginalEffects)):
            return f"vcov({symbol})"
        if settings is None or settings.vcov == "classic":
            return f"vcov({symbol})"
        if settings.vcov == "HC1":
            return f'sandwich::vcovHC({symbol}, type = "HC1")'
        return f'sandwich::vcovCL({symbol}, cluster = ~{settings.cluster}, type = "HC1")'

    def _needs_sandwich(self, operations) -> bool:
        return any(
            (isinstance(op, OLS) and op.vcov != "classic")
            or isinstance(op, (StandardErrorTable, BreuschPagan, IV))
            for op in operations
        )

    # --- Başlık ve yardımcılar ------------------------------------------
    def imports(self, operations: tuple[Operation, ...]) -> list[str]:
        ops = flatten(operations)
        packages: list[str] = []
        if self._needs_sandwich(ops):
            packages += ["sandwich", "lmtest"]
        elif any(isinstance(op, BinaryChoice) for op in ops) or any(
            isinstance(op, ShowModel) and isinstance(self.models.get(op.model), BinaryChoice) for op in ops
        ):
            packages.append("lmtest")
        if any(isinstance(op, LoadHansen) and op.member.lower().endswith(".dta") for op in ops):
            packages.append("haven")
        if any(isinstance(op, (IV, Tobit)) for op in ops):
            packages.append("AER")
        if any(isinstance(op, QuantileRegression) for op in ops):
            packages.append("quantreg")
        if not packages:
            return []
        lines = [f"# Gerekli paketler: install.packages(c({_quoted(packages)}))"]
        if "sandwich" in packages:
            lines += ["library(sandwich)", "library(lmtest)"]
        elif "lmtest" in packages:
            lines.append("library(lmtest)")
        return lines + [""]

    def helpers(self, operations: tuple[Operation, ...], *, with_checks: bool) -> list[str]:
        lines: list[str] = []
        loads = [op for op in operations if isinstance(op, LoadHansen)]
        if loads:
            stata_file = loads[0].member.lower().endswith(".dta")
            example = loads[0].member if stata_file else "cps09mar.txt"
            lines += [
                "options(timeout = max(600, getOption(\"timeout\")))",
                f'hansen_arsiv <- "{HANSEN_ARCHIVE_URL}"',
                "",
                f"# Dosyayı kendiniz indirdiyseniz yolunu yazın (ör. \"{example}\").",
                "# NULL bırakırsanız veri Hansen'in sayfasından otomatik indirilir.",
                "yerel_dosya <- NULL",
                "",
            ]
            if stata_file:
                lines += [
                    "# Hansen'in .dta dosyası değişken adlarını taşır; Python, R ve Stata'da aynı olsun diye",
                    "# adlar küçük harfe çevrilir. Yerel dosya olarak ders notlarının öğretim CSV'si de verilebilir.",
                    "hansen_verisi <- function(dosya_adi, yerel_dosya = NULL) {",
                    "  if (!is.null(yerel_dosya) && grepl(\"\\\\.csv$\", tolower(yerel_dosya))) {",
                    "    veri <- read.csv(yerel_dosya)",
                    "  } else {",
                    "    if (is.null(yerel_dosya)) {",
                    "      gecici <- tempfile(fileext = \".zip\")",
                    "      download.file(hansen_arsiv, gecici, mode = \"wb\", quiet = TRUE)",
                    "      uyeler <- unzip(gecici, list = TRUE)$Name",
                    "      uye <- uyeler[tolower(basename(uyeler)) == tolower(dosya_adi)][1]",
                    "      if (is.na(uye)) stop(dosya_adi, \" Hansen arşivinde bulunamadı.\")",
                    "      yerel_dosya <- unzip(gecici, files = uye, exdir = tempdir(), junkpaths = TRUE)",
                    "      unlink(gecici)",
                    "    }",
                    "    veri <- as.data.frame(haven::read_dta(yerel_dosya))",
                    "  }",
                    "  names(veri) <- tolower(names(veri))",
                    "  veri",
                    "}",
                    "",
                ]
            else:
                lines += [
                    "# Hansen'in .txt dosyalarında başlık satırı yoktur ve değerler boşlukla ayrılır;",
                    "# değişken adları açıklama belgesindeki sırayla verilir.",
                    "hansen_verisi <- function(dosya_adi, sutunlar, yerel_dosya = NULL) {",
                    "  if (!is.null(yerel_dosya)) return(read.table(yerel_dosya, col.names = sutunlar))",
                    "  gecici <- tempfile(fileext = \".zip\")",
                    "  download.file(hansen_arsiv, gecici, mode = \"wb\", quiet = TRUE)",
                    "  uyeler <- unzip(gecici, list = TRUE)$Name",
                    "  uye <- uyeler[tolower(basename(uyeler)) == dosya_adi][1]",
                    "  if (is.na(uye)) stop(dosya_adi, \" Hansen arşivinde bulunamadı.\")",
                    "  read.table(unz(gecici, uye), col.names = sutunlar)",
                    "}",
                    "",
                ]
        if with_checks:
            lines += [
                "kontrol_et <- function(etiket, deger, beklenen, ondalik = 4) {",
                "  tolerans <- 0.5 * 10^(-ondalik) + 1e-12",
                "  durum <- if (abs(deger - beklenen) <= tolerans) \"OK  \" else \"HATA\"",
                "  cat(sprintf(\"  %s %s: %.*f  (notlar: %s)\\n\", durum, etiket, ondalik, deger,",
                "              sprintf(\"%.*f\", ondalik, beklenen)))",
                "  if (abs(deger - beklenen) > tolerans) stop(etiket, \" notlarla uyuşmuyor.\")",
                "}",
                "",
            ]
        return lines

    def helper_names(self, operations: tuple[Operation, ...]) -> list[str]:
        ops = flatten(operations)
        names: list[str] = []
        if any(isinstance(op, BinaryChoice) and op.vcov == "robust" for op in ops):
            names.append("dayanikli_vcov")
        if any(isinstance(op, MarginalEffects) for op in ops):
            names.append("marjinal_etkiler")
        return names

    def helper_code(self, name: str) -> list[str]:
        if name == "dayanikli_vcov":
            return _ROBUST_HELPER + [""]
        if name == "marjinal_etkiler":
            return _MARGINAL_EFFECTS_HELPER + [""]
        return []

    # --- İşlemler --------------------------------------------------------
    def operation(self, op: Operation) -> list[str]:
        lines = self._operation(op)
        if self.quiet:
            lines = [line for line in lines if not line.lstrip().startswith(("print(", "cat("))]
        return lines

    def _operation(self, op: Operation) -> list[str]:
        limited = self._limited_operation(op)
        if limited is not None:
            return limited
        if isinstance(op, StandardErrorTable):
            t = op.term
            models = ", ".join(f"{m} = {m}" for m in op.models)
            return [
                "# Aynı katsayı, iki farklı belirsizlik ölçüsü: klasik ve HC1 standart hata",
                f"modeller <- list({models})",
                f"{op.result} <- t(sapply(modeller, function(m) c(",
                f'  katsayi = coef(m)[["{t}"]],',
                f'  klasik_sh = sqrt(diag(vcov(m)))[["{t}"]],',
                f'  hc1_sh = sqrt(diag(vcovHC(m, type = "HC1")))[["{t}"]],',
                "  r2 = summary(m)$r.squared",
                ")))",
                f"print(round({op.result}, 4))",
            ]
        if isinstance(op, BreuschPagan):
            return [
                "# Breusch–Pagan (Koenker, n·R²): bptest varsayılanı studentize = TRUE",
                f"bp <- bptest({op.model})",
                f"{op.name}_lm <- unname(bp$statistic)",
                f"{op.name}_p <- bp$p.value",
                f'cat(sprintf("LM = %.2f, p = %.2e\\n", {op.name}_lm, {op.name}_p))',
            ]
        if isinstance(op, LinearCombination):
            weights = ", ".join(f"`{_term(t)}` = {E.format_number(w)}" for t, w in op.weights)
            return [
                f"# {op.comment}: a'β ve √(a'Va)",
                f"a <- c({weights})",
                f"{op.name} <- sum(a * coef({op.model})[names(a)])",
                f"V <- {self.vcov(op.model)}[names(a), names(a), drop = FALSE]",
                f"{op.name}_se <- sqrt(drop(t(a) %*% V %*% a))",
                f'cat(sprintf("{op.comment}: %.4f (SH %.4f)\\n", {op.name}, {op.name}_se))',
            ]
        if isinstance(op, DeltaMethod):
            dialect = self.dialect("")
            coefs = E.coefficients(op.expr)
            gradient = ", ".join(E.render(E.derivative(op.expr, c), dialect) for c in coefs)
            terms = ", ".join(f'"{_term(c.term)}"' for c in coefs)
            return [
                f"# {op.comment}",
                f"{op.name} <- {E.render(op.expr, dialect)}",
                "# Delta yöntemi: SH = √(g'(β)' V g'(β))",
                f"turev <- c({gradient})",
                f"V <- {self.vcov(op.model)}[c({terms}), c({terms}), drop = FALSE]",
                f"{op.name}_se <- sqrt(drop(t(turev) %*% V %*% turev))",
                f'cat(sprintf("{op.comment}: %.2f (delta SH %.3f)\\n", {op.name}, {op.name}_se))',
            ]
        if isinstance(op, NewSample):
            frame = f"{op.frame} <- data.frame(id = seq_len({op.nobs}))"
            if op.seed is None:
                return [frame]
            return [
                "# Sabit tohum: betik her çalıştırmada aynı veriyi üretir",
                f"set.seed({op.seed})",
                frame,
            ]
        if isinstance(op, ClusterDraw):
            a, b = E.format_number(op.first), E.format_number(op.second)
            return [
                f"# {op.comment} (her küme için bir çekiliş)",
                f"kume_soku <- rnorm({op.groups}, mean = {a}, sd = {b})",
                f"{op.frame}${op.name} <- kume_soku[{op.frame}${op.cluster}]",
            ]
        if isinstance(op, Draw):
            a, b = E.format_number(op.first), E.format_number(op.second)
            call = f"rnorm(nrow({op.frame}), mean = {a}, sd = {b})" if op.distribution == "normal" \
                else f"runif(nrow({op.frame}), min = {a}, max = {b})"
            return [f"# {op.comment}", f"{op.frame}${op.name} <- {call}"]
        if isinstance(op, Predict):
            if op.kind == "index":
                settings = self.models.get(op.model)
                kind = "lp" if isinstance(settings, Tobit) else "link"
                return ["# Doğrusal indeks x'β (olasılık değil)",
                        f'{op.frame}${op.name} <- predict({op.model}, type = "{kind}")']
            function = "fitted" if op.kind == "fitted" else "resid"
            return [f"{op.frame}${op.name} <- {function}({op.model})"]
        if isinstance(op, Summaries):
            entries = [f'  "{label}" = {_STAT[stat]}({op.frame}${variable})' for label, variable, stat in op.rows]
            return [f"{op.result} <- c(", ",\n".join(entries), ")", f"print(round({op.result}, 10))"]
        if isinstance(op, Plot):
            return self._plot(op)
        if isinstance(op, LoadHansen):
            if op.member.lower().endswith(".dta"):
                return [
                    f"# Hansen'in {op.member.rsplit('.', 1)[0]} veri seti; değişken adları dosyada kayıtlı",
                    f'{op.frame} <- hansen_verisi("{op.member}", yerel_dosya)',
                    f"print(dim({op.frame}))  # gözlem, değişken",
                ]
            names = ", ".join(f'"{c}"' for c in op.columns)
            return [
                f"# Değişken sırası: Hansen'in {op.dataset} açıklama belgesi",
                f"sutunlar <- c({names})",
                f'{op.frame} <- hansen_verisi("{op.member}", sutunlar, yerel_dosya)',
                f"print(dim({op.frame}))  # gözlem, değişken",
            ]
        if isinstance(op, DropMissing):
            return [
                f"# {op.comment}",
                *_wrapped(f"eksiksiz <- complete.cases({op.frame}[, c(", [f'"{v}"' for v in op.variables], ")])"),
                f"{op.frame} <- {op.frame}[eksiksiz, ]",
                f"print(nrow({op.frame}))  # analiz örneklemi",
            ]
        if isinstance(op, Derive):
            rhs = E.render(op.expr, self.dialect(op.frame))
            return [f"# {op.comment}", f"{op.frame}${op.name} <- {rhs}"]
        if isinstance(op, Describe):
            names = ", ".join(f'"{name}"' for name in op.variables)
            stats = ", ".join(f"{s} = {_STAT[s]}(x)" for s in op.stats)
            return [
                f"degiskenler <- c({names})",
                f"{op.result} <- t(sapply({op.frame}[degiskenler], function(x) c({stats})))",
                f"print(round({op.result}, 4))",
            ]
        if isinstance(op, GroupSummary):
            lines = [f"{op.result} <- data.frame("]
            entries = [
                f"  {name} = tapply({op.frame}${variable}, {op.frame}${op.by}, {_STAT[stat]})"
                for name, variable, stat in op.columns
            ]
            lines.append(",\n".join(entries))
            lines += [")", f"print(round({op.result}, 4))"]
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
            settings = self.models.get(op.model)
            if settings is None or (isinstance(settings, OLS) and settings.vcov == "classic"):
                return [f"summary({op.model})", f"confint({op.model})"]
            return [f"coeftest({op.model}, vcov. = {self.vcov(op.model)})"]
        if isinstance(op, RegressionTable):
            models = ", ".join(f'"({i})" = {name}' for i, name in enumerate(op.models, start=1))
            terms = ", ".join(f'"{_term(t)}"' for t in op.terms)
            vcov_expr = self.vcov(op.models[0], name="m")
            return [
                f"modeller <- list({models})",
                f"terimler <- c({terms})",
                f"{op.result} <- sapply(modeller, function(m) {{",
                "  b <- coef(m)[terimler]",
                f"  s <- sqrt(diag({vcov_expr}))[terimler]",
                '  ifelse(is.na(b), "", sprintf("%.4f (%.4f)", b, s))',
                "})",
                f"rownames({op.result}) <- terimler",
                f"print(noquote({op.result}))",
                'cat("N :", sapply(modeller, nobs), "\\n")',
                'cat("R2:", sprintf("%.4f", sapply(modeller, function(m) summary(m)$r.squared)), "\\n")',
            ]
        if isinstance(op, Scalar):
            rhs = E.render(op.expr, self.dialect(""))
            return [
                f"# {op.comment}",
                f"{op.name} <- {rhs}",
                f'cat("{op.comment}:", sprintf("%.{op.decimals}f", {op.name}), "\\n")',
            ]
        if isinstance(op, GroupMeanPlot):
            labels = [label for _, label in sorted(op.group_labels)]
            count = len(labels)
            colors = ", ".join(_COLORS[:count])
            pch = ", ".join(str(15 + i) for i in range(count))
            label_text = ", ".join(f'"{label}"' for label in labels)
            return [
                f"ortalama <- tapply({op.frame}${op.y}, list({op.frame}${op.x}, {op.frame}${op.group}), mean)",
                f"matplot(as.numeric(rownames(ortalama)), ortalama, type = \"b\", lty = 1,",
                f"        pch = c({pch}), col = c({colors}),",
                f'        xlab = "{op.x_label}", ylab = "{op.y_label}",',
                f'        main = "{op.title}")',
                f'legend("topleft", legend = c({label_text}), pch = c({pch}),',
                f"       col = c({colors}), bty = \"n\")",
            ]
        if isinstance(op, ProjectionPlot):
            return [
                f"ortalama <- tapply({op.frame}${op.y}, {op.frame}${op.x}, mean)",
                f"gozlem <- tapply({op.frame}${op.y}, {op.frame}${op.x}, length)",
                "plot(as.numeric(names(ortalama)), ortalama, pch = 16, col = \"#107C89\",",
                "     cex = sqrt(gozlem) / 30,",
                f'     xlab = "{op.x_label}", ylab = "{op.y_label}",',
                f'     main = "{op.title}")',
                f'abline(a = coef({op.model})[["(Intercept)"]], b = coef({op.model})[["{op.x}"]], lwd = 2)',
                'legend("topleft", legend = c("Koşullu ortalama", "OLS doğrusal projeksiyonu"),',
                '       pch = c(16, NA), lty = c(NA, 1), col = c("#107C89", "black"), bty = "n")',
            ]
        raise TypeError(f"R üreticisi bu işlemi tanımıyor: {type(op).__name__}")

    # --- Sınırlı bağımlı değişken işlemleri -------------------------------
    def _limited_operation(self, op: Operation) -> list[str] | None:
        if isinstance(op, KeepIf):
            parts = " & ".join(
                f"{op.frame}${variable} {operator} {E.format_number(value)}"
                for variable, operator, value in op.conditions
            )
            return [
                f"# {op.comment}",
                f"{op.frame} <- {op.frame}[{parts}, ]",
                f"print(nrow({op.frame}))  # analiz örneklemi",
            ]
        if isinstance(op, Recode):
            expression = str(op.other)
            for values, code in reversed(op.mapping):
                if len(values) == 1:
                    condition = f"{op.frame}${op.source} == {E.format_number(values[0])}"
                else:
                    condition = f"{op.frame}${op.source} %in% c({', '.join(E.format_number(v) for v in values)})"
                expression = f"ifelse({condition}, {code}, {expression})"
            return [
                f"# {op.comment}",
                f"{op.frame}${op.name} <- {expression}",
                f"print(table({op.frame}${op.name}))",
            ]
        if isinstance(op, BinaryChoice):
            terms = [f"factor({r})" if r in op.categorical else r for r in op.regressors]
            formula = f"{op.outcome} ~ " + " + ".join(terms)
            name = "Logit" if op.link == "logit" else "Probit"
            lines = [
                f"# {name} (MLE); " + ("dayanıklı kovaryans dayanikli_vcov() ile: gözlenen Hessian'lı sandviç"
                                      if op.vcov == "robust" else "klasik kovaryans: ters bilgi matrisi"),
            ]
            call = self._call(op.name, "glm", formula, op.frame, f'family = binomial(link = "{op.link}")')
            lines += call
            shown = continuous_terms(op) if op.categorical else list(op.regressors)
            if shown:
                lines.append(
                    f"print(coeftest({op.name}, vcov. = {self.vcov(op.name)})[c({_quoted(shown)}), , drop = FALSE])"
                )
            return lines
        if isinstance(op, MarginalEffects):
            settings = self.models[op.model]
            terms = f"c({_quoted(op.terms)})"
            call = (f'{op.name} <- ortalama_marjinal_etkiler({op.model}, {self.vcov(op.model)}, '
                    f'"{link_of(settings)}", {terms}')
            if op.discrete:
                call += f", kesikli = c({_quoted(op.discrete)})"
            return [
                "# Ortalama marjinal etkiler: sürekli değişkende türev, kategorik değişkende referans düzeyine",
                "# göre olasılık farkı; standart hatalar delta yöntemiyle, modelin kovaryansıyla",
                call + ")",
                f"print(round(cbind(AME = coef({op.name}), SH = sqrt(diag(vcov({op.name})))), 4))",
            ]
        if isinstance(op, AverageProfile):
            data = self.models[op.model].frame
            values = _r_values(op.values)
            return [
                f"# {op.variable} bütün gözlemlerde sırayla aynı değere eşitlenir; diğer değişkenler gözlenen",
                "# değerlerinde kalır. Her değerde tahmin edilen olasılıkların ortalaması alınır.",
                f"degerler <- {values}",
                f"{op.name} <- data.frame(olasilik = sapply(degerler, function(deger) {{",
                f"  kopya <- {data}",
                f"  kopya${op.variable} <- deger",
                f'  mean(predict({op.model}, newdata = kopya, type = "response"))',
                "}), row.names = degerler)",
                f"print(round({op.name}, 4))",
                f'plot(degerler, {op.name}$olasilik, type = "b", pch = 16, col = "#107C89",',
                f'     xlab = "{op.x_label}", ylab = "{op.y_label}", main = "{op.title}")',
            ]
        if isinstance(op, Tobit):
            formula = f"{op.outcome} ~ " + " + ".join(op.regressors)
            lines = [f"# Tobit: {op.outcome} soldan {E.format_number(op.left)} noktasında sansürlü; MLE "
                     "(AER::tobit, survival::survreg üzerine)"]
            call = self._call(op.name, "AER::tobit", formula, op.frame, f"left = {E.format_number(op.left)}")
            lines += call
            shown = ["(Intercept)", op.regressors[0]] if len(op.regressors) > 5 else ["(Intercept)", *op.regressors]
            lines.append(f"print(round(coef({op.name})[c({_quoted(shown)})], 4))")
            lines.append(f'cat("sigma:", round({op.name}$scale, 4), "\\n")')
            return lines
        if isinstance(op, QuantileRegression):
            formula = f"{op.outcome} ~ " + " + ".join(op.regressors)
            lines = [f"# Kantil regresyon, q = {E.format_number(op.q)}"
                     + (" (medyan / LAD); çözüm tek olmayabilir, rq bunu uyarıyla bildirir" if op.q == 0.5 else "")]
            call = self._call(op.name, "quantreg::rq", formula, op.frame, f"tau = {E.format_number(op.q)}")
            lines += call
            shown = ["(Intercept)", op.regressors[0]] if len(op.regressors) > 5 else ["(Intercept)", *op.regressors]
            lines.append(f"print(round(coef({op.name})[c({_quoted(shown)})], 4))")
            return lines
        if isinstance(op, ProfileCurves):
            return self._profile_curves(op)
        if isinstance(op, TobitTargets):
            latent = f"{op.curves}${op.model}"
            return [
                "# Tobit'in üç hedefi: z = x'β/σ; P(Y>0|x) = Φ(z), m(x) = Φ(z)x'β + σφ(z), m#(x) = x'β + σφ(z)/Φ(z)",
                f"s <- {op.model}$scale",
                f"z <- {latent} / s",
                f"{op.result} <- data.frame(",
                f"  gizli = {latent},",
                "  p_poz = pnorm(z),",
                f"  gozlenen = pnorm(z) * {latent} + s * dnorm(z),",
                f"  poz_ort = {latent} + s * dnorm(z) / pnorm(z),",
                f"  row.names = rownames({op.curves})",
                ")",
                f"print(round({op.result}, 3))",
            ]
        if isinstance(op, TobitFitCheck):
            settings = self.models[op.model]
            left = E.format_number(settings.left)
            y = f"{settings.frame}${settings.outcome}"
            return [
                "# Model kontrolü: Tobit'in ima ettiği P(Y>0) ve E[Y], örneklem üzerinde ortalanır",
                f'xb <- predict({op.model}, type = "lp")',
                f"z <- (xb - {left}) / {op.model}$scale",
                f"{op.result} <- data.frame(",
                f"  model = c(mean(pnorm(z)), mean({left} + pnorm(z) * (xb - {left}) + {op.model}$scale * dnorm(z))),",
                f"  veri = c(mean({y} > {left}), mean({y})),",
                '  row.names = c("p_poz", "ortalama")',
                ")",
                f"print(round({op.result}, 4))",
            ]
        return None

    def _profile_curves(self, op: ProfileCurves) -> list[str]:
        others = profile_others(op, self.models)
        dialect = self.dialect("P")
        values = _r_values(op.values)
        lines = [
            f"# Profil: {op.variable} ızgarasında x'β; türetilen terimler ızgaradan, diğer regresörler",
            "# örneklem ortalamasında. OLS/LAD'de x'β tahmin edilen ortalama/medyan, Tobit'te gizli ortalama.",
            "profil_tasarimi <- function(degerler) {",
            f"  P <- data.frame({op.variable} = degerler)",
        ]
        for name, expression in op.derived:
            lines.append(f"  P${name} <- {E.render(expression, dialect)}")
        lines += _wrapped("  for (ad in c(", [f'"{name}"' for name in others], ")) {")
        lines += [
            f"    P[[ad]] <- mean({op.frame}[[ad]])",
            "  }",
            '  cbind("(Intercept)" = 1, as.matrix(P))',
            "}",
            "dogrusal_indeks <- function(m, P) drop(P[, names(coef(m))] %*% coef(m))",
            f"degerler <- {values}",
            "izgara <- profil_tasarimi(degerler)",
            f"{op.result} <- data.frame(",
        ]
        for index, (model, _) in enumerate(op.models):
            lines.append(f"  {model} = dogrusal_indeks({model}, izgara),")
        lines += [
            "  row.names = degerler",
            ")",
            f"print(round({op.result}, 2))",
            f"ince <- profil_tasarimi(seq({E.format_number(op.plot_grid[0])}, {E.format_number(op.plot_grid[1])}, "
            f"length.out = {int(op.plot_grid[2])}))",
            f"egri <- cbind({', '.join(f'dogrusal_indeks({model}, ince)' for model, _ in op.models)})",
        ]
        colors = ", ".join(f'"{color}"' for color in _COLORS_HEX[: len(op.models)])
        labels = ", ".join(f'"{label}"' for _, label in op.models)
        lines += [
            f'matplot(ince[, "{op.variable}"], egri, type = "l", lty = 1, lwd = 2, col = c({colors}),',
            f'        xlab = "{op.x_label}", ylab = "{op.y_label}", main = "{op.title}")',
            f'legend("topright", legend = c({labels}), col = c({colors}), lty = 1, lwd = 2, bty = "n")',
        ]
        return lines

    # --- Tahminler -------------------------------------------------------
    @staticmethod
    def _call(name: str, function: str, formula: str, data: str, extra: str = "") -> list[str]:
        """``name <- function(formula, data = data[, extra])``; uzun formül satırlara bölünür."""

        tail = f", {extra}" if extra else ""
        single = f"{name} <- {function}({formula}, data = {data}{tail})"
        if len(formula) <= 120 and len(single) <= 160:
            return [single]
        body = _formula_lines(formula)
        ending = [f"  data = {data},", f"  {extra}"] if extra else [f"  data = {data}"]
        return [f"{name} <- {function}(", *[f"  {line}" for line in body[:-1]], f"  {body[-1]},",
                *ending, ")"]

    def _ols(self, op: OLS) -> list[str]:
        terms = [f"factor({r})" if r in op.categorical else r for r in op.regressors]
        formula = f"{op.outcome} ~ " + " + ".join(terms)
        lines: list[str] = []
        data = op.frame
        if self.needs_sample(op):
            data = f"veri_{op.name}"
            used = self.used_variables(op)
            if op.where is not None:
                variable, value = op.where
                lines.append(f"# Tahmin örneklemi: {variable} = {E.format_number(value)} olan, eksiksiz gözlemler")
                lines.append(
                    f"{data} <- na.omit(subset({op.frame}, {variable} == {E.format_number(value)}, "
                    f"select = c({', '.join(used)})))"
                )
            else:
                lines.append("# Tahmin örneklemi: model ve küme değişkeni eksiksiz gözlemler")
                lines.append(f"{data} <- na.omit({op.frame}[, c({_quoted(used)}), drop = FALSE])")
        lines += self._call(op.name, "lm", formula, data)
        if op.categorical:
            lines.insert(0, categorical_comment(op, "#"))
            shown = continuous_terms(op)
            if shown and op.vcov == "classic":
                lines.append(f"print(round(coef({op.name})[c({_quoted(shown)})], 4))")
            elif shown:
                lines.append(f"print(coeftest({op.name}, vcov. = {self.vcov(op.name)})[c({_quoted(shown)}), ])")
            return lines
        shown = self.shown_terms(op)
        if shown:
            if op.vcov == "classic":
                lines.append(f"print(round(coef({op.name})[c({_quoted(shown)})], 4))")
            else:
                lines.append(
                    f"print(coeftest({op.name}, vcov. = {self.vcov(op.name)})[c({_quoted(shown)}), , drop = FALSE])"
                )
        elif op.vcov == "classic":
            lines.append(f"print(round(coef({op.name}), 4))")
        else:
            lines.append(f"print(coeftest({op.name}, vcov. = {self.vcov(op.name)}))")
        return lines

    def _iv(self, op: IV) -> list[str]:
        structural = " + ".join((*op.endogenous, *op.exogenous))
        instruments = " + ".join((*op.instruments, *op.exogenous))
        formula = f"{op.outcome} ~ {structural} | {instruments}"
        lines = [
            f"# 2SLS: {', '.join(op.endogenous)} içsel, araç {', '.join(op.instruments)}; "
            + ("dışsal kontroller '|' işaretinin iki yanında da yer alır" if op.exogenous
               else "'|' işaretinden sonra araçlar"),
            '# Kovaryans: vcovHC(type = "HC1") → Python (debiased=True) ve Stata (small) ile aynı standart hata',
        ]
        lines += self._call(op.name, "AER::ivreg", formula, op.frame)
        lines.append(
            f"print(coeftest({op.name}, vcov. = {self.vcov(op.name)})[c({_quoted(self.shown_terms(op))}), , drop = FALSE])"
        )
        return lines

    def _effect_table(self, op: EffectTable) -> list[str]:
        lines = [
            f"# {op.title}" if op.title else "# Tahminler yan yana",
            "satir <- function(m, terim, V) c(coef(m)[[terim]], sqrt(diag(V))[[terim]], nobs(m))",
            "satirlar <- rbind(",
        ]
        for index, (label, model, term) in enumerate(op.rows):
            ending = "," if index < len(op.rows) - 1 else ""
            call = f'satir({model}, "{self._key(model, term)}", {self.vcov(model)}){ending}'
            entry = f'  "{label}" = {call}'
            if len(entry) > 100:
                lines += [f'  "{label}" =', f"    {call}"]
            else:
                lines.append(entry)
        lines += [
            ")",
            f"{op.result} <- data.frame(tahmin = satirlar[, 1], SH = satirlar[, 2], N = satirlar[, 3],",
            f"{' ' * (len(op.result) + len(' <- data.frame('))}row.names = rownames(satirlar))",
            "# p-değeri: normal yaklaşımla iki yönlü, 2Φ(−|tahmin/SH|)",
            f"{op.result}$p <- 2 * pnorm(-abs({op.result}$tahmin / {op.result}$SH))",
            f'print(round({op.result}[, c("tahmin", "SH", "p", "N")], 4))',
        ]
        return lines

    def _monte_carlo(self, op: MonteCarlo) -> list[str]:
        lines = [
            f"# {op.comment}",
            f"# {op.reps} tekrar; tohum döngüden önce bir kez ayarlanır",
            f"set.seed({op.seed})",
            f"sonuclar <- vector(\"list\", {op.reps})",
            f"for (tekrar in seq_len({op.reps})) {{",
        ]
        self.quiet = True
        try:
            for inner in op.body:
                lines += [f"  {line}" if line else "" for line in self.operation(inner)]
        finally:
            self.quiet = False
        dialect = self.dialect("")
        lines.append("  sonuclar[[tekrar]] <- c(")
        for index, (name, expression) in enumerate(op.collect):
            ending = "," if index < len(op.collect) - 1 else ""
            lines.append(f"    {name} = {E.render(expression, dialect)}{ending}")
        lines += [
            "  )",
            "}",
            f"{op.result} <- as.data.frame(do.call(rbind, sonuclar))",
            f"print(summary({op.result}))",
        ]
        if op.coverage:
            lines.append(f"# %95 güven aralığı: tahmin ± {CI_MULTIPLIER}·SH; gerçek değeri kapsayan tekrarların payı")
        for estimate, standard_error, truth in op.coverage:
            key = coverage_key(op.result, estimate)
            lines += [
                f"{key} <- mean(abs({op.result}${estimate} - {E.format_number(truth)}) <= "
                f"{CI_MULTIPLIER} * {op.result}${standard_error})",
                f'cat(sprintf("Kapsama oranı, {estimate} (gerçek değer {E.format_number(truth)}): %.3f\\n", {key}))',
            ]
        return lines

    def _histogram(self, op: Histogram) -> list[str]:
        lower, upper = E.format_number(op.lower), E.format_number(op.upper)
        styles = histogram_styles(len(op.columns))
        series = [f"{op.table}${column}" for column, _ in op.columns]
        lines = [
            f"# [{lower}, {upper}] dışındaki değerler çizilmez; kaç tane olduğu aşağıda yazdırılır",
            f"kutular <- seq({lower}, {upper}, length.out = {op.bins + 1})",
            f"aralikta <- function(x) x[x >= {lower} & x <= {upper}]",
            f"sayimlar <- sapply(list({', '.join(series)}),",
            "                   function(x) hist(aralikta(x), breaks = kutular, plot = FALSE)$counts)",
        ]
        for index, (item, style) in enumerate(zip(series, styles)):
            color = f'adjustcolor("{style.color}", 0.55)'
            if index == 0:
                lines += [
                    f'hist(aralikta({item}), breaks = kutular, col = {color}, border = "white",',
                    f'     ylim = c(0, max(sayimlar)), xlab = "{op.x_label}", ylab = "Tekrar sayısı",',
                    f'     main = "{op.title}")',
                ]
            else:
                lines.append(f'hist(aralikta({item}), breaks = kutular, col = {color}, border = "white", add = TRUE)')
        fills = [f'adjustcolor("{style.color}", 0.55)' for style in styles]
        labels = [f'"{label}"' for _, label in op.columns]
        ltys = ["NA"] * len(op.columns)
        colors = ["NA"] * len(op.columns)
        for index, (value, label) in enumerate(op.references):
            color, lty = _REFERENCE_STYLES[index % len(_REFERENCE_STYLES)]
            lines.append(f"abline(v = {E.format_number(value)}, col = {color}, lty = {lty}, lwd = 2)")
            fills.append("NA")
            labels.append(f'"{label}"')
            ltys.append(lty)
            colors.append(color)
        lines += [
            f"legend(\"topright\", legend = c({', '.join(labels)}),",
            f"       fill = c({', '.join(fills)}), border = NA,",
            f"       lty = c({', '.join(ltys)}), col = c({', '.join(colors)}), lwd = 2, bty = \"n\")",
            f"print(sapply({op.table}[, c({_quoted(column for column, _ in op.columns)})],",
            f"             function(x) sum(x < {lower} | x > {upper})))  # aralık dışında kalan değer sayısı",
        ]
        return lines

    def _plot(self, op: Plot) -> list[str]:
        frame, x = op.frame, op.x
        grid = E.Dialect(
            variable=lambda name: "izgara",
            coefficient=lambda model, term: f'coef({model})[["{self._key(model, term)}"]]',
            functions=self.dialect(frame).functions,
            power="^",
        )
        setup = [f"izgara <- seq(min({frame}${x}), max({frame}${x}), length.out = 200)"]
        ranges: list[str] = []
        drawing: list[str] = []
        legend = {"label": [], "col": [], "pch": [], "lty": [], "lwd": []}
        curves = 0
        for layer, style in zip(op.layers, layer_styles(op.layers)):
            color = f'"{style.color}"'
            if isinstance(layer, MeanPoints):
                mean, count = f"ort_{layer.y}", f"n_{layer.y}"
                setup += [
                    f"{mean} <- tapply({frame}${layer.y}, {frame}${x}, mean)",
                    f"{count} <- tapply({frame}${layer.y}, {frame}${x}, length)",
                ]
                ranges.append(mean)
                drawing.append(
                    f"points(as.numeric(names({mean})), {mean}, pch = 16, col = {color}, cex = sqrt({count}) / 12)"
                )
                marks = ("16", "NA", "NA")
            elif isinstance(layer, Scatter):
                ranges.append(f"{frame}${layer.y}")
                drawing.append(
                    f"points({frame}${x}, {frame}${layer.y}, pch = 16, cex = 0.5, col = adjustcolor({color}, 0.4))"
                )
                marks = ("16", "NA", "NA")
            elif isinstance(layer, Curve):
                curves += 1
                name = f"egri_{curves}"
                values = E.render(layer.expr, grid)
                if not E.variables(layer.expr):
                    values = f"rep({values}, length(izgara))"
                setup.append(f"{name} <- {values}")
                ranges.append(name)
                lty = 2 if style.dashed else 1
                pattern = f", lty = {lty}" if style.dashed else ""
                drawing.append(f"lines(izgara, {name}, col = {color}, lwd = 2{pattern})")
                marks = ("NA", str(lty), "2")
            elif isinstance(layer, ModelLine):
                ranges.append(f"fitted({layer.model})")
                lty = 2 if style.dashed else 1
                drawing.append(
                    f'abline(a = coef({layer.model})[["(Intercept)"]], b = coef({layer.model})[["{x}"]], '
                    f"col = {color}, lwd = 2, lty = {lty})"
                )
                marks = ("NA", str(lty), "2")
            else:  # ZeroLine
                ranges.append("0")
                drawing.append(f"abline(h = 0, col = {color}, lty = 3)")
                marks = ("NA", "3", "1")
            legend["label"].append(f'"{layer.label}"')
            legend["col"].append(color)
            legend["pch"].append(marks[0])
            legend["lty"].append(marks[1])
            legend["lwd"].append(marks[2])
        lines = setup + [
            f"plot(NA, xlim = range({frame}${x}), ylim = range(c({', '.join(ranges)})),",
            f'     xlab = "{op.x_label}", ylab = "{op.y_label}", main = "{op.title}")',
            *drawing,
            f'legend("topleft", legend = c({", ".join(legend["label"])}),',
            f'       col = c({", ".join(legend["col"])}), pch = c({", ".join(legend["pch"])}),',
            f'       lty = c({", ".join(legend["lty"])}), lwd = c({", ".join(legend["lwd"])}), bty = "n")',
        ]
        return lines

    # --- Notlarla karşılaştırma -----------------------------------------
    def target(self, target) -> str:
        if isinstance(target, StatTarget):
            values = f"{target.frame}${target.variable}"
            if target.where is not None:
                variable, value = target.where
                values = f"{values}[{target.frame}${variable} == {E.format_number(value)}]"
            if target.stat == "count":
                return f"sum(!is.na({values}))"
            return f"{_STAT[target.stat]}({values})"
        if isinstance(target, CoefTarget):
            coefficient = f'coef({target.model})[["{self._key(target.model, target.term)}"]]'
            if target.quantity == "coef":
                return coefficient
            if target.quantity == "se_hc1":
                return f'sqrt(diag(vcovHC({target.model}, type = "HC1")))[["{_term(target.term)}"]]'
            if target.quantity == "p":
                return f"2 * pnorm(-abs({coefficient} / {self._standard_error(target.model, target.term)}))"
            return self._standard_error(target.model, target.term)
        if isinstance(target, ModelTarget):
            if target.quantity == "r2":
                return f"summary({target.model})$r.squared"
            return f"nobs({target.model})"
        if isinstance(target, ScalarTarget):
            return target.name
        if isinstance(target, TableTarget):
            return f'{target.table}["{table_row_text(target.row)}", "{target.column}"]'
        raise TypeError(f"Tanınmayan hedef: {type(target).__name__}")

    def check_lines(self, checks: tuple[Check, ...]) -> list[str]:
        lines = ['cat("Notlarla karşılaştırma:\\n")']
        for check in checks:
            expected = f"{check.expected:.{check.decimals}f}"
            lines.append(
                f'kontrol_et("{check.label}", {self.target(check.target)}, {expected}, {check.decimals})'
            )
        return lines

    def closing(self) -> list[str]:
        return ['cat("\\nBütün değerler ders notlarıyla uyuşuyor.\\n")']
