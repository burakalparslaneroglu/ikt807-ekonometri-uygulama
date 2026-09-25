"""Laboratuvar tanımından R kodu üretir.

Varsayılan yığın temel R'dır. Dayanıklı çıkarım gereken konularda ``sandwich`` ve
``lmtest`` kullanılır; kovaryans türü kodda açıkça yazılır (ör. ``type = "HC1"``).
"""

from __future__ import annotations

from core.codegen.base import HANSEN_ARCHIVE_URL, Generator, layer_styles
from core.labs import expr as E
from core.labs.spec import (
    OLS,
    Check,
    CoefTarget,
    Curve,
    Derive,
    Describe,
    Draw,
    MeanPoints,
    ModelLine,
    NewSample,
    Plot,
    Predict,
    Scatter,
    Summaries,
    ZeroLine,
    GroupMeanPlot,
    GroupSummary,
    LoadHansen,
    ModelTarget,
    Operation,
    ProjectionPlot,
    RegressionTable,
    Scalar,
    ScalarTarget,
    ShowModel,
    StatTarget,
)

_STAT = {"count": "length", "sum": "sum", "mean": "mean", "sd": "sd", "median": "median", "min": "min", "max": "max"}
_COLORS = ('"#107C89"', '"#B3392F"', '"#2F9E6B"', '"#07373D"')


def _term(term: str) -> str:
    return "(Intercept)" if term == E.INTERCEPT else term


class RGenerator(Generator):
    language = "R"
    comment = "#"

    def dialect(self, frame: str) -> E.Dialect:
        return E.Dialect(
            variable=lambda name: f"{frame}${name}",
            coefficient=lambda model, term: f'coef({model})[["{_term(term)}"]]',
            functions={"log": "log", "exp": "exp", "sqrt": "sqrt", "maximum": "pmax", "minimum": "pmin", "round": "round"},
            power="^",
        )

    def vcov(self, model: str, name: str | None = None) -> str:
        """Modelin kovaryans matrisi ifadesi; ``name`` verilirse o değişken adıyla yazılır."""

        symbol = name or model
        settings = self.models.get(model)
        if settings is None or settings.vcov == "classic":
            return f"vcov({symbol})"
        if settings.vcov == "HC1":
            return f'sandwich::vcovHC({symbol}, type = "HC1")'
        return f'sandwich::vcovCL({symbol}, cluster = ~{settings.cluster}, type = "HC1")'

    def _needs_sandwich(self, operations) -> bool:
        return any(isinstance(op, OLS) and op.vcov != "classic" for op in operations)

    # --- Başlık ve yardımcılar ------------------------------------------
    def imports(self, operations: tuple[Operation, ...]) -> list[str]:
        lines: list[str] = []
        if self._needs_sandwich(operations):
            lines += [
                "# Gerekli paketler: install.packages(c(\"sandwich\", \"lmtest\"))",
                "library(sandwich)",
                "library(lmtest)",
                "",
            ]
        return lines

    def helpers(self, operations: tuple[Operation, ...], *, with_checks: bool) -> list[str]:
        lines: list[str] = []
        if any(isinstance(op, LoadHansen) for op in operations):
            lines += [
                "options(timeout = max(600, getOption(\"timeout\")))",
                f'hansen_arsiv <- "{HANSEN_ARCHIVE_URL}"',
                "",
                "# Dosyayı kendiniz indirdiyseniz yolunu yazın (ör. \"cps09mar.txt\").",
                "# NULL bırakırsanız veri Hansen'in sayfasından otomatik indirilir.",
                "yerel_dosya <- NULL",
                "",
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
                "              format(beklenen)))",
                "  if (abs(deger - beklenen) > tolerans) stop(etiket, \" notlarla uyuşmuyor.\")",
                "}",
                "",
            ]
        return lines

    # --- İşlemler --------------------------------------------------------
    def operation(self, op: Operation) -> list[str]:
        if isinstance(op, NewSample):
            return [
                "# Sabit tohum: betik her çalıştırmada aynı veriyi üretir",
                f"set.seed({op.seed})",
                f"{op.frame} <- data.frame(id = seq_len({op.nobs}))",
            ]
        if isinstance(op, Draw):
            a, b = E.format_number(op.first), E.format_number(op.second)
            call = f"rnorm(nrow({op.frame}), mean = {a}, sd = {b})" if op.distribution == "normal" \
                else f"runif(nrow({op.frame}), min = {a}, max = {b})"
            return [f"# {op.comment}", f"{op.frame}${op.name} <- {call}"]
        if isinstance(op, Predict):
            function = "fitted" if op.kind == "fitted" else "resid"
            return [f"{op.frame}${op.name} <- {function}({op.model})"]
        if isinstance(op, Summaries):
            entries = [f'  "{label}" = {_STAT[stat]}({op.frame}${variable})' for label, variable, stat in op.rows]
            return [f"{op.result} <- c(", ",\n".join(entries), ")", f"print(round({op.result}, 10))"]
        if isinstance(op, Plot):
            return self._plot(op)
        if isinstance(op, LoadHansen):
            names = ", ".join(f'"{c}"' for c in op.columns)
            return [
                f"# Değişken sırası: Hansen'in {op.dataset} açıklama belgesi",
                f"sutunlar <- c({names})",
                f'{op.frame} <- hansen_verisi("{op.member}", sutunlar, yerel_dosya)',
                f"print(dim({op.frame}))  # gözlem, değişken",
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
            terms = [f"factor({r})" if r in op.categorical else r for r in op.regressors]
            formula = f"{op.outcome} ~ " + " + ".join(terms)
            lines = [f"{op.name} <- lm({formula}, data = {op.frame})"]
            if op.categorical:
                lines.insert(0, "# Doymuş model: eğitimin her değeri için ayrı bir kukla")
            elif op.vcov == "classic":
                lines.append(f"print(round(coef({op.name}), 4))")
            else:
                lines.append(f"print(coeftest({op.name}, vcov. = {self.vcov(op.name)}))")
            return lines
        if isinstance(op, ShowModel):
            settings = self.models.get(op.model)
            if settings is None or settings.vcov == "classic":
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
                f'cat("{op.comment}:", sprintf("%.2f", {op.name}), "\\n")',
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

    def _plot(self, op: Plot) -> list[str]:
        frame, x = op.frame, op.x
        grid = E.Dialect(
            variable=lambda name: "izgara",
            coefficient=lambda model, term: f'coef({model})[["{_term(term)}"]]',
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
                setup.append(f"{name} <- {E.render(layer.expr, grid)}")
                ranges.append(name)
                drawing.append(f"lines(izgara, {name}, col = {color}, lwd = 2)")
                marks = ("NA", "1", "2")
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
            return f"{_STAT[target.stat]}({values})"
        if isinstance(target, CoefTarget):
            if target.quantity == "coef":
                return f'coef({target.model})[["{_term(target.term)}"]]'
            return f'sqrt(diag({self.vcov(target.model)}))[["{_term(target.term)}"]]'
        if isinstance(target, ModelTarget):
            if target.quantity == "r2":
                return f"summary({target.model})$r.squared"
            return f"nobs({target.model})"
        if isinstance(target, ScalarTarget):
            return target.name
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
