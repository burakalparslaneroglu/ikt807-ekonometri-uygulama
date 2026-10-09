"""Notlar §2.1.1 ve §3.10 için iki ek ders deneyi; aynı işlem DSL'si."""

from __future__ import annotations

import numpy as np

from core.labs import expr as E
from core.labs.runner import LabState
from core.labs.sezgi import SimExperiment, SimMetric, SimParameter, number, plain
from core.labs.spec import Draw, NewSample, Derive, OLS, Plot, Scatter, Curve, ModelLine, NoteRef, ScalarTable

N = SimParameter("n", "Gözlem sayısı", 200, 5000, 2000, 200,
                 "Aynı tohumla n'yi değiştirin; tek örneklemdeki sapma yanlılık kanıtı değildir.", integer=True)


def _collinearity(p):
    rho = float(p["rho"])
    return (
        NewSample("sim", int(p["n"]), 820),
        Draw("sim", "x2", "normal", 0, 1, "X_2"),
        Draw("sim", "v", "normal", 0, 1, "Bağımsız V"),
        Derive("sim", "x1", E.add(E.mul(rho, E.var("x2")), E.mul(float(np.sqrt(1-rho*rho)), E.var("v"))),
               "X_1 = ρ X_2 + √(1−ρ²) V"),
        Draw("sim", "e", "normal", 0, 1, "Bağımsız e, varyansı 1"),
        Derive("sim", "y", E.add(E.add(E.add(1, E.mul(0.5, E.var("x1"))), E.mul(0.8, E.var("x2"))), E.var("e")),
               "Y = 1 + 0,5 X_1 + 0,8 X_2 + e"),
        OLS("yardimci", "sim", "x1", ("x2",)),
        OLS("tam", "sim", "y", ("x1", "x2")),
        Plot("sim", "x2", (Scatter("x1", "Gözlemler"), ModelLine("yardimci", "X_1'in X_2 üzerine projeksiyonu")),
             "X_2", "X_1", "Ortak hareket artarken ayrı bilgi azalır"),
        ScalarTable((("Tam model: X_1 katsayısı", E.coef("tam", "x1")),
                     ("Tam model: X_1 standart hatası", E.se("tam", "x1")),
                     ("Tam model: X_2 katsayısı", E.coef("tam", "x2")),
                     ("Tam model: X_2 standart hatası", E.se("tam", "x2"))), "katsayilar", decimals=4),
    )


def collinearity_se(state: LabState) -> float:
    """σ²=1 altında tasarıma koşullu gerçek SH; FWL artık karelerinden."""
    return float(1 / np.sqrt(np.square(state.models["yardimci"].resid).sum()))


def _collinearity_metrics(state, p):
    model = state.models["tam"]
    return (
        SimMetric("Gerçek β₁", "0,5000", "DGP'de ρ değişirken sabit."),
        SimMetric("Tahmin β̂₁", plain(model.params["x1"]), "Tam model; X_2 içeride."),
        SimMetric("Tahmin edilen SH", plain(model.bse["x1"]), "Klasik SH; bu DGP homoskedastik."),
        SimMetric("Gerçek koşullu SH", plain(collinearity_se(state)), "σ²=1 ve FWL artık kareleriyle tam koşullu değer."),
        SimMetric("Anakütle varyans şişmesi", plain(1 / (1-p["rho"]**2), 2), "1/(1−ρ²); sonlu örneklem SH oranı değildir."),
    )


COLLINEARITY = SimExperiment(
    topic_key="konu02", number=4, title="Çoklu doğrusal bağlantı: yanlılık mı, belirsizlik mi?",
    question="Doğru modelde regresörler birlikte hareket edince katsayı neden oynaklaşır?",
    note=NoteRef("2.1.1", 0), parameters=(N, SimParameter("rho", "Regresör korelasyonu ρ", 0, 0.95, 0.8, 0.05,
                                                               "ρ = 1 rank kaybıdır; deney 0,95 ile sınırlıdır.")),
    dgp=lambda p: (r"X_{2i},V_i,e_i\overset{ind}{\sim}N(0,1),\quad \text{gözlemler i.i.d.}",
                   rf"X_{{1i}}=\rho X_{{2i}}+\sqrt{{1-\rho^2}}V_i,\quad\rho={number(p['rho'],2)}",
                   r"Y_i=1+0{,}5X_{1i}+0{,}8X_{2i}+e_i,\quad\operatorname{Var}(e_i\mid X_{1i},X_{2i})=1"),
    dgp_note="X_1 ve X_2 skaler regresörler; V bağımsız bileşen, e bağımsız hata. İki regresör de modele girer. "
             "FWL artığı r_1 = X_1 − proj(X_1|1,X_2); anakütlede Var(r_1)=1−ρ².",
    look_at=("**Grafik** — ρ arttıkça X_1'in X_2'den ayrı hareketi azalır.",
             "**Ölçüler** — gerçek β₁ sabit kalırken koşullu SH büyür; tek bir tahminin kayması sistematik yanlılık değildir."),
    build=_collinearity, metrics=_collinearity_metrics,
    takeaway=lambda s,p: "Tam model doğru ve E[e|X]=0: OLS tasarıma koşullu yansızdır. Korelasyon, hedefi değiştirmeden "
                        "ayrı regresör etkisini öğrenmeyi zorlaştırır. ρ=1'de iki eğim ayrı tanımlanamaz; büyük n rank kaybını çözmez.",
    labels=(("x1", "X_1"), ("x2", "X_2")),
    tables=(("katsayilar", "Tam model: katsayılar ve standart hatalar"),),
)


def collider_target(b: float) -> float:
    return 1 - b / 2


def _collider(p):
    b = float(p["b"])
    return (
        NewSample("sim", int(p["n"]), 830),
        Draw("sim", "q", "uniform", 0, 1, "Tedavinin bağımsız atama çekilişi"),
        Derive("sim", "d", E.compare("gt", E.var("q"), 0.5), "D = 1{Q > 0,5}, rassal atama"),
        Draw("sim", "u", "normal", 0, 1, "Gözlenmeyen U"),
        Draw("sim", "v", "normal", 0, 1, "Bağımsız V"),
        Draw("sim", "e", "normal", 0, 1, "Bağımsız e"),
        Derive("sim", "m", E.add(E.add(E.mul(b, E.var("d")), E.var("u")), E.var("v")), "M = b D + U + V: çarpışan kontrol"),
        Derive("sim", "y", E.add(E.add(E.var("d"), E.var("u")), E.var("e")), "Y = θ D + U + e, θ=1"),
        OLS("dogru", "sim", "y", ("d",)),
        OLS("yanlis_kontrol", "sim", "y", ("d", "m")),
        ScalarTable((("Kontrolsüz: D katsayısı", E.coef("dogru", "d")),
                     ("Kontrolsüz: D standart hatası", E.se("dogru", "d")),
                     ("M kontrollü: D katsayısı", E.coef("yanlis_kontrol", "d")),
                     ("M kontrollü: D standart hatası", E.se("yanlis_kontrol", "d")),
                     ("M kontrollü: M katsayısı", E.coef("yanlis_kontrol", "m")),
                     ("M kontrollü: M standart hatası", E.se("yanlis_kontrol", "m"))), "katsayilar", decimals=4),
        Plot("sim", "m", (Scatter("u", "U ve M"), Curve(E.mul(0.5, E.var("m")), "E[U|M,D=0]"),
                           Curve(E.mul(0.5, E.sub(E.var("m"), b)), "E[U|M,D=1]", dashed=True)),
             "Tedavi sonrası kontrol M", "Gözlenmeyen U",
             "Simülasyonda gözlenen U: D → M ← U → Y yolu"),
    )


COLLIDER = SimExperiment(
    topic_key="konu03", number=4, title="Yanlış kontrol: rassal atamada bile yanlılık",
    question="Tedavi sonrası bir çarpışanı kontrol etmek neden yeni yanlılık yaratır?",
    note=NoteRef("3.10", 0), parameters=(N, SimParameter("b", "D'nin kontrol M üzerindeki etkisi b", 0, 2, 1, 0.25,
                                                               "b=0 iken D → M oku kaybolur; b>0 bu yolu oluşturur.")),
    dgp=lambda p: (r"D_i\sim\operatorname{Bernoulli}(0{,}5),\quad U_i,V_i,e_i\overset{ind}{\sim}N(0,1),\quad D_i\perp(U_i,V_i,e_i)",
                   rf"Y_i=\theta D_i+U_i+e_i,\quad\theta=1,\qquad M_i=bD_i+U_i+V_i,\quad b={number(p['b'],2)}",
                   r"D\longrightarrow M\longleftarrow U\longrightarrow Y,\qquad\operatorname{plim}\widehat\beta_D^{\,Y\sim1+D+M}=1-\frac b2"),
    dgp_note="D skaler tedavi göstergesi; U gözlenmeyen sonuç belirleyicisi; M tedaviden sonra ölçülen çarpışan; "
             "V ve e bağımsız hatalar. M sonucu etkileyen bir aracı değildir. Gözlemler i.i.d.; U yalnız simülasyonda görünür.",
    look_at=("**Tablo** — Y~1+D toplam etki θ=1'i; Y~1+D+M ise 1−b/2 projeksiyon katsayısını hedefler.",
             "**Deney** — b'yi 0'dan 2'ye artırın; n'yi büyütmek yanlış kontrolün yarattığı yanlılığı gidermez."),
    build=_collider,
    metrics=lambda s,p: (
        SimMetric("Toplam etki θ (DGP)", "1,0000", "Rassal atamayla tanımlanır."),
        SimMetric("Kontrolsüz β̂_D", plain(s.models["dogru"].params["d"]), "Y~1+D."),
        SimMetric("M kontrollü β̂_D", plain(s.models["yanlis_kontrol"].params["d"]), "Y~1+D+M."),
        SimMetric("M kontrollü hedef (DGP)", plain(collider_target(p["b"])), "1−b/2; koşullandırmayla açılan yol.")),
    takeaway=lambda s,p: "Rassal atama D ile U'yu bağımsız kılar. b>0 iken M'ye koşullandırmak bu bağımsızlığı bozar: "
                        "aynı M'de yüksek D, ortalamada düşük U ile eşleşir. Kontrol sayısını artırmak nedensel geçerliliği artırmaz. "
                        "b=0'da çarpışan yol yoktur; iki regresyon da θ=1'i hedefler.",
    labels=(("d", "Tedavi D"), ("m", "Çarpışan M")),
    tables=(("katsayilar", "Rassal atamada kontrolsüz ve M kontrollü model"),),
)
