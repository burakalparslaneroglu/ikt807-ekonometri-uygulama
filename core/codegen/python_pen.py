"""Python kodu: model seçimi, düzenlileştirme ve DML işlemleri (Konu 11–12).

Yardımcı fonksiyonlar uygulamanın hesabıyla (``core.labs.penalized``) aynı formülleri ve aynı çözücüyü kullanır:
Ridge kapalı biçim (özdeğer ayrışımı), Lasso ve Elastic Net scikit-learn ``enet_path`` (sıcak başlangıç, tolerans
1e-12). Üretilen betik uygulamanın sayılarını verir; eşitlik testlerle denetlenir.
"""

from __future__ import annotations

from core.codegen.base import DARK, GRAY, SERIES_COLORS
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
    dml_fold_key,
    penalized_key,
)

PENALTY_NAMES = {"ols": "Cezasız EKK", "ridge": "Ridge", "lasso": "Lasso", "enet": "Elastic Net"}
SCALE_NOTES = {
    "ridge": "Ridge, SSE ölçeği: (Y − Xβ)'(Y − Xβ) + λβ'β",
    "lasso": "Lasso, yazılım ölçeği: (1/(2n))‖Y − Xβ‖² + λ‖β‖₁",
    "enet": "Elastic Net, yazılım ölçeği: (1/(2n))‖Y − Xβ‖² + λ[r‖β‖₁ + (1 − r)/2·‖β‖²]",
}

PENALIZED_HELPER = [
    "def ceza_izgarasi(ust, alt, nokta):",
    '    """10^ust\'ten 10^alt\'a logaritmik eşit aralıklı ceza ızgarası, büyükten küçüğe."""',
    "    return np.logspace(ust, alt, nokta)",
    "",
    "",
    "def tasarim(egitim, diger, sayisal, kategorik=(), olcekle=True):",
    '    """Eğitim verisinden öğrenilen ölçek ve kategori düzeyleriyle iki tasarım matrisi ve terim adları.',
    "",
    "    Sürekli değişkenler eğitim ortalaması ve (n'e bölünen) standart sapmasıyla ölçeklenir. Kategori göstergeleri",
    "    eğitim verisinde görülen düzeylerden kurulur, ilk düzey referanstır; eğitimde görülmeyen düzey sıfır alır.",
    '    """',
    "    xa = egitim[list(sayisal)].to_numpy(dtype=float)",
    "    xb = diger[list(sayisal)].to_numpy(dtype=float)",
    "    if olcekle:",
    "        ortalama, ss = xa.mean(axis=0), xa.std(axis=0)",
    "        xa, xb = (xa - ortalama) / ss, (xb - ortalama) / ss",
    "    bloklar_a, bloklar_b, adlar = [xa], [xb], list(sayisal)",
    "    for ad in kategorik:",
    "        duzeyler = np.sort(egitim[ad].unique())[1:]",
    "        bloklar_a.append((egitim[ad].to_numpy()[:, None] == duzeyler[None, :]).astype(float))",
    "        bloklar_b.append((diger[ad].to_numpy()[:, None] == duzeyler[None, :]).astype(float))",
    '        adlar += [f"{ad}={float(duzey):g}" for duzey in duzeyler]',
    "    return np.column_stack(bloklar_a), np.column_stack(bloklar_b), adlar",
    "",
    "",
    'def ceza_yolu(xa, ya, xb, izgara, ceza, l1=1.0):',
    '    """Izgaradaki her ceza için sabit terim, katsayılar ve xb tahminleri; sabit terim cezasızdır (veri merkezlenir).',
    "",
    '    ceza: "ols"; "ridge" (SSE ölçeği, (Y − Xβ)\'(Y − Xβ) + λβ\'β); "lasso" veya "enet" (yazılım ölçeği,',
    "    (1/(2n))‖Y − Xβ‖² + λ[r‖β‖₁ + (1 − r)/2·‖β‖²], r = l1). Lasso yolu sıcak başlangıçla, çok küçük",
    "    toleransla çözülür: R glmnet ve Stata'daki koordinat inişi aynı optimuma yakınsar.",
    '    """',
    "    izgara = np.atleast_1d(np.asarray(izgara, dtype=float))",
    "    xo, yo = xa.mean(axis=0), ya.mean()",
    "    xc, yc = xa - xo, ya - yo",
    '    if ceza == "ols":',
    "        katsayi = np.repeat(np.linalg.lstsq(xc, yc, rcond=None)[0][:, None], len(izgara), axis=1)",
    '    elif ceza == "ridge":',
    "        deger, vektor = np.linalg.eigh(xc.T @ xc)",
    "        katsayi = vektor @ ((vektor.T @ (xc.T @ yc))[:, None] / (deger[:, None] + izgara[None, :]))",
    "    else:",
    '        oran = 1.0 if ceza == "lasso" else l1',
    "        _, katsayi, _ = enet_path(xc, yc, l1_ratio=oran, alphas=izgara, precompute=True, tol=1e-12,",
    "                                  max_iter=1_000_000)",
    "    sabit = yo - xo @ katsayi",
    "    return sabit, katsayi, sabit[None, :] + xb @ katsayi",
    "",
    "",
    'def ceza_cv(veri, y, sayisal, kategorik, kat, izgara, ceza, kural="min", l1_oranlari=(1.0,), olcekle=True):',
    '    """K-katlı CV: ölçek ve göstergeler her katta o katın eğitim verisinden öğrenilir.',
    "",
    "    Ölçüt kat ortalama karesel hatalarının ağırlıksız ortalaması, SH katlar arası standart sapma/√K. \"min\": en",
    '    küçük CV; "1se": en küçüğün bir SH\'si içindeki en büyük ceza. Döndürür: (L1 ağırlığı, ceza, CV tablosu).',
    '    """',
    "    kat = np.asarray(kat)",
    "    anahtarlar = np.unique(kat)",
    "    yv = veri[y].to_numpy(dtype=float)",
    '    oranlar = list(l1_oranlari) if ceza == "enet" else [1.0]',
    "    hata = np.empty((len(oranlar), len(anahtarlar), len(izgara)))",
    "    for j, k in enumerate(anahtarlar):",
    "        e, v = kat != k, kat == k",
    "        xa, xb, _ = tasarim(veri[e], veri[v], sayisal, kategorik, olcekle)",
    "        for r, oran in enumerate(oranlar):",
    "            tahmin = ceza_yolu(xa, yv[e], xb, izgara, ceza, oran)[2]",
    "            hata[r, j] = ((yv[v][:, None] - tahmin) ** 2).mean(axis=0)",
    "    ortalama = hata.mean(axis=1)",
    "    sh = hata.std(axis=1, ddof=1) / np.sqrt(len(anahtarlar))",
    "    r, g = np.unravel_index(np.argmin(ortalama), ortalama.shape)  # eşitlikte ilk (en büyük) ceza",
    '    if kural == "1se":',
    "        g = int(np.flatnonzero(ortalama[r] <= ortalama[r, g] + sh[r, g])[0])",
    '    tablo = pd.DataFrame({"lambda": izgara, "cv_ort": ortalama[r], "cv_sh": sh[r]})',
    "    return oranlar[r], float(izgara[g]), tablo",
    "",
    "",
    "class CezaSonucu:",
    '    """params (sabit dahil), lam, l1, test_mse, egitim_mse, sifirdan, norm, cv, yol, tahmin, secilen, nobs."""',
    "",
    "    def __init__(self, **alanlar):",
    "        self.__dict__.update(alanlar)",
    "",
    "",
    "def _sonuc(sabit, b, lam, oran, yv, tahmin, e, cv=None, yol=None):",
    "    artik = yv - tahmin",
    "    return CezaSonucu(",
    '        params=pd.concat([pd.Series({"Intercept": sabit}), b]), lam=lam, l1=oran,',
    "        test_mse=float(np.mean(artik[~e] ** 2)) if (~e).any() else float(\"nan\"),",
    "        egitim_mse=float(np.mean(artik[e] ** 2)), sifirdan=int((b != 0).sum()), norm=float(np.sqrt((b**2).sum())),",
    "        cv=cv, yol=yol, tahmin=tahmin, secilen=list(b.index[b != 0]), nobs=int(e.sum()),",
    "    )",
    "",
    "",
    'def ceza_modeli(veri, y, sayisal, kategorik=(), ceza="ols", izgara=None, egitim=None, kat=None, kural="min",',
    "                l1_oranlari=(1.0,), olcekle=True, yol=False):",
    '    """Eğitim satırlarında (egitim = 1) tahmin; ceza eğitim katlarıyla (kat) CV\'de seçilir; test: egitim = 0."""',
    "    e = np.ones(len(veri), dtype=bool) if egitim is None else np.asarray(egitim).astype(bool)",
    "    egt = veri[e]",
    '    oran = l1_oranlari[0] if ceza == "enet" else 1.0',
    "    lam, cv = 0.0, None",
    '    if ceza != "ols":',
    "        oran, lam, cv = ceza_cv(egt, y, sayisal, kategorik, np.asarray(kat)[e], izgara, ceza, kural, l1_oranlari,",
    "                                olcekle)",
    "    xa, xb, adlar = tasarim(egt, veri, sayisal, kategorik, olcekle)",
    "    yv = veri[y].to_numpy(dtype=float)",
    "    sabit, katsayi, tahmin = ceza_yolu(xa, yv[e], xb, [lam], ceza, oran)",
    "    tablo = None",
    "    if yol:  # bütün eğitim örnekleminde ızgara boyunca katsayılar",
    "        tablo = pd.DataFrame(ceza_yolu(xa, yv[e], xa[:1], izgara, ceza, oran)[1].T, columns=adlar,",
    '                             index=pd.Index(izgara, name="lambda"))',
    "    return _sonuc(sabit[0], pd.Series(katsayi[:, 0], index=adlar), lam, oran, yv, tahmin[:, 0], e, cv, tablo)",
]

POST_HELPER = [
    "def post_secim(veri, y, kaynak, sayisal, kategorik=(), egitim=None, olcekle=True):",
    '    """Post-Lasso: kaynak modelin sıfırdan farklı katsayılı terimleri üzerinde cezasız EKK (aynı örneklem ve ölçek)."""',
    "    e = np.ones(len(veri), dtype=bool) if egitim is None else np.asarray(egitim).astype(bool)",
    "    xa, xb, adlar = tasarim(veri[e], veri, sayisal, kategorik, olcekle)",
    "    sec = [adlar.index(ad) for ad in kaynak.secilen]",
    "    yv = veri[y].to_numpy(dtype=float)",
    '    sabit, katsayi, tahmin = ceza_yolu(xa[:, sec], yv[e], xb[:, sec], [0.0], "ols")',
    "    b = pd.Series(0.0, index=adlar)",
    "    b.iloc[sec] = katsayi[:, 0]",
    "    return _sonuc(sabit[0], b, 0.0, 1.0, yv, tahmin[:, 0], e)",
]

DML_HELPER = [
    "class DMLSonucu:",
    '    """θ̂ ve SH; statsmodels benzeri params, bse ve nobs; artıklar (u, v) ve kat tablosu."""',
    "",
    "    def __init__(self, theta, se, nobs, artiklar=None, katlar=None):",
    "        self.theta, self.se, self.nobs = float(theta), float(se), int(nobs)",
    '        self.params, self.bse = pd.Series({"theta": self.theta}), pd.Series({"theta": self.se})',
    "        self.artiklar, self.katlar = artiklar, katlar",
    "",
    "",
    "def artik_regresyonu(u, v, kume=None):",
    '    """θ̂ = Σv̂û/Σv̂² (sabitsiz); SH: HC1 (n/(n − 1)) ya da küme-dayanıklı (skorlar küme içinde toplanır, G/(G − 1))."""',
    "    teta = (v @ u) / (v @ v)",
    "    skor = v * (u - teta * v)",
    "    if kume is None:",
    "        orta = (skor @ skor) * len(u) / (len(u) - 1)",
    "    else:",
    "        toplam = pd.Series(skor).groupby(np.asarray(kume)).sum().to_numpy()",
    "        orta = (toplam @ toplam) * len(toplam) / (len(toplam) - 1)",
    "    return float(teta), float(np.sqrt(orta) / (v @ v))",
    "",
    "",
    'def dml_capraz(veri, y, d, ozellikler, dis_kat=None, ic_kat=None, izgara=None, kural="min", kume=None,',
    '               ogrenici="lasso"):',
    '    """DML2: her dış kat k için m_Y(X) = E[Y|X] ve m_D(X) = E[D|X] k dışındaki gözlemlerde öğrenilir, k\'de artık',
    "    alınır; θ̂ artıkların artıklar üzerine sabitsiz regresyonudur. dis_kat None: çapraz uyarlama yok",
    '    (artıklaştırma). ogrenici "lasso": ceza iç katlarla (ic_kat) CV\'de seçilir; "ols": cezasız EKK.',
    '    """',
    "    n = len(veri)",
    "    dis = np.ones(n, dtype=int) if dis_kat is None else np.asarray(dis_kat).astype(int)",
    "    yv, dv = veri[y].to_numpy(dtype=float), veri[d].to_numpy(dtype=float)",
    "    y_sapka, d_sapka, katlar = np.zeros(n), np.zeros(n), []",
    "    for k in np.unique(dis):",
    "        degerlendirme = dis == k",
    "        e = ~degerlendirme if dis_kat is not None else np.ones(n, dtype=bool)",
    "        egt = veri[e]",
    '        kayit = {"kat": int(k)}',
    '        for hedef, depo, ek in ((y, y_sapka, "y"), (d, d_sapka, "d")):',
    '            if ogrenici == "ols":',
    '                lam, ceza = 0.0, "ols"',
    "            else:",
    '                _, lam, _ = ceza_cv(egt, hedef, ozellikler, (), np.asarray(ic_kat)[e], izgara, "lasso", kural)',
    '                ceza = "lasso"',
    "            xa, xb, _ = tasarim(egt, veri[degerlendirme], ozellikler)",
    "            _, katsayi, tahmin = ceza_yolu(xa, egt[hedef].to_numpy(dtype=float), xb, [lam], ceza)",
    "            depo[degerlendirme] = tahmin[:, 0]",
    '            kayit[f"lambda_{ek}"], kayit[f"sifirdan_{ek}"] = lam, int((katsayi[:, 0] != 0).sum())',
    "        katlar.append(kayit)",
    "    u, v = yv - y_sapka, dv - d_sapka",
    "    teta, sh = artik_regresyonu(u, v, kume)",
    '    artiklar = pd.DataFrame({"u": u, "v": v}, index=veri.index)',
    '    return DMLSonucu(teta, sh, n, artiklar, pd.DataFrame(katlar).set_index("kat"))',
]

DOUBLE_SELECTION_HELPER = [
    'def cift_secim(veri, y, d, kontroller, kat, izgara, kural="1se"):',
    '    """S_Y: Y ve her X, [1, D] üzerinde artıklaştırıldıktan sonra (D cezalanmasın diye) Lasso; S_D: D\'nin X üzerine',
    "    Lasso'su. Döndürür: yalnız-sonuç Post-Lasso Y ~ D + S_Y, double selection Y ~ D + (S_Y ∪ S_D) (HC1) ve",
    '    seçim kümeleri."""',
    "    kontroller = list(kontroller)",
    "    Z = np.column_stack([np.ones(len(veri)), veri[d].to_numpy(dtype=float)])",
    "    blok = veri[kontroller + [y]].to_numpy(dtype=float)",
    "    artik = pd.DataFrame(blok - Z @ np.linalg.lstsq(Z, blok, rcond=None)[0], columns=kontroller + [y],",
    "                         index=veri.index)",
    '    s_y = ceza_modeli(artik, y, kontroller, (), "lasso", izgara, kat=kat, kural=kural).secilen',
    '    s_d = ceza_modeli(veri, d, kontroller, (), "lasso", izgara, kat=kat, kural=kural).secilen',
    "    birlesim = [ad for ad in kontroller if ad in s_y or ad in s_d]",
    "",
    "    def hc1_ols(adlar):",
    '        return smf.ols(f"{y} ~ " + " + ".join([d] + adlar), data=veri).fit(cov_type="HC1")',
    "",
    '    return hc1_ols(s_y), hc1_ols(birlesim), {"sonuc": s_y, "tedavi": s_d, "birlesim": birlesim}',
]

COMPLEXITY_HELPER = [
    "def karmasiklik(veri, x, y, en_buyuk, egitim):",
    '    """Polinom derecesi 1, …, D için eğitim MSE (σ̂² = SSR/n), AIC = n + n·log(2πσ̂²) + 2K,',
    "    BIC = n + n·log(2πσ̂²) + K·log n (K = d + 2: katsayılar ve σ²), LOOCV (kaldıraçla kesin) ve test MSE.",
    '    """',
    "    e = np.asarray(egitim).astype(bool)",
    "    xv, yv = veri[x].to_numpy(dtype=float), veri[y].to_numpy(dtype=float)",
    "    satirlar = []",
    "    for d in range(1, en_buyuk + 1):",
    "        X = np.column_stack([xv**p for p in range(d + 1)])",
    "        q, r = np.linalg.qr(X[e])",
    "        b = np.linalg.solve(r, q.T @ yv[e])",
    "        artik = yv[e] - X[e] @ b",
    "        h = (q**2).sum(axis=1)  # kaldıraç değerleri",
    "        n, K = int(e.sum()), d + 2",
    "        s2 = float(artik @ artik) / n",
    "        temel = n + n * np.log(2 * np.pi * s2)",
    "        satirlar.append({",
    '            "derece": d, "egitim_mse": s2, "loocv": np.mean((artik / (1 - h)) ** 2),',
    '            "test_mse": np.mean((yv[~e] - X[~e] @ b) ** 2), "aic": temel + 2 * K, "bic": temel + K * np.log(n),',
    "        })",
    '    return pd.DataFrame(satirlar).set_index("derece")',
]


def _number(value: float) -> str:
    return E.format_number(value)


def _quoted(names) -> list[str]:
    return [f'"{name}"' for name in names]


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


def _list(gen, values) -> tuple[list[str], str]:
    """Adlı liste (ilk kullanımda tanımıyla) ya da satır içi liste."""

    lines, name = gen.use_list(values)
    if name is not None:
        return lines, name
    return [], "[" + ", ".join(_quoted(values)) + "]"


def list_definition(name: str, values) -> list[str]:
    return _wrapped(f"{name} = [", _quoted(values), "]")


def _grid(spec) -> str:
    high, low, count = spec
    return f"ceza_izgarasi({_number(high)}, {_number(low)}, {int(count)})"


def helper_names(ops) -> list[str]:
    names: list[str] = []
    if any(isinstance(op, (Penalized, PostSelection, CrossFitDML, DoubleSelection)) for op in ops):
        names.append("ceza")
    if any(isinstance(op, PostSelection) for op in ops):
        names.append("post")
    if any(isinstance(op, (CrossFitDML, DMLSplits)) for op in ops):
        names.append("dml")
    if any(isinstance(op, DoubleSelection) for op in ops):
        names.append("cift")
    if any(isinstance(op, ComplexityCurve) for op in ops):
        names.append("karmasiklik")
    return names


HELPERS = {
    "ceza": PENALIZED_HELPER,
    "post": POST_HELPER,
    "dml": DML_HELPER,
    "cift": DOUBLE_SELECTION_HELPER,
    "karmasiklik": COMPLEXITY_HELPER,
}


def needs_enet(ops) -> bool:
    return any(
        (isinstance(op, Penalized) and op.penalty in ("lasso", "enet"))
        or (isinstance(op, CrossFitDML) and op.learner == "lasso")
        or isinstance(op, (DoubleSelection, DMLSplits))
        for op in ops
    )


PLOTS = (DotPlot, CVCurve, CoefPath, EstimatePlot, LinePlot)


def operation(gen, op) -> list[str] | None:
    """Konu 11–12 işlemlerinin Python kodu; tanımadığı işlemde ``None``."""

    if isinstance(op, RowNumber):
        return [f"# {op.comment}", f'{op.frame}["{op.name}"] = np.arange(1, len({op.frame}) + 1)']
    if isinstance(op, GroupRank):
        return [
            f"# {op.comment}",
            f'{op.frame}["{op.name}"] = {op.frame}["{op.source}"].rank(method="dense").astype(int)',
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
        return _cv_curve(gen, op)
    if isinstance(op, CoefPath):
        return _coef_path(op)
    if isinstance(op, CrossFitDML):
        return _crossfit(gen, op)
    if isinstance(op, DMLSplits):
        return _splits(gen, op)
    if isinstance(op, DoubleSelection):
        return _double_selection(gen, op)
    if isinstance(op, EstimatePlot):
        return _estimate_plot(op)
    if isinstance(op, ComplexityCurve):
        return [
            "# Polinom derecesi boyunca model seçim ölçütleri; en küçük değerlerin dereceleri yazdırılır",
            f'{op.result} = karmasiklik({op.frame}, "{op.x}", "{op.y}", {op.max_degree}, {op.frame}["{op.sample}"])',
            f"print({op.result}.round(4))",
            *[f'{op.name}_d_{column} = int({op.result}["{column}"].idxmin())' for column in
              ("egitim_mse", "loocv", "test_mse", "aic", "bic")],
            f'print("En küçük ölçütün derecesi — LOOCV:", {op.name}_d_loocv, "AIC:", {op.name}_d_aic, "BIC:", '
            f'{op.name}_d_bic, "test:", {op.name}_d_test_mse)',
        ]
    if isinstance(op, LinePlot):
        return _line_plot(op)
    return None


def _dictionary(gen, op: Dictionary) -> list[str]:
    lines = [f"# {op.comment}"]
    if op.powers:
        lines += [
            f"for ad in [{', '.join(_quoted(op.powers))}]:",
            f"    for kuvvet in range(2, {op.degree + 1}):",
            f'        {op.frame}[f"{{ad}}_{{kuvvet}}"] = {op.frame}[ad].astype(float) ** kuvvet',
        ]
    if op.interactions:
        lines += [
            *_wrapped("taban = [", _quoted(op.base), "]"),
            "for i, a in enumerate(taban):",
            "    for b in taban[i + 1:]:",
            f'        {op.frame}[f"{{a}}_x_{{b}}"] = {op.frame}[a].astype(float) * {op.frame}[b].astype(float)',
        ]
    from core.labs.spec import dictionary_terms

    definition, name = gen.use_list(dictionary_terms(op))
    lines += definition
    if name is not None:
        lines.append(f"print(len({name}), \"terim\")")
    return lines


def _draw_columns(op: DrawColumns) -> list[str]:
    frame = op.frame
    if op.rho == 0:
        value = "rng.normal(0, 1, size=len({frame}))".format(frame=frame)
        return [
            f"# {op.comment}",
            "sutunlar = {}",
            f"for j in range(1, {op.count} + 1):",
            f'    sutunlar[f"{op.prefix}{{j}}"] = {value}',
            f"{frame} = pd.concat([{frame}, pd.DataFrame(sutunlar, index={frame}.index)], axis=1)",
        ]
    rho = _number(op.rho)
    return [
        f"# {op.comment}",
        f"# x1 = z1, xj = {rho}·x(j−1) + √(1 − {rho}²)·zj: Corr(xj, xk) = {rho}^|j − k|, her sütunun varyansı 1",
        "sutunlar = {}",
        f"for j in range(1, {op.count} + 1):",
        f"    z = rng.normal(0, 1, size=len({frame}))",
        f'    sutunlar[f"{op.prefix}{{j}}"] = z if j == 1 else {rho} * sutunlar[f"{op.prefix}{{j - 1}}"] + '
        f"np.sqrt(1 - {rho}**2) * z",
        f"{frame} = pd.concat([{frame}, pd.DataFrame(sutunlar, index={frame}.index)], axis=1)",
    ]


def _penalized(gen, op: Penalized) -> list[str]:
    lines: list[str] = []
    numeric_lines, numeric = _list(gen, op.numeric)
    lines += numeric_lines
    categorical = "()"
    if op.categorical:
        categorical_lines, categorical = _list(gen, op.categorical)
        lines += categorical_lines
    arguments = [op.frame, f'"{op.outcome}"', numeric, categorical]
    if op.penalty != "ols":
        arguments += [f'"{op.penalty}"', _grid(op.grid)]
    else:
        arguments.append('"ols"')
    if op.sample:
        arguments.append(f'egitim={op.frame}["{op.sample}"]')
    if op.penalty != "ols":
        arguments.append(f'kat={op.frame}["{op.folds}"]')
        if op.rule != "min":
            arguments.append(f'kural="{op.rule}"')
        if op.penalty == "enet":
            arguments.append(f"l1_oranlari=({', '.join(_number(r) for r in op.l1_ratios)},)")
    if not op.standardize:
        arguments.append("olcekle=False")
    if op.path:
        arguments.append("yol=True")
    if op.penalty == "ols":
        comment = f"# {op.name}: cezasız EKK" + (" (eğitim örnekleminde)" if op.sample else "")
    else:
        high, low, count = op.grid
        rule = "en küçük CV" if op.rule == "min" else "bir standart hata kuralı"
        comment = (f"# {SCALE_NOTES[op.penalty]}; ceza {count} noktalı ızgarada (10^{_number(high)} … 10^{_number(low)}) "
                   f"CV ile, {rule}")
    lines.append(comment)
    lines += _wrapped(f"{op.name} = ceza_modeli(", arguments, ")")
    lines += _scalar_lines(gen, op.name)
    if op.penalty == "ols":
        lines.append(f'print(f"{op.name}: test MSE = {{{op.name}.test_mse:.4f}}, katsayı = {{{op.name}.sifirdan}}")')
    else:
        lines.append(
            f'print(f"{op.name}: λ = {{{op.name}.lam:.6g}}, test MSE = {{{op.name}.test_mse:.4f}}, '
            f'sıfırdan farklı katsayı = {{{op.name}.sifirdan}}")'
        )
    return lines


_ATTRIBUTES = {"test_mse": "test_mse", "egitim_mse": "egitim_mse", "sifirdan": "sifirdan", "norm": "norm",
               "lambda": "lam", "l1": "l1"}


def _scalar_lines(gen, model: str) -> list[str]:
    """Kontrollerin kullandığı model skalerleri (ör. ``ridge_lambda``), ayrı değişken olarak."""

    return [
        f"{penalized_key(model, quantity)} = {model}.{attribute}"
        for quantity, attribute in _ATTRIBUTES.items()
        if penalized_key(model, quantity) in gen.scalar_refs
    ]


def _post_selection(gen, op: PostSelection) -> list[str]:
    source: Penalized = gen.models[op.source]
    lines: list[str] = []
    numeric_lines, numeric = _list(gen, source.numeric)
    lines += numeric_lines
    categorical = "()"
    if source.categorical:
        categorical_lines, categorical = _list(gen, source.categorical)
        lines += categorical_lines
    arguments = [source.frame, f'"{source.outcome}"', op.source, numeric, categorical]
    if source.sample:
        arguments.append(f'egitim={source.frame}["{source.sample}"]')
    if not source.standardize:
        arguments.append("olcekle=False")
    lines.append(f"# Post-Lasso: {op.source} modelinin seçtiği terimlerle cezasız EKK")
    lines += _wrapped(f"{op.name} = post_secim(", arguments, ")")
    lines += _scalar_lines(gen, op.name)
    lines += [
        f'print(f"{op.name}: test MSE = {{{op.name}.test_mse:.4f}}, seçilen terim = {{{op.name}.sifirdan}}")',
    ]
    return lines


def _metrics(op: ModelMetrics) -> list[str]:
    lines = ["# Dış-örneklem karşılaştırması: test MSE, sıfırdan farklı katsayı, ‖β̂‖₂ ve seçilen ceza", "satirlar = ["]
    for model, label in op.rows:
        lines.append(f'    ("{model}", "{label}", {model}),')
    lines += [
        "]",
        f"{op.result} = pd.DataFrame(",
        '    [{"model": ad, "etiket": etiket, "test_mse": m.test_mse, "sifirdan": m.sifirdan, "norm": m.norm,',
        '      "lambda": m.lam if m.lam > 0 else np.nan} for ad, etiket, m in satirlar]',
        ').set_index("model")',
        f"print({op.result}.round(6))",
    ]
    return lines


def _labels(pairs) -> str:
    return "{" + ", ".join(f'"{key}": "{label}"' for key, label in pairs) + "}"


def _dot_plot(op: DotPlot) -> list[str]:
    color = SERIES_COLORS[0][0]
    return [
        f"etiketler = {_labels(op.labels)}",
        f"degerler = {op.table}[\"{op.column}\"]",
        "konum = np.arange(len(degerler))[::-1]",
        "fig, ax = plt.subplots(figsize=(8, 4))",
        f'ax.plot(degerler, konum, "o", color="{color}", markersize=9, label="{op.x_label}")',
        "for deger, y in zip(degerler, konum):",
        f'    ax.annotate(f"{{deger:.{op.decimals}f}}", (deger, y), xytext=(8, 0), textcoords="offset points", va="center")',
        "ax.set_yticks(konum)",
        "ax.set_yticklabels([etiketler[ad] for ad in degerler.index])",
        "ax.margins(x=0.25)",
        f'ax.set_xlabel("{op.x_label}")',
        f'ax.set_title("{op.title}")',
        'ax.legend(loc="lower right")',
        'ax.grid(axis="x", alpha=0.3)',
        "plt.show()",
    ]


def _cv_curve(gen, op: CVCurve) -> list[str]:
    color, dark = SERIES_COLORS[0][0], DARK[0]
    model = op.model
    return [
        f"cv = {model}.cv",
        "fig, ax = plt.subplots(figsize=(8, 5))",
        f'ax.fill_between(cv["lambda"], cv["cv_ort"] - cv["cv_sh"], cv["cv_ort"] + cv["cv_sh"], color="{color}",',
        '                alpha=0.18, linewidth=0, label="±1 SH (katlar arası)")',
        f'ax.plot(cv["lambda"], cv["cv_ort"], color="{color}", linewidth=2, label="CV ortalama karesel hatası")',
        f'ax.axvline({model}.lam, color="{dark}", linestyle="--", linewidth=1.5, label=f"Seçilen λ = {{{model}.lam:.4g}}")',
        'ax.set_xscale("log")',
        f'ax.set_xlabel("{op.x_label}")',
        'ax.set_ylabel("CV ortalama karesel hatası")',
        f'ax.set_title("{op.title}")',
        "ax.legend()",
        "ax.grid(alpha=0.3)",
        "plt.show()",
    ]


def _coef_path(op: CoefPath) -> list[str]:
    color, gray, dark = SERIES_COLORS[0][0], GRAY[0], DARK[0]
    model = op.model
    return [
        f"yol = {model}.yol",
        f"vurgulu = [{', '.join(_quoted(op.highlight))}]",
        "digerleri = [ad for ad in yol.columns if ad not in vurgulu]",
        "fig, ax = plt.subplots(figsize=(8, 5))",
        "for i, ad in enumerate(digerleri):",
        f'    ax.plot(yol.index, yol[ad], color="{gray}", linewidth=0.8, label="{op.other_label}" if i == 0 else None)',
        "for i, ad in enumerate(vurgulu):",
        f'    ax.plot(yol.index, yol[ad], color="{color}", linewidth=2, label="{op.highlight_label}" if i == 0 else None)',
        f'ax.axvline({model}.lam, color="{dark}", linestyle="--", linewidth=1.5, label=f"CV ile seçilen λ = {{{model}.lam:.3g}}")',
        'ax.set_xscale("log")',
        "ax.invert_xaxis()  # ceza büyükten küçüğe: değişkenler modele sırayla girer",
        f'ax.set_xlabel("{op.x_label}")',
        'ax.set_ylabel("Katsayı")',
        f'ax.set_title("{op.title}")',
        "ax.legend()",
        "ax.grid(alpha=0.3)",
        "plt.show()",
    ]


def _crossfit(gen, op: CrossFitDML) -> list[str]:
    lines, features = _list(gen, op.features)
    arguments = [op.frame, f'"{op.outcome}"', f'"{op.treatment}"', features]
    arguments.append(f'{op.frame}["{op.outer}"]' if op.outer else "None")
    if op.learner == "lasso":
        arguments += [f'{op.frame}["{op.inner}"]', _grid(op.grid)]
        if op.rule != "min":
            arguments.append(f'"{op.rule}"')
    if op.cluster:
        arguments.append(f'kume={op.frame}["{op.cluster}"]')
    if op.learner != "lasso":
        arguments.append(f'ogrenici="{op.learner}"')
    learner = "Lasso (ceza iç katlarla CV'de)" if op.learner == "lasso" else "cezasız EKK"
    if op.outer:
        lines.append(f"# DML2: yardımcı modeller {learner}; her dış katın artığı o katı görmemiş modelden")
    else:
        lines.append(f"# Artıklaştırma: yardımcı modeller {learner}, bütün örneklemde (çapraz uyarlama yok)")
    se = "okul/küme SH" if op.cluster else "HC1 SH"
    lines += _wrapped(f"{op.name} = dml_capraz(", arguments, ")")
    if not gen.quiet:
        lines.append(f'print(f"θ̂ = {{{op.name}.theta:.4f}} ({se} {{{op.name}.se:.4f}}), n = {{{op.name}.nobs}}")')
    if op.outer and op.learner == "lasso":
        lines.append(f"print({op.name}.katlar)")
    if op.learner == "lasso":
        for part in ("y", "d"):
            for statistic in ("min", "max"):
                key = dml_fold_key(op.name, part, statistic)
                if key in gen.scalar_refs:
                    lines.append(f'{key} = {op.name}.katlar["sifirdan_{part}"].{statistic}()')
    if op.residuals is not None:
        u, v = op.residuals
        lines.append(f'{op.frame}["{u}"], {op.frame}["{v}"] = {op.name}.artiklar["u"], {op.name}.artiklar["v"]')
    return lines


def _splits(gen, op: DMLSplits) -> list[str]:
    source: CrossFitDML = gen.models[op.dml]
    lines, features = _list(gen, source.features)
    rules = [f'("{key}", "{label}", {_number(multiplier)})' for key, label, multiplier in op.rules]
    arguments = [source.frame, f'"{source.outcome}"', f'"{source.treatment}"', features, "kat"]
    if source.learner == "lasso":
        arguments += [f'{source.frame}["{source.inner}"]', _grid(source.grid)]
        if source.rule != "min":
            arguments.append(f'"{source.rule}"')
    if source.cluster:
        arguments.append(f'kume={source.frame}["{source.cluster}"]')
    if source.learner != "lasso":
        arguments.append(f'ogrenici="{source.learner}"')
    lines += [
        f"# Bölme duyarlılığı: dış katlar ⌊{op.folds}{{r·c}}⌋ + 1; aynı veri, öğrenici ve kat sayısı, yalnız okulların",
        "# katlara dağılımı değişir. Medyan birleştirme: θ̂_med = medyan θ̂_s, σ̂²_med = medyan{σ̂²_s + (θ̂_s − θ̂_med)²}",
        *_wrapped("kurallar = [", rules, "]"),
        "satirlar = []",
        "for anahtar, etiket, c in kurallar:",
        f'    x = {source.frame}["{op.rank}"].to_numpy(dtype=float) * c',
        f"    kat = np.floor({op.folds} * (x - np.floor(x))) + 1",
        *[f"    {line}" for line in _wrapped("s = dml_capraz(", arguments, ")")],
        '    satirlar.append({"kural": anahtar, "etiket": etiket, "theta": s.theta, "sh": s.se})',
        f'{op.result} = pd.DataFrame(satirlar).set_index("kural")',
        f'medyan = {op.result}["theta"].median()',
        f'medyan_sh = np.sqrt(np.median({op.result}["sh"] ** 2 + ({op.result}["theta"] - medyan) ** 2))',
        f"{op.name} = DMLSonucu(medyan, medyan_sh, len({source.frame}))",
        f'{op.name}_min, {op.name}_max = {op.result}["theta"].min(), {op.result}["theta"].max()',
        f"print({op.result}.round(3))",
        f'print(f"Medyan θ̂ = {{medyan:.3f}} (SH {{medyan_sh:.3f}}); aralık {{{op.name}_min:.3f}} – {{{op.name}_max:.3f}}")',
    ]
    return lines


def _double_selection(gen, op: DoubleSelection) -> list[str]:
    lines, controls = _list(gen, op.controls)
    arguments = [op.frame, f'"{op.outcome}"', f'"{op.treatment}"', controls, f'{op.frame}["{op.folds}"]', _grid(op.grid)]
    if op.rule != "1se":
        arguments.append(f'"{op.rule}"')
    rule = "bir standart hata kuralı" if op.rule == "1se" else "en küçük CV"
    lines += [
        f"# Yalnız-sonuç Post-Lasso ve double selection; Lasso cezaları CV ile, {rule}",
        *_wrapped(f"{op.name}_sonuc, {op.name}, {op.name}_kumeler = cift_secim(", arguments, ")"),
    ]
    tracked = ", ".join(_quoted(op.track))
    for suffix, key in (("ny", "sonuc"), ("nd", "tedavi"), ("n", "birlesim")):
        lines.append(f'{op.name}_{suffix} = len({op.name}_kumeler["{key}"])')
        if op.track:
            lines.append(f'{op.name}_{suffix}_iz = len(set({op.name}_kumeler["{key}"]) & {{{tracked}}})')
    if not gen.quiet:
        lines.append(f'print("Seçilen kontroller:", {{ad: len(k) for ad, k in {op.name}_kumeler.items()}})')
        lines.append(f'print({op.name}_sonuc.params["{op.treatment}"], {op.name}.params["{op.treatment}"])')
    return lines


def _estimate_plot(op: EstimatePlot) -> list[str]:
    color, gray, red = SERIES_COLORS[0][0], GRAY[0], SERIES_COLORS[1][0]
    lines = ["satirlar = ["]
    for label, model, term in op.rows:
        key = "Intercept" if term == E.INTERCEPT else term
        lines.append(f'    ("{label}", {model}.params["{key}"], {model}.bse["{key}"]),')
    lines += [
        "]",
        f'{op.result} = pd.DataFrame(satirlar, columns=["etiket", "tahmin", "sh"]).set_index("etiket")',
        f'{op.result}["alt"] = {op.result}["tahmin"] - {CI_MULTIPLIER} * {op.result}["sh"]',
        f'{op.result}["ust"] = {op.result}["tahmin"] + {CI_MULTIPLIER} * {op.result}["sh"]',
        f"print({op.result}.round(3))",
        f"konum = np.arange(len({op.result}))[::-1]",
        "fig, ax = plt.subplots(figsize=(8, 5))",
        f'ax.errorbar({op.result}["tahmin"], konum, xerr={CI_MULTIPLIER} * {op.result}["sh"], fmt="o", color="{color}",',
        '            capsize=4, linewidth=2, label="Tahmin ve %95 güven aralığı")',
    ]
    if op.splits:
        lines += [
            f'ax.scatter({op.splits}["theta"], np.full(len({op.splits}), konum[{op.splits_row}] - 0.25), s=18, '
            f'color="{gray}",',
            '           label="Farklı kat kurallarıyla DML tahminleri")',
        ]
    if op.truth is not None:
        lines.append(f'ax.axvline({_number(op.truth)}, color="{red}", linestyle="--", linewidth=1.5, label="{op.truth_label}")')
    lines += [
        "ax.set_yticks(konum)",
        f"ax.set_yticklabels({op.result}.index)",
        f'ax.set_xlabel("{op.x_label}")',
        f'ax.set_title("{op.title}")',
        "ax.legend()",
        'ax.grid(axis="x", alpha=0.3)',
        "plt.show()",
    ]
    return lines


def _line_plot(op: LinePlot) -> list[str]:
    columns = [column for column, _ in op.columns]
    lines = [f"cizim = {op.table}[[{', '.join(_quoted(columns))}]]"]
    if op.relative:
        lines.append("cizim = cizim - cizim.min()  # her ölçütten kendi en küçük değeri çıkarılır")
    lines.append("fig, ax = plt.subplots(figsize=(8, 5))")
    for (column, label), (color, _) in zip(op.columns, SERIES_COLORS):
        lines.append(f'ax.plot(cizim.index, cizim["{column}"], marker="o", color="{color}", linewidth=2, label="{label}")')
    lines += [
        f'ax.set_xlabel("{op.x_label}")',
        f'ax.set_ylabel("{op.y_label}")',
        f'ax.set_title("{op.title}")',
        "ax.legend()",
        "ax.grid(alpha=0.3)",
        "plt.show()",
    ]
    return lines
