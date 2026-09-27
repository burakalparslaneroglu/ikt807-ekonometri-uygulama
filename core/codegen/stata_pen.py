"""Stata kodu: model seçimi, düzenlileştirme ve DML işlemleri (Konu 11–12).

Stata'nın ``lasso`` komutu kat değişkeni ve serbest ceza ızgarası almadığı için hesap kodda yazılı Mata
fonksiyonlarıyla yapılır: Ridge kapalı biçim (özdeğer ayrışımı), Lasso ve Elastic Net sıcak başlangıçlı koordinat
inişi (Gram matrisiyle, göreli değişim 1e-12'nin altına inene kadar), cezasız EKK QR. Bu, scikit-learn
``enet_path`` ve R ``glmnet`` ile aynı amaç fonksiyonunun aynı optimumudur; katlar ve ızgara açık verildiği için
sayılar Python ve R ile aynıdır. DML modelleri ``ereturn post`` ile Stata tahmin sonucu olarak saklanır
(``_b[theta]``, ``_se[theta]``); ``estimates table`` ve kontroller bu sonuçları okur.
"""

from __future__ import annotations

from core.codegen.base import DARK, GRAY, SERIES_COLORS
from core.codegen.python_pen import SCALE_NOTES
from core.labs import expr as E
from core.labs.runner import CI_MULTIPLIER
from core.labs.spec import (
    CoefPath,
    COMPLEXITY_COLUMNS,
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
)

PENALIZED_HELPER = [
    "* Konu 11–12 yardımcıları (Mata). Ceza ölçekleri: Ridge (Y − Xβ)'(Y − Xβ) + λβ'β; Lasso ve Elastic Net",
    "* (1/(2n))‖Y − Xβ‖² + λ[r‖β‖₁ + (1 − r)/2·‖β‖²], r = L1 ağırlığı. Sabit terim cezasızdır: veri merkezlenir.",
    "* Sürekli değişkenler eğitim ortalaması ve (n'e bölünen) standart sapmasıyla ölçeklenir; kategori göstergeleri",
    "* eğitimde görülen düzeylerden kurulur, ilk düzey referanstır. Sonuçlar Stata skalerlerine ve matrislerine yazılır:",
    "* <ad>_test_mse, <ad>_egitim_mse, <ad>_sifirdan, <ad>_norm, <ad>_lambda, <ad>_l1; <ad>_b, <ad>_cv, <ad>_yol.",
    "capture mata: mata drop ceza_izgarasi()",
    "capture mata: mata drop pen_*()",
    "mata:",
    "real matrix pen_sec(real matrix X, real colvector e)",
    "{",
    "    return(cols(X) ? select(X, e) : J(sum(e), 0, .))",
    "}",
    "",
    "real rowvector ceza_izgarasi(real scalar ust, real scalar alt, real scalar nokta)",
    "{",
    "    real rowvector u",
    "    u = ust :+ (0..(nokta - 1)) :* ((alt - ust) / (nokta - 1))",
    "    u[nokta] = alt",
    "    return(10 :^ u)",
    "}",
    "",
    "void pen_tasarim(real matrix Sa, real matrix Sb, real matrix Ka, real matrix Kb, real scalar olcekle,",
    "                 string rowvector sayisal, string rowvector kategorik, real matrix XA, real matrix XB,",
    "                 string rowvector adlar)",
    "{",
    "    real rowvector ort, ss, duzey",
    "    real scalar j, i",
    "    XA = Sa",
    "    XB = Sb",
    "    adlar = sayisal",
    "    if (olcekle & cols(Sa) > 0) {",
    "        ort = mean(Sa)",
    "        ss = sqrt(mean((Sa :- ort) :^ 2))",
    "        XA = (Sa :- ort) :/ ss",
    "        XB = (Sb :- ort) :/ ss",
    "    }",
    "    for (j = 1; j <= cols(Ka); j++) {",
    "        duzey = uniqrows(Ka[., j])'",
    "        if (cols(duzey) > 1) {",
    "            duzey = duzey[2..cols(duzey)]",
    "            XA = XA, (Ka[., j] :== duzey)",
    "            XB = XB, (Kb[., j] :== duzey)",
    "            for (i = 1; i <= cols(duzey); i++) adlar = adlar, (kategorik[j] + \"_\" + strofreal(duzey[i]))",
    "        }",
    "    }",
    "}",
    "",
    "// Aktif küme ve işaretlerle KKT sistemini kesin çözer: (G_AA + ek·I)b_A = c_A − esik·s_A. İşaretler tutarlı ve",
    "// aktif olmayanlarda |c_j − G_j·b| ≤ esik ise b optimumdur (1 döner); değilse koordinat inişi sürer (0 döner).",
    "real scalar pen_cila(real matrix G, real colvector c, real colvector b, real colvector q, real scalar esik,",
    "                     real scalar ek)",
    "{",
    "    real colvector A, s, bA, z, r",
    "    real scalar j",
    "    A = selectindex(b :!= 0)",
    "    if (rows(A) == 0) {",
    "        if (max(abs(c)) > esik) return(0)",
    "        q = J(rows(b), 1, 0)",
    "        return(1)",
    "    }",
    "    s = sign(b[A])",
    "    bA = lusolve(G[A, A] + ek * I(rows(A)), c[A] - esik * s)",
    "    if (sum(sign(bA) :!= s) > 0) return(0)",
    "    z = J(rows(b), 1, 0)",
    "    z[A] = bA",
    "    r = c - G * z",
    "    for (j = 1; j <= rows(b); j++) {",
    "        if (z[j] == 0 & abs(r[j]) > esik * (1 + 1e-10)) return(0)",
    "    }",
    "    b = z",
    "    q = G * z",
    "    return(1)",
    "}",
    "",
    "real matrix pen_yol(real matrix xc, real colvector yc, real rowvector izgara, string scalar ceza, real scalar l1)",
    "{",
    "    real matrix G, V, K",
    "    real rowvector L",
    "    real colvector c, b, q",
    "    real scalar n, p, g, j, t, eski, yeni, rho, esik, ek, degisim, olcek",
    "    n = rows(xc)",
    "    p = cols(xc)",
    "    if (ceza == \"ols\") return(J(1, cols(izgara), qrsolve(xc, yc)))",
    "    G = cross(xc, xc)",
    "    c = cross(xc, yc)",
    "    K = J(p, cols(izgara), 0)",
    "    if (ceza == \"ridge\") {",
    "        symeigensystem(G, V, L)",
    "        for (g = 1; g <= cols(izgara); g++) K[., g] = V * ((V' * c) :/ (L' :+ izgara[g]))",
    "        return(K)",
    "    }",
    "    // Lasso / Elastic Net: koordinat inişi, ızgara boyunca sıcak başlangıç; q = G·b",
    "    b = J(p, 1, 0)",
    "    q = J(p, 1, 0)",
    "    for (g = 1; g <= cols(izgara); g++) {",
    "        esik = n * izgara[g] * l1",
    "        ek = n * izgara[g] * (1 - l1)",
    "        for (t = 1; t <= 1000000; t++) {",
    "            degisim = 0",
    "            olcek = 0",
    "            for (j = 1; j <= p; j++) {",
    "                eski = b[j]",
    "                rho = c[j] - q[j] + G[j, j] * eski",
    "                yeni = (rho > esik ? (rho - esik) / (G[j, j] + ek) : (rho < -esik ? (rho + esik) / (G[j, j] + ek) : 0))",
    "                if (yeni != eski) {",
    "                    q = q + G[., j] * (yeni - eski)",
    "                    b[j] = yeni",
    "                    degisim = max((degisim, abs(yeni - eski) * sqrt(G[j, j])))",
    "                }",
    "                olcek = max((olcek, abs(yeni) * sqrt(G[j, j])))",
    "            }",
    "            if (degisim <= 1e-12 * olcek) break",
    "            if (pen_cila(G, c, b, q, esik, ek)) break",
    "        }",
    "        K[., g] = b",
    "    }",
    "    return(K)",
    "}",
    "",
    "real matrix pen_tahmin(real matrix XA, real colvector ya, real matrix XB, real rowvector izgara,",
    "                       string scalar ceza, real scalar l1, real matrix K)",
    "{",
    "    real rowvector xo, sabit",
    "    real scalar yo",
    "    xo = mean(XA)",
    "    yo = mean(ya)",
    "    K = pen_yol(XA :- xo, ya :- yo, izgara, ceza, l1)",
    "    sabit = yo :- xo * K",
    "    return((XB * K) :+ sabit)",
    "}",
    "",
    "void pen_cv(real colvector y, real matrix S, real matrix Kt, real colvector kat, real rowvector izgara,",
    "            string scalar ceza, real rowvector oranlar, string scalar kural, real scalar olcekle,",
    "            real scalar r_sec, real scalar g_sec, real matrix ORT, real matrix SH)",
    "{",
    "    real colvector anahtar, e, v",
    "    real matrix XA, XB, H, P, K",
    "    string rowvector adlar",
    "    real scalar KK, R, G, k, r, g, en_kucuk, sinir",
    "    anahtar = uniqrows(kat)",
    "    KK = rows(anahtar)",
    "    R = cols(oranlar)",
    "    G = cols(izgara)",
    "    H = J(R * KK, G, .)",
    "    for (k = 1; k <= KK; k++) {",
    "        e = kat :!= anahtar[k]",
    "        v = kat :== anahtar[k]",
    "        pen_tasarim(select(S, e), select(S, v), pen_sec(Kt, e), pen_sec(Kt, v), olcekle, J(1, cols(S), \"\"),",
    "                    J(1, cols(Kt), \"\"), XA, XB, adlar)",
    "        for (r = 1; r <= R; r++) {",
    "            P = pen_tahmin(XA, select(y, e), XB, izgara, ceza, oranlar[r], K)",
    "            H[(r - 1) * KK + k, .] = mean((select(y, v) :- P) :^ 2)",
    "        }",
    "    }",
    "    ORT = J(R, G, .)",
    "    SH = J(R, G, .)",
    "    for (r = 1; r <= R; r++) {",
    "        ORT[r, .] = mean(H[((r - 1) * KK + 1)..(r * KK), .])",
    "        for (g = 1; g <= G; g++) SH[r, g] = sqrt(variance(H[((r - 1) * KK + 1)..(r * KK), g])) / sqrt(KK)",
    "    }",
    "    // En küçük CV (eşitlikte ilk, yani en büyük ceza); 1se: en küçüğün bir SH'si içindeki en büyük ceza",
    "    r_sec = 1",
    "    g_sec = 1",
    "    en_kucuk = ORT[1, 1]",
    "    for (r = 1; r <= R; r++) {",
    "        for (g = 1; g <= G; g++) {",
    "            if (ORT[r, g] < en_kucuk) {",
    "                en_kucuk = ORT[r, g]",
    "                r_sec = r",
    "                g_sec = g",
    "            }",
    "        }",
    "    }",
    "    if (kural == \"1se\") {",
    "        sinir = en_kucuk + SH[r_sec, g_sec]",
    "        for (g = 1; g <= G; g++) {",
    "            if (ORT[r_sec, g] <= sinir) {",
    "                g_sec = g",
    "                break",
    "            }",
    "        }",
    "    }",
    "}",
    "",
    "void pen_sakla(string scalar ad, real scalar sabit, real colvector b, real colvector y, real colvector tahmin,",
    "               real colvector e, real scalar lam, real scalar oran, string rowvector adlar)",
    "{",
    "    real colvector artik",
    "    artik = y - tahmin",
    "    st_numscalar(ad + \"_test_mse\", (sum(e :== 0) > 0 ? mean(select(artik, e :== 0) :^ 2) : .))",
    "    st_numscalar(ad + \"_egitim_mse\", mean(select(artik, e) :^ 2))",
    "    st_numscalar(ad + \"_sifirdan\", sum(b :!= 0))",
    "    st_numscalar(ad + \"_norm\", sqrt(sum(b :^ 2)))",
    "    st_numscalar(ad + \"_lambda\", lam)",
    "    st_numscalar(ad + \"_l1\", oran)",
    "    st_matrix(ad + \"_b\", (sabit, b'))",
    "    st_matrixcolstripe(ad + \"_b\", (J(cols(adlar) + 1, 1, \"\"), (\"sabit\", adlar)'))",
    "}",
    "",
    "void pen_model(string scalar ad, string scalar yad, string scalar sayisal, string scalar kategorik,",
    "               string scalar ceza, real rowvector izgara, string scalar egitimad, string scalar katad,",
    "               string scalar kural, real rowvector oranlar, real scalar olcekle, real scalar yol)",
    "{",
    "    real colvector y, e, tahmin",
    "    real matrix S, Kt, XA, XB, ORT, SH, K, KY",
    "    string rowvector adlar",
    "    real scalar r_sec, g_sec, lam, oran, yo",
    "    real rowvector xo",
    "    y = st_data(., yad)",
    "    S = st_data(., tokens(sayisal))",
    "    Kt = (kategorik == \"\" ? J(rows(y), 0, .) : st_data(., tokens(kategorik)))",
    "    e = (egitimad == \"\" ? J(rows(y), 1, 1) : (st_data(., egitimad) :== 1))",
    "    oran = (ceza == \"enet\" ? oranlar[1] : 1)",
    "    lam = 0",
    "    if (ceza != \"ols\") {",
    "        pen_cv(select(y, e), select(S, e), pen_sec(Kt, e), select(st_data(., katad), e), izgara, ceza,",
    "               (ceza == \"enet\" ? oranlar : 1), kural, olcekle, r_sec, g_sec, ORT, SH)",
    "        lam = izgara[g_sec]",
    "        oran = (ceza == \"enet\" ? oranlar[r_sec] : 1)",
    "        st_matrix(ad + \"_cv\", (izgara', ORT[r_sec, .]', SH[r_sec, .]'))",
    "        st_matrixcolstripe(ad + \"_cv\", (J(3, 1, \"\"), (\"lambda\" \\ \"cv_ort\" \\ \"cv_sh\")))",
    "    }",
    "    pen_tasarim(select(S, e), S, pen_sec(Kt, e), Kt, olcekle, tokens(sayisal), tokens(kategorik), XA, XB, adlar)",
    "    tahmin = pen_tahmin(XA, select(y, e), XB, lam, ceza, oran, K)",
    "    xo = mean(XA)",
    "    yo = mean(select(y, e))",
    "    pen_sakla(ad, yo - xo * K, K, y, tahmin, e, lam, oran, adlar)",
    "    if (yol) {",
    "        KY = pen_yol(XA :- xo, select(y, e) :- yo, izgara, ceza, oran)",
    "        st_matrix(ad + \"_yol\", (izgara', KY'))",
    "        st_matrixcolstripe(ad + \"_yol\", (J(cols(adlar) + 1, 1, \"\"), (\"lambda\", adlar)'))",
    "    }",
    "    printf(\"%s: lambda = %g, test MSE = %9.4f, sıfırdan farklı katsayı = %g\\n\", ad, lam,",
    "           st_numscalar(ad + \"_test_mse\"), st_numscalar(ad + \"_sifirdan\"))",
    "}",
    "end",
]

POST_HELPER = [
    "* Post-Lasso: kaynak modelin sıfırdan farklı katsayılı terimleri üzerinde cezasız EKK (aynı örneklem ve ölçek)",
    "mata:",
    "void pen_post(string scalar ad, string scalar kaynak, string scalar yad, string scalar sayisal,",
    "              string scalar kategorik, string scalar egitimad, real scalar olcekle)",
    "{",
    "    real colvector y, e, sec, b, tahmin",
    "    real matrix S, Kt, XA, XB, B, K",
    "    string rowvector adlar",
    "    y = st_data(., yad)",
    "    S = st_data(., tokens(sayisal))",
    "    Kt = (kategorik == \"\" ? J(rows(y), 0, .) : st_data(., tokens(kategorik)))",
    "    e = (egitimad == \"\" ? J(rows(y), 1, 1) : (st_data(., egitimad) :== 1))",
    "    pen_tasarim(select(S, e), S, pen_sec(Kt, e), Kt, olcekle, tokens(sayisal), tokens(kategorik), XA, XB, adlar)",
    "    B = st_matrix(kaynak + \"_b\")",
    "    sec = selectindex(B[2..cols(B)] :!= 0)",
    "    tahmin = pen_tahmin(XA[., sec], select(y, e), XB[., sec], 0, \"ols\", 1, K)",
    "    b = J(cols(adlar), 1, 0)",
    "    b[sec] = K[., 1]",
    "    pen_sakla(ad, mean(select(y, e)) - mean(XA[., sec]) * K, b, y, tahmin, e, 0, 1, adlar)",
    "    printf(\"%s: test MSE = %9.4f, seçilen terim = %g\\n\", ad, st_numscalar(ad + \"_test_mse\"), cols(sec))",
    "}",
    "end",
]

DML_HELPER = [
    "* DML2: her dış kat k için m_Y(X) = E[Y|X] ve m_D(X) = E[D|X] k dışındaki gözlemlerde öğrenilir, k'de artık",
    "* alınır; θ̂ = Σv̂û/Σv̂² (sabitsiz). Dış kat değişkeni boşsa çapraz uyarlama yoktur (artıklaştırma). Lasso",
    "* cezası iç katlarla CV'de seçilir. SH: HC1 (n/(n − 1)) ya da küme-dayanıklı (skorlar küme içinde, G/(G − 1)).",
    "* Sonuç: <ad>_theta, <ad>_se, <ad>_n skalerleri; <ad>_b ve <ad>_V (ereturn post için); <ad>_katlar.",
    "capture mata: mata drop dml_capraz()",
    "capture mata: mata drop ortanca()",
    "mata:",
    "real scalar ortanca(real colvector x)",
    "{",
    "    real colvector s",
    "    real scalar m",
    "    s = sort(x, 1)",
    "    m = rows(s)",
    "    return(mod(m, 2) ? s[(m + 1) / 2] : (s[m / 2] + s[m / 2 + 1]) / 2)",
    "}",
    "",
    "void dml_capraz(string scalar ad, string scalar yad, string scalar dad, string scalar ozellik,",
    "                string scalar disad, string scalar icad, real rowvector izgara, string scalar kural,",
    "                string scalar kumead, string scalar ogrenici, string scalar uad, string scalar vad)",
    "{",
    "    real colvector Y, D, O, I, yh, dh, anahtar, deg, e, u, v, skor, C, idler, toplam, hedef, P",
    "    real matrix X, XA, XB, ORT, SH, katlar, K",
    "    string rowvector adlar",
    "    real scalar n, j, t, lam, r_sec, g_sec, teta, orta, g, G, sh",
    "    Y = st_data(., yad)",
    "    D = st_data(., dad)",
    "    X = st_data(., tokens(ozellik))",
    "    n = rows(Y)",
    "    O = (disad == \"\" ? J(n, 1, 1) : st_data(., disad))",
    "    I = (icad == \"\" ? J(n, 1, 1) : st_data(., icad))",
    "    anahtar = uniqrows(O)",
    "    yh = J(n, 1, 0)",
    "    dh = J(n, 1, 0)",
    "    katlar = J(rows(anahtar), 5, .)",
    "    for (j = 1; j <= rows(anahtar); j++) {",
    "        deg = O :== anahtar[j]",
    "        e = (disad == \"\" ? J(n, 1, 1) : (O :!= anahtar[j]))",
    "        katlar[j, 1] = anahtar[j]",
    "        for (t = 1; t <= 2; t++) {",
    "            hedef = (t == 1 ? Y : D)",
    "            lam = 0",
    "            if (ogrenici == \"lasso\") {",
    "                pen_cv(select(hedef, e), select(X, e), J(sum(e), 0, .), select(I, e), izgara, \"lasso\", 1, kural, 1,",
    "                       r_sec, g_sec, ORT, SH)",
    "                lam = izgara[g_sec]",
    "            }",
    "            pen_tasarim(select(X, e), select(X, deg), J(sum(e), 0, .), J(sum(deg), 0, .), 1, J(1, cols(X), \"\"),",
    "                        J(1, 0, \"\"), XA, XB, adlar)",
    "            P = pen_tahmin(XA, select(hedef, e), XB, lam, (ogrenici == \"lasso\" ? \"lasso\" : \"ols\"), 1, K)",
    "            if (t == 1) yh[selectindex(deg)] = P",
    "            else dh[selectindex(deg)] = P",
    "            katlar[j, 2 * t] = lam",
    "            katlar[j, 2 * t + 1] = sum(K[., 1] :!= 0)",
    "        }",
    "    }",
    "    u = Y - yh",
    "    v = D - dh",
    "    teta = cross(v, u) / cross(v, v)",
    "    skor = v :* (u - teta :* v)",
    "    if (kumead == \"\") {",
    "        orta = cross(skor, skor) * n / (n - 1)",
    "    }",
    "    else {",
    "        C = st_data(., kumead)",
    "        idler = uniqrows(C)",
    "        G = rows(idler)",
    "        toplam = J(G, 1, 0)",
    "        for (g = 1; g <= G; g++) toplam[g] = sum(select(skor, C :== idler[g]))",
    "        orta = cross(toplam, toplam) * G / (G - 1)",
    "    }",
    "    sh = sqrt(orta) / cross(v, v)",
    "    st_numscalar(ad + \"_theta\", teta)",
    "    st_numscalar(ad + \"_se\", sh)",
    "    st_numscalar(ad + \"_n\", n)",
    "    st_matrix(ad + \"_b\", teta)",
    "    st_matrixcolstripe(ad + \"_b\", (\"\", \"theta\"))",
    "    st_matrix(ad + \"_V\", sh^2)",
    "    st_matrixcolstripe(ad + \"_V\", (\"\", \"theta\"))",
    "    st_matrixrowstripe(ad + \"_V\", (\"\", \"theta\"))",
    "    st_matrix(ad + \"_katlar\", katlar)",
    "    st_matrixcolstripe(ad + \"_katlar\", (J(5, 1, \"\"), (\"kat\" \\ \"lambda_y\" \\ \"sifirdan_y\" \\ \"lambda_d\" \\ \"sifirdan_d\")))",
    "    st_numscalar(ad + \"_sifirdan_y_min\", min(katlar[., 3]))",
    "    st_numscalar(ad + \"_sifirdan_y_max\", max(katlar[., 3]))",
    "    st_numscalar(ad + \"_sifirdan_d_min\", min(katlar[., 5]))",
    "    st_numscalar(ad + \"_sifirdan_d_max\", max(katlar[., 5]))",
    "    if (uad != \"\") {",
    "        st_store(., st_addvar(\"double\", uad), u)",
    "        st_store(., st_addvar(\"double\", vad), v)",
    "    }",
    "}",
    "end",
    "* DML sonucunu Stata tahmin sonucu olarak saklar: _b[theta], _se[theta], e(N)",
    "capture program drop dml_kaydet",
    "program define dml_kaydet",
    "    args ad",
    "    ereturn post `ad'_b `ad'_V, obs(`=scalar(`ad'_n)')",
    "    estimates store `ad'",
    "end",
]

DOUBLE_SELECTION_HELPER = [
    "* S_Y: Y ve her X, [1, D] üzerinde artıklaştırıldıktan sonra (D cezalanmasın diye) Lasso; S_D: D'nin X",
    "* üzerine Lasso'su. Seçilen kontroller <ad>_sy, <ad>_sd ve <ad>_birlesim yerel makrolarına yazılır.",
    "capture mata: mata drop cift_secim()",
    "mata:",
    "real rowvector pen_secim(real colvector y, real matrix X, real colvector kat, real rowvector izgara,",
    "                         string scalar kural)",
    "{",
    "    real matrix ORT, SH, XA, XB, K",
    "    string rowvector adlar",
    "    real scalar r_sec, g_sec",
    "    pen_cv(y, X, J(rows(y), 0, .), kat, izgara, \"lasso\", 1, kural, 1, r_sec, g_sec, ORT, SH)",
    "    pen_tasarim(X, X, J(rows(y), 0, .), J(rows(y), 0, .), 1, J(1, cols(X), \"\"), J(1, 0, \"\"), XA, XB, adlar)",
    "    (void) pen_tahmin(XA, y, XB, izgara[g_sec], \"lasso\", 1, K)",
    "    return((K[., 1] :!= 0)')",
    "}",
    "",
    "void cift_secim(string scalar ad, string scalar yad, string scalar dad, string scalar kontrol,",
    "                string scalar katad, real rowvector izgara, string scalar kural, string scalar izle)",
    "{",
    "    real colvector Y, D, kat",
    "    real matrix X, Z, M",
    "    real rowvector sy, sd, birlesim, maske",
    "    string rowvector adlar, iz",
    "    real scalar i",
    "    Y = st_data(., yad)",
    "    D = st_data(., dad)",
    "    X = st_data(., tokens(kontrol))",
    "    kat = st_data(., katad)",
    "    adlar = tokens(kontrol)",
    "    Z = (J(rows(Y), 1, 1), D)",
    "    M = (X, Y)",
    "    M = M - Z * qrsolve(Z, M)",
    "    sy = pen_secim(M[., cols(M)], M[., 1..cols(X)], kat, izgara, kural)",
    "    sd = pen_secim(D, X, kat, izgara, kural)",
    "    birlesim = sy :| sd",
    "    st_local(ad + \"_sy\", invtokens(select(adlar, sy)))",
    "    st_local(ad + \"_sd\", invtokens(select(adlar, sd)))",
    "    st_local(ad + \"_birlesim\", invtokens(select(adlar, birlesim)))",
    "    iz = tokens(izle)",
    "    maske = J(1, cols(adlar), 0)",
    "    for (i = 1; i <= cols(iz); i++) maske = maske :| (adlar :== iz[i])",
    "    st_numscalar(ad + \"_ny\", sum(sy))",
    "    st_numscalar(ad + \"_nd\", sum(sd))",
    "    st_numscalar(ad + \"_n\", sum(birlesim))",
    "    st_numscalar(ad + \"_ny_iz\", sum(sy :& maske))",
    "    st_numscalar(ad + \"_nd_iz\", sum(sd :& maske))",
    "    st_numscalar(ad + \"_n_iz\", sum(birlesim :& maske))",
    "}",
    "end",
]

HELPERS = {
    "ceza": PENALIZED_HELPER,
    "post": POST_HELPER,
    "dml": DML_HELPER,
    "cift": DOUBLE_SELECTION_HELPER,
}


def _number(value: float) -> str:
    return E.format_number(value)


def _grid(spec) -> str:
    high, low, count = spec
    return f"ceza_izgarasi({_number(high)}, {_number(low)}, {int(count)})"


def list_definition(name: str, values) -> list[str]:
    lines: list[str] = []
    current = f"local {name}"
    for value in values:
        if len(current) + len(value) + 1 > 100:
            lines.append(current + " ///")
            current = "    "
        current = f"{current} {value}" if current.strip() else f"    {value}"
    lines.append(current)
    return lines


def _list(gen, values) -> tuple[list[str], str]:
    """Adlı liste (yerel makro) ya da satır içi değişken listesi."""

    lines, name = gen.use_list(values)
    if name is not None:
        return lines, f"`{name}'"
    return [], " ".join(values)


def operation(gen, op, command) -> list[str] | None:
    """Konu 11–12 işlemlerinin Stata kodu; tanımadığı işlemde ``None``."""

    if isinstance(op, RowNumber):
        return [f"* {op.comment}", f"generate double {op.name} = _n"]
    if isinstance(op, GroupRank):
        return [f"* {op.comment}", f"egen double {op.name} = group({op.source})"]
    if isinstance(op, Dictionary):
        return _dictionary(gen, op)
    if isinstance(op, DrawColumns):
        return _draw_columns(op)
    if isinstance(op, Penalized):
        return _penalized(gen, op, command)
    if isinstance(op, PostSelection):
        return _post_selection(gen, op, command)
    if isinstance(op, ModelMetrics):
        return _metrics(op)
    if isinstance(op, DotPlot):
        return _dot_plot(op)
    if isinstance(op, CVCurve):
        return _cv_curve(op)
    if isinstance(op, CoefPath):
        return _coef_path(gen, op)
    if isinstance(op, CrossFitDML):
        return _crossfit(gen, op, command)
    if isinstance(op, DMLSplits):
        return _splits(gen, op, command)
    if isinstance(op, DoubleSelection):
        return _double_selection(gen, op, command)
    if isinstance(op, EstimatePlot):
        return _estimate_plot(op)
    if isinstance(op, ComplexityCurve):
        return _complexity(op)
    if isinstance(op, LinePlot):
        return _line_plot(op)
    return None


def _dictionary(gen, op: Dictionary) -> list[str]:
    lines = [f"* {op.comment}"]
    if op.powers:
        lines += [
            f"foreach ad in {' '.join(op.powers)} {{",
            f"    forvalues kuvvet = 2/{op.degree} {{",
            "        generate double `ad'_`kuvvet' = `ad'^`kuvvet'",
            "    }",
            "}",
        ]
    if op.interactions:
        lines += [
            f"local taban {' '.join(op.base)}",
            "local k : word count `taban'",
            "forvalues i = 1/`=`k' - 1' {",
            "    forvalues j = `=`i' + 1'/`k' {",
            "        local a : word `i' of `taban'",
            "        local b : word `j' of `taban'",
            "        generate double `a'_x_`b' = `a' * `b'",
            "    }",
            "}",
        ]
    definition, name = gen.use_list(dictionary_terms(op))
    lines += definition
    if name is not None:
        lines.append(f"display \"Sözlükte `: word count `{name}'' terim\"")
    return lines


def _draw_columns(op: DrawColumns) -> list[str]:
    if op.rho == 0:
        return [
            f"* {op.comment}",
            f"forvalues j = 1/{op.count} {{",
            f"    generate double {op.prefix}`j' = rnormal()",
            "}",
        ]
    rho = _number(op.rho)
    return [
        f"* {op.comment}",
        f"* x1 = z1, xj = {rho}·x(j−1) + √(1 − {rho}²)·zj: Corr(xj, xk) = {rho}^|j − k|, her sütunun varyansı 1",
        f"generate double {op.prefix}1 = rnormal()",
        f"forvalues j = 2/{op.count} {{",
        f"    generate double {op.prefix}`j' = {rho} * {op.prefix}`=`j' - 1' + sqrt(1 - {rho}^2) * rnormal()",
        "}",
    ]


def _penalized(gen, op: Penalized, command) -> list[str]:
    lines, numeric = _list(gen, op.numeric)
    categorical = ""
    if op.categorical:
        extra, categorical = _list(gen, op.categorical)
        lines += extra
    grid = _grid(op.grid) if op.grid is not None else "0"
    ratios = "(" + ", ".join(_number(r) for r in op.l1_ratios) + ")" if op.penalty == "enet" else "1"
    if op.penalty == "ols":
        lines.append(f"* {op.name}: cezasız EKK" + (" (eğitim örnekleminde)" if op.sample else ""))
    else:
        high, low, count = op.grid
        rule = "en küçük CV" if op.rule == "min" else "bir standart hata kuralı"
        lines.append(f"* {SCALE_NOTES[op.penalty]}; ceza {count} noktalı ızgarada (10^{_number(high)} … "
                     f"10^{_number(low)}) CV ile, {rule}")
    call = (
        f'mata: pen_model("{op.name}", "{op.outcome}", "{numeric}", "{categorical}", "{op.penalty}", {grid}, '
        f'"{op.sample or ""}", "{op.folds or ""}", "{op.rule}", {ratios}, {int(op.standardize)}, {int(op.path)})'
    )
    return lines + [call]


def _post_selection(gen, op: PostSelection, command) -> list[str]:
    source: Penalized = gen.models[op.source]
    lines, numeric = _list(gen, source.numeric)
    categorical = ""
    if source.categorical:
        extra, categorical = _list(gen, source.categorical)
        lines += extra
    lines.append(f"* Post-Lasso: {op.source} modelinin seçtiği terimlerle cezasız EKK")
    call = (
        f'mata: pen_post("{op.name}", "{op.source}", "{source.outcome}", "{numeric}", "{categorical}", '
        f'"{source.sample or ""}", {int(source.standardize)})'
    )
    return lines + [call]


def _metrics(op: ModelMetrics) -> list[str]:
    models = " ".join(model for model, _ in op.rows)
    table = op.result
    lines = [
        "* Dış-örneklem karşılaştırması: test MSE, sıfırdan farklı katsayı, ‖β̂‖₂ ve seçilen ceza",
        f"matrix {table} = J({len(op.rows)}, 4, .)",
        f"matrix rownames {table} = {models}",
        f"matrix colnames {table} = test_mse sifirdan norm lambda",
        "local i = 0",
        f"foreach m in {models} {{",
        "    local ++i",
        f"    matrix {table}[`i', 1] = scalar(`m'_test_mse)",
        f"    matrix {table}[`i', 2] = scalar(`m'_sifirdan)",
        f"    matrix {table}[`i', 3] = scalar(`m'_norm)",
        f"    matrix {table}[`i', 4] = cond(scalar(`m'_lambda) > 0, scalar(`m'_lambda), .)",
        f"    scalar {table}_test_mse_`m' = scalar(`m'_test_mse)",
        f"    scalar {table}_sifirdan_`m' = scalar(`m'_sifirdan)",
        f"    scalar {table}_norm_`m' = scalar(`m'_norm)",
        f"    scalar {table}_lambda_`m' = scalar(`m'_lambda)",
        "}",
        f"matrix list {table}, format(%12.6g)",
    ]
    return lines


def _dot_plot(op: DotPlot) -> list[str]:
    color = SERIES_COLORS[0][1]
    count = len(op.labels)
    labels = " ".join(f'{count - index} "{label}"' for index, (_, label) in enumerate(op.labels))
    return [
        "* Yatay nokta grafiği; satırlar tablo sırasıyla yukarıdan aşağıya",
        "preserve",
        "clear",
        f"quietly svmat double {op.table}, names(col)",
        "generate double konum = _N - _n + 1",
        f"twoway (scatter konum {op.column}, msymbol(O) mcolor(\"{color}\") mlabel({op.column}) "
        f"mlabformat(%9.{op.decimals}f) mlabposition(3)), ///",
        f"       ylabel({labels}, angle(0) noticks) ytitle(\"\") yscale(range(0.5 {count + 0.5})) ///",
        f'       legend(order(1 "{op.x_label}")) xtitle("{op.x_label}") title("{op.title}", size(medium))',
        "restore",
    ]


def _cv_curve(op: CVCurve) -> list[str]:
    color, dark = SERIES_COLORS[0][1], DARK[1]
    model = op.model
    return [
        f"* CV eğrisi ve ±1 SH bandı ({model}); dikey çizgi seçilen ceza",
        "preserve",
        "clear",
        f"quietly svmat double {model}_cv, names(col)",
        "generate double cv_alt = cv_ort - cv_sh",
        "generate double cv_ust = cv_ort + cv_sh",
        'twoway (rarea cv_alt cv_ust lambda, fcolor("195 222 226") lcolor("195 222 226")) ///',
        f'       (line cv_ort lambda, lcolor("{color}") lwidth(medthick)), ///',
        f"       xscale(log) xline(`=scalar({model}_lambda)', lpattern(dash) lcolor(\"{dark}\")) ///",
        '       legend(order(2 "CV ortalama karesel hatası" 1 "±1 SH (katlar arası)")) ///',
        f"       note(\"Kesikli çizgi: seçilen λ = `: display %9.4g scalar({model}_lambda)'\") ///",
        f'       xtitle("{op.x_label}") ytitle("CV ortalama karesel hatası") title("{op.title}", size(medium))',
        "restore",
    ]


def _coef_path(gen, op: CoefPath) -> list[str]:
    color, gray, dark = SERIES_COLORS[0][1], GRAY[1], DARK[1]
    source: Penalized = gen.models[op.model]
    others = [name for name in source.numeric if name not in op.highlight]
    first_colored = len(others) + 1
    return [
        f"* Katsayı yolu ({op.model}); ceza büyükten küçüğe: değişkenler modele sırayla girer",
        "preserve",
        "clear",
        f"quietly svmat double {op.model}_yol, names(col)",
        'local gri ""',
        f"foreach v in {' '.join(others)} {{",
        f'    local gri `gri\' (line `v\' lambda, lcolor("{gray}") lwidth(vthin))',
        "}",
        'local renkli ""',
        f"foreach v in {' '.join(op.highlight)} {{",
        f'    local renkli `renkli\' (line `v\' lambda, lcolor("{color}") lwidth(medthick))',
        "}",
        "twoway `gri' `renkli', ///",
        f"       xscale(log reverse) xline(`=scalar({op.model}_lambda)', lpattern(dash) lcolor(\"{dark}\")) ///",
        f'       legend(order({first_colored} "{op.highlight_label}" 1 "{op.other_label}")) ///',
        f'       note("Kesikli çizgi: CV ile seçilen λ") xtitle("{op.x_label}") ytitle("Katsayı") ///',
        f'       title("{op.title}", size(medium))',
        "restore",
    ]


def _crossfit(gen, op: CrossFitDML, command) -> list[str]:
    lines, features = _list(gen, op.features)
    grid = _grid(op.grid) if op.grid is not None else "0"
    learner = "Lasso (ceza iç katlarla CV'de)" if op.learner == "lasso" else "cezasız EKK"
    if op.outer:
        lines.append(f"* DML2: yardımcı modeller {learner}; her dış katın artığı o katı görmemiş modelden")
    else:
        lines.append(f"* Artıklaştırma: yardımcı modeller {learner}, bütün örneklemde (çapraz uyarlama yok)")
    u, v = op.residuals if op.residuals is not None else ("", "")
    if op.residuals is not None:
        lines.append(f"capture drop {u} {v}")
    call = (
        f'mata: dml_capraz("{op.name}", "{op.outcome}", "{op.treatment}", "{features}", "{op.outer or ""}", '
        f'"{op.inner or ""}", {grid}, "{op.rule}", "{op.cluster or ""}", "{op.learner}", "{u}", "{v}")'
    )
    lines.append(call)
    lines.append(f"dml_kaydet {op.name}")
    se = "okul/küme SH" if op.cluster else "HC1 SH"
    lines.append(f'display "θ̂ = " %9.4f scalar({op.name}_theta) " ({se} " %9.4f scalar({op.name}_se) ")"')
    if op.outer and op.learner == "lasso":
        lines.append(f"matrix list {op.name}_katlar, format(%9.4g)")
    return lines


def _splits(gen, op: DMLSplits, command) -> list[str]:
    source: CrossFitDML = gen.models[op.dml]
    lines, features = _list(gen, source.features)
    grid = _grid(source.grid) if source.grid is not None else "0"
    keys = " ".join(key for key, _, _ in op.rules)
    multipliers = " ".join(_number(multiplier) for _, _, multiplier in op.rules)
    table = op.result
    call = (
        f'mata: dml_capraz("bolme_s", "{source.outcome}", "{source.treatment}", "{features}", "bolme_kat", '
        f'"{source.inner or ""}", {grid}, "{source.rule}", "{source.cluster or ""}", "{source.learner}", "", "")'
    )
    lines += [
        f"* Bölme duyarlılığı: dış katlar ⌊{op.folds}{{r·c}}⌋ + 1; aynı veri, öğrenici ve kat sayısı, yalnız okulların",
        "* katlara dağılımı değişir. Medyan birleştirme: θ̂_med = medyan θ̂_s, σ̂²_med = medyan{σ̂²_s + (θ̂_s − θ̂_med)²}",
        f"local carpanlar {multipliers}",
        f"matrix {table} = J({len(op.rules)}, 2, .)",
        f"matrix rownames {table} = {keys}",
        f"matrix colnames {table} = theta sh",
        "local s = 0",
        "foreach c of local carpanlar {",
        "    local ++s",
        "    capture drop bolme_kat",
        f"    quietly generate double bolme_kat = floor({op.folds} * ({op.rank} * `c' - floor({op.rank} * `c'))) + 1",
        f"    {call}",
        f"    matrix {table}[`s', 1] = scalar(bolme_s_theta)",
        f"    matrix {table}[`s', 2] = scalar(bolme_s_se)",
        "}",
        "drop bolme_kat",
        f"matrix list {table}, format(%9.3f)",
        f'mata: T = st_matrix("{table}"); m = ortanca(T[., 1]); st_numscalar("{op.name}_theta", m); '
        f'st_numscalar("{op.name}_se", sqrt(ortanca(T[., 2] :^ 2 + (T[., 1] :- m) :^ 2)))',
        f'mata: st_numscalar("{op.name}_min", min(T[., 1])); st_numscalar("{op.name}_max", max(T[., 1])); '
        f'st_numscalar("{op.name}_n", st_nobs())',
        f"matrix {op.name}_b = (scalar({op.name}_theta))",
        f"matrix colnames {op.name}_b = theta",
        f"matrix {op.name}_V = (scalar({op.name}_se)^2)",
        f"matrix colnames {op.name}_V = theta",
        f"matrix rownames {op.name}_V = theta",
        f"dml_kaydet {op.name}",
        f'display "Medyan θ̂ = " %6.3f scalar({op.name}_theta) " (SH " %6.3f scalar({op.name}_se) "); aralık " ///',
        f'    %6.3f scalar({op.name}_min) " – " %6.3f scalar({op.name}_max)',
    ]
    return lines


def _double_selection(gen, op: DoubleSelection, command) -> list[str]:
    lines, controls = _list(gen, op.controls)
    rule = "bir standart hata kuralı" if op.rule == "1se" else "en küçük CV"
    track = ""
    if op.track:
        lines.append(f"local {op.name}_izle {' '.join(op.track)}")
        track = f"`{op.name}_izle'"
    call = (
        f'mata: cift_secim("{op.name}", "{op.outcome}", "{op.treatment}", "{controls}", "{op.folds}", '
        f'{_grid(op.grid)}, "{op.rule}", "{track}")'
    )
    lines += [f"* Yalnız-sonuç Post-Lasso ve double selection; Lasso cezaları CV ile, {rule}", call]
    lines += command(f"quietly regress {op.outcome} {op.treatment} `{op.name}_sy', vce(robust)")
    lines.append(f"estimates store {op.name}_sonuc")
    lines += command(f"quietly regress {op.outcome} {op.treatment} `{op.name}_birlesim', vce(robust)")
    lines.append(f"estimates store {op.name}")
    lines.append(
        f'display "Seçilen kontroller: sonuç " scalar({op.name}_ny) ", tedavi " scalar({op.name}_nd) '
        f'", birleşim " scalar({op.name}_n)'
    )
    return lines


def _estimate_plot(op: EstimatePlot) -> list[str]:
    color, gray, red = SERIES_COLORS[0][1], GRAY[1], SERIES_COLORS[1][1]
    count = len(op.rows)
    table = op.result
    lines = [f"matrix {table} = J({count}, 4, .)", f"matrix colnames {table} = tahmin sh alt ust"]
    for index, (_, model, term) in enumerate(op.rows, start=1):
        key = "_cons" if term == E.INTERCEPT else term
        lines += [
            f"quietly estimates restore {model}",
            f"matrix {table}[{index}, 1] = _b[{key}]",
            f"matrix {table}[{index}, 2] = _se[{key}]",
            f"matrix {table}[{index}, 3] = _b[{key}] - {CI_MULTIPLIER}*_se[{key}]",
            f"matrix {table}[{index}, 4] = _b[{key}] + {CI_MULTIPLIER}*_se[{key}]",
        ]
    labels = " ".join(f'{count - index} "{label}"' for index, (label, _, _) in enumerate(op.rows))
    lines += [
        f"matrix list {table}, format(%9.3f)",
        "preserve",
        "clear",
        f"quietly svmat double {table}, names(col)",
        "generate double konum = _N - _n + 1",
    ]
    layers = [
        f'(rcap alt ust konum, horizontal lcolor("{color}") lwidth(medthick))',
        f'(scatter konum tahmin, msymbol(O) mcolor("{color}"))',
    ]
    legend = ['2 "Tahmin ve %95 güven aralığı"']
    options = []
    if op.splits:
        row = count - op.splits_row
        lines += [
            "local eski = _N",
            f"quietly set obs `=_N + rowsof({op.splits})'",
            "generate double bolme_theta = .",
            "generate double bolme_konum = .",
            f"forvalues s = 1/`=rowsof({op.splits})' {{",
            f"    quietly replace bolme_theta = {op.splits}[`s', 1] in `=`eski' + `s''",
            f"    quietly replace bolme_konum = {row} - 0.25 in `=`eski' + `s''",
            "}",
        ]
        layers.append(f'(scatter bolme_konum bolme_theta, msymbol(o) msize(small) mcolor("{gray}"))')
        legend.append('3 "Farklı kat kurallarıyla DML tahminleri"')
    if op.truth is not None:
        options.append(f'xline({_number(op.truth)}, lpattern(dash) lcolor("{red}"))')
        lines.append(f'local not note("Kesikli çizgi: {op.truth_label}")')
    else:
        lines.append('local not ""')
    lines += [
        f"twoway {' '.join(layers)}, ///",
        f"       ylabel({labels}, angle(0) labsize(small) noticks) ytitle(\"\") {' '.join(options)} ///",
        f"       yscale(range(0.5 {count + 0.5})) legend(order({' '.join(legend)})) `not' ///",
        f'       xtitle("{op.x_label}") title("{op.title}", size(medium))',
        "restore",
    ]
    return lines


def _complexity(op: ComplexityCurve) -> list[str]:
    table = op.result
    degree = op.max_degree
    return [
        "* Polinom derecesi 1, …, D: eğitim MSE (σ̂² = SSR/n), AIC = n + n·log(2πσ̂²) + 2K, BIC = n + n·log(2πσ̂²) + K·log n",
        "* (K = d + 2: katsayılar ve σ²), LOOCV (kaldıraçla kesin: [ê/(1 − h)]² ortalaması) ve test MSE",
        f"matrix {table} = J({degree}, 6, .)",
        f"matrix colnames {table} = derece egitim_mse loocv test_mse aic bic",
        f"forvalues d = 1/{degree} {{",
        "    capture drop kuvvet* p_tahmin p_kaldirac p_loo p_test",
        "    forvalues k = 1/`d' {",
        f"        quietly generate double kuvvet`k' = {op.x}^`k'",
        "    }",
        f"    quietly regress {op.y} kuvvet1-kuvvet`d' if {op.sample} == 1",
        "    quietly predict double p_tahmin",
        "    quietly predict double p_kaldirac if e(sample), hat",
        "    scalar p_n = e(N)",
        "    scalar p_s2 = e(rss) / p_n",
        f"    quietly generate double p_loo = (({op.y} - p_tahmin) / (1 - p_kaldirac))^2 if e(sample)",
        f"    quietly generate double p_test = ({op.y} - p_tahmin)^2 if {op.sample} == 0",
        "    quietly summarize p_loo",
        f"    matrix {table}[`d', 3] = r(mean)",
        "    quietly summarize p_test",
        f"    matrix {table}[`d', 4] = r(mean)",
        f"    matrix {table}[`d', 1] = `d'",
        f"    matrix {table}[`d', 2] = p_s2",
        f"    matrix {table}[`d', 5] = p_n + p_n*ln(2*_pi*p_s2) + 2*(`d' + 2)",
        f"    matrix {table}[`d', 6] = p_n + p_n*ln(2*_pi*p_s2) + (`d' + 2)*ln(p_n)",
        "}",
        "drop kuvvet* p_tahmin p_kaldirac p_loo p_test",
        f"matrix list {table}, format(%10.4f)",
        "* Her ölçütün en küçük olduğu derece",
        "local j = 1",
        f"foreach c in {' '.join(COMPLEXITY_COLUMNS_STATA)} {{",
        "    local ++j",
        f"    scalar {op.name}_d_`c' = 1",
        f"    forvalues d = 2/{degree} {{",
        f"        if el({table}, `d', `j') < el({table}, scalar({op.name}_d_`c'), `j') scalar {op.name}_d_`c' = `d'",
        "    }",
        "}",
        f'display "En küçük ölçütün derecesi — LOOCV: " scalar({op.name}_d_loocv) ", AIC: " scalar({op.name}_d_aic) ///',
        f'    ", BIC: " scalar({op.name}_d_bic) ", test: " scalar({op.name}_d_test_mse)',
    ]


COMPLEXITY_COLUMNS_STATA = ("egitim_mse", "loocv", "test_mse", "aic", "bic")
"""Stata tablosundaki sütun sırası (ilk sütun derece)."""
assert set(COMPLEXITY_COLUMNS_STATA) == set(COMPLEXITY_COLUMNS)


def _line_plot(op: LinePlot) -> list[str]:
    lines = [
        "preserve",
        "clear",
        f"quietly svmat double {op.table}, names(col)",
    ]
    if op.relative:
        lines += ["* Her ölçütten kendi en küçük değeri çıkarılır"]
        for column, _ in op.columns:
            lines += [f"quietly summarize {column}", f"quietly replace {column} = {column} - r(min)"]
    layers = []
    legend = []
    for index, ((column, label), (_, rgb)) in enumerate(zip(op.columns, SERIES_COLORS), start=1):
        layers.append(f'(connected {column} derece, lcolor("{rgb}") mcolor("{rgb}") msymbol(O) lwidth(medthick))')
        legend.append(f'{index} "{label}"')
    lines += [
        f"twoway {' '.join(layers)}, ///",
        f"       legend(order({' '.join(legend)})) xtitle(\"{op.x_label}\") ytitle(\"{op.y_label}\") ///",
        f'       title("{op.title}", size(medium))',
        "restore",
    ]
    return lines
