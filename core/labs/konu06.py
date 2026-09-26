"""Konu 6 uygulama laboratuvarı: sıfır yığılmasından sansürlü modele.

Ders notları §6.16'nın (Adım 1–4, §6.16.5–6) birebir karşılığıdır. Veri: Hansen'in CHJ2004
dosyası (Cox, Hansen ve Jimenez 2004, Filipinler kentsel haneleri). Hansen'in arşivindeki
dosya, ham ``urban.dta``'ya oluşturma dosyasının (``CHJ2004create.do``) uygulandığı hâlidir:
``income`` düzeltilmiş gelirdir, gelir dağılımının üst %2'si ve negatif gelirler çıkarılmıştır
(8.863 → 8.684 hane). Her ``Check`` notlarda basılı bir sayıdır.
"""

from __future__ import annotations

from core.hansen_data import CHJ2004_CONTROLS
from core.labs import expr as E
from core.labs.spec import (
    OLS,
    Check,
    Derive,
    LabSpec,
    LabStep,
    LoadHansen,
    NoteRef,
    ProfileCurves,
    QuantileRegression,
    ReproClass,
    StatTarget,
    Summaries,
    TableTarget,
    Tobit,
    TobitFitCheck,
    TobitTargets,
)

SECTION = "6.16"
FRAME = "chj"
KNOTS = (10, 20, 50, 100, 150)
SPLINES = tuple(f"spline{k}" for k in KNOTS)
REGRESSORS = ("income_k", *SPLINES, *CHJ2004_CONTROLS)
GRID = (0.0, 10.0, 20.0, 50.0, 100.0, 150.0)

# Tablo 6.4: seçilmiş gelir düzeylerinde tahmin edilen transferler (bin peso).
CURVES = {
    "ols": (16.62, 10.09, 7.00, 6.70, 7.37, 7.72),
    "tobit": (15.70, 9.24, 4.91, 3.37, 3.50, 3.60),
    "lad": (10.60, 3.50, 2.98, 2.63, 2.26, 2.24),
}
# Tablo 6.5: Tobit'in ima ettiği üç hedef, aynı profillerde.
TARGETS = {
    "p_poz": (0.787, 0.680, 0.598, 0.568, 0.570, 0.572),
    "gozlenen": (18.10, 13.35, 10.58, 9.68, 9.76, 9.81),
    "poz_ort": (23.00, 19.62, 17.68, 17.05, 17.10, 17.14),
}
_MODEL_LABELS = {"ols": "OLS", "tobit": "Tobit gizli ortalama", "lad": "LAD medyan"}
_TARGET_LABELS = {"p_poz": "P(Y>0|x)", "gozlenen": "m(x)", "poz_ort": "m#(x)"}


def _curve_checks() -> tuple[Check, ...]:
    return tuple(
        Check(f"{_MODEL_LABELS[model]}, gelir {int(income)}", TableTarget("egriler", income, model), value, 2)
        for model, values in CURVES.items()
        for income, value in zip(GRID, values)
    )


def _target_checks() -> tuple[Check, ...]:
    return tuple(
        Check(f"{_TARGET_LABELS[column]}, gelir {int(income)}", TableTarget("hedefler", income, column), value,
              3 if column == "p_poz" else 2)
        for column, values in TARGETS.items()
        for income, value in zip(GRID, values)
    )


STEPS = (
    LabStep(
        number=1,
        title="Hansen'in veri hazırlama zincirini yeniden üretmek",
        note=NoteRef(SECTION, 1),
        explanation=(
            "Ham veri 8.863 hane içerir. Hansen'in oluşturma dosyası önce net hükümet dışı transferi "
            "$transfers=tabroad+tdomestic+tinkind-tgifts$ olarak tanımlar, sonra gelirden transferleri ve emekli "
            "maaşını çıkarır ($income^{adj}=income-transfers-pension$), düzeltilmiş gelirin üst %2'sini ve negatif "
            "gelirleri dışlar: 8.684 hane kalır. **Hansen'in veri arşivindeki `CHJ2004.dta` bu adımların sonucudur**: "
            "`income` sütunu zaten düzeltilmiş gelirdir; düzeltme yeniden uygulanmaz.\n\n"
            "Sansürlü sonuç, yurt dışından, yurt içinden ve ayni alınan transferlerin toplamıdır: "
            "$Y=tabroad+tdomestic+tinkind\\ge0$. Modellerde transfer ve gelir bin peso ölçeğindedir."
        ),
        operations=(
            LoadHansen("chj2004", "CHJ2004.dta", FRAME),
            Derive(FRAME, "received", E.add(E.add(E.var("tabroad"), E.var("tdomestic")), E.var("tinkind")),
                   "Alınan hükümet dışı transfer: yurt dışı + yurt içi + ayni (peso)"),
            Derive(FRAME, "received_k", E.div(E.var("received"), 1000), "Transfer, bin peso"),
            Derive(FRAME, "income_k", E.div(E.var("income"), 1000),
                   "Gelir, bin peso (arşivdeki income zaten düzeltilmiş gelirdir)"),
            Derive(FRAME, "sifir", E.mul(100, E.compare("eq", E.var("received"), 0)),
                   "Transfer almayan hane göstergesi, yüzde ölçeğinde (ortalaması sıfır payıdır)"),
            Summaries(
                FRAME,
                (
                    ("Hane sayısı", "received", "count"),
                    ("Sıfır transfer payı (%)", "sifir", "mean"),
                    ("Ortalama transfer (peso)", "received", "mean"),
                    ("Medyan transfer (peso)", "received", "median"),
                ),
                "ozet",
            ),
        ),
        checks=(
            Check("Hane sayısı", StatTarget(FRAME, "received", "count"), 8684, decimals=0),
            Check("Sıfır transfer payı (%)", StatTarget(FRAME, "sifir", "mean"), 17.6, decimals=1),
            Check("Ortalama transfer (peso)", StatTarget(FRAME, "received", "mean"), 7709, decimals=0),
            Check("Medyan transfer (peso)", StatTarget(FRAME, "received", "median"), 1200, decimals=0),
        ),
        reproducibility=ReproClass.EXACT,
        takeaway=(
            "Hanelerin yaklaşık %17,6'sı hiç transfer almıyor. Ortalama transfer yaklaşık 7.709 peso, medyan yalnız "
            "1.200 peso: ortalama–medyan farkı dağılımın güçlü sağ çarpıklığını gösterir. Ham veriden analiz örneklemine "
            "geçiş yeniden üretilemezse, doğru yazılımla hesaplanmış bir Tobit katsayısı bile araştırmayı güvenilir kılmaz."
        ),
        code_note=(
            "Ham `urban.dta` ile çalışıyorsanız önce Hansen'in `CHJ2004create.do` adımlarını uygulayın; arşivdeki "
            "`CHJ2004.dta` üzerinde bu adımları tekrarlamak geliri ikinci kez düzeltir."
        ),
    ),
    LabStep(
        number=2,
        title="Gelir etkisini doğrusal olmak zorunda bırakmamak",
        note=NoteRef(SECTION, 2, ("Şekil 6.3", "Tablo 6.4")),
        explanation=(
            "Gelir fonksiyonu beş düğümlü doğrusal spline ile esnekleştirilir: düğümler 10, 20, 50, 100 ve 150 bin "
            "pesoda; her düğüm için $(gelir-k)_+=\\max\\{gelir-k,0\\}$ terimi eklenir. Yaş, eğitim kategorileri, medeni "
            "durum, cinsiyet, çocuklar, hane büyüklüğü ve istihdam durumuna ilişkin 15 kontrol vardır.\n\n"
            "Aynı spesifikasyon üç tahmin ediciyle: **OLS** gözlenen koşullu ortalamayı, **LAD** (medyan regresyonu) "
            "koşullu medyanı, **Tobit** ise normal-homoskedastik modelde gizli sonucun ortalamasını hedefler. Eğriler, "
            "kontroller örneklem ortalamasındayken $x'\\hat\\beta$'dır."
        ),
        operations=(
            *(Derive(FRAME, f"spline{k}", E.maximum(E.sub(E.var("income_k"), k), 0), f"Spline terimi (gelir − {k})₊")
              for k in KNOTS),
            OLS("ols", FRAME, "received_k", REGRESSORS, vcov="HC1"),
            QuantileRegression("lad", FRAME, "received_k", REGRESSORS, 0.5),
            Tobit("tobit", FRAME, "received_k", REGRESSORS, 0.0),
            ProfileCurves(
                models=(("ols", "OLS"), ("tobit", "Tobit (gizli ortalama)"), ("lad", "LAD (medyan)")),
                frame=FRAME,
                variable="income_k",
                derived=tuple((f"spline{k}", E.maximum(E.sub(E.var("income_k"), k), 0)) for k in KNOTS),
                values=GRID,
                result="egriler",
                plot_grid=(0.0, 200.0, 201),
                x_label="Toplam gelir (bin peso)",
                y_label="Tahmin edilen transfer (bin peso)",
                title="CHJ2004: gelir-transfer profili",
            ),
        ),
        checks=_curve_checks(),
        reproducibility=ReproClass.CONVENTION,
        takeaway=(
            "Düşük gelirde transferler gelirle birlikte hızla azalır; yaklaşık 20–50 bin peso sonrasında profil büyük "
            "ölçüde yataylaşır. Bu, özel transferlerin alıcı hanenin gelir kaybını kısmen telafi ettiği fikriyle "
            "uyumludur. Şekil üç tahmin edicide kalıcıdır, düzeyler ise belirgin biçimde farklıdır: tahmin hedefleri "
            "(ortalama, medyan, gizli ortalama) farklı olduğu için çizgilerin farklı düzeyde olması \"hangisi doğru\" "
            "sorusuna indirgenemez. LAD ve medyan regresyonu Konu 7'de ayrıntılı işlenir."
        ),
        code_note=(
            "OLS ve Tobit üç dilde aynı sayıyı verir (Tobit: Python'da kodda tanımlı MLE, R `AER::tobit`, Stata "
            "`tobit, ll(0)`). LAD doğrusal programlamanın kesin çözümüdür (Python `scipy.optimize.linprog`, R "
            "`quantreg::rq`, Stata `qreg`). Bu örneklemde çözüm tek değildir (R \"nonunique\" uyarısı verir): diller "
            "aynı amaç değerini veren farklı köşe çözümleri seçebilir; eğriler iki ondalıkta aynıdır. Kontrollerin "
            "ortalamaları örneklemin tamamından alınır."
        ),
    ),
    LabStep(
        number=3,
        title="Tobit katsayısını OLS katsayısı gibi okumamak",
        note=NoteRef(SECTION, 3, ("Tablo 6.5",)),
        explanation=(
            "Tobit'teki $\\beta_j$ gizli sonuç $Y^*$ denkleminin eğimidir. Aynı katsayı setinden üç farklı hedef "
            "türetilir ($z=x'\\beta/\\sigma$):\n\n"
            "1. gizli sonuç: $m^*(x)=x'\\beta$,\n"
            "2. sansürlenmeme olasılığı: $P(Y>0\\mid x)=\\Phi(z)$,\n"
            "3. gözlenen ortalama: $m(x)=\\Phi(z)\\,x'\\beta+\\sigma\\phi(z)$; pozitif gözlemlerde ise "
            "$m^{\\#}(x)=x'\\beta+\\sigma\\phi(z)/\\Phi(z)$.\n\n"
            "Aşağıda bu hedefler Tablo 6.4'teki profillerde hesaplanıyor. Son olarak model bir kontrolden geçiyor: "
            "Tobit'in ima ettiği $P(Y>0)$ ve $\\mathbb E[Y]$ örneklem ortalamaları verideki karşılıklarıyla "
            "karşılaştırılıyor."
        ),
        operations=(
            TobitTargets("tobit", "egriler", "hedefler"),
            TobitFitCheck("tobit", "kontrol"),
        ),
        checks=(
            *_target_checks(),
            Check("Model: ortalama P(Y>0)", TableTarget("kontrol", "p_poz", "model"), 0.579, 3),
            Check("Veri: pozitif transfer payı", TableTarget("kontrol", "p_poz", "veri"), 0.824, 3),
            Check("Model: ortalama E[Y] (bin peso)", TableTarget("kontrol", "ortalama", "model"), 11.46, 2),
            Check("Veri: ortalama transfer (bin peso)", TableTarget("kontrol", "ortalama", "veri"), 7.71, 2),
        ),
        reproducibility=ReproClass.CONVENTION,
        takeaway=(
            "Aynı Tobit tahmini, 150 bin peso gelirde gizli ortalama olarak 3,60, gözlenen ortalama olarak 9,81, pozitif "
            "transfer alanlarda 17,14 bin peso verir: \"Tobit katsayısı\" tek başına hangi hedefin yorumlandığını "
            "söylemez. Model kontrolü daha da öğreticidir: Tobit ortalamada P(Y>0) = 0,579 öngörürken verideki pozitif "
            "transfer payı 0,824; ima edilen ortalama 11,46, gözlenen 7,71 bin peso. Normal-homoskedastik Tobit, çok "
            "sayıda küçük pozitif transferi ve uzun sağ kuyruğu aynı anda yakalayamıyor. Hansen'in CLAD gibi dağılım "
            "varsayımına daha az bağlı yöntemleri tartışmasının nedeni budur."
        ),
    ),
    LabStep(
        number=4,
        title="Sıfırların ekonomik anlamını sorgulamak",
        note=NoteRef(SECTION, 4),
        explanation=(
            "Bu örnekte sıfır, transfer alınmadığını ifade eder ve klasik sansürlü model öğretimi için uygundur. Fakat "
            "birçok tezde sıfır iki aşamalı bir karar sürecinden doğabilir: bir firmanın Ar-Ge harcaması sıfırsa önce "
            "\"Ar-Ge yapma kararı\", ardından pozitif harcama miktarı ayrı mekanizmalarla belirlenebilir. Tek denklemli "
            "Tobit iki marjini aynı parametre setine zorladığı için böyle bir durumda aşırı kısıtlayıcı olabilir."
        ),
        takeaway=(
            "$Y$'de çok sayıda sıfır görmek tek başına Tobit seçme gerekçesi değildir. Sıfırın sansürleme, gerçek ekonomik "
            "köşe çözümü, iki aşamalı katılım kararı veya veri kaydı mekanizmasından hangisine karşılık geldiği "
            "açıklanmalıdır."
        ),
    ),
    LabStep(
        number=5,
        title="CLAD karşılaştırması ve tezinizde sansürlü sonuç protokolü",
        note=NoteRef("6.16.5", 0, ("§6.16.6",)),
        explanation=(
            "Hansen aynı örnekte sansürlü LAD (CLAD) tahminini de raporlar ve güçlü çarpıklık ile sansürleme nedeniyle "
            "CLAD'ı dayanıklı bir alternatif olarak tartışır. Amaç sayısal ayrıntı değil, model varsayımları "
            "değiştiğinde estimand'ın ve tahmin edicinin de değişebileceğini görmektir.\n\n"
            "**Protokol:**\n\n"
            "1. Sıfır veya sınır değerinin veri üretim mekanizmasını açıklayın.\n"
            "2. Sansürleme ile truncation ve örneklem seçimini birbirinden ayırın.\n"
            "3. Gizli sonuç yorumunun ekonomik olarak anlamlı olup olmadığını tartışın.\n"
            "4. Tobit kullanıyorsanız normalite ve homoskedastisite varsayımlarının güçlü olduğunu açıkça belirtin.\n"
            "5. Mümkünse OLS/LAD/Tobit veya iki parçalı alternatiflerle duyarlılık gösterin.\n"
            "6. Tobit katsayısı yerine hangi marjinal etkinin raporlandığını belirtin.\n"
            "7. Örneklem seçiminde dışlama değişkeninin neden seçim denklemini etkileyip sonuç denklemini doğrudan "
            "etkilemediğini savunun."
        ),
        takeaway=(
            "Adım 3'teki model kontrolü (ima edilen ile gözlenen sıfır payı ve ortalama) her Tobit uygulamasında "
            "raporlanabilecek basit bir tanıdır: büyük bir uyumsuzluk, normal-homoskedastik varsayımın veriyle "
            "çeliştiğinin işaretidir."
        ),
    ),
)


KONU06_LAB = LabSpec(
    topic_key="konu06",
    title="Sıfır Yığılmasından Sansürlü Modele",
    dataset="chj2004",
    note_section=SECTION,
    steps=STEPS,
    labels=(
        ("(sabit)", "Sabit"),
        ("received", "Alınan transfer (peso)"),
        ("received_k", "Alınan transfer (bin peso)"),
        ("income_k", "Gelir (bin peso)"),
        ("ols", "OLS"),
        ("lad", "LAD (medyan)"),
        ("tobit", "Tobit"),
        *((f"spline{k}", f"(gelir − {k})₊") for k in KNOTS),
    ),
)
