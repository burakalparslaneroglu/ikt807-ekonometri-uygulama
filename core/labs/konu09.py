"""Konu 9 uygulama laboratuvarı: RDD sonuç tablosunu tasarım olarak okumak.

Ders notları §9.15'in (Adım 1–4 ve §9.15.5) birebir karşılığıdır. Adım 2 laboratuvarın Tablo 9.2'sini ve
aynı sayıları taşıyan §9.8'deki Tablo 9.1'i, Adım 3 §9.7'deki Hansen tahminlerini, Adım 4 Şekil 9.1'i yeniden
üretir. Veri: Hansen'in LM2007 dosyası (Ludwig ve Miller 2007), eşik değişkeni ve sonucu eksik olmayan 2.783
ilçe. Her ``Check`` notlarda basılı bir sayıdır.

Bant genişliği Hansen'in ölçeğindedir: çekirdekler birim varyanslıdır ve h çekirdeğin standart
sapmasıdır. Üçgen çekirdekte ağırlık eşikten h√6, dikdörtgen çekirdekte h√3 uzaklıkta sıfırlanır.
"""

from __future__ import annotations

from core.labs import expr as E
from core.labs.spec import (
    RDD,
    Check,
    CoefTarget,
    Derive,
    DropMissing,
    EffectTable,
    LabSpec,
    LabStep,
    LoadHansen,
    ModelTarget,
    NoteRef,
    Plot,
    RDDCurve,
    RDDTable,
    ReproClass,
    Scalar,
    ScalarTarget,
    StatTarget,
    Summaries,
    TableTarget,
    VLine,
)

SECTION = "9.15"
FRAME = "hs"
X = "povrate60"
Y = "mort_age59_related_posths"
CUTOFF = 59.1984
MAIN_H = 8.0

# Tablo 9.1 (§9.8) ve laboratuvar tablosu (§9.15.2): h → (n_h, τ̂, SH, alt %95, üst %95).
TABLE_91 = {
    4.0: (565, -2.10, 1.01, -4.08, -0.13),
    6.0: (794, -1.86, 0.83, -3.48, -0.24),
    8.0: (1041, -1.51, 0.71, -2.90, -0.11),
    10.0: (1325, -1.38, 0.62, -2.60, -0.15),
    12.0: (1649, -1.25, 0.57, -2.37, -0.12),
}
COLUMNS = (("n", "etkin örneklem n_h", 0), ("tahmin", "τ̂", 2), ("sh", "SH", 2), ("alt", "alt %95", 2),
           ("ust", "üst %95", 2))


def _model(h: float) -> str:
    return f"rdd_h{int(h)}"


def _table_checks() -> tuple[Check, ...]:
    return tuple(
        Check(f"h = {int(h)}: {label}", TableTarget("bant", h, column), value, decimals)
        for h, values in TABLE_91.items()
        for (column, label, decimals), value in zip(COLUMNS, values)
    )


STEPS = (
    LabStep(
        number=1,
        title="Katsayıdan önce tasarım satırlarını okuyun",
        note=NoteRef(SECTION, 1),
        explanation=(
            "Bir RDD tablosunda katsayıdan önce tasarımı tanımlayan satırlar okunur: eşik $c$; eşik değişkeninin "
            "(running variable) tanımı ve yönü; tasarımın keskin (sharp) mı bulanık (fuzzy) mı olduğu; çekirdek "
            "türü; yerel polinom derecesi; ana bant genişliği ve etkin örneklem büyüklüğü; standart hata ve yanlılık "
            "düzeltmesi yöntemi.\n\n"
            "Hansen'in LM2007 verisinde birim ABD ilçesidir. Eşik değişkeni $X$ 1960 yoksulluk oranı, eşik "
            "$c=59{,}1984$; sonuç 1973–1983 döneminde 5–9 yaş çocuklarda Head Start ile ilişkili nedenlerden ölüm "
            "oranıdır. Federal hükümet 1965'te en yoksul 300 ilçeye Head Start başvurusu hazırlama desteği vermiştir: "
            "tedavi göstergesi $D=\\mathbf 1\\{X\\ge c\\}$, tasarım keskindir. Analiz örneklemi eşik değişkeni ve "
            "sonucu eksik olmayan ilçelerdir (Hansen'in LM2007.dta dosyasında bu ilçeler zaten çıkarılmıştır)."
        ),
        operations=(
            LoadHansen("lm2007", "LM2007.dta", FRAME),
            DropMissing(FRAME, (X, Y), "Eşik değişkeni veya sonuç değişkeni eksik ilçeler"),
            Derive(FRAME, "esik_sagi", E.compare("ge", E.var(X), CUTOFF),
                   "Eşiğin sağı: D = 1{1960 yoksulluk oranı ≥ 59,1984}"),
            Summaries(
                FRAME,
                (
                    ("İlçe sayısı", X, "count"),
                    ("Eşiğin sağındaki ilçe sayısı", "esik_sagi", "sum"),
                    ("1960 yoksulluk oranı: en küçük", X, "min"),
                    ("1960 yoksulluk oranı: en büyük", X, "max"),
                ),
                "tasarim",
            ),
        ),
        checks=(
            Check("Analiz örneklemi (ilçe)", StatTarget(FRAME, X, "count"), 2783, decimals=0),
            Check("Eşiğin sağındaki ilçe", StatTarget(FRAME, "esik_sagi", "sum"), 294, decimals=0),
        ),
        reproducibility=ReproClass.EXACT,
        takeaway=(
            "Eşik $c=59{,}1984$; 2.783 ilçeden 294'ü eşiğin sağındadır. Eşiğin sağı, 1960 yoksulluk oranı en yüksek "
            "olan ve 1965'te başvuru desteği alan 300 ilçeden sonuç verisi eksik olmayanlardır. \"Eşiğin sağındaki "
            "katsayı negatiftir\" demek tek başına yeterli değildir: sağ tarafın hangi tedavi durumunu temsil ettiği "
            "kurumsal kuraldan okunur. Burada sağ taraf destek alan, yani tedavi edilen ilçelerdir."
        ),
    ),
    LabStep(
        number=2,
        title="Bant genişliği tablosunu sonuç avcılığı için değil duyarlılık için kullanmak",
        note=NoteRef(SECTION, 2, ("Tablo 9.2", "Tablo 9.1")),
        explanation=(
            "Yerel doğrusal RDD, pencere içindeki ilçelerde\n\n"
            "$$Y_i=\\alpha+\\tau D_i+\\beta_1R_i+\\beta_2D_iR_i+u_i,\\qquad R_i=X_i-c,$$\n\n"
            "regresyonunu çekirdek ağırlıklarıyla tahmin eder; $\\widehat\\tau$ eşikteki sıçramadır. Hansen "
            "çekirdekleri birim varyansa ölçekler: $h$ çekirdeğin standart sapmasıdır. Birim varyanslı üçgen çekirdek\n\n"
            "$$K(u)=\\frac{1}{\\sqrt6}\\Bigl(1-\\frac{|u|}{\\sqrt6}\\Bigr)\\mathbf 1\\{|u|\\le\\sqrt6\\}$$\n\n"
            "olup ilçe $i$'nin ağırlığı $K(R_i/h)$'dir: ağırlık eşikten $h\\sqrt6$ uzaklıkta sıfıra iner. $n_h$ pozitif "
            "ağırlık alan ilçe sayısıdır. Standart hata ağırlıklı regresyonun HC1 standart hatası, güven aralığı "
            "$\\widehat\\tau\\pm1{,}96\\cdot\\text{SH}$'dir. Aynı tahmin $h\\in\\{4,6,8,10,12\\}$ için tekrarlanır."
        ),
        operations=(
            *(RDD(_model(h), FRAME, X, Y, CUTOFF, h) for h in TABLE_91),
            RDDTable(
                tuple((h, _model(h)) for h in TABLE_91), "bant",
                "Bant genişliği h (Hansen ölçeği; üçgen çekirdekte pencere ±h√6)", "Eşikte tahmini sıçrama τ̂",
                "LM2007: bant genişliği duyarlılığı",
            ),
        ),
        checks=_table_checks(),
        reproducibility=ReproClass.CONVENTION,
        takeaway=(
            "Bant genişliği arttıkça etkin örneklem büyür (565'ten 1.649'a), standart hata düşer (1,01'den 0,57'ye) "
            "ve tahmin mutlak değerce küçülür (−2,10'dan −1,25'e). Tablo \"$h=4$ daha büyük etki verdiği için tercih "
            "edilir\" ya da \"$h=12$ en küçük standart hataya sahip olduğu için doğrudur\" diye okunmaz: her satır farklı "
            "bir yerellik–varyans dengesi kurar. Ana sonuç, makul bant aralığında işaretin negatif kalmasıdır; büyüklük "
            "yerellik kararına duyarlıdır. Tezde ana bant önceden tanımlı veya veri-temelli bir kuralla seçilir, tablo "
            "duyarlılık analizi olarak sunulur."
        ),
        code_note=(
            "Her bant genişliğinde aynı ağırlıklı regresyon: Python'da `statsmodels` `WLS(...).fit(cov_type=\"HC1\")`, "
            "R'de `lm(..., weights = w)` ve `sandwich::vcovHC(type = \"HC1\")`, Stata'da `regress ... [aw = w], "
            "vce(robust)`; üç yazılım aynı sayıları verir. Hazır RDD paketlerinin (ör. rdrobust) varsayılanları "
            "farklıdır: bant genişliği ölçeği, veri-temelli bant seçimi ve yanlılık düzeltmesi başka sayılar üretir."
        ),
    ),
    LabStep(
        number=3,
        title="Hansen'in tahminini yeniden üretmek: bant genişliği ölçeği",
        note=NoteRef(SECTION, 3, ("§9.7",)),
        explanation=(
            "Hansen'in kitap uygulamasında tercih edilen yerel tahmin $-1{,}51$ (SH 0,71) değeridir; Adım 2'deki "
            "$h=8$ satırı bu sayıyı yeniden üretir. Eşikteki iki limit tahmini aynı regresyondan okunur: solda "
            "$\\widehat\\alpha$, sağda $\\widehat\\alpha+\\widehat\\tau$. Aynı bant genişliğinin dikdörtgen çekirdek "
            "karşılığı $\\pm8\\sqrt3$ puanlık penceredir; bu pencerede $D$, $R$ ve $D\\cdot R$ üzerine basit EKK "
            "Hansen'in raporladığı ikinci tahmini verir.\n\n"
            "Aynı \"$h=8$\" pencerenin yarı genişliği olarak okunursa (ölçeklenmemiş üçgen çekirdek) pencere $\\pm8$'e "
            "daralır. Bu bir yöntem farkı değil ölçek farkıdır: yazılımlar ve makaleler bant genişliğini farklı "
            "ölçeklerde raporlayabilir."
        ),
        operations=(
            Scalar("sol", E.coef(_model(MAIN_H), E.INTERCEPT), "Eşiğin solu (h = 8)", percent=False),
            Scalar("sag", E.add(E.coef(_model(MAIN_H), E.INTERCEPT), E.coef(_model(MAIN_H), "D")),
                   "Eşiğin sağı (h = 8)", percent=False),
            Scalar("pencere_ucgen", E.mul(MAIN_H, E.sqrt(6)), "Üçgen pencere 8√6", percent=False, decimals=1),
            Scalar("pencere_dikdortgen", E.mul(MAIN_H, E.sqrt(3)), "Dikdörtgen pencere 8√3", percent=False),
            RDD("rdd_dikdortgen", FRAME, X, Y, CUTOFF, MAIN_H, kernel="rectangular"),
            RDD("rdd_pencere8", FRAME, X, Y, CUTOFF, MAIN_H, scale="window"),
            EffectTable(
                (
                    ("Üçgen, h = 8 (±19,6)", _model(MAIN_H), "D"),
                    ("Dikdörtgen, h = 8 (±13,86)", "rdd_dikdortgen", "D"),
                    ("Üçgen, pencere ±8", "rdd_pencere8", "D"),
                ),
                "olcek", title="Aynı \"h = 8\", üç pencere: eşikteki sıçrama", se_label="SH (HC1)",
            ),
        ),
        checks=(
            Check("Eşiğin solunda tahmini ölüm oranı (h = 8)", ScalarTarget("sol"), 3.31, decimals=2),
            Check("Eşiğin sağında tahmini ölüm oranı (h = 8)", ScalarTarget("sag"), 1.80, decimals=2),
            Check("Üçgen çekirdek penceresi 8√6", ScalarTarget("pencere_ucgen"), 19.6, decimals=1),
            Check("Dikdörtgen çekirdek penceresi 8√3", ScalarTarget("pencere_dikdortgen"), 13.86, decimals=2),
            Check("Dikdörtgen çekirdek: τ̂", CoefTarget("rdd_dikdortgen", "D"), -1.55, decimals=2),
            Check("Dikdörtgen çekirdek: SH", CoefTarget("rdd_dikdortgen", "D", "se"), 0.74, decimals=2),
            Check("Dikdörtgen çekirdek: ilçe sayısı", ModelTarget("rdd_dikdortgen", "nobs"), 757, decimals=0),
            Check("Pencere ±8: τ̂", CoefTarget("rdd_pencere8", "D"), -2.25, decimals=2),
            Check("Pencere ±8: SH", CoefTarget("rdd_pencere8", "D", "se"), 1.08, decimals=2),
            Check("Pencere ±8: ilçe sayısı", ModelTarget("rdd_pencere8", "nobs"), 482, decimals=0),
        ),
        reproducibility=ReproClass.CONVENTION,
        takeaway=(
            "Birim varyanslı üçgen çekirdekle $h=8$ eşiğe 19,6 puandan yakın 1.041 ilçeye pozitif ağırlık verir; sıçrama "
            "−1,51 (0,71), eşiğin solunda 3,31, sağında 1,80. Dikdörtgen çekirdek aynı yerellikte (±13,86; 757 ilçe) "
            "−1,55 (0,74) verir: bant genişliği aynı ölçekte tanımlandığında çekirdek seçimi ikinci derecededir. Pencere "
            "±8'e daraltılırsa 482 ilçe kalır ve tahmin −2,25 (1,08) olur. Replikasyonda sayılar tutmadığında önce "
            "tanım farkları aranır: analiz örneklemi, eşik değişkeninin merkezlenmesi, eşik, çekirdek, bant genişliğinin "
            "ölçeği, yerel polinom derecesi, kovaryatlar ve standart hata/yanlılık düzeltmesi prosedürü."
        ),
    ),
    LabStep(
        number=4,
        title="RDD grafiğini \"güzel sıçrama\" diye değil belirsizlikle birlikte okumak",
        note=NoteRef(SECTION, 4, ("Şekil 9.1",)),
        explanation=(
            "Ana RDD grafiğinde eşiğin iki yanında ayrı yerel doğrusal eğriler ve güven bantları gösterilir. Her "
            "$x_0$ noktasında yalnız o taraftaki ilçelerle üçgen çekirdekli ($h=8$, pencere $\\pm8\\sqrt6$) ağırlıklı "
            "EKK'nin sabit terimi hesaplanır; bant noktasal %95 güven aralığıdır (HC1). Görsel değerlendirmede üç şey "
            "birlikte aranır:\n\n"
            "1. eşikte koşullu ortalamada sıçrama,\n"
            "2. eşik çevresinde yeterli veri desteği,\n"
            "3. güven bantlarının sıçrama büyüklüğüne kıyasla genişliği."
        ),
        operations=(
            Plot(
                FRAME, X,
                (
                    RDDCurve(Y, CUTOFF, MAIN_H, "Yerel doğrusal tahmin (h = 8) ve %95 güven bandı"),
                    VLine(CUTOFF, "Eşik c = 59,1984"),
                ),
                "1960 yoksulluk oranı (eşik değişkeni)", "5–9 yaş Head Start ilişkili ölüm oranı",
                "LM2007: eşiğin iki yanında yerel doğrusal tahmin", x_range=(15.0, 82.0),
            ),
        ),
        reproducibility=ReproClass.EXACT,
        takeaway=(
            "Eşiğin sağında tahmini koşullu ortalama daha düşüktür: negatif bir yerel etkiyle uyumlu. Eğrilerin eşikteki "
            "değerleri Adım 3'teki 3,31 ve 1,80'dir. Bantlar eşikte genişler, çünkü sınırda gözlemler tek taraftadır. "
            "Belirgin görünen bir sıçrama, bant genişliği ve standart hata dikkate alınmadığında aşırı ikna edici "
            "görünebilir; tersine, nokta bulutunun gürültülü olması geçerli yerel tahminin bilgi taşımadığı anlamına "
            "gelmez. Hansen kutulanmış ortalamalar (binned means) yerine yerel tahmini ve belirsizliğini çizmeyi önerir."
        ),
        code_note=(
            "Eğriler her tarafta 120 noktada hesaplanır; her nokta ayrı bir ağırlıklı EKK'dir. Python'da NumPy ile "
            "vektörel, R'de `sapply`, Stata'da Mata ile."
        ),
    ),
    LabStep(
        number=5,
        title="Bir RDD makalesini değerlendirmek için kısa kontrol listesi",
        note=NoteRef("9.15.5", 0),
        explanation=(
            "1. Eşik kuralı kurumsal olarak açık mı?\n"
            "2. Birimler eşik değişkenini hassas biçimde manipüle edebilir mi?\n"
            "3. Tedavi olasılığı eşikte gerçekten değişiyor mu?\n"
            "4. Önceden belirlenmiş kovaryatlarda ve plasebo sonuçlarda sıçrama var mı?\n"
            "5. Ana bant genişliği nasıl seçildi?\n"
            "6. Alternatif makul bant genişliklerindeki sonuçlar gösterildi mi?\n"
            "7. Global yüksek dereceli polinom yerine yerel yöntem kullanılmış mı?\n"
            "8. Tahmin eşik çevresindeki birimler için yerel etki olarak mı yorumlanıyor?"
        ),
        takeaway=(
            "Makale dili için örnek: \"Yerel doğrusal RDD tahminleri, eşik civarındaki ilçelerde program desteği ile "
            "ölüm oranı arasında negatif bir yerel etki bulunduğunu göstermektedir. Tahminin büyüklüğü bant genişliğine "
            "duyarlı olmakla birlikte işaret makul bant aralığında korunmaktadır. Sonuçlar eşikten uzak ilçelere "
            "otomatik olarak genellenmemektedir.\""
        ),
    ),
)


KONU09_LAB = LabSpec(
    topic_key="konu09",
    title="RDD Sonuç Tablosunu Tasarım Olarak Okumak",
    dataset="lm2007",
    note_section=SECTION,
    steps=STEPS,
    labels=(
        (E.INTERCEPT, "Sabit (eşiğin solunda limit)"),
        (X, "1960 yoksulluk oranı"),
        (Y, "Head Start ilişkili ölüm oranı (5–9 yaş)"),
        ("esik_sagi", "Eşiğin sağı (D)"),
        ("D", "Eşikteki sıçrama (D)"),
        ("R", "R = X − c"),
        ("DR", "D·R"),
        *((_model(h), f"Üçgen çekirdek, h = {int(h)}") for h in TABLE_91),
        ("rdd_dikdortgen", "Dikdörtgen çekirdek, h = 8"),
        ("rdd_pencere8", "Üçgen çekirdek, pencere ±8"),
    ),
    consistency_notes=(
        "Notların bant genişliği tablosu (Tablo 9.1) ve Head Start şekli bir betikten üretilmemişti; h'yi pencerenin "
        "yarı genişliği olarak kullanıyordu ve Hansen'in −1,51 (0,71) tahminini vermiyordu. Hansen'in birim varyanslı "
        "çekirdek ölçeği (üçgende pencere ±h√6) ile −1,51 (0,71) yeniden üretildi; notlar, sunum ve uygulama bu ölçeğe "
        "göre güncellendi.",
    ),
)
