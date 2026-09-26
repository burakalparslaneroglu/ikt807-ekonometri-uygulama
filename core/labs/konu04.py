"""Konu 4 uygulama laboratuvarı: Card verisiyle ilk aşamadan 2SLS'ye.

Ders notları §4.17'nin (Adım 1–4 ve §4.17.5) birebir karşılığıdır. Veri: Hansen'in
Card1995 dosyası. Her ``Check`` notlarda basılı bir sayıdır. Kontroller ve deneyim
tanımı notların analiz betiğiyle aynıdır: deneyim = yaş − eğitim − 6.
"""

from __future__ import annotations

from core.labs import expr as E
from core.labs.spec import (
    IV,
    OLS,
    Check,
    CoefTarget,
    Derive,
    DropMissing,
    EffectTable,
    LabSpec,
    LabStep,
    LoadHansen,
    NoteRef,
    Predict,
    ReproClass,
    Scalar,
    ScalarTarget,
    StatTarget,
)

SECTION = "4.17"
FRAME = "card"
CONTROLS = (
    "exp76", "exp762_100", "black", "smsa76r", "reg76r", "smsa66r",
    *(f"reg66{i}" for i in range(2, 10)),
)


def _coef(model: str, term: str, expected: float, label: str, quantity: str = "coef", decimals: int = 4) -> Check:
    return Check(label, CoefTarget(model, term, quantity), expected, decimals)


def _percent(model: str) -> E.Expr:
    return E.mul(100, E.sub(E.exp(E.coef(model, "ed76")), 1))


STEPS = (
    LabStep(
        number=1,
        title="Yapısal denklem, ilk aşama ve indirgenmiş biçim",
        note=NoteRef(SECTION, 1),
        explanation=(
            "Soru: eğitimin ücret üzerindeki etkisi. Eğitim tercihi yetenek, aile geçmişi veya beklenen kazanç "
            "gibi gözlenmeyen etkenlerle ilişkili olabileceği için OLS nedensel getiriyi tanımlamayabilir. Araç: "
            "bireyin büyüdüğü yerde dört yıllık bir koleje yakınlık (`nearc4`). Üç denklem (notların §4.3 "
            "gösterimiyle):\n\n"
            "$$\\begin{aligned}"
            "\\text{Yapısal:}\\quad lwage_i &= \\beta\\, education_i + W_i'\\gamma + e_i\\\\"
            "\\text{İlk aşama:}\\quad education_i &= \\pi\\, nearc4_i + W_i'\\delta + v_i\\\\"
            "\\text{İndirgenmiş biçim:}\\quad lwage_i &= \\lambda\\, nearc4_i + W_i'\\rho + \\eta_i"
            "\\end{aligned}$$\n\n"
            "$W_i$: sabit terim ile deneyim, deneyimin karesi, ırk, metropolitan alan ve bölge göstergeleri. "
            "Analiz örneklemi bu değişkenlerin hepsi gözlenen bireylerdir."
        ),
        operations=(
            LoadHansen("card1995", "Card1995.dta", FRAME),
            Derive(FRAME, "exp76", E.sub(E.sub(E.var("age76"), E.var("ed76")), 6), "Potansiyel deneyim: yaş − eğitim − 6"),
            Derive(FRAME, "exp762_100", E.div(E.power(E.var("exp76"), 2), 100), "Deneyimin karesi / 100"),
            DropMissing(FRAME, ("lwage76", "ed76", "nearc4", *CONTROLS),
                        "Analiz örneklemi: ücret, eğitim, araç ve bütün kontrolleri gözlenen bireyler"),
        ),
        checks=(Check("Analiz örneklemi (N)", StatTarget(FRAME, "lwage76", "count"), 3010, decimals=0),),
        takeaway=(
            "Bu üç denklem ayrı raporlama nesneleridir. İyi bir IV makalesi yalnız ikinci aşama katsayısını "
            "gösterip ilk aşamayı gizlememelidir."
        ),
    ),
    LabStep(
        number=2,
        title="Sonuçları tek bir tanımlama zinciri olarak okumak",
        note=NoteRef(SECTION, 2, ("Tablo 4.1",)),
        explanation=(
            "Dört tahmin aynı kontrollerle ve HC1 standart hatasıyla: OLS, ilk aşama, indirgenmiş biçim ve 2SLS. "
            "İlk aşama gücü robust $t$ istatistiğinin karesiyle ($F=t^2$) özetlenir. Tek endojen değişken ve tek "
            "dışlanmış araçta $\\hat\\lambda/\\hat\\pi$ oranı (dolaylı en küçük kareler) 2SLS katsayısını verir."
        ),
        operations=(
            OLS("ols", FRAME, "lwage76", ("ed76", *CONTROLS), vcov="HC1"),
            OLS("ilk", FRAME, "ed76", ("nearc4", *CONTROLS), vcov="HC1"),
            OLS("indirgenmis", FRAME, "lwage76", ("nearc4", *CONTROLS), vcov="HC1"),
            IV("iv", FRAME, "lwage76", ("ed76",), ("nearc4",), CONTROLS),
            EffectTable(
                (
                    ("OLS eğitim katsayısı", "ols", "ed76"),
                    ("İlk aşama: nearc4 → eğitim", "ilk", "nearc4"),
                    ("İndirgenmiş biçim: nearc4 → log ücret", "indirgenmis", "nearc4"),
                    ("2SLS eğitim katsayısı", "iv", "ed76"),
                ),
                "tablo_41",
                title="Tablo 4.1: OLS, ilk aşama, indirgenmiş biçim ve 2SLS",
                se_label="HC1 SH",
            ),
            Scalar(
                "ilk_asama_F",
                E.power(E.div(E.coef("ilk", "nearc4"), E.se("ilk", "nearc4")), 2),
                "İlk aşama F (robust t²)",
                percent=False,
            ),
            Scalar(
                "oran",
                E.div(E.coef("indirgenmis", "nearc4"), E.coef("ilk", "nearc4")),
                "Oran λ̂/π̂ (dolaylı EKK)",
                percent=False,
                decimals=4,
            ),
            Scalar("ols_yuzde", _percent("ols"), "OLS kesin yüzde etki", decimals=1),
            Scalar("iv_yuzde", _percent("iv"), "2SLS kesin yüzde etki", decimals=1),
        ),
        checks=(
            _coef("ols", "ed76", 0.0747, "OLS eğitim katsayısı"),
            _coef("ols", "ed76", 0.0036, "OLS HC1 SH", "se"),
            _coef("ilk", "nearc4", 0.3199, "İlk aşama katsayısı"),
            _coef("ilk", "nearc4", 0.0851, "İlk aşama HC1 SH", "se"),
            _coef("indirgenmis", "nearc4", 0.0421, "İndirgenmiş biçim katsayısı"),
            _coef("indirgenmis", "nearc4", 0.0175, "İndirgenmiş biçim HC1 SH", "se"),
            _coef("iv", "ed76", 0.1315, "2SLS eğitim katsayısı"),
            _coef("iv", "ed76", 0.0541, "2SLS HC1 SH", "se"),
            Check("İlk aşama F", ScalarTarget("ilk_asama_F"), 14.14, decimals=2),
            Check("Oran λ̂/π̂", ScalarTarget("oran"), 0.1315),
            Check("OLS yüzde etki", ScalarTarget("ols_yuzde"), 7.8, decimals=1),
            Check("2SLS yüzde etki", ScalarTarget("iv_yuzde"), 14.1, decimals=1),
        ),
        reproducibility=ReproClass.CONVENTION,
        takeaway=(
            "İlk aşama: koleje yakınlık, diğer kontroller sabitken yaklaşık 0,32 yıl daha fazla eğitimle ilişkili; "
            "robust $F\\simeq14{,}14$. Bu, aracın eğitimle kayda değer ilişkisini gösterir, fakat \"$F>10$ ise araç "
            "kesinlikle güçlüdür\" evrensel bir kural değildir. Dolaylı en küçük kareler oranı "
            "$\\hat\\lambda/\\hat\\pi=0{,}04207/0{,}31990\\approx0{,}1315$, 2SLS katsayısının kendisidir; tablodaki "
            "yuvarlanmış değerlerle $0{,}0421/0{,}3199\\approx0{,}1316$ çıkar, fark yalnız yuvarlamadandır. "
            "OLS'nin kesin yüzde dönüşümü yaklaşık %7,8, 2SLS'ninki %14,1; ancak "
            "\"IV daha büyük, o halde OLS aşağı yanlı\" sonucu otomatik değildir: araç varsayımları, ölçüm hatası, "
            "heterojen eğitim getirileri ve IV'nin hangi alt popülasyonun yerel etkisini tanımladığı birlikte "
            "düşünülmelidir."
        ),
        code_note=(
            "2SLS standart hatası üç dilde aynı ayarla hesaplanır: Python `debiased=True`, R "
            "`vcovHC(type = \"HC1\")`, Stata `vce(robust) small`. Stata'da `small` yazılmazsa dayanıklı kovaryans "
            "n/(n−k) ile ölçeklenmez ve SH 0,0540 çıkar (notlar: 0,0541). Yazılım tablolarındaki p-değerleri "
            "t(n−k) dağılımıyla, uygulamadaki tablo normal yaklaşımla hesaplanır."
        ),
    ),
    LabStep(
        number=3,
        title="Araç geçerliliğini p-değerinden türetmemek",
        note=NoteRef(SECTION, 3),
        explanation=(
            "IV için iki ayrı soru vardır:\n\n"
            "1. **İlgililik (relevance):** `nearc4` eğitimle ilişkili mi? İlk aşama bunu inceler.\n"
            "2. **Dışsallık/dışlama:** `nearc4` ücreti eğitim dışında başka bir kanal üzerinden etkiliyor mu ve "
            "yapısal hata ile ortogonal mi?"
        ),
        takeaway=(
            "İlk soru veriden doğrudan araştırılabilir; ikinci soru büyük ölçüde ekonomik ve kurumsal argümana "
            "dayanır. İlk aşamanın güçlü olması dışlama kısıtını kanıtlamaz. Tam tanımlı modelde tek araç için "
            "aşırı tanımlama testi de yoktur."
        ),
    ),
    LabStep(
        number=4,
        title="2SLS'i yazılım komutunun arkasındaki iki projeksiyon olarak görmek",
        note=NoteRef(SECTION, 4),
        explanation=(
            "2SLS, endojen eğitim değişkeninin **araçların açıkladığı bileşenini** kullanır. Bunu elle görmek için "
            "ilk aşamanın uyum değerlerini ($\\widehat{education}$) alır, log ücreti bunlar ve aynı kontroller "
            "üzerine regres ederiz. Katsayı 2SLS ile birebir aynıdır; standart hata ise aynı değildir."
        ),
        operations=(
            Predict("ilk", FRAME, "ed76_hat", "fitted"),
            OLS("elle", FRAME, "lwage76", ("ed76_hat", *CONTROLS), vcov="HC1"),
            EffectTable(
                (
                    ("2SLS (yazılımın IV komutu)", "iv", "ed76"),
                    ("Elle ikinci aşama: tahmin edilen eğitim", "elle", "ed76_hat"),
                ),
                "iki_asama",
                title="Aynı katsayı, farklı standart hata",
                se_label="HC1 SH",
            ),
        ),
        checks=(_coef("elle", "ed76_hat", 0.1315, "Elle ikinci aşama katsayısı"),),
        reproducibility=ReproClass.CONVENTION,
        code_note=(
            "Tablodaki 2SLS satırı Adım 2'deki ayarlarla hesaplanır (Python `debiased=True`, R "
            "`vcovHC(type = \"HC1\")`, Stata `vce(robust) small`); elle ikinci aşama sıradan HC1 OLS'dir."
        ),
        takeaway=(
            "Elle ikinci aşamanın standart hatası 2SLS standart hatası değildir: $\\widehat{education}$ aynı "
            "örneklemden tahmin edilmiştir ve ikinci aşama artığı $Y-\\widehat X'\\hat\\beta$ yapısal hata için doğru "
            "artık değildir; doğru artık $Y-X'\\hat\\beta_{2SLS}$'dir (§4.8). Bu veride iki SH'nin farkı küçüktür "
            "(tabloda karşılaştırın); genel olarak fark büyük olabilir ve yönü önceden bilinemez. Çıkarım için "
            "2SLS rutininin IV-robust kovaryansı kullanılır. Bir ayrıntı: elle ikinci aşamanın p-değeri indirgenmiş "
            "biçiminkiyle aynıdır, çünkü tam tanımlı modelde tahmin edilen eğitim araç ve kontrollerin doğrusal bir "
            "fonksiyonudur; bu regresyon indirgenmiş biçimin yeniden ölçeklenmiş halidir."
        ),
    ),
    LabStep(
        number=5,
        title="Bir IV makalesindeki tabloyu nasıl okumalıyız?",
        note=NoteRef("4.17.5", 0),
        explanation=(
            "Bir IV sonuç tablosunda şunları arayın:\n\n"
            "1. OLS karşılaştırması,\n"
            "2. ilk aşama katsayıları ve zayıf araç tanıları,\n"
            "3. hangi değişkenlerin araç, hangilerinin dahil edilen dışsal kontrol olduğu,\n"
            "4. 2SLS katsayısı ve uygun robust/küme standart hatası,\n"
            "5. birden fazla araç varsa aşırı tanımlama sonuçları,\n"
            "6. IV estimand'ının hangi popülasyona ilişkin olduğunun tartışılması,\n"
            "7. aracın dışlama kısıtını destekleyen kurumsal gerekçe."
        ),
        takeaway=(
            "Tez yazımında kaçınılacak ifade: \"İçsellik testi anlamlı olduğu için IV modeli doğrudur\" veya "
            "\"ilk aşama F istatistiği 10'u geçtiği için araç geçerlidir\". IV tasarımının güvenilirliği, araç "
            "ilgililiği ile dışlama/dışsallık argümanının birlikte savunulmasına bağlıdır."
        ),
    ),
)


KONU04_LAB = LabSpec(
    topic_key="konu04",
    title="Card Verisiyle İlk Aşamadan 2SLS'ye",
    dataset="card1995",
    note_section=SECTION,
    steps=STEPS,
    labels=(
        ("(sabit)", "Sabit"),
        ("lwage76", "Log saatlik ücret (1976)"),
        ("ed76", "Eğitim yılı"),
        ("nearc4", "Dört yıllık koleje yakınlık"),
        ("exp76", "Deneyim"),
        ("exp762_100", "Deneyim²/100"),
        ("black", "Siyahi"),
        ("smsa76r", "Metropol (1976)"),
        ("reg76r", "Güney (1976)"),
        ("smsa66r", "Metropol (1966)"),
        ("ed76_hat", "Tahmin edilen eğitim"),
        ("ols", "OLS"),
        ("ilk", "İlk aşama"),
        ("indirgenmis", "İndirgenmiş biçim"),
        ("iv", "2SLS"),
        ("elle", "Elle ikinci aşama"),
    ),
)
