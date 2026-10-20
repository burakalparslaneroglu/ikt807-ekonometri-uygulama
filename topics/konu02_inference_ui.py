"""Small teaching panels; all reference calculations live in core/."""
from inspect import cleandoc

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.inference_konu02 import classical_f_test, normal_reference_curve, normal_test, vif_summary


def _md(text: str) -> None:
    st.markdown(cleandoc(text))


def _number(value: float, decimals: int = 4) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")


def _tails() -> None:
    _md(r"""**Notlar §2.9–2.9.1 · Aynı istatistik, farklı alternatif.**
    Burada $H_0$ sınırında standart normal referansı kullanıyoruz.
    $\Phi$ standart normal dağılım fonksiyonudur. Test yönü veriyi görmeden gerekçelendirilir; tahminin işaretine bakıp seçilmez.""")
    names = {"Çift taraflı": "two-sided", "Sağ taraflı": "greater", "Sol taraflı": "less"}
    direction = st.radio("Alternatifin yönü", tuple(names), horizontal=True, key="konu02_test_yonu")
    observed = st.slider("Gözlenen t istatistiği", -4.5, 4.5, 1.8, .1, key="konu02_test_t")
    alpha = st.selectbox("Anlamlılık düzeyi α", (.01, .05, .10), index=1,
                         format_func=lambda a: f"%{100*a:g}", key="konu02_test_alpha")
    result = normal_test(observed, names[direction], alpha)
    formulas = {
        "two-sided": r"H_1:\beta_j\ne\beta_{j,0},\qquad p=2[1-\Phi(|t_{obs}|)]",
        "greater": r"H_1:\beta_j>\beta_{j,0},\qquad p=1-\Phi(t_{obs})",
        "less": r"H_1:\beta_j<\beta_{j,0},\qquad p=\Phi(t_{obs})",
    }
    st.latex(formulas[result.alternative])
    x, density, shaded = normal_reference_curve(result)
    figure = go.Figure()
    figure.add_scatter(x=x, y=shaded, fill="tozeroy", line={"width":0},
                       fillcolor="rgba(22,139,155,.35)", name="p-değerinin alanı")
    figure.add_scatter(x=x, y=density, line={"color":"#16474c","width":2}, name="H₀ altında N(0,1)")
    for critical in result.critical_values:
        figure.add_vline(x=critical, line_dash="dash", line_color="#af4b36")
    figure.add_vline(x=result.observed, line_color="#202a30")
    figure.update_layout(height=330, margin={"t":20,"b":35,"l":40,"r":15},
                         xaxis_title="Test istatistiği T", yaxis_title="Sıfır hipotezi altında yoğunluk",
                         legend={"orientation":"h","y":1.15}, yaxis={"rangemode":"tozero"})
    st.plotly_chart(figure, width="stretch", key="konu02_p_alani")
    st.caption("Mavi alan p-değeridir; kesikli kırmızı çizgi α düzeyinin kritik sınırı, siyah çizgi gözlenen t'dir.")
    left, right = st.columns(2)
    left.metric("p-değeri", _number(result.p_value))
    right.metric("Karar (p < α)", "H₀ reddedilir" if result.reject else "H₀ reddedilemez")
    _md(r"""**p bir anlamlılık eşiğidir.** Aynı test ailesinde reddetme düzeylerinin
    alt sınırını verir. $p<\alpha$ kuralında $\alpha=p$ eşitliğinde reddedilmez;
    $p\le\alpha$ kuralında eşik dahildir. Karar yuvarlanmış sayıdan verilmez.
    Bu yorum, sonucu gördükten sonra $\alpha$ seçme gerekçesi değildir.

    $t=1{,}8$ için çift taraflı $p\approx0{,}0719$, sağ $p\approx0{,}0359$,
    sol $p\approx0{,}9641$. İşaret ters yöndeyse tek taraflı p büyük olur;
    çift taraflı p her zaman ikiye bölünmez. p, $H_0$'ın doğru olma olasılığı değildir.

    **Güven sınırı:** sağ test $[\widehat\beta_j-z_{1-\alpha}se,\infty)$,
    sol test $(-\infty,\widehat\beta_j+z_{1-\alpha}se]$ ile eşleşir.
    Normal-homoskedastik model ve klasik SH ile tam testte $\Phi$ yerine
    $t_{n-k}$ dağılım fonksiyonu kullanılır. HC/küme SH bu tam sonucu sağlamaz.""")


def _f_test() -> None:
    _md(r"""**Notlar §2.9.3 · Kısıtın uyuma maliyeti.** Aynı 100 gözlemde,
    sabit dahil $k=4$ katsayılı kısıtsız modelin $SSE_u=96$ olsun.
    $H_0:\beta_2=\beta_3=0$ için $q=2$ kısıt vardır. Aşağıda kısıtlı modelin
    varsayımsal $SSE_r$ değerini değiştiriyoruz; gerçek veri tahmini yapılmıyor.""")
    sse_r = st.slider("Kısıtlı modelin artık kareler toplamı SSE_r", 96.0, 144.0, 108.0, 1.0,
                      key="konu02_f_sse_r")
    result = classical_f_test(96, sse_r, n=100, k=4, q=2)
    st.latex(r"F=\frac{(SSE_r-SSE_u)/q}{SSE_u/(n-k)}"
             r"=\frac{(SSE_r-96)/2}{96/96}.")
    a, b, c = st.columns(3)
    a.metric("F istatistiği", _number(result.statistic, 2))
    b.metric("F₂,₉₆ sağ-kuyruk p", _number(result.p_value, 5))
    c.metric("%5 kritik değer", _number(result.critical_value, 3))
    st.info("Ortak hipotez reddedilir." if result.reject else "Ortak hipotez reddedilemez.")
    _md(r"""Pay kısıt başına uyum kaybı, payda kısıtsız hata varyansı tahminidir.
    Varsayılan örnekte $F=6$ ve $p\approx0{,}00350$: ortak ret, her iki katsayının
    ayrı ayrı anlamlı olduğunu söylemez. Paydanın serbestlik derecesi **kısıtsız**
    modelden gelir: $n-k=96$.

    Tam $F_{q,n-k}$ dağılımı için $\boldsymbol e\mid\boldsymbol X\sim
    N(0,\sigma^2\boldsymbol I_n)$, tam rank ve doğrusal kısıtlar gerekir. Genel homoskedastik
    büyük örneklem referansı $\chi_q^2/q$'dur. Sağ kuyruk pozitif katsayı testi
    değildir: $q=1$ ile $F=t^2$ çift taraflı klasik t testine eşdeğerdir.

    HC/küme kovaryansıyla kurulan Wald, SSE veya R² farkına indirgenmez.
    Dayanıklı $W/q$'nun “F” diye raporlanması klasik tam dağılımı geri getirmez.""")


def _trio() -> None:
    _md("""**Notlar §2.9.4 · Aynı kısıt, farklı hesaplama noktası.**
    Olabilirlik karşılaştırmasında modeller aynı örneklemde ve aynı olabilirlik
    tanımıyla tahmin edilir. Bu bölüm LR'ye giriş yapar; ayrıntısı sonra ele alınır.""")
    st.table(pd.DataFrame([
        ("Wald", "Yalnız kısıtsız", "Kısıttan uzaklık / tahmini kovaryans"),
        ("LM / skor", "Yalnız kısıtlı", "Tam modelin skoru ve bilgisi, kısıtlı tahminde"),
        ("LR", "Kısıtlı ve kısıtsız", "İki maksimum log-olabilirliğin farkı"),
    ], columns=["Test", "Gereken model tahmini", "Ölçtüğü sapma"]))
    st.latex(r"LR=2\{\ell_n(\widehat\theta_u)-\ell_n(\widetilde\theta_r)\}.")
    _md(r"""$\theta$ katsayıları ve yardımcı parametreleri; $\ell_n$ toplam
    log-olabilirliği, $\widehat\theta_u$ kısıtsız ve $\widetilde\theta_r$ kısıtlı
    en çok olabilirlik tahminini gösterir. LM'de **tam modelin** skoru kısıtlı
    tahminde değerlendirilir; kısıtsız optimumda skor sıfır olduğundan orada
    hesaplanan skor LM testi vermez.

    Doğru, düzgün, tanımlanmış ve iç noktada parametreli olabilirlik modelinde
    üçü de $H_0$ altında asimptotik $\chi_q^2$ referansına sahiptir; $q$ bağımsız
    kısıt sayısıdır. Sonlu örneklem istatistikleri aynı olmak zorunda değildir.
    Sınır parametrelerinde bu referans ayrıca sorgulanır.

    Wald tutarlı HC/küme kovaryansıyla olabilirlik olmadan da kurulabilir.
    Alışılmış LR ve model bilgisine dayalı LM, yanlış olabilirlik/varyans altında
    otomatik dayanıklı değildir; uygun skor ve kalibrasyon ayrıca gerekir.""")


def _heteroskedasticity() -> None:
    _md(r"""**Notlar §2.11 · Artık kareleri hangi değişkenlerle değişiyor?**
    Ortalama modeli doğru, $\mathbb E[e_i\mid X_i]=0$ olsun.
    $H_0:\operatorname{Var}(e_i\mid X_i)=\sigma^2$ sınanır.

    1. OLS artıklarını al: $\widehat e_i=Y_i-X_i'\widehat\beta$.
    2. Önceden seçilen $Z_i$ ile sabitli yardımcı regresyon kur:
       $\widehat e_i^2=\delta_0+Z_i'\delta+v_i$.
    3. $R^2_{aux}$ bu yardımcı regresyona aittir; asıl modelin R²'si kullanılmaz.
    4. $LM_h=nR^2_{aux}$ için $\chi^2_{q_h}$ sağ-kuyruk p-değerini kullan;
       $q_h$ sabit dışındaki bağımsız yardımcı sütun sayısıdır.""")
    st.table(pd.DataFrame([
        ("Koenker–Breusch–Pagan", "Önceden seçilen varyans değişkenleri Z"),
        ("White", "Regresörler + kareleri + çapraz çarpımları"),
    ], columns=["Test", "Yardımcı regresör seçimi"]))
    _md(r"""İki sürekli skaler regresör için White kümesi
    $(X_{1i},X_{2i},X_{1i}^2,X_{1i}X_{2i},X_{2i}^2)'$ ve ayrıca sabittir:
    bağımsız sütunlarla $q_h=5$. Kuklada $D_i^2=D_i$ yeni sütun/kısıt değildir.

    **Yeterli çerçeve:** i.i.d. gözlemler; $H_0$ altında $e_i$ ile $X_i$ bağımsız;
    sonlu dördüncü hata momenti, $\operatorname{Var}(e_i^2)>0$, yardımcı
    değişkenlerde sonlu ikinci moment ve yeterli rank. Koenker normallik istemez;
    buradaki dayanıklılık seri/küme bağımlılığını kapsamaz.

    **Laboratuvar Adım 4:** yaklaşık 175,16 istatistiği öğrencilendirilmiş Koenker
    sürümüdür: Python `het_breuschpagan(..., robust=True)` ve R
    `bptest(..., studentize=TRUE)`. 

    Reddedememek homoskedastisitenin kanıtı değildir. White her olası varyans
    desenini yakalayamaz; çok yardımcı terim küçük örneklemde güç/kalibrasyon
    sorunları yaratabilir. Yanlış ortalama biçimi de ret üretebilir.
    HC kullanmayı bu ön-testin anlamlı çıkmasına bağlamayın. White testi ile
    White/HC kovaryansı farklı işlemlerdir: biri tanılama, diğeri çıkarım.

    Uygulama ayrıntıları:
    [Koenker–BP](https://www.statsmodels.org/v0.14.6/generated/statsmodels.stats.diagnostic.het_breuschpagan.html),
    [White](https://www.statsmodels.org/v0.14.6/generated/statsmodels.stats.diagnostic.het_white.html).""")


def _vif() -> None:
    _md(r"""**Notlar §2.12.3 · Eşik yerine neyi ölçtüğümüzü okuyalım.**
    $R_j^2$, $X_j$'nin sabit ve diğer regresörler üzerindeki yardımcı regresyonuna
    aittir. VIF varyans, karekökü homoskedastik standart hata çarpanıdır.""")
    r_squared = st.slider("Yardımcı regresyon R²", 0.0, .99, .147, .001, format="%.3f",
                          key="konu02_vif_r2")
    vif, se_factor = vif_summary(r_squared)
    st.latex(r"VIF_j=\frac1{1-R_j^2},\qquad SH\text{ çarpanı}=\sqrt{VIF_j}.")
    a, b = st.columns(2)
    a.metric("VIF · varyans çarpanı", _number(vif, 2))
    b.metric("Homoskedastik SH çarpanı", _number(se_factor, 2))
    st.table(pd.DataFrame([
        ("1", "Doğrusal ilişki yok"), ("1 < VIF < 5", "Genellikle sınırlı/orta; bağlamı incele"),
        ("5 ≤ VIF < 10", "Belirgin bağlantı için inceleme işareti"),
        ("VIF ≥ 10", "Yaygın güçlü uyarı eşiği"),
    ], columns=["VIF", "Pratik yorum"]))
    _md(r"""Varsayılan $R_j^2\approx0{,}147$, notlardaki eğitim örneğidir:
    VIF yaklaşık 1,17 ve SH çarpanı yaklaşık 1,08. Hata varyansı ve $SST_j$
    aynı tutulunca varyans yüzde 17, homoskedastik SH yaklaşık yüzde 8 artar.

    **5 ve 10 pratik uyarı eşikleridir; hipotez testinin kritik değerleri veya
    modelin geçerlilik sınırları değildir.** Yüksek VIF nedeniyle gerekli kontrol
    otomatik çıkarılmaz. Polinom/etkileşimler doğal bağlantı yaratabilir.
    $R_j^2=1$ tam bağlantıdır; VIF sonlu değildir. HC altında aynı kesin varyans
    veya SH çarpanı yorumu otomatik geçerli değildir.""")


def render_inference_refresher() -> None:
    with st.expander("Hipotez testleri ve tanılama · Notlar §2.9–2.12", expanded=False):
        views = {"Test yönü ve p-değeri": _tails, "F testi": _f_test,
                 "Wald / LM / LR": _trio, "Heteroskedastisite testleri": _heteroskedasticity,
                 "VIF eşikleri": _vif}
        selected = st.selectbox("İncelenecek başlık", tuple(views), key="konu02_cikarim_baslik")
        views[selected]()
