"""Konu 02: güvenilir çıkarım — dayanıklı ve küme standart hataları, FWL, delta yöntemi.

Uygulama sekmesi ders notları §2.13'ü gerçek Hansen verisiyle adım adım yeniden
üretir. Sezgi sekmesi bilinen bir veri üretim süreciyle dört kontrollü deney sunar.
Kendini sına sekmesi haftanın kavramlarını dört soru türüyle sınar.
"""

from __future__ import annotations

from inspect import cleandoc

import streamlit as st

from core.labs.registry import get_lab
from core.labs.sezgi_konu02 import KONU02_EXPERIMENTS
from core.quiz.registry import get_quiz
from topics.konu02_inference_ui import render_inference_refresher
from topics.lab_ui import render_lab
from topics.quiz_ui import render_quiz
from topics.shared import render_topic_header, topic_tabs
from topics.sim_ui import render_experiments

TOPIC_KEY = "konu02"


def _render_ols_properties() -> None:
    """Notes §2.5: distinguish the three questions before discussing inference."""
    with st.expander("OLS'nin örnekleme özellikleri · Notlar §2.5", expanded=False):
        st.markdown(cleandoc(
            r"""Aynı veri üretim sürecinden farklı örneklemler çektiğimizde
            $\widehat\beta$ değişir. Önce **sapmasızlık (yansızlık)**, sonra
            **tutarlılık**, ardından **asimptotik dağılım** sorusunu ele alalım.

            Model $Y_i=X_i'\beta+e_i$: $Y_i,e_i$ skaler;
            $X_i,\beta$ ise $k\times1$ sütun vektörleridir. Kalın $\boldsymbol X$
            bütün örneklemin $n\times k$ tasarım matrisi, $\boldsymbol e$ hata
            vektörüdür. Temel çerçeve: i.i.d. gözlemler, sabit $k$, sonlu
            ikinci momentler ve pozitif tanımlı $\boldsymbol Q_{XX}=\mathbb E[X_iX_i']$.

            **2.5.1 · Sonlu örneklemde sapmasızlık.** Aynı büyüklükteki
            örneklemlerden hesaplanan tahminlerin ortalaması doğru merkezde mi?
            Ek olarak $\mathbb E[e_i\mid X_i]=0$ ve $\boldsymbol X$'in olasılık birle
            tam sütun ranklı olduğunu varsayalım. Bağımsız örnekleme sayesinde
            $\mathbb E[\boldsymbol e\mid\boldsymbol X]=0$; dolayısıyla:"""
        ))
        st.latex(
            r"\mathbb E[\widehat\beta-\beta\mid\boldsymbol X]"
            r"=(\boldsymbol X'\boldsymbol X)^{-1}\boldsymbol X'\mathbb E[\boldsymbol e\mid\boldsymbol X]=0."
        )
        st.markdown(cleandoc(
            r"""Bu **koşullu sapmasızlıktır**. Koşulsuz $\mathbb E[\widehat\beta]=\beta$
            için ayrıca $\mathbb E\|\widehat\beta\|<\infty$ gerekir. Tam rank tek başına
            bu momentin varlığını garanti etmez. Sapmasızlık, tek tek tahminlerin
            doğru veya birbirine yakın olması demek değildir.

            **2.5.2 · Büyük örneklemde tutarlılık.** Örneklem büyüdükçe sabit
            bir toleransın dışına düşme olasılığı sıfıra gider mi? Burada daha
            zayıf $\mathbb E[X_ie_i]=0$ yeterlidir. Büyük sayılar yasası ve ters almanın
            limitte sürekliliği ile:"""
        ))
        st.latex(
            r"\widehat\beta-\beta="
            r"\left(\frac1n\sum_{i=1}^nX_iX_i'\right)^{-1}"
            r"\left(\frac1n\sum_{i=1}^nX_ie_i\right)"
            r"\xrightarrow{p}\boldsymbol Q_{XX}^{-1}0=0."
        )
        st.markdown(cleandoc(
            r"""Pozitif tanımlı popülasyon matrisi, örneklem matrisinin tersinin
            var olma olasılığını bire yaklaştırır; her sonlu örneklemde tam rank
            garantisi vermez. Tekil örneklemlerde tahmin ediciye sabit bir değer
            atanması olasılık limitini değiştirmez. Hedef doğrusal projeksiyon
            katsayısıdır; nedensel yorum için ayrıca tanımlama gerekir.

            **2.5.3 · Asimptotik dağılım.** Tahmin hatası hangi hızda küçülür,
            standart hatanın dayanağı nedir? Hansen'in Varsayım 7.2'sini izleyerek
            $\mathbb E[|Y_i|^4]<\infty$ ve $\mathbb E[\|X_i\|^4]<\infty$ ekleyelim. Bunlar
            $\boldsymbol\Omega=\mathbb E[X_iX_i'e_i^2]$ matrisinin sonlu olması için
            yeterlidir. Merkezi limit teoremi ve Slutsky ile:"""
        ))
        st.latex(
            r"\sqrt n(\widehat\beta-\beta)\xrightarrow{d}N(0,\boldsymbol V_\beta),"
            r"\qquad\boldsymbol V_\beta=\boldsymbol Q_{XX}^{-1}"
            r"\boldsymbol\Omega\boldsymbol Q_{XX}^{-1}."
        )
        st.markdown(cleandoc(
            r"""$\boldsymbol V_\beta$ ölçeklenmiş hatanın limit kovaryansıdır;
            $\widehat\beta$ için büyük örneklem kovaryans yaklaşımı
            $\boldsymbol V_\beta/n$ ölçeğindedir. Dördüncü momentler yeterli bir
            koşuldur; i.i.d. skor $X_ie_i$ için merkezi limit teoremi sonlu
            ikinci momentle uygulanabilir.

            **Üç ayrı özellik:** $\mathbb E[e\mid X]=0$, $\mathbb E[Xe]=0$ sonucunu verir;
            tersi genel olarak doğru değildir. Tutarlı bir tahmin edici küçük
            örneklemde sapmalı olabilir. Sapmasızlık da tek başına tutarlılık
            garantisi vermez. Hata normalliği ve homoskedastisite bu sonuçlar
            için gerekli değildir.

            Kaynak sırası: ders notları §2.5.1–2.5.3; Hansen §4.5, §4.7,
            §7.2–7.3. Aşağıdaki deney ve laboratuvarların veri üretim süreçleri
            kendi panellerinde ayrıca belirtilir."""
        ))


def render() -> None:
    render_topic_header(TOPIC_KEY)
    _render_ols_properties()
    render_inference_refresher()
    application, intuition, self_test = topic_tabs(TOPIC_KEY)
    with application:
        render_lab(get_lab(TOPIC_KEY))
    with intuition:
        render_experiments(KONU02_EXPERIMENTS)
    with self_test:
        render_quiz(get_quiz(TOPIC_KEY))
