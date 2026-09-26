"""Konu 3 uygulama laboratuvarı: rastgele atama ile seçilmiş alt grubu ayırmak.

Ders notları §3.15'in (Adım 1–4 ve §3.15.5) birebir karşılığıdır. Veri: Hansen'in
DDK2011 dosyası (Duflo, Dupas ve Kremer 2011, Kenya okul deneyi). Her ``Check`` notlarda
basılı bir sayıdır; p-değerleri notlardaki gibi normal yaklaşımla hesaplanır.
"""

from __future__ import annotations

from core.labs.spec import (
    OLS,
    Check,
    CoefTarget,
    DropMissing,
    EffectTable,
    LabSpec,
    LabStep,
    LoadHansen,
    ModelTarget,
    NoteRef,
    ReproClass,
    StatTarget,
)

SECTION = "3.15"
FRAME = "ddk"
CLUSTER = "schoolid"
CONTROLS = ("std_mark", "girl", "agetest", "sbm", "etpteacher")
BALANCE = (
    ("std_mark", "Başlangıç standart puanı", -0.0314, 0.0140, 0.024),
    ("girl", "Kadın öğrenci", 0.0225, 0.0128, 0.079),
    ("agetest", "Test yaşı", 0.1820, 0.0850, 0.032),
    ("sbm", "Okul yönetim komitesi (SBM)", 0.0150, 0.0924, 0.871),
    ("etpteacher", "Ek öğretmen", -0.0217, 0.0119, 0.067),
)

P_VALUE_NOTE = (
    "p-değerleri notlardaki gibi normal yaklaşımla hesaplanır: $2\\Phi(-|t|)$. Stata `estimates table` "
    "küme-dayanıklı SH için t(G−1) dağılımını (G = okul sayısı), R `coeftest` ise t(n−k) dağılımını "
    "kullanır; bu yüzden yazılım tablolarındaki p-değerleri üçüncü basamakta farklı görünebilir "
    "(ör. ham fark için Stata 0,076). Kontrol satırları üç dilde de normal yaklaşımı kullanır. Küme "
    "düzeltmesi $G/(G-1)\\cdot(N-1)/(N-K)$ üç dilde aynıdır; standart hatalar birebir tutar."
)


def _coef(model: str, term: str, expected: float, label: str, quantity: str = "coef", decimals: int = 4) -> Check:
    return Check(label, CoefTarget(model, term, quantity), expected, decimals)


def _balance_model(variable: str) -> OLS:
    return OLS(f"d_{variable}", FRAME, variable, ("tracking",), vcov="cluster", cluster=CLUSTER)


STEPS = (
    LabStep(
        number=1,
        title="Araştırma sorusu ve atama birimi",
        note=NoteRef(SECTION, 1),
        explanation=(
            "Temel soru: öğrencileri başlangıç başarısına göre sınıflara ayıran **tracking** uygulamasının "
            "dönem sonu toplam test puanına etkisi. Müdahale öğrenci düzeyinde değil, **okul düzeyinde** "
            "atanmıştır; aynı okuldaki öğrencilerin hata terimleri bağımsız kabul edilmemelidir. Çıkarımda "
            "okul düzeyinde kümelenmiş standart hata kullanmak tasarımın parçasıdır. Analiz örneklemi, toplam "
            "test puanı gözlenen öğrencilerdir."
        ),
        operations=(
            LoadHansen("ddk2011", "DDK2011.dta", FRAME),
            DropMissing(FRAME, ("totalscore", "tracking", CLUSTER),
                        "Analiz örneklemi: toplam puanı, tedavi durumu ve okulu gözlenen öğrenciler"),
        ),
        checks=(Check("Analiz örneklemi (N)", StatTarget(FRAME, "totalscore", "count"), 5795, decimals=0),),
        takeaway=(
            "Estimand, rastgele atanan tracking programının ortalama etkisidir. Randomization, tedavi göstergesi "
            "ile potansiyel sonuçlar arasındaki bağımsızlığı tasarımla destekler; regresyon kontrolleri "
            "tanımlamayı yaratmak için değil, dengesizlikleri azaltmak ve hassasiyeti artırmak için kullanılır."
        ),
        code_note=(
            "Hansen'in DDK2011 dosyası toplam puanı eksik öğrencileri zaten dışarıda bırakır; eksik değer "
            "ayıklaması burada analiz örneklemini açıkça tanımlamak için yazılmıştır (0 gözlem çıkar)."
        ),
    ),
    LabStep(
        number=2,
        title="Denge tablosunu mekanik bir sınav gibi okumamak",
        note=NoteRef(SECTION, 2, ("Tablo 3.1",)),
        explanation=(
            "Her başlangıç değişkeni için $x_i=\\alpha+\\delta\\,tracking_i+e_i$ regresyonu okul düzeyinde "
            "kümelenmiş standart hatayla tahmin edilir; $\\hat\\delta$ iki grup arasındaki farktır. Her "
            "regresyon o değişkenin gözlendiği öğrencilerle yapılır, bu yüzden N satırdan satıra değişir."
        ),
        operations=tuple(_balance_model(variable) for variable, *_ in BALANCE)
        + (
            EffectTable(
                tuple((label, f"d_{variable}", "tracking") for variable, label, *_ in BALANCE),
                "tablo_31",
                title="Tablo 3.1: tracking grupları arasında seçilmiş başlangıç değişkeni farkları",
                se_label="Küme SH",
            ),
        ),
        checks=tuple(
            check
            for variable, label, difference, se, p in BALANCE
            for check in (
                _coef(f"d_{variable}", "tracking", difference, f"{label}: fark"),
                _coef(f"d_{variable}", "tracking", se, f"{label}: küme SH", "se"),
                _coef(f"d_{variable}", "tracking", p, f"{label}: p-değeri", "p", 3),
            )
        ),
        reproducibility=ReproClass.CONVENTION,
        takeaway=(
            "Bazı p-değerlerinin yüzde 5 veya yüzde 10'un altında olması randomization'ın başarısız olduğunu "
            "kanıtlamaz. Rastgele atama sonlu örneklemde her kovaryatın iki grupta tam eşit olacağını garanti "
            "etmez; çok sayıda kovaryat incelenince bazı farkların yalnız örnekleme değişkenliğiyle belirgin "
            "görünmesi olağandır. Denge tablosu bir geçerlik sınavı değil, örneklem kompozisyonunu anlamak ve "
            "önceden belirlenmiş kovaryat ayarlamasının yararını değerlendirmek için kullanılır."
        ),
        code_note=P_VALUE_NOTE,
    ),
    LabStep(
        number=3,
        title="Ham fark ile kovaryat ayarlı tahmini karşılaştırmak",
        note=NoteRef(SECTION, 3, ("Tablo 3.2",)),
        explanation=(
            "**Ham fark:** $totalscore_i=\\alpha+\\tau\\,tracking_i+e_i$. **Kovaryat ayarlı** model başlangıç "
            "standart puanını, kadın göstergesini, test yaşını, SBM ve ek öğretmen göstergelerini ekler. İki "
            "modelde de standart hatalar okul düzeyinde kümelenir. Kovaryat ayarlı modelde yalnız bütün "
            "kontrolleri gözlenen öğrenciler kalır."
        ),
        operations=(
            OLS("ham", FRAME, "totalscore", ("tracking",), vcov="cluster", cluster=CLUSTER),
            OLS("ayarli", FRAME, "totalscore", ("tracking", *CONTROLS), vcov="cluster", cluster=CLUSTER),
            EffectTable(
                (("Ham fark", "ham", "tracking"), ("Kovaryat ayarlı", "ayarli", "tracking")),
                "tablo_32",
                title="Tablo 3.2: tracking etkisi için iki temel tahmin",
                se_label="Küme SH",
            ),
        ),
        checks=(
            _coef("ham", "tracking", 1.258, "Ham fark: τ̂", decimals=3),
            _coef("ham", "tracking", 0.704, "Ham fark: küme SH", "se", 3),
            _coef("ham", "tracking", 0.074, "Ham fark: p-değeri", "p", 3),
            Check("Ham fark: n", ModelTarget("ham", "nobs"), 5795, decimals=0),
            _coef("ayarli", "tracking", 1.375, "Kovaryat ayarlı: τ̂", decimals=3),
            _coef("ayarli", "tracking", 0.699, "Kovaryat ayarlı: küme SH", "se", 3),
            _coef("ayarli", "tracking", 0.049, "Kovaryat ayarlı: p-değeri", "p", 3),
            Check("Kovaryat ayarlı: n", ModelTarget("ayarli", "nobs"), 5135, decimals=0),
        ),
        reproducibility=ReproClass.CONVENTION,
        takeaway=(
            "İki nokta tahmini birbirine yakın: program etkisi yaklaşık 1,3–1,4 test puanı. p-değerinin 0,074'ten "
            "0,049'a geçmesi farklı bir ekonomik hikâye değildir; \"anlamlı/anlamsız\" ikili dili yanıltıcıdır. "
            "Ayrıca sütunlar arasında örneklem değişir (5.795 → 5.135): fark yalnız \"kontroller eklendi\" diye "
            "yorumlanmamalı, örneklem kompozisyonunun da değiştiği görülmelidir."
        ),
        code_note=P_VALUE_NOTE,
    ),
    LabStep(
        number=4,
        title="Aynı veride seçilmiş bir grubun neden nedensel karşılaştırma olmadığını görmek",
        note=NoteRef(SECTION, 4),
        explanation=(
            "Tracking okullarında öğrenciler başlangıç başarısına göre düşük ve yüksek akışlara ayrılmıştır. "
            "`lowstream` göstergesi rastgele tedavi değildir; başlangıç performansına göre mekanik olarak oluşur. "
            "Bunu görmek için yalnız tracking okullarında $std\\_mark_i=\\alpha+\\delta\\,lowstream_i+e_i$ "
            "regresyonunu tahmin ederiz."
        ),
        operations=(
            OLS("akim", FRAME, "std_mark", ("lowstream",), vcov="cluster", cluster=CLUSTER, where=("tracking", 1.0)),
            EffectTable(
                (("Tracking okullarında düşük akış (lowstream)", "akim", "lowstream"),),
                "akim_tablosu",
                title="Seçilmiş grup: düşük akıştaki öğrencilerin başlangıç puanı farkı",
                se_label="Küme SH",
            ),
        ),
        checks=(
            _coef("akim", "lowstream", -1.574, "δ̂ (lowstream)", decimals=3),
            _coef("akim", "lowstream", 0.016, "Küme SH", "se", 3),
        ),
        takeaway=(
            "Düşük akıştaki öğrencilerin başlangıç performansı sistematik biçimde çok daha düşüktür. Dönem sonu "
            "puanını düşük ve yüksek akış arasında doğrudan karşılaştırmak lowstream'in nedensel etkisini "
            "vermez: seçim, değişkenin tanımına gömülüdür. Aynı veri setinde tracking için randomization güçlü "
            "bir tanımlama kaynağıdır; lowstream karşılaştırması için aynı argüman kullanılamaz."
        ),
        code_note=(
            "Kodda kritik satır yalnız regresyon formülü değildir: küme değişkeni (`schoolid`) atama biriminin "
            "okul olduğunu çıkarım aşamasına taşır. Tedavi okul, firma, köy veya bölge düzeyinde atanmışsa "
            "standart hata stratejisi de bu tasarımla uyumlu olmalıdır."
        ),
    ),
    LabStep(
        number=5,
        title="Makale çıktısını okurken sorulacak tanımlama soruları",
        note=NoteRef("3.15.5", 0),
        explanation=(
            "1. Tedavi nasıl atanmıştır: rastgele mi, politika kuralıyla mı, bireysel seçimle mi?\n"
            "2. Atama birimi ile gözlem birimi aynı mıdır?\n"
            "3. Standart hatalar hangi düzeyde kümelenmiştir?\n"
            "4. Kovaryatlar tedaviden önce mi ölçülmüştür?\n"
            "5. Kovaryat eklenince örneklem değişiyor mu?\n"
            "6. Sonuç \"program etkisi\" mi, yoksa seçilmiş gruplar arası koşullu fark mı?\n"
            "7. Yalnız p-değerinin eşik değiştirmesine bakılarak ekonomik sonuç abartılıyor mu?"
        ),
        takeaway=(
            "Tanımlama, regresyon komutundan önce gelir. Aynı OLS komutu rastgele atanmış bir tedavi katsayısı "
            "için nedensel bir estimand'ı tahmin edebilirken, seçilmiş bir grup göstergesi için yalnız koşullu "
            "ilişkiyi özetleyebilir. Yöntemin adı değil, değişkenin veri üretim sürecindeki rolü nedensel "
            "yorumu belirler."
        ),
    ),
)


KONU03_LAB = LabSpec(
    topic_key="konu03",
    title="Rastgele Atama ile Seçilmiş Alt Grubu Ayırmak",
    dataset="ddk2011",
    note_section=SECTION,
    steps=STEPS,
    labels=(
        ("(sabit)", "Sabit"),
        ("totalscore", "Toplam test puanı"),
        ("tracking", "Tracking"),
        ("schoolid", "Okul"),
        ("std_mark", "Başlangıç standart puanı"),
        ("girl", "Kadın öğrenci"),
        ("agetest", "Test yaşı"),
        ("sbm", "Okul yönetim komitesi (SBM)"),
        ("etpteacher", "Ek öğretmen"),
        ("lowstream", "Düşük akış (lowstream)"),
        ("ham", "Ham fark"),
        ("ayarli", "Kovaryat ayarlı"),
        ("akim", "Akım karşılaştırması"),
    ),
)
