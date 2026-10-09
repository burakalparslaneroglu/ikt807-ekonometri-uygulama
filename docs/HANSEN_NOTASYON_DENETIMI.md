# Hansen notasyon denetimi — 9 Ekim 2026

Uygulamanın 12 konusu, güncel ders notlarının notasyon rehberi ve ilgili konu/laboratuvar metinleriyle karşılaştırıldı. Temel referans Hansen'in 11 Mart 2021 Econometrics sürümü (§1.3 ve §3.10–3.11). Yerel ders notları ve veri dosyaları bu depoya kopyalanmadı.

## Temel karar

`Y = X'β + e` tek gözlemdir: Y ve e skaler, X ve β k×1 vektördür. Hansen'in convention'ında rassal vektörün kalın olmaması skaler olduğu anlamına gelmez. Örneklem matrisleri uygulamada kalın italik yazılır: **Y = Xβ + e**, X n×k. Gerçekleşmeler küçük harfle gösterilir. Skaler X kullanılan deneylerde sabit terimli regresör vektörü (1, X)' olur.

Her sayfanın başına açılabilir ortak ve konuya özgü notasyon rehberi eklendi (`core/notation.py`, `topics/shared.py`). Üç sekmenin sırası ve araştırma soruları korundu.

## Bulgular ve düzeltmeler

| Konu | Kontrol ve yapılan düzeltme |
|---|---|
| 1 — Regresyon | Anakütle X/Y ile örneklem matrisleri ayrıldı. Kalın italik matrisler, artık vektörü ve normal denklemlerde 0_k tanımlandı. |
| 2 — Çıkarım | FWL/eksik değişken formüllerindeki rassal skalerler X_1/X_2 olarak yazıldı. Doğrusal birleşimin ağırlıkları, ilgili kovaryans alt matrisi ve tahmini kovaryans açıklandı. Küme düzeyindeki rassal X_g ayrıldı. |
| 3 — Tanımlama | Potansiyel sonuçlar Y_1/Y_0 ve Y_{1i}/Y_{0i}; nedensel etki θ olarak yazıldı. ATE/ATT ve sabit etki ayrımı tanımlandı. |
| 4 — IV | Yığılmış 2SLS artığında gereksiz X transpozu kaldırıldı. Araç projeksiyonu ve boyutları tanımlandı. LATE etkileri θ, potansiyel tedaviler D_1/D_0 ile gösterildi. Notlardaki indirgenmiş biçim hatası η korundu. |
| 5 — Ayrık sonuçlar | Hessian Q ve skor kovaryansı Ω kullanıldı; matris ve tek parametreli skaler karşılıklar ayrıldı. AME delta gradyanı ve tahmini kovaryans açıklandı. |
| 6 — Seçim | Cov(e,u) = σ_21, Var(u)=1 normalizasyonu ve ters Mills oranı λ ayrı tanımlandı. Skaler X/Z kullanan simülasyonda seçim indeksi açık biçimde yazıldı. |
| 7 — Kantil | β_τ vektörü ile β_{j,τ} bileşeni ayrıldı. Fark testi aynı j için yazıldı. HK matrisleri kalın italik, kovaryans ise tahmin edicinin ölçeğinde gösterildi. Q_τ[Y\|X] kullanıldı. |
| 8 — Parametrik olmayan | K_i yerel ağırlıkları tanımlandı. Yerel doğrusal minimizasyonun iki parametresi ile m tahmini ayrıldı. Kısmen doğrusal modelin kaynak gösterimle eşleştirmesi ve g/m_Y farkı açıklandı. |
| 9 — RDD | Eşik etkisi θ(c), yerel kısa yazım θ ve bulanık RDD hedefi θ_FRD olarak ayrıldı. Sonuç/tedavi sıçramaları Δ_Y/Δ_D tanımlandı. Tablo ve grafik etiketleri eşleştirildi. |
| 10 — Bootstrap | Tek gözlem çiftleri ve yığılmış bootstrap matrisleri ayrıldı. Skaler X kullanan wild örneğinde uyum değeri β̂_0 + β̂_1 X_i olarak açıldı. |
| 11 — Düzenlileştirme | Örneklem matrisleri, p ve özdeğer r_j açıklandı. Hansen SSE cezası λ ile yazılım cezası λ_y; Lasso için 2nλ_y dönüşümü; Elastic Net L1/L2 ağırlıkları ayrıldı. Grafik, tablo ve CV etiketleri aynı convention'a geçirildi. |
| 12 — DML | Katlar A_k, yardımcı fonksiyonlar, skaler artıklar ve gözlem skoru tanımlandı. Ürün hızına tedavi yardımcı hatasının kare terimi eklendi. Medyan birleştirmesinde SH_s² = tahmini varyans ayrımı düzeltildi. |

**Major:** Yığılmış 2SLS artık formülü boyutsal olarak yanlıştı. DML'nin artıklaştırma skoru için yalnız iki yardımcı hatanın çarpımını sınırlayan eski koşul eksikti. İkisi de mevcut kaynak notlara uyumlu biçimde düzeltildi; tahmin algoritmaları değiştirilmedi.

DML'de δ_D = m̂_D − m_D ve δ_Y = m̂_Y − m_Y için skorun koşullu ortalama kalan terimi E[δ_D δ_Y − θδ_D²]'dir. Bu nedenle, diğer düzenlilik koşulları altında yeterli hız koşulu

    ||δ_D||_{P,2} (||δ_Y||_{P,2} + ||δ_D||_{P,2}) = o_p(n^(-1/2))

olarak yazıldı. Bağımsız Gauss kareselleştirmesiyle, δ_Y=0 ve δ_D=n^(-1/5)X örneğinde çarpım sıfır olsa da √n E[skor] = −θ n^(1/10) olduğu doğrulandı. Her iki hatanın o_p(n^(-1/4)) olması bu kalan terim için yeterlidir; tek başına tam bir DML limit teoremi değildir.

**Minor/Editorial:** Kantil katsayısı/bileşeni, tahmin edici varyansı/limit kovaryansı, gerçekleşme/rassal değişken ve kalın italik matris ayrımları düzeltildi. Ek yerel semboller rehberde veya ilgili soruda tanımlandı.

**NOTE_CONSISTENCY_ISSUE — kaynak notta kalan açıklama:** Yerel `12_dml_butunlesik_arastirma.tex` dosyasındaki “DML açısından kriter neden farklı?” kutusu (satır 473), hâlâ yalnız iki farklı yardımcı hatanın çarpımından söz ediyor. Aynı dosyanın §12.8'deki açık norm koşulu ve hemen altındaki açıklaması kare terimini doğru biçimde içeriyor. Uygulama bu açık koşulu izliyor. Bu çalışma Streamlit deposunu hedeflediği için dışarıdaki not dosyası değiştirilmedi; kutu metni de norm koşuluna göre güncellenmeli. Bu, yeni bir teorik tercih gerektirmeyen ancak kaynak metinde ayrıca kayda geçirilmesi gereken bir tutarlılık düzeltmesidir.

## Korunan davranış

- Önceden değiştirilmiş 13 dosyadaki düzeltmeler korundu; özellikle ayrışma, QR katsayı sırası, RDD kaynak-sayı karşılaştırması ve koşullu yorumlar geri alınmadı.
- Notların 12 laboratuvarından üretilen Python/R/Stata betiklerinin `NOTES_MD5` değeri değişmedi: `e6eb2b280607de0fe2a5b242b5c5a38d`. Eski teknik kontrol/işlem etiketleri bu kod uyumluluğu için kaynakta kalabilir; uygulama tablosu ve RDD grafiklerinde güncel θ/λ_y gösterilir.
- İşlem tanımları, sayısal hedefler, veri yolları ve tohumlar korundu. Notasyon değişiklikleri hesapların değerini değiştirmez.
- Cevap motoru yeni gösterimleri kabul ederken eski Y(1)/Y(0), τ ve σ_eu girişlerini de kabul eder. Bu uyumluluk, yeni convention'ı öğretirken önceki sürüm kullanıcılarının doğru cevaplarını reddetmemeyi sağlar.
- Commit, push veya harici yayın yapılmadı.

## Doğrulama

881 testlik tam paket çalıştırıldı: 780 geçti, 99 atlandı; eski τ gösterimini bekleyen iki RDD test ifadesi θ gösterimine uyarlanarak tekrar çalıştırıldı ve 2/2 geçti. Aynı üretim kodu için toplam 782 test doğrulandı; açık başarısızlık kalmadı. Son `pytest --lf` sonucu başarılıdır. 99 atlamanın 75'i Rscript/R readxl ortamına, 24'ü testlere sağlanmamış gerçek Hansen veri dosyalarına bağlıdır; bu yollar doğrulanmış sayılmıyor.

Python compileall, git diff --check ve üretilen not kodu hash kontrolü geçti. Gerçek Streamlit/KaTeX denetiminde 932 metin/formül bloğundan 1842 matematik gösterimi üretildi; son kontrolde ayrıştırma hatası veya tanımsız komut yoktu. Tüm soru setleri, not ve alternatif laboratuvarların statik açıklamaları, rehberler ve 36 Sezgi deneyinin varsayılan DGP'leri kapsandı. Dinamik kendi-veri yolları ilgili mevcut testlerle denetlendi; her olası kullanıcı dosyası tarayıcıda tek tek denenmedi.

Ayrıntılı test kayıtları `TASK_STATE.md` içinde; geçici denetim betikleri ve loglar git tarafından izlenmeyen `tmp/notation_audit` altındadır.
