# Mimari

## Sınırlar

Uygulama üç katmandan oluşur:

1. `app.py` ortak sayfa kabuğunu kurar ve seçili konuya yönlendirir.
2. `topics/` Streamlit bileşenlerini ve öğrenciye gösterilen metni oluşturur.
3. `core/` Streamlit'ten bağımsız metadata, veri doğrulama, sonuç sözleşmesi, state geçişi ve soru seçimini taşır.

Ekonometrik hesaplama yalnız `core/` altında yer alır. Plotly figürü oluşturma ve Streamlit yerleşimi `topics/` katmanında kalır.

Konu 01-12 sayısal sahipliği:

- `core/ols.py`: OLS, projeksiyon değişmezleri ve FWL.
- `core/inference.py`: klasik, HC1, küme kovaryansı ve delta yöntemi.
- `core/diagnostics.py`: kaldıraç, studentized artık ve Cook uzaklığı.
- `core/functional_forms.py`: log, karesel ve etkileşim yorum dönüşümleri.
- `core/simulation.py`: seed kontrollü ücret ve çıkarım DGP'si.
- `core/datasets.py`: dışarıdan sağlanan hazırlanmış CPS CSV adaptörü.
- `core/causal.py`: potansiyel sonuç, seçim ayrıştırması, içsellik limiti ve okul-kümeli deney DGP'si.
- `core/iv.py`: koşullu Wald, ilk aşama, robust 2SLS ve zayıf araç Monte Carlo hesapları.
- `core/discrete.py`: LPM, Logit, Probit ve ikili sonuç DGP sözleşmeleri.
- `core/marginal_effects.py`: AME, kukla sonlu farkı ve delta-yöntemi standart hataları.
- `core/limited_outcomes.py`: Tobit MLE, koşullu beklentiler ve Heckman iki aşama.
- `core/quantile.py`: check-loss, doğrusal program benchmark'ı ve robust kantil profilleri.
- `core/nonparametric.py`: kernel ağırlıkları, yerel sabit/doğrusal tahmin ve spline seri yaklaşımı.
- `core/cross_validation.py`: gözlem veya küme bazlı dış-kat bandwidth seçimi.
- `core/partialling.py`: kısmen doğrusal model için örneklem-içi esnek artıklaştırma.
- `core/rdd.py`: sharp/fuzzy yerel polinom RDD, etkin örneklem ve tasarım tanıları.
- `core/resampling.py`: gözlem veya küme biriminde seed'li yeniden örnekleme indeksleri.
- `core/bootstrap.py`: pairs/wild OLS bootstrap dağılımları ve güven aralıkları.
- `core/regularization.py`: seyrek DGP, Ridge/Lasso katsayı yolları ve yanlılık-varyans eğrisi.
- `core/pipelines.py`: kat-içi ölçeklemeli Lasso CV, 1-SE kuralı ve ayrılmış test karşılaştırması.
- `core/dml.py`: double selection, gözlem/grup bazlı cross-fitting ve ortogonal hedef tahmini.
- `core/research_workflow.py`: tahmin hedefinden yeniden üretilebilirliğe araştırma denetim aşamaları.
- `core/code_recipes.py`: Sezgi sekmesindeki 48 bölüm için bağımsız Python betiği ve Colab not defteri üretimi.
- `core/labs/spec.py`: ders notu laboratuvar şeması (işlemler, notlardaki sayılar, tekrarlanabilirlik sınıfı).
- `core/labs/expr.py`: türetilmiş değişkenler ve katsayı dönüşümleri için küçük ifade dili; pandas'ta değerlendirilir ve üç dile çevrilir.
- `core/labs/runner.py`: laboratuvarı çalıştırır ve notlarla karşılaştırır. OLS statsmodels ile (üretilen Python koduyla aynı kütüphane), 2SLS linearmodels `IV2SLS(...).fit(cov_type="robust", debiased=True)` ile aynı formüllerle numpy üzerinde hesaplanır; eşitlik testle denetlenir. Tahmin örneklemi (eksik değerler, alt grup, küme değişkeni) açıkça kurulur.
- `core/labs/limited.py`: sınırlı bağımlı değişken hesapları. Logit/Probit statsmodels ile (dayanıklı kovaryans: gözlenen Hessian'lı sandviç, `HC0`); ortalama marjinal etkiler (sürekli terimde türev, kategorik/kesikli terimde referansa göre fark, delta yöntemi SH); ortalama olasılık profili; Tobit (Olsen parametrelemesiyle Newton, R `AER::tobit` ile aynı çözüm); LAD (statsmodels `QuantReg`); ızgara profilleri ve Tobit'in üç hedefi. Monte Carlo için vektörleştirilmiş Logit/Probit Newton'u.
- `core/labs/konuNN.py`: konu laboratuvarları (şu an Konu 1–6).
- `core/labs/sezgi.py`: Sezgi deneylerinin şeması (soru, DGP, parametreler, ölçüler, yorum).
- `core/labs/sezgi_konuNN.py`: konu deneyleri (Konu 1–6'da üçer deney).
- `topics/sim_ui.py`: Sezgi sekmesinin ortak arayüzü.
- `core/quiz/model.py`: "Kendini sına" soru türleri ve notlandırma kuralları.
- `core/quiz/expression.py`: öğrencinin yazdığı formülü güvenli okuma (eval yok; izinli sözdizimi ağacı), LaTeX önizleme ve sayısal eşdeğerlik.
- `core/quiz/konuNN.py`: konu soru setleri (Konu 1: 25, Konu 2–6: 24'er soru).
- `topics/quiz_ui.py`: "Kendini sına" sekmesinin arayüzü.
- `core/codegen/`: Python, R ve Stata üreticileri.
- `core/hansen_data.py`: Hansen veri arşivi indirme, arşivde dosya bulma (`.txt` ve `.dta`), yüklenen dosyayı doğrulama (cps09mar, DDK2011, Card1995, CHJ2004).

## Uygulama laboratuvarı akışı

Bir konu laboratuvarı tek bir `LabSpec` tanımıdır ve dört çıktıyı birlikte besler:

1. `topics/lab_ui.py` adımları, tabloları, grafikleri ve kodu gösterir.
2. `core/labs/runner.py` hesabı yapar; her `Check` notlarda basılı bir sayıdır.
3. `core/codegen/` aynı işlemleri Python, R ve Stata'ya çevirir.
4. `tests/test_lab_konuNN.py` uygulamanın ve üretilen Python/R kodunun notlardaki sayıları ürettiğini doğrular.

Yeni bir konu eklemek, yeni bir `LabSpec` yazmak ve gerekiyorsa üç üreticiye yeni işlem türünü öğretmek demektir. Bir sayı veya işlem yalnız tanımda değişir.

Terim adları dilden bağımsızdır: kategorik değişkenin bir düzeyi `race4=2` biçiminde yazılır ve üreticiler bunu statsmodels'te `C(race4)[T.2]`, R'de `factor(race4)2`, Stata'da `2.race4` olarak çevirir. Marjinal etki sonuçları katsayıları etkiler olan bir model gibi saklanır (`EffectTable` ve `CoefTarget` onları doğrudan kullanır); tablo hücreleri `TableTarget` ile notlarla karşılaştırılır (Stata'da skaler adı `tablo_sütun_satır`, en çok 32 karakter).

## Sezgi deneyleri

Bir Sezgi deneyi (`SimExperiment`), kaydırıcı değerlerinden işlem listesi üreten bir tanımdır. Simülasyon işlemleri (`NewSample`, `Draw`) Python'da `numpy.random.default_rng` ile uygulamayla aynı sırada çekiliş yapar; bu yüzden üretilen Python kodu uygulamadaki sayıların aynısını verir. R (`set.seed` + `rnorm`) ve Stata (`set seed` + `rnormal()`) aynı dağılımdan farklı çekiliş yapar; deneyler "yalnız dağılımda aynı" sınıfındadır.

Grafikler katmanlardan kurulur (`MeanPoints`, `Scatter`, `Curve`, `ModelLine`, `ZeroLine`); katman renkleri `core/codegen/base.layer_styles` ile uygulamada ve üç dilde aynıdır.

`MonteCarlo` işlemi bir işlem bloğunu yeni çekilişlerle tekrarlar ve seçilen katsayı/standart hata ifadelerini bir tabloda toplar (isteğe bağlı olarak güven aralığı kapsama oranı). Rastgele sayı üreteci döngüden önce bir kez tohumlanır; Python kodu uygulamayla aynı çekiliş sırasını izler. Uygulama, blok yalnız normal çekiliş, türetme, OLS (alt örneklemli dahil), 2SLS, kategoriksiz Logit/Probit ve doğrusal indeks içeriyorsa tekrarları vektörleştirerek toplu hesaplar (aynı bit akışı, aynı formüller; döngüyle eşitliği testle denetlenir); öğrenci kodu okunaklı döngü olarak kalır. Stata sonuçları `postfile` ile geçici dosyada (`tempfile`) toplar; do-dosyası bitince dosya silinir. `Histogram` sonuç tablosundaki sütunların dağılımını aynı kutularla üç dilde çizer.

Konu 1–6 yeni yapıya taşındığı için `core/code_recipes.py` içindeki konu01–konu06 tarifleri arayüzde kullanılmaz; diğer konular taşınınca kaldırılacaktır.

## Registry akışı

`core/topic_registry.py` 12 konu için tek başlık ve pedagojik metadata kaynağıdır. Sidebar seçenekleri, konu başlığı, araştırma sorusu, estimand, tanımlama odağı ve başlangıç soruları aynı kayıttan okunur.

`core/data_registry.py` veri kaynağı, gözlem birimi, örneklem kısıtı, beklenen sütunlar, izinli konular, küme ve yeniden örnekleme birimi ile yeniden dağıtım durumunu tutar. `core/datasets.py` bu şemayı çalışma zamanında doğrular; lisans kapısı çözülene kadar gerçek veri depoya alınmaz.

## State akışı

Konu değiştiğinde `core/session_utils.py` önceki konunun soru indeksini ve cevap görünürlüğünü sıfırlar. Metin ölçeği konu state'inden bağımsızdır. Yeni soru cevap görünürlüğünü kapatır.

## Sonuç sözleşmesi

`core/types.py` içindeki `ModelResult`, `EstimandMetadata`, `InferenceSpec` ve `TuningSpec` daha sonraki yöntem dallarının ortak sözleşmesidir. Kovaryans, cluster, seed, bandwidth, lambda, fold ve optimizasyon ayarları örtük varsayılan olarak bırakılmaz.

## Test katmanları

- Saf unit test: registry, metadata ve state geçişleri.
- Kaynak/kontrat testi: 12 başlık, render fonksiyonları ve private veri politikası.
- Streamlit AppTest: app entrypoint, konu geçişi, metin ölçeği ve soru/cevap state'i.
- Sayısal benchmark: OLS ve FWL eşitlikleri, statsmodels HC1/küme kovaryansı, delta yöntemi, etki tanıları ve deterministik DGP.
- Konu 01-02 AppTest: mekanizma uyarısı, veri modu, çıkarım seçimi, fonksiyonel biçim ve etkili gözlem state'i.
- Konu 03-04: Uygulama/Sezgi/Kendini sına sekmeleri, deneylerin ekonometrik doğruluğu (seçim ayrıştırması, zayıflama, olasılık limiti, Wald = 2SLS, zayıf araç kapsaması, LATE), numpy 2SLS ile linearmodels eşitliği, `.dta` okuma.
- Konu 05-06: Uygulama/Sezgi/Kendini sına sekmeleri; AME'nin statsmodels ile, Tobit'in R `AER::tobit` çözümüyle eşitliği; Probit dayanıklı kovaryansının gözlenen Hessian'lı sandviç olması; Sezgi varsayılanlarının notlardaki Tablo 6.2 ve 6.3'ü birebir üretmesi; dışlama kısıtı olmadan Heckman varyansı; Heckman Monte Carlo'sunun vektörleştirilmiş yolla döngünün eşitliği.
- Konu 07-08 AppTest: kantil hedefi, kernel/bandwidth state'i ve CPS/DDK lisans kapıları.
- Konu 09-10 AppTest: RDD yönü, manipülasyon/ilk aşama uyarıları, bootstrap yöntemi, örnekleme birimi ve LM/CPS lisans kapıları.
- Konu 11-12 AppTest: lambda seçim kuralı, kat-içi ölçekleme, CPS/DDK lisans kapıları, OOF cross-fitting ve araştırma akışı state'i.
- Kod tarifi testi: 12 konunun dört bölümünü kapsayan 48 Python betiğinin derlenmesi ve dış veri olmadan çalışması; Colab JSON sözleşmesi.

## Öğrenci kodu akışı

Her konu sekmesi `render_reproduction_code` ile merkezi tarif kaydına bağlanır. Python çıktısı sabit rastgelelik tohumu, veri hazırlama, tahmin ve raporlama adımlarını içerir. Colab çıktısı aynı kodu ve açık paket kurulum hücresini taşır. Gerçek CPS/DDK dosyaları yalnız öğrenci tarafından sağlanır; yokluğunda kontrollü öğretim örneği kullanılır.

## Veri yayınlama kapısı

Gerçek CSV/DTA dosyaları `references_private/` altında yerel tutulur. Açık lisans veya yazılı yeniden dağıtım izni doğrulanmadan `data/` altında public kopya oluşturulmaz.

## Kendini sına

Her soru tek bir kavramı sınar (`concept` alanı); aynı sette iki soru aynı kavramı sınayamaz ve setin
beyan ettiği bütün not bölümleri en az bir soruyla kapsanır. Bu iki kural testle denetlenir. Sorular
notların bölüm sonu egzersizlerini tekrar etmez. Sayısal cevaplar notlardaki basılı değerlerden
alınır ve gerçek veriyle test edilir.

Denklem soruları için öğrenci girdisi Python sözdizimi ağacına çevrilir; yalnız sayılar, tanımlı
semboller, `+ - * / ^` ve `exp/log/sqrt` kabul edilir. Eşdeğerlik, sembollerin rastgele değerlerinde
sayısal karşılaştırmayla sınanır.

## Ortak testler

`tests/test_all_labs.py` ve `tests/test_all_quizzes.py` kayıtlı bütün laboratuvar ve soru setlerini
kendiliğinden kapsar: adım numaralandırması, üç dilde kod üretimi, Stata kuralları, notlardaki sayıların
uygulamada ve üretilen Python/R kodunda yeniden üretilmesi, soru setinde kavram tekilliği, bölüm kapsamı,
cevap anahtarı dengesi. Yeni bir konu kayda eklendiğinde ayrıca test yazmadan bu sözleşmelere tabidir.
