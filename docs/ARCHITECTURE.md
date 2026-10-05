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
- `core/labs/limited.py`: sınırlı bağımlı değişken hesapları. Logit/Probit statsmodels ile (dayanıklı kovaryans: gözlenen Hessian'lı sandviç, `HC0`); ortalama marjinal etkiler (sürekli terimde türev, kategorik/kesikli terimde referansa göre fark, delta yöntemi SH); ortalama olasılık profili; Tobit (Olsen parametrelemesiyle Newton, R `AER::tobit` ile aynı çözüm); ızgara profilleri ve Tobit'in üç hedefi. Monte Carlo için vektörleştirilmiş Logit/Probit Newton'u.
- `core/labs/quantreg.py`: kantil regresyon ve LAD. Koenker'in dual doğrusal programı HiGHS (`scipy.optimize.linprog`) ile kesin çözülür; çözüm tekse R `quantreg::rq` (Barrodale–Roberts) ve Stata `qreg` ile aynı köşe çözümüdür. Standart hatalar Hendricks–Koenker sandviçidir (R `summary.rq(se = "nid")`; Hall–Sheather bant genişliği). İki kantil arasındaki katsayı farkı ortak asimptotik kovaryansla sınanır. HiGHS aynı süreçte eşzamanlı çağrılmaz: Windows'ta eşzamanlı çağrılar erişim ihlaline yol açabildiği için bütün çözümler süreç genelinde bir kilitle sıraya konur (ayrı Streamlit oturumları dahil).
- `core/labs/smoothing.py`: Gauss çekirdekli yerel doğrusal ve yerel sabit (Nadaraya–Watson) tahmin (kapalı biçim), birini-dışarıda-bırak ve küme-silmeli çapraz doğrulama, kutu ortalamaları, Robinson artıklaştırması.
- `core/labs/rdd.py`: keskin RDD. Hansen'in birim varyanslı çekirdekleri (üçgen: pencere ±h√6, dikdörtgen: ±h√3; `scale="window"` ile h pencerenin yarı genişliğidir), eşik çevresinde `[1, D, R, D·R]` üzerine ağırlıklı EKK ve HC1 sandviç kovaryansı (statsmodels `WLS(...).fit(cov_type="HC1")`, R `lm(weights = w)` + `vcovHC(type = "HC1")`, Stata `regress [aw = w], vce(robust)` ile aynı sayı); bant genişliği tablosu; eşiğin iki yanında noktasal %95 bantlı yerel doğrusal eğri (her noktada o taraftaki gözlemlerle ağırlıklı EKK sabit terimi, eşikte RDD limitlerine eşit).
- `core/labs/resample.py`: pairs, wild (Rademacher) ve küme bootstrap tekrarları. Çekiliş sırası üretilen Python koduyla aynıdır (`rng.integers(0, n, n)`, `rng.choice([-1.0, 1.0], n)`, kümelerde `rng.integers(0, G, G)`); her tekrar tasarım matrisiyle `np.linalg.lstsq` ile çözülür, istenirse HC1 SH (percentile-t) de hesaplanır. Özet: SH (ddof = 1) ve percentile sınırları (`np.quantile`; R `quantile(type = 7)` ile aynı tanım).
- `core/labs/konuNN.py`: konu laboratuvarları (şu an Konu 1–10).
- `core/labs/sezgi.py`: Sezgi deneylerinin şeması (soru, DGP, parametreler, ölçüler, yorum).
- `core/labs/sezgi_konuNN.py`: konu deneyleri (Konu 1–10'da üçer deney).
- `topics/sim_ui.py`: Sezgi sekmesinin ortak arayüzü.
- `core/quiz/model.py`: "Kendini sına" soru türleri ve notlandırma kuralları.
- `core/quiz/expression.py`: öğrencinin yazdığı formülü güvenli okuma (eval yok; izinli sözdizimi ağacı), LaTeX önizleme ve sayısal eşdeğerlik.
- `core/quiz/konuNN.py`: konu soru setleri (Konu 1: 25, Konu 2–10: 24'er soru).
- `topics/quiz_ui.py`: "Kendini sına" sekmesinin arayüzü.
- `core/codegen/`: Python, R ve Stata üreticileri. `python_np.py`, `r_np.py` ve `stata_np.py` kantil regresyon ve parametrik olmayan işlemleri üretir: Python'da `linprog` ile kesin çözüm ve kodda yazılı Hendricks–Koenker kovaryansı; R'de `quantreg::rq` ve `summary(se = "nid")`; Stata'da `qreg` ve kovaryansı `ereturn repost` ile yazan `hk_sh` programı. Yerel doğrusal tahmin ve CV Python'da NumPy, R'de matris işlemleri, Stata'da Mata ile aynı formüllerle yazılır. Tam Python betikleri içe aktarmalardan hemen sonra çıktıyı UTF-8'e çevirir: Windows'ta dosyaya ya da başka programa yönlendirilen çıktı yerel kod sayfasını (ör. cp1254) kullanır ve τ, β̂ gibi karakterleri yazamaz. Python kodu `.dta` dosyalarını uygulama gibi `convert_categoricals=False` ile okur ve tamsayı sütunları int64'e, float32 sütunları float64'e çevirir: Hansen'in dosyalarında tamsayılar byte/int/long olarak saklanabilir ve pandas'ın okuduğu int8 sütunda kare alma uyarısız taşar. `python_rdd_boot.py`, `r_rdd_boot.py` ve `stata_rdd_boot.py` RDD ve bootstrap işlemlerini üretir: RDD her dilde kodda yazılı bir yardımcıdır (Python `rdd_yerel_dogrusal`, R `rdd_yerel_dogrusal`, Stata `rdd_yd` programı; eğri Python/R'de vektör işlemleri, Stata'da Mata `rdd_egrisi`); bootstrap Python'da uygulamayla aynı çekiliş sırasını izleyen döngü, R'de `sample.int` + `lm.fit`, Stata'da `bsample` (kümede `bsample, cluster()`) ve `postfile` ile geçici dosyadır.
- `core/hansen_data.py`: Hansen veri arşivi indirme, arşivde dosya bulma (`.txt` ve `.dta`), yüklenen dosyayı doğrulama (cps09mar, DDK2011, Card1995, CHJ2004, LM2007; DDK2011'de Konu 8 için `percentile` sütunu, LM2007'de `povrate60` ve `mort_age59_related_posths` gerekir).

## Uygulama laboratuvarı akışı

Bir konu laboratuvarı tek bir `LabSpec` tanımıdır ve dört çıktıyı birlikte besler:

1. `topics/lab_ui.py` adımları, tabloları, grafikleri ve kodu gösterir.
2. `core/labs/runner.py` hesabı yapar; her `Check` notlarda basılı bir sayıdır.
3. `core/codegen/` aynı işlemleri Python, R ve Stata'ya çevirir.
4. `tests/test_lab_konuNN.py` uygulamanın ve üretilen Python/R kodunun notlardaki sayıları ürettiğini doğrular.

Yeni bir konu eklemek, yeni bir `LabSpec` yazmak ve gerekiyorsa üç üreticiye yeni işlem türünü öğretmek demektir. Bir sayı veya işlem yalnız tanımda değişir.

Terim adları dilden bağımsızdır: kategorik değişkenin bir düzeyi `race4=2` biçiminde yazılır ve üreticiler bunu statsmodels'te `C(race4)[T.2]`, R'de `factor(race4)2`, Stata'da `2.race4` olarak çevirir. Marjinal etki sonuçları katsayıları etkiler olan bir model gibi saklanır (`EffectTable` ve `CoefTarget` onları doğrudan kullanır); tablo hücreleri `TableTarget` ile notlarla karşılaştırılır (Stata'da skaler adı `tablo_sütun_satır`, en çok 32 karakter).

## Ek veri kaynakları: alternatif örnek ve kendi verin

Uygulama sekmesinde ek kaynakları olan konularda (`core/labs/ornekler.py`) en üstte veri kaynağı seçilir
(`LabSpec.source`: `notlar`, `alternatif`, `kendi`). Notlardaki laboratuvar (`core/labs/konuNN.py`) değişmez; her konu
için ek bir **genel uygulama** yazılır (`core/labs/ornek_konuNN.py`): notlardaki adımlar aynı numaralar ve aynı
işlemlerle, verisi bir `Case`'ten (roller → sütunlar) gelecek biçimde.

- **Alternatif örnek** (`alternative()`): Hansen'in arşivindeki başka bir gerçek veriyle kurulur. Tanım veriden önce
  kurulur; kontrollerin beklenen değerleri bu verinin tam örneklemindeki sayılardır (`ALT_EXPECTED`, `with_expected`)
  ve metinlerdeki sayılar onlardır. Testler sayıları bağımsız bir hesapla (numpy, statsmodels) ve üretilen Python/R
  koduyla doğrular. Veri paneli notlardakiyle aynıdır; oturum anahtarları `konuNN_alternatif_` önekiyle ayrılır.
- **Kendi verin** (`CustomLab`): öğrenci dosyayı yükler, sütunları rollere atar (`Role`), açık/kapalı seçenekleri
  (`Option`, ör. sonucun logaritması) seçer. `core/labs/kendi_veri.py` dosyayı üretilen kodun okuyacağı biçimde okur ve
  temizler (İKT 217/305 ile aynı kurallar); sonuç bir `ReadFile` işlemidir: temizlenmiş değerler tanımın içindedir,
  kod aynı değerleri dosyadan elde eder. Kontrollerin beklenen değerleri uygulamanın hesabıdır (`with_app_values`);
  yorumlar sonuçlardan yazılır (`LabStep.note_for`). Konuya özgü doğrulama (ör. tam doğrusal bağlantı, logaritma için
  analiz örnekleminde pozitif sonuç, koşullu ortalama için en çok 30 değer) açık bir iletiyle reddeder. Sayısal
  hassasiyet `ornek.stable_design` ile denetlenir: sütunları birim uzunluğa ölçeklenmiş tasarımın koşul sayısı en çok
  3.000 olmalı (neredeyse doğrusal bağlantı, ör. yıl ve karesi, QR ile sözde ters çözümünü ayrıştırır) ve
  statsmodels'in sözde ters çözümü, ortalanıp ölçeklenmiş tasarımdaki çözümle 1e-9 içinde aynı olmalı (kötü ölçek
  yalnız sözde tersi bozar; R'nin QR çözümü etkilenmez). Tam uyumda (R² ≈ 1) standart hataya bağlı kontroller kodda
  karşılaştırılmaz. Python okuyucusu sayısal sütunları uygulama gibi ondalıklı sayıya, R okuyucusu (Excel'de metin
  olarak saklanmış sayılar dahil) `as.numeric` ile sayıya çevirir. Kod Python ve R'da üretilir (`languages_for`);
  Stata üreticisi `ReadFile` için hata verir.

Yeni işlemler: `ReadFile` (dosya okuma ve temizleme), `CompleteCases` (tam gözlemler), `Indicator` (kategoriden 0/1
gösterge). Notların kodunu değiştirmeyen isteğe bağlı alanlar: `RegressionTable.title`, `ProjectionPlot.relative_size`
(nokta boyutu en kalabalık gruba göre), `LabStep.note_for`. Kontrol başlığı, `kontrol_et` metni ve toleransı, betik
başlığı ve dosya adı (`ikt807_konuNN_alternatif`, `ikt807_konuNN_kendi_verim`) kaynağa göre yazılır; kendi verinde
tolerans büyük sayılar için göreli bir pay içerir. Notlardaki 12 laboratuvarın üç dildeki bütün kodunun özeti
`tests/test_lab_variants.py::NOTES_MD5` ile kilitlidir.

Konu 3–4 (Blok B) ile eklenenler: `GroupMean` (bir değişkenin grup ortalaması, isteğe bağlı bir koşulu sağlayan
satırlardan, grubun bütün satırlarına yazılır; Python `where().groupby().transform("mean")`, R `ave(ifelse(...), FUN =
mean(na.rm = TRUE))`, Stata `egen double ... = mean(cond(...)), by()`; eksik grup ya da koşul değeri dillerde farklı
işlendiği için bu sütunlar eksiksiz olmalıdır, uygulama aksi hâlde hata verir), `IV.categorical` (2SLS'te kategorik
dışsal kontroller; en küçük düzey referans: Python `C()`, R `factor()`, Stata `i.`), `Role.shared` (bir rolün sütunu
ek sütun olarak da seçilebilir: seçilmiş grubun dayandığı başlangıç puanı denge tablosunda da yer alır) ve
`CustomLab.suggest` (dosya ilk açıldığında konuya özgü rol önerileri; başka bir role önerilen sütun genel tahminlerde
kullanılmaz). Büyük tasarımda (`runner.LARGE_DESIGN`: tasarım matrisi 3 milyon hücre ve üstü) OLS sonucu katsayı,
standart hata (HC1 dahil), uyum değerleri ve artıkları önbelleğe aldıktan sonra tasarım matrisini ve sözde tersini
bırakır (`runner._slim`); tasarımı yeniden isteyen işlemler (Breusch–Pagan, bootstrap) böyle bir modelde açık bir
hatayla durur. Eşik yalnız AK1991'in 329.509 gözlemli modellerini etkiler: notlardaki laboratuvarların en büyük
tasarımı 1,47 milyon hücredir (Konu 2), kendi verin en çok 10.000 satırdır.

Konu 5–8 (Blok C) ile eklenenler, notların kodunu değiştirmeden:

- `Tobit.scale`: sonuç büyük ölçekliyse (standart sapma 1.000 ve üstü) R `survreg` sonucu bu ölçeğe bölerek tahmin
  eder ve katsayıları, doğrusal tahmini ve σ'yı geri ölçekler (büyük ölçekte `survreg`'in yakınsaması bozuluyor).
- `codegen.r_gen.r_row_name`: R'nin `data.frame` satır adının yazımı (R kısa olduğunda bilimsel gösterim kullanır,
  ör. `1e+05`); tablo kontrolleri satırı bu adla bulur.
- `quantreg.solution_range` / `fitted_value_range`: bütün optimal kantil (LAD) çözümleri üzerinde g'β'nın en küçük ve
  en büyük değeri. Tümleyici gevşeklikle: dual çözüm dejenere değilse çözüm tektir; değilse optimal küme k değişkenli
  küçük bir doğrusal programdır. Kendi verinde bir kontrol yalnız bu aralık gösterim penceresinin
  (yuvarlanmış değer ± 0,5·10⁻ᵈ) içindeyse kodda karşılaştırılır: Konu 6'da LAD eğrisi (`lad_guard`), Konu 7'de her
  kantil modelinin katsayıları, kesin yüzde etkiler ve 0,10–0,90 farkı (`stability`). Hendricks–Koenker standart
  hatası τ ± h kantillerinin çözümlerini kullanır; onlar sayısal olarak tek değilse standart hata kontrolleri çıkarılır,
  yoğunluk matrisi H tekilse (az değerli sonuç) dosya açık bir iletiyle reddedilir.
- `smoothing.cv_curve` önbelleği veri başına, h başına tutulur (aynı veriyle ızgaranın bir parçası yeniden
  hesaplanmaz); `cv_conditioning` ve `local_conditioning` yerel doğrusal tahminin göreli belirleyicisini
  ρ = (S₀S₂ − S₁²)/(S₀S₂) verir. ρ küçükse (bir noktanın çevresinde ağırlığı anlamlı tek bir farklı değer kalır) formül
  0/0 ya da yuvarlama gürültüsü üretir ve diller farklı sayı verir (R grafiği çizemez). Konu 8'in kendi verisinde CV
  ızgarasının alt ucu, dışarıda bırakılan bütün tahminlerde ve grafik eğrisinde ρ ≥ 10⁻⁶ olan en küçük h'ye
  yükseltilir; notların formülü değişmez.
- Ondalıklar ölçeğe göre seçilir: Konu 5'te profil olasılıkları, Konu 6'da eğriler, Konu 7'de her terimin katsayısı,
  Konu 8'de tahminler; gösterim çok fazla ondalık gerektiriyorsa (Konu 7'de 8, Konu 8'de 6 ondalıktan fazla) dosya
  birimi değiştirme önerisiyle reddedilir. Arayüz tabloları kontrollerin ondalığını kullanır.
- Konu 8'in kendi verisinde kontroller katsayılar değil seçilmiş noktalardaki tahminler olduğu için `stable_design`
  yerine tahmin sınaması kullanılır (`ornek_konu08._stable_fits`): doğrusal, kübik ve spline tasarımının koşul sayısı
  en çok 10⁶ (R'nin QR'ı sütun düşürmez) ve uygulamanın tahminleri ortalanıp ölçeklenmiş tasarımdakilerle 10⁻⁹ içinde
  aynı. Çarpık bir açıklayıcı (ör. gelir) böylece kabul edilir; AL1999'un kendisi koşul sayısı 2·10⁴'tür.

Öğrencinin dosyası, seçimleri ve ondan kurulan hesap yalnız oturum belleğindedir (`st.session_state`); ortak önbelleğe
girmez. Kaynak seçimi, adım seçimi ve kendi verin paneli, Streamlit'in çizilmeyen widget durumunu silmesine karşı gölge
anahtarlarla (`_kalici_`) korunur (`topics/kendi_veri_ui.py`): öğrenci kaynak ya da konu değiştirip döndüğünde ya da
geçersiz bir seçimi düzelttiğinde kaldığı adımda devam eder.

## Sezgi deneyleri

Bir Sezgi deneyi (`SimExperiment`), kaydırıcı değerlerinden işlem listesi üreten bir tanımdır. Simülasyon işlemleri (`NewSample`, `Draw`) Python'da `numpy.random.default_rng` ile uygulamayla aynı sırada çekiliş yapar; bu yüzden üretilen Python kodu uygulamadaki sayıların aynısını verir. R (`set.seed` + `rnorm`) ve Stata (`set seed` + `rnormal()`) aynı dağılımdan farklı çekiliş yapar; deneyler "yalnız dağılımda aynı" sınıfındadır.

Grafikler katmanlardan kurulur (`MeanPoints`, `Scatter`, `Curve`, `ModelLine`, `ZeroLine`, `BinMeans`, `LocalCurve`, `RDDCurve`, `VLine`); katman renkleri `core/codegen/base.layer_styles` ile uygulamada ve üç dilde aynıdır.

`MonteCarlo` işlemi bir işlem bloğunu yeni çekilişlerle tekrarlar ve seçilen katsayı/standart hata ifadelerini bir tabloda toplar (isteğe bağlı olarak güven aralığı kapsama oranı). Rastgele sayı üreteci döngüden önce bir kez tohumlanır; Python kodu uygulamayla aynı çekiliş sırasını izler. Uygulama, blok yalnız normal çekiliş, türetme, OLS (alt örneklemli dahil), 2SLS, kategoriksiz Logit/Probit ve doğrusal indeks içeriyorsa tekrarları vektörleştirerek toplu hesaplar (aynı bit akışı, aynı formüller; döngüyle eşitliği testle denetlenir); öğrenci kodu okunaklı döngü olarak kalır. Stata sonuçları `postfile` ile geçici dosyada (`tempfile`) toplar; do-dosyası bitince dosya silinir. `Histogram` sonuç tablosundaki sütunların dağılımını aynı kutularla üç dilde çizer.

`Bootstrap` işlemi bir modeli (pairs, wild veya küme) yeniden örnekler ve seçilen ifadeleri (`BOOT` modelinin katsayıları ve HC1 SH'leri) her tekrarda toplar; özet skalerler `{sonuç}_{sütun}_{se|lo|hi}` adıyla saklanır ve `Scalar`/`ScalarTable` bunlara `ref()` ile başvurur. Tohum verilmezse deneyin üreteci veri çekilişlerinin ardından kullanılmaya devam eder (Konu 10 Deney 1 böylece notlardaki Tablo 10.1'i birebir üretir). Bootstrap sayısı taşıyan `Check`'ler `mc_tolerance` alır: uygulama ve Python kodu sayıyı birebir üretir, R ve Stata betikleri aynı sayıyı Monte Carlo toleransı içinde denetler.

## Model seçimi, düzenlileştirme ve DML (Konu 11–12)

`core/labs/penalized.py` hesabın tek kaynağıdır. `Penalized` işlemi cezasız EKK, Ridge, Lasso ve Elastic Net tahmin eder; sabit terim cezasızdır (veri eğitim ortalamasıyla merkezlenir). Ceza ölçekleri: Ridge Hansen'in SSE ölçeğinde (scikit-learn `Ridge(alpha)` ile aynı), Lasso ve Elastic Net yazılım ölçeğinde (`(1/(2n))‖Y − Xβ‖² + λ[r‖β‖₁ + (1 − r)/2‖β‖²]`; Hansen ölçeğinde 2nλ). Sürekli değişkenler eğitim verisinin ortalama ve (n'e bölünen) standart sapmasıyla ölçeklenir; kategori göstergeleri eğitimde görülen düzeylerden kurulur, ilk düzey referanstır. CV'de ölçekleme ve göstergeler her katta yeniden öğrenilir; ölçüt kat MSE'lerinin ağırlıksız ortalamasıdır, SH katlar arası standart sapma/√K. Izgara büyükten küçüğe sıralıdır; eşitlikte en büyük ceza, `1se` kuralında en küçük CV'nin bir SH'si içindeki ilk (en büyük) ceza seçilir.

Katlar ve bölmeler rastgele sayı üreteciyle değil açık kuralla kurulur (`RowNumber`, `GroupRank`, altın oran φ ile u = {iφ}; DML'de okul sıra numarası r ile ⌊K{rc}⌋ + 1): Python, R ve Stata aynı eğitim/test bölmesini, aynı katları ve aynı sayıları üretir. `CrossFitDML` DML2'yi (sabitsiz artık regresyonu, HC1 veya küme SH), `DMLSplits` bölme duyarlılığını ve Chernozhukov vd. (2018) medyan birleştirmesini, `DoubleSelection` yalnız-sonuç Post-Lasso ile double selection'ı, `ComplexityCurve` polinom derecesi boyunca eğitim MSE, LOOCV, AIC, BIC ve test MSE'yi hesaplar. Grafik işlemleri (`CVCurve`, `CoefPath`, `DotPlot`, `EstimatePlot`, `LinePlot`) `topics/selection_ui.py` ile uygulamada, `core/codegen/*_pen.py` ile üç dilde aynı renklerle çizilir.

Üreticiler aynı algoritmayı yazar: Python `sklearn.linear_model.enet_path` (sıcak başlangıç, tolerans 1e-12), R `glmnet` (`standardize = FALSE`, `intercept = FALSE`, merkezlenmiş veri, `thresh = 1e-16`, `glmnet.control(fdev = 0, devmax = 1)`), Stata do-dosyasındaki Mata koordinat inişi (Gram matrisiyle, her taramada KKT koşuluyla parlatma; Stata 14 yeterli). Ridge üç dilde kapalı biçimle (özdeğer ayrışımı) hesaplanır. Monte Carlo bloklarında EKK öğrenicili `CrossFitDML` ve `DrawColumns` de vektörleştirilir (döngüyle eşitliği testle denetlenir).

Konu 1–12 yeni yapıya taşındığı için `core/code_recipes.py` içindeki tarifler, `topics/shared.render_reproduction_code` ve eski `core/regularization.py`, `core/dml.py`, `core/pipelines.py`, `core/research_workflow.py` modülleri arayüzde kullanılmaz; kendi testleri sürdükçe depoda kalırlar ve ayrı bir temizlik adımında kaldırılabilir.

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
- Konu 07-08: Uygulama/Sezgi/Kendini sına sekmeleri; kesin kantil çözümünün, Hendricks–Koenker standart hatalarının ve kantil farkı testinin R `quantreg` ile eşitliği; LP çözümünün alt-gradyan koşulu; yerel doğrusal tahminin ağırlıklı EKK sabit terimi olması ve CV'nin kaba kuvvet hesapla eşitliği; Sezgi deneylerinin kuramsal değerleri (kuyruk kantili SH oranı, kalın kuyrukta LAD'ın OLS'e üstünlüğü, sınırda Nadaraya–Watson yanlılığı, doğrusal kontrolün yanlılığı ve Robinson).
- Konu 09-10: Uygulama/Sezgi/Kendini sına sekmeleri; RDD'nin statsmodels WLS ve R `lm` + `sandwich::vcovHC` ile eşitliği; eğrinin noktasal ağırlıklı EKK ile ve eşikte RDD limitleriyle eşitliği; eksik değerli gözlemlerin atılması ve LM2007'nin CSV olarak yüklenmesi (değişken adları küçük harfe çevrilir); bootstrap çekiliş sırasının (pairs, wild, küme) üretilen Python koduyla aynı olması; Sezgi varsayılanlarının notlardaki Tablo 10.1'i birebir üretmesi; Sezgi deneylerinin kuramsal değerleri (global polinomun eşik yanlılığı, yerel doğrusal yanlılığın ≈ −2,4h² olması, bulanık RDD'de Wald oranı ve zayıf ilk aşama, percentile aralığının monoton dönüşüme uyumu, kümeli veride pairs bootstrap'ın belirsizliği küçümsemesi).
- Konu 11-12: Uygulama/Sezgi/Kendini sına sekmeleri; Ridge, Lasso ve Elastic Net yolunun scikit-learn `Ridge`, `Lasso`, `ElasticNet` ile, CV'nin `Pipeline` + `GridSearchCV` + `PredefinedSplit` ile, artık regresyonunun statsmodels sabitsiz HC1 ve küme kovaryansıyla, AIC/BIC'in log-olabilirlikle ve LOOCV'nin açık dışarıda bırakma döngüsüyle eşitliği; Lasso KKT koşulu; çapraz uyarlamanın yalnız diğer katları kullanması ve artıklaştırmanın FWL ile eşitliği; R `glmnet` karşılığının notlardaki sayıları üretmesi; Sezgi varsayılanlarının notlardaki Tablo 11.1 ve 12.1'i birebir üretmesi; Sezgi deneylerinin kuramsal değerleri (eğitim hatasının monotonluğu, BIC'in AIC'den az parametre seçmesi, ön test sonrası kapsamanın düşmesi, naif ve ortogonal moment yanlılıkları, gözlenmeyen karıştırıcı altında DML yanlılığı κ²/(1 + κ²)).
- Kod tarifi testi: 12 konunun dört bölümünü kapsayan 48 Python betiğinin derlenmesi ve dış veri olmadan çalışması; Colab JSON sözleşmesi.
- Ek veri kaynakları (`test_lab_variants.py`, `test_lab_sources_ui.py`): notların kod özeti; alternatif örneğin notlarla aynı adımları, sayılarının bağımsız hesapla ve üretilen Python/R koduyla eşitliği, Stata kuralları; kendi verinde örnek dosya, Türkçe CSV (; ve ondalık virgül, cp1254), işaretli sütun adları, ayrılmış kod adları, tam uyum, logaritması alınamayan sonuç ve rol kurallarının iki dilde aynı sayıyı vermesi; AppTest ile kaynak seçimi, dosya yükleme ve seçimlerin korunması. Konu 5–8 için `test_lab_variants_konu05_08.py`: CHJ2004, CK1994, Card1995 (kantil) ve AL1999 alternatiflerinin bağımsız hesabı (statsmodels ve sayısal türevle AME, primal doğrusal programla LAD ve kantil regresyon, elle Hendricks–Koenker, genel amaçlı optimizasyonla Tobit, döngüyle ağırlıklı en küçük kareler CV'si) ve metin sayıları; kendi verinde ayrışma, tek olmayan LAD ve kantil çözümleri, az değerli sonuç, terim ölçekleri, iki CV'nin aynı h'yi seçmesi, seyrek uç değerler, çarpık açıklayıcı ve tam doğrusal sonuç kurallarının iki dilde aynı sayıyı vermesi. Konu 3–4 için ayrıca `test_lab_variants_konu03_04.py`: DS2004 ve AK1991 alternatiflerinin bütün kontrol değerlerinin ve metindeki sayılarının uygulamanın kodundan ayrı numpy hesabıyla (küme-dayanıklı ve HC1 sandviç, elle 2SLS) doğrulanması; `GroupMean`, kategorik kontrollü 2SLS ve `_slim` birim testleri; kendi verinde küme yokken HC1, az küme, birlikte seçilen roller, Adım 4'ün alt örneklemi, sıfır ilk aşama, zayıf araç, logaritmasız sonuç ve boş hücreli kontrol kurallarının iki dilde aynı sayıyı vermesi.

## Öğrenci kodu akışı

Bütün konularda öğrenci kodu laboratuvar ve deney tanımlarından üretilir (`core/codegen/`). Uygulama sekmesinde her adımın kodu ve bütün laboratuvar tek dosya olarak (Python, R, Stata) indirilir; dosya veriyi Hansen'in arşivinden indirir ve sonunda notlardaki sayılarla karşılaştırır. Alternatif örnekte betik aynı veriyle uygulamanın verdiği sayılarla, kendi verinde (Python ve R) öğrencinin dosyasını okuyup uygulamanın sayılarıyla karşılaştırır. Sezgi sekmesinde her deneyin kodu şu anki kaydırıcı değerleriyle üretilir.

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
uygulamada ve üretilen Python/R kodunda yeniden üretilmesi (Python betikleri Türkçe Windows kod sayfası
`PYTHONIOENCODING=cp1254` altında çalıştırılır; `.dta` verili konularda uygulama ve Python kodu, tamsayıları
Stata `compress` gibi en küçük türde saklanmış bir kopyayla da sınanır), soru setinde kavram tekilliği, bölüm kapsamı,
cevap anahtarı dengesi. Yeni bir konu kayda eklendiğinde ayrıca test yazmadan bu sözleşmelere tabidir.
