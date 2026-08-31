# IKT 807 Streamlit Uygulaması: Geliştirme Planı

Durum: Kullanıcı incelemesine hazır ilk plan
Tarih: 28 Ağustos 2026
Ders: IKT 807 - Ekonometrik Modelleme ve Uygulamaları
Program: İktisat Tezli Yüksek Lisans Programı

## 1. Amaç ve Başarı Ölçütü

Uygulama, ders notlarının veya sunumların ekrana taşınmış bir kopyası olmayacaktır. Her haftanın yaklaşık 30 dakikalık uygulama bölümünde öğrencinin aşağıdaki karar zincirini veri ve simülasyon üzerinde izlemesini sağlayacaktır:

> Araştırma sorusu -> estimand -> tanımlama varsayımları -> tahmin yöntemi -> belirsizlik -> tanı/duyarlılık -> ekonomik yorum

Başarılı ilk sürüm:

- 12 konunun her birinde 2-4 yüksek değerli ve birbiriyle ilişkili etkileşim sunar.
- Nokta tahmini ile çıkarımı, model uyumu ile tanımlamayı ve istatistiksel anlamlılık ile ekonomik önemi ayırır.
- Ders notlarındaki terminoloji, notasyon, örneklem tanımı ve yorumlama sınırlarını korur.
- Gerçek veri laboratuvarını kontrollü simülasyonla tamamlar; biri diğerinin yerine geçmez.
- Aynı seed ve ayarlarla aynı sonucu üretir; kullanılan veri, örneklem, model, kovaryans ve tuning tercihlerini görünür kılar.
- Saf hesaplama katmanı, sayısal referans testleri ve Streamlit AppTest kontrolleriyle sürdürülebilir olur.
- Streamlit Community Cloud üzerinde gizli anahtar veya çalışma zamanı LLM/API gerektirmeden çalışır.

## 2. Keşif Özeti ve Kaynak Hiyerarşisi

### 2.1 Yerel kaynaklar

Bağlayıcı içerik kaynakları:

1. `Ders Notları/bolumler/01_...tex` - `12_...tex`: 12 konunun güncel LaTeX kaynakları.
2. `Ders Notları/main.pdf`: 281 sayfalık, 26 Ağustos 2026 tarihli derlenmiş ders notu.
3. `Sunum dosyaları/sunumlar/konu01_...tex` - `konu12_...tex`: 12 sunumun güncel kaynakları.
4. `Yönerge/IKT807_Ders_Yonergesi_Revize.tex`: haftalık akış, öğrenme çıktıları ve 30 dakikalık uygulama çerçevesi.
5. `Ders Notları/UYGULAMA_SABLONU.md`: gerçek veri laboratuvarlarının ortak pedagojik omurgası.
6. `Ders Notları/SOURCE_DATA_NOTES.md`: Hansen veri kaynakları ve hazırlanmış öğretim kopyaları.
7. `Ders Notları/scripts/`, `tables/`, `figures/`: yeniden üretim betikleri ve sayısal referans çıktıları.

Kaynaklarda 24 Ağustos 2026 tarihli Hansen uyum denetimi yapılmış; ders notu ve sunumlarda 230 tablo değeri eşleştirilmiştir. Uygulama geliştirmesinde bu sürüm bağlayıcı kabul edilecektir.

### 2.2 Repo durumu

`C:\Ekonometri YL Dr` bir Git deposu değildir ve alt klasörlerde `.git` dizini bulunmamıştır. Ayrı bir IKT 807 uygulama deposu oluşturulmadan uygulama kodu yazılmamalıdır. IKT 305 referans deposu yalnız mimari örnektir; IKT 807 kodunun hedefi veya başlangıç dalı değildir.

Önerilen yeni yerel depo adı:

```text
C:\Ekonometri YL Dr\ikt807-ekonometri-uygulama
```

Bu ad ve varsa GitHub uzak depo adresi kullanıcı onayıyla kesinleşecektir.

### 2.3 Referans mimari

IKT 305 referans deposundaki ince `app.py`, Streamlit'ten bağımsız `core/`, konu bazlı `topics/`, merkezi veri kaydı, deterministik soru motoru ve AppTest yaklaşımı korunmaya değerdir. IKT 807'de bunlara aşağıdaki yeni sözleşmeler eklenmelidir:

- Estimand, tanımlama ve çıkarım metadata'sı taşıyan ortak sonuç nesneleri.
- Kümeleme ve yeniden örnekleme birimini veri metadata'sında zorunlu alan yapan kontroller.
- Bandwidth, lambda, fold, seed ve optimizasyon ayarlarını sonuçla birlikte saklayan tuning metadata'sı.
- Başarısız yakınsama, rank eksikliği, zayıf araç, az küme ve veri sızıntısı için görünür hata/uyarı durumları.

Referans: [IKT 305 deposu](https://github.com/burakalparslaneroglu/ekonometri-uygulama) ve [mimari belgesi](https://github.com/burakalparslaneroglu/ekonometri-uygulama/blob/main/docs/ARCHITECTURE.md).

## 3. Kapsam Kararları

### 3.1 İlk sürümde yapılacaklar

- 12 konu ve her konuda ortak bir öğrenme akışı.
- Kontrollü simülasyonlar ve ders notundaki beş gerçek veri ailesi.
- Plotly grafikler, okunabilir sonuç tabloları, varsayım ve duyarlılık panelleri.
- Ekrandaki güncel sonuca bağlı deterministik kısa sorular.
- Saf sayısal testler, kaynak-sonuç regresyon testleri ve seçilmiş AppTest senaryoları.
- Türkçe kullanıcı arayüzü; uluslararası terim ilk kullanımda parantez içinde.

### 3.2 İlk sürümde yapılmayacaklar

- Ders notunun tam metnini veya sunum slaytlarını uygulamaya kopyalamak.
- Serbest model formülü veya serbest Python kodu çalıştırmak.
- Runtime LLM, dış AI API veya kullanıcı hesabı gerektirmek.
- Panel veri ve fark-içinde-fark eklemek; bunlar ders kapsamı dışında bırakılmıştır.
- Konu sırasını aşan aktif öğretim yapmak. Örneğin Konu 03'te 2SLS algoritması, Konu 08'de DML cross-fitting ayrıntısı öğretilmez.
- `doubleml` veya `rdrobust` gibi özel paketleri yalnız yöntem adı nedeniyle otomatik eklemek.

## 4. A - Konu Matrisi

| Konu | Ana amaç | Etkileşimler | Veri | Core modülü | Temel test |
|---|---|---|---|---|---|
| 01. Ampirik Ekonometrik Modelleme ve Regresyonun Temeli | Koşullu ortalama, doğrusal projeksiyon ve OLS'nin farklı rollerini ayırmak | 1) Dağılım ve koşullandırma simülatörü.<br>2) OLS geometrisi: artık-uydurulan değer ortogonalliği.<br>3) CPS ücret-egitim koşullu ortalama ve çok sütunlu tablo okuma laboratuvarı. | CPS 2009 + kontrollü DGP | `ols.py`, `simulation.py`, `datasets.py` | Kapalı biçim OLS; normal denklemler; artık ortogonalliği; CPS referans katsayıları |
| 02. Doğrusal Regresyon ve Güvenilir Çıkarım | Ceteris paribus yorumunu, fonksiyonel biçimi ve güvenilir belirsizlik ölçümünü birlikte kurmak | 1) FWL artıklaştırma laboratuvarı.<br>2) Düzey/log/karesel/etkileşim yorum aracı.<br>3) Klasik-HC1-küme standart hata karşılaştırması.<br>4) Kaldıraç ve etkili gözlem duyarlılığı. | CPS 2009 + kontrollü heteroskedastik/kümeli DGP | `ols.py`, `inference.py`, `diagnostics.py`, `functional_forms.py` | FWL eşitliği; HC1 benchmark; delta yöntemi; leverage/Cook ölçüleri |
| 03. Tanımlama, Nedensellik ve İçsellik | Estimand, tanımlama, tahmin edici ve tahmini ayırmak; büyük örneklemin yanlılığı çözmediğini göstermek | 1) Potansiyel sonuçlar ve seçim ayrıştırması.<br>2) İçsellik korelasyonu ve örneklem büyüklüğü DGP'si.<br>3) Nedensel yol/koşullandırma senaryoları.<br>4) DDK rastgele atama ile seçilmiş alt grup karşılaştırması. | DDK2011 + kontrollü içsellik DGP | `causal.py`, `simulation.py`, `clustered_inference.py` | Bilinen DGP bias limiti; seed determinismi; DDK ham/ayarlı küme-SH sonucu |
| 04. Araçsal Değişkenler ve 2SLS | Uygunluk, dışsallık ve dışlama kısıtını ayırmak; Wald ve 2SLS zincirini görünür yapmak | 1) IV geçerlilik koşulları senaryo matrisi.<br>2) Wald oranı laboratuvarı.<br>3) İlk aşama -> indirgenmiş biçim -> 2SLS projeksiyonları.<br>4) Araç gücü Monte Carlo duyarlılığı ve Card çıktısı okuma. | Card1995 + kontrollü IV DGP | `iv.py`, `simulation.py`, `inference.py` | Wald=just-identified 2SLS; Card referans sonuçları; weak-IV target recovery; sıradan ikinci-aşama OLS SH'nin reddi |
| 05. İkili ve Ayrık Sonuç Modelleri | Model katsayısı, tahmin edilen olasılık ve marjinal etkiyi ayırmak | 1) LPM-Logit-Probit olasılık eğrileri.<br>2) Katsayı-AME-sonlu fark dönüşümü.<br>3) Yaş profili ve bireysel/ortalama marjinal etki.<br>4) Sonuç desteğine göre model ailesi karar akışı. | CPS 2009, 35 yaş ve altı çalışan erkekler | `discrete.py`, `marginal_effects.py`, `inference.py` | Logit/Probit olasılık ve AME benchmark; kukla sonlu fark; delta yöntemi |
| 06. Sansürleme, Kesilme ve Örneklem Seçimi | Gizli ve gözlenen sonuçları; sansürleme, kesilme ve seçimi ayırmak | 1) Gizli -> gözlenen sonuç sansürleme simülatörü.<br>2) Üç koşullu ortalama ve marjinal etki karşılaştırması.<br>3) OLS-Tobit-LAD gerçek veri profili.<br>4) Heckman seçim DGP'si ve dışlama kısıtı duyarlılığı. | CHJ2004 + kontrollü Tobit/Heckman DGP | `limited_outcomes.py`, `optimization.py`, `marginal_effects.py` | Tobit MLE yakınsama ve skor kontrolleri; bilinen DGP; CHJ örneklem/özet benchmark; seçim düzeltmesi |
| 07. Kantil Regresyon | Ortalama ve koşullu kantil estimand'larının farklı olduğunu göstermek | 1) Check-loss eğimi ve seçilen kantil.<br>2) OLS ile seçili kantil doğrularını karşılaştırma.<br>3) Tau boyunca katsayı profili ve güven aralıkları.<br>4) CPS makale tablosu okuma soruları. | CPS 2009 | `quantile.py`, `inference.py` | Check-loss minimizasyonu; statsmodels benchmark; CPS kantil katsayıları; seed'li bootstrap seçeneği |
| 08. Parametrik Olmayan ve Yarı Parametrik Regresyon | Esnekliğin tuning kararı olduğunu; bandwidth ve boyutluluk maliyetini göstermek | 1) Kernel ağırlıkları ve effective neighborhood.<br>2) Yerel sabit-yerel doğrusal sınır davranışı.<br>3) Bandwidth/CV duyarlılığı.<br>4) Doğrusal-seri/spline-yerel doğrusal ve partialling-out karşılaştırması. | DDK2011 tracking okullarındaki kız öğrenciler + kontrollü eğri | `nonparametric.py`, `cross_validation.py`, `partialling.py` | Kernel ağırlık toplamı; yerel doğrusal benchmark; h=6.2/12.3 referans eğrileri; leakage kontrolü |
| 09. Regresyon Süreksizliği Tasarımı | Yerel estimand'ı, eşik çevresindeki tanımlamayı ve bandwidth duyarlılığını birlikte okumak | 1) Sharp RDD DGP ve yerel doğrusal sıçrama.<br>2) Bandwidth/kernel/polinom duyarlılığı.<br>3) LM2007 Head Start grafik ve sonuç tablosu okuma.<br>4) Sharp-fuzzy ayrımı, yoğunluk ve placebo tanıları. | LM2007 Head Start + kontrollü sharp/fuzzy RDD | `rdd.py`, `local_polynomial.py`, `diagnostics.py` | Bilinen sıçrama recovery; LM h-grid referansları; eşik tarafı/yön testi; fuzzy Wald oranı |
| 10. Bootstrap ve Yeniden Örnekleme | Yeniden örnekleme birimini, Monte Carlo hatasını ve güven aralığı türlerini görünür yapmak | 1) Tekrar tekrar örneklem çekme animasyonu.<br>2) B ve seed ile Monte Carlo hatası.<br>3) Pairs-wild-parametrik bootstrap karşılaştırması.<br>4) Analitik/normal/percentile/percentile-t güven aralıkları. | CPS 5.000 gözlemlik sabit alt örneklem + kontrollü DGP | `resampling.py`, `bootstrap.py`, `inference.py` | Seed determinismi; B arttıkça MC hata davranışı; CPS B=1000 benchmark; yeniden örnekleme birimi doğrulaması |
| 11. Model Seçimi, Çapraz Doğrulama ve Düzenlileştirme | Tahmin amacı ile yapısal çıkarımı ayırmak; scaling, CV ve ceza yolunu doğru kurmak | 1) Yanlılık-varyans simülasyonu.<br>2) Ridge/Lasso katsayı yolları.<br>3) Kat içi standardizasyon ve lambda-min/1se CV görünümü.<br>4) CPS hold-out model karşılaştırması ve Post-Lasso. | CPS 20.000 gözlemlik sabit alt örneklem + kontrollü yüksek boyutlu DGP | `regularization.py`, `cross_validation.py`, `pipelines.py` | Ortonormal kapalı biçim; sklearn benchmark; leakage testi; CPS test MSE ve seçili değişken sayısı |
| 12. Double/Debiased Machine Learning ve Bütünleşik Araştırma Akışı | Yüksek boyutlu yardımcı modelleri düşük boyutlu hedef çıkarımından ayırmak; cross-fitting ve ortogonalizasyonu öğretmek | 1) Naif outcome-only seçim, double selection ve partialling-out.<br>2) Fold bazında out-of-fold tahmin ve veri sızıntısı görünümü.<br>3) DDK OLS-DML karşılaştırması; okul bazlı GroupKFold.<br>4) Fold/seed/learner duyarlılığı ve bütünleşik tez iş akışı. | DDK2011 + kontrollü n=800, p=180 DGP | `dml.py`, `partialling.py`, `cross_validation.py`, `research_workflow.py` | Bilinen theta recovery; yalnız OOF tahmin değişmezi; grup katı sızıntı testi; DDK OLS/DML benchmark; split-seed duyarlılığı |

## 5. B - Veri Matrisi

| Dataset | Kaynak | Gözlem birimi | Örneklem kısıtı | Konular | Commit edilebilir mi? |
|---|---|---|---|---|---|
| `cps09mar_ikt807.csv` (50.742 satır) | Hansen `cps09mar`; Mart 2009 CPS | Tam zamanlı çalışan birey | Ders betiğindeki ücret, çalışma süresi ve eksiksiz değişken filtreleri; Konu 5'te erkek ve yaş <=35; Konu 10'da seed 807 ile n=5.000; Konu 11'de seed 807 ile n=20.000 | 01, 02, 05, 07, 10, 11 | **Beklemede.** Kaynak indirmeye açık, fakat açık yeniden dağıtım lisansı bulunamadı. Yazılı onay/lisans teyidi olmadan public repoya konmayacak. |
| `DDK2011_ikt807.csv` (7.022 satır) | Duflo-Dupas-Kremer okul tracking verisi; Hansen kaynak paketi | Öğrenci; küme birimi okul | Sonuç için n=5.795; eksiksiz kovaryat için n=5.135; Konu 8'de tracking=1, girl=1 ve complete-case n=1.487 | 03, 08, 12 | **Beklemede.** Aynı lisans kapısı; okul kimliği ve kaynak atfı korunacak. |
| `Card1995_ikt807.csv` (3.010 satır) | Card kolej yakınlığı/eğitim IV verisi; Hansen kaynak paketi | Birey | `lwage76`, `ed76`, `nearc4` ve tanımlı kontrol setinde complete-case | 04 | **Beklemede.** Kaynak/lisans teyidi gerekli. |
| `CHJ2004_ikt807.csv` (8.684 satır) | Cox-Hansen-Jimenez hane transfer verisi; Hansen oluşturma dosyası | Hane | Düzeltilmiş gelirin üst %2'si ve negatif gelir çıkarılır; transfer sonucu >=0 | 06 | **Beklemede.** Kaynak/lisans teyidi gerekli; dönüşüm zinciri metadata'da saklanacak. |
| `LM2007_ikt807.csv` (2.810 satır) | Ludwig-Miller Head Start RDD; Hansen kaynak paketi | İlçe/coğrafi birim | Running variable `povrate60`, outcome `mort_age59_related_postHS`; h-grid etkin örneklemi her tahminde yeniden hesaplanır | 09 | **Beklemede.** Kaynak/lisans teyidi gerekli. |
| Kontrollü simülasyonlar | Uygulama içi, ders notundaki DGP'lerle uyumlu üretim | Simülasyona göre birey/okul/hane | Her senaryoda seed, n, parametreler ve bağımlılık yapısı görünür | 01-12 | Evet; uygulama kodunun parçası olarak üretilecek. |

Veri yayınlama kapısı için önerilen sıra:

1. Kaynak veri paketinin ve tekil veri setlerinin yeniden dağıtım koşullarını yazılı olarak doğrula.
2. İzin varsa yalnız gerekli sütunları ve öğretim örneklemlerini, kaynak ve dönüşüm metadata'sıyla commit et.
3. İzin yoksa public uygulama için resmi URL'den checksum doğrulamalı indirme/ön işleme veya ayrı izinli alternatif veri belirle.
4. Belirsizlik sürerken geliştirmeyi `references_private/` altındaki yerel kopyalarla yap; bu klasörü `.gitignore` ile koru.

Hansen'in resmi sayfası veri, program ve kaynak veri ziplerini indirime açmaktadır; sayfada açık bir yeniden dağıtım lisansı görünmemektedir: [Bruce Hansen - Econometrics Data Sets](https://users.ssc.wisc.edu/~bhansen/econometrics/).

## 6. C - Yöntem ve Dependency Matrisi

| Yöntem / ihtiyaç | Ana paket | Alternatif | Referans test | Risk ve karar |
|---|---|---|---|---|
| Streamlit UI ve state | `streamlit` | Yok | `st.testing.v1.AppTest` widget/state senaryoları | Sürüm pinlenecek; AppTest tek sayfayı çalıştırdığı için konu seçimi entrypoint üzerinden test edilecek. [Resmi AppTest belgesi](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest) |
| Veri ve matris hesabı | `pandas`, `numpy`, `scipy` | Yok | Kapalı biçim ve ders tablosu benchmarkları | Ortak alt sınır/üst sınır pinleri; float toleransları açık yazılacak. |
| OLS, robust/küme çıkarım, GLM, Logit/Probit, QuantReg | `statsmodels` | Gerekli küçük formüller NumPy | Statsmodels + kapalı biçim karşılaştırması | Kovaryans ve finite-sample correction varsayılanları sonuç metadata'sında açık olacak. |
| IV/2SLS ve IV kovaryansı | `linearmodels` | Şeffaf NumPy 2SLS + sandviç implementasyonu | Card tablosu, Wald eşitliği, simüle DGP | `linearmodels` tercih edilir; robust/clustered ve `debiased` seçimi görünür. [IV2SLS fit belgesi](https://bashtage.github.io/linearmodels/iv/iv/linearmodels.iv.model.IV2SLS.fit.html) |
| Tobit MLE | `scipy.optimize` + saf likelihood | `statsmodels.base.model.GenericLikelihoodModel` | Simüle Tobit recovery, sayısal gradient, CHJ benchmark | Statsmodels'ta kararlı public Tobit API'sine güvenilmeyecek. Yakınsama, tolerans, iterasyon ve sigma pozitifliği zorunlu raporlanacak. |
| Heckman iki aşama | `statsmodels` Probit + OLS | Saf formül | Simüle seçim DGP ve rho=0 özel durumu | İlk sürümde öğretim amaçlı iki aşama; full-information MLE yok. Dışlama kısıtı görünür. |
| Marjinal etkiler / delta yöntemi | `statsmodels` + saf Jacobian | Sonlu fark doğrulaması | Analitik-sonlu fark eşitliği, CPS AME | Değerlendirme noktası ve kukla değişkende sonlu fark zorunlu metadata. |
| Yerel doğrusal/kernel/RDD | NumPy/SciPy saf ağırlıklı LS | `statsmodels.nonparametric` | DDK eğrileri, LM h-grid, bilinen sıçrama DGP | Öğretim şeffaflığı için küçük implementasyon; tek noktada ortak local-polynomial çekirdeği kullanılacak. |
| Bootstrap / Monte Carlo | NumPy `Generator` + statsmodels | SciPy bootstrap yalnız basit istatistiklerde | Seed determinismi, CPS B=1000 benchmark | Yeniden örnekleme birimi API'de zorunlu; pahalı sonuçlar parametre+seed ile cache edilecek. |
| Ridge/Lasso/Elastic Net/CV/pipeline | `scikit-learn` | Saf kapalı biçim yalnız referans testinde | Ortonormal çözüm, leakage, CPS test MSE | Standardizasyon ve kodlama fold içinde; test seti tuning'e giremez. |
| DML / cross-fitting / Random Forest | `scikit-learn` + saf ortogonal skor | `doubleml` daha sonra değerlendirilebilir | OOF değişmezi, bilinen theta, DDK GroupKFold | İlk sürümde özel paket yok; mekanizma şeffaf tutulacak. Kat, seed, learner ve tuning metadata'sı zorunlu. |
| Grafik | `plotly` | Streamlit native tablo/grafik | Eksen/legend kaynak testleri; görsel smoke | Grafik verisi core'da, figür kurma topics/UI katmanında. |
| Test | `pytest`, `pytest-cov` (dev), Streamlit AppTest | Playwright yalnız manuel/görsel kabul için | Sayısal, kaynak tutarlılığı, smoke, AppTest | Ağır Monte Carlo testleri küçük deterministic parametrelerle; yavaş replikasyon testleri ayrı marker. |

İlk `requirements.txt` için önerilen çekirdek: Streamlit, pandas, NumPy, SciPy, statsmodels, Plotly, scikit-learn ve linearmodels. Kesin sürüm aralıkları foundation dalında Python 3.12 ve Streamlit Cloud kurulumu denenerek kilitlenecektir.

## 7. Önerilen Mimari

```text
ikt807-ekonometri-uygulama/
├── app.py
├── AGENTS.md
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── .streamlit/
│   └── config.toml
├── assets/
│   └── styles.css
├── core/
│   ├── app_config.py
│   ├── types.py
│   ├── data_registry.py
│   ├── datasets.py
│   ├── metadata.py
│   ├── formatters.py
│   ├── simulation.py
│   ├── question_engine.py
│   ├── session_utils.py
│   ├── ui_preferences.py
│   ├── ols.py
│   ├── inference.py
│   ├── diagnostics.py
│   ├── causal.py
│   ├── iv.py
│   ├── discrete.py
│   ├── limited_outcomes.py
│   ├── quantile.py
│   ├── nonparametric.py
│   ├── rdd.py
│   ├── resampling.py
│   ├── regularization.py
│   └── dml.py
├── topics/
│   ├── konu01_ampirik_modelleme.py
│   ├── ...
│   └── konu12_dml_arastirma_akisi.py
├── tests/
│   ├── conftest.py
│   ├── test_data_registry.py
│   ├── test_numerical_*.py
│   ├── test_topic_contracts.py
│   └── test_app_*.py
├── docs/
│   ├── ARCHITECTURE.md
│   └── IKT807_APPLICATION_PLAN.md
└── references_private/         # .gitignore içinde
    ├── ders_notlari/
    └── sunumlar/
```

Yalnız gerçekten kullanılan modüller oluşturulacaktır. Yukarıdaki yöntem dosyaları büyümeden önce küçük tutulabilir; ortak bir davranış en az iki yerde gerçek tekrar üretmeden yeni soyutlama eklenmez.

### 7.1 Ortak sonuç sözleşmeleri

`core/types.py` içinde frozen dataclass veya eşdeğer tiplerle şu yapılar önerilir:

- `DatasetMetadata`: kaynak, gözlem birimi, örneklem tanımı, değişken/birim açıklamaları, izinli konu/model eşleşmeleri, cluster/resampling birimi.
- `EstimandMetadata`: hedefin adı, matematiksel/yalın dil tanımı, hedef anakütle, nedensel yorum koşulu.
- `InferenceSpec`: kovaryans türü, küçük örneklem düzeltmesi, cluster alanı, güven düzeyi.
- `TuningSpec`: seed, bandwidth, kernel, lambda, fold sayısı, split birimi, standardizasyon ve optimizasyon toleransı.
- `ModelResult`: ham sayısal sonuç, standart hata/GA, tanılar, uyarılar ve yukarıdaki metadata bağlantıları.

UI yalnız biçimlendirilmiş bir tabloyu değil, bu sözleşmeden üretilen başlık, dipnot ve uyarıları kullanmalıdır. Böylece “HC1 seçildi ama ekranda classic yazıyor” türü kaynak-UI ayrışmaları test edilebilir.

### 7.2 `app.py` sınırı

`app.py` yalnız:

- `st.set_page_config`,
- ortak CSS ve metin ölçeği,
- sidebar konu seçimi,
- aktif konu state senkronizasyonu,
- `TOPIC_RENDERERS[topic_id]()` çağrısı

içermelidir. Veri hazırlama, tahmin, simülasyon veya sonuç biçimlendirme mantığı `app.py` içinde bulunmamalıdır.

## 8. UI ve Pedagojik Akış

### 8.1 Ortak konu düzeni

Her konu sayfası aynı bilişsel ritmi kullanır:

1. **Araştırma sorusu ve hedef:** kısa, konuya özgü problem ve estimand.
2. **Mekanizma:** bir veya iki kontrollü etkileşim; öğrenci varsayımı değiştirir.
3. **Veri laboratuvarı:** kaynak, örneklem, model ve sonuç metadata'sı birlikte görünür.
4. **Tanı ve duyarlılık:** yönteme özgü ayar/varsayım değişince sonuç izlenir.
5. **Makale çıktısını okuma:** katsayıdan önce hangi satırların okunacağı gösterilir.
6. **Uygulama sorusu:** güncel sonuçtan üretilen deterministik soru ve gizli cevap.

Konu içinde sekmeler yalnız bu ana görünümleri ayırmak için kullanılabilir: `Mekanizma`, `Veri laboratuvarı`, `Duyarlılık`, `Çıktı okuma`. Sekme içeriği birbirini tekrar etmemelidir.

### 8.2 Görsel dil

Yerel ders notu ve sunumlarla uyumlu açık tema:

- Ana koyu renk: `#07373D`
- Derin teal: `#0C5B65`
- Etkileşim rengi: `#107C89`
- Vurgu teal: `#15A4B5`
- Olumlu/uygun: `#2F9E6B`
- Uyarı/hata: `#B3392F`

Renk tek bilgi kanalı olmayacak; işaret, çizgi biçimi, etiket ve açıklama ile desteklenecektir. Grafiklerde eksen adı, birim, legend ve veri/model bağlamı zorunludur. Ham teknik alanlar (`beta_hat`, `resid`, `tau_hat`, `mse_cv`) UI'da Türkçe etiketle gösterilir. Küçük p-değerleri `p < 0.001` biçimindedir.

Metin ölçeği `%100`, `%110`, `%120`, `%130` seçenekleriyle tek CSS değişkeninden uygulanır. Plotly yazı boyutları aynı ölçeğin kontrollü karşılığını kullanır; `em` ile iki kez ölçekleme yapılmaz.

### 8.3 Soru motoru

- Soru güncel veri/model/tuning kimliğine bağlıdır.
- Sıra deterministiktir; font ölçeği veya salt görünüm değişikliği soruyu değiştirmez.
- `[Cevabı göster / gizle]` ve `[Yeni soru]` ortak bileşendir.
- Yeni soru cevabı kapatır.
- Cevap metni soru oluşturulurken UI'a verilmez; session state'te kontrollü tutulur.
- Her soru estimand, varsayım, çıktı okuma veya duyarlılık kategorilerinden birine sahiptir.

## 9. Sayısal Doğruluk Sözleşmeleri

1. **OLS:** Robust standart hata katsayıyı değiştirmez ve içselliği çözmez.
2. **IV:** Uygunluk, dışsallık ve dışlama ayrı gösterilir. İkinci aşamanın sıradan OLS standart hatası hiçbir ekranda 2SLS standart hatası diye kullanılmaz.
3. **Logit/Probit:** Katsayı olasılık etkisi değildir. Marjinal etkinin değerlendirme noktası ve kukla değişkende sonlu fark belirtilir.
4. **Tobit:** Gizli sonuç, sansürlenmeme olasılığı ve gözlenen koşullu ortalama ayrıdır. Tobit katsayısı gözlenen Y üzerindeki marjinal etki diye etiketlenmez.
5. **Kantil regresyon:** `tau` hedef kantili belirler. OLS ile QR farklı estimand'lar olabilir; “hangisi doğru?” dili kullanılmaz.
6. **Parametrik olmayan yöntemler:** Bandwidth/baz/tuning kararı ve sınır davranışı görünürdür.
7. **RDD:** Tahmin yereldir; cutoff, yön, kernel, bandwidth, polinom derecesi ve etkin örneklem raporlanır. Sharp/fuzzy ayrımı zorunludur.
8. **Bootstrap:** Yeniden örnekleme birimi, B ve seed raporlanır. Bootstrap tanımlama sorununu çözmez.
9. **Düzenlileştirme:** Scaling ve değişken türetme fold içinde yapılır; test seti tuning için kullanılmaz. Tahmin başarısı yapısal çıkarım değildir.
10. **DML:** Yardımcı modeller yalnız eğitim katında fit edilir; her gözlem out-of-fold tahmin alır. Cross-fitting, ortogonal skor ve split-seed duyarlılığı görünürdür. DML gözlenmeyen içselliği çözmez.

Bu sözleşmeler hem metin testlerine hem sayısal testlere dönüştürülecektir.

## 10. Test Stratejisi

### 10.1 Saf sayısal testler

- OLS kapalı biçim, FWL ve projection değişmezleri.
- HC0-HC3, cluster ve delta yöntemi benchmarkları.
- Wald, just-identified IV ve 2SLS sandviç kovaryansı.
- Logit/Probit olasılık, AME ve sonlu fark.
- Tobit likelihood, gradient/yakınsama ve bilinen DGP recovery.
- Quantile check-loss ve katsayı benchmarkı.
- Kernel ağırlıkları, yerel doğrusal ve bandwidth grid.
- Sharp/fuzzy RDD sıçrama recovery.
- Bootstrap seed/B/re-sampling-unit davranışı.
- Ridge/Lasso kapalı biçim, CV leakage ve test ayrımı.
- DML out-of-fold, GroupKFold ve theta recovery.

### 10.2 Kaynak tutarlılığı testleri

- 12 başlık `app_config` ve topic badge'lerinde birebir aynı.
- Her topic `render()` fonksiyonuna sahip.
- Her grafik en az X/Y etiketi ve gerektiğinde legend içeriyor.
- Her sonuç tablosu veri, örneklem, estimand ve inference/tuning metadata'sına bağlanıyor.
- Ders notu referans tabloları açık toleransla (`pytest.approx`) yeniden üretiliyor.
- UI'da yasaklı ham teknik adlar ve yanlış nedensel kalıplar bulunmuyor.

### 10.3 AppTest ve görsel kabul

Foundation sonrasında en az Konu 01, 04, 06, 09 ve 12 için tam AppTest; diğer tüm konular için smoke test yazılır. Uygulama tamamlandığında tüm konuların temel widget etkileşimi AppTest kapsamına alınır.

Manuel kabul iki masaüstü ve bir dar ekran genişliğinde, `%100` ve `%130` ölçeklerde yapılır. Kontrol listesi: başlık/sidebar, widget state, uzun matematik, grafik eksenleri/legend, tablo taşması, metric etiketleri, soru düğmeleri, uyarı metinleri, hesaplama süresi ve hata durumları.

## 11. Performans ve Yeniden Üretilebilirlik

- Tüm rassallık `np.random.default_rng(seed)` üzerinden geçer.
- Varsayılan ders seed'i `807`; konuya özgü sabitler metadata'da açıkça tanımlanır.
- Bootstrap, Monte Carlo, kantil profili, RDD h-grid, CV/path ve DML cross-fitting sonuçları `@st.cache_data` ile önbelleklenir.
- Cache anahtarı veri kimliği/versiyonu, örneklem filtresi, model, tüm tuning parametreleri ve seed'i kapsar.
- Etkileşim sırasında hedef süre: hafif kontroller <0,5 sn; tek model <1 sn; ağır grid/yeniden örnekleme ilk çalıştırma tercihen <5 sn.
- Bootstrap tekrar sayısı UI'da pedagojik aralıkla sınırlanır; düşük B'de Monte Carlo uyarısı verilir.
- Sonuç metadata'sı en az veri kaynağı, örneklem kısıtı, formül, kovaryans türü, seed, bandwidth/lambda/fold ve yazılım sürümlerini içerir.

## 12. D - Branch Matrisi

| Branch | Konular | Core | Topics | Test | Manuel kontrol |
|---|---|---|---|---|---|
| `chore/ikt807-foundation` | Ortak kabuk | config, types, data registry, formatters, session, question engine, CSS | Topic placeholder sözleşmeleri; geniş içerik yok | Registry, topic contract, AppTest shell, text-scale | Sidebar, `%100/%130`, dar ekran, hata durumu |
| `feature/konu01-02-regression` | 01-02 | OLS, inference, diagnostics, functional forms, CPS adapter | Konu 01-02 | OLS/FWL/HC/cluster/delta + AppTest 01-02 | CPS grafik/tablo, ölçek, leverage, soru state |
| `feature/konu03-04-identification-iv` | 03-04 | causal DGP, cluster inference, IV | Konu 03-04 | Bias limit, DDK benchmark, Wald/2SLS, weak IV + AppTest 04 | Cluster etiketi, ilk aşama, yanlış SE engeli |
| `feature/konu05-06-discrete-limited` | 05-06 | discrete, marginal effects, Tobit/Heckman | Konu 05-06 | AME/delta, Tobit/selection DGP, CHJ benchmark + AppTest 06 | Probability/AME labels, latent/observed ayrımı, convergence |
| `feature/konu07-08-quantile-nonparametric` | 07-08 | quantile, local regression, CV, partialling | Konu 07-08 | Check-loss, QR benchmark, kernel/local-linear/CV | Tau profili, bandwidth, sınır davranışı, yavaşlık |
| `feature/konu09-10-rdd-bootstrap` | 09-10 | RDD/local polynomial, resampling/bootstrap | Konu 09-10 | RDD recovery/LM grid, bootstrap determinism/benchmark + AppTest 09 | Cutoff/yön, etkin n, B/seed, resampling unit |
| `feature/konu11-12-regularization-dml` | 11-12 | pipelines, regularization, DML, research workflow | Konu 11-12 | Leakage, path/CV, OOF/GroupKFold/theta + AppTest 12 | Lambda/fold metadata, learner/seed sensitivity, workflow |
| `fix/final-application-audit` | 01-12 | Yalnız doğrulamada bulunan düzeltmeler | Tüm konular | Tam pytest, compileall, diff-check, tüm topic smoke | İki masaüstü + dar ekran, `%100/%130`, performans, içerik denetimi |

Her branch sonunda zorunlu doğrulama:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall app.py core topics tests
git diff --check
```

Ardından canlı kontrol:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## 13. Foundation Dalı Kabul Kriterleri

Foundation tamamlanmış sayılmadan önce:

- Ayrı IKT 807 repo ve uzak depo doğrulanmış olmalı.
- `references_private/` ve yerel veri kaynakları `.gitignore` ile korunmalı.
- 12 kesin başlık tek `APP_CONFIG`/registry kaynağından gelmeli.
- `app.py` yalnız kabuk ve yönlendirme içermeli.
- Ortak metadata ve sonuç tipleri en az bir sahte konu ile test edilmeli.
- Metin ölçeği `%100-%130` ve soru state bağımsızlığı test edilmeli.
- Veri lisans kapısının durumu README'de açık olmalı; belirsiz veri public commit'e girmemeli.
- Python 3.12 ortamında runtime ve dev bağımlılıkları kurulmalı.
- Streamlit Community Cloud için import ve cold-start smoke testi yapılmalı.

## 14. Açık Kararlar ve Önerilen Varsayımlar

| Karar | Öneri | Neden |
|---|---|---|
| Yeni repo konumu/adı | `C:\Ekonometri YL Dr\ikt807-ekonometri-uygulama` | IKT 305'ten ve özel ders kaynaklarından fiziksel olarak ayrılır. |
| Veri yayınlama | Lisans teyidine kadar public commit yok | Resmi sayfa indirme sağlıyor, fakat açık yeniden dağıtım lisansı göstermiyor. |
| IV implementasyonu | `linearmodels` ana sonuç + küçük şeffaf NumPy öğretim hesabı | Güvenilir kovaryans ve tanılarla projeksiyon sezgisini birlikte sağlar. |
| Tobit | SciPy tabanlı şeffaf MLE | Kararlı public Tobit API'sine bağımlılığı önler; ders likelihood'ı görünür kalır. |
| Heckman | İlk sürümde iki aşama ve simülasyon | 30 dakikalık laboratuvar kapsamına uygun; full MLE ek risk yaratır. |
| RDD | Ortak yerel doğrusal çekirdek; `rdrobust` yok | Dersin bandwidth/kernel mantığı şeffaf kalır; benchmark testleriyle doğrulanır. |
| DML | sklearn + açık cross-fitting; `doubleml` yok | OOF tahmin, grup katı ve ortogonal skor öğrenciden saklanmaz. |
| Konu 5 kapsamı | LPM/Logit/Probit aktif lab; multinomial/ordered/count karar akışında | Ders notu kapsamı korunur, fakat 30 dakikalık laboratuvar aşırı dağılmaz. |

## 15. İlk Uygulama Adımı

Bu plan kullanıcı tarafından onaylandıktan sonra yalnız `chore/ikt807-foundation` dalı açılacaktır. İlk uygulama turu konu hesaplamalarına başlamaz; repo iskeleti, bağımlılıklar, ortak metadata sözleşmeleri, veri lisans kapısı, shell, metin ölçeği, soru motoru ve test altyapısını kurar.

Foundation PR/dal incelemesi tamamlanmadan Konu 01-12'nin geniş implementasyonuna geçilmez.
