# IKT 807 Ekonometrik Modelleme ve Uygulamaları

İzmir Bakırçay Üniversitesi İktisat Tezli Yüksek Lisans Programı için geliştirilen Türkçe, etkileşimli Streamlit laboratuvarıdır. Uygulama yöntemleri komut listesi olarak değil, araştırma sorusu -> tahmin hedefi -> tanımlama -> tahmin -> çıkarım -> duyarlılık -> ekonomik yorum zinciri içinde ele alır.

## Güncel kapsam

Yeni yapıdaki her konu üç sekmeden oluşur:

- **Uygulama:** ders notlarındaki uygulama laboratuvarını adım adım, gerçek Hansen verisiyle yeniden üretir. Her adımda notlardaki tablo ve şekiller, notlarla sayı sayı karşılaştırma ve seçilen dilde (Python, R, Stata) kod bulunur. Son adımda bütün laboratuvar tek dosya olarak indirilir; dosya sonunda sonuçları notlardaki basılı değerlerle karşılaştırır. Ek veri kaynakları olan konularda sekmenin üstünde üç seçenek vardır (aşağıda).
- **Sezgi:** veri üretim süreci (DGP) bilinen kontrollü deneyler. Her deney aynı sırayı izler: soru, DGP ve parametreleri, neye bakıyoruz, sonuç, ne gördük ve seçilen dilde kod. Gerçek veride görülemeyen nesneler (ör. gerçek koşullu ortalama) tahminlerle yan yana gösterilir.

- **Kendini sına:** haftanın kavramlarını sınayan soru seti. Dört tür: çoktan seçmeli, doğru–yanlış, boşluk doldurma ve denklem yazma. Her soru tek bir kavramı sınar ve notlardaki bir bölüme bağlıdır; yanlış cevaplara göre tekrar edilecek bölümler listelenir. Denklem sorularında yazılan formül anında önizlenir ve doğru cevaba eşdeğerliği sayısal olarak sınanır (`100(exp(b)-1)` ile `100*exp(b)-100` aynı kabul edilir).

Yeni yapı **bütün konular (1–12)** için hazırdır:

| Konu | Uygulama (notlardaki sayılar) | Sezgi deneyleri | Kendini sına |
|---|---|---|---|
| 1 | §1.13, CPS, 39 kontrol | koşullu ortalama ve OLS doğrusu; artıkların cebirsel özellikleri; OLS eğimi ve nedensel etki | 25 soru |
| 2 | §2.13, CPS, 19 kontrol: klasik/HC1 SH, Breusch–Pagan, doğrusal birleşim, delta yöntemi | heteroskedastisite; FWL ve eksik değişken; kümelenmiş veri | 24 soru |
| 3 | §3.15, DDK2011, 26 kontrol: denge tablosu, ham ve kovaryat ayarlı tracking etkisi (okul-kümeli), seçilmiş alt grup | seçim ayrıştırması; ölçüm hatası; büyük örneklem ve içsellik (Monte Carlo) | 24 soru |
| 4 | §4.17, Card1995, 14 kontrol: OLS, ilk aşama, indirgenmiş biçim, 2SLS, ilk aşama F, Wald oranı, elle ikinci aşama | Wald oranı ve dışlama kısıtı; zayıf araç (Monte Carlo); LATE | 24 soru |
| 5 | §5.15 ve §5.9 Tablo 5.1, CPS, 58 kontrol: LPM, Logit, Probit, AME (türev ve sonlu fark), ölçek çarpanı, yaş profili | LPM ve Logit; ölçek normalizasyonu; kukla değişkende türev ve sonlu fark | 24 soru |
| 6 | §6.16, CHJ2004, 44 kontrol: veri zinciri, spline'lı OLS/LAD/Tobit profilleri, Tobit'in üç hedefi ve model kontrolü | sansürleme (Tablo 6.2); Heckman iki aşama (Tablo 6.3); dışlama kısıtı olmadan Heckman (Monte Carlo) | 24 soru |
| 7 | §7.16, CPS, 45 kontrol: beş kantilde kesin kantil regresyon, Hendricks–Koenker SH, OLS karşılaştırması, katsayı profili, kantiller arası fark testi | konum ve ölçek: kantil doğruları; kalın kuyruklu hata: OLS ve LAD (Monte Carlo); kuyruk kantillerinde belirsizlik (Monte Carlo) | 24 soru |
| 8 | §8.13, DDK2011, 18 kontrol: doğrusal model, birini-dışarıda-bırak ve okul-kümeli CV ile bant genişliği, kübik polinom, kübik spline, yerel doğrusal tahminler | bant genişliği ve yanlılık–varyans; sınırda Nadaraya–Watson ve yerel doğrusal; kısmen doğrusal model: doğrusal kontrol ve Robinson (Monte Carlo) | 24 soru |
| 9 | §9.15, LM2007, 37 kontrol: tasarım satırları, beş bant genişliğinde keskin RDD (birim varyanslı üçgen çekirdek, HC1), eşikteki iki limit, dikdörtgen çekirdek ve bant genişliği ölçeği, güven bantlı RDD grafiği | yerel doğrusal ve global polinom (Monte Carlo); bant genişliği, yanlılık ve kapsama (Monte Carlo); bulanık RDD: yerel Wald oranı ve zayıf ilk aşama (Monte Carlo) | 24 soru |
| 10 | §10.17, CPS (50.742 gözlem), 6 kontrol: HC1 ve 1.000 tekrarlı pairs bootstrap (SH, percentile aralığı) | heteroskedastik regresyon: analitik, pairs ve wild bootstrap (Tablo 10.1); doğrusal olmayan dönüşüm: delta yöntemi ve percentile aralığı; kümeli veride yeniden örnekleme birimi | 24 soru |
| 11 | §11.15, CPS (50.742 gözlem; açık kuralla 38.055 eğitim / 12.687 test), 12 kontrol: basit ve zengin OLS, 5-katlı CV ile Ridge ve Lasso, test MSE, sıfırdan farklı katsayı, seçilen ceza ve ölçeği, CV eğrileri | Ridge, Lasso, Elastic Net ve Post-Lasso (Tablo 11.1); polinom derecesi: eğitim hatası, LOOCV, AIC, BIC ve test hatası; ön test sonrası güven aralığının kapsaması (Monte Carlo) | 24 soru |
| 12 | §12.15, DDK2011 (108 okul, 5.135 öğrenci), 15 kontrol: ham fark, kovaryat ayarlı OLS, 27 terimli sözlükte Lasso yardımcı modelli ve okul-kümeli çapraz uyarlamalı DML, 11 kat kuralında duyarlılık ve medyan birleştirme | yalnız sonuç seçimi, double selection, artıklaştırma ve DML (Tablo 12.1); ortogonal moment: yardımcı parametre hatasına duyarlılık (Monte Carlo); gözlenmeyen karıştırıcı altında DML (Monte Carlo) | 24 soru |

Kod dili kenar çubuğundan bir kez seçilir ve bütün konularda geçerlidir.

### Uygulama sekmesinde veri kaynağı

Notlardaki örnek ders sırasında notlarla aynı sayıları gösterdiği için, aynı adımlar iki ek kaynakla da yapılabilir.
Sekmenin üstündeki seçim varsayılan olarak notlardaki örnektir; notların laboratuvarı ve üretilen kodu değişmez
(üç dildeki bütün kodun özeti testle kilitlidir).

- **Notlardaki örnek:** yukarıdaki laboratuvar.
- **Alternatif örnek:** aynı adımlar, aynı işlemler ve aynı kod üreticisi; veri Hansen'in arşivindeki başka bir gerçek
  veri setidir ve notlardaki gibi çalışma anında indirilir ya da yüklenir. Metinler bu verinin sayılarıyla yazılmıştır;
  sayılar testlerde bağımsız bir hesapla ve üretilen Python/R koduyla doğrulanır. İndirilen betik (Python, R, Stata)
  sonunda uygulamanın aynı veriyle verdiği değerlerle karşılaştırır.
- **Kendi verini yükle:** aynı adımlar öğrencinin Excel (.xlsx) ya da CSV dosyasıyla kurulur. Öğrenci sütunları
  rollere atar (ör. sonuç, açıklayıcı değişken, grup), sonucun logaritmasının alınıp alınmayacağını seçer; yorumlar
  sonuçlardan yazılır. Kod Python ve R'da üretilir: dosyayı okuma ve temizleme kuralları (boşluklar, "NA", ondalık
  virgül, Türkçe CSV, Excel'de metin olarak saklanmış sayılar) iki dilde birebir aynıdır ve betik uygulamanın
  sayılarıyla karşılaştırır. İki dilin aynı sayıyı veremeyeceği seçimler açık bir iletiyle reddedilir: tam doğrusal
  bağlantı, neredeyse doğrusal bağlantı (ör. dar aralıkta yıl ve karesi; iletide yaşı ya da `yıl − 2015` gibi bir
  farkı kullanma önerilir) ve kötü ölçek (ör. milyonlarla ölçülen bir kontrolün karesi; iletide birimi değiştirme
  önerilir). Konu 3'te kümelenmiş standart hata için her tahmin örnekleminde (denge satırları, kovaryat ayarlı model,
  Adım 4) en az 4 küme gerekir. Konu 4'te kontroller sabitken endojen değişkenle ilişkisiz araç (sıfır ilk aşama) ve
  araçla kontrollerin tam doğrusal fonksiyonu olan endojen değişken (tam uyumlu ilk aşama) reddedilir; ilk aşama F 10'un
  altındaysa yorum zayıf aracı belirtir. Konu 5'te Logit'i sonsuza götüren tam ya da yarı-tam ayrışma, Konu 6'da sıfır
  payı %5–50 dışındaki sonuç ve Tobit'in tahmin edilemediği seçimler reddedilir; LAD çözümü gösterim basamağında tek
  değilse LAD eğrisi gösterilir ama kodda karşılaştırılmaz. Konu 7'de sonucun en az 10 farklı değeri olmalıdır
  (az değerli sonuçta Hendricks–Koenker standart hatası hesaplanamaz); bir kantilde doğrusal programın çözümü gösterim
  basamağında tek değilse o modelin kontrolleri karşılaştırılmaz ve yorum bunu söyler. Konu 8'de en çok 2.500 gözlem
  kullanılır (CV n×n ağırlık matrisiyle); tam doğrusal sonuç reddedilir; CV ızgarasının alt ucu, dışarıda bırakılan
  bütün tahminlerin ve grafik eğrisinin sayısal olarak hesaplanabildiği en küçük h'ye yükseltilir (uçtaki seyrek
  değerler) ve iki CV aynı h'yi seçerse karşılaştırma bandı ana bandın iki katıdır. Konu 9'da eşik değeri ve isteğe
  bağlı ana bant genişliği metin alanına yazılır (ondalık virgül ya da nokta; binlik de olabilecek "5.000" gibi bir
  yazım reddedilir); eşik, eşik değişkeninin gözlenen aralığında olmalıdır ve en dar pencerede eşiğin iki yanında en az
  10 gözlem ile 3 farklı değer, küme seçildiyse en az 10 küme bulunmalıdır; eşik değişkeni değişince eşik alanı yeni
  sütunun medyanına döner, bant genişliği alanı boşalır. Konu 10'da küme bootstrap'ı için
  en az 20 küme gerekir; 1.000 tekrarın birinde bile hedef katsayı tanımsız kalıyorsa (ör. seyrek iki değerli
  açıklayıcı bir tekrarda sabit) seçim reddedilir. Konu 11'de zengin model en çok 200 terim içerebilir ve eğitim
  örnekleminde en az 2·(terim sayısı + 1) gözlem ister; eğitim örnekleminde ya da bir CV katının eğitim kısmında sabit
  kalan ya da önceki terimlerin tam doğrusal birleşimi olan sözlük terimleri çıkarılır (metin bunu söyler);
  Ridge ızgarası eğitim gözlemi sayısından, Lasso ızgarası bütün katsayıları sıfır yapan en küçük cezadan kurulur.
  Konu 12'de en çok 5.000 gözlem ve 100 sözlük terimi kullanılır; küme seçilirse dış ve iç katlar ile standart hata
  kümeye göre kurulur (en az 20 küme). Örnek dosya kurgusal veriden bellekte üretilir (`core/labs/ornek_veri.py`: ücret
  verisi, okul deneyi, araç değişkeni, iş arama, bağış, çalışma saati, burs, sınıf, konut ve mesleki kurs verisi;
  DGP'ler dosyada yazılıdır). Okuma kuralları İKT 217 ve İKT 305 uygulamalarıyla aynıdır.

| Konu | Alternatif örnek | Kendi verinde roller |
|---|---|---|
| 1 | Card (1995), NLSYM 1976: 3.010 genç erkek; 39 kontrol (koşullu ortalamalar, projeksiyon, üç sütunlu tablo: eğitim; + deneyim profili; + siyahi göstergesi) | sonuç, az değerli açıklayıcı değişken (≤ 30 değer), karesiyle eklenen sürekli kontrol, iki kategorili grup |
| 2 | Card (1995): 19 kontrol (M1–M3, klasik/HC1 SH, Breusch–Pagan, eğitim × siyahi etkileşimi, doğrusal birleşim, delta yöntemi) | sonuç, açıklayıcı değişken, iki kategorili grup (etkileşim), ek sayısal kontroller |
| 3 | Di Tella ve Schargrodsky (2004), Buenos Aires: 1994 saldırısından sonra Yahudi kurumu bulunan 37 bloğa polis koruması; 876 blok, saldırı sonrası 4.380 blok-ay; 32 kontrol (denge tablosu, ham ve kovaryat ayarlı fark, seçilmiş grup ve ortalamaya dönüş; blok düzeyinde küme SH). Atama rastgele değildir; metinler dışsal atama varsayımını açıkça yazar | sonuç, iki kategorili tedavi, isteğe bağlı atama birimi (küme SH; seçilmezse HC1), tedavi öncesi başlangıç değişkenleri (denge tablosu ve kovaryat ayarı), birlikte seçilen seçilmiş grup göstergesi ve dayandığı başlangıç değişkeni |
| 4 | Angrist ve Krueger (1991), 1980 nüfus sayımı: 1930–1939 doğumlu 329.509 erkek, araç yılın ilk çeyreğinde doğmuş olmak; 14 kontrol (OLS, ilk aşama, indirgenmiş biçim, 2SLS, ilk aşama F, dolaylı EKK oranı, kesin yüzde dönüşüm, elle ikinci aşama; doğum yılı ve bölge göstergeleri) | sonuç (isteğe bağlı logaritma), endojen açıklayıcı değişken, tek araç, dışsal kontroller |
| 5 | Cox, Hansen ve Jimenez (2004), Filipinler: 8.684 kentsel hane, sonuç yurt dışından havale almak; 62 kontrol (LPM, Logit ve Probit, ortalama marjinal etkiler ve delta yöntemi standart hataları, Logit ölçek çarpanı, yaş profili, kadın hane reisi göstergesinde sonlu fark) | iki kategorili sonuç, sürekli profil değişkeni, iki kategorili gösterge, ek kontroller |
| 6 | Card ve Krueger (1994), New Jersey ve Pennsylvania: 410 fast-food restoranı, iki tur; sonuç tam zamanlı çalışan sayısı, profil değişkeni restoranın günde açık kaldığı saat (spline düğümleri 12 ve 16); 44 kontrol (OLS, LAD ve Tobit profilleri, Tobit'in üç hedefi, model kontrolü) | sıfırda yığılan negatif olmayan sonuç, ana açıklayıcı (düğümleri çeyreklerde spline), ek kontroller |
| 7 | Card (1995), NLSYM 1976: 3.010 erkek, log saatlik ücret; 45 kontrol (beş kantilde kantil regresyon, Hendricks–Koenker standart hataları, OLS, eğitim ve siyahi göstergesinin katsayı profili, kesin yüzde etkiler, 0,90–0,10 farkı ve ortak kovaryansla standart hatası) | sonuç (isteğe bağlı logaritma), ana açıklayıcı, karesiyle eklenen sürekli kontrol, iki kategorili grup, ek kontroller (en çok 6) |
| 8 | Angrist ve Lavy (1999), İsrail 1991: 1.013 okulda 2.049 dördüncü sınıf, sınıfın ortalama matematik puanı ve okulun dezavantajlı öğrenci yüzdesi; 18 kontrol (birini dışarıda bırakan ve okul düzeyinde küme-silmeli CV ile h, doğrusal model, kübik polinom, kübik spline ve iki bantta yerel doğrusal tahmin) | sonuç, sürekli açıklayıcı (en az 20 farklı değer), isteğe bağlı küme (en az 10 küme; seçilmezse HC1) |
| 9 | Angrist ve Lavy (1999), İsrail 1991: 1.001 okulda 2.018 beşinci sınıf; eşik değişkeni okulun 5. sınıf kaydı, Maimonides eşiği 41, sonuç sınıfın ortalama okuma puanı; 40 kontrol (tasarım satırları ve sınıf mevcudunda ilk aşama sıçraması, beş bant genişliğinde keskin RDD, eşikteki iki limit, dikdörtgen çekirdek ve bant genişliği ölçeği; okul düzeyinde küme SH, güven bantlı grafik) | sonuç, eşik değişkeni (en az 20 farklı değer), yazılan eşik değeri, isteğe bağlı ana h (varsayılan: ±h√6 penceresi gözlemlerin yaklaşık %40'ı; tablo 0,5h–1,5h), tedavi tarafı (sağ ya da sol), isteğe bağlı küme (küme SH; seçilmezse HC1) |
| 10 | Duflo, Dupas ve Kremer (2011), Kenya: 121 okulda 5.795 öğrenci, toplam puanın tracking'e regresyonu; 10 kontrol (HC1, 1.000 tekrarlı pairs bootstrap, okul-kümeli SH ve okul düzeyinde küme bootstrap'ı: standart hata ve percentile aralığı) | sonuç, hedef açıklayıcı, kontroller (en çok 8), isteğe bağlı küme (en az 20 küme; seçilirse Adım 3'te küme SH ve küme bootstrap'ı); 1.000 tekrar, tohum 807 |
| 11 | Card (1995), NLSYM 1976: ücreti ve IQ dahil 15 aday değişkeni gözlenen 2.034 erkek, açık kuralla 1.524 eğitim / 510 test; 13 kontrol (basit OLS, 132 terimli sözlükte zengin OLS, 5-katlı CV ile Ridge ve Lasso; test ve eğitim MSE, sıfırdan farklı katsayı, seçilen ceza) | sonuç, basit modelin 1–6 değişkeni, zengin model için ek sayısal adaylar (en çok 12) ve en çok üç kategorik değişken (her kategoride en az 10 gözlem), "kareler ve ikili etkileşimler" seçeneği |
| 12 | Card (1995), NLSYM 1976: 2.997 erkek, eğitim yılının log ücrete etkisi, yalnız tedavi öncesi sekiz kontrol; 14 kontrol (kontrolsüz regresyon, doğrusal kontrollü OLS, 42 terimli sözlükte Lasso yardımcı modelli çapraz uyarlamalı DML, 11 kat kuralında duyarlılık ve medyan) | sonuç, tedavi (iki değerli ya da sürekli), 2–8 tedavi öncesi kontrol (sözlük: kareler ve ikili çarpımlar), isteğe bağlı küme (en az 20 küme) |

Ek veri kaynakları bütün konular (1–12) için hazırdır; her blokta alternatif veri seti ve roller önceden onaylandı.

### Üç dilde aynı sayı sözleşmesi

| Sınıf | Yöntemler | Beklenti |
|---|---|---|
| Birebir aynı | OLS, HC1, Logit/Probit ve Tobit katsayıları, AME, kantil regresyon katsayıları (kesin çözüm tekse), yerel doğrusal tahmin ve CV ile seçilen bant genişliği | Ondalık düzeyinde eşit |
| Ayar sabitlenince aynı | Keskin RDD (çekirdek, bant genişliği ölçeği ve ağırlıklı EKK + HC1 kodda yazılı: Python `WLS(...).fit(cov_type="HC1")`, R `lm(weights = w)` + `vcovHC(type = "HC1")`, Stata `regress [aw = w], vce(robust)`; hazır RDD paketlerinin varsayılanları farklıdır), Küme SH ve p-değeri dağılımı, 2SLS dayanıklı SH (Python `debiased=True`, R `vcovHC(type = "HC1")`, Stata `vce(robust) small`), Logit/Probit dayanıklı SH (gözlenen Hessian'lı sandviç: Python `HC0`, R `dayanikli_vcov()`, Stata `vce(robust)`; Stata ayrıca n/(n−1) ile çarpar), LAD (çözüm tek olmayabilir; eğriler aynı), kantil regresyon SH (Hendricks–Koenker: R `se = "nid"`; Python ve Stata'da kodda yazılı), Ridge, Lasso ve Elastic Net (ceza ölçeği, ızgara, ölçekleme ve kat kuralı kodda yazılı: Python `enet_path(tol=1e-12)`, R `glmnet(standardize = FALSE, thresh = 1e-16)`, Stata Mata koordinat inişi), DML (yardımcı model, kat kuralı ve küme SH kodda yazılı) | Paket varsayılanları farklı; kodda açıkça sabitlenir. Konu 11–12'de eğitim/test bölmesi, CV katları ve DML bölmeleri rastgele sayıyla değil açık kuralla (ör. u = {iφ}) kurulduğu için üç dilde aynıdır |
| Yalnız dağılımda aynı | Bootstrap ve Sezgi deneylerindeki simülasyonlar | Diller farklı rastgele sayı üreteci kullanır; Python kodu uygulamanın çekilişini birebir yapar, R ve Stata betikleri bootstrap sayılarını Monte Carlo toleransıyla denetler |

Her adım hangi sınıfta olduğunu arayüzde gösterir.

### Öğrenci tarafında yazılım

- **Python:** `pandas`, `numpy`, `statsmodels`, `matplotlib`; 2SLS için `linearmodels`; kantil regresyonun kesin çözümü için `scipy`; Lasso ve Elastic Net yolu için `scikit-learn`; kendi verinde Excel dosyası için `openpyxl`.
- **R:** temel R; dayanıklı çıkarım gereken konularda `sandwich` ve `lmtest`, Hansen'in `.dta` dosyaları için `haven`, 2SLS ve Tobit için `AER`, kantil regresyon için `quantreg`, Lasso ve Elastic Net için `glmnet`; kendi verinde Excel dosyası için `readxl`.
- **Stata:** 14 ve üstü. Konu 11–12'de Ridge, Lasso, Elastic Net, çapraz doğrulama ve DML do-dosyasındaki Mata fonksiyonlarıyla hesaplanır; Stata 16'nın `lasso` komutları gerekmez.

## Teknik yapı

- `app.py`: sayfa yapılandırması, ortak başlık, sidebar ve konu yönlendirmesi.
- `core/`: Streamlit'ten bağımsız metadata, regresyon, çıkarım, tanı, simülasyon, veri doğrulama, kod tarifi, durum ve soru mantığı.
- `core/labs/`: ders notu laboratuvarlarının dilden bağımsız tanımları (adımlar, işlemler, notlardaki sayılar) ve yürütücü.
- `core/codegen/`: aynı tanımdan Python, R ve Stata kodu üreten üreticiler.
- `core/hansen_data.py`: Hansen veri arşivini çalışma anında indirme ve yüklenen dosyayı doğrulama.
- `topics/`: konu bazlı Streamlit render modülleri.
- `assets/`: ortak CSS.
- `tests/`: registry, sözleşme ve Streamlit AppTest kontrolleri.
- `docs/`: mimari ve uygulama planı.

## Kurulum

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## Çalıştırma

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Test ve doğrulama

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall app.py core topics tests
git diff --check
```

Notlardaki sayılarla karşılaştıran testler gerçek veri gerektirir. Veri depoda tutulmadığı için dosya
yolu ortam değişkeniyle verilir; verilmezse bu testler atlanır:

```powershell
$env:IKT807_HANSEN_CPS09MAR_PATH = "C:\veri\cps09mar.txt"
$env:IKT807_HANSEN_DDK2011_PATH = "C:\veri\DDK2011.dta"
$env:IKT807_HANSEN_CARD1995_PATH = "C:\veri\Card1995.dta"
$env:IKT807_HANSEN_CHJ2004_PATH = "C:\veri\CHJ2004.dta"
$env:IKT807_HANSEN_LM2007_PATH = "C:\veri\LM2007.dta"
$env:IKT807_HANSEN_DS2004_PATH = "C:\veri\DS2004.dta"
$env:IKT807_HANSEN_AK1991_PATH = "C:\veri\AK1991.dta"
$env:IKT807_HANSEN_CK1994_PATH = "C:\veri\CK1994.dta"
$env:IKT807_HANSEN_AL1999_PATH = "C:\veri\AL1999.dta"
.\.venv\Scripts\python.exe -m pytest -q
```

Bu testler üretilen Python ve R betiklerini gerçekten çalıştırır (R için `Rscript` PATH'te olmalıdır;
gerekli R paketleri: `sandwich`, `lmtest`, `haven`, `AER`, `quantreg`, Konu 11–12 için `glmnet`, kendi verin testleri
için `readxl`; `readxl` kurulu değilse yalnız Excel okuyan R adımları atlanır).
Stata lisans gerektirdiği için otomatik test edilemez; do-dosyası sonunda kendi kontrollerini yapar.

Ek veri kaynaklarının testleri `tests/test_lab_variants.py` (alternatif örneğin sayıları, kendi verinde örnek dosya,
Türkçe CSV, işaretli sütun adları, tam uyum, logaritması alınamayan sonuç, notların kod özeti) ve
`tests/test_lab_sources_ui.py` (Streamlit AppTest: kaynak seçimi, dosya yükleme, seçimlerin korunması) içindedir;
Konu 3–4'ün bağımsız hesapları ve kendi veri kuralları `tests/test_lab_variants_konu03_04.py`, Konu 5–8'inkiler
`tests/test_lab_variants_konu05_08.py`, Konu 9–12'ninkiler `tests/test_lab_variants_konu09_12.py` (RDD küme SH'sinin
statsmodels ile eşitliği dahil) içindedir. Alternatif örneklerin gerçek veri testleri
`IKT807_HANSEN_CARD1995_PATH` (Konu 1, 2, 7, 11 ve 12), `IKT807_HANSEN_DS2004_PATH` (Konu 3),
`IKT807_HANSEN_AK1991_PATH` (Konu 4), `IKT807_HANSEN_CHJ2004_PATH` (Konu 5), `IKT807_HANSEN_CK1994_PATH` (Konu 6),
`IKT807_HANSEN_AL1999_PATH` (Konu 8 ve 9) ve `IKT807_HANSEN_DDK2011_PATH` (Konu 10) ile çalışır; kendi verin testleri
veri dosyası gerektirmez (örnek dosyalar kurgusaldır).

## Veri politikası

Ders notlarındaki uygulamalar Hansen'in *Econometrics* veri arşivine dayanır. Gerçek veri dosyaları depoya eklenmez. Uygulama sekmesi veriyi iki yoldan alır:

1. **Hansen'in sayfasından çalışma anında indirme.** Arşiv (`Econometrics Data.zip`) sunucuda bir kez indirilir ve önbelleğe alınır; bütün oturumlar aynı kopyayı kullanır. Hansen'in sayfası `~bhansen` → `~behansen` adresine yönlenmektedir; önce güncel, sonra eski adres denenir.
2. **Dosya yükleme.** Hansen'in dosyası (`cps09mar.txt`, `DDK2011.dta`, `Card1995.dta`, `CHJ2004.dta`, `LM2007.dta`; alternatif örnekler için `DS2004.dta`, `AK1991.dta`, `CK1994.dta`, `AL1999.dta`) yüklenebilir; cps09mar, DDK2011 ve LM2007 için ders notlarının öğretim CSV'si de kabul edilir (Card1995 ve CHJ2004 öğretim CSV'lerinde laboratuvarın türettiği ham değişkenler yoktur). Dosya yalnız oturumda tutulur.

`cps09mar` başlıksız `.txt` olarak okunur (değişken adları açıklama belgesindeki sırayla). DDK2011, Card1995 ve CHJ2004, değişken adlarını taşıyan `.dta` dosyalarından okunur; adlar üç dilde aynı olsun diye küçük harfe çevrilir. DDK2011 ve Card1995'te eksik değerler olağandır; analiz örneklemi laboratuvar adımlarında açıkça kurulur. CHJ2004, Hansen'in oluşturma dosyasının ham `urban.dta`'ya uygulanmış hâlidir (düzeltilmiş gelir, 8.684 hane). LM2007, Hansen'in `LM2007_create.do` dosyasıyla Ludwig ve Miller (2007) verisinden oluşturulmuştur: eşik değişkeni (`povrate60`) ve sonucu (`mort_age59_related_posths`) eksik olmayan 2.783 ilçe. DS2004 (Di Tella ve Schargrodsky 2004; 876 blok × Nisan–Aralık 1994 = 7.884 satır, Temmuz yalnız 1–17) ve AK1991 (Angrist ve Krueger 1991; 329.509 erkek) yalnız Uygulama sekmesinin alternatif örneklerinde kullanılır; ikisi de `.dta`'dan okunur, DS2004'te mahalle (`barrio`) metin sütunudur. CK1994 (Card ve Krueger 1994; 410
fast-food restoranı × iki anket dalgası = 820 satır) ve AL1999 (Angrist ve Lavy 1999; 4. ve 5. sınıflar, 4.067 sınıf)
de yalnız alternatif örneklerde (Konu 6; Konu 8'de 4., Konu 9'da 5. sınıflar) kullanılır ve `.dta`'dan okunur. AK1991'in modelleri büyük olduğu için tasarım matrisi bellekte tutulmaz (`docs/ARCHITECTURE.md`).

Yerel geliştirmede `IKT807_HANSEN_CPS09MAR_PATH` (eski adı `IKT807_CPS_PATH`), `IKT807_HANSEN_DDK2011_PATH`, `IKT807_HANSEN_CARD1995_PATH`, `IKT807_HANSEN_CHJ2004_PATH`, `IKT807_HANSEN_LM2007_PATH`, `IKT807_HANSEN_DS2004_PATH`, `IKT807_HANSEN_AK1991_PATH`, `IKT807_HANSEN_CK1994_PATH` ve
`IKT807_HANSEN_AL1999_PATH` verilirse veri otomatik yüklenir. Öğrencilere verilen Python, R ve Stata kodları da veriyi aynı arşivden indirir; internet yoksa betiğin başındaki yerel dosya satırına yol yazılır.

Her veri setinin kaynağı, gözlem birimi, örneklem kısıtı, değişken birimi, küme/yeniden örnekleme birimi ve yeniden dağıtım durumu `core/data_registry.py` içinde kayıtlıdır.

"Kendi verini yükle" seçeneğinde yüklenen dosya yalnız oturumun belleğinde işlenir: diske, ortak önbelleğe ya da günlüğe yazılmaz (en çok 5 MB ve 10.000 satır). Arayüz kişisel veri içeren dosya yüklenmemesini hatırlatır.

## Dağıtım

Hedef ortam Streamlit Community Cloud ve Python 3.12'dir. Uygulama çalışma zamanında LLM, dış AI API veya gizli anahtar kullanmaz. Açık dağıtım kontrollü simülasyonları ve oturum içi veri yükleme kapılarını kullanır; lisanslı gerçek veri dosyalarını içermez.

## Ekonometrik yorumlama ilkeleri

Dayanıklı standart hata nokta tahminini veya içselliği düzeltmez. Model uyumu tanımlama değildir. Logit/Probit ve Tobit katsayıları doğrudan gözlenen sonuç marjinal etkisi değildir. Yeniden örnekleme tanımlama sorununu çözmez. Düzenlileştirme ve DML, araştırma tasarımının yerine geçmez.

## Kullanım ve lisans

Kaynak kod [MIT Lisansı](LICENSE) ile yayımlanır. Dış veri setleri bu lisansın kapsamında değildir; kendi kaynak koşullarına ve yeniden dağıtım izinlerine tabidir. Bu depoda gerçek CPS, DDK, Card, CHJ veya LM2007 veri dosyası bulunmaz.
