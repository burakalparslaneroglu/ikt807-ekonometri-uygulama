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
  altındaysa yorum zayıf aracı belirtir. Örnek
  dosya kurgusal veriden bellekte üretilir (`core/labs/ornek_veri.py`: ücret verisi, okul deneyi, araç değişkeni
  verisi; DGP'ler dosyada yazılıdır). Okuma kuralları İKT 217 ve İKT 305 uygulamalarıyla aynıdır.

| Konu | Alternatif örnek | Kendi verinde roller |
|---|---|---|
| 1 | Card (1995), NLSYM 1976: 3.010 genç erkek; 39 kontrol (koşullu ortalamalar, projeksiyon, üç sütunlu tablo: eğitim; + deneyim profili; + siyahi göstergesi) | sonuç, az değerli açıklayıcı değişken (≤ 30 değer), karesiyle eklenen sürekli kontrol, iki kategorili grup |
| 2 | Card (1995): 19 kontrol (M1–M3, klasik/HC1 SH, Breusch–Pagan, eğitim × siyahi etkileşimi, doğrusal birleşim, delta yöntemi) | sonuç, açıklayıcı değişken, iki kategorili grup (etkileşim), ek sayısal kontroller |
| 3 | Di Tella ve Schargrodsky (2004), Buenos Aires: 1994 saldırısından sonra Yahudi kurumu bulunan 37 bloğa polis koruması; 876 blok, saldırı sonrası 4.380 blok-ay; 32 kontrol (denge tablosu, ham ve kovaryat ayarlı fark, seçilmiş grup ve ortalamaya dönüş; blok düzeyinde küme SH). Atama rastgele değildir; metinler dışsal atama varsayımını açıkça yazar | sonuç, iki kategorili tedavi, isteğe bağlı atama birimi (küme SH; seçilmezse HC1), tedavi öncesi başlangıç değişkenleri (denge tablosu ve kovaryat ayarı), birlikte seçilen seçilmiş grup göstergesi ve dayandığı başlangıç değişkeni |
| 4 | Angrist ve Krueger (1991), 1980 nüfus sayımı: 1930–1939 doğumlu 329.509 erkek, araç yılın ilk çeyreğinde doğmuş olmak; 14 kontrol (OLS, ilk aşama, indirgenmiş biçim, 2SLS, ilk aşama F, dolaylı EKK oranı, kesin yüzde dönüşüm, elle ikinci aşama; doğum yılı ve bölge göstergeleri) | sonuç (isteğe bağlı logaritma), endojen açıklayıcı değişken, tek araç, dışsal kontroller |

Diğer konular blok blok eklenecektir; her blokta alternatif veri seti ve roller önceden onaylanır.

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
.\.venv\Scripts\python.exe -m pytest -q
```

Bu testler üretilen Python ve R betiklerini gerçekten çalıştırır (R için `Rscript` PATH'te olmalıdır;
gerekli R paketleri: `sandwich`, `lmtest`, `haven`, `AER`, `quantreg`, kendi verin testleri için `readxl`; `readxl`
kurulu değilse yalnız Excel okuyan R adımları atlanır).
Stata lisans gerektirdiği için otomatik test edilemez; do-dosyası sonunda kendi kontrollerini yapar.

Ek veri kaynaklarının testleri `tests/test_lab_variants.py` (alternatif örneğin sayıları, kendi verinde örnek dosya,
Türkçe CSV, işaretli sütun adları, tam uyum, logaritması alınamayan sonuç, notların kod özeti) ve
`tests/test_lab_sources_ui.py` (Streamlit AppTest: kaynak seçimi, dosya yükleme, seçimlerin korunması) içindedir;
Konu 3–4'ün bağımsız hesapları ve kendi veri kuralları `tests/test_lab_variants_konu03_04.py` içindedir. Alternatif
örneklerin gerçek veri testleri `IKT807_HANSEN_CARD1995_PATH` (Konu 1–2), `IKT807_HANSEN_DS2004_PATH` (Konu 3) ve
`IKT807_HANSEN_AK1991_PATH` (Konu 4) ile çalışır; kendi verin testleri veri dosyası gerektirmez (örnek dosyalar
kurgusaldır).

## Veri politikası

Ders notlarındaki uygulamalar Hansen'in *Econometrics* veri arşivine dayanır. Gerçek veri dosyaları depoya eklenmez. Uygulama sekmesi veriyi iki yoldan alır:

1. **Hansen'in sayfasından çalışma anında indirme.** Arşiv (`Econometrics Data.zip`) sunucuda bir kez indirilir ve önbelleğe alınır; bütün oturumlar aynı kopyayı kullanır. Hansen'in sayfası `~bhansen` → `~behansen` adresine yönlenmektedir; önce güncel, sonra eski adres denenir.
2. **Dosya yükleme.** Hansen'in dosyası (`cps09mar.txt`, `DDK2011.dta`, `Card1995.dta`, `CHJ2004.dta`, `LM2007.dta`; alternatif örnekler için `DS2004.dta`, `AK1991.dta`) yüklenebilir; cps09mar, DDK2011 ve LM2007 için ders notlarının öğretim CSV'si de kabul edilir (Card1995 ve CHJ2004 öğretim CSV'lerinde laboratuvarın türettiği ham değişkenler yoktur). Dosya yalnız oturumda tutulur.

`cps09mar` başlıksız `.txt` olarak okunur (değişken adları açıklama belgesindeki sırayla). DDK2011, Card1995 ve CHJ2004, değişken adlarını taşıyan `.dta` dosyalarından okunur; adlar üç dilde aynı olsun diye küçük harfe çevrilir. DDK2011 ve Card1995'te eksik değerler olağandır; analiz örneklemi laboratuvar adımlarında açıkça kurulur. CHJ2004, Hansen'in oluşturma dosyasının ham `urban.dta`'ya uygulanmış hâlidir (düzeltilmiş gelir, 8.684 hane). LM2007, Hansen'in `LM2007_create.do` dosyasıyla Ludwig ve Miller (2007) verisinden oluşturulmuştur: eşik değişkeni (`povrate60`) ve sonucu (`mort_age59_related_posths`) eksik olmayan 2.783 ilçe. DS2004 (Di Tella ve Schargrodsky 2004; 876 blok × Nisan–Aralık 1994 = 7.884 satır, Temmuz yalnız 1–17) ve AK1991 (Angrist ve Krueger 1991; 329.509 erkek) yalnız Uygulama sekmesinin alternatif örneklerinde kullanılır; ikisi de `.dta`'dan okunur, DS2004'te mahalle (`barrio`) metin sütunudur. AK1991'in modelleri büyük olduğu için tasarım matrisi bellekte tutulmaz (`docs/ARCHITECTURE.md`).

Yerel geliştirmede `IKT807_HANSEN_CPS09MAR_PATH` (eski adı `IKT807_CPS_PATH`), `IKT807_HANSEN_DDK2011_PATH`, `IKT807_HANSEN_CARD1995_PATH`, `IKT807_HANSEN_CHJ2004_PATH`, `IKT807_HANSEN_LM2007_PATH`, `IKT807_HANSEN_DS2004_PATH` ve `IKT807_HANSEN_AK1991_PATH` verilirse veri otomatik yüklenir. Öğrencilere verilen Python, R ve Stata kodları da veriyi aynı arşivden indirir; internet yoksa betiğin başındaki yerel dosya satırına yol yazılır.

Her veri setinin kaynağı, gözlem birimi, örneklem kısıtı, değişken birimi, küme/yeniden örnekleme birimi ve yeniden dağıtım durumu `core/data_registry.py` içinde kayıtlıdır.

"Kendi verini yükle" seçeneğinde yüklenen dosya yalnız oturumun belleğinde işlenir: diske, ortak önbelleğe ya da günlüğe yazılmaz (en çok 5 MB ve 10.000 satır). Arayüz kişisel veri içeren dosya yüklenmemesini hatırlatır.

## Dağıtım

Hedef ortam Streamlit Community Cloud ve Python 3.12'dir. Uygulama çalışma zamanında LLM, dış AI API veya gizli anahtar kullanmaz. Açık dağıtım kontrollü simülasyonları ve oturum içi veri yükleme kapılarını kullanır; lisanslı gerçek veri dosyalarını içermez.

## Ekonometrik yorumlama ilkeleri

Dayanıklı standart hata nokta tahminini veya içselliği düzeltmez. Model uyumu tanımlama değildir. Logit/Probit ve Tobit katsayıları doğrudan gözlenen sonuç marjinal etkisi değildir. Yeniden örnekleme tanımlama sorununu çözmez. Düzenlileştirme ve DML, araştırma tasarımının yerine geçmez.

## Kullanım ve lisans

Kaynak kod [MIT Lisansı](LICENSE) ile yayımlanır. Dış veri setleri bu lisansın kapsamında değildir; kendi kaynak koşullarına ve yeniden dağıtım izinlerine tabidir. Bu depoda gerçek CPS, DDK, Card, CHJ veya LM2007 veri dosyası bulunmaz.
