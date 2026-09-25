# IKT 807 Ekonometrik Modelleme ve Uygulamaları

İzmir Bakırçay Üniversitesi İktisat Tezli Yüksek Lisans Programı için geliştirilen Türkçe, etkileşimli Streamlit laboratuvarıdır. Uygulama yöntemleri komut listesi olarak değil, araştırma sorusu -> tahmin hedefi -> tanımlama -> tahmin -> çıkarım -> duyarlılık -> ekonomik yorum zinciri içinde ele alır.

## Güncel kapsam

Her konu iki sekmeden oluşur:

- **Uygulama:** ders notlarındaki uygulama laboratuvarını adım adım, gerçek Hansen verisiyle yeniden üretir. Her adımda notlardaki tablo ve şekiller, notlarla sayı sayı karşılaştırma ve seçilen dilde (Python, R, Stata) kod bulunur. Son adımda bütün laboratuvar tek dosya olarak indirilir; dosya sonunda sonuçları notlardaki basılı değerlerle karşılaştırır.
- **Sezgi:** veri üretim süreci (DGP) bilinen kontrollü deneyler. Her deney aynı sırayı izler: soru, DGP ve parametreleri, neye bakıyoruz, sonuç, ne gördük ve seçilen dilde kod. Gerçek veride görülemeyen nesneler (ör. gerçek koşullu ortalama) tahminlerle yan yana gösterilir.

- **Kendini sına:** haftanın kavramlarını sınayan soru seti. Dört tür: çoktan seçmeli, doğru–yanlış, boşluk doldurma ve denklem yazma. Her soru tek bir kavramı sınar ve notlardaki bir bölüme bağlıdır; yanlış cevaplara göre tekrar edilecek bölümler listelenir. Denklem sorularında yazılan formül anında önizlenir ve doğru cevaba eşdeğerliği sayısal olarak sınanır (`100(exp(b)-1)` ile `100*exp(b)-100` aynı kabul edilir).

Yeni yapı şu an **Konu 1** için hazırdır: Uygulama (Notlar §1.13, Adım 1–9, 39 sayısal kontrol), Sezgi (üç deney: koşullu ortalama ile OLS doğrusu, OLS artıklarının cebirsel özellikleri, OLS eğimi ile nedensel etki) ve Kendini sına (25 soru). Diğer konular ders sırasıyla taşınacaktır. Kod dili kenar çubuğundan bir kez seçilir ve bütün konularda geçerlidir.

### Üç dilde aynı sayı sözleşmesi

| Sınıf | Yöntemler | Beklenti |
|---|---|---|
| Birebir aynı | OLS, HC1, 2SLS, Logit/Probit, Tobit, AME | Ondalık düzeyinde eşit |
| Ayar sabitlenince aynı | Küme SH, kantil regresyon SH, çekirdek bant genişliği, Lasso λ | Paket varsayılanları farklı; kodda açıkça sabitlenir |
| Yalnız dağılımda aynı | Bootstrap, çapraz doğrulama, DML bölmeleri | Diller farklı rastgele sayı üreteci kullanır |

Her adım hangi sınıfta olduğunu arayüzde gösterir.

### Öğrenci tarafında yazılım

- **Python:** `pandas`, `numpy`, `statsmodels`, `matplotlib`.
- **R:** temel R; dayanıklı çıkarım gereken konularda `sandwich` ve `lmtest`.
- **Stata:** 14 ve üstü. Konu 11–12'deki `lasso` ve çapraz uyarlamalı komutlar Stata 16 gerektirir.

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
.\.venv\Scripts\python.exe -m pytest -q
```

Bu testler üretilen Python ve R betiklerini gerçekten çalıştırır (R için `Rscript` PATH'te olmalıdır).
Stata lisans gerektirdiği için otomatik test edilemez; do-dosyası sonunda kendi kontrollerini yapar.

## Veri politikası

Ders notlarındaki uygulamalar Hansen'in *Econometrics* veri arşivine dayanır. Gerçek veri dosyaları depoya eklenmez. Uygulama sekmesi veriyi iki yoldan alır:

1. **Hansen'in sayfasından çalışma anında indirme.** Arşiv (`Econometrics Data.zip`) sunucuda bir kez indirilir ve önbelleğe alınır; bütün oturumlar aynı kopyayı kullanır. Hansen'in sayfası `~bhansen` → `~behansen` adresine yönlenmektedir; önce güncel, sonra eski adres denenir.
2. **Dosya yükleme.** Hansen'in `cps09mar.txt` / `.dta` dosyası veya ders notlarının öğretim CSV'si yüklenebilir. Dosya yalnız oturumda tutulur.

Yerel geliştirmede `IKT807_HANSEN_CPS09MAR_PATH` (eski adı `IKT807_CPS_PATH`) verilirse veri otomatik yüklenir. Öğrencilere verilen Python, R ve Stata kodları da veriyi aynı arşivden indirir; internet yoksa betiğin başındaki yerel dosya satırına yol yazılır.

Her veri setinin kaynağı, gözlem birimi, örneklem kısıtı, değişken birimi, küme/yeniden örnekleme birimi ve yeniden dağıtım durumu `core/data_registry.py` içinde kayıtlıdır.

## Dağıtım

Hedef ortam Streamlit Community Cloud ve Python 3.12'dir. Uygulama çalışma zamanında LLM, dış AI API veya gizli anahtar kullanmaz. Açık dağıtım kontrollü simülasyonları ve oturum içi veri yükleme kapılarını kullanır; lisanslı gerçek veri dosyalarını içermez.

## Ekonometrik yorumlama ilkeleri

Dayanıklı standart hata nokta tahminini veya içselliği düzeltmez. Model uyumu tanımlama değildir. Logit/Probit ve Tobit katsayıları doğrudan gözlenen sonuç marjinal etkisi değildir. Yeniden örnekleme tanımlama sorununu çözmez. Düzenlileştirme ve DML, araştırma tasarımının yerine geçmez.

## Kullanım ve lisans

Kaynak kod [MIT Lisansı](LICENSE) ile yayımlanır. Dış veri setleri bu lisansın kapsamında değildir; kendi kaynak koşullarına ve yeniden dağıtım izinlerine tabidir. Bu depoda gerçek CPS, DDK, Card, CHJ veya LM2007 veri dosyası bulunmaz.
