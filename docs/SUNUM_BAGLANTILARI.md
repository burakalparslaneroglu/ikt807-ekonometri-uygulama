# Sunumdan doğrudan ders görünümüne geçiş

Yayımlanmış uygulama: https://iktt807-ekonometri-uygulama.streamlit.app/

Bağlantı biçimi:

- `?konu=07&sekme=sezgi&deney=2`: Konu 07, Sezgi, Deney 2.
- `?konu=04&sekme=uygulama&adim=4`: Konu 04, Uygulama, notlardaki örnek, Adım 4.
- `?konu=03&sekme=sezgi&deney=4&p_b=2`: ek çarpışan kontrol deneyini b=2 ile açar.
- `?konu=02&sekme=sezgi&deney=4&p_rho=0.95`: ek çoklu doğrusal bağlantı deneyini ρ=0,95 ile açar.

`konu=konu07` de kabul edilir. `sekme=sinav` Kendini sına sekmesini açar. Konu 00 için ayrı bir uygulama sayfası yoktur; giriş sunumu uygulamanın ana adresini açar.

Sezgi bağlantısı deneyin varsayılan parametrelerini yükler; `p_<anahtar>` verilirse yalnız geçerli aralık ve kaydırıcı adımındaki değerler kabul edilir. Parametreler, deneyde görünen DGP'nin aynı sayılarıdır. Tohumlar deney tanımında sabittir. Uygulama bağlantısı, oturumda önceki bir alternatif/kendi veri seçimi olsa bile ders notlarındaki örneği seçer. Gerçek veri henüz yüklenmemişse veri paneli ve seçilmiş adım görünür; veri yüklemesi bu yönlendirmeden ayrı bir işlemdir.

URL yalnız ilk açılışta veya URL parametreleri değişince uygulanır. Sonraki konu/deney/sekme/kaydırıcı seçimleri geri alınmaz. Geçersiz hedef açık bir uyarı üretir; başka bir deneye sessizce yönlendirilmez. Tanınmayan parametreler gezinmeyi etkilemez.

Sekme seçiminin sunum bağlantısıyla yönetilmesi Streamlit 1.64+ gerektirir. Yerel deneme için aynı sorguyu `http://localhost:8501/` adresine ekleyebilirsiniz. Yayımlanmış sürümün bu davranışı kullanması için uygulama değişikliklerinin GitHub'a push edilip Streamlit dağıtımına geçmesi gerekir.

## İki ek sezgi deneyi

- Konu 02, Deney 4: doğru belirlenmiş tam modelde regresör korelasyonu arttıkça ayrı bilgi ve tahmin kesinliği azalır; hedef ve koşullu yansızlık korunur. Gerçek koşullu standart hata FWL artık karelerinden hesaplanır ve bağımsız `(X'X)^{-1}` hesabıyla sınanır. ρ=1 rank kaybıdır ve kaydırıcı dışında tutulur.
- Konu 03, Deney 4: rassal atamada tedavi sonrası çarpışan M'yi kontrol etmek yeni yanlılık yaratır. DGP `Y=D+U+e`, `M=bD+U+V`; kontrolsüz hedef 1, M kontrollü projeksiyon hedefi `1-b/2`. U,V,e bağımsız standart normal ve D bunlardan bağımsız Bernoulli(0,5). Anakütle kovaryans sistemiyle bağımsız sayısal kontrol yapılır.

Bu örnekler yeni bir tahmin yöntemi eklemez; mevcut OLS işlem tanımları ve Python/R/Stata üreticileri kullanılır. R ve Stata farklı rassal sayı üreteçleri nedeniyle aynı sonlu örneklem sayılarını vermek zorunda değildir. Özgün uygulama laboratuvarı adımları ve sayısal kontrol hedefleri korunur. Özel ders notları ve sunum dosyaları bu depoya eklenmez.
