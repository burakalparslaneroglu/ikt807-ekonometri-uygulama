# IKT 807 Ekonometrik Modelleme ve Uygulamaları

İzmir Bakırçay Üniversitesi İktisat Tezli Yüksek Lisans Programı için geliştirilen Türkçe, etkileşimli Streamlit laboratuvarıdır. Uygulama yöntemleri komut listesi olarak değil, araştırma sorusu -> tahmin hedefi -> tanımlama -> tahmin -> çıkarım -> duyarlılık -> ekonomik yorum zinciri içinde ele alır.

## Güncel kapsam

Ortak uygulama kabuğu ve 12 konu kaydının tamamı etkileşimli laboratuvar olarak aktiftir. Laboratuvarlar regresyon ve çıkarım, tanımlama ve 2SLS, ikili ve sınırlı sonuç modelleri, kantil/çekirdek yöntemleri, regresyon süreksizliği, yeniden örnekleme, sızıntısız model seçimi ve çapraz uyarlamalı DML yöntemlerini kontrollü veri üretim süreçleri üzerinde gösterir.

Her konudaki dört laboratuvar bölümünün sonunda **Uygulama kodu** alanı bulunur. Öğrenci seçili bölümü bağımsız çalıştıran Python betiğini veya gerekli paket kurulum hücresi eklenmiş Colab not defterini indirebilir. Lisanslı veri dosyası bulunmadığında kodlar aynı değişken yapısını öğreten kontrollü örnekle çalışır.

## Teknik yapı

- \`app.py\`: sayfa yapılandırması, ortak başlık, sidebar ve konu yönlendirmesi.
- \`core/\`: Streamlit'ten bağımsız metadata, regresyon, çıkarım, tanı, simülasyon, veri doğrulama, kod tarifi, durum ve soru mantığı.
- \`topics/\`: konu bazlı Streamlit render modülleri.
- \`assets/\`: ortak CSS.
- \`tests/\`: registry, sözleşme ve Streamlit AppTest kontrolleri.
- \`docs/\`: mimari ve uygulama planı.

## Kurulum

Windows PowerShell:

\`\`\`powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
\`\`\`

## Çalıştırma

\`\`\`powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
\`\`\`

## Test ve doğrulama

\`\`\`powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall app.py core topics tests
git diff --check
\`\`\`

## Veri politikası

Ders notlarındaki uygulamalar Hansen'in \`Econometrics\` veri paketine dayanmaktadır. Kaynak verilerin açık yeniden dağıtım lisansı doğrulanana kadar gerçek veri dosyaları public repoya eklenmez. Yerel geliştirme kopyaları \`references_private/\` altında tutulur ve Git tarafından izlenmez.

Her veri setinin kaynağı, gözlem birimi, örneklem kısıtı, değişken birimi, küme/yeniden örnekleme birimi ve yeniden dağıtım durumu \`core/data_registry.py\` içinde kayıtlıdır. Yeniden dağıtım izni doğrulanmayan veri dosyaları kaynak kod deposuna alınmaz.

Konu 01, 05, 07, 10 ve 11 veri laboratuvarları hazırlanmış CPS CSV dosyasını yalnız çalışma zamanında kabul eder. Dosya arayüzden yüklenebilir veya \`IKT807_CPS_PATH\` ortam değişkeniyle gösterilebilir; uygulama dosyayı depoya kopyalamaz. Konu 12 DDK dosyasında okul bazlı GroupKFold uygular. DDK, Card, CHJ ve LM2007 CSV dosyaları aynı politika altında yalnız oturum içinde şema doğrulamasından geçirilir.

## Dağıtım

Hedef ortam Streamlit Community Cloud ve Python 3.12'dir. Uygulama çalışma zamanında LLM, dış AI API veya gizli anahtar kullanmaz. Açık dağıtım kontrollü simülasyonları ve oturum içi veri yükleme kapılarını kullanır; lisanslı gerçek veri dosyalarını içermez.

## Ekonometrik yorumlama ilkeleri

Dayanıklı standart hata nokta tahminini veya içselliği düzeltmez. Model uyumu tanımlama değildir. Logit/Probit ve Tobit katsayıları doğrudan gözlenen sonuç marjinal etkisi değildir. Yeniden örnekleme tanımlama sorununu çözmez. Düzenlileştirme ve DML, araştırma tasarımının yerine geçmez.

## Kullanım ve lisans

Kaynak kod [MIT Lisansı](LICENSE) ile yayımlanır. Dış veri setleri bu lisansın kapsamında değildir; kendi kaynak koşullarına ve yeniden dağıtım izinlerine tabidir. Bu depoda gerçek CPS, DDK, Card, CHJ veya LM2007 veri dosyası bulunmaz.
