# Streamlit Hansen notasyon uyumu — 9 Ekim 2026

## Sonraki çalışma: sunum bağlantıları ve ek deneyler — 9 Ekim 2026

- Önceki notasyon denetimi kullanıcı tarafından commit/push edildi; bu çalışma temiz main...origin/main üzerinde başladı. Bu görevde commit/push yapılmadı.
- `core/navigation.py`: doğrulanmış konu/sekme/deney-adım URL'leri; bağlantı yalnız yeni URL'de uygulanır; geçersiz değer uyarısı; Sezgi varsayılan/preset parametreleri; notlardaki laboratuvar veri kaynağı ve gölge anahtarları.
- `app.py` ve 12 konu sarmalayıcısı: widget'lardan önce hedefi uygular; ortak `topic_tabs()` ile gerçek seçili sekmeyi yönetir. Üç sekmenin mevcut hesap/çizim davranışı korunur. Streamlit alt sınırı 1.64.
- `core/labs/sezgi_ek_ders.py`: Konu 02 Sezgi 4 (çoklu doğrusal bağlantı, koşullu SH) ve Konu 03 Sezgi 4 (çarpışan kontrol, 1−b/2 projeksiyon hedefi); mevcut OLS/ScalarTable/Plot DSL'si, sabit bağımsız çekiliş tohumları 820/830. Python/R/Stata tek tanımdan üretilir.
- Kaynak ders notlarına DGP/ek semboller eklendi (§2.1.1/§3.10); 13 Beamer sunumu 51 bağlantıyla derlendi. Özel materyaller public depoya kopyalanmadı. Yerel kaynak projesinin state/rehber/denetim kayıtları yetkili çalışma alanında tutuluyor.
- Yeni testler bağımsız ters tasarım kovaryansını, anakütle kovaryans sistemiyle çarpışan hedefini, üretilen Python hesabını, bütün URL hedeflerini, bozuk parametreleri, eski veri kaynağı/gölge anahtarını ve gerçek AppTest etkileşimini sınar.
- Tam pytest: 907 test, 805 geçti / 101 atlandı / yeni çarpışan deneyinin üretilen tablo nesnesinde 1 hata. Tablo mevcut ScalarTable biçimine geçirildi. Sonrasında ilgili 68 test geçti, 7 R testi atlandı; tam taramadaki tek başarısız test `pytest --lf -q` ile yeniden geçti. Etkin sonuç 806 doğrulanan / 101 atlanan, açık başarısız test yok. Tam 907 test bu düzeltmeden sonra yeniden çalıştırılmadı.
- 101 atlama: R/R readxl çalışma ortamı (77) ve verilmemiş Hansen LM2007/DS2004/AK1991/CK1994/AL1999 dosyaları (24). Stata betikleri üretildi; Stata çalışma zamanı yürütülmedi.
- `compileall` ve `git diff --check` geçti. Python kodu gerçek yürütmeyle arayüz hesaplarıyla eşleştirildi; mevcut not laboratuvarları/numerik hedefleri değiştirilmedi.
- Gerçek tarayıcı: Sezgi bağlantısı doğru sekme, deney ve p_b=2 ile açıldı; Card uygulaması doğru not veri kaynağı ve Adım 4'ü açtı. Yerel sunucu 8509 test için çalıştırıldı.
- Son tarayıcı denetimi de geçti: Konu 02/Sezgi4/rho=0.95 doğru açıldı; Uygulama↔Sezgi elle geçişi korundu. Yeni iki deneyin rho=0.95/b=2 uç ayarlarındaki Python betikleri model ve sayısal tablolarla 1e-12 toleransında eşleşti. Son belge/diff kontrolü tamamlandı; yerel iş bitti.
- Yayımlanmış sürüm yeni yönlendirme/deney koduna henüz geçirilmedi; kullanıcı commit/push sonrasında dağıtımın güncellenmesini beklemelidir. Agent dağıtım yapmadı. QA için başlatılan 8509 sunucusu denetim sonunda durduruldu.

## Amaç ve sınırlar

- IKT 807 uygulamasının bütün konu sayfalarını güncel Hansen uyumlu ders notlarına göre incelemek ve notasyon uyumsuzluklarını düzeltmek.
- Tek gözlem / yığılmış örneklem; rassal değişken / gerçekleşme; skaler / vektör / matris boyutları birlikte açıklanır.
- Ek semboller ilk kullanımlarında tanımlanır. Mevcut yöntem mimarisi, gerçek veri ve notlardaki sayısal kontroller korunur.
- Konu sırası ve Uygulama / Sezgi / Kendini sına yapısı korunur; ortak kod üretimi sürdürülür.
- Commit ve push kullanıcı tarafından yapılacak; bu çalışmada yapılmayacak.
- Notlar ve sunumlar tamamlanmış, tek kopyaya geçirilmiş kaynaklardır; uygulama için yerel referans olarak okunur. Ders materyali ve veri public depoya kopyalanmaz.

## Başlangıç durumu

- Çalışma ağacı main üzerinde, 13 dosyada önceden yapılmış değişiklikler var: labs konu05/07/08/09, ornek_konu05/07/08; quiz konu03/05/07/08/09/10.
- Bu değişiklikler korunacak. Başlangıç farkları ve Python kaynak hash'leri git tarafından izlenmeyen tmp/notation_audit altında kaydedildi.
- Kaynak notasyon rehberi: yerel güncel ders notlarının on_bolum/hansen_notasyon_rehberi.tex dosyası; Hansen 11 Mart 2021 sürümü, §1.3, §3.10–3.11.
- Yerel sanal ortam Python 3.12.0; Streamlit 1.64.0, NumPy 2.5.3, pytest 8.4.2.

## Durum

12 konunun incelemesi ve düzeltmeleri tamamlandı. Uygulama/notlardaki örnek, alternatif örnek, kendi veri açıklamaları, Sezgi deneyleri ve Kendini sına soruları kapsandı. Bulgular ve yapılan değişiklikler `docs/HANSEN_NOTASYON_DENETIMI.md` içinde.

- Yeni ortak/konu rehberi: `core/notation.py`, `topics/shared.py`.
- İlgili `core/labs/konuNN.py`, `ornek_konuNN.py`, `sezgi_konuNN.py` ve `core/quiz/konuNN.py` metinleri düzeltildi.
- RDD ve ceza etiketleri: `topics/lab_ui.py`, `topics/selection_ui.py`; bilimsel hesaplara dokunulmadı.
- Yeni ve eski notasyonla denklem girişlerini birlikte değerlendiren mevcut testler genişletildi (Konu 3/4, 5/6, 11/12).
- Git işlemi yapılmadı; başlangıçtaki 13 dosyanın değişiklikleri korundu.

## Doğrulama durumu

- Başlangıç kontrolü: 37 test geçti; mevcut not betiklerinin hash'i doğru.
- İlk hedefli kontrol: 103 geçti, 6 atlandı (ortam/veri bağımlılığı).
- Yeni gösterim ve geriye uyumluluk: 66 denklem/soru testi geçti. İlk tam geçişte saptanan Y(1)/Y(0) ve τ girişleri regresyonları düzeltildi; eski girişler de kabul ediliyor.
- Gerçek Streamlit/KaTeX denetimi: 932 metin/formül bloğu, 1842 matematik gösterimi; son kontrolde ayrıştırma hatası, tanımsız kırmızı komut ve Streamlit istisnası yok. Not ve alternatif laboratuvarların statik açıklamaları, tüm sorular/sembolleri, 36 Sezgi deneyinin varsayılan DGP'leri ve rehberler kapsandı. Dinamik kendi-veri metinleri mevcut testlerle kontrol edilir; olası tüm kullanıcı veri dosyaları tarayıcıda tek tek denenmedi.
- Birinci konu rehberinin gerçek ders sayfasında kalın italik matris görünümü görsel olarak doğrulandı.
- DML kare terimi bağımsız Gauss kareselleştirmesiyle doğrulandı (`tmp/notation_audit/check_rate.py`): sonuç yardımcı fonksiyonu tam doğru olsa da tedavi hatasının karesi kalıyor.
- Python compileall ve git diff --check geçti; yalnız Git'in yerel LF/CRLF bilgilendirmeleri var.
- Son tam pytest: 881 test, 780 geçti / 99 atlandı / 2 eski RDD notasyonu bekleyen test başarısız (793,26 saniye). Başarısızlıkların ikisi de θ değişikliğine uyarlanmamış test ifadeleri: arayüz sütunu τ̂ ve sol taraftaki etkinin −τ̂ yazımı. Testler yeni θ̂/−θ̂ gösterimine uyarlandı; üretim koduna ek değişiklik yapılmadı.
- Son `pytest --lf -q --junitxml=tmp/notation_audit/pytest_recheck.xml`: 2/2 geçti. Böylece aynı üretim kodunun 881 testlik paketinde toplam 782 geçti, 99 ortam/veri bağımlılığından atlandı; açık başarısızlık yok. Tam koşunun JUnit kaydı `pytest_final.xml`, düzeltilen iki testin kaydı `pytest_recheck.xml`.
- Atlanan 99 testin 75'i Rscript/R readxl ortamına, 24'ü testlere verilmemiş gerçek Hansen dosyalarına bağlı: LM2007 (3), DS2004 (4), AK1991 (4), CK1994 (4), AL1999 (9). Bu testlerin gerçek veri/R çalıştırması tamamlanmış sayılmıyor.
- En son compileall ve git diff --check sonucu başarılı. Commit, staging ve push yok.

Geçici denetim betikleri ve çıktıları `tmp/notation_audit` altında ve git tarafından izlenmiyor. Notların kod hash'i hedefi: `e6eb2b280607de0fe2a5b242b5c5a38d`.

## Kapsam dışındaki kaynak bulgusu

`NOTE_CONSISTENCY_ISSUE`: Yerel `Ders Notları/bolumler/12_dml_butunlesik_arastirma.tex:473`, “DML açısından kriter neden farklı?” kutusunda yalnız iki farklı hatanın çarpımından söz ediyor. Aynı kaynak §12.8 satır 317–324 norm koşulunu ve m_D karesini doğru içeriyor. Uygulama açık norm koşulunu izler. Not dosyası bu Streamlit çalışmasında değiştirilmedi; finalde kutunun da güncellenmesi gerektiği kullanıcıya bildirilecek. Ayrıntı denetim raporunda.

## Tamamlanma ölçütleri

- Tüm konu içeriklerinin notasyon eşleştirmesi; ek sembol ve boyut tanımları.
- İlgili testler ve gerekli tam pytest, compileall, git diff --check kontrolleri.
- Arayüzde matematik gösteriminin doğrulanması; başlangıç değişikliklerinin korunması.
- İncelenebilir yerel farklar ve doğrulama raporu; commit/push yapılmaması.

## Sonuç

Streamlit kapsamındaki değişiklikler ve mevcut ortamın çalıştırabildiği doğrulamalar tamamlandı. Kaynak notun DML kutusu ve ortam nedeniyle atlanan doğrulamalar yukarıda açıkça kaydedildi. Son uygulama inceleme için yerel `http://127.0.0.1:8507` üzerinde açık; geçici matematik denetim sunucusu kapatıldı. Kullanıcı commit ve push işlemlerini kendisi yapacak.
