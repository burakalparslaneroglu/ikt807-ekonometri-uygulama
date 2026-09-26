# IKT 807 Geliştirme Kuralları

1. Güncel yerel ders notları konu sırası, terminoloji, notasyon, estimand, varsayım dili ve yorumlama sınırları için bağlayıcıdır.
2. Ders notu, sunum, uygulama veya literatür arasında uyuşmazlık ya da notlarda hata bulunursa sessiz düzeltme yapılmaz. Düzeltme aynı turda kaynağından başlar: önce ders notu (LaTeX), sonra ilgili sunum, sonra uygulama; notlar ve sunumlar sıfır hatayla yeniden derlenir. Her değişiklik raporlanır. Notasyon veya terim gibi yoruma açık seçimler gerekçesiyle bildirilir; içerik kararı gerektiren uyuşmazlıklar `NOTE_CONSISTENCY_ISSUE` olarak raporlanır ve kullanıcı kararı beklenir. Notlar ve sunumlar bu depoda değil, öğretim elemanının yerel klasöründe tutulur (kural 12).
3. Look-ahead öğretim yapılmaz. Sonraki konunun yöntemi önceki konuda aktif laboratuvar olarak açılmaz.
4. Ekonometrik hesaplama, veri hazırlama, simülasyon ve soru üretimi Streamlit'ten bağımsız `core/` katmanında tutulur.
5. `app.py` yalnız ortak shell, navigation ve seçili topic `render()` çağrısını içerir.
6. Runtime LLM, dış AI API veya gizli anahtar kullanılmaz.
7. Rassallık `np.random.default_rng(seed)` ile yönetilir. Seed ve tuning ayarları sonuç metadata'sında görünürdür.
8. Yeni yöntem sayısal benchmark testi olmadan eklenmez.
9. Tanımlama varsayımı açık değilse nedensel dil kullanılmaz.
10. Estimand, estimator ve estimate ayrımı UI metninde ve sonuç sözleşmelerinde korunur.
11. Robust standart hata, bootstrap, düzenlileştirme veya DML içselliği çözüyor gibi sunulmaz.
12. Private ders materyali ve lisansı doğrulanmamış veri public repoya commit edilmez. `references_private/` izlenmez.
13. Her grafikte eksen adı, gerektiğinde birim, legend ve model/veri bağlamı bulunur.
14. Ham teknik değişken adları öğrenci arayüzünde açıklamasız gösterilmez.
15. Ön işleme ve tuning fold içinde yapılır; test verisi model seçimine sızmaz.
16. Her branch sonunda pytest, compileall ve `git diff --check` çalıştırılır.

## Konu şablonu

17. Her konu üç sekmeden oluşur ve bu sıra korunur:
    - **Uygulama:** ders notlarındaki uygulama laboratuvarının adımları birebir (`core/labs/konuNN.py`). Notlarda basılı her sayı bir `Check` olarak tanıma girer; uygulama ve üretilen Python/R kodu bu sayıları gerçek veriyle yeniden üretmelidir.
    - **Sezgi:** DGP'si açıkça yazılmış kontrollü deneyler (`core/labs/sezgi_konuNN.py`). Her deney şu sırayı izler: soru → DGP ve parametreleri → neye bakıyoruz → sonuç → ne gördük → kod. Deneyler bilinen gerçeği (ör. gerçek koşullu ortalama) tahminle yan yana gösterir.
    - **Kendini sına:** dört türde soru seti (`core/quiz/konuNN.py`): çoktan seçmeli, doğru–yanlış, boşluk doldurma, denklem yazma. Her soru tek bir kavramı sınar ve notlardaki bir bölüme bağlıdır; kavram tekilliği ve bölüm kapsamı testle denetlenir. Sorular bölüm sonu egzersizlerini tekrar etmez.
18. Python, R ve Stata kodu aynı tanımdan üretilir; elle yazılmış dile özgü şablon eklenmez. Her adım veya deney, üç dilde aynı sayının hangi anlamda beklendiğini (birebir / ayar sabitlenince / yalnız dağılımda) gösterir.
19. Yeni yapıya taşınan konu `tests/test_topic_contracts.py` içindeki `MIGRATED_TOPICS` kümesine eklenir.
