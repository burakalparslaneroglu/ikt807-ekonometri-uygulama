# IKT 807 Geliştirme Kuralları

1. Güncel yerel ders notları konu sırası, terminoloji, notasyon, estimand, varsayım dili ve yorumlama sınırları için bağlayıcıdır.
2. Ders notu ile yazılım veya literatür arasında içerik uyuşmazlığı görülürse sessiz düzeltme yapılmaz. Durum \`NOTE_CONSISTENCY_ISSUE\` olarak raporlanır ve kullanıcı kararı beklenir.
3. Look-ahead öğretim yapılmaz. Sonraki konunun yöntemi önceki konuda aktif laboratuvar olarak açılmaz.
4. Ekonometrik hesaplama, veri hazırlama, simülasyon ve soru üretimi Streamlit'ten bağımsız \`core/\` katmanında tutulur.
5. \`app.py\` yalnız ortak shell, navigation ve seçili topic \`render()\` çağrısını içerir.
6. Runtime LLM, dış AI API veya gizli anahtar kullanılmaz.
7. Rassallık \`np.random.default_rng(seed)\` ile yönetilir. Seed ve tuning ayarları sonuç metadata'sında görünürdür.
8. Yeni yöntem sayısal benchmark testi olmadan eklenmez.
9. Tanımlama varsayımı açık değilse nedensel dil kullanılmaz.
10. Estimand, estimator ve estimate ayrımı UI metninde ve sonuç sözleşmelerinde korunur.
11. Robust standart hata, bootstrap, düzenlileştirme veya DML içselliği çözüyor gibi sunulmaz.
12. Private ders materyali ve lisansı doğrulanmamış veri public repoya commit edilmez. \`references_private/\` izlenmez.
13. Her grafikte eksen adı, gerektiğinde birim, legend ve model/veri bağlamı bulunur.
14. Ham teknik değişken adları öğrenci arayüzünde açıklamasız gösterilmez.
15. Ön işleme ve tuning fold içinde yapılır; test verisi model seçimine sızmaz.
16. Her branch sonunda pytest, compileall ve \`git diff --check\` çalıştırılır.
