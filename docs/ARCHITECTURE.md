# Mimari

## Sınırlar

Uygulama üç katmandan oluşur:

1. \`app.py\` ortak sayfa kabuğunu kurar ve seçili konuya yönlendirir.
2. \`topics/\` Streamlit bileşenlerini ve öğrenciye gösterilen metni oluşturur.
3. \`core/\` Streamlit'ten bağımsız metadata, veri doğrulama, sonuç sözleşmesi, state geçişi ve soru seçimini taşır.

Ekonometrik hesaplama yalnız \`core/\` altında yer alır. Plotly figürü oluşturma ve Streamlit yerleşimi \`topics/\` katmanında kalır.

Konu 01-12 sayısal sahipliği:

- \`core/ols.py\`: OLS, projeksiyon değişmezleri ve FWL.
- \`core/inference.py\`: klasik, HC1, küme kovaryansı ve delta yöntemi.
- \`core/diagnostics.py\`: kaldıraç, studentized artık ve Cook uzaklığı.
- \`core/functional_forms.py\`: log, karesel ve etkileşim yorum dönüşümleri.
- \`core/simulation.py\`: seed kontrollü ücret ve çıkarım DGP'si.
- \`core/datasets.py\`: dışarıdan sağlanan hazırlanmış CPS CSV adaptörü.
- \`core/causal.py\`: potansiyel sonuç, seçim ayrıştırması, içsellik limiti ve okul-kümeli deney DGP'si.
- \`core/iv.py\`: koşullu Wald, ilk aşama, robust 2SLS ve zayıf araç Monte Carlo hesapları.
- \`core/discrete.py\`: LPM, Logit, Probit ve ikili sonuç DGP sözleşmeleri.
- \`core/marginal_effects.py\`: AME, kukla sonlu farkı ve delta-yöntemi standart hataları.
- \`core/limited_outcomes.py\`: Tobit MLE, koşullu beklentiler ve Heckman iki aşama.
- \`core/quantile.py\`: check-loss, doğrusal program benchmark'ı ve robust kantil profilleri.
- \`core/nonparametric.py\`: kernel ağırlıkları, yerel sabit/doğrusal tahmin ve spline seri yaklaşımı.
- \`core/cross_validation.py\`: gözlem veya küme bazlı dış-kat bandwidth seçimi.
- \`core/partialling.py\`: kısmen doğrusal model için örneklem-içi esnek artıklaştırma.
- \`core/rdd.py\`: sharp/fuzzy yerel polinom RDD, etkin örneklem ve tasarım tanıları.
- \`core/resampling.py\`: gözlem veya küme biriminde seed'li yeniden örnekleme indeksleri.
- \`core/bootstrap.py\`: pairs/wild OLS bootstrap dağılımları ve güven aralıkları.
- \`core/regularization.py\`: seyrek DGP, Ridge/Lasso katsayı yolları ve yanlılık-varyans eğrisi.
- \`core/pipelines.py\`: kat-içi ölçeklemeli Lasso CV, 1-SE kuralı ve ayrılmış test karşılaştırması.
- \`core/dml.py\`: double selection, gözlem/grup bazlı cross-fitting ve ortogonal hedef tahmini.
- \`core/research_workflow.py\`: tahmin hedefinden yeniden üretilebilirliğe araştırma denetim aşamaları.
- \`core/code_recipes.py\`: 48 laboratuvar bölümü için bağımsız Python betiği ve Colab not defteri üretimi.

## Registry akışı

\`core/topic_registry.py\` 12 konu için tek başlık ve pedagojik metadata kaynağıdır. Sidebar seçenekleri, konu başlığı, araştırma sorusu, estimand, tanımlama odağı ve başlangıç soruları aynı kayıttan okunur.

\`core/data_registry.py\` veri kaynağı, gözlem birimi, örneklem kısıtı, beklenen sütunlar, izinli konular, küme ve yeniden örnekleme birimi ile yeniden dağıtım durumunu tutar. \`core/datasets.py\` bu şemayı çalışma zamanında doğrular; lisans kapısı çözülene kadar gerçek veri depoya alınmaz.

## State akışı

Konu değiştiğinde \`core/session_utils.py\` önceki konunun soru indeksini ve cevap görünürlüğünü sıfırlar. Metin ölçeği konu state'inden bağımsızdır. Yeni soru cevap görünürlüğünü kapatır.

## Sonuç sözleşmesi

\`core/types.py\` içindeki \`ModelResult\`, \`EstimandMetadata\`, \`InferenceSpec\` ve \`TuningSpec\` daha sonraki yöntem dallarının ortak sözleşmesidir. Kovaryans, cluster, seed, bandwidth, lambda, fold ve optimizasyon ayarları örtük varsayılan olarak bırakılmaz.

## Test katmanları

- Saf unit test: registry, metadata ve state geçişleri.
- Kaynak/kontrat testi: 12 başlık, render fonksiyonları ve private veri politikası.
- Streamlit AppTest: app entrypoint, konu geçişi, metin ölçeği ve soru/cevap state'i.
- Sayısal benchmark: OLS ve FWL eşitlikleri, statsmodels HC1/küme kovaryansı, delta yöntemi, etki tanıları ve deterministik DGP.
- Konu 01-02 AppTest: mekanizma uyarısı, veri modu, çıkarım seçimi, fonksiyonel biçim ve etkili gözlem state'i.
- Konu 03-04 AppTest: atama mekanizması, lisans kapıları, dışlama ihlali ve IV geçerlilik state'i.
- Konu 05-06 AppTest: olasılık-etki dönüşümü, Tobit yakınsaması, veri mekanizması ve seçim dışlama state'i.
- Konu 07-08 AppTest: kantil hedefi, kernel/bandwidth state'i ve CPS/DDK lisans kapıları.
- Konu 09-10 AppTest: RDD yönü, manipülasyon/ilk aşama uyarıları, bootstrap yöntemi, örnekleme birimi ve LM/CPS lisans kapıları.
- Konu 11-12 AppTest: lambda seçim kuralı, kat-içi ölçekleme, CPS/DDK lisans kapıları, OOF cross-fitting ve araştırma akışı state'i.
- Kod tarifi testi: 12 konunun dört bölümünü kapsayan 48 Python betiğinin derlenmesi ve dış veri olmadan çalışması; Colab JSON sözleşmesi.

## Öğrenci kodu akışı

Her konu sekmesi \`render_reproduction_code\` ile merkezi tarif kaydına bağlanır. Python çıktısı sabit rastgelelik tohumu, veri hazırlama, tahmin ve raporlama adımlarını içerir. Colab çıktısı aynı kodu ve açık paket kurulum hücresini taşır. Gerçek CPS/DDK dosyaları yalnız öğrenci tarafından sağlanır; yokluğunda kontrollü öğretim örneği kullanılır.

## Veri yayınlama kapısı

Gerçek CSV/DTA dosyaları \`references_private/\` altında yerel tutulur. Açık lisans veya yazılı yeniden dağıtım izni doğrulanmadan \`data/\` altında public kopya oluşturulmaz.
