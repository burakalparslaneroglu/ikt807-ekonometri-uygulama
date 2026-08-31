"""IKT 807 konu sırası ve ortak pedagojik metadata kaydı."""

from __future__ import annotations

from core.types import TopicMetadata


TOPICS: tuple[TopicMetadata, ...] = (
    TopicMetadata(
        key="konu01",
        number=1,
        title="Ampirik Ekonometrik Modelleme ve Regresyonun Temeli",
        short_title="Ampirik Modelleme ve Regresyon",
        guiding_question="Eğitim ile ücret arasındaki gözlenen ilişki hangi anakütle büyüklüğünü özetliyor?",
        estimand="Ücretin eğitim ve gözlenen özelliklere göre koşullu ortalaması ile en iyi doğrusal öngörücü katsayısı.",
        identification_focus="Koşullu ilişki, doğrusal projeksiyon ve nedensel etki aynı nesne değildir.",
        application_focus="CPS ücret-egitim ilişkisi; koşullu ortalamadan çok sütunlu regresyon tablosuna geçiş.",
        dataset_ids=("cps09mar",),
        methods=("Koşullu ortalama", "OLS", "Doğrusal projeksiyon"),
        questions=(
            (
                "Koşullu ortalama doğrusal değilse OLS katsayısı anlamsız mı olur?",
                "Hayır. OLS, koşullu ortalamanın en iyi doğrusal öngörücüsünü verir; hedef artık tam koşullu ortalama değil doğrusal projeksiyondur.",
            ),
            (
                "Eğitim katsayısının pozitif olması tek başına nedensel eğitim getirisi gösterir mi?",
                "Hayır. Gözlemsel veride yetenek, aile geçmişi ve seçim gibi faktörler ek tanımlama varsayımları gerektirir.",
            ),
        ),
    ),
    TopicMetadata(
        key="konu02",
        number=2,
        title="Doğrusal Regresyonun Modern Kullanımı ve Güvenilir Çıkarım",
        short_title="Doğrusal Regresyon ve Çıkarım",
        guiding_question="Aynı katsayı farklı standart hata ve fonksiyonel biçim tercihleri altında nasıl okunmalıdır?",
        estimand="Belirli kontrol seti ve fonksiyonel biçim altında doğrusal koşullu ilişki.",
        identification_focus="Robust veya küme-dayanıklı standart hata katsayıyı ve tanımlamayı değiştirmez.",
        application_focus="CPS üzerinde model spesifikasyonu, HC1 ve ekonomik büyüklük karşılaştırması.",
        dataset_ids=("cps09mar",),
        methods=("FWL", "HC1", "Küme-dayanıklı çıkarım", "Delta yöntemi"),
        questions=(
            (
                "HC1 standart hata kullanıldığında OLS eğitim katsayısı neden aynı kalır?",
                "HC1 aynı OLS nokta tahminini kullanır; yalnız katsayının tahmini kovaryans matrisini değiştirir.",
            ),
            (
                "Daha küçük p-değeri daha büyük ekonomik etki anlamına gelir mi?",
                "Hayır. P-değeri etki büyüklüğünü değil, sıfır hipotezi altındaki örnekleme kanıtını belirsizlikle birlikte özetler.",
            ),
        ),
    ),
    TopicMetadata(
        key="konu03",
        number=3,
        title="Tanımlama, Nedensellik ve İçsellik",
        short_title="Tanımlama ve İçsellik",
        guiding_question="Gözlenen bir grup farkı hangi varsayımlar altında nedensel etkiye dönüşür?",
        estimand="Potansiyel sonuçlar çerçevesinde ortalama tedavi etkisi veya açıkça tanımlanmış koşullu etki.",
        identification_focus="Tanımlama veri dağılımını hedef parametreye bağlayan varsayımdır; iyi uyum onun yerine geçmez.",
        application_focus="DDK okul düzeyi rastgele ataması ile seçilmiş alt grup karşılaştırmasının ayrılması.",
        dataset_ids=("ddk2011",),
        methods=("Potansiyel sonuçlar", "İçsellik DGP", "Küme-dayanıklı çıkarım"),
        questions=(
            (
                "Örneklem büyüdükçe içsellik yanlılığı neden otomatik olarak kaybolmaz?",
                "Büyük örneklem yanlış hedef çevresindeki örnekleme belirsizliğini azaltabilir; açıklayıcı değişken ile hata ilişkisini ortadan kaldırmaz.",
            ),
            (
                "DDK uygulamasında standart hatalar neden okul düzeyinde kümelenir?",
                "Müdahale okul düzeyinde atanır ve aynı okuldaki öğrenci hataları bağımlı olabilir; çıkarım atama birimine saygı göstermelidir.",
            ),
        ),
    ),
    TopicMetadata(
        key="konu04",
        number=4,
        title="Araçsal Değişkenler ve İki Aşamalı En Küçük Kareler",
        short_title="Araçsal Değişkenler ve 2SLS",
        guiding_question="Koleje yakınlık eğitimin ücret etkisini hangi koşullar altında tanımlar?",
        estimand="Geçerli araç ve homojenlik/monotonluk koşullarına göre yapısal katsayı veya yerel ortalama tedavi etkisi.",
        identification_focus="Uygunluk, dışsallık ve dışlama kısıtı ayrı gerekçelerdir; ilk aşama yalnız uygunluğu gösterir.",
        application_focus="Card verisinde ilk aşama, indirgenmiş biçim, Wald oranı ve 2SLS.",
        dataset_ids=("card1995",),
        methods=("Wald", "2SLS", "Zayıf araç tanısı", "LATE"),
        questions=(
            (
                "Güçlü bir ilk aşama aracın dışlama kısıtını kanıtlar mı?",
                "Hayır. İlk aşama uygunluğu ölçer; aracın sonucu yalnız eğitim üzerinden etkilemesi ayrıca savunulmalıdır.",
            ),
            (
                "İkinci aşamada sıradan OLS standart hatası neden kullanılamaz?",
                "Tahmin edilen endojen değişken ilk aşamadan gelir. Sıradan OLS kovaryansı bu iki aşamalı tahmin belirsizliğini doğru taşımaz.",
            ),
        ),
    ),
    TopicMetadata(
        key="konu05",
        number=5,
        title="İkili ve Ayrık Sonuç Değişkenleri: Olasılık Modelleri ve Sınırlı Bağımlı Değişkenler",
        short_title="İkili ve Ayrık Sonuç Modelleri",
        guiding_question="Bir değişken evli olma olasılığıyla ne ölçüde ilişkilidir ve bu etki hangi ölçekte raporlanmalıdır?",
        estimand="Koşullu gerçekleşme olasılığı ve açık değerlendirme kuralıyla ortalama marjinal etki.",
        identification_focus="Logit ve Probit katsayıları tek indeks ölçeğindedir; doğrudan olasılık farkı değildir.",
        application_focus="CPS evlilik örneğinde LPM, Logit, Probit ve ortalama marjinal etkiler.",
        dataset_ids=("cps09mar",),
        methods=("LPM", "Logit", "Probit", "Ortalama marjinal etki"),
        questions=(
            (
                "Logit katsayısı neden yüzde puan değişimi olarak okunmaz?",
                "Katsayı log-odds/tek indeks ölçeğindedir. Olasılık etkisi kovaryat değerine bağlıdır ve marjinal etkiyle hesaplanır.",
            ),
            (
                "Kukla değişkende marjinal etki nasıl hesaplanmalıdır?",
                "Türev yerine değişken 0 ve 1 yapıldığında tahmin edilen olasılıklar arasındaki sonlu fark kullanılmalıdır.",
            ),
        ),
    ),
    TopicMetadata(
        key="konu06",
        number=6,
        title="Sansürleme, Kesilme ve Örneklem Seçimi",
        short_title="Sansürleme ve Örneklem Seçimi",
        guiding_question="Sıfır gözlenen transferler gizli sonuç, gözlenen sonuç ve seçim mekanizmasını nasıl ayırmamızı gerektirir?",
        estimand="Gizli sonuç, sansürlenmeme olasılığı veya gözlenen koşullu ortalama üzerindeki açıkça seçilmiş etki.",
        identification_focus="Sansürleme, kesilme ve örneklem seçimi farklı veri üretim süreçleridir.",
        application_focus="CHJ hane transferlerinde OLS, Tobit ve LAD; seçim için kontrollü Heckman DGP'si.",
        dataset_ids=("chj2004",),
        methods=("Tobit", "LAD", "Heckman iki aşama"),
        questions=(
            (
                "Tobit katsayısı gözlenen transfer üzerindeki marjinal etki midir?",
                "Hayır. Katsayı gizli sonuç denkleminin eğimidir; gözlenen ortalama ve pozitif olma olasılığı için ayrı marjinal etkiler gerekir.",
            ),
            (
                "Çok sayıda sıfır gözlem tek başına Tobit kullanmayı gerektirir mi?",
                "Hayır. Sıfırların ekonomik ve ölçümsel mekanizması sansürleme varsayımıyla uyumlu olmalıdır.",
            ),
        ),
    ),
    TopicMetadata(
        key="konu07",
        number=7,
        title="Kantil Regresyon ve Dağılımsal Heterojenlik",
        short_title="Kantil Regresyon",
        guiding_question="Eğitim-ücret ilişkisi koşullu ücret dağılımının farklı noktalarında nasıl değişir?",
        estimand="Belirli tau düzeyinde koşullu kantil fonksiyonunun katsayısı.",
        identification_focus="OLS koşullu ortalamayı, kantil regresyon koşullu kantili hedefler; farklılık hata göstergesi değildir.",
        application_focus="CPS ücret verisinde eğitim katsayısının tau boyunca profili.",
        dataset_ids=("cps09mar",),
        methods=("LAD", "Check loss", "Kantil regresyon"),
        questions=(
            (
                "OLS ile medyan regresyon katsayıları farklıysa hangisi doğrudur?",
                "İkisi farklı tahmin hedeflerini izler. Araştırma sorusu koşullu ortalama mı medyan mı istediğine göre yöntem seçilir.",
            ),
            (
                "İki kantil katsayısının ayrı ayrı anlamlı olması farklarının anlamlı olduğunu gösterir mi?",
                "Hayır. Katsayı farkının belirsizliği ortak kovaryans veya uygun bootstrap/test ile ayrıca değerlendirilmelidir.",
            ),
        ),
    ),
    TopicMetadata(
        key="konu08",
        number=8,
        title="Parametrik Olmayan ve Yarı Parametrik Regresyon",
        short_title="Parametrik Olmayan Regresyon",
        guiding_question="Başlangıç başarısı ile test puanı ilişkisini tek bir doğrusal biçime zorlamadan nasıl tahmin ederiz?",
        estimand="Seçilen düzgünleştirme kuralı altında koşullu ortalama fonksiyonu veya kısmen doğrusal hedef katsayı.",
        identification_focus="Parametrik olmayan yöntem varsayımsız değildir; bant genişliği ve baz seçimi düzenlilik ve ayar varsayımları taşır.",
        application_focus="DDK düzeyleme okullarındaki kız öğrencilerde yerel doğrusal eğri ve kümeli çapraz doğrulama.",
        dataset_ids=("ddk2011",),
        methods=("Çekirdek regresyon", "Yerel doğrusal", "Seri/eğri bazları", "Artıklaştırma"),
        questions=(
            (
                "Daha küçük bant genişliği her zaman daha iyi midir?",
                "Hayır. Daha yerel tahmin yanlılığı azaltabilir fakat varyansı artırır; veri ve araştırma amacı dengeyi belirler.",
            ),
            (
                "Kümeli veride çapraz doğrulama katları neden gözlem düzeyinde rastgele kurulmayabilir?",
                "Aynı kümenin eğitim ve değerlendirme tarafına dağılması bilgi sızıntısına ve aşırı iyimser hata ölçüsüne yol açabilir.",
            ),
        ),
    ),
    TopicMetadata(
        key="konu09",
        number=9,
        title="Regresyon Süreksizliği Tasarımı",
        short_title="Regresyon Süreksizliği",
        guiding_question="Bir uygunluk eşiği çevresindeki sıçrama hangi yerel nedensel etkiyi tanımlar?",
        estimand="Eşik noktasındaki yerel kesin etki veya bulanık tasarımda yerel Wald oranı.",
        identification_focus="Potansiyel sonuçların eşik çevresinde sürekliliği ve manipülasyon olmaması temel tasarım varsayımlarıdır.",
        application_focus="LM2007 Head Start verisinde yerel doğrusal RDD ve bant genişliği duyarlılığı.",
        dataset_ids=("lm2007",),
        methods=("Kesin RDD", "Bulanık RDD", "Yerel doğrusal", "Yoğunluk/sahte eşik tanıları"),
        questions=(
            (
                "En büyük sıçramayı veren bant genişliği neden otomatik olarak seçilmez?",
                "Bant genişliği sonuç avcılığı için değil önceden tanımlı veya veri-temelli yanlılık-varyans dengesiyle seçilmelidir.",
            ),
            (
                "RDD tahmini tüm anakütle için ortalama etki midir?",
                "Genellikle hayır. Tahmin eşik çevresindeki birimler için yerel etkidir.",
            ),
        ),
    ),
    TopicMetadata(
        key="konu10",
        number=10,
        title="Yeniden Örnekleme ile Çıkarım: Jackknife ve Bootstrap",
        short_title="Bootstrap ve Yeniden Örnekleme",
        guiding_question="Bir eğitim katsayısının örnekleme belirsizliğini analitik ve yeniden örnekleme yollarıyla nasıl karşılaştırırız?",
        estimand="Sabit modelde eğitim katsayısı; bootstrap hedefi bu tahmin edicinin örnekleme dağılımıdır.",
        identification_focus="Bootstrap belirsizliği yeniden hesaplar; model veya tanımlama hatasını onarmaz.",
        application_focus="CPS alt örnekleminde çiftler yeniden örneklemesi, HC1 ve güven aralığı karşılaştırması.",
        dataset_ids=("cps09mar",),
        methods=("Birini dışarıda bırakma", "Çiftler yeniden örneklemesi", "Çarpanlı yeniden örnekleme", "Öğrencileştirilmiş yüzdelik"),
        questions=(
            (
                "Bootstrap tekrar sayısı B arttığında hangi belirsizlik azalır?",
                "Bootstrap hesaplamasının Monte Carlo hatası azalır; orijinal örneklemin bilgi içeriği artmaz.",
            ),
            (
                "Kümeli veride satırları tek tek yeniden örneklemek neden yanlış olabilir?",
                "Bağımlılık yapısını bozabilir. Yeniden örnekleme birimi tasarımdaki bağımsız kümeler olmalıdır.",
            ),
        ),
    ),
    TopicMetadata(
        key="konu11",
        number=11,
        title="Model Seçimi, Çapraz Doğrulama ve Düzenlileştirme",
        short_title="Model Seçimi ve Düzenlileştirme",
        guiding_question="Dış örneklem tahmini için iyi model ile hedef katsayı çıkarımı için iyi model nasıl ayrılır?",
        estimand="Öngörü kaybı veya ayrıca tanımlanmış düşük boyutlu hedef parametre.",
        identification_focus="Düşük test hatası nedensel tanımlama sağlamaz; seçim sonrası klasik çıkarım otomatik geçerli değildir.",
        application_focus="CPS ücret tahmininde OLS, Ridge ve Lasso; kat içi ölçekleme ve ayrılmış sınama değerlendirmesi.",
        dataset_ids=("cps09mar",),
        methods=("K-katlı CV", "Ridge", "Lasso", "Elastic Net", "Post-Lasso"),
        questions=(
            (
                "Test seti lambda seçiminde kullanılırsa neden artık tarafsız değerlendirme verisi değildir?",
                "Model kararı test performansına uyarlanmış olur; raporlanan hata seçim sürecine bilgi sızdığı için iyimserleşir.",
            ),
            (
                "Lasso'dan sonra seçilen değişkenlerde OLS çalıştırmak klasik güven aralığını otomatik geçerli yapar mı?",
                "Hayır. Değişken seçiminin belirsizliği devam eder; hedefe özel selection-aware çıkarım gerekir.",
            ),
        ),
    ),
    TopicMetadata(
        key="konu12",
        number=12,
        title="Çift/Yanlılıktan Arındırılmış Makine Öğrenmesi ve Bütünleşik Araştırma Akışı",
        short_title="DML ve Araştırma Akışı",
        guiding_question="Esnek yardımcı modeller kullanırken düşük boyutlu hedef katsayı için geçerli çıkarımı nasıl koruruz?",
        estimand="Kısmen doğrusal modelde düşük boyutlu hedef katsayı theta.",
        identification_focus="Ortogonalizasyon ve çapraz uyarlama tanımlama varsayımının yerine geçmez; yalnız yardımcı tahmin hatasının etkisini yönetir.",
        application_focus="DDK deneyinde okul bazlı grup-katlı bölme ile OLS ve rastgele orman DML karşılaştırması.",
        dataset_ids=("ddk2011",),
        methods=("Çift seçim", "Artıklaştırma", "Ortogonal skor", "Çapraz uyarlama"),
        questions=(
            (
                "DML sonucu neden makine öğrenmesi sayesinde otomatik nedensel olmaz?",
                "Nedensel yorum randomization, unconfoundedness, IV veya başka bir tasarım varsayımından gelir; DML gözlenen yardımcı yapıyı esnek yönetir.",
            ),
            (
                "Aynı okulun öğrencileri neden farklı çapraz uyarlama katlarına bölünmemelidir?",
                "Model aynı okulun eğitim gözlemlerinden değerlendirme öğrencileri hakkında doğrudan bilgi öğrenebilir; grup-katlı bölme bu sızıntıyı önler.",
            ),
        ),
    ),
)

TOPICS_BY_KEY = {topic.key: topic for topic in TOPICS}


def list_topics() -> tuple[TopicMetadata, ...]:
    """Konuları ders sırasıyla döndürür."""

    return TOPICS


def get_topic(topic_key: str) -> TopicMetadata:
    """Konu anahtarını doğrular."""

    try:
        return TOPICS_BY_KEY[topic_key]
    except KeyError as error:
        raise ValueError(f"Desteklenmeyen konu: {topic_key}") from error
