"""Lisans kapılı Hansen öğretim verileri için metadata kaydı."""

from __future__ import annotations

from collections.abc import Iterable

from core.types import DatasetMetadata, VariableMetadata


def _variable(name: str, label: str, description: str, unit: str) -> VariableMetadata:
    return VariableMetadata(name=name, label=label, description=description, unit=unit)


DATASETS: tuple[DatasetMetadata, ...] = (
    DatasetMetadata(
        dataset_id="cps09mar",
        title="CPS 2009 - Ücret ve bireysel özellikler",
        source="Bruce E. Hansen, Econometrics veri paketi; Mart 2009 CPS.",
        observation_unit="Tam zamanlı çalışan birey",
        sample_definition="Ders notu üretim betiğindeki ücret ve çalışma süresi filtreleri; 50.742 gözlem.",
        variables=(
            _variable("age", "Yaş", "Bireyin yaşı.", "yıl"),
            _variable("female", "Kadın", "Kadın gösterge değişkeni.", "0/1"),
            _variable("hisp", "Hispanik", "Hispanik köken göstergesi.", "0/1"),
            _variable("education", "Eğitim", "Tamamlanan eğitim yılı.", "yıl"),
            _variable("earnings", "Kazanç", "Dönem kazancı.", "ABD doları"),
            _variable("hours", "Çalışma saati", "Haftalık çalışma saati.", "saat"),
            _variable("week", "Çalışılan hafta", "Yıl içinde çalışılan hafta.", "hafta"),
            _variable("hrwage", "Saatlik ücret", "Hesaplanan saatlik ücret.", "ABD doları/saat"),
            _variable("experience", "Deneyim", "Potansiyel iş deneyimi.", "yıl"),
            _variable("experience2_100", "Deneyim karesi", "Deneyim karesinin 100'e bölünmüş hali.", "yıl kare / 100"),
            _variable("region", "Bölge", "Yerleşim bölgesi kategorisi.", "kategori"),
            _variable("race", "Irk", "Irk kategorisi.", "kategori"),
            _variable("marital", "Medeni durum", "Medeni durum kategorisi.", "kategori"),
            _variable("lwage", "Log saatlik ücret", "Saatlik ücretin doğal logaritması.", "log birim"),
        ),
        expected_columns=(
            "age", "female", "hisp", "education", "earnings", "hours", "week",
            "hrwage", "experience", "experience2_100", "region", "race", "marital", "lwage",
        ),
        allowed_topics=("konu01", "konu02", "konu05", "konu07", "konu10", "konu11"),
    ),
    DatasetMetadata(
        dataset_id="ddk2011",
        title="DDK2011 - Okul tracking deneyi",
        source="Duflo, Dupas ve Kremer; Hansen Econometrics kaynak veri paketi.",
        observation_unit="Öğrenci",
        sample_definition="Hazırlanmış öğretim kopyasında 7.022 öğrenci; konuya göre complete-case filtreleri.",
        variables=(
            _variable("pupilid", "Öğrenci kimliği", "Öğrenci tanımlayıcısı.", "kimlik"),
            _variable("schoolid", "Okul kimliği", "Atama ve çıkarım kümesi.", "kimlik"),
            _variable("tracking", "Tracking", "Okul düzeyi müdahale göstergesi.", "0/1"),
            _variable("sbm", "Okul yönetim komitesi", "SBM göstergesi.", "0/1"),
            _variable("girl", "Kız öğrenci", "Cinsiyet göstergesi.", "0/1"),
            _variable("agetest", "Test yaşı", "Test tarihindeki yaş.", "yıl"),
            _variable("etpteacher", "Ek öğretmen", "Ek öğretmen müdahalesi göstergesi.", "0/1"),
            _variable("lowstream", "Düşük başarı sınıfı", "Tracking okulunda alt sınıf göstergesi.", "0/1"),
            _variable("std_mark", "Başlangıç standart puanı", "Başlangıç başarısı.", "standart puan"),
            _variable("percentile", "Başlangıç yüzdeliği", "Başlangıç başarı yüzdelik dilimi.", "yüzdelik"),
            _variable("totalscore", "Toplam test puanı", "Dönem sonu toplam puan.", "puan"),
        ),
        expected_columns=(
            "pupilid", "schoolid", "tracking", "sbm", "girl", "agetest",
            "etpteacher", "lowstream", "std_mark", "percentile", "totalscore",
        ),
        allowed_topics=("konu03", "konu08", "konu10", "konu12"),
        cluster_variable="schoolid",
        resampling_unit="okul",
    ),
    DatasetMetadata(
        dataset_id="card1995",
        title="Card1995 - Koleje yakınlık ve eğitim",
        source="Card (1995); Hansen Econometrics kaynak veri paketi.",
        observation_unit="Birey",
        sample_definition="Ücret, eğitim, araç ve kontrol setinde complete-case 3.010 gözlem; Konu 11 alternatifinde IQ dahil "
                          "aday değişkenlerin hepsi gözlenen 2.034, Konu 12 alternatifinde tedavi öncesi kontrolleri gözlenen "
                          "2.997 erkek. NLSYM hane temelli bir örneklemdir (kardeşler); dosyada hane kimliği yoktur.",
        variables=tuple(
            _variable(name, label, label, unit)
            for name, label, unit in (
                ("lwage76", "1976 log ücret", "log birim"),
                ("wage76", "1976 saatlik ücret", "sent"),
                ("age76", "1976 yaş", "yıl"),
                ("ed76", "1976 eğitim", "yıl"),
                ("nearc4", "Dört yıllık koleje yakınlık", "0/1"),
                ("exp76", "1976 deneyim", "yıl"),
                ("exp762_100", "Deneyim karesi", "yıl kare / 100"),
                ("black", "Siyah", "0/1"),
                ("smsa76r", "1976 metropolitan alan", "0/1"),
                ("reg76r", "1976 bölge", "kategori"),
                ("smsa66r", "1966 metropolitan alan", "0/1"),
                ("reg662", "1966 bölge 2", "0/1"),
                ("reg663", "1966 bölge 3", "0/1"),
                ("reg664", "1966 bölge 4", "0/1"),
                ("reg665", "1966 bölge 5", "0/1"),
                ("reg666", "1966 bölge 6", "0/1"),
                ("reg667", "1966 bölge 7", "0/1"),
                ("reg668", "1966 bölge 8", "0/1"),
                ("reg669", "1966 bölge 9", "0/1"),
                ("south66", "1966 güney", "0/1"),
                ("momed", "Anne eğitimi", "yıl"),
                ("daded", "Baba eğitimi", "yıl"),
                ("momdad14", "14 yaşında anne ve babayla", "0/1"),
                ("libcrd14", "14 yaşında kütüphane kartı", "0/1"),
                ("enroll76", "1976'da okula kayıtlı", "0/1"),
                ("kww", "KWW bilgi testi", "puan"),
                ("iq", "IQ", "puan"),
            )
        ),
        expected_columns=(
            "lwage76", "wage76", "age76", "ed76", "nearc4", "exp76", "exp762_100", "black",
            "smsa76r", "reg76r", "smsa66r", "reg662", "reg663", "reg664",
            "reg665", "reg666", "reg667", "reg668", "reg669", "south66", "momed", "daded", "momdad14", "libcrd14",
            "enroll76", "kww", "iq",
        ),
        allowed_topics=("konu01", "konu02", "konu04", "konu07", "konu11", "konu12"),
    ),
    DatasetMetadata(
        dataset_id="chj2004",
        title="CHJ2004 - Hane transferleri",
        source="Cox, Hansen ve Jimenez (2004); Hansen Econometrics kaynak veri paketi.",
        observation_unit="Hane",
        sample_definition="Düzeltilmiş gelirin üst yüzde 2'si ve negatif gelir çıkarıldı; 8.684 hane.",
        variables=tuple(
            _variable(name, label, label, unit)
            for name, label, unit in (
                ("received", "Alınan transfer", "peso"),
                ("income_adj", "Düzeltilmiş gelir", "peso"),
                ("primary", "İlkokul", "0/1"),
                ("somesecondary", "Bir miktar ortaöğretim", "0/1"),
                ("secondary", "Ortaöğretim", "0/1"),
                ("someuniversity", "Bir miktar üniversite", "0/1"),
                ("university", "Üniversite", "0/1"),
                ("age10c", "Yaş ölçeği", "10 yıl"),
                ("married", "Evli", "0/1"),
                ("female", "Kadın", "0/1"),
                ("marriedf", "Evli kadın etkileşimi", "0/1"),
                ("child1", "0-1 yaş çocuk", "adet"),
                ("child7", "2-7 yaş çocuk", "adet"),
                ("child15", "8-15 yaş çocuk", "adet"),
                ("size", "Hane büyüklüğü", "kişi"),
                ("bothwork", "İki eş de çalışıyor", "0/1"),
                ("notemployed", "İstihdamda değil", "0/1"),
            )
        ),
        expected_columns=(
            "received", "income_adj", "primary", "somesecondary", "secondary",
            "someuniversity", "university", "age10c", "married", "female",
            "marriedf", "child1", "child7", "child15", "size", "bothwork", "notemployed",
        ),
        allowed_topics=("konu05", "konu06"),
    ),
    DatasetMetadata(
        dataset_id="lm2007",
        title="LM2007 - Head Start RDD",
        source="Ludwig ve Miller (2007); Hansen Econometrics kaynak veri paketi.",
        observation_unit="İlçe veya coğrafi birim",
        sample_definition="Eşik değişkeni ve ölüm oranı sonucu eksik olmayan 2.783 ilçe (kaynak dosya 2.810 satır).",
        variables=(
            _variable("povrate60", "1960 yoksulluk oranı", "RDD eşik değişkeni.", "oran"),
            _variable(
                "mort_age59_related_postHS",
                "Head Start ilişkili ölüm oranı",
                "5-9 yaş Head Start ilişkili ölüm sonucu.",
                "ölüm oranı",
            ),
        ),
        expected_columns=("povrate60", "mort_age59_related_postHS"),
        allowed_topics=("konu09",),
    ),
    DatasetMetadata(
        dataset_id="ds2004",
        title="DS2004 - Saldırı sonrası polis koruması ve araç hırsızlığı",
        source="Di Tella ve Schargrodsky (2004); Hansen Econometrics kaynak veri paketi.",
        observation_unit="Blok × ay",
        sample_definition="Buenos Aires'in üç mahallesinde 876 blok, Nisan–Aralık 1994 (Temmuz yalnız 1–17); 7.884 satır. "
                          "Uygulama sekmesinin alternatif örneği saldırı sonrası tam ayları (Ağustos–Aralık) kullanır.",
        variables=tuple(
            _variable(name, label, label, unit)
            for name, label, unit in (
                ("block", "Blok", "kimlik"),
                ("barrio", "Mahalle", "kategori"),
                ("sameblock", "Blokta Yahudi kurumu", "0/1"),
                ("distance", "En yakın Yahudi kurumuna uzaklık", "blok"),
                ("public", "Blokta kamu binası", "0/1"),
                ("gasstation", "Blokta benzin istasyonu", "0/1"),
                ("bank", "Blokta banka", "0/1"),
                ("thefts", "Aylık araç hırsızlığı", "adet"),
                ("month", "Ay", "ay"),
                ("oneblock", "Bir blok uzakta Yahudi kurumu", "0/1"),
            )
        ),
        expected_columns=("block", "barrio", "sameblock", "distance", "public", "gasstation", "bank", "thefts",
                          "month", "oneblock"),
        allowed_topics=("konu03",),
        cluster_variable="block",
    ),
    DatasetMetadata(
        dataset_id="ak1991",
        title="AK1991 - Doğum çeyreği ve eğitim",
        source="Angrist ve Krueger (1991); Hansen Econometrics kaynak veri paketi.",
        observation_unit="Birey",
        sample_definition="1980 nüfus sayımı, 1930–1939 doğumlu erkekler; 329.509 gözlem.",
        variables=tuple(
            _variable(name, label, label, unit)
            for name, label, unit in (
                ("logwage", "Log haftalık kazanç", "log dolar"),
                ("edu", "Tamamlanan eğitim", "yıl"),
                ("qob", "Doğum çeyreği", "1–4"),
                ("yob", "Doğum yılı", "yıl"),
                ("ageq", "Çeyrek dahil yaş", "yıl"),
                ("black", "Siyah", "0/1"),
                ("smsa", "Metropol alanda çalışıyor", "0/1"),
                ("married", "Evli ve eşiyle yaşıyor", "0/1"),
                ("region", "Bölge", "kategori"),
                ("state", "Doğum eyaleti", "FIPS kodu"),
            )
        ),
        expected_columns=("logwage", "edu", "qob", "yob", "ageq", "black", "smsa", "married", "region", "state"),
        allowed_topics=("konu04",),
    ),
    DatasetMetadata(
        dataset_id="ck1994",
        title="CK1994 - Asgari ücret ve fast-food istihdamı",
        source="Card ve Krueger (1994); Hansen Econometrics kaynak veri paketi.",
        observation_unit="Restoran × anket dalgası",
        sample_definition="New Jersey ve Pennsylvania'da 410 fast-food restoranı, 1992 Şubat–Mart ve Kasım–Aralık "
                          "anketleri; 820 satır. Uygulama sekmesinin alternatif örneği tam zamanlı çalışan sayısını "
                          "kullanır.",
        variables=tuple(
            _variable(name, label, label, unit)
            for name, label, unit in (
                ("store", "Restoran", "kimlik"),
                ("chain", "Zincir", "1–4"),
                ("co_owned", "Şirkete ait", "0/1"),
                ("state", "New Jersey", "0/1"),
                ("empft", "Tam zamanlı çalışan", "kişi"),
                ("emppt", "Yarı zamanlı çalışan", "kişi"),
                ("nmgrs", "Yönetici", "kişi"),
                ("wage_st", "Başlangıç ücreti", "dolar/saat"),
                ("hoursopen", "Günlük açık saat", "saat"),
                ("nregisters", "Kasa sayısı", "adet"),
                ("time", "İkinci anket dalgası", "0/1"),
            )
        ),
        expected_columns=("store", "chain", "co_owned", "state", "empft", "emppt", "nmgrs", "wage_st", "hoursopen",
                          "nregisters", "time"),
        allowed_topics=("konu06",),
        cluster_variable="store",
    ),
    DatasetMetadata(
        dataset_id="al1999",
        title="AL1999 - Sınıf büyüklüğü ve başarı (Maimonides kuralı)",
        source="Angrist ve Lavy (1999); Hansen Econometrics kaynak veri paketi.",
        observation_unit="Sınıf",
        sample_definition="İsrail'de 4. ve 5. sınıflar (1991); eksik test puanı, 44'ten büyük sınıf ve 6'dan küçük "
                          "kayıt çıkarıldı; 4.067 sınıf. Uygulama sekmesinin alternatif örnekleri Konu 8'de 4. sınıfları, "
                          "Konu 9'da 5. sınıfları kullanır.",
        variables=tuple(
            _variable(name, label, label, unit)
            for name, label, unit in (
                ("schlcode", "Okul", "kimlik"),
                ("grade", "Sınıf düzeyi", "4/5"),
                ("classid", "Sınıf sıra numarası", "sıra"),
                ("classize", "Sınıf mevcudu", "öğrenci"),
                ("enrollment", "Okulun o sınıf düzeyindeki öğrenci sayısı", "öğrenci"),
                ("avgmath", "Matematik ortalaması", "puan"),
                ("avgverb", "Okuma (dil) ortalaması", "puan"),
                ("disadvantaged", "Dezavantajlı öğrenci yüzdesi", "%"),
            )
        ),
        expected_columns=("schlcode", "grade", "classid", "classize", "enrollment", "avgmath", "avgverb",
                          "disadvantaged"),
        allowed_topics=("konu08", "konu09"),
        cluster_variable="schlcode",
        resampling_unit="okul",
    ),
)

DATASETS_BY_ID = {dataset.dataset_id: dataset for dataset in DATASETS}


def list_datasets() -> tuple[DatasetMetadata, ...]:
    """Veri setlerini kararlı sırada döndürür."""

    return DATASETS


def get_dataset_metadata(dataset_id: str) -> DatasetMetadata:
    """Veri seti anahtarını doğrular."""

    try:
        return DATASETS_BY_ID[dataset_id]
    except KeyError as error:
        raise ValueError(f"Desteklenmeyen veri seti: {dataset_id}") from error


def validate_dataset_columns(dataset_id: str, columns: Iterable[str]) -> None:
    """Beklenen sütunların tamamının bulunduğunu doğrular."""

    metadata = get_dataset_metadata(dataset_id)
    available = set(columns)
    missing = sorted(set(metadata.expected_columns) - available)
    if missing:
        raise ValueError(
            f"{metadata.title} beklenen sütunları içermiyor: {', '.join(missing)}"
        )
