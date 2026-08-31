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
        allowed_topics=("konu03", "konu08", "konu12"),
        cluster_variable="schoolid",
        resampling_unit="okul",
    ),
    DatasetMetadata(
        dataset_id="card1995",
        title="Card1995 - Koleje yakınlık ve eğitim",
        source="Card (1995); Hansen Econometrics kaynak veri paketi.",
        observation_unit="Birey",
        sample_definition="Ücret, eğitim, araç ve kontrol setinde complete-case 3.010 gözlem.",
        variables=tuple(
            _variable(name, label, label, unit)
            for name, label, unit in (
                ("lwage76", "1976 log ücret", "log birim"),
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
            )
        ),
        expected_columns=(
            "lwage76", "ed76", "nearc4", "exp76", "exp762_100", "black",
            "smsa76r", "reg76r", "smsa66r", "reg662", "reg663", "reg664",
            "reg665", "reg666", "reg667", "reg668", "reg669",
        ),
        allowed_topics=("konu04",),
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
        allowed_topics=("konu06",),
    ),
    DatasetMetadata(
        dataset_id="lm2007",
        title="LM2007 - Head Start RDD",
        source="Ludwig ve Miller (2007); Hansen Econometrics kaynak veri paketi.",
        observation_unit="İlçe veya coğrafi birim",
        sample_definition="Running variable ve ölüm oranı sonucu bulunan 2.810 gözlem.",
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
