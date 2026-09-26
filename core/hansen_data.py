"""Hansen'in Econometrics veri arşivi: çalışma anında indirme ve yükleme.

Veri depoya kopyalanmaz. Uygulama arşivi Hansen'in sayfasından indirir veya
kullanıcının yüklediği dosyayı okur. Hansen'in sayfası 2026'da ``~bhansen`` →
``~behansen`` adresine yönleniyor; önce güncel adres, sonra eski adres denenir.

Dosya biçimleri: ``cps09mar`` başlıksız ``.txt`` olarak okunur (değişken adları
açıklama belgesindeki sırayla). ``DDK2011``, ``Card1995``, ``CHJ2004`` ve ``LM2007`` Stata ``.dta``
olarak okunur: adlar dosyada kayıtlıdır ve üç dilde aynı olsun diye küçük harfe çevrilir.
"""

from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from dataclasses import dataclass

import pandas as pd

HANSEN_ARCHIVE_URLS = (
    "https://users.ssc.wisc.edu/~behansen/econometrics/Econometrics%20Data.zip",
    "https://www.ssc.wisc.edu/~bhansen/econometrics/Econometrics%20Data.zip",
)

CPS09MAR_COLUMNS = (
    "age",
    "female",
    "hisp",
    "education",
    "earnings",
    "hours",
    "week",
    "union",
    "uncov",
    "region",
    "race",
    "marital",
)
CPS09MAR_REQUIRED = ("age", "female", "education", "earnings", "hours", "week")
CPS09MAR_ROWS = 50742

DDK2011_REQUIRED = (
    "totalscore", "tracking", "schoolid", "std_mark", "girl", "agetest", "sbm", "etpteacher", "lowstream",
    "percentile",
)
CARD1995_REQUIRED = (
    "lwage76", "ed76", "nearc4", "age76", "black", "smsa76r", "reg76r", "smsa66r",
    *(f"reg66{i}" for i in range(2, 10)),
)
CHJ2004_CONTROLS = (
    "primary", "somesecondary", "secondary", "someuniversity", "university", "age", "married", "female",
    "marriedf", "child1", "child7", "child15", "size", "bothwork", "notemployed",
)
CHJ2004_REQUIRED = ("tabroad", "tdomestic", "tinkind", "tgifts", "income", "transfers", *CHJ2004_CONTROLS)
LM2007_REQUIRED = ("povrate60", "mort_age59_related_posths")

DATASET_MEMBERS = {
    "cps09mar": "cps09mar.txt", "ddk2011": "DDK2011.dta", "card1995": "Card1995.dta", "chj2004": "CHJ2004.dta",
    "lm2007": "LM2007.dta",
}
DATASET_COLUMNS = {"cps09mar": CPS09MAR_COLUMNS}
"""Başlıksız ``.txt`` dosyaları için değişken adları (açıklama belgesindeki sıra)."""
DATASET_REQUIRED = {
    "cps09mar": CPS09MAR_REQUIRED, "ddk2011": DDK2011_REQUIRED, "card1995": CARD1995_REQUIRED,
    "chj2004": CHJ2004_REQUIRED, "lm2007": LM2007_REQUIRED,
}
DATASET_COMPLETE = {"cps09mar": CPS09MAR_REQUIRED, "chj2004": CHJ2004_REQUIRED}
"""Eksik değer içermemesi gereken sütunlar. DDK2011 ve Card1995'te eksik değerler olağandır;
analiz örneklemi laboratuvar adımlarında açıkça kurulur."""
DATASET_ROWS = {"cps09mar": CPS09MAR_ROWS, "ddk2011": 5795, "card1995": 3613, "chj2004": 8684, "lm2007": 2783}
DATASET_SAMPLE = {"lm2007": LM2007_REQUIRED}
"""Tam örneklemi bu sütunlarda eksiksiz satır sayısıyla denetlenen veri setleri. Hansen'in LM2007.dta dosyası
(LM2007_create.do) eşik değişkeni veya sonuç değişkeni eksik ilçeleri çıkarır: 2.783 ilçe. Kaynak
headstart.dta ve ders notlarının öğretim CSV'si 2.810 satırdır; laboratuvar aynı 2.783 ilçeyi açıkça seçer."""


class HansenDataError(RuntimeError):
    """Veri indirilemediğinde veya dosya beklenen yapıda olmadığında."""


@dataclass(frozen=True)
class LoadedData:
    frame: pd.DataFrame
    source: str
    matches_hansen: bool


def download_archive(timeout: float = 120.0) -> bytes:
    """Hansen'in veri arşivini indirir; güncel adres başarısız olursa eskisini dener."""

    errors: list[str] = []
    for url in HANSEN_ARCHIVE_URLS:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (IKT807)"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as error:  # ağ hataları sınıflandırılmadan kullanıcıya iletilir
            errors.append(f"{url}: {error}")
    raise HansenDataError("Hansen veri arşivi indirilemedi. " + " | ".join(errors))


def find_member(archive: bytes, filename: str) -> str:
    """Arşivde klasör yapısından ve büyük/küçük harften bağımsız olarak dosyayı adıyla bulur."""

    target = filename.lower()
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        for name in bundle.namelist():
            if name.lower().rsplit("/", 1)[-1] == target:
                return name
    raise HansenDataError(f"{filename} Hansen arşivinde bulunamadı.")


def extract_member(archive: bytes, filename: str) -> bytes:
    member = find_member(archive, filename)
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        return bundle.read(member)


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def read_table(data: bytes, filename: str, dataset: str | None = None) -> pd.DataFrame:
    """Hansen'in veri dosyasını veya öğretim CSV'sini okur.

    Hansen'in ``.txt`` dosyaları başlıksızdır ve boşlukla ayrılır; değişken adları
    veri setinin açıklama belgesindeki sırayla atanır. Başlık satırı olan dosyalar
    (ör. ders notlarının öğretim CSV'si) adlarını kendi başlığından alır. ``.dta``
    dosyalarında adlar küçük harfe çevrilir; değer etiketleri kategoriye dönüştürülmez.
    """

    lowered = filename.lower()
    if lowered.endswith(".dta"):
        frame = pd.read_stata(io.BytesIO(data), convert_categoricals=False)
        frame.columns = [str(column).lower() for column in frame.columns]
        return frame
    if lowered.endswith((".xlsx", ".xls")):
        raise HansenDataError("Excel yerine .txt veya .dta dosyası yükleyin.")

    text = data.decode("utf-8-sig", errors="replace")
    first = next((line for line in text.splitlines() if line.strip()), "")
    if not first:
        raise HansenDataError("Dosya boş.")
    separator = "," if "," in first else r"\s+"
    tokens = [token for token in re.split(r"[,\s]+", first.strip()) if token]
    has_header = not all(_is_number(token) for token in tokens)
    if has_header:
        frame = pd.read_csv(io.StringIO(text), sep=separator)
        frame.columns = [str(column).lower() for column in frame.columns]
        return frame

    frame = pd.read_csv(io.StringIO(text), sep=separator, header=None)
    columns = DATASET_COLUMNS.get(dataset or "")
    if columns is None:
        raise HansenDataError("Başlıksız dosyada değişken adları bilinmiyor.")
    if frame.shape[1] != len(columns):
        raise HansenDataError(
            f"Başlıksız dosyada {frame.shape[1]} sütun var; Hansen'in {dataset} dosyasında "
            f"{len(columns)} sütun beklenir."
        )
    frame.columns = list(columns)
    return frame


def validate(dataset: str, frame: pd.DataFrame) -> bool:
    """Gerekli sütunları denetler; Hansen'in tam örneklemi mi olduğunu döndürür."""

    required = DATASET_REQUIRED[dataset]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise HansenDataError(
            "Dosyada beklenen sütunlar yok: " + ", ".join(missing)
            + f". Hansen'in {DATASET_MEMBERS[dataset]} dosyasını veya ders notlarının öğretim CSV'sini yükleyin."
        )
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    complete = DATASET_COMPLETE.get(dataset, ())
    if complete and frame[list(complete)].isna().any().any():
        raise HansenDataError("Gerekli sütunlarda eksik değer var.")
    sample = DATASET_SAMPLE.get(dataset)
    rows = int(frame[list(sample)].notna().all(axis=1).sum()) if sample else len(frame)
    return rows == DATASET_ROWS[dataset]


def load_from_archive(dataset: str, archive: bytes) -> LoadedData:
    member = DATASET_MEMBERS[dataset]
    frame = read_table(extract_member(archive, member), member, dataset)
    matches = validate(dataset, frame)
    return LoadedData(frame=frame, source="Hansen veri arşivi", matches_hansen=matches)


def load_from_upload(dataset: str, data: bytes, filename: str) -> LoadedData:
    frame = read_table(data, filename, dataset)
    matches = validate(dataset, frame)
    return LoadedData(frame=frame, source=f"Yüklenen dosya: {filename}", matches_hansen=matches)
