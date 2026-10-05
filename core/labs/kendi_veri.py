"""Öğrencinin kendi veri dosyası: okuma, sütun seçimi, temizleme ve doğrulama ("Kendi verini yükle").

Uygulama dosyayı üretilen kodun okuyacağı biçimde okur. Excel için ``pandas.read_excel`` (openpyxl) kullanılır; CSV için
ayırıcı, ondalık işareti ve karakter kodlaması algılanır ve kodda açıkça yazılır. Temizleme kuralları ``ReadFile``
belgesindekilerle ve üretilen Python/R koduyla aynıdır (``clean_name``, ``clean_text``, ``code_text``); bu yüzden
uygulamanın sayıları kodun sayılarıyla aynı çıkar. İki dilin aynı sonucu vereceği güvence altına alınamayan dosya ve
sütunlar açık bir iletiyle reddedilir ya da seçeneklerden çıkarılır.

Okuma ve temizleme kuralları İKT 217 ve İKT 305 uygulamalarıyla aynıdır. Dosya yalnız bellekte işlenir: diske, ortak
önbelleğe ya da günlüğe yazılmaz.
"""

from __future__ import annotations

import csv
import datetime
import io
import keyword
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from core.labs.spec import ReadFile

MAX_MEGABYTES = 5
MAX_BYTES = MAX_MEGABYTES * 1024 * 1024
MAX_ROWS = 10_000
MAX_LABEL = 40
"""Kategori etiketinin en çok karakter sayısı (tablolar ve grafikler okunur kalsın)."""
MAX_EXTRA = 12
"""Ek sütun seçiminde ilk açılışta önerilen en çok sütun."""
MAX_TEXT_INTEGER = 10 ** 15
"""Sayıların ve kodların büyüklük sınırı: üstünde Python ile R aynı değeri ya da yazımı vermeyebilir (2^53 üstündeki
tam sayılar, int64 toplamlarında taşma)."""
NA_VALUES = ("", "NA")
"""Eksik değer sayılan hücreler (iki dilde aynı): boş hücre ve NA yazısı."""
ORDER_RULES = {
    "alfabetik": "Alfabetik (Türkçe sıra)",
    "dosya": "Dosyadaki ilk görülme sırası",
    "frekans": "Frekansa göre (çoktan aza)",
}
USES = ("kategorik", "sayisal", "serbest")
"""Sütunun kullanımı: kategorik değişken, sayısal değişken ya da olduğu gibi (yalnız veri tablosunda gösterilir)."""

_R_RESERVED = {
    "if", "else", "repeat", "while", "function", "for", "next", "break", "TRUE", "FALSE", "NULL", "Inf", "NaN", "NA",
    "in",
}
RESERVED_CODES = frozenset((
    # genel uygulamaların türettiği sütunlar (ör. log sonuç, gösterge, etkileşim, kontrolün karesi)
    "log_sonuc", "grup01", "etkilesim", "kontrol_kare_100", "veri", "ham", "tedavi01", "secilmis01", "icsel_hat",
    "sonuc01", "gosterge01", "sifir_yuzde", "dugum1", "dugum2", "dugum3", "kare_terim", "kup_terim",
    # genel uygulamaların model adları (etiket sözlüğünde öğrencinin sütunuyla karışmasınlar)
    "ols", "lad", "tobit", "lpm", "logit", "probit", "ame_logit", "ame_probit", "ilk", "indirgenmis", "iv", "elle",
    "q10", "q25", "q50", "q75", "q90", "dogrusal", "kubik", "spline",
    # notlardaki ve Blok A–B alternatif örneklerindeki bazı türetilmiş adlar (kodlar yan yana okunurken karışmasın;
    # sonraki alternatiflerin genel sözcük olan adları, ör. gelir, egitim, ayrılmaz)
    "hrwage", "lwage", "experience", "experience2_100", "edu_female", "ed_black", "onceki", "yuksek", "q1", "edu_hat",
    # sonuç tablolarının sütunları ve ekranda Türkçe etiketi sabit olan adlar (``topics.lab_ui``)
    "n", "ort_sonuc", "ort_duzey", "medyan_duzey", "ortalama", "gozlem", "katsayi", "klasik_sh", "hc1_sh", "r2",
    "model", "terim", "deger", "tahmin", "sh", "p", "alt", "ust", "count", "mean", "std", "sd", "median", "min",
    "max", "sum", "coef", "se", "t", "nobs", "intercept", "sabit", "x", "y",
))
"""Öğrencinin sütunlarına verilmeyen kod adları: uygulamanın türettiği sütunlarla ve sabit etiketli tablo sütunlarıyla
karışmasınlar (ör. "Gözlem" sütunu ``gozlem_2`` olur). Ekrandaki ad dosyadaki addır."""
_ASCII = str.maketrans("çğıöşüÇĞİÖŞÜâîûÂÎÛ", "cgiosuCGIOSUaiuAIU")
_ALPHABET = "abcçdefgğhıijklmnoöprsştuüvyz"
_NUMBER_TEXT = re.compile(r"[+-]?[0-9]+(?:[.,][0-9]+)?")
_GROUPED_NUMBER = re.compile(r"[+-]?[0-9]{1,3}(?:[.,][0-9]{3})+(?:[.,][0-9]+)?")
_DOT_DECIMAL = re.compile(r"^\s*[+-]?[0-9]+\.[0-9]+\s*$", re.MULTILINE)
_COMMA_DECIMAL = re.compile(r"^\s*[+-]?[0-9]+,[0-9]+\s*$", re.MULTILINE)
_ID_NAMES = {"no", "id", "sıra", "sira", "sıra no", "sira no", "numara", "kimlik", "kimlik no", "#"}


def thousands(value: int) -> str:
    """Binlik ayırıcısı nokta olan tam sayı (10.000)."""

    return f"{int(value):,}".replace(",", ".")


def _list(items: list[str]) -> str:
    """Türkçe sıralama: "A", "A ve B", "A, B ve C"."""

    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " ve " + items[-1]


def _quoted(items: Iterable[object], limit: int = 3) -> str:
    shown = [f"“{_printable(item)}”" for item in list(items)[:limit]]
    return _list(shown) if shown else ""


def _printable(value: object) -> str:
    """İletide gösterilecek metin: denetim karakterleri görünür bir işarete çevrilir."""

    return "".join("�" if unicodedata.category(character) in ("Cc", "Zl", "Zp") else character
                   for character in str(value))


class UploadError(ValueError):
    """Öğrenciye gösterilecek, dosyanın ya da seçimin kullanılmasını engelleyen hata."""


# --- İki dilde aynı temizleme kuralları -----------------------------------------------------------

def clean_name(value: object) -> str:
    """Sütun adı: bölünmez boşluk boşluğa çevrilir, baştaki ve sondaki boşluklar silinir (koddaki kural)."""

    return str(value).replace("\xa0", " ").strip(" \t\r\n")


def clean_text(value: object) -> str | None:
    """Metin hücresi: üretilen koddaki ``temiz_metin`` ile aynı kural. Boş kalan hücre ve NA eksik değerdir."""

    if pd.isna(value):
        return None
    text = str(value).replace("\xa0", " ").strip(" \t\r\n")
    return None if text in NA_VALUES else text


def code_text(value: object) -> str | None:
    """Tam sayı kodu kategori etiketi olur (2.0 → "2"): üretilen koddaki ``kod_metni`` ile aynı kural."""

    return None if pd.isna(value) else str(int(value))


def _bad_characters(text: str) -> bool:
    """Kodda dizge ya da açıklama satırı olarak yazılamayan işaretler: tırnak, ters bölü ve denetim karakterleri."""

    return any(mark in text for mark in ('"', "\\")) or any(
        unicodedata.category(character) in ("Cc", "Zl", "Zp") for character in text)


# --- Tablolar -----------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class UploadedTable:
    """Yüklenen dosyanın okunmuş hâli ve kodda kullanılacak okuma ayarları.

    ``frame`` yalnız kullanılabilir sütunları taşır; sütun adları temizlenmiştir (``clean_name``). ``strip_names``:
    dosyadaki adlardan en az biri temizlenince değişir; kod da adları aynı kuralla temizler. ``notes``: seçeneklere
    alınmayan sütunlar için öğrenciye gösterilecek açıklamalar.
    """

    file_name: str
    file_format: str
    frame: pd.DataFrame
    sheet: str | None = None
    sheets: tuple[str, ...] = ()
    separator: str = ","
    decimal: str = "."
    encoding: str = "utf-8-sig"
    strip_names: bool = False
    notes: tuple[str, ...] = ()

    @property
    def columns(self) -> list[str]:
        return [str(column) for column in self.frame.columns]


@dataclass(frozen=True)
class Selection:
    """Analize alınan bir sütun: koddaki adı, dosyadaki adı, kullanımı (``USES``) ve zorunlu olup olmadığı.

    Zorunlu bir sütunda değeri olmayan satırlar analizden çıkarılır; diğer sütunlardaki boş hücreler yerinde kalır.
    """

    name: str
    original: str
    use: str
    required: bool = False


@dataclass(frozen=True, eq=False)
class Prepared:
    """Temizlenmiş veri, onu tanımlayan ``ReadFile`` işlemi ve öğrenciye gösterilecek notlar."""

    read: ReadFile
    frame: pd.DataFrame
    notes: tuple[str, ...] = field(default_factory=tuple)


# --- Okuma -----------------------------------------------------------------------------

def excel_sheets(data: bytes) -> list[str]:
    try:
        return [str(name) for name in pd.ExcelFile(io.BytesIO(data), engine="openpyxl").sheet_names]
    except Exception as error:  # bozuk ya da parola korumalı dosya
        raise UploadError("Excel dosyası açılamadı. Dosyanın .xlsx biçiminde ve parolasız olduğunu kontrol edin.") \
            from error


def _decode(data: bytes) -> tuple[str, str]:
    """CSV dosyasının karakter kodlaması ve metni. Dosyanın tamamı çözülür: önce UTF-8 (BOM'lu ya da BOM'suz), olmazsa
    Windows-1254 (Türkçe)."""

    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise UploadError("CSV dosyası UTF-16 kodlamalı. Excel'de “CSV UTF-8 (virgülle ayrılmış)” ya da “CSV "
                          "(noktalı virgülle ayrılmış)” biçiminde yeniden kaydedin.")
    for encoding in ("utf-8-sig", "cp1254"):
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        if "\x00" in text:
            raise UploadError("CSV dosyası metin biçiminde değil (boş karakterler içeriyor). Dosyayı Excel'de CSV "
                              "olarak yeniden kaydedin.")
        return encoding, text
    raise UploadError("CSV dosyasının karakter kodlaması tanınmadı. Dosyayı UTF-8 olarak kaydedin.")


def _separator(text: str) -> str:
    sample = text[:65536].replace("\r\n", "\n").replace("\r", "\n")
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,\t").delimiter
    except csv.Error:
        first = next((line for line in sample.splitlines() if line.strip()), "")
        return max((";", ",", "\t"), key=first.count)


def _decimal(text: str, separator: str) -> str:
    """Ondalık işareti. Türkçe Excel "CSV (noktalı virgülle ayrılmış)" kaydında ayırıcı ``;``, ondalık işareti
    virgüldür; virgülle ya da sekmeyle ayrılmış dosyada noktadır. Tek sütunlu dosyada ayırıcı yoktur; ondalık işareti
    sayıların yazımından belirlenir."""

    first = next((line for line in text.splitlines() if line.strip()), "")
    if separator in first:
        return "," if separator == ";" else "."
    return "." if len(_DOT_DECIMAL.findall(text)) > len(_COMMA_DECIMAL.findall(text)) else ","


def detect_csv(data: bytes) -> tuple[str, str, str]:
    """CSV dosyasının karakter kodlaması, ayırıcısı ve ondalık işareti (metin olarak yazılmış sayılar ayrıca
    denetlenir)."""

    encoding, text = _decode(data)
    separator = _separator(text)
    return encoding, separator, _decimal(text, separator)


def _stray_quotes(text: str, separator: str) -> bool:
    """Tırnak işareti yalnız hücrenin başında açılıp sonunda kapanabilir (içerideki tırnak iki kez yazılır: "a""b").
    pandas hücre ortasındaki tırnağı harf sayar, R ise alıntının başı sayar; böyle bir dosyada iki dil ayrışır."""

    quoted, start = False, True
    index, size = 0, len(text)
    while index < size:
        character = text[index]
        if quoted:
            if character == '"':
                if index + 1 < size and text[index + 1] == '"':
                    index += 2
                    continue
                quoted = False
                following = text[index + 1] if index + 1 < size else "\n"
                if following not in (separator, "\n", "\r"):
                    return True
            index += 1
            continue
        if character == '"':
            if not start:
                return True
            quoted = True
        start = character in (separator, "\n", "\r")
        index += 1
    return quoted


def _csv_header(text: str, separator: str) -> list[str]:
    """Başlık satırının ham hücreleri (pandas'ın sütun adlarını değiştirmeden önceki hâli)."""

    try:
        for row in csv.reader(io.StringIO(text, newline=""), delimiter=separator):
            if not row:
                continue  # boş satır: pandas ve R atlar
            if len(row) == 1 and not row[0].strip():
                break
            return row
    except csv.Error as error:
        raise UploadError("CSV dosyasının başlık satırı okunamadı.") from error
    raise UploadError("Dosyanın ilk satırı boş ya da yalnız boşluk içeriyor; sütun adları ilk satırda olmalı.")


def read_upload(file_name: str, data: bytes, sheet: str | None = None) -> UploadedTable:
    """Dosyayı üretilen kodun okuyacağı ayarlarla okur ve kullanılabilir sütunları belirler."""

    if len(data) > MAX_BYTES:
        raise UploadError(f"Dosya {MAX_MEGABYTES} MB'tan büyük. Daha küçük bir dosya yükleyin ya da dosyada yalnız "
                          "gereken sütunları bırakın.")
    lower = file_name.lower()
    if lower.endswith(".xlsx"):
        sheets = tuple(excel_sheets(data))
        chosen = sheet if sheet in sheets else sheets[0]
        try:
            frame = pd.read_excel(io.BytesIO(data), sheet_name=chosen, na_values=list(NA_VALUES),
                                  keep_default_na=False, engine="openpyxl")
            first = pd.read_excel(io.BytesIO(data), sheet_name=chosen, header=None, nrows=1, na_filter=False,
                                  dtype=object, engine="openpyxl")
        except Exception as error:
            raise UploadError(f"“{chosen}” sayfası okunamadı.") from error
        header = list(first.iloc[0]) if len(first) else []
        settings = {"file_format": "xlsx", "sheet": chosen, "sheets": sheets}
    elif lower.endswith(".csv"):
        encoding, text = _decode(data)
        separator = _separator(text)
        decimal = _decimal(text, separator)
        if any(line and not line.strip() and separator not in line for line in text.splitlines()):
            # pandas böyle bir satırı atlar, R boş bir gözlem sayabilir: satır sayıları ayrışmasın
            raise UploadError("Dosyada yalnız boşluk içeren satırlar var; bu satırları silin.")
        if '"' in text and _stray_quotes(text, separator):
            raise UploadError('CSV dosyasında bir hücrenin ortasında tırnak işareti (") var; R bu hücreyi farklı okur. '
                              "Tırnakları silin ya da dosyayı Excel'de CSV olarak yeniden kaydedin.")
        header = _csv_header(text, separator)
        try:
            frame = pd.read_csv(io.BytesIO(data), sep=separator, decimal=decimal, encoding=encoding,
                                na_values=list(NA_VALUES), keep_default_na=False)
        except Exception as error:
            raise UploadError("CSV dosyası okunamadı. İlk satırda sütun adları olmalı ve her satırda aynı sayıda "
                              "alan bulunmalı.") from error
        settings = {"file_format": "csv", "separator": separator, "decimal": decimal, "encoding": encoding}
    else:
        raise UploadError("Yalnız Excel (.xlsx) ve CSV (.csv) dosyaları okunur.")
    if frame.shape[1] == 0 or len(frame) == 0:
        raise UploadError("Dosyada veri bulunamadı. İlk satırda sütun adları, altında gözlemler olmalı.")
    if not frame.index.equals(pd.RangeIndex(len(frame))):
        raise UploadError("Başlık satırında veri satırlarından daha az alan var; ilk satırdaki her sütuna bir ad "
                          "yazın.")
    if len(frame) > MAX_ROWS:
        raise UploadError(f"Dosyada {thousands(len(frame))} satır var; en çok {thousands(MAX_ROWS)} satır okunur.")
    usable, strip_names, notes = _usable_columns(frame, header)
    return UploadedTable(file_name=file_name, frame=usable, strip_names=strip_names, notes=notes, **settings)


def _blank(cell: object) -> bool:
    return cell is None or (isinstance(cell, float) and math.isnan(cell)) or (isinstance(cell, str) and not cell)


def _usable_columns(frame: pd.DataFrame, header: list[object]) -> tuple[pd.DataFrame, bool, tuple[str, ...]]:
    """Başlık satırından kullanılabilir sütunlar. Adı metin olmayan, adsız, adı "NA" olan ya da kodda yazılamayan
    işaretler içeren sütunlar seçeneklere alınmaz; aynı adı taşıyan sütunlar dosyayı reddettirir (pandas ve R yinelenen
    adları farklı biçimde değiştirir)."""

    width = frame.shape[1]
    cells = list(header)[:width] + [None] * max(0, width - len(header))
    if all(_blank(cell) or (isinstance(cell, str) and not clean_name(cell)) for cell in cells):
        raise UploadError("Dosyanın ilk satırı boş; sütun adları ilk satırda olmalı.")
    positions: list[int] = []
    names: list[str] = []
    seen: list[str] = []
    not_text: list[str] = []
    unnamed: list[int] = []
    unusable: list[str] = []
    for position, cell in enumerate(cells):
        has_data = bool(frame.iloc[:, position].notna().any())
        if _blank(cell):
            if has_data:
                unnamed.append(position + 1)
            continue
        if not isinstance(cell, str):
            not_text.append(str(cell))
            seen.append(clean_name(cell))
            continue
        name = clean_name(cell)
        if not name:
            if has_data:
                unnamed.append(position + 1)
            continue
        seen.append(name)
        if name in NA_VALUES or _bad_characters(name):
            unusable.append(name)
            continue
        positions.append(position)
        names.append(name)
    duplicates = sorted({name for name in seen if seen.count(name) > 1}, key=turkish_key)
    if duplicates:
        raise UploadError(f"Aynı adı taşıyan sütunlar var: {_quoted(duplicates, 5)}. Dosyada sütun adlarını farklı "
                          "yapın (baştaki ve sondaki boşluklar ad farkı sayılmaz).")
    if not positions:
        raise UploadError("Dosyada kullanılabilir adı olan bir sütun yok. İlk satırdaki her sütuna metin bir ad "
                          "yazın.")
    notes = []
    if not_text:
        notes.append(f"Adı sayı ya da tarih olan sütunlar seçeneklerde yok: {_quoted(not_text, 5)}. Kullanmak için "
                     "Excel'de başlığı metin olarak yazın (ör. “Yıl 2020”).")
    if unnamed:
        notes.append("Başlık hücresi boş olan sütunlar seçeneklerde yok (soldan sıra: "
                     f"{', '.join(str(item) for item in unnamed[:8])}).")
    if unusable:
        notes.append(f"Adı “NA” olan ya da adında tırnak, ters bölü veya satır sonu bulunan sütunlar seçeneklerde "
                     f"yok: {_quoted(unusable, 5)}.")
    usable = frame.iloc[:, positions].copy()
    usable.columns = names
    strip_names = any(cells[position] != name for position, name in zip(positions, names))
    return usable, strip_names, tuple(notes)


# --- Adlar ve sıralar ------------------------------------------------------------------

def code_name(original: str, taken: Iterable[str] = ()) -> str:
    """Kodda kullanılacak ad: Türkçe karakterler ASCII'ye, diğer işaretler alt çizgiye çevrilir (ör. "Ödeme yöntemi"
    → ``odeme_yontemi``). Python ve R'nin ayrılmış sözcükleri, uygulamanın iç adları (``RESERVED_CODES``) ve daha önce
    verilen adlar kullanılmaz."""

    used = set(taken) | RESERVED_CODES
    name = re.sub(r"[^a-z0-9]+", "_", str(original).translate(_ASCII).lower()).strip("_")[:24].rstrip("_")
    if not name or not name[0].isalpha():
        name = f"degisken_{name}".rstrip("_") if name else "degisken"
    if keyword.iskeyword(name) or name in _R_RESERVED:
        name = f"{name}_"
    candidate, index = name, 2
    while candidate in used:
        candidate, index = f"{name}_{index}", index + 1
    return candidate


def turkish_key(text: str) -> tuple:
    """Türkçe alfabe sırası (ç, ğ, ı, ö, ş, ü yerinde) ve sayılar değerleriyle (2 < 10)."""

    key: list[tuple] = []
    for part in re.split(r"([0-9]+)", str(text)):
        if not part:
            continue
        if part[0] in "0123456789":
            key.append((0, int(part), ""))
            continue
        for character in part:
            low = "ı" if character == "I" else "i" if character == "İ" else character.lower()
            if low in _ALPHABET:
                key.append((1, _ALPHABET.index(low), character))
            else:
                key.append((2, ord(low), character))
    return tuple(key)


def fold(text: str) -> str:
    """Türkçe küçük harf (I → ı, İ → i); karşılaştırmalar için."""

    return str(text).replace("I", "ı").replace("İ", "i").lower().strip()


TIME_WORDS = frozenset(("yıl", "yil", "year", "dönem", "donem", "period", "tarih", "date", "ay", "month", "çeyrek",
                        "ceyrek", "quarter", "hafta", "week", "zaman", "time", "t"))
"""Dönem sütunu olabilecek adların sözcükleri (öneriler için; öğrenci değiştirebilir)."""


def name_words(name: str) -> tuple[str, ...]:
    """Sütun adının sözcükleri, Türkçe küçük harfle (ör. "Yıllık kazanç (bin TL)" → yıllık, kazanç, bin, tl)."""

    return tuple(word for word in re.split(r"[^a-zçğıöşüâîû0-9]+", fold(name)) if word)


def time_like(name: str) -> bool:
    """Adı bir dönem sütununa benzeyen sütun (Yıl, year, Dönem, Ay …)."""

    words = name_words(name)
    return bool(words) and (words[0] in TIME_WORDS or (len(words) == 1 and words[0] in TIME_WORDS))


def category_order(values: pd.Series, rule: str) -> tuple[str, ...]:
    """Kategorilerin tablo ve grafik sırası (``ORDER_RULES``); boş hücreler kategori sayılmaz."""

    present = values.dropna().astype(str)
    found = [str(value) for value in pd.unique(present)]
    if rule == "dosya":
        return tuple(found)
    if rule == "alfabetik":
        return tuple(sorted(found, key=turkish_key))
    if rule == "frekans":
        counts = present.value_counts()
        return tuple(sorted(found, key=lambda item: (-int(counts[item]), turkish_key(item))))
    raise ValueError(f"Desteklenmeyen sıra: {rule}")


# --- Sütun türü ------------------------------------------------------------------------

def _contains(values: Iterable[object], types: tuple[type, ...]) -> bool:
    return any(isinstance(value, types) for value in values)


def _numeric(value: object) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


def series_kind(series: pd.Series, use: str, label: str, decimal: str | None = None) -> str:
    """Sütunun koddaki dönüşümü (``FILE_COLUMN_KINDS``); kullanılamıyorsa ``UploadError``.

    Python ile R'nin aynı değeri üretemediği hücreler reddedilir: tarih ve saatler, DOĞRU/YANLIŞ değerleri, sonsuz
    değerler, metin sütunundaki ondalıklı ya da çok büyük sayılar ve ondalık işareti belirsiz metin sayılar.
    ``decimal``: CSV dosyasının ondalık işareti (Excel için ``None``).
    """

    present = series.dropna()
    if pd.api.types.is_datetime64_any_dtype(series) or _contains(present, (datetime.date, datetime.time)):
        raise UploadError(f"“{label}” tarih ya da saat değerleri içeriyor. Tarih ve saatler bu uygulamada kullanılmaz; "
                          "seçimden çıkarın ya da dosyada metne çevirin.")
    if pd.api.types.is_bool_dtype(series) or _contains(present, (bool, np.bool_)):
        raise UploadError(f"“{label}” DOĞRU/YANLIŞ (mantıksal) değerler içeriyor; bu uygulamada kullanılmaz. Dosyada "
                          "metne çevirin (ör. Evet, Hayır).")
    numeric = pd.api.types.is_numeric_dtype(series)
    numbers = [float(value) for value in present if _numeric(value)]
    if not all(math.isfinite(value) for value in numbers):
        raise UploadError(f"“{label}” sütununda sonsuz değerler (Inf) var; bu hücreleri dosyada silin.")
    if use == "sayisal":
        if any(abs(value) >= MAX_TEXT_INTEGER for value in numbers):
            raise UploadError(f"“{label}” sütununda çok büyük sayılar var (10^15 ve üstü). Dosyada birimi değiştirin "
                              "(ör. bin TL ya da milyon TL).")
        if numeric:
            return "sayi"
        _check_number_texts(present, label, decimal)
        return "sayi_metin"
    if not numeric:
        if _contains(present, (float, np.floating)):
            raise UploadError(f"“{label}” sütununda metinle birlikte ondalıklı sayılar var; kategori ya da metin "
                              "sütununda ondalıklı sayı kullanılmaz. Dosyada bu hücreleri metne çevirin ya da sütunu "
                              "seçmeyin.")
        if any(abs(value) >= MAX_TEXT_INTEGER for value in numbers):
            raise UploadError(f"“{label}” sütununda metinle birlikte çok büyük sayılar var; dosyada bu hücreleri metne "
                              "çevirin.")
    if use == "kategorik":
        if numeric:
            values = present.astype(float)
            if not (values == values.round()).all():
                raise UploadError(f"“{label}” ondalıklı sayılar içeriyor; kategorik değişken olarak kullanılamaz.")
            if any(abs(value) >= MAX_TEXT_INTEGER for value in numbers):
                raise UploadError(f"“{label}” sütununda çok büyük kodlar var (10^15 ve üstü); dosyada bu kodları "
                                  "metne çevirin (ör. başına bir harf ekleyin).")
            return "kod"
        return "metin"
    return "sayi" if numeric else "metin"


def column_kind(table: UploadedTable, original: str, use: str) -> str:
    return series_kind(table.frame[original], use, original, table.decimal if table.file_format == "csv" else None)


def _check_number_texts(values: pd.Series, label: str, decimal: str | None) -> None:
    """Metin olarak saklanmış sayılar. Tek bir ondalık işareti (nokta ya da virgül) kabul edilir; binlik ayırıcılı,
    işareti karışık ya da işareti belirsiz değerler reddedilir (yanlış okunmuş bir sayı sessizce sonucu değiştirirdi).
    """

    texts = [text for text in (clean_text(value) for value in values if not _numeric(value)) if text is not None]
    bad = [text for text in texts if not _NUMBER_TEXT.fullmatch(text)]
    grouped = [text for text in bad if _GROUPED_NUMBER.fullmatch(text)]
    if grouped:
        raise UploadError(f"“{label}” sütunundaki sayılar binlik ayırıcı içeriyor (ör. {_quoted(grouped, 2)}). "
                          "Dosyada sayıları binlik ayırıcı olmadan yazın.")
    if bad:
        examples = _quoted(sorted(set(bad), key=turkish_key))
        raise UploadError(f"“{label}” sütununda sayı olmayan değerler var: {examples}. Boş hücre ve NA eksik değer "
                          "sayılır; diğer işaretleri dosyada silin ya da başka bir sütun seçin.")
    if any(len(re.split(r"[.,]", text.lstrip("+-"))[0]) > 15 for text in texts):
        raise UploadError(f"“{label}” sütununda çok büyük sayılar var (10^15 ve üstü). Dosyada birimi değiştirin "
                          "(ör. bin TL ya da milyon TL).")
    dots = [text for text in texts if "." in text]
    commas = [text for text in texts if "," in text]
    # Noktadan sonra üç basamaktan farklı bir değer (ör. 12.5) binlik ayırıcı olamaz: virgüllü sayı yoksa nokta bu
    # sütunun ondalık işaretidir (kod da metin sayıları aynı kuralla çevirir).
    dot_decimal = bool(dots) and not commas and any(len(text.split(".")[-1]) != 3 for text in dots)
    if decimal == "," and dots and not dot_decimal:
        raise UploadError(f"“{label}” sütununda noktalı sayılar var (ör. {_quoted(dots, 2)}). Bu dosyada ondalık "
                          "işareti virgüldür; nokta binlik ayırıcı olabilir. Dosyada sayıları binlik ayırıcı olmadan "
                          "yazın.")
    if dots and commas:
        raise UploadError(f"“{label}” sütununda hem noktalı ({_quoted(dots, 1)}) hem virgüllü ({_quoted(commas, 1)}) "
                          "sayılar var; ondalık işareti belirsiz. Dosyada tek bir ondalık işareti kullanın.")
    marked = commas if decimal == "." else (commas or dots) if decimal is None else []
    if marked and all(len(re.split(r"[.,]", text)[-1]) == 3 for text in marked):
        raise UploadError(f"“{label}” sütunundaki {_quoted(marked, 2)} gibi değerlerde ayırıcıdan sonra hep üç basamak "
                          "var; işaret binlik ayırıcı da olabilir. Dosyada sayıları binlik ayırıcı olmadan yazın ya "
                          "da hücreleri sayı biçimine çevirin.")


def usable(table: UploadedTable, original: str, use: str) -> bool:
    """Sütun bu kullanım için reddedilmeden seçilebilir mi (öneriler için)."""

    try:
        kind = column_kind(table, original, use)
        if use == "kategorik":
            series = table.frame[original]
            values = series.map(code_text) if kind == "kod" else series.map(clean_text)
            _check_labels(values.dropna(), original)
    except UploadError:
        return False
    return True


def id_like(table: UploadedTable, original: str) -> bool:
    """Gözlem numarası gibi görünen sütun (No, ID, Sıra ya da 1, 2, 3, … biçiminde artan tam sayılar)."""

    if fold(original) in _ID_NAMES:
        return True
    series = table.frame[original].dropna()
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series) or len(series) < 3:
        return False
    values = series.to_numpy(dtype=float)
    return bool(np.all(values == np.round(values)) and np.all(np.diff(values) == 1))


# --- Hazırlama ---------------------------------------------------------------------------

def prepare(table: UploadedTable, selections: Iterable[Selection], *, frame: str = "veri",
            comment: str = "Yüklediğiniz veri dosyası") -> Prepared:
    """Seçilen sütunları üretilen kodun yaptığı gibi temizler ve ``ReadFile`` işlemini kurar."""

    chosen = list(selections)
    if not chosen:
        raise UploadError("En az bir sütun seçin.")
    originals = [item.original for item in chosen]
    if len(set(originals)) != len(originals):
        raise UploadError("Aynı sütun birden fazla kez seçildi; her rol için farklı bir sütun seçin.")
    missing_columns = [original for original in originals if original not in table.frame.columns]
    if missing_columns:
        raise UploadError(f"Şu sütunlar dosyada yok: {_quoted(missing_columns)}.")
    kinds = [column_kind(table, item.original, item.use) for item in chosen]
    data = table.frame[originals].copy()
    data.columns = [item.name for item in chosen]
    for item, kind in zip(chosen, kinds):
        if kind in ("metin", "sayi_metin"):
            data[item.name] = data[item.name].map(clean_text)
    required = [item.name for item in chosen if item.required] or [item.name for item in chosen]
    before = len(data)
    data = data.dropna(subset=required).reset_index(drop=True)
    dropped = before - len(data)
    if len(data) == 0:
        raise UploadError("Seçilen temel sütunlarda değeri olan satır kalmadı.")
    for item, kind in zip(chosen, kinds):
        if kind == "kod":
            data[item.name] = data[item.name].map(code_text)
        elif kind == "sayi_metin":
            try:
                data[item.name] = pd.to_numeric(data[item.name].str.replace(",", ".", regex=False))
            except (ValueError, OverflowError) as error:
                raise UploadError(f"“{item.original}” sütunundaki değerler sayıya çevrilemedi; dosyada sayıları "
                                  "sayı biçimine çevirin.") from error
    for item in chosen:
        if item.use == "kategorik":
            _check_labels(data[item.name].dropna(), item.original)
    notes = []
    if dropped:
        names = _list([f"“{item.original}”" for item in chosen if item.name in required])
        scope = (f"Seçilen sütunlardan ({names}) birinde" if len(required) == len(chosen)
                 else f"Temel sütunlarda ({names})")
        notes.append(f"{scope} boş hücre bulunan {dropped} satır çıkarıldı; analizde {len(data)} gözlem var.")
    for item in chosen:
        if item.name not in required:
            blanks = int(data[item.name].isna().sum())
            if blanks:
                notes.append(f"“{item.original}” sütununda {blanks} boş hücre var; bu sütunu kullanan adımlarda yalnız "
                             "değeri olan gözlemler kullanılır.")
    read = ReadFile(
        frame=frame,
        file_name=table.file_name,
        file_format=table.file_format,
        columns=tuple((item.name, item.original, kind) for item, kind in zip(chosen, kinds)),
        rows=tuple(tuple(_python(value) for value in row) for row in data.itertuples(index=False, name=None)),
        comment=comment,
        sheet=table.sheet,
        separator=table.separator,
        decimal=table.decimal,
        encoding=table.encoding,
        dropped=dropped,
        required=tuple(required),
        strip_names=table.strip_names,
    )
    return Prepared(read=read, frame=data, notes=tuple(notes))


def _python(value: object) -> object:
    """numpy sayısını Python sayısına, eksik değeri ``None``'a çevirir (tanımdaki değerler düz Python değerleridir)."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, np.floating) and np.isnan(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _check_labels(values: pd.Series, label: str) -> None:
    """Kategori etiketleri: tabloların toplam satırıyla karışmamalı, kısa olmalı, kodda yazılabilmeli."""

    categories = [str(item) for item in pd.unique(values)]
    long = [item for item in categories if len(item) > MAX_LABEL]
    if long:
        raise UploadError(f"“{label}” sütununda {MAX_LABEL} karakterden uzun kategori adları var "
                          f"(ör. “{_printable(long[0][:MAX_LABEL])}…”). Daha kısa adlar kullanın.")
    if any(_bad_characters(item) for item in categories):
        raise UploadError(f"“{label}” sütunundaki kategori adlarında tırnak, ters bölü, satır sonu ya da sekme var; "
                          "dosyada düzeltin.")


def check_levels(values: pd.Series, label: str, low: int, high: int) -> None:
    """Kategorik bir rolün kategori sayısı [low, high] aralığında olmalı (boş hücreler sayılmaz)."""

    count = values.dropna().nunique()
    if count < low:
        raise UploadError(f"“{label}” sütununda {count} farklı kategori var; bu adım için en az {low} gerekir.")
    if count > high:
        raise UploadError(f"“{label}” sütununda {count} farklı kategori var; en çok {high} kategori okunur. "
                          "Kategorileri birleştirin ya da başka bir sütun seçin.")


def sample_excel(frame, sheet: str = "Veri") -> bytes:
    """Örnek dosya, yüklemeye hazır Excel biçiminde (bellekte). ``frame`` bir veri çerçevesi ya da sayfa adı → veri
    çerçevesi sözlüğüdür (ör. bir panel ve bir deney sayfası)."""

    sheets = frame if isinstance(frame, dict) else {sheet: frame}
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, data in sheets.items():
            data.to_excel(writer, sheet_name=name, index=False)
    return buffer.getvalue()
