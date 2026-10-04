"""Uygulama sekmesinde "Kendi verini yükle" paneli: dosya yükleme, sütun seçimi ve doğrulama.

Hesap ``core.labs.kendi_veri`` ve ``core.labs.ornek`` içindedir; bu modül yalnız seçimleri toplar. Yüklenen dosya ve
ondan kurulan uygulama yalnız bu oturumun belleğinde (``st.session_state``) tutulur; ortak önbelleğe, diske ya da
günlüğe yazılmaz.

Veri kaynağı değişince ya da başka bir konuya geçilince Streamlit bu paneldeki widget'ların durumunu siler. Dosya ve
seçimler bu yüzden widget dışı anahtarlarda da saklanır (``_kalici_`` önekli gölge anahtarlar); öğrenci panele
döndüğünde kaldığı yerden devam eder. Panel İKT 217 ve İKT 305 uygulamalarındakiyle aynı kuralları izler.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Callable

import pandas as pd
import streamlit as st

from core.labs import kendi_veri as K
from core.labs.ornek import CustomChoices, CustomLab, custom_case, md
from core.labs.ornekler import get_variants
from core.labs.spec import LabSpec

NONE = "— seçilmedi —"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
PRIVACY = (
    "Dosyanız yalnız bu oturumda, sunucunun belleğinde işlenir; kaydedilmez ve başkalarıyla paylaşılmaz. Kişisel veri "
    f"(ad, kimlik numarası, iletişim bilgisi) içeren dosya yüklemeyin. En çok {K.MAX_MEGABYTES} MB ve "
    f"{K.thousands(K.MAX_ROWS)} satır; ilk satırda sütun adları olmalı."
)
SHADOW = "_kalici_"


@st.cache_data(show_spinner=False)
def _sample_file(topic_key: str) -> bytes:
    """Örnek dosya (kurgusal veri); öğrenci verisi içermez, ortak önbellekte tutulabilir."""

    return K.sample_excel(get_variants(topic_key).custom.sample())


def session_cache(key: str, token: object, factory: Callable[[], object]) -> object:
    """Oturumluk önbellek: ``token`` değişmedikçe ``factory`` yeniden çalışmaz (öğrenci verisi ortak önbelleğe
    girmez)."""

    stored = st.session_state.get(key)
    if isinstance(stored, tuple) and len(stored) == 2 and stored[0] == token:
        return stored[1]
    value = factory()
    st.session_state[key] = (token, value)
    return value


# --- Seçimlerin saklanması ---------------------------------------------------------------

def shadow(key: str) -> str:
    return f"{SHADOW}{key}"


def restore(key: str) -> None:
    """Widget çizilmeden önce: durumu silinmişse (ör. veri kaynağı ya da konu değişti) son seçim geri yüklenir."""

    if key not in st.session_state and shadow(key) in st.session_state:
        st.session_state[key] = st.session_state[shadow(key)]


def remember(key: str) -> None:
    if key in st.session_state:
        st.session_state[shadow(key)] = st.session_state[key]


def _file_key(topic_key: str) -> str:
    return f"{topic_key}_kendi_yuklenen"


def _on_upload(topic_key: str) -> None:
    """Öğrenci dosyayı değiştirince ya da kaldırınca saklanan dosya güncellenir."""

    uploaded = st.session_state.get(f"{topic_key}_kendi_dosya")
    if uploaded is None:
        _remove_file(topic_key)
    else:
        st.session_state[_file_key(topic_key)] = (uploaded.name, uploaded.getvalue())


def _remove_file(topic_key: str) -> None:
    """Dosya ve ondan okunan her şey oturum belleğinden silinir."""

    for name in ("yuklenen", "tablo", "uygulama", "ozet", "ozet_sayfa", "sayfalar", "hesap"):
        st.session_state.pop(f"{topic_key}_kendi_{name}", None)
    _forget_choices(topic_key)


def _forget_choices(topic_key: str, names: tuple[str, ...] = ("rol_", "ek", "sira", "sayfa", "kategori_",
                                                                 "secenek_")) -> None:
    """Yeni dosya yüklenince önceki dosyanın sütun seçimleri silinir (başka sayfa seçilince sayfa seçimi kalır,
    diğerleri silinir)."""

    prefix = f"{topic_key}_kendi_"
    for key in list(st.session_state.keys()):
        text = str(key)
        bare = text[len(SHADOW):] if text.startswith(SHADOW) else text
        if bare.startswith(prefix) and bare[len(prefix):].startswith(names):
            del st.session_state[key]


# --- Öneriler -----------------------------------------------------------------------------

def _numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _varies(table: K.UploadedTable, column: str) -> bool:
    """Sütunun en az iki farklı değeri ve en az üç dolu hücresi var (sabit sütun özet ve ilişki için kullanılmaz)."""

    values = table.frame[column].dropna()
    return len(values) >= 3 and values.nunique() >= 2


def _guess(role, table: K.UploadedTable, used: set[str]) -> str:
    """Önerilen sütun: kullanımına uyan ve reddedilmeyecek ilk sütun. Sayısal rolde gözlem numarası gibi görünen
    sütunlar önerilmez; kategorik rolde her kategoride en az iki gözlem olmalı."""

    candidates = [column for column in table.columns if column not in used]
    if role.use == "sayisal":
        numeric = [column for column in candidates if _numeric(table.frame[column]) and _varies(table, column)
                   and K.usable(table, column, "sayisal")]
        for column in numeric:
            if not K.id_like(table, column) and not K.time_like(column):
                return column
        if role.required and numeric:
            return numeric[0]
    elif role.use == "kategorik":
        numeric_ok = role.required or role.levels[1] <= 2
        for column in candidates:
            series = table.frame[column]
            if _numeric(series) and not numeric_ok:
                continue
            counts = series.map(K.clean_text).dropna().value_counts()
            if role.levels[0] <= len(counts) <= role.levels[1] and counts.min() >= 2 \
                    and K.usable(table, column, "kategorik"):
                return column
    return NONE


# --- Panel ---------------------------------------------------------------------------------

def _render_roles(topic_key: str, custom: CustomLab, table: K.UploadedTable) -> CustomChoices:
    options = [NONE, *table.columns]
    used: set[str] = set()
    roles: dict[str, str | None] = {}
    columns = st.columns(2)
    hints = dict(custom.suggest(table)) if custom.suggest is not None else {}
    for index, role in enumerate(custom.roles):
        key = f"{topic_key}_kendi_rol_{role.key}"
        restore(key)
        if key not in st.session_state or st.session_state[key] not in options:
            if role.key in hints and hints[role.key] in options:
                st.session_state[key] = hints[role.key]
            elif role.required or (role.suggest and not hints):
                st.session_state[key] = _guess(role, table, used)
            else:
                st.session_state[key] = NONE
        label = role.label + ("" if role.required else " (isteğe bağlı)")
        steps = ", ".join(str(step) for step in role.steps)
        value = columns[index % 2].selectbox(label, options, key=key, help=f"{role.help} Adım: {steps}.")
        remember(key)
        roles[role.key] = None if value == NONE else value
        if value != NONE:
            used.add(value)
    extra: tuple[str, ...] = ()
    if custom.extra_columns:
        key = f"{topic_key}_kendi_ek"
        restore(key)
        remaining = [column for column in table.columns if column not in used]
        if key in st.session_state:
            st.session_state[key] = [column for column in st.session_state[key] if column in remaining]
        else:  # ilk açılışta kullanılabilir sayısal sütunlar (gözlem numarası ya da dönem gibi görünenler hariç)
            st.session_state[key] = [
                column for column in remaining
                if K.usable(table, column, custom.extra_use)
                and not (custom.extra_use == "sayisal" and (K.id_like(table, column) or K.time_like(column)
                                                            or not _numeric(table.frame[column])
                                                            or not _varies(table, column)))
            ][:custom.max_extra]
        extra = tuple(st.multiselect(custom.extra_label, remaining, key=key, placeholder="Sütun seçin",
                                     max_selections=custom.max_extra, help=custom.extra_help or None))
        remember(key)
    order = "alfabetik"
    if any(roles.get(role) for role in custom.order_roles):
        key = f"{topic_key}_kendi_sira"
        restore(key)
        order = st.selectbox("Kategori sırası", list(K.ORDER_RULES), format_func=K.ORDER_RULES.get, key=key,
                             help="Tablolarda ve seçeneklerde kategorilerin sırası.")
        remember(key)
    options = _render_options(topic_key, custom, table, roles, extra)
    return CustomChoices(roles=roles, extra=extra, order=order, options=options)


def _analysis_rows(custom: CustomLab, table: K.UploadedTable, roles: dict[str, str | None],
                   extra: tuple[str, ...]) -> pd.Series:
    """Analize girecek satırlar: zorunlu rollerin (ve gerekiyorsa ek sütunların) hepsinde değeri olanlar (boşluk ve NA
    eksik sayılır, uygulamanın kuralıyla)."""

    columns = [roles[role.key] for role in custom.roles if role.required and roles.get(role.key)]
    if custom.extra_required:
        columns += list(extra)
    keep = pd.Series(True, index=table.frame.index)
    for column in dict.fromkeys(columns):
        keep &= table.frame[column].map(K.clean_text).notna()
    return keep


def _render_options(topic_key: str, custom: CustomLab, table: K.UploadedTable, roles: dict[str, str | None],
                    extra: tuple[str, ...]) -> dict[str, bool]:
    """Konuya özgü açık/kapalı seçenekler (ör. sonucun logaritması). Bağlı olduğu rolün sütunu analiz örnekleminde
    seçeneğe uymuyorsa seçenek kapalı ve pasif gösterilir; öğrencinin önceki tercihi saklı kalır."""

    values: dict[str, bool] = {}
    rows = _analysis_rows(custom, table, roles, extra) if custom.options else None
    for option in custom.options:
        key = f"{topic_key}_kendi_secenek_{option.key}"
        original = roles.get(option.role) if option.role else None
        allowed = option.allowed is None or not original or option.allowed(table.frame.loc[rows, original])
        if not allowed:
            st.toggle(option.label, value=False, disabled=True, key=f"{key}_pasif", help=option.help)
            st.caption(option.blocked)
            values[option.key] = False
            continue
        restore(key)
        if key not in st.session_state:
            st.session_state[key] = option.default
        values[option.key] = bool(st.toggle(option.label, key=key, help=option.help))
        remember(key)
    return values


def _render_picks(topic_key: str, custom: CustomLab, case) -> dict[str, str]:
    picks: dict[str, str] = {}
    for role in custom.roles:
        if not role.pick or role.key not in case.roles:
            continue
        categories = list(case.orders[case.roles[role.key]])
        key = f"{topic_key}_kendi_kategori_{role.key}"
        restore(key)
        if st.session_state.get(key) not in categories:
            st.session_state[key] = case.levels[role.key]
        picks[role.key] = st.selectbox(f"{role.pick} · {md(case.labels[case.roles[role.key]])}", categories, key=key)
        remember(key)
    return picks


def _current_file(topic_key: str, uploaded) -> tuple[str, bytes] | None:
    """Kullanılan dosya: yükleyicideki dosya ya da (yükleyicinin durumu silindiyse) saklanan son dosya."""

    if uploaded is not None:
        current = (uploaded.name, uploaded.getvalue())
        if st.session_state.get(_file_key(topic_key)) != current:
            st.session_state[_file_key(topic_key)] = current
        return current
    return st.session_state.get(_file_key(topic_key))


def render_custom(topic_key: str, custom: CustomLab) -> LabSpec | None:
    """Veri panelini gösterir; geçerli seçimlerle kurulan uygulamayı döndürür (yoksa ``None``)."""

    st.markdown(custom.intro)
    st.caption(PRIVACY)
    left, right = st.columns([4, 1], vertical_alignment="bottom")
    uploaded = left.file_uploader("Veri dosyası (.xlsx ya da .csv)", type=["xlsx", "csv"],
                                  key=f"{topic_key}_kendi_dosya", max_upload_size=K.MAX_MEGABYTES,
                                  on_change=_on_upload, args=(topic_key,))
    right.download_button(
        "Örnek dosya", data=_sample_file(topic_key), file_name=f"ikt807_{topic_key}_ornek.xlsx", mime=XLSX_MIME,
        key=f"{topic_key}_kendi_ornek", icon=":material/download:", width="stretch",
        help="Kurgusal bir örnek veri. İndirip yükleyerek seçenekleri deneyebilirsiniz.",
    )
    current = _current_file(topic_key, uploaded)
    if current is None:
        st.info("Başlamak için bir dosya yükleyin. Biçimi görmek için örnek dosyayı indirip yükleyebilirsiniz.",
                icon=":material/upload_file:")
        return None
    name, data = current
    if uploaded is None:
        note, button = st.columns([4, 1], vertical_alignment="center")
        note.caption(f"Kullanılan dosya: {md(name)}. Başka bir dosya yükleyerek değiştirebilirsiniz.")
        button.button("Dosyayı kaldır", key=f"{topic_key}_kendi_kaldir", on_click=_remove_file, args=(topic_key,),
                      icon=":material/delete:", width="stretch")
    digest = hashlib.md5(data).hexdigest()
    token = (digest, name)
    if st.session_state.get(f"{topic_key}_kendi_ozet") != token:
        _forget_choices(topic_key)
        st.session_state[f"{topic_key}_kendi_ozet"] = token
    try:
        sheet = None
        if name.lower().endswith(".xlsx"):
            sheets = session_cache(f"{topic_key}_kendi_sayfalar", token, lambda: K.excel_sheets(data))
            if len(sheets) > 1:
                key = f"{topic_key}_kendi_sayfa"
                restore(key)
                if st.session_state.get(key) not in sheets:
                    st.session_state[key] = sheets[0]
                sheet = st.selectbox("Sayfa", sheets, key=key)
                remember(key)
        if st.session_state.get(f"{topic_key}_kendi_ozet_sayfa") != (token, sheet):
            if f"{topic_key}_kendi_ozet_sayfa" in st.session_state:  # aynı dosyada başka sayfa: sütunlar değişir
                _forget_choices(topic_key, ("rol_", "ek", "sira", "kategori_", "secenek_"))
            st.session_state[f"{topic_key}_kendi_ozet_sayfa"] = (token, sheet)
        table = session_cache(f"{topic_key}_kendi_tablo", (token, sheet), lambda: K.read_upload(name, data, sheet))
    except K.UploadError as error:
        st.error(md(str(error)), icon=":material/error:")
        return None
    with st.expander(f"Dosyanın ilk satırları ({K.thousands(len(table.frame))} satır, {len(table.columns)} sütun)"):
        st.dataframe(table.frame.head(8), hide_index=True)
    for note in table.notes:
        st.caption(md(note))
    with st.container(border=True):
        st.markdown("**Sütunları seçin**")
        try:
            choices = _render_roles(topic_key, custom, table)
            case, notes = custom_case(custom, table, choices)
            picks = _render_picks(topic_key, custom, case)
            if picks:
                choices = replace(choices, picks=picks)
                case, notes = custom_case(custom, table, choices)
            key = (token, sheet, tuple(sorted(choices.roles.items(), key=str)), choices.extra, choices.order,
                   tuple(sorted(choices.picks.items())), tuple(sorted(choices.options.items())))
            spec = session_cache(f"{topic_key}_kendi_uygulama", key, lambda: custom.build(case))
        except K.UploadError as error:
            st.error(md(str(error)), icon=":material/error:")
            return None
        except Exception as error:  # beklenmeyen veri: öğrenciye anlaşılır bir ileti, ayrıntı türüyle
            st.error(f"Dosya bu seçimlerle işlenemedi ({type(error).__name__}: {md(str(error))}). Seçimleri "
                     "değiştirin ya da dosyayı kontrol edin.", icon=":material/error:")
            return None
    for note in notes:
        st.caption(md(note))
    return spec
