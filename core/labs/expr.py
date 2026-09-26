"""Laboratuvar tanımlarında kullanılan küçük ifade dili.

Türetilmiş değişkenler ve katsayı dönüşümleri bir kez burada tanımlanır; aynı
ifade hem uygulamada (pandas) değerlendirilir hem de Python, R ve Stata koduna
çevrilir. Böylece üç dildeki kod ile uygulamanın hesabı tek kaynaktan gelir.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

import numpy as np
import pandas as pd
from scipy import special, stats


INTERCEPT = "(sabit)"
"""Sabit terim için dilden bağımsız ad. Her üretici kendi adına çevirir."""


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class Const:
    value: float


@dataclass(frozen=True)
class BinOp:
    op: str
    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class Call:
    fn: str
    args: tuple["Expr", ...]


@dataclass(frozen=True)
class Coef:
    """Tahmin edilmiş bir modelin katsayısı (skaler)."""

    model: str
    term: str


@dataclass(frozen=True)
class StdErr:
    """Tahmin edilmiş bir modelin katsayısının standart hatası (modelin kendi kovaryans türüyle)."""

    model: str
    term: str


Expr = Union[Var, Const, BinOp, Call, Coef, StdErr]

BINARY_OPS = ("+", "-", "*", "/", "^")
COMPARISONS = {"le": "<=", "lt": "<", "ge": ">=", "gt": ">", "eq": "==", "ne": "!="}
"""Karşılaştırma fonksiyonları: koşul sağlanırsa 1, değilse 0 (gösterge değişkeni)."""
FUNCTIONS = (
    "log", "exp", "sqrt", "maximum", "minimum", "round", "floor", "positive",
    "logistic", "normcdf", "normpdf", "sin", "cos", *COMPARISONS,
)
_BINARY_FUNCTIONS = ("maximum", "minimum", *COMPARISONS)
_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3}
_ATOM = 4


# --- Kurucular: tanım dosyalarının okunaklı kalması için -------------------

def var(name: str) -> Var:
    return Var(name)


def const(value: float) -> Const:
    return Const(float(value))


def _wrap(value: "Expr | float | int") -> "Expr":
    if isinstance(value, (int, float)):
        return Const(float(value))
    return value


def add(a, b) -> BinOp:
    return BinOp("+", _wrap(a), _wrap(b))


def sub(a, b) -> BinOp:
    return BinOp("-", _wrap(a), _wrap(b))


def mul(a, b) -> BinOp:
    return BinOp("*", _wrap(a), _wrap(b))


def div(a, b) -> BinOp:
    return BinOp("/", _wrap(a), _wrap(b))


def power(a, b) -> BinOp:
    return BinOp("^", _wrap(a), _wrap(b))


def log(a) -> Call:
    return Call("log", (_wrap(a),))


def exp(a) -> Call:
    return Call("exp", (_wrap(a),))


def maximum(a, b) -> Call:
    return Call("maximum", (_wrap(a), _wrap(b)))


def sin(a) -> Call:
    """Sinüs (radyan): simülasyonlarda bilinen doğrusal olmayan koşullu ortalama için."""

    return Call("sin", (_wrap(a),))


def cos(a) -> Call:
    return Call("cos", (_wrap(a),))


def minimum(a, b) -> Call:
    return Call("minimum", (_wrap(a), _wrap(b)))


def floor(a) -> Call:
    return Call("floor", (_wrap(a),))


def rounded(a) -> Call:
    """En yakın tam sayıya yuvarlama (sürekli çekilişlerde yarım değer olasılığı sıfırdır)."""

    return Call("round", (_wrap(a),))


def positive(a) -> Call:
    """Gösterge fonksiyonu: argüman sıfırdan büyükse 1, değilse 0."""

    return Call("positive", (_wrap(a),))


def logistic(a) -> Call:
    """Lojistik dağılım fonksiyonu Λ(a) = 1/(1 + e^(−a))."""

    return Call("logistic", (_wrap(a),))


def normcdf(a) -> Call:
    """Standart normal dağılım fonksiyonu Φ(a)."""

    return Call("normcdf", (_wrap(a),))


def normpdf(a) -> Call:
    """Standart normal yoğunluk φ(a)."""

    return Call("normpdf", (_wrap(a),))


def compare(name: str, a, b) -> Call:
    """Gösterge: ``a`` ile ``b`` karşılaştırması doğruysa 1, değilse 0 (``name``: le, lt, ge, gt, eq, ne)."""

    if name not in COMPARISONS:
        raise ValueError(f"Desteklenmeyen karşılaştırma: {name}")
    return Call(name, (_wrap(a), _wrap(b)))


def coef(model: str, term: str) -> Coef:
    return Coef(model, term)


def se(model: str, term: str) -> StdErr:
    return StdErr(model, term)


# --- Doğrulama ---------------------------------------------------------------

def validate(expr: Expr) -> None:
    if isinstance(expr, (Var, Const, Coef, StdErr)):
        return
    if isinstance(expr, BinOp):
        if expr.op not in BINARY_OPS:
            raise ValueError(f"Desteklenmeyen işlem: {expr.op}")
        validate(expr.left)
        validate(expr.right)
        return
    if isinstance(expr, Call):
        if expr.fn not in FUNCTIONS:
            raise ValueError(f"Desteklenmeyen fonksiyon: {expr.fn}")
        expected = 2 if expr.fn in _BINARY_FUNCTIONS else 1
        if len(expr.args) != expected:
            raise ValueError(f"{expr.fn} {expected} argüman alır.")
        for argument in expr.args:
            validate(argument)
        return
    raise TypeError(f"Tanınmayan ifade düğümü: {type(expr).__name__}")


def variables(expr: Expr) -> set[str]:
    """İfadenin başvurduğu veri değişkenleri."""

    if isinstance(expr, Var):
        return {expr.name}
    if isinstance(expr, BinOp):
        return variables(expr.left) | variables(expr.right)
    if isinstance(expr, Call):
        found: set[str] = set()
        for argument in expr.args:
            found |= variables(argument)
        return found
    return set()


# --- Değerlendirme -----------------------------------------------------------

def evaluate(
    expr: Expr,
    frame: pd.DataFrame | None = None,
    coefficient: Callable[[str, str], float] | None = None,
    standard_error: Callable[[str, str], float] | None = None,
):
    """İfadeyi bir veri çerçevesi (vektör) veya katsayılar (skaler) üzerinde hesaplar."""

    def again(node: Expr):
        return evaluate(node, frame, coefficient, standard_error)

    if isinstance(expr, Const):
        return expr.value
    if isinstance(expr, Var):
        if frame is None:
            raise ValueError(f"'{expr.name}' için veri çerçevesi gerekir.")
        return np.asarray(frame[expr.name], dtype=float)
    if isinstance(expr, Coef):
        if coefficient is None:
            raise ValueError("Katsayı ifadesi için tahmin edilmiş model gerekir.")
        return coefficient(expr.model, expr.term)
    if isinstance(expr, StdErr):
        if standard_error is None:
            raise ValueError("Standart hata ifadesi için tahmin edilmiş model gerekir.")
        return standard_error(expr.model, expr.term)
    if isinstance(expr, BinOp):
        left = again(expr.left)
        right = again(expr.right)
        if expr.op == "+":
            return left + right
        if expr.op == "-":
            return left - right
        if expr.op == "*":
            return left * right
        if expr.op == "/":
            return left / right
        return left**right
    if isinstance(expr, Call):
        values = [again(argument) for argument in expr.args]
        if expr.fn == "log":
            return np.log(values[0])
        if expr.fn == "exp":
            return np.exp(values[0])
        if expr.fn == "sqrt":
            return np.sqrt(values[0])
        if expr.fn == "round":
            return np.rint(values[0])
        if expr.fn == "floor":
            return np.floor(values[0])
        if expr.fn == "positive":
            return np.where(np.asarray(values[0]) > 0, 1.0, 0.0)
        if expr.fn == "logistic":
            return special.expit(values[0])
        if expr.fn == "normcdf":
            return stats.norm.cdf(values[0])
        if expr.fn == "normpdf":
            return stats.norm.pdf(values[0])
        if expr.fn == "sin":
            return np.sin(values[0])
        if expr.fn == "cos":
            return np.cos(values[0])
        if expr.fn in COMPARISONS:
            left, right = np.asarray(values[0]), np.asarray(values[1])
            outcome = {
                "le": left <= right, "lt": left < right, "ge": left >= right,
                "gt": left > right, "eq": left == right, "ne": left != right,
            }[expr.fn]
            return np.where(outcome, 1.0, 0.0)
        if expr.fn == "minimum":
            return np.minimum(values[0], values[1])
        return np.maximum(values[0], values[1])
    raise TypeError(f"Tanınmayan ifade düğümü: {type(expr).__name__}")


# --- Dile çevirme ------------------------------------------------------------

@dataclass(frozen=True)
class Dialect:
    """Bir hedef dilin ifade sözdizimi.

    ``functions`` değerleri ya bir ad (``np.exp`` → ``np.exp(x)``) ya da ``{0}``
    yer tutuculu bir kalıptır (``as.numeric({0} > 0)``).
    """

    variable: Callable[[str], str]
    coefficient: Callable[[str, str], str]
    functions: dict[str, str]
    power: str
    standard_error: Callable[[str, str], str] | None = None


def format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value))


def _precedence(expr: Expr) -> int:
    return _PRECEDENCE[expr.op] if isinstance(expr, BinOp) else _ATOM


def render(expr: Expr, dialect: Dialect) -> str:
    """İfadeyi gereksiz parantez olmadan hedef dile yazar."""

    if isinstance(expr, Const):
        return format_number(expr.value)
    if isinstance(expr, Var):
        return dialect.variable(expr.name)
    if isinstance(expr, Coef):
        return dialect.coefficient(expr.model, expr.term)
    if isinstance(expr, StdErr):
        if dialect.standard_error is None:
            raise ValueError("Bu dilde standart hata ifadesi tanımlı değil.")
        return dialect.standard_error(expr.model, expr.term)
    if isinstance(expr, Call):
        arguments = [render(argument, dialect) for argument in expr.args]
        spec = dialect.functions[expr.fn]
        if "{0}" in spec:
            return spec.format(*arguments)
        return f"{spec}({', '.join(arguments)})"
    parent = _PRECEDENCE[expr.op]
    left = render(expr.left, dialect)
    right = render(expr.right, dialect)
    if _precedence(expr.left) < parent or (
        expr.op == "^" and _precedence(expr.left) == parent
    ):
        left = f"({left})"
    if _precedence(expr.right) < parent or (
        expr.op in ("-", "/") and _precedence(expr.right) == parent
    ):
        right = f"({right})"
    symbol = dialect.power if expr.op == "^" else expr.op
    return f"{left} {symbol} {right}"


# --- Sembolik türev (delta yöntemi için) --------------------------------------

def _simplify(node: Expr) -> Expr:
    if isinstance(node, BinOp):
        left, right = _simplify(node.left), _simplify(node.right)
        zero_l = isinstance(left, Const) and left.value == 0
        zero_r = isinstance(right, Const) and right.value == 0
        one_l = isinstance(left, Const) and left.value == 1
        one_r = isinstance(right, Const) and right.value == 1
        if node.op == "*":
            if zero_l or zero_r:
                return Const(0.0)
            if one_l:
                return right
            if one_r:
                return left
        if node.op == "+":
            if zero_l:
                return right
            if zero_r:
                return left
        if node.op == "-" and zero_r:
            return left
        if node.op == "/" and zero_l:
            return Const(0.0)
        if node.op == "/" and one_r:
            return left
        if isinstance(left, Const) and isinstance(right, Const):
            return Const(float(evaluate(BinOp(node.op, left, right))))
        return BinOp(node.op, left, right)
    if isinstance(node, Call):
        return Call(node.fn, tuple(_simplify(argument) for argument in node.args))
    return node


def derivative(node: Expr, target: Coef) -> Expr:
    """``node``'un ``target`` katsayısına göre türevi (sadeleştirilmiş)."""

    def d(item: Expr) -> Expr:
        if isinstance(item, (Const, Var, StdErr)):
            return Const(0.0)
        if isinstance(item, Coef):
            return Const(1.0 if item == target else 0.0)
        if isinstance(item, BinOp):
            a, b = item.left, item.right
            if item.op in ("+", "-"):
                return BinOp(item.op, d(a), d(b))
            if item.op == "*":
                return BinOp("+", BinOp("*", d(a), b), BinOp("*", a, d(b)))
            if item.op == "/":
                return BinOp("/", BinOp("-", BinOp("*", d(a), b), BinOp("*", a, d(b))), BinOp("^", b, Const(2.0)))
            if isinstance(b, Const):
                return BinOp("*", BinOp("*", b, BinOp("^", a, Const(b.value - 1))), d(a))
            raise ValueError("Delta yöntemi: üs sabit olmalıdır.")
        if isinstance(item, Call):
            a = item.args[0]
            if item.fn == "exp":
                return BinOp("*", item, d(a))
            if item.fn == "log":
                return BinOp("/", d(a), a)
            if item.fn == "sqrt":
                return BinOp("/", d(a), BinOp("*", Const(2.0), item))
            if item.fn == "logistic":
                return BinOp("*", BinOp("*", item, BinOp("-", Const(1.0), item)), d(a))
            if item.fn == "normcdf":
                return BinOp("*", Call("normpdf", (a,)), d(a))
            if item.fn == "normpdf":
                return BinOp("*", BinOp("*", BinOp("-", Const(0.0), a), item), d(a))
        raise ValueError(f"Delta yöntemi bu ifadeyi türevleyemez: {item}")

    return _simplify(d(node))


def coefficients(node: Expr) -> list[Coef]:
    """İfadedeki katsayılar, ilk görülme sırasıyla."""

    found: list[Coef] = []
    if isinstance(node, Coef):
        found.append(node)
    elif isinstance(node, BinOp):
        found += coefficients(node.left) + coefficients(node.right)
    elif isinstance(node, Call):
        for argument in node.args:
            found += coefficients(argument)
    unique: list[Coef] = []
    for item in found:
        if item not in unique:
            unique.append(item)
    return unique
