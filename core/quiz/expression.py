"""Öğrencinin yazdığı formülü güvenli biçimde okur, önizler ve doğru cevapla karşılaştırır.

Güvenlik: ``eval`` kullanılmaz. Metin Python'un sözdizimi ağacına çevrilir ve yalnız
izin verilen düğümler (sayılar, tanımlı semboller, + − × ÷ ^, exp/log/sqrt)
``core.labs.expr`` ifadelerine dönüştürülür; başka her şey reddedilir.

Eşdeğerlik: iki ifade, sembollerin rastgele seçilmiş değerlerinde sayısal olarak
karşılaştırılır. Böylece ``100(exp(b)-1)`` ile ``100*exp(b) - 100`` aynı kabul edilir.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.labs import expr as E

FUNCTIONS = {"exp": "exp", "log": "log", "ln": "log", "sqrt": "sqrt"}


class FormulaError(ValueError):
    """Formül okunamadığında öğrenciye gösterilecek açıklamayla."""


@dataclass(frozen=True)
class Symbol:
    name: str
    latex: str
    meaning: str
    low: float = 0.5
    high: float = 3.0
    aliases: tuple[str, ...] = ()


_TOKEN = re.compile(r"\s*(?:(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|([A-Za-z_]\w*)|(\*\*|[-+*/()]))")


def _normalize(text: str) -> str:
    text = text.strip()
    for old, new in (("−", "-"), ("×", "*"), ("·", "*"), ("÷", "/"), ("^", "**"), ("[", "("), ("]", ")")):
        text = text.replace(old, new)
    return re.sub(r"(?<=\d),(?=\d)", ".", text)


def _with_implicit_products(text: str) -> str:
    """``100(exp(b)-1)`` → ``100*(exp(b)-1)``, ``2b`` → ``2*b``."""

    tokens: list[tuple[str, str]] = []
    position = 0
    while position < len(text):
        match = _TOKEN.match(text, position)
        if not match or match.end() == position:
            if text[position:].strip() == "":
                break
            raise FormulaError(f"Tanınmayan karakter: '{text[position:].strip()[0]}'")
        number, name, operator = match.groups()
        if number is not None:
            tokens.append(("sayi", number))
        elif name is not None:
            tokens.append(("ad", name))
        else:
            tokens.append(("islem", operator))
        position = match.end()
    pieces: list[str] = []
    for index, (kind, value) in enumerate(tokens):
        if index:
            previous_kind, previous = tokens[index - 1]
            left_closed = previous_kind == "sayi" or previous == ")" or (
                previous_kind == "ad" and previous.lower() not in FUNCTIONS
            )
            right_open = kind in ("sayi", "ad") or value == "("
            if left_closed and right_open:
                pieces.append("*")
        pieces.append(value)
    return "".join(pieces)


def _convert(node: ast.AST, symbols: dict[str, Symbol]) -> E.Expr:
    if isinstance(node, ast.Expression):
        return _convert(node.body, symbols)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return E.Const(float(node.value))
    if isinstance(node, ast.Name):
        if node.id in symbols:
            return E.Var(node.id)
        allowed = ", ".join(symbols) or "yok"
        raise FormulaError(f"'{node.id}' tanımlı bir sembol değil. Kullanılabilecek semboller: {allowed}.")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        operand = _convert(node.operand, symbols)
        return E.BinOp("-", E.Const(0.0), operand) if isinstance(node.op, ast.USub) else operand
    if isinstance(node, ast.BinOp):
        operators = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "^"}
        symbol = operators.get(type(node.op))
        if symbol is None:
            raise FormulaError("Yalnız + − * / ^ işlemleri kullanılabilir.")
        return E.BinOp(symbol, _convert(node.left, symbols), _convert(node.right, symbols))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        function = FUNCTIONS.get(node.func.id.lower())
        if function is None or len(node.args) != 1:
            raise FormulaError("Fonksiyon olarak yalnız exp(), log() ve sqrt() kullanılabilir.")
        return E.Call(function, (_convert(node.args[0], symbols),))
    raise FormulaError("Bu ifade okunamadı. Yalnız sayılar, semboller ve + − * / ^ ( ) kullanın.")


def parse(text: str, symbols: tuple[Symbol, ...]) -> E.Expr:
    """Metni güvenli biçimde ifade ağacına çevirir; okunamazsa ``FormulaError``."""

    if not text or not text.strip():
        raise FormulaError("Boş ifade.")
    if len(text) > 200:
        raise FormulaError("İfade çok uzun.")
    replacements = sorted(
        ((alias, symbol.name) for symbol in symbols for alias in symbol.aliases),
        key=lambda pair: -len(pair[0]),
    )
    for alias, name in replacements:
        text = text.replace(alias, f" {name} ")
    prepared = _with_implicit_products(_normalize(text))
    try:
        tree = ast.parse(prepared, mode="eval")
    except SyntaxError as error:
        raise FormulaError("Parantezleri ve işlemleri kontrol edin.") from error
    return _convert(tree, {symbol.name: symbol for symbol in symbols})


def equivalent(candidate: E.Expr, answer: E.Expr, symbols: tuple[Symbol, ...], draws: int = 16) -> bool:
    """İki ifadenin sembollerin rastgele değerlerinde aynı sonucu verip vermediği."""

    rng = np.random.default_rng(20260926)
    frame = pd.DataFrame(
        {symbol.name: rng.uniform(symbol.low, symbol.high, size=draws) for symbol in symbols}
    )
    with np.errstate(all="ignore"):
        try:
            got = np.asarray(E.evaluate(candidate, frame), dtype=float) * np.ones(draws)
            expected = np.asarray(E.evaluate(answer, frame), dtype=float) * np.ones(draws)
        except (ValueError, KeyError, ZeroDivisionError, OverflowError):
            return False
    usable = np.isfinite(expected)
    if not usable.any() or not np.isfinite(got[usable]).all():
        return False
    return bool(np.allclose(got[usable], expected[usable], rtol=1e-9, atol=1e-12))


def _latex_number(value: float) -> str:
    text = f"{value:.10g}"
    return text.replace(".", "{,}")


_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3}


def latex(expression: E.Expr, symbols: tuple[Symbol, ...]) -> str:
    """Önizleme için LaTeX; öğrenci yazdığının nasıl okunduğunu görür."""

    names = {symbol.name: symbol.latex for symbol in symbols}

    def level(node: E.Expr) -> int:
        return _PRECEDENCE[node.op] if isinstance(node, E.BinOp) else 4

    def wrap(node: E.Expr, minimum: int) -> str:
        text = render(node)
        return f"\\left({text}\\right)" if level(node) < minimum else text

    def render(node: E.Expr) -> str:
        if isinstance(node, E.Const):
            return _latex_number(node.value)
        if isinstance(node, E.Var):
            return names.get(node.name, node.name)
        if isinstance(node, E.Call):
            inner = render(node.args[0])
            if node.fn == "exp":
                return f"e^{{{inner}}}"
            if node.fn == "sqrt":
                return f"\\sqrt{{{inner}}}"
            return f"\\log\\left({inner}\\right)"
        if isinstance(node, E.BinOp):
            if node.op == "-" and isinstance(node.left, E.Const) and node.left.value == 0:
                return f"-{wrap(node.right, 2)}"
            if node.op == "/":
                return f"\\frac{{{render(node.left)}}}{{{render(node.right)}}}"
            if node.op == "^":
                return f"{{{wrap(node.left, 4)}}}^{{{render(node.right)}}}"
            if node.op == "*":
                return f"{wrap(node.left, 2)} \\cdot {wrap(node.right, 2)}"
            right = wrap(node.right, 2) if node.op == "-" else render(node.right)
            return f"{render(node.left)} {node.op} {right}"
        raise TypeError(type(node).__name__)

    return render(expression)
