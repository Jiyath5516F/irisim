"""A small C preprocessor good enough for Iris shader packs.

Used for
  * shaders.properties / block.properties (Iris runs JCPP over them with the
    option values as macros),
  * extracting the *active* text of a GLSL program (comments kept, inactive
    #if branches removed) so const / comment directives (RENDERTARGETS,
    colortexNFormat, workGroups, ...) are read from the branch that is really
    compiled — like Iris, which parses directives from JCPP output.

It evaluates #if/#ifdef/#ifndef/#elif/#else/#endif and tracks #define/#undef
(object-like and simple function-like macros). Text lines are passed through
unexpanded except by `expand()` on request.
"""
import re

_TOKEN_RE = re.compile(r"""
    (?P<num>0[xX][0-9a-fA-F]+[uUlL]*|\d+\.\d*(?:[eE][-+]?\d+)?[fF]?|\.\d+(?:[eE][-+]?\d+)?[fF]?|\d+(?:[eE][-+]?\d+)?[uUlLfF]*)
  | (?P<id>[A-Za-z_]\w*)
  | (?P<op>&&|\|\||==|!=|<=|>=|<<|>>|[()!~*/%+\-<>&^|?:,])
  | (?P<ws>\s+)
  | (?P<other>.)
""", re.X)

_IDENT_RE = re.compile(r"[A-Za-z_]\w*")


class CppError(Exception):
    pass


def _tokens(expr):
    out = []
    for m in _TOKEN_RE.finditer(expr):
        kind = m.lastgroup
        if kind == "ws":
            continue
        if kind == "other":
            raise CppError(f"unexpected character {m.group()!r} in expression {expr!r}")
        out.append((kind, m.group()))
    return out


def _parse_number(text):
    t = text.rstrip("uUlL")
    if t.lower().startswith("0x"):
        return int(t, 16)
    if re.fullmatch(r"\d+", t):
        # C octal literals are not used in packs; treat as decimal
        return int(t)
    t = text.rstrip("fF")
    return float(t)


class _ExprParser:
    """Precedence-climbing evaluator over already macro-expanded tokens."""
    BIN = {
        "||": 2, "&&": 3, "|": 4, "^": 5, "&": 6, "==": 7, "!=": 7,
        "<": 8, ">": 8, "<=": 8, ">=": 8, "<<": 9, ">>": 9, "+": 10, "-": 10, "*": 11, "/": 11, "%": 11,
    }

    def __init__(self, toks):
        self.toks = toks
        self.i = 0
        self.used_float = False

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else (None, None)

    def take(self):
        t = self.peek()
        self.i += 1
        return t

    def parse(self):
        v = self.ternary()
        if self.i != len(self.toks):
            raise CppError(f"trailing tokens in #if expression: {' '.join(t for _, t in self.toks[self.i:])}")
        return v

    def ternary(self):
        c = self.binary(1)
        if self.peek()[1] == "?":
            self.take()
            a = self.ternary()
            if self.take()[1] != ":":
                raise CppError("expected ':' in ternary")
            b = self.ternary()
            return a if c else b
        return c

    def binary(self, min_prec):
        lhs = self.unary()
        while True:
            kind, op = self.peek()
            if kind != "op" or op not in self.BIN or self.BIN[op] < min_prec:
                return lhs
            prec = self.BIN[op]
            self.take()
            if op == "&&":
                rhs = self.binary(prec + 1)
                lhs = 1 if (lhs and rhs) else 0
                continue
            if op == "||":
                rhs = self.binary(prec + 1)
                lhs = 1 if (lhs or rhs) else 0
                continue
            rhs = self.binary(prec + 1)
            lhs = self.apply(op, lhs, rhs)

    def apply(self, op, a, b):
        if isinstance(a, float) or isinstance(b, float):
            self.used_float = True
        if op == "+": return a + b
        if op == "-": return a - b
        if op == "*": return a * b
        if op == "/":
            if b == 0:
                raise CppError("division by zero in #if")
            return a / b if (isinstance(a, float) or isinstance(b, float)) else int(a / b)
        if op == "%":
            if b == 0:
                raise CppError("modulo by zero in #if")
            return a % b
        if op == "<<": return int(a) << int(b)
        if op == ">>": return int(a) >> int(b)
        if op == "<": return int(a < b)
        if op == ">": return int(a > b)
        if op == "<=": return int(a <= b)
        if op == ">=": return int(a >= b)
        if op == "==": return int(a == b)
        if op == "!=": return int(a != b)
        if op == "&": return int(a) & int(b)
        if op == "^": return int(a) ^ int(b)
        if op == "|": return int(a) | int(b)
        raise CppError("bad operator " + op)

    def unary(self):
        kind, t = self.take()
        if kind is None:
            raise CppError("unexpected end of #if expression")
        if kind == "op":
            if t == "(":
                v = self.ternary()
                if self.take()[1] != ")":
                    raise CppError("missing ')'")
                return v
            if t == "!":
                return 0 if self.unary() else 1
            if t == "~":
                return ~int(self.unary())
            if t == "-":
                return -self.unary()
            if t == "+":
                return self.unary()
            raise CppError(f"unexpected operator {t}")
        if kind == "num":
            v = _parse_number(t)
            if isinstance(v, float):
                self.used_float = True
            return v
        if kind == "id":
            # identifiers left after expansion evaluate to 0 (C rule); true/false for GLSL-ish packs
            if t == "true":
                return 1
            return 0
        raise CppError(f"unexpected token {t}")


class MiniCpp:
    def __init__(self, defines=None):
        # name -> (params tuple or None, body)
        self.macros = {}
        for k, v in (defines or {}).items():
            self.define(k, v)
        self.errors = []
        self.warnings = []

    # ---------------------------------------------------------------- macros
    def define(self, name, body="1", params=None):
        self.macros[name] = (params, "" if body is None else str(body))

    def define_line(self, rest):
        m = re.match(r"([A-Za-z_]\w*)(\(([^)]*)\))?\s*(.*)$", rest)
        if not m:
            return
        name = m.group(1)
        params = None
        if m.group(2) is not None:
            params = tuple(p.strip() for p in m.group(3).split(",") if p.strip())
        body = _strip_comments_inline(m.group(4)).strip()
        self.macros[name] = (params, body)

    def is_defined(self, name):
        return name in self.macros

    def expand(self, text, depth=0, active=None):
        """Expand macros in a text fragment (no comment awareness)."""
        if depth > 32:
            return text
        active = active or frozenset()
        out = []
        pos = 0
        for m in _IDENT_RE.finditer(text):
            if m.start() < pos:
                continue
            name = m.group()
            if name not in self.macros or name in active:
                continue
            params, body = self.macros[name]
            if params is None:
                out.append(text[pos:m.start()])
                out.append(self.expand(body, depth + 1, active | {name}))
                pos = m.end()
            else:
                # function-like: need '(' args ')'
                j = m.end()
                while j < len(text) and text[j].isspace():
                    j += 1
                if j >= len(text) or text[j] != "(":
                    continue
                args, end = _split_args(text, j)
                if args is None:
                    continue
                sub = body
                for p, a in zip(params, args):
                    sub = re.sub(r"\b%s\b" % re.escape(p), a.strip(), sub)
                out.append(text[pos:m.start()])
                out.append(self.expand(sub, depth + 1, active | {name}))
                pos = end
        out.append(text[pos:])
        return "".join(out)

    def eval_condition(self, expr):
        expr = _strip_comments_inline(expr)
        # resolve defined X / defined(X) before expansion
        def repl(m):
            name = m.group(1) or m.group(2)
            return " 1 " if name in self.macros else " 0 "
        expr = re.sub(r"\bdefined\s*(?:\(\s*([A-Za-z_]\w*)\s*\)|([A-Za-z_]\w*))", repl, expr)
        expr = self.expand(expr)
        toks = _tokens(expr)
        if not toks:
            raise CppError("empty #if expression")
        p = _ExprParser(toks)
        v = p.parse()
        if p.used_float:
            self.warnings.append(f"float arithmetic in preprocessor condition: {expr.strip()}")
        return bool(v)

    # ---------------------------------------------------------------- processing
    def process(self, text, keep_directives=False):
        """Return the active text. Conditional directives are removed;
        #define/#undef are applied (and kept if keep_directives)."""
        lines = text.split("\n")
        out = []
        # stack entries: [parent_active, this_branch_active, any_branch_taken]
        stack = []
        active = True
        in_comment = False
        i = 0
        n = len(lines)
        while i < n:
            line = lines[i]
            i += 1
            starts_in_comment = in_comment
            in_comment = _comment_state_after(line, in_comment)
            stripped = line.lstrip()
            if starts_in_comment or not stripped.startswith("#"):
                if active:
                    out.append(line)
                continue
            # directive (join continuation lines)
            full = stripped
            while full.endswith("\\") and i < n:
                full = full[:-1] + " " + lines[i].strip()
                in_comment = _comment_state_after(lines[i], in_comment)
                i += 1
            m = re.match(r"#\s*(\w+)\s*(.*)$", full)
            if not m:
                if active:
                    out.append(line)
                continue
            d, rest = m.group(1), m.group(2)
            if d in ("if", "ifdef", "ifndef"):
                if not active:
                    stack.append([False, False, True])
                else:
                    if d == "if":
                        try:
                            cond = self.eval_condition(rest)
                        except CppError as e:
                            self.errors.append(f"#if {rest.strip()}: {e}")
                            cond = False
                    else:
                        name = _strip_comments_inline(rest).strip().split()[0] if rest.strip() else ""
                        cond = (name in self.macros) == (d == "ifdef")
                    stack.append([True, cond, cond])
                active = all(s[0] and s[1] for s in stack)
                continue
            if d == "elif":
                if not stack:
                    self.errors.append("#elif without #if")
                    continue
                top = stack[-1]
                if top[0] and not top[2]:
                    try:
                        cond = self.eval_condition(rest)
                    except CppError as e:
                        self.errors.append(f"#elif {rest.strip()}: {e}")
                        cond = False
                    top[1] = cond
                    top[2] = cond
                else:
                    top[1] = False
                active = all(s[0] and s[1] for s in stack)
                continue
            if d == "else":
                if not stack:
                    self.errors.append("#else without #if")
                    continue
                top = stack[-1]
                top[1] = top[0] and not top[2]
                top[2] = True
                active = all(s[0] and s[1] for s in stack)
                continue
            if d == "endif":
                if not stack:
                    self.errors.append("#endif without #if")
                    continue
                stack.pop()
                active = all(s[0] and s[1] for s in stack)
                continue
            if not active:
                continue
            if d == "define":
                self.define_line(rest)
                if keep_directives:
                    out.append(line)
                continue
            if d == "undef":
                self.macros.pop(rest.strip().split()[0] if rest.strip() else "", None)
                if keep_directives:
                    out.append(line)
                continue
            if d == "error":
                self.errors.append("#error " + rest.strip())
                continue
            # version / extension / pragma / line / include / unknown: keep as text
            out.append(line)
        if stack:
            self.errors.append(f"{len(stack)} unterminated #if block(s)")
        return "\n".join(out)


def _split_args(text, open_idx):
    depth = 0
    args = []
    cur = []
    for j in range(open_idx, len(text)):
        c = text[j]
        if c == "(":
            depth += 1
            if depth == 1:
                continue
        elif c == ")":
            depth -= 1
            if depth == 0:
                args.append("".join(cur))
                return args, j + 1
        elif c == "," and depth == 1:
            args.append("".join(cur))
            cur = []
            continue
        cur.append(c)
    return None, open_idx


def _strip_comments_inline(s):
    s = re.sub(r"/\*.*?\*/", " ", s)
    k = s.find("//")
    if k >= 0:
        s = s[:k]
    return s


def _comment_state_after(line, in_comment):
    """Track whether we are inside a /* */ block comment at the end of `line`."""
    i = 0
    n = len(line)
    while i < n:
        if in_comment:
            k = line.find("*/", i)
            if k < 0:
                return True
            in_comment = False
            i = k + 2
        else:
            a = line.find("/*", i)
            b = line.find("//", i)
            if a < 0:
                return False
            if 0 <= b < a:
                return False
            in_comment = True
            i = a + 2
    return in_comment


# ---------------------------------------------------------------- properties files
def preprocess_properties(text, defines):
    """Iris-style preprocessing of a .properties file (PropertiesPreprocessor)."""
    cmds = ("#if", "#ifdef", "#ifndef", "#elif", "#else", "#endif", "#define", "#undef", "#error", "#include",
            "#line", "#pragma", "#warning")
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            if any(line.startswith(c) and (len(line) == len(c) or not (line[len(c)].isalnum() or line[len(c)] == "_"))
                   for c in cmds):
                lines.append(line)
            else:
                lines.append("")
            continue
        lines.append(line.replace("#", ""))
    pp = MiniCpp(defines)
    out = pp.process("\n".join(lines))
    return out, pp


def parse_properties(text):
    """java.util.Properties-like parser -> ordered list of (key, value)."""
    items = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].lstrip()
        i += 1
        if not line or line[0] in "#!":
            continue
        while _ends_with_continuation(line) and i < len(lines):
            line = line[:-1] + lines[i].lstrip()
            i += 1
        m = re.match(r"((?:\\.|[^=:\s])+)\s*(?:[=:]\s*|\s+|$)(.*)$", line)
        if not m:
            continue
        key = m.group(1).replace("\\", "")
        value = m.group(2).strip()
        items.append((key, value))
    return items


def _ends_with_continuation(line):
    n = len(line) - len(line.rstrip("\\"))
    return n % 2 == 1
