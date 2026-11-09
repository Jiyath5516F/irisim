"""shaders.properties custom uniforms (`uniform.<type>.<name>` / `variable.<type>.<name>`).

Implements the common subset of the OptiFine/Iris expression language: arithmetic,
comparisons, && || !, vec2/vec3/vec4 constructors, component access (.x .y .z .w
.r .g .b .a), and the functions sin cos asin acos tan atan atan2 torad todeg min max
clamp abs floor ceil exp exp2 frac log log2 pow random round signum sqrt fmod if
between equals in smooth smoothstep. Expressions are translated to Python and
evaluated once per frame with the standard uniforms as the namespace."""
import math
import random
import re

import numpy as np


def _vec(*a):
    return np.array([float(x) for x in a], np.float64)


def _c(v, comp):
    i = "xyzw".find(comp)
    if i < 0:
        i = "rgba".find(comp)
    v = np.atleast_1d(np.asarray(v, np.float64))
    return float(v[i]) if i < v.size else 0.0


def _if(*args):
    # if(cond, val, [cond2, val2, ...], else)
    n = len(args)
    i = 0
    while i + 1 < n:
        if args[i]:
            return args[i + 1]
        i += 2
    return args[-1] if n % 2 == 1 else 0.0


def _frac(x):
    return x - math.floor(x)


def _signum(x):
    return (x > 0) - (x < 0)


FUNCS = {
    "sin": math.sin, "cos": math.cos, "asin": math.asin, "acos": math.acos, "tan": math.tan, "atan": math.atan,
    "atan2": math.atan2, "torad": math.radians, "todeg": math.degrees, "min": min, "max": max,
    "clamp": lambda x, a, b: min(max(x, a), b), "abs": abs, "floor": math.floor, "ceil": math.ceil, "exp": math.exp,
    "exp2": lambda x: 2.0 ** x, "frac": _frac, "log": math.log, "log2": math.log2, "pow": math.pow,
    "random": random.random, "round": round, "signum": _signum, "sqrt": lambda x: math.sqrt(max(0.0, x)),
    "fmod": math.fmod, "_if": _if, "between": lambda x, a, b: a <= x <= b,
    "equals": lambda a, b, eps: abs(a - b) <= eps, "_in": lambda x, *vals: x in vals,
    "smoothstep": lambda a, b, x: (lambda t: t * t * (3 - 2 * t))(min(max((x - a) / (b - a), 0.0), 1.0)),
    "vec2": _vec, "vec3": _vec, "vec4": _vec, "_c": _c, "true": True, "false": False, "pi": math.pi,
}


def translate(expr):
    e = expr
    e = re.sub(r"\bif\s*\(", "_if(", e)
    e = re.sub(r"\bin\s*\(", "_in(", e)
    e = e.replace("&&", " and ").replace("||", " or ")
    e = re.sub(r"!(?!=)", " not ", e)
    e = re.sub(r"\b([A-Za-z_]\w*)\.([xyzwrgba])\b", r"_c(\1, '\2')", e)
    return e


class CustomUniforms:
    def __init__(self, decls, log):
        self.log = log
        self.items = []
        self.smooth_state = {}
        for kind, typ, name, expr in decls:
            try:
                code = compile(translate(expr), f"<{kind}.{typ}.{name}>", "eval")
            except SyntaxError as e:
                log.warn(f"{kind}.{typ}.{name}: cannot parse '{expr}': {e}")
                continue
            self.items.append((kind, typ, name, expr, code))

    def evaluate(self, U):
        if not self.items:
            return {}
        ns = dict(FUNCS)
        ns.update(U)
        dt = float(U.get("frameTime", 1 / 60))
        call = [0]

        def smooth(*args):
            # smooth([id,] value, [fadeUpTime, [fadeDownTime]])
            if len(args) >= 2 and isinstance(args[0], (int, float)) and len(args) == 4:
                key, v, up, down = args
            else:
                key = ("site", call[0])
                v = args[0]
                up = args[1] if len(args) > 1 else 1.0
                down = args[2] if len(args) > 2 else up
            call[0] += 1
            prev = self.smooth_state.get(key)
            if prev is None:
                self.smooth_state[key] = v
                return v
            # Iris SmoothFloat: half-lives are given in deciseconds (computeDecay(halfLife * 0.1F))
            t = (up if v > prev else down) * 0.1
            a = 1.0 - math.exp(-dt * math.log(2) / max(t, 1e-6))
            nv = prev + (v - prev) * a
            self.smooth_state[key] = nv
            return nv
        ns["smooth"] = smooth
        out = {}
        for kind, typ, name, expr, code in self.items:
            try:
                v = eval(code, {"__builtins__": {}}, ns)  # noqa: S307 — pack-provided expressions, local tool
                if typ == "bool":
                    v = bool(v)
                elif typ == "int":
                    v = int(v)
                elif typ == "float":
                    v = float(v)
                else:
                    v = np.asarray(v, np.float64)
                ns[name] = v
                if kind == "uniform":
                    out[name] = v
            except Exception as e:  # noqa: BLE001
                self.log.once(("cu", name), f"WARN  {kind}.{typ}.{name} = {expr}: {e}")
        return out
