"""Fast tests without OpenGL: option discovery / rewriting, pack location, the mini preprocessor."""
import os
import zipfile

import pytest

from irisim import cli, cpp, options, pack

HERE = os.path.dirname(os.path.abspath(__file__))
MINIMAL = os.path.join(HERE, "..", "examples", "minimal_pack")
SHADERS = os.path.join(MINIMAL, "shaders")


def test_discover_options_minimal_pack():
    opts = options.discover_options(SHADERS)
    assert opts["SHADOWS"].kind == "bool" and opts["SHADOWS"].default is True
    assert opts["DEBUG_NORMALS"].kind == "bool" and opts["DEBUG_NORMALS"].default is False
    assert opts["FOG_DENSITY"].kind == "value" and opts["FOG_DENSITY"].values == ["0.0", "0.5", "1.0", "2.0"]
    assert opts["shadowMapResolution"].kind == "const" and opts["shadowMapResolution"].default == "1024"


def test_unreferenced_define_is_not_an_option(tmp_path):
    (tmp_path / "a.glsl").write_text("#define USED\n#define UNUSED\n#ifdef USED\n#endif\n"
                                     "const float notAnIrisConst = 1.0; // [1.0 2.0]\n")
    opts = options.discover_options(str(tmp_path))
    assert "USED" in opts
    assert "UNUSED" not in opts
    assert "notAnIrisConst" not in opts


def test_apply_overrides():
    ov = {"SHADOWS": False, "DEBUG_NORMALS": True, "FOG_DENSITY": "2.0", "shadowMapResolution": "2048"}
    assert options.apply_option_overrides("#define SHADOWS // x", ov).startswith("//#define SHADOWS")
    assert options.apply_option_overrides("//#define DEBUG_NORMALS // y", ov).startswith("#define DEBUG_NORMALS")
    assert "#define FOG_DENSITY 2.0" in options.apply_option_overrides("#define FOG_DENSITY 1.0 // [0.0 1.0 2.0]", ov)
    assert "= 2048;" in options.apply_option_overrides("const int shadowMapResolution = 1024; // [1024 2048]", ov)
    assert options.apply_option_overrides("float x = 1.0;", ov) == "float x = 1.0;"


def test_include_resolution_and_line_map():
    pack.set_shaders_dir(SHADERS)
    src = pack.load_stage(os.path.join(SHADERS, "composite.fsh"), {"FOG_DENSITY": "0.5"})
    text = "\n".join(src.lines)
    assert "#define FOG_DENSITY 0.5" in text
    assert ("lib/settings.glsl", 1) in src.origins


def test_find_shaders_dir_variants(tmp_path):
    assert cli.find_shaders_dir(MINIMAL, []) == os.path.abspath(SHADERS)
    assert cli.find_shaders_dir(SHADERS, []) == os.path.abspath(SHADERS)
    zpath = tmp_path / "pack.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for root, _, files in os.walk(SHADERS):
            for f in files:
                p = os.path.join(root, f)
                z.write(p, os.path.join("shaders", os.path.relpath(p, SHADERS)))
    tmp = []
    found = cli.find_shaders_dir(str(zpath), tmp)
    assert os.path.isfile(os.path.join(found, "final.fsh"))
    for t in tmp:
        import shutil
        shutil.rmtree(t)


def test_minicpp_conditions():
    pp = cpp.MiniCpp({"A": "1", "B": "2"})
    out = pp.process("#if A && B == 2\nyes\n#else\nno\n#endif\n#ifdef C\nc\n#endif\n")
    assert "yes" in out and "no" not in out and "\nc\n" not in out


def test_unknown_scenario_is_rejected():
    with pytest.raises(SystemExit):
        cli.parse_args([MINIMAL, "--scenario", "does-not-exist"])
