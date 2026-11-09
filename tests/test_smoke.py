"""End-to-end smoke test: render the minimal example pack (needs Mesa llvmpipe + Xvfb)."""
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
MINIMAL = os.path.join(HERE, "..", "examples", "minimal_pack")

needs_display = pytest.mark.skipif(shutil.which("xvfb-run") is None and not os.environ.get("DISPLAY"),
                                   reason="needs xvfb-run or an X display")


@needs_display
def test_render_minimal_pack(tmp_path):
    cmd = [sys.executable, "-m", "irisim", MINIMAL, "--scenario", "noon", "--width", "160", "--height", "90",
           "--frames", "2", "--out", str(tmp_path), "--name", "smoke", "-q"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    out = tmp_path / "smoke"
    log = (out / "log.txt").read_text() if (out / "log.txt").exists() else ""
    assert r.returncode == 0, r.stdout + r.stderr + log
    assert (out / "final.png").is_file()
    assert "failed programs: 0" in log
    assert "!! NaN" not in log
    passes = os.listdir(out / "passes")
    assert any("composite_colortex0" in p for p in passes)
    assert any("shadow_shadowtex0" in p for p in passes)
    from PIL import Image
    import numpy as np
    img = np.asarray(Image.open(out / "final.png")).astype(np.float32)
    assert img.shape == (90, 160, 3)
    assert img.std() > 5.0, "final image is flat"


@needs_display
def test_option_override_changes_image(tmp_path):
    def render(name, *extra):
        cmd = [sys.executable, "-m", "irisim", MINIMAL, "--scenario", "noon", "--width", "96", "--height", "54",
               "--frames", "1", "--dump", "none", "--out", str(tmp_path), "--name", name, "-q", *extra]
        assert subprocess.run(cmd, timeout=600).returncode == 0
        from PIL import Image
        import numpy as np
        return np.asarray(Image.open(tmp_path / name / "final.png")).astype(np.float32)
    a = render("default")
    b = render("normals", "--on", "DEBUG_NORMALS")
    assert abs(a - b).mean() > 2.0
