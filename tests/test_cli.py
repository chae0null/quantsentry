"""CLI smoke tests via typer's CliRunner (no network, no subprocess)."""

from typer.testing import CliRunner

from quantsentry.cli import app

runner = CliRunner()


def test_make_zoo_and_scan(tmp_path):
    out = tmp_path / "zoo"
    res = runner.invoke(app, ["make-zoo", "--out", str(out), "--n-clean", "3", "--n-backdoor", "2"])
    assert res.exit_code == 0, res.output
    assert (out / "manifest.json").exists()

    res = runner.invoke(app, ["scan", "toy:backdoor:100"])
    assert res.exit_code == 0, res.output
    assert "verdict" in res.output.lower()
    assert "suspicious" in res.output.lower()


def test_scan_clean(tmp_path):
    res = runner.invoke(app, ["scan", "toy:clean:0"])
    assert res.exit_code == 0, res.output
    assert "verdict" in res.output.lower()


def test_scan_npz(tmp_path):
    out = tmp_path / "zoo"
    runner.invoke(app, ["make-zoo", "--out", str(out), "--n-clean", "2", "--n-backdoor", "2"])
    npz = next(out.glob("backdoor-*.npz"))
    res = runner.invoke(app, ["scan", str(npz)])
    assert res.exit_code == 0, res.output
    assert "verdict" in res.output.lower()
