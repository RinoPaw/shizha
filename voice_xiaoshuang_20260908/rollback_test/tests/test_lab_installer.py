import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_model_manifest_has_unique_destinations_and_hashes() -> None:
    payload = json.loads((ROOT / "deploy" / "models.manifest.json").read_text("utf-8"))
    models = payload["models"]
    assert len(models) == 4
    assert len({model["destination"] for model in models}) == len(models)
    for model in models:
        assert model["repo_id"]
        assert model["revision"]
        assert model["files"]
        for digest in model["sha256"].values():
            assert len(digest) == 64
            int(digest, 16)


def test_bootstrap_targets_repository_installer() -> None:
    text = (ROOT / "bootstrap-shizha.cmd").read_text("utf-8")
    assert "RinoPaw/shizha/main/deploy/install-lab.ps1" in text
    assert "ExecutionPolicy Bypass -File $i" in text
    assert "shizha.env" in text


def test_installer_keeps_secrets_out_of_git() -> None:
    text = (ROOT / "deploy" / "install-lab.ps1").read_text("utf-8")
    assert "shizha.env" in text
    assert 'Destination (Join-Path $Project ".env")' in text
    assert "AI_API_KEY=" not in text
    assert "XF_API_SECRET=" not in text
