"""Package the verified combined Dataset and current English execution guide."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = (args.out or root / "dist").resolve()
    out.mkdir(parents=True, exist_ok=True)
    status = json.loads((root / "VALIDATION_STATUS.json").read_text(encoding="utf-8"))
    if status.get("version") != "0.4.0" or not status.get("source_preservation", {}).get("passed"):
        raise RuntimeError("Expected current source-preservation evidence before packaging")
    if status.get("unit_tests", {}).get("failed") != 0 or not status.get("installed_distribution", {}).get("passed"):
        raise RuntimeError("Expected successful tests and installed-package checks")
    import markdown2
    guide_body = markdown2.markdown((root / "docs/DATASET_GUIDE.md").read_text(encoding="utf-8"), extras=["fenced-code-blocks", "tables"])
    guide = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Y-Map Dataset Guide</title><style>body{font:16px/1.7 system-ui,sans-serif;color:#203342;background:#eef3f6;margin:0}main{max-width:1040px;margin:30px auto;background:white;padding:38px;border-top:7px solid #187c87}h1,h2{color:#125963}h2{margin-top:32px}pre{background:#142e3c;color:#e8f5fa;padding:18px;overflow:auto;border-radius:6px}code{font-family:Consolas,monospace}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #dce5eb;padding:9px;text-align:left}th{background:#e9f3f4}a{color:#126e80}@media(max-width:700px){main{padding:20px;margin:0}table{display:block;overflow:auto}}</style></head><body><main>' + str(guide_body) + '</main></body></html>'
    (root / "docs/DATASET_GUIDE.html").write_text(guide, encoding="utf-8")
    (out / "YMap_Step_by_Step_EN.html").write_text(guide, encoding="utf-8")
    (root / "docs/START_HERE_EN.md").write_text("# Y-Map execution guide\n\nUse [DATASET_GUIDE.md](DATASET_GUIDE.md) for the combined Dataset and current commands. The previous release is preserved in archives/YMap_Research_Framework_v0_3.zip.\n", encoding="utf-8")
    (root / "docs/START_HERE_EN.html").write_text(guide, encoding="utf-8")
    source_manifest = json.loads((root / "provenance/imported_archive_manifest.json").read_text(encoding="utf-8"))
    imported_paths = {record["project_path"] for record in source_manifest["files"]}
    files = []
    transient_roots = {"runs", "build", "dist", ".venv", ".git", ".pytest_cache"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        name = relative.as_posix()
        if name == "RELEASE_MANIFEST.json":
            continue
        if name not in imported_paths:
            if relative.parts[0] in transient_roots or "__pycache__" in relative.parts or any(part.endswith(".egg-info") for part in relative.parts) or path.suffix in {".pyc", ".pyo"}:
                continue
        files.append(path)
    manifest = {"format_version": 2, "archive_root": root.name, "version": "0.4.0", "files": []}
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        manifest["files"].append({"path":path.relative_to(root).as_posix(),"sha256":digest.hexdigest(),"bytes":path.stat().st_size})
    manifest_path = root / "RELEASE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    files.append(manifest_path)
    archive = out / "YMap_Research_Framework_v0_4.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as package:
        for path in files:
            package.write(path, root.name + "/" + path.relative_to(root).as_posix())
    with zipfile.ZipFile(archive) as package:
        bad = package.testzip()
        if bad is not None:
            raise RuntimeError(f"Release ZIP CRC verification failed: {bad}")
    print(json.dumps({"archive":str(archive),"bytes":archive.stat().st_size,"files":len(files),"imported_original_files_preserved":len(imported_paths),"guide":str(out/"YMap_Step_by_Step_EN.html")}, indent=2))


if __name__ == "__main__":
    main()
