import argparse
import json
from pathlib import Path

from PIL import Image


NATIVE_FILENAMES = {
    "UQ-force.png",
    "UQ-force-rescaled.png",
    "UQ-QbC-force.png",
    "UQ-RND-force.png",
    "UQ-diff-UQ-parity.png",
    "UQ-diff-fdiff-parity.png",
    "UQ-force-qbc-rnd-fdiff-scatter.png",
    "UQ-force-qbc-rnd-fdiff-scatter-truncated.png",
    "UQ-force-qbc-rnd-identity-scatter.png",
    "UQ-force-qbc-rnd-identity-scatter-truncated.png",
    "UQ-force-fdiff-parity.png",
    "UQ-force-rescaled-fdiff-parity.png",
    "explained_variance.png",
    "coverage_score.png",
    "Final_sampled_PCAview.png",
    "Final_sampled_PCAview_by_pool.png",
    "DIRECT_PCA_feature_coverage.png",
    "Random_PCA_feature_coverage.png",
}

FIGURE_SIZE_RULES = {
    "coverage_score.png": (15, 5),
    "Final_sampled_PCAview.png": (12, 10),
    "Final_sampled_PCAview_by_pool.png": (14, 10),
    "DIRECT_PCA_feature_coverage.png": (10, 8),
    "Random_PCA_feature_coverage.png": (10, 8),
}


def expected_size(filename: str, fig_dpi: int) -> tuple[int, int]:
    width_inches, height_inches = FIGURE_SIZE_RULES.get(filename, (8, 6))
    return int(width_inches * fig_dpi), int(height_inches * fig_dpi)


def scan_view_dir(view_dir: Path, fig_dpi: int, strict_native: bool) -> tuple[list[dict], list[str]]:
    issues: list[str] = []
    rows: list[dict] = []
    if not view_dir.exists() or not view_dir.is_dir():
        return rows, [f"目录不存在: {view_dir}"]
    for file in sorted(view_dir.iterdir()):
        if not file.is_file():
            continue
        if file.suffix.lower() != ".png":
            issues.append(f"发现非 PNG 文件: {file}")
            continue
        with Image.open(file) as image:
            width, height = image.size
            mode = image.mode
            dpi = image.info.get("dpi")
        in_native = file.name in NATIVE_FILENAMES
        exp_w, exp_h = expected_size(file.name, fig_dpi)
        size_ok = abs(width - exp_w) <= 2 and abs(height - exp_h) <= 2
        dpi_ok = False
        if isinstance(dpi, tuple) and len(dpi) == 2:
            dpi_ok = abs(dpi[0] - fig_dpi) <= 2 and abs(dpi[1] - fig_dpi) <= 2
        mode_ok = mode in {"RGBA", "RGB"}
        if strict_native and not in_native:
            issues.append(f"非原生图名: {file.name}")
        if in_native and not size_ok:
            issues.append(f"尺寸不匹配: {file.name}, got={width}x{height}, expected={exp_w}x{exp_h}")
        if in_native and not dpi_ok:
            issues.append(f"DPI 不匹配: {file.name}, got={dpi}, expected≈{fig_dpi}")
        if in_native and not mode_ok:
            issues.append(f"色彩模式异常: {file.name}, mode={mode}")
        rows.append(
            {
                "file": file.name,
                "path": str(file),
                "native": in_native,
                "width": width,
                "height": height,
                "mode": mode,
                "dpi": dpi,
                "expected_width": exp_w,
                "expected_height": exp_h,
                "size_ok": size_ok,
                "dpi_ok": dpi_ok,
                "mode_ok": mode_ok,
            }
        )
    return rows, issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view-dir", action="append", required=True)
    parser.add_argument("--fig-dpi", type=int, default=300)
    parser.add_argument("--strict-native", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    all_rows: list[dict] = []
    all_issues: list[str] = []
    for path in args.view_dir:
        rows, issues = scan_view_dir(Path(path), args.fig_dpi, args.strict_native)
        all_rows.extend(rows)
        all_issues.extend(issues)

    result = {"total_files": len(all_rows), "issues": all_issues, "rows": all_rows}
    if args.json_out:
        args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if all_issues:
        for issue in all_issues:
            print(issue)
        return 1
    print(f"PASS: checked {len(all_rows)} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
