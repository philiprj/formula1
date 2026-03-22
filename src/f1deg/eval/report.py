"""Generate evaluation reports in Markdown format."""

import json
from pathlib import Path


def generate_cv_report(
    cv_results: dict,
    model_name: str,
    output_dir: Path,
) -> Path:
    """Generate a Markdown evaluation report from CV results.

    Args:
        cv_results: Output from leave_one_race_out_cv().
        model_name: Name of the model (e.g., "linear", "bayesian").
        output_dir: Directory to write the report.

    Returns:
        Path to the generated report.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"eval_report_{model_name}.md"

    lines = [
        f"# Evaluation Report: {model_name}",
        "",
        "## Aggregate Metrics",
        "",
    ]

    agg = cv_results.get("aggregate", {})
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for key, value in sorted(agg.items()):
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.4f} |")
        else:
            lines.append(f"| {key} | {value} |")

    lines.extend(["", "## Per-Fold Results", ""])

    fold_results = cv_results.get("fold_results", [])
    if fold_results:
        headers = sorted(fold_results[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for fold in fold_results:
            row = []
            for h in headers:
                v = fold.get(h, "")
                if isinstance(v, float):
                    row.append(f"{v:.4f}")
                else:
                    row.append(str(v))
            lines.append("| " + " | ".join(row) + " |")

    report_text = "\n".join(lines) + "\n"
    report_path.write_text(report_text)

    # Also save raw results as JSON for programmatic access
    json_path = output_dir / f"cv_results_{model_name}.json"
    json_path.write_text(json.dumps(cv_results, indent=2, default=str))

    return report_path


def generate_comparison_report(
    results: dict[str, dict],
    output_dir: Path,
) -> Path:
    """Generate a comparison report across multiple models.

    Args:
        results: Dict mapping model_name -> cv_results.
        output_dir: Directory to write the report.

    Returns:
        Path to the generated report.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "model_comparison.md"

    lines = [
        "# Model Comparison Report",
        "",
        "## Summary",
        "",
        "| Model | MAE (s) | RMSE (s) | PI Coverage | PI Width (s) |",
        "|-------|---------|----------|-------------|--------------|",
    ]

    for name, result in sorted(results.items()):
        agg = result.get("aggregate", {})
        mae = agg.get("mae", float("nan"))
        rmse = agg.get("rmse", float("nan"))
        pi_cov = agg.get("pi_coverage_95", float("nan"))
        pi_width = agg.get("pi_width_mean", float("nan"))
        lines.append(f"| {name} | {mae:.3f} | {rmse:.3f} | {pi_cov:.3f} | {pi_width:.3f} |")

    report_text = "\n".join(lines) + "\n"
    report_path.write_text(report_text)
    return report_path
