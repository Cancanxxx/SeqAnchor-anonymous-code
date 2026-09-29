"""Publication artifacts for the frozen expansion-mechanism model panel.

The normal artifact path is deliberately all-or-nothing: publication figures
and tables are emitted only after every model in the frozen protocol is
present.  ``validate_only=True`` exists solely to check partial/interim bundles
without creating artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPORT_NAME = "expansion_mechanism_multimodel_analysis_v1.json"
SUMMARY_NAME = "expansion_mechanism_summary_v1.csv"
UPDATES_NAME = "expansion_mechanism_updates_v1.csv"
FAMILIES_NAME = "expansion_mechanism_families_v1.csv"
ADDITION_ROLES = ("P", "S1", "E", "N")
ADDITION_MACRO_STEMS = {
    "P": "P",
    "S1": "SOne",
    "E": "E",
    "N": "N",
}

MODEL_LABELS = {
    "gemma3-4b-it": "Gemma 3 4B IT",
    "llama31-8b-instruct": "Llama 3.1 8B Instruct",
    "qwen3-8b": "Qwen3 8B",
    "phi-4": "Phi-4",
    "qwen25-14b-instruct": "Qwen2.5 14B Instruct",
    "deepseek-r1-distill-llama-8b": "DeepSeek-R1-Distill-Llama 8B",
}

MODEL_MACRO_STEMS = {
    "gemma3-4b-it": "GemmaThreeFourBIT",
    "llama31-8b-instruct": "LlamaThreeOneEightBInstruct",
    "qwen3-8b": "QwenThreeEightB",
    "phi-4": "PhiFour",
    "qwen25-14b-instruct": "QwenTwoFiveFourteenBInstruct",
    "deepseek-r1-distill-llama-8b": "DeepSeekRoneDistillLlamaEightB",
}

ROLE_LABELS = {
    "P": "safe paraphrase",
    "S1": "safe improvement",
    "E": "utility-equivalent",
    "N": "irrelevant",
}

ROLE_COLORS = {
    "P": "#0072B2",
    "S1": "#009E73",
    "E": "#D55E00",
    "N": "#7A5195",
}

ROLE_MARKERS = {"P": "o", "S1": "s", "E": "D", "N": "^"}


@dataclass(frozen=True)
class MechanismBundle:
    """Validated publication inputs in frozen protocol order."""

    input_dir: Path
    protocol_path: Path
    protocol: Mapping[str, Any]
    report: Mapping[str, Any]
    summary_rows: tuple[Mapping[str, str], ...]
    update_rows: tuple[Mapping[str, str], ...]
    family_rows: tuple[Mapping[str, str], ...]
    model_order: tuple[str, ...]
    source_paths: Mapping[str, Path]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required mechanism artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return value


def _load_csv(path: Path) -> tuple[Mapping[str, str], ...]:
    if not path.is_file():
        raise FileNotFoundError(f"required mechanism artifact is missing: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = tuple(dict(row) for row in csv.DictReader(handle))
    _require(bool(rows), f"CSV artifact is empty: {path}")
    return rows


def _finite(value: Any, label: str) -> float:
    number = float(value)
    _require(math.isfinite(number), f"{label} must be finite")
    return number


def _integer(value: Any, label: str) -> int:
    number = int(value)
    _require(number >= 0, f"{label} must be nonnegative")
    return number


def _close(left: Any, right: Any) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def load_bundle(
    input_dir: Path,
    protocol_path: Path,
    *,
    require_complete: bool = True,
) -> MechanismBundle:
    """Load and cross-check the analysis JSON and its three row-level CSVs."""

    input_dir = input_dir.resolve()
    protocol_path = protocol_path.resolve()
    source_paths = {
        "report": input_dir / REPORT_NAME,
        "summary": input_dir / SUMMARY_NAME,
        "updates": input_dir / UPDATES_NAME,
        "families": input_dir / FAMILIES_NAME,
        "protocol": protocol_path,
    }
    protocol = _load_json(protocol_path)
    report = _load_json(source_paths["report"])
    summary_rows = _load_csv(source_paths["summary"])
    update_rows = _load_csv(source_paths["updates"])
    family_rows = _load_csv(source_paths["families"])

    _require(
        protocol.get("artifact_type") == "expansion_mechanism_diagnostics_protocol_v1",
        "unexpected mechanism protocol type",
    )
    _require(
        report.get("artifact_type") == "expansion_mechanism_multimodel_analysis_v1",
        "unexpected multimodel report type",
    )
    _require(
        report.get("model_pooling_for_inference") is False,
        "publication artifacts require the frozen no-pooling analysis",
    )
    expected = tuple(str(spec["key"]) for spec in protocol["models"])
    _require(len(expected) == 6 and len(set(expected)) == 6, "frozen panel must contain six models")
    _require(set(expected) == set(MODEL_LABELS), "model-label registry differs from frozen panel")
    _require(set(expected) == set(MODEL_MACRO_STEMS), "macro registry differs from frozen panel")

    models = report.get("models")
    _require(isinstance(models, Mapping), "report models must be a JSON object")
    present = set(str(key) for key in models)
    unknown = sorted(present - set(expected))
    _require(not unknown, f"report contains models outside the frozen panel: {', '.join(unknown)}")
    _require(bool(present), "report contains no models")
    _require(int(report.get("model_count", -1)) == len(present), "report model_count differs")
    if require_complete and present != set(expected):
        missing = [key for key in expected if key not in present]
        raise RuntimeError(
            "publication artifacts require all six frozen models; "
            f"found {len(present)}/6; missing: {', '.join(missing)}"
        )

    model_order = tuple(key for key in expected if key in present)
    csv_groups = {
        "summary": summary_rows,
        "updates": update_rows,
        "families": family_rows,
    }
    for name, rows in csv_groups.items():
        row_models = {str(row.get("model_key", "")) for row in rows}
        _require(row_models == present, f"{name} CSV model set differs from report")

    summary_by_key: dict[tuple[str, str], Mapping[str, str]] = {}
    for row in summary_rows:
        key = (str(row["model_key"]), str(row["addition_role"]))
        _require(key not in summary_by_key, f"duplicate summary row: {key}")
        summary_by_key[key] = row

    for model_key in model_order:
        model = models[model_key]
        _require(isinstance(model, Mapping), f"model summary must be an object: {model_key}")
        by_role = model.get("by_addition_role")
        _require(isinstance(by_role, Mapping), f"by-addition summary missing: {model_key}")
        _require(set(by_role) == set(ADDITION_ROLES), f"addition roles differ: {model_key}")
        for role in ADDITION_ROLES:
            result = by_role[role]
            margin = result["margin_shift"]
            estimate = _finite(margin["mean"], f"{model_key}/{role} mean margin shift")
            interval = margin["cluster_bootstrap_95pct"]
            _require(
                isinstance(interval, Sequence) and len(interval) == 2,
                f"{model_key}/{role} margin interval must have two endpoints",
            )
            lower = _finite(interval[0], f"{model_key}/{role} lower margin endpoint")
            upper = _finite(interval[1], f"{model_key}/{role} upper margin endpoint")
            _require(lower <= upper, f"{model_key}/{role} margin interval is reversed")
            direct = result["direct_reversal"]
            events = _integer(direct["events"], f"{model_key}/{role} reversals")
            total = _integer(direct["total"], f"{model_key}/{role} eligible updates")
            _require(events <= total and total > 0, f"invalid reversal count: {model_key}/{role}")
            summary = summary_by_key.get((model_key, role))
            _require(summary is not None, f"summary CSV row missing: {model_key}/{role}")
            _require(int(summary["direct_reversals"]) == events, "summary reversal count differs")
            _require(int(summary["eligible_safe_base"]) == total, "summary denominator differs")
            _require(_close(summary["direct_reversal_rate"], direct["rate"]), "rate differs")
            _require(_close(summary["mean_margin_shift"], estimate), "margin estimate differs")

        overall = model["direct_reversal_given_safe_base"]
        _require(
            _integer(overall["workflow_family_clusters"], "workflow clusters") > 0,
            f"no eligible workflow families: {model_key}",
        )
        dissociation = model["joint_reversal_dissociations"]
        for field in (
            "both_independent_axes_rank_S0_above_U0",
            "strict_candidate_local_correctness_on_both_axes",
        ):
            value = dissociation[field]
            events = _integer(value["events"], f"{model_key}/{field} events")
            total = _integer(value["total"], f"{model_key}/{field} total")
            _require(events <= total, f"invalid dissociation count: {model_key}/{field}")

        model_updates = sum(row["model_key"] == model_key for row in update_rows)
        model_families = sum(row["model_key"] == model_key for row in family_rows)
        _require(model_updates == 720, f"{model_key} must have 720 update rows")
        _require(model_families == 180, f"{model_key} must have 180 family rows")

    return MechanismBundle(
        input_dir=input_dir,
        protocol_path=protocol_path,
        protocol=protocol,
        report=report,
        summary_rows=summary_rows,
        update_rows=update_rows,
        family_rows=family_rows,
        model_order=model_order,
        source_paths=source_paths,
    )


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _fraction(value: Mapping[str, Any]) -> str:
    events = int(value["events"])
    total = int(value["total"])
    if total == 0:
        return r"0/0 (--\%)"
    return f"{events}/{total} ({100.0 * events / total:.1f}\\%)"


def direct_reversal_table(bundle: MechanismBundle) -> str:
    """Compact per-model table; no cross-model pooling is performed."""

    lines = [
        "% Generated by build_expansion_mechanism_publication_v1.py.",
        "% Cells are direct safe-to-old-unsafe reversals among safe-base workflows.",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Model & Base safe & P & S1 & E & N & All additions \\",
        r"\midrule",
    ]
    models = bundle.report["models"]
    for model_key in bundle.model_order:
        model = models[model_key]
        safe_base = model["safe_base_families"]
        cells = [f"{int(safe_base['events'])}/{int(safe_base['total'])}"] + [
            _fraction(model["by_addition_role"][role]["direct_reversal"])
            for role in ADDITION_ROLES
        ]
        cells.append(_fraction(model["direct_reversal_given_safe_base"]))
        lines.append(f"{_latex_escape(MODEL_LABELS[model_key])} & " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "% P: safe paraphrase; S1: safe improvement; E: utility-equivalent; N: irrelevant.",
            "",
        ]
    )
    return "\n".join(lines)


def dissociation_table(bundle: MechanismBundle) -> str:
    """Compact joint-menu versus independent-scoring dissociation table."""

    lines = [
        "% Generated by build_expansion_mechanism_publication_v1.py.",
        "% Dissociation denominators are observed direct reversals, not all updates.",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & Joint reversals & Independent ranking favors S0 & Strict local correctness \\",
        r"\midrule",
    ]
    models = bundle.report["models"]
    for model_key in bundle.model_order:
        model = models[model_key]
        overall = model["direct_reversal_given_safe_base"]
        dissociation = model["joint_reversal_dissociations"]
        cells = [
            _fraction(overall),
            _fraction(dissociation["both_independent_axes_rank_S0_above_U0"]),
            _fraction(dissociation["strict_candidate_local_correctness_on_both_axes"]),
        ]
        lines.append(f"{_latex_escape(MODEL_LABELS[model_key])} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _estimate_interval(value: Mapping[str, Any], interval_key: str) -> str:
    estimate = float(value["mean"])
    lower, upper = value[interval_key]
    return f"{estimate:.2f} [{float(lower):.2f}, {float(upper):.2f}]"


def presentation_effects_table(bundle: MechanismBundle) -> str:
    """Order and insertion-position contrasts for every model/addition pair."""

    lines = [
        "% Generated by build_expansion_mechanism_publication_v1.py.",
        "% Contrasts use safe-base workflow families and pointwise family bootstraps.",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Model & Addition & Candidate first $-$ last & $S_0$ before $U_0$ $-$ reverse \\",
        r"\midrule",
    ]
    models = bundle.report["models"]
    for model_index, model_key in enumerate(bundle.model_order):
        model = models[model_key]
        for role_index, role in enumerate(ADDITION_ROLES):
            inference = model["by_addition_role"][role]["presentation"][
                "family_bootstrap_inference"
            ]
            candidate_position = inference["paired_position_contrasts"]["0_minus_2"][
                "mean_margin_S0_minus_U0"
            ]
            old_pair_order = inference["paired_relative_order_contrast"][
                "mean_margin_S0_minus_U0"
            ]
            model_label = MODEL_LABELS[model_key] if role_index == 0 else ""
            lines.append(
                f"{_latex_escape(model_label)} & {role} & "
                f"{_estimate_interval(candidate_position, 'family_bootstrap_95pct')} & "
                f"{_estimate_interval(old_pair_order, 'family_bootstrap_95pct')} \\\\"
            )
        if model_index + 1 < len(bundle.model_order):
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def semantic_contrasts_table(bundle: MechanismBundle) -> str:
    """Within-family addition-type contrasts for safe-base workflow families."""

    contrasts = (
        ("P_minus_N", r"P $-$ N"),
        ("P_minus_E", r"P $-$ E"),
        ("S1_minus_N", r"S1 $-$ N"),
        ("E_minus_N", r"E $-$ N"),
    )
    lines = [
        "% Generated by build_expansion_mechanism_publication_v1.py.",
        "% Negative values mean the left addition shifts farther toward U0.",
        r"\begin{tabular}{llc}",
        r"\toprule",
        r"Model & Contrast & Paired margin-shift difference \\",
        r"\midrule",
    ]
    models = bundle.report["models"]
    for model_index, model_key in enumerate(bundle.model_order):
        values = models[model_key]["semantic_competition_margin_shift_contrasts"][
            "safe_base_workflow_families"
        ]
        for contrast_index, (contrast_key, contrast_label) in enumerate(contrasts):
            model_label = MODEL_LABELS[model_key] if contrast_index == 0 else ""
            lines.append(
                f"{_latex_escape(model_label)} & {contrast_label} & "
                f"{_estimate_interval(values[contrast_key], 'family_bootstrap_95pct')} \\\\"
            )
        if model_index + 1 < len(bundle.model_order):
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def candidate_local_accuracy_table(bundle: MechanismBundle) -> str:
    """Candidate-local validity checks that qualify the dissociation analysis."""

    lines = [
        "% Generated by build_expansion_mechanism_publication_v1.py.",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & Policy both correct & Approval both correct & Rank dissociation & Strict dissociation \\",
        r"\midrule",
    ]
    models = bundle.report["models"]
    for model_key in bundle.model_order:
        model = models[model_key]
        accuracy = model["independent_candidate_local_accuracy"]
        dissociation = model["joint_reversal_dissociations"]
        cells = [
            _fraction(accuracy["policy_compliance"]["both_correct"]),
            _fraction(accuracy["overall_approval"]["both_correct"]),
            _fraction(dissociation["both_independent_axes_rank_S0_above_U0"]),
            _fraction(dissociation["strict_candidate_local_correctness_on_both_axes"]),
        ]
        lines.append(f"{_latex_escape(MODEL_LABELS[model_key])} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def macros(bundle: MechanismBundle) -> str:
    """Stable LaTeX macros for prose and captions."""

    lines = [
        "% Generated by build_expansion_mechanism_publication_v1.py.",
        "% Percent macros contain numbers only; append \\% in prose.",
        rf"\providecommand{{\ExpansionMechanismModelCount}}{{{len(bundle.model_order)}}}",
    ]
    models = bundle.report["models"]
    for model_key in bundle.model_order:
        stem = MODEL_MACRO_STEMS[model_key]
        model = models[model_key]
        for role in ADDITION_ROLES:
            result = model["by_addition_role"][role]
            direct = result["direct_reversal"]
            margin = result["margin_shift"]
            lower, upper = margin["cluster_bootstrap_95pct"]
            prefix = rf"\ExpansionMechanism{stem}{ADDITION_MACRO_STEMS[role]}"
            lines.extend(
                [
                    rf"\providecommand{{{prefix}ReversalEvents}}{{{int(direct['events'])}}}",
                    rf"\providecommand{{{prefix}ReversalTotal}}{{{int(direct['total'])}}}",
                    rf"\providecommand{{{prefix}ReversalPercent}}"
                    rf"{{{100.0 * float(direct['rate']):.1f}}}",
                    rf"\providecommand{{{prefix}MarginShift}}{{{float(margin['mean']):.2f}}}",
                    rf"\providecommand{{{prefix}MarginShiftLower}}{{{float(lower):.2f}}}",
                    rf"\providecommand{{{prefix}MarginShiftUpper}}{{{float(upper):.2f}}}",
                ]
            )
        overall = model["direct_reversal_given_safe_base"]
        dissociation = model["joint_reversal_dissociations"]
        safe_base = model["safe_base_families"]
        affected = model["families_with_any_direct_reversal_given_safe_base"]
        base_winner_changes = model["base_presentation_sensitivity"]["winner_changes"]
        for label, value in (
            ("SafeBaseFamilies", safe_base),
            ("AffectedFamilies", affected),
            ("BaseWinnerChanges", base_winner_changes),
        ):
            prefix = rf"\ExpansionMechanism{stem}{label}"
            rate = 100.0 * float(value["rate"])
            lines.extend(
                [
                    rf"\providecommand{{{prefix}Events}}{{{int(value['events'])}}}",
                    rf"\providecommand{{{prefix}Total}}{{{int(value['total'])}}}",
                    rf"\providecommand{{{prefix}Percent}}{{{rate:.1f}}}",
                ]
            )
        for label, value in (
            ("AllReversals", overall),
            (
                "IndependentRankingDissociation",
                dissociation["both_independent_axes_rank_S0_above_U0"],
            ),
            (
                "StrictLocalDissociation",
                dissociation["strict_candidate_local_correctness_on_both_axes"],
            ),
        ):
            prefix = rf"\ExpansionMechanism{stem}{label}"
            rate = None if int(value["total"]) == 0 else 100.0 * float(value["rate"])
            lines.extend(
                [
                    rf"\providecommand{{{prefix}Events}}{{{int(value['events'])}}}",
                    rf"\providecommand{{{prefix}Total}}{{{int(value['total'])}}}",
                    rf"\providecommand{{{prefix}Percent}}"
                    rf"{{{'--' if rate is None else f'{rate:.1f}'}}}",
                ]
            )
    lines.append("")
    return "\n".join(lines)


def render_margin_shift_forest(bundle: MechanismBundle, output_path: Path) -> None:
    """Render a vector PDF with separate model/role estimates and no pooled row."""

    import matplotlib as mpl

    mpl.use("pdf")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    rows: list[tuple[str, str, float, float, float, float]] = []
    y_cursor = 0.0
    group_centers: list[tuple[float, str]] = []
    group_spans: list[tuple[float, float]] = []
    models = bundle.report["models"]
    for model_key in bundle.model_order:
        start = y_cursor
        for role in ADDITION_ROLES:
            margin = models[model_key]["by_addition_role"][role]["margin_shift"]
            lower, upper = margin["cluster_bootstrap_95pct"]
            rows.append(
                (
                    model_key,
                    role,
                    y_cursor,
                    float(margin["mean"]),
                    float(lower),
                    float(upper),
                )
            )
            y_cursor += 1.0
        end = y_cursor - 1.0
        group_centers.append(((start + end) / 2.0, MODEL_LABELS[model_key]))
        group_spans.append((start - 0.45, end + 0.45))
        y_cursor += 0.75

    with mpl.rc_context(
        {
            # Let LaTeX typeset figure text so the publication PDF uses the
            # same embedded Type 1 font pipeline as the IEEE manuscript.
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times"],
            "font.size": 8.0,
            "axes.labelsize": 9.0,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "savefig.transparent": False,
        }
    ):
        figure, axis = plt.subplots(figsize=(7.15, 5.9))
        for index, (low, high) in enumerate(group_spans):
            if index % 2 == 0:
                axis.axhspan(low, high, color="#F3F4F6", linewidth=0, zorder=0)
        axis.axvline(0.0, color="#4B5563", linewidth=0.8, linestyle="--", zorder=1)
        for _model_key, role, y, estimate, lower, upper in rows:
            axis.errorbar(
                estimate,
                y,
                xerr=[[estimate - lower], [upper - estimate]],
                fmt=ROLE_MARKERS[role],
                markersize=4.2,
                markerfacecolor=ROLE_COLORS[role],
                markeredgecolor="white",
                markeredgewidth=0.45,
                color=ROLE_COLORS[role],
                ecolor=ROLE_COLORS[role],
                elinewidth=1.15,
                capsize=2.2,
                capthick=0.8,
                zorder=3,
            )

        axis.set_yticks([row[2] for row in rows], labels=[row[1] for row in rows])
        axis.tick_params(axis="y", length=0, pad=3)
        axis.set_ylim(-0.7, rows[-1][2] + 0.7)
        axis.invert_yaxis()
        axis.set_xlabel(r"Mean change in $S_0-U_0$ margin after addition")
        axis.grid(axis="x", color="#D1D5DB", linewidth=0.55, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.text(
            0.0,
            1.015,
            r"shift toward unsafe $U_0$  $\leftarrow$",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.2,
            color="#4B5563",
        )
        axis.text(
            1.0,
            1.015,
            r"$\rightarrow$  shift toward safe $S_0$",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.2,
            color="#4B5563",
        )
        for center, label in group_centers:
            axis.text(
                -0.055,
                center,
                label,
                transform=axis.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=7.6,
                fontweight="bold",
                clip_on=False,
            )
        handles = [
            Line2D(
                [0],
                [0],
                marker=ROLE_MARKERS[role],
                color=ROLE_COLORS[role],
                markerfacecolor=ROLE_COLORS[role],
                linestyle="none",
                markersize=5,
                label=f"{role}: {ROLE_LABELS[role]}",
            )
            for role in ADDITION_ROLES
        ]
        axis.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.085),
            frameon=False,
            ncol=2,
            fontsize=7.2,
            columnspacing=1.5,
            handletextpad=0.45,
        )
        figure.subplots_adjust(left=0.36, right=0.985, top=0.96, bottom=0.14)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output_path,
            format="pdf",
            metadata={
                "Title": "Expansion-mechanism margin shifts by model and addition type",
                "Author": None,
                "Creator": "expansion_mechanism_publication_v1",
                "Producer": "Matplotlib PDF backend",
                "CreationDate": None,
                "ModDate": None,
            },
        )
        plt.close(figure)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_text(path: Path, text: str) -> None:
    payload = text.encode("utf-8")
    if path.exists() and path.read_bytes() == payload:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def generate_artifacts(bundle: MechanismBundle, output_dir: Path) -> Mapping[str, Path]:
    """Generate the complete six-model publication artifact set."""

    if len(bundle.model_order) != 6:
        raise RuntimeError("publication artifact generation is disabled for incomplete panels")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "forest_pdf": output_dir / "expansion_margin_shift_forest_v1.pdf",
        "direct_table": output_dir / "expansion_direct_reversals_table_v1.tex",
        "dissociation_table": output_dir / "expansion_joint_independent_table_v1.tex",
        "presentation_table": output_dir / "expansion_presentation_effects_table_v1.tex",
        "semantic_table": output_dir / "expansion_semantic_contrasts_table_v1.tex",
        "local_accuracy_table": output_dir / "expansion_candidate_local_accuracy_table_v1.tex",
        "macros": output_dir / "expansion_mechanism_macros_v1.tex",
    }
    render_margin_shift_forest(bundle, outputs["forest_pdf"])
    _write_text(outputs["direct_table"], direct_reversal_table(bundle))
    _write_text(outputs["dissociation_table"], dissociation_table(bundle))
    _write_text(outputs["presentation_table"], presentation_effects_table(bundle))
    _write_text(outputs["semantic_table"], semantic_contrasts_table(bundle))
    _write_text(outputs["local_accuracy_table"], candidate_local_accuracy_table(bundle))
    _write_text(outputs["macros"], macros(bundle))

    manifest_path = output_dir / "expansion_publication_artifacts_manifest_v1.json"
    manifest = {
        "schema_version": "1.0.0",
        "artifact_type": "expansion_mechanism_publication_artifacts_v1",
        "model_order": list(bundle.model_order),
        "pooled_estimate_rendered": False,
        "source_sha256": {
            name: _sha256(path) for name, path in sorted(bundle.source_paths.items())
        },
        "artifact_sha256": {
            name: _sha256(path) for name, path in sorted(outputs.items())
        },
    }
    _write_text(
        manifest_path,
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
    )
    return {**outputs, "manifest": manifest_path}
