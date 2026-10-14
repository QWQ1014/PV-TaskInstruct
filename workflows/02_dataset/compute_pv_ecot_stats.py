import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_ACTIVE = DEFAULT_PROJECT / "cleaned_v2" / "pv_ecot_core_train_v2_active_enriched.jsonl"
DEFAULT_SAFE = DEFAULT_PROJECT / "cleaned_v2" / "pv_ecot_core_train_v2_safe_no_revision_enriched.jsonl"
DEFAULT_FINAL = (
    DEFAULT_PROJECT
    / "Extraction_project"
    / "data"
    / "pv_result_mechanism_attribution_style_v3_full.jsonl"
)
DEFAULT_MERGED_RESULTS = DEFAULT_PROJECT / "Extraction_project" / "data" / "merged_results"
LAYER_ORDER = {
    "L1_Intervention": 1,
    "L2_Structure_Composition": 2,
    "L3_Mechanism_Physics": 3,
    "L4_Device_Parameter": 4,
    "L5_Performance": 5,
}
LAYER_LABELS = list(LAYER_ORDER)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pct(part: int | float, whole: int | float) -> str:
    if not whole:
        return "0.00%"
    return f"{part / whole * 100:.2f}%"


def cell_type(row: dict[str, Any]) -> str:
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    meta = row.get("meta_info") if isinstance(row.get("meta_info"), dict) else {}
    return str(provenance.get("final_cell_type") or meta.get("cell_type") or "unknown")


def source_file(row: dict[str, Any]) -> str:
    return str(row.get("source_file") or "unknown")


def collect_sample_path_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    entities: set[tuple[Any, ...]] = set()
    relations: set[tuple[Any, ...]] = set()
    per_source: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"entities": set(), "relations": set(), "relation_occurrences": 0}
    )
    relation_counts: list[int] = []
    evidence_counts: list[int] = []
    predicate_counter: Counter[str] = Counter()
    layer_counter: Counter[str] = Counter()
    unique_entity_layer_counter: Counter[str] = Counter()
    relation_occurrences = 0
    evidence_occurrences = 0

    for row in rows:
        src = source_file(row)
        provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
        supporting_relations = provenance.get("supporting_relations")
        if not isinstance(supporting_relations, list):
            supporting_relations = []
        relation_counts.append(len(supporting_relations))

        supporting_evidences = provenance.get("supporting_evidences")
        if not isinstance(supporting_evidences, list):
            supporting_evidences = []
        evidence_counts.append(len(supporting_evidences))

        for rel in supporting_relations:
            if not isinstance(rel, dict):
                continue
            relation_occurrences += 1
            relation_sig = (
                src,
                rel.get("subject_id"),
                rel.get("raw_predicate"),
                rel.get("object_id"),
            )
            relations.add(relation_sig)
            per_source[src]["relations"].add(relation_sig)
            per_source[src]["relation_occurrences"] += 1
            predicate_counter[str(rel.get("predicate_group") or "unknown")] += 1

            evidences = rel.get("evidences")
            if isinstance(evidences, list):
                evidence_occurrences += len(evidences)

            for role in ("subject", "object"):
                ent = (
                    src,
                    rel.get(f"{role}_id"),
                    rel.get(f"{role}_name"),
                    rel.get(f"{role}_layer"),
                )
                entities.add(ent)
                per_source[src]["entities"].add(ent)
                layer_counter[str(rel.get(f"{role}_layer") or "unknown")] += 1

    entity_counts_by_source = [len(v["entities"]) for v in per_source.values()]
    relation_counts_by_source = [len(v["relations"]) for v in per_source.values()]

    for entity in entities:
        unique_entity_layer_counter[str(entity[3] or "unknown")] += 1

    return {
        "unique_path_entities": len(entities),
        "unique_path_relations": len(relations),
        "relation_occurrences": relation_occurrences,
        "evidence_occurrences_in_relations": evidence_occurrences,
        "avg_unique_entities_per_source": mean(entity_counts_by_source) if entity_counts_by_source else 0,
        "avg_unique_relations_per_source": mean(relation_counts_by_source) if relation_counts_by_source else 0,
        "avg_relations_per_sample": mean(relation_counts) if relation_counts else 0,
        "median_relations_per_sample": median(relation_counts) if relation_counts else 0,
        "min_relations_per_sample": min(relation_counts) if relation_counts else 0,
        "max_relations_per_sample": max(relation_counts) if relation_counts else 0,
        "avg_supporting_evidence_per_sample": mean(evidence_counts) if evidence_counts else 0,
        "median_supporting_evidence_per_sample": median(evidence_counts) if evidence_counts else 0,
        "entity_layer_mentions": dict(layer_counter.most_common()),
        "unique_entity_layer_counts": dict(unique_entity_layer_counter.most_common()),
        "predicate_group_occurrences": dict(predicate_counter.most_common()),
    }


def relation_layers(rel: dict[str, Any]) -> tuple[str, str]:
    return (
        str(rel.get("subject_layer") or "unknown"),
        str(rel.get("object_layer") or "unknown"),
    )


def node_key(rel: dict[str, Any], role: str) -> tuple[str, str]:
    return (
        str(rel.get(f"{role}_id") or rel.get(f"{role}_name") or "unknown"),
        str(rel.get(f"{role}_layer") or "unknown"),
    )


def has_directed_path(
    adjacency: dict[tuple[str, str], set[tuple[str, str]]],
    starts: set[tuple[str, str]],
    target_layers: set[str],
) -> bool:
    seen: set[tuple[str, str]] = set()
    stack = list(starts)
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        if node[1] in target_layers:
            return True
        stack.extend(adjacency.get(node, set()) - seen)
    return False


def longest_directed_path_edges(
    adjacency: dict[tuple[str, str], set[tuple[str, str]]],
    starts: set[tuple[str, str]],
    max_depth: int = 12,
) -> int:
    best = 0
    stack = [(node, 0, {node}) for node in starts]
    while stack:
        node, depth, seen = stack.pop()
        best = max(best, depth)
        if depth >= max_depth:
            continue
        for nxt in adjacency.get(node, set()):
            if nxt in seen:
                continue
            stack.append((nxt, depth + 1, seen | {nxt}))
    return best


def collect_mechanism_structure_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    layer_presence: Counter[str] = Counter()
    layer_count_distribution: Counter[int] = Counter()
    edge_span_counter: Counter[str] = Counter()

    full_five_layer = 0
    at_least_four_layers = 0
    at_least_three_layers = 0
    has_l1_l5 = 0
    has_l1_to_l5_path = 0
    has_l1_to_l4_or_l5_path = 0
    has_multi_hop_path = 0
    ordered_relation_count = 0
    known_layer_relation_count = 0
    longest_paths: list[int] = []

    for row in rows:
        provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
        supporting_relations = provenance.get("supporting_relations")
        if not isinstance(supporting_relations, list):
            supporting_relations = []

        layers: set[str] = set()
        nodes_by_layer: dict[str, set[tuple[str, str]]] = defaultdict(set)
        adjacency: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)

        for rel in supporting_relations:
            if not isinstance(rel, dict):
                continue
            subj_layer, obj_layer = relation_layers(rel)
            subj = node_key(rel, "subject")
            obj = node_key(rel, "object")
            adjacency[subj].add(obj)
            nodes_by_layer[subj_layer].add(subj)
            nodes_by_layer[obj_layer].add(obj)
            if subj_layer in LAYER_ORDER:
                layers.add(subj_layer)
            if obj_layer in LAYER_ORDER:
                layers.add(obj_layer)

            if subj_layer in LAYER_ORDER and obj_layer in LAYER_ORDER:
                known_layer_relation_count += 1
                delta = LAYER_ORDER[obj_layer] - LAYER_ORDER[subj_layer]
                if delta >= 0:
                    ordered_relation_count += 1
                edge_span_counter[f"L{LAYER_ORDER[subj_layer]}->L{LAYER_ORDER[obj_layer]}"] += 1

        for layer in layers:
            layer_presence[layer] += 1
        layer_count_distribution[len(layers)] += 1
        if len(layers) >= 3:
            at_least_three_layers += 1
        if len(layers) >= 4:
            at_least_four_layers += 1
        if set(LAYER_LABELS).issubset(layers):
            full_five_layer += 1
        if "L1_Intervention" in layers and "L5_Performance" in layers:
            has_l1_l5 += 1

        l1_nodes = nodes_by_layer.get("L1_Intervention", set())
        if l1_nodes and has_directed_path(adjacency, l1_nodes, {"L5_Performance"}):
            has_l1_to_l5_path += 1
        if l1_nodes and has_directed_path(
            adjacency, l1_nodes, {"L4_Device_Parameter", "L5_Performance"}
        ):
            has_l1_to_l4_or_l5_path += 1
        longest = longest_directed_path_edges(adjacency, l1_nodes) if l1_nodes else 0
        longest_paths.append(longest)
        if longest >= 2:
            has_multi_hop_path += 1

    layer_rows = []
    for layer in LAYER_LABELS:
        layer_rows.append(
            {
                "layer": layer,
                "samples_with_layer": layer_presence[layer],
                "sample_pct": pct(layer_presence[layer], total),
            }
        )

    return {
        "layer_rows": layer_rows,
        "layer_count_distribution": dict(sorted(layer_count_distribution.items())),
        "path_rows": [
            {
                "metric": "samples_with_at_least_3_layers",
                "count": at_least_three_layers,
                "pct": pct(at_least_three_layers, total),
                "definition": "Sample includes entities from at least three PV-ECoT mechanism layers.",
            },
            {
                "metric": "samples_with_at_least_4_layers",
                "count": at_least_four_layers,
                "pct": pct(at_least_four_layers, total),
                "definition": "Sample includes entities from at least four PV-ECoT mechanism layers.",
            },
            {
                "metric": "samples_with_all_5_layers",
                "count": full_five_layer,
                "pct": pct(full_five_layer, total),
                "definition": "Sample includes L1 intervention, L2 structure/composition, L3 mechanism/physics, L4 device parameter, and L5 performance.",
            },
            {
                "metric": "samples_with_L1_and_L5",
                "count": has_l1_l5,
                "pct": pct(has_l1_l5, total),
                "definition": "Sample contains both intervention and final performance layers.",
            },
            {
                "metric": "samples_with_directed_L1_to_L4_or_L5_path",
                "count": has_l1_to_l4_or_l5_path,
                "pct": pct(has_l1_to_l4_or_l5_path, total),
                "definition": "Directed supporting-relation graph has a path from intervention to device parameter or performance.",
            },
            {
                "metric": "samples_with_directed_L1_to_L5_path",
                "count": has_l1_to_l5_path,
                "pct": pct(has_l1_to_l5_path, total),
                "definition": "Directed supporting-relation graph has a path from intervention to final performance.",
            },
            {
                "metric": "samples_with_multi_hop_path_from_L1",
                "count": has_multi_hop_path,
                "pct": pct(has_multi_hop_path, total),
                "definition": "Longest directed path from an intervention node has at least two edges.",
            },
            {
                "metric": "ordered_layer_relation_ratio",
                "count": ordered_relation_count,
                "pct": pct(ordered_relation_count, known_layer_relation_count),
                "definition": "Relations whose object layer is the same as or downstream of the subject layer among known-layer relations.",
            },
        ],
        "edge_span_counter": dict(edge_span_counter.most_common()),
        "avg_longest_path_edges_from_L1": mean(longest_paths) if longest_paths else 0,
        "median_longest_path_edges_from_L1": median(longest_paths) if longest_paths else 0,
    }


def extract_graph_payload(obj: dict[str, Any]) -> dict[str, Any]:
    result = obj.get("result")
    if isinstance(result, dict):
        return result
    return obj


def collect_full_graph_stats(merged_dir: Path) -> dict[str, Any] | None:
    if not merged_dir.exists():
        return None

    graph_files = sorted(merged_dir.glob("*.json"))
    if not graph_files:
        return None

    paper_rows = []
    total_entities = 0
    total_relations = 0
    valid_graphs = 0
    layer_counter: Counter[str] = Counter()
    predicate_counter: Counter[str] = Counter()
    entity_type_counter: Counter[str] = Counter()

    for path in graph_files:
        with path.open("r", encoding="utf-8") as f:
            payload = extract_graph_payload(json.load(f))
        entities = payload.get("entities")
        relations = payload.get("relations")
        if not isinstance(entities, list):
            entities = []
        if not isinstance(relations, list):
            relations = []
        valid = bool(entities and relations)
        if valid:
            valid_graphs += 1
        total_entities += len(entities)
        total_relations += len(relations)
        for entity in entities:
            if isinstance(entity, dict):
                layer_counter[str(entity.get("meta_layer") or entity.get("entity_type") or "unknown")] += 1
                entity_type_counter[str(entity.get("entity_type") or "unknown")] += 1
        for rel in relations:
            if isinstance(rel, dict):
                predicate_counter[str(rel.get("predicate_group") or "unknown")] += 1
        paper_rows.append(
            {
                "source_file": path.name,
                "entities": len(entities),
                "relations": len(relations),
                "valid_graph": valid,
            }
        )

    return {
        "graph_files": len(graph_files),
        "valid_graphs": valid_graphs,
        "total_entities": total_entities,
        "total_relations": total_relations,
        "avg_entities_per_valid_graph": total_entities / valid_graphs if valid_graphs else 0,
        "avg_relations_per_valid_graph": total_relations / valid_graphs if valid_graphs else 0,
        "entity_layer_counts": dict(layer_counter.most_common()),
        "entity_type_counts": dict(entity_type_counter.most_common()),
        "predicate_group_counts": dict(predicate_counter.most_common()),
        "paper_rows": paper_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute PV-ECoT dataset statistics.")
    parser.add_argument("--active", type=Path, default=DEFAULT_ACTIVE)
    parser.add_argument("--safe", type=Path, default=DEFAULT_SAFE)
    parser.add_argument("--final", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--merged-results", type=Path, default=DEFAULT_MERGED_RESULTS)
    parser.add_argument("--out-dir", type=Path, default=Path("pv_ecot_stats"))
    args = parser.parse_args()

    active_rows = load_jsonl(args.active)
    safe_rows = load_jsonl(args.safe)
    final_rows = load_jsonl(args.final)

    active_sources = {source_file(row) for row in active_rows}
    safe_sources = {source_file(row) for row in safe_rows}

    safe_by_index = {i: row for i, row in enumerate(safe_rows, start=1)}
    final_sources = []
    for row in final_rows:
        meta = row.get("meta_info") if isinstance(row.get("meta_info"), dict) else {}
        try:
            source_index = int(meta.get("source_index"))
        except (TypeError, ValueError):
            continue
        if source_index in safe_by_index:
            final_sources.append(source_file(safe_by_index[source_index]))

    funnel_rows = [
        {
            "stage": "active_mechanism_samples",
            "rows": len(active_rows),
            "unique_sources": len(active_sources),
            "removed_from_previous": "",
            "stage_loss_rate": "",
        },
        {
            "stage": "safe_no_revision_samples",
            "rows": len(safe_rows),
            "unique_sources": len(safe_sources),
            "removed_from_previous": len(active_rows) - len(safe_rows),
            "stage_loss_rate": pct(len(active_rows) - len(safe_rows), len(active_rows)),
        },
        {
            "stage": "final_rewritten_instruction_samples",
            "rows": len(final_rows),
            "unique_sources": len(set(final_sources)),
            "removed_from_previous": len(safe_rows) - len(final_rows),
            "stage_loss_rate": pct(len(safe_rows) - len(final_rows), len(safe_rows)),
        },
    ]
    write_csv(
        args.out_dir / "funnel_stats.csv",
        funnel_rows,
        ["stage", "rows", "unique_sources", "removed_from_previous", "stage_loss_rate"],
    )

    coverage_counter = Counter(cell_type(row) for row in safe_rows)
    sources_by_type: dict[str, set[str]] = defaultdict(set)
    for row in safe_rows:
        sources_by_type[cell_type(row)].add(source_file(row))
    coverage_rows = []
    for label, sample_count in coverage_counter.most_common():
        source_count = len(sources_by_type[label])
        coverage_rows.append(
            {
                "cell_type": label,
                "samples": sample_count,
                "sample_pct": pct(sample_count, len(safe_rows)),
                "unique_sources": source_count,
                "source_pct": pct(source_count, len(safe_sources)),
            }
        )
    write_csv(
        args.out_dir / "coverage_by_cell_type.csv",
        coverage_rows,
        ["cell_type", "samples", "sample_pct", "unique_sources", "source_pct"],
    )

    simple_counters = []
    for name, counter in [
        ("chain_type", Counter((row.get("meta_info") or {}).get("chain_type", "unknown") for row in safe_rows)),
        ("sample_type", Counter((row.get("meta_info") or {}).get("sample_type", "unknown") for row in safe_rows)),
        ("chain_quality", Counter((row.get("meta_info") or {}).get("chain_quality", "unknown") for row in safe_rows)),
        ("grounded_quality", Counter((row.get("meta_info") or {}).get("grounded_quality", "unknown") for row in safe_rows)),
        ("final_input_mode", Counter((row.get("meta_info") or {}).get("input_mode", "unknown") for row in final_rows)),
        ("final_answer_style", Counter((row.get("meta_info") or {}).get("answer_style", "unknown") for row in final_rows)),
    ]:
        for label, count in counter.most_common():
            denominator = len(final_rows) if name.startswith("final_") else len(safe_rows)
            simple_counters.append({"metric": name, "label": label, "count": count, "pct": pct(count, denominator)})
    write_csv(args.out_dir / "categorical_stats.csv", simple_counters, ["metric", "label", "count", "pct"])

    path_stats = collect_sample_path_stats(safe_rows)
    structure_stats = collect_mechanism_structure_stats(safe_rows)
    write_csv(
        args.out_dir / "sampled_path_graph_stats.csv",
        [
            {
                key: value
                for key, value in path_stats.items()
                if not isinstance(value, dict)
            }
        ],
        [
            "unique_path_entities",
            "unique_path_relations",
            "relation_occurrences",
            "evidence_occurrences_in_relations",
            "avg_unique_entities_per_source",
            "avg_unique_relations_per_source",
            "avg_relations_per_sample",
            "median_relations_per_sample",
            "min_relations_per_sample",
            "max_relations_per_sample",
            "avg_supporting_evidence_per_sample",
            "median_supporting_evidence_per_sample",
        ],
    )
    write_csv(
        args.out_dir / "entity_layer_mentions.csv",
        [{"layer": k, "mentions": v} for k, v in path_stats["entity_layer_mentions"].items()],
        ["layer", "mentions"],
    )
    write_csv(
        args.out_dir / "sampled_unique_entity_layer_counts.csv",
        [
            {"layer": k, "unique_entities": v}
            for k, v in path_stats["unique_entity_layer_counts"].items()
        ],
        ["layer", "unique_entities"],
    )
    write_csv(
        args.out_dir / "predicate_group_occurrences.csv",
        [{"predicate_group": k, "occurrences": v} for k, v in path_stats["predicate_group_occurrences"].items()],
        ["predicate_group", "occurrences"],
    )
    write_csv(
        args.out_dir / "mechanism_layer_coverage.csv",
        structure_stats["layer_rows"],
        ["layer", "samples_with_layer", "sample_pct"],
    )
    write_csv(
        args.out_dir / "sample_layer_count_distribution.csv",
        [
            {
                "covered_layer_count": layer_count,
                "samples": count,
                "pct": pct(count, len(safe_rows)),
            }
            for layer_count, count in structure_stats["layer_count_distribution"].items()
        ],
        ["covered_layer_count", "samples", "pct"],
    )
    write_csv(
        args.out_dir / "path_completeness_stats.csv",
        structure_stats["path_rows"],
        ["metric", "count", "pct", "definition"],
    )
    write_csv(
        args.out_dir / "layer_transition_counts.csv",
        [
            {"transition": transition, "relations": count}
            for transition, count in structure_stats["edge_span_counter"].items()
        ],
        ["transition", "relations"],
    )

    graph_stats = collect_full_graph_stats(args.merged_results)
    alignment_rows: list[dict[str, Any]] = []
    if graph_stats:
        graph_sources = {row["source_file"] for row in graph_stats["paper_rows"]}
        safe_from_graph = [row for row in safe_rows if source_file(row) in graph_sources]
        final_from_graph = 0
        final_sources_from_graph: set[str] = set()
        final_without_graph_source = 0
        for row in final_rows:
            meta = row.get("meta_info") if isinstance(row.get("meta_info"), dict) else {}
            try:
                source_index = int(meta.get("source_index"))
            except (TypeError, ValueError):
                continue
            src = source_file(safe_by_index.get(source_index, {}))
            if src in graph_sources:
                final_from_graph += 1
                final_sources_from_graph.add(src)
            else:
                final_without_graph_source += 1

        alignment_rows = [
            {
                "metric": "graph_sources",
                "count": len(graph_sources),
                "note": "JSON graph files found in merged/refined graph directory",
            },
            {
                "metric": "safe_sources_overlapping_graph_dir",
                "count": len(safe_sources & graph_sources),
                "note": "Safe sample source files also present as graph files",
            },
            {
                "metric": "safe_sources_missing_from_graph_dir",
                "count": len(safe_sources - graph_sources),
                "note": "Safe sample source files not found in graph directory",
            },
            {
                "metric": "safe_rows_from_graph_dir",
                "count": len(safe_from_graph),
                "note": "Safe samples whose source file is present in graph directory",
            },
            {
                "metric": "final_rows_from_graph_dir",
                "count": final_from_graph,
                "note": "Final rewritten samples whose source file is present in graph directory",
            },
            {
                "metric": "final_unique_sources_from_graph_dir",
                "count": len(final_sources_from_graph),
                "note": "Unique graph sources represented in final rewritten samples",
            },
            {
                "metric": "final_rows_without_graph_source",
                "count": final_without_graph_source,
                "note": "Final rewritten samples whose source file is absent from graph directory",
            },
        ]
        write_csv(
            args.out_dir / "graph_sample_alignment.csv",
            alignment_rows,
            ["metric", "count", "note"],
        )

        write_csv(
            args.out_dir / "full_graph_paper_stats.csv",
            graph_stats["paper_rows"],
            ["source_file", "entities", "relations", "valid_graph"],
        )
        write_csv(
            args.out_dir / "full_graph_entity_layer_counts.csv",
            [
                {"layer": k, "entities": v}
                for k, v in graph_stats["entity_layer_counts"].items()
            ],
            ["layer", "entities"],
        )
        write_csv(
            args.out_dir / "full_graph_entity_type_counts.csv",
            [
                {"entity_type": k, "entities": v}
                for k, v in graph_stats["entity_type_counts"].items()
            ],
            ["entity_type", "entities"],
        )
        write_csv(
            args.out_dir / "full_graph_predicate_group_counts.csv",
            [
                {"predicate_group": k, "relations": v}
                for k, v in graph_stats["predicate_group_counts"].items()
            ],
            ["predicate_group", "relations"],
        )

    summary_lines = [
        "# PV-ECoT Statistics Summary",
        "",
        "## Funnel",
        "",
        "| Stage | Rows | Unique Sources | Removed | Loss Rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in funnel_rows:
        summary_lines.append(
            f"| {row['stage']} | {row['rows']} | {row['unique_sources']} | "
            f"{row['removed_from_previous']} | {row['stage_loss_rate']} |"
        )
    summary_lines.extend(
        [
            "",
            "## Coverage By Cell Type",
            "",
            "| Cell Type | Samples | Sample % | Unique Sources | Source % |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in coverage_rows:
        summary_lines.append(
            f"| {row['cell_type']} | {row['samples']} | {row['sample_pct']} | "
            f"{row['unique_sources']} | {row['source_pct']} |"
        )
    summary_lines.extend(
        [
            "",
            "## Sampled Mechanism Path Graph Stats",
            "",
            f"- Unique entities in sampled paths: {path_stats['unique_path_entities']}",
            f"- Unique relations in sampled paths: {path_stats['unique_path_relations']}",
            f"- Average unique entities per source: {path_stats['avg_unique_entities_per_source']:.2f}",
            f"- Average unique relations per source: {path_stats['avg_unique_relations_per_source']:.2f}",
            f"- Average supporting relations per sample: {path_stats['avg_relations_per_sample']:.2f}",
            f"- Average supporting evidence snippets per sample: {path_stats['avg_supporting_evidence_per_sample']:.2f}",
            "",
            "Note: sampled-path graph statistics are conservative counts from mechanism paths used by generated samples, not necessarily full-paper graph totals.",
            "",
            "## Mechanism Chain Coverage",
            "",
            "| Layer | Samples With Layer | Sample % |",
            "|---|---:|---:|",
        ]
    )
    for row in structure_stats["layer_rows"]:
        summary_lines.append(f"| {row['layer']} | {row['samples_with_layer']} | {row['sample_pct']} |")
    summary_lines.extend(
        [
            "",
            "## Path Completeness",
            "",
            "| Metric | Count | Percent |",
            "|---|---:|---:|",
        ]
    )
    for row in structure_stats["path_rows"]:
        summary_lines.append(f"| {row['metric']} | {row['count']} | {row['pct']} |")
    summary_lines.extend(
        [
            "",
            f"- Average longest directed path from L1: {structure_stats['avg_longest_path_edges_from_L1']:.2f} edges",
            f"- Median longest directed path from L1: {structure_stats['median_longest_path_edges_from_L1']} edges",
        ]
    )
    if graph_stats:
        summary_lines.extend(
            [
                "",
                "## Full Graph Stats",
                "",
                f"- Graph files: {graph_stats['graph_files']}",
                f"- Valid graphs: {graph_stats['valid_graphs']}",
                f"- Total entities: {graph_stats['total_entities']}",
                f"- Total relations: {graph_stats['total_relations']}",
                f"- Average entities per valid graph: {graph_stats['avg_entities_per_valid_graph']:.2f}",
                f"- Average relations per valid graph: {graph_stats['avg_relations_per_valid_graph']:.2f}",
            ]
        )
        if alignment_rows:
            summary_lines.extend(
                [
                    "",
                    "## Graph-Sample Alignment",
                    "",
                    "| Metric | Count | Note |",
                    "|---|---:|---|",
                ]
            )
            for row in alignment_rows:
                summary_lines.append(f"| {row['metric']} | {row['count']} | {row['note']} |")
    else:
        summary_lines.extend(
            [
                "",
                "## Full Graph Stats",
                "",
                f"`{args.merged_results}` was not found or contained no graph JSON files. Use the merged graph directory to compute full-paper entity/relation totals.",
            ]
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Wrote statistics to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
