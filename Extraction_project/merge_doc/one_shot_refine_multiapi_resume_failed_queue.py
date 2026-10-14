import json
import asyncio
import os
import re
import sys
import logging
import time
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional

# ----------------------------------------------------
# 1. 动态路径与基础配置
# ----------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from Extraction_project.llm_api.model_api_factory import get_model_api
    from Extraction_project.config import API_CONFIG
except ImportError as e:
    print(f"导入失败! 路径: {PROJECT_ROOT} | 错误: {e}")
    raise

# ================= 配置区域 =================
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
INPUT_DIR = os.getenv("PV_MERGED_RESULTS_DIR", os.path.join(REPO_ROOT, "outputs", "merged_results"))
OUTPUT_DIR = os.getenv("PV_REFINED_RESULTS_DIR", os.path.join(REPO_ROOT, "outputs", "refined_results"))

# 支持多个 API：按顺序配置即可，内部会交替分配任务（轮询）
API_NAMES = [
"silicon_deepseek_r1_0528"
]

MAX_RETRIES = 1
API_TIMEOUT_SECONDS = 1000

# 断点续传 / 运行状态
RESUME_MODE = True  # 为 True 时：若输出文件已存在且有效，则自动跳过
OVERWRITE_EXISTING = False  # 为 True 时：即使已存在结果，也强制重跑
RUN_STATUS_FILE = os.path.join(OUTPUT_DIR, "_run_status.json")
FAILED_CASES_JSONL = os.path.join(OUTPUT_DIR, "_failed_cases.jsonl")
FAILED_RAW_RESP_DIR = os.path.join(OUTPUT_DIR, "_failed_raw_responses")
SKIP_PREVIOUS_FAILED_IN_MAIN = False

SAVE_VALIDATION_REPORT = True
REPORT_DIR = os.path.join(OUTPUT_DIR, "_validation_reports")

USE_ONE_SHOT_REFINEMENT = True
ONE_SHOT_FALLBACK_TO_LEGACY = False
ONE_SHOT_TEMPERATURE = 0.0


def _calc_total_concurrency(api_names: List[str]) -> int:
    if "API_CONFIG" not in globals():
        return max(len(api_names), 1)
    total = 0
    for name in api_names:
        total += max(int(API_CONFIG.get(name, {}).get("concurrency", 1)), 1)
    return max(total, 1)


MAX_CONCURRENCY = _calc_total_concurrency(API_NAMES)

os.makedirs(OUTPUT_DIR, exist_ok=True)
if SAVE_VALIDATION_REPORT:
    os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(FAILED_RAW_RESP_DIR, exist_ok=True)


# ----------------------------------------------------
# 1.1 断点续传与进度工具
# ----------------------------------------------------
def atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(tmp_path, path)


def load_json_file(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def append_jsonl_line(path: str, item: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_failed_case_files(path: str) -> set[str]:
    files = set()
    if not os.path.exists(path):
        return files
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            src = str(obj.get("file") or obj.get("source_file") or "").strip()
            if src:
                files.add(src)
    return files


def classify_error_type(message: str) -> str:
    m = normalize_text(message).lower()
    if "timeout" in m:
        return "api_timeout"
    if "json parsing failed" in m or "unterminated string" in m or "expecting property name" in m or "no json object found" in m:
        return "json_parse_failed"
    if "one-shot" in m:
        return "one_shot_failed"
    return "processing_failed"


def save_failed_raw_response(file_name: str, api_name: str, stage: str, attempt: int, raw_response: str) -> str:
    stem = safe_filename_stem(file_name)
    api_safe = re.sub(r'[\/:*?"<>|]+', '_', api_name)
    stage_safe = re.sub(r'[\/:*?"<>|]+', '_', stage)
    path = os.path.join(FAILED_RAW_RESP_DIR, f"{stem}.{api_safe}.{stage_safe}.attempt{attempt}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(raw_response or "")
    return path


def append_failed_case(file_name: str, api_name: str, stage: str, error_message: str, retry_count: int = 0, raw_response_file: str = "") -> None:
    item = {
        "file": file_name,
        "source_file": file_name,
        "api_name": api_name,
        "stage": stage,
        "error_type": classify_error_type(error_message),
        "error_message": error_message,
        "retry_count": retry_count,
        "raw_response_file": raw_response_file,
        "updated_at": now_iso(),
    }
    append_jsonl_line(FAILED_CASES_JSONL, item)


def is_effective_output_file(path: str) -> bool:
    data = load_json_file(path)
    if not isinstance(data, dict):
        return False

    if "result" in data and isinstance(data["result"], dict):
        data = data["result"]

    return isinstance(data, dict) and (
            "entities" in data or "relations" in data or "validation_summary" in data
    )


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_seconds(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def build_initial_run_status(total_files: int, pending_files: List[str], skipped_files: List[str], skipped_failed_files: Optional[List[str]] = None) -> Dict[str, Any]:
    skipped_failed_files = skipped_failed_files or []
    files_payload = {
        os.path.basename(fp): {
            "status": "skipped_existing",
            "updated_at": now_iso(),
            "message": "已存在有效输出，断点续传自动跳过"
        }
        for fp in skipped_files
    }
    for fp in skipped_failed_files:
        files_payload[os.path.basename(fp)] = {
            "status": "skipped_failed",
            "updated_at": now_iso(),
            "message": "历史失败样本，主流程跳过，等待后续专项修复"
        }

    return {
        "updated_at": now_iso(),
        "resume_mode": RESUME_MODE,
        "overwrite_existing": OVERWRITE_EXISTING,
        "skip_previous_failed_in_main": SKIP_PREVIOUS_FAILED_IN_MAIN,
        "api_names": API_NAMES,
        "max_retries": MAX_RETRIES,
        "max_concurrency": MAX_CONCURRENCY,
        "total_files": total_files,
        "pending_files": len(pending_files),
        "skipped_existing_files": len(skipped_files),
        "skipped_failed_files": len(skipped_failed_files),
        "summary": {
            "done": len(skipped_files) + len(skipped_failed_files),
            "success": 0,
            "warning": 0,
            "failed": 0,
            "skipped_existing": len(skipped_files),
            "skipped_failed": len(skipped_failed_files),
        },
        "files": files_payload
    }


def update_run_status(
        run_status: Dict[str, Any],
        file_name: str,
        api_name: str,
        status: str,
        message: str
) -> Dict[str, Any]:
    run_status["updated_at"] = now_iso()
    files = run_status.setdefault("files", {})
    files[file_name] = {
        "status": status,
        "api_used": api_name,
        "updated_at": now_iso(),
        "message": message
    }

    summary = run_status.setdefault("summary", {})
    summary["success"] = sum(1 for x in files.values() if x.get("status") == "success")
    summary["warning"] = sum(1 for x in files.values() if x.get("status") == "warning")
    summary["failed"] = sum(1 for x in files.values() if x.get("status") == "failed")
    summary["skipped_existing"] = sum(1 for x in files.values() if x.get("status") == "skipped_existing")
    summary["skipped_failed"] = sum(1 for x in files.values() if x.get("status") == "skipped_failed")
    summary["done"] = (
            summary["success"]
            + summary["warning"]
            + summary["failed"]
            + summary["skipped_existing"]
            + summary["skipped_failed"]
    )
    return run_status


def log_progress(
        *,
        done: int,
        total: int,
        success: int,
        warning: int,
        failed: int,
        skipped_existing: int,
        skipped_failed: int,
        start_time: float
) -> None:
    elapsed = max(time.time() - start_time, 1e-9)
    pct = (done / total * 100.0) if total else 100.0
    remaining = max(total - done, 0)
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = format_seconds(remaining / rate) if rate > 0 else "--:--"

    logging.info(
        " 总进度: %s/%s (%.1f%%) | success=%s | warning=%s | failed=%s | skipped_existing=%s | skipped_failed=%s | elapsed=%s | ETA=%s",
        done, total, pct, success, warning, failed, skipped_existing, skipped_failed,
        format_seconds(elapsed), eta
    )


# ----------------------------------------------------
# 2. 五层映射与类型归一
# ----------------------------------------------------
BASE_ONTOLOGY_MAPPING = {
    "Material": "L1_Intervention",
    "Additive_or_Dopant": "L1_Intervention",
    "Interface_Layer": "L1_Intervention",
    "Process_Condition": "L1_Intervention",
    "Structural_Morphology_Change": "L2_Structure_Composition",
    "Chemical_Interaction": "L2_Structure_Composition",
    "Defect_State": "L3_Mechanism_Physics",
    "Carrier_Dynamics": "L3_Mechanism_Physics",
    "Recombination": "L3_Mechanism_Physics",
    "Device_Parameter": "L4_Device_Parameter",
    "Performance_Metric": "L5_Performance",
}

VALID_META_LAYERS = {
    "L1_Intervention",
    "L2_Structure_Composition",
    "L3_Mechanism_Physics",
    "L4_Device_Parameter",
    "L5_Performance",
}

ENTITY_TYPE_ALIASES = {
    "Additive_Or_Dopant": "Additive_or_Dopant",
    "InterfaceLayer": "Interface_Layer",
    "ProcessCondition": "Process_Condition",
    "StructuralMorphologyChange": "Structural_Morphology_Change",
    "ChemicalInteraction": "Chemical_Interaction",
    "DefectState": "Defect_State",
    "CarrierDynamics": "Carrier_Dynamics",
    "DeviceParameter": "Device_Parameter",
    "PerformanceMetric": "Performance_Metric",
}

ALLOWED_PREDICATE_GROUPS = {
    "structural_formation",
    "causal_change",
    "positive_effect",
    "negative_effect",
    "passivation_effect",
    "transport_effect",
    "stability_effect",
    "correlation",
    "process_enabling",
}

# ----------------------------------------------------
# 3. 提示词
# ----------------------------------------------------
ONE_SHOT_REFINEMENT_PROMPT = """You are a meticulous photovoltaic knowledge graph refinement expert working WITHIN ONE PAPER ONLY.

Paper Title:
{title}

Target Problem:
{problem}

Material System:
{material_system}

Research Focus:
{research_focus}

Task:
You will refine the ENTIRE paper graph in ONE SHOT.
Use the full paper-level context to decide entity consolidation globally.

Primary objective:
Produce the MOST CONSOLIDATED paper graph that is still scientifically safe.
Do NOT behave like a conservative deduplicator that leaves many near-duplicate entities untouched.
Within one paper, wording variation alone is NOT a valid reason to keep multiple entities separate.

However, consolidation must remain conceptually precise:
you must merge true alternative phrasings of the SAME concept,
but you must NOT collapse different levels of a causal chain into one entity.

Input entities already include local relation contexts, numeric signatures, concept signatures, metric/state/condition hints, and relation/evidence counts.
You must use these hints actively for global consolidation.

Hard constraints:
1. Never invent new entity IDs. Use only the existing input IDs.
2. Never invent entities that are not present in the input.
3. Do not merge entities with clearly conflicting final numerical values.
4. Do not merge across clearly different result states or clearly different experimental conditions.
5. Do not merge merely related concepts.
6. Each input ID may appear in at most one merge cluster.
7. Do not merge entities from different conceptual levels unless they are truly interchangeable phrasings of the exact same paper-specific concept.
8. If two entities cannot replace each other in the paper without changing the scientific meaning, keep them separate.

Required merge policy:
A. If two or more L4/L5 entities refer to the SAME metric/outcome in this paper and have the same or equivalent final value / delta value / numeric signature, they SHOULD normally be merged.
B. If one expression is a trend/result phrasing and another is the same final scientific outcome, they SHOULD be merged when no state/condition conflict exists.
C. If several intervention entities are clearly different phrasings of the SAME paper-specific stack / layer sequence / back contact / treatment, merge them into one canonical intervention concept.
D. Do not preserve multiple entities only because one is shorter, one is longer, one is narrative, one is symbolic, or one contains extra wording like "improved", "boosted", "rises by", "increase in", etc.
E. If there are obvious near-duplicate entities and merge_plan is empty, that is considered a bad answer.
F. Do NOT use singleton_updates as a way to rename several near-duplicate entities while still leaving them separate. If they are the same concept, merge them.

High-priority duplicate patterns that usually SHOULD be merged when no hard conflict exists:
- same metric + same numeric signature + same state + same condition
- same metric + same delta value expressed in different wording
- same intervention stack/process/contact expressed in long vs short forms
- same paper-specific result expressed as "increase in X", "X rises by ...", "improved X", "ΔX = ...", or "boosted X"

High-priority patterns that usually should stay separate:
- 16.4% efficiency vs 16.5% efficiency
- VOC gain vs final VOC absolute value
- JSC gain vs final JSC absolute value
- champion result vs stabilized result vs retention result
- intervention entity vs mechanism entity vs performance entity, unless they are truly the same concept
- structure formation vs optical/mechanistic effect vs performance outcome
- generic "performance improvement" phrasing vs a concrete efficiency result with a specific value


Critical anti-overmerge rule:
Do NOT merge two entities merely because they are causally linked, appear in the same reasoning chain, or frequently co-occur in the same evidence sentence.

Keep conceptually distinct levels separate, even if one leads to the other:
- intervention / structure / mechanism / device parameter / performance outcome should remain separate unless they are clearly alternative phrasings of the SAME concept
- cause and effect are not the same entity
- mechanism and downstream result are not the same entity
- process/structure and performance gain are not the same entity
- generic performance wording and a specific quantified metric are not automatically the same entity

Very important anti-overmerge examples:
- enhanced light trapping ≠ higher effective diffusion length
- enhanced light trapping ≠ higher IQE in the infrared
- higher IQE in the infrared ≠ higher effective diffusion length
- formation of a dielectric mirror ≠ higher infrared sensitivity
- improved back side reflection ≠ higher infrared response
- improved passivation ≠ open-circuit voltage gain
- open-circuit voltage gain ≠ efficiency gain
- short-circuit current density gain ≠ efficiency gain
- improved back contact performance ≠ cell efficiency
- improved back contact performance ≠ absolute power conversion efficiency gain
- reduced surface recombination velocity ≠ passivation
- passivation improvement ≠ reflection enhancement

Interchangeability test (must pass before merging):
For every proposed merge cluster, check all items against all others:
1. Are they describing the same scientific concept, not just adjacent steps in a causal chain?
2. Could one mention replace the other in the paper without changing the scientific meaning?
3. Are they at the same conceptual level?
4. If numeric, do they refer to the same metric and same final/delta value?
5. Is the merged canonical name something a scientist would naturally use as ONE concept, rather than a slash-joined mixture of cause and effect?

If any answer is NO, keep them separate.

Canonical naming rule:
The canonical name should represent ONE concept only.
Do NOT produce a canonical name that artificially joins different concepts with "/" or "and" just to compress the graph.
Bad style examples:
- "enhanced light trapping / higher IQE in infrared"
- "higher infrared sensitivity / dielectric mirror formation"
These indicate overmerge and should usually be avoided.

Few-shot examples (follow these patterns):

Example 1: same VOC gain, different wording -> MERGE
Input-like entities:
- E1: increased open circuit voltage (ΔVOC = 15 mV)
- E2: open circuit voltage rises by ΔV_OC = 15mV
- E3: improved open circuit voltage V_OC by 15 mV
Good decision:
{{
  "merge_plan": [
    {{
      "survivor_entity_id": "E1",
      "merged_entity_ids": ["E1", "E2", "E3"],
      "canonical_name": "open-circuit voltage gain (ΔV_OC = 15 mV)",
      "entity_type": "Device_Parameter",
      "meta_layer": "L4_Device_Parameter",
      "aliases": ["increased open circuit voltage", "open circuit voltage rises by ΔV_OC = 15mV"],
      "reason": "same metric and same delta value in different wording"
    }}
  ],
  "singleton_updates": [],
  "notes": []
}}

Example 2: same JSC gain, trend vs result phrasing -> MERGE
Input-like entities:
- E4: increased short circuit current density (ΔJ_SC = 0.8 mA cm^-2)
- E5: short circuit current increases by ΔJ_SC = 0.8 mA cm^-2
Good decision: merge E4 and E5 into one canonical JSC-gain entity.

Example 3: same efficiency gain, different wording -> MERGE
Input-like entities:
- E6: absolute efficiency gain of 0.4%
- E7: boosted absolute power conversion efficiency (Δη = 0.4%)
- E8: power conversion efficiency improves by Δη = 0.4%
Good decision: merge E6/E7/E8 into one canonical efficiency-gain entity because they describe the same paper-specific Δη result.

Example 4: same paper-specific intervention, long vs short descriptions -> MERGE
Input-like entities:
- E9: novel back contact
- E10: novel low-temperature back contact sequence
- E11: low-temperature a-Si:H-based back contact layer sequence
- E12: novel low-temperature back side layer sequence based upon plasma-deposited hydrogenated amorphous silicon (a-Si:H)
Good decision: merge them into one canonical intervention entity if they refer to the same back-contact stack in this paper.

Example 5: numerically different outcomes -> KEEP SEPARATE
Input-like entities:
- E13: power conversion efficiency up to 16.5%
- E14: increased efficiency (16.4%)
Good decision: keep separate because the final numerical outcomes differ.

Example 6: causal chain neighbors -> KEEP SEPARATE
Input-like entities:
- E15: formation of a dielectric mirror
- E16: higher sensitivity in the infrared spectral range
- E17: enhanced light trapping
Good decision: keep separate because these are causally linked but not interchangeable phrasings of one concept.

Example 7: mechanism vs downstream result -> KEEP SEPARATE
Input-like entities:
- E18: enhanced light trapping
- E19: higher IQE in the infrared
- E20: higher effective diffusion length
Good decision: keep separate unless the paper explicitly uses them as exact alternate names for the same concept, which is rare.

Example 8: generic performance wording vs quantified result -> KEEP SEPARATE
Input-like entities:
- E21: improved back contact performance
- E22: cell efficiency (η)
- E23: absolute power conversion efficiency gain (Δη = 0.4%)
Good decision: keep separate unless the text clearly shows that one is merely a direct alternate wording of the exact same quantified outcome.

Operational instructions:
1. First internally scan the full entity list for likely duplicate groups.
2. For each likely duplicate group, make an explicit merge-or-separate decision.
3. Before finalizing any merge cluster, apply the Interchangeability test.
4. Then output the final JSON.
5. Prefer fewer, stronger canonical entities when scientifically justified.
6. If a merge cluster is valid, include all equivalent IDs in that cluster in one step instead of partial pairwise merging.
7. If there is a tension between "more consolidation" and "scientific conceptual precision", choose conceptual precision.

Return ONLY valid JSON with this schema:
{{
  "merge_plan": [
    {{
      "survivor_entity_id": "existing input id",
      "merged_entity_ids": ["existing id 1", "existing id 2"],
      "canonical_name": "best canonical concept name",
      "entity_type": "normalized entity type",
      "meta_layer": "one of L1_Intervention / L2_Structure_Composition / L3_Mechanism_Physics / L4_Device_Parameter / L5_Performance",
      "aliases": ["optional alias 1", "optional alias 2"],
      "reason": "short reason"
    }}
  ],
  "singleton_updates": [
    {{
      "entity_id": "existing input id",
      "entity_name": "updated or canonicalized name",
      "entity_type": "normalized entity type",
      "meta_layer": "one of the 5 labels",
      "aliases": ["optional alias 1", "optional alias 2"],
      "reason": "short reason"
    }}
  ],
  "notes": ["optional note"]
}}

Compactness rules:
- Omit unchanged singleton entities from singleton_updates.
- Put only genuine merge clusters into merge_plan.
- But do NOT omit obvious duplicate clusters; explicitly merge them.
- Do NOT create fake compactness by merging neighboring concepts in a causal chain.

Input entity payload:
{entities_json}
"""

LOCAL_ONTOLOGY_PROMPT = """You are a photovoltaic knowledge engineer.

Paper Title:
{title}

Task:
Map each NEW entity type below into exactly one of:
- L1_Intervention
- L2_Structure_Composition
- L3_Mechanism_Physics
- L4_Device_Parameter
- L5_Performance

Rules:
1. Return ONLY a JSON object.
2. Keys must be the original new types.
3. Values must be one of the 5 labels above.
4. Do not invent extra keys.

NEW TYPES:
{new_types}
"""

LOCAL_ALIGNMENT_PROMPT = """You are a meticulous photovoltaic scientist doing entity alignment WITHIN ONE PAPER ONLY.

Paper Title:
{title}

Target Problem:
{problem}

Entity Type:
{entity_type}

Task:
Group only the raw entities that are definitely the same concept in THIS paper.
Do not merge merely related concepts.
Do not merge measurement results with interventions.
Do not merge material/process names with device parameters unless they are clearly the same concept.
Do not merge if there is any ambiguity.
Do not merge entities that represent different final numerical values, different result states, or clearly different experimental conditions.

Return ONLY JSON:
{{
  "Canonical Name A": ["raw a1", "raw a2"],
  "Canonical Name B": ["raw b1", "raw b2"]
}}

Candidate entities with local context:
{entities_json}
"""

LOCAL_CANONICAL_MERGE_PROMPT = """You are a photovoltaic knowledge consolidation expert.

Task:
Given a group of candidate entities from ONE paper that are already roughly about the same concept,
produce a semantically improved canonical merge decision.

Your goal is NOT simple deduplication.
Your goal is to rewrite/fuse paper-specific mentions into the best canonical scientific concept.

Important rules:
1. You may merge generic and specific mentions if they describe the SAME final scientific concept.
2. If two L4/L5 entities contain clearly conflicting final numerical values, DO NOT merge them.
3. If one mention is a trend expression like "from A to B" and another is the final value "B",
   they MAY be merged into one final-result concept.
4. Preserve paper-specific meaning. Do not over-generalize away important scientific content.
5. Do not merge across clearly different result states (e.g. champion PCE vs stabilized PCE vs long-term retention).
6. Return only valid JSON.

Return format:
{{
  "clusters": [
    {{
      "canonical_name": "best canonical concept name",
      "member_ids": ["PE_1", "PE_7"],
      "reason": "short reason"
    }}
  ],
  "keep_separate": [
    {{
      "entity_ids": ["PE_8", "PE_9"],
      "reason": "short reason"
    }}
  ]
}}

Candidate entities:
{entities_json}
"""


# ----------------------------------------------------
# 4. 通用工具
# ----------------------------------------------------
def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_key(text: Any) -> str:
    return normalize_text(text).lower()


def normalize_alias_list(values: Any, entity_name: str = "") -> List[str]:
    if not isinstance(values, list):
        values = []

    aliases = []
    seen = set()
    name_key = normalize_key(entity_name) if entity_name else ""

    for v in values:
        a = normalize_text(v)
        if not a:
            continue
        k = normalize_key(a)
        if not k or k == name_key or k in seen:
            continue
        seen.add(k)
        aliases.append(a)

    return aliases


def normalize_entity_record(ent: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(ent, dict):
        return None

    entity_id = ent.get("entity_id")
    entity_name = normalize_text(ent.get("entity_name"))
    entity_type = normalize_entity_type(ent.get("entity_type", "Material"))

    if not entity_id or not entity_name:
        return None

    raw_meta = normalize_text(ent.get("meta_layer"))
    if raw_meta in VALID_META_LAYERS:
        meta_layer = raw_meta
    else:
        meta_layer = infer_meta_layer_from_type_and_name(entity_type, entity_name) or "Unknown"

    new_ent = dict(ent)
    new_ent["entity_name"] = entity_name
    new_ent["entity_type"] = entity_type
    new_ent["meta_layer"] = meta_layer
    new_ent["aliases"] = normalize_alias_list(ent.get("aliases", []), entity_name=entity_name)
    new_ent["merged_entity_ids"] = normalize_alias_list(ent.get("merged_entity_ids", []))
    return new_ent


def normalize_relation_record(rel: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(rel, dict):
        return None

    sub = rel.get("subject_id")
    obj = rel.get("object_id")
    raw_pred = normalize_text(rel.get("raw_predicate") or rel.get("predicate"))
    pred_group = normalize_predicate_group(rel.get("predicate_group"))

    if not sub or not obj or not raw_pred:
        return None

    evidences = rel.get("evidences", rel.get("evidence", []))
    if isinstance(evidences, str):
        evidences = [evidences]
    clean_evidences = []
    seen = set()
    for x in evidences:
        ev = normalize_text(x)
        if not ev:
            continue
        k = normalize_key(ev)
        if k in seen:
            continue
        seen.add(k)
        clean_evidences.append(ev)

    return {
        "subject_id": sub,
        "raw_predicate": raw_pred,
        "predicate_group": pred_group,
        "object_id": obj,
        "evidences": clean_evidences
    }


def normalize_entity_type(entity_type: Any) -> str:
    if not isinstance(entity_type, str):
        return "Material"

    et = normalize_text(entity_type)
    if not et:
        return "Material"

    et = et.replace("-", " ").replace("_", " ")
    et = "_".join([w.capitalize() for w in et.split()])

    if et in ENTITY_TYPE_ALIASES:
        return ENTITY_TYPE_ALIASES[et]

    return et


def normalize_predicate_group(v: Any) -> str:
    if not isinstance(v, str):
        return "causal_change"
    v = normalize_text(v).lower().replace(" ", "_").replace("-", "_")
    if v in ALLOWED_PREDICATE_GROUPS:
        return v

    if any(k in v for k in ["form", "construct", "recrystall", "morpholog"]):
        return "structural_formation"
    if any(k in v for k in ["passivat", "trap", "defect"]):
        return "passivation_effect"
    if any(k in v for k in ["transport", "extract", "conduct", "mobility"]):
        return "transport_effect"
    if any(k in v for k in ["stability", "retain", "degradation", "lifetime"]):
        return "stability_effect"
    if any(k in v for k in ["reduce", "decrease", "suppress", "lower", "loss"]):
        return "negative_effect"
    if any(k in v for k in ["increase", "enhance", "improve", "boost", "promote"]):
        return "positive_effect"
    if any(k in v for k in ["enable", "allow", "facilitate"]):
        return "process_enabling"
    if any(k in v for k in ["associate", "correlat", "link", "reflect", "results"]):
        return "correlation"

    return "causal_change"


def extract_json_block(raw_text: str) -> str:
    if not raw_text:
        raise ValueError("empty response")

    text = raw_text.strip()

    if "```json" in text:
        return text.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0].strip()

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return match.group(0).strip()

    raise ValueError(f"no json object found: {raw_text[:200]}")


async def ask_llm_json(model_api, prompt: str, temperature: float = 0.05, debug_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    messages = [{"role": "user", "content": prompt}]
    last_err = None
    last_raw_response = ""

    for attempt in range(MAX_RETRIES):
        try:
            response = await asyncio.wait_for(
                model_api.call(messages=messages, temperature=temperature),
                timeout=API_TIMEOUT_SECONDS
            )
            last_raw_response = response or ""
            clean_resp = extract_json_block(response)
            parsed = json.loads(clean_resp)
            if not isinstance(parsed, dict):
                raise ValueError("response JSON is not an object")
            return parsed
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)

    raw_response_file = ""
    if last_raw_response and debug_context:
        raw_response_file = save_failed_raw_response(
            file_name=str(debug_context.get("file_name") or "unknown"),
            api_name=str(debug_context.get("api_name") or "unknown_api"),
            stage=str(debug_context.get("stage") or "llm_json"),
            attempt=MAX_RETRIES,
            raw_response=last_raw_response,
        )
    msg = f"LLM JSON parsing failed after retries: {last_err}"
    if raw_response_file:
        msg += f" | raw_response_file={raw_response_file}"
    raise RuntimeError(msg)


def safe_filename_stem(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    stem = re.sub(r'[\\/:*?"<>|]+', "_", stem)
    return stem


# ----------------------------------------------------
# 5. 五层推断
# ----------------------------------------------------
def infer_meta_layer_from_type_and_name(entity_type: str, entity_name: str) -> Optional[str]:
    if entity_type in BASE_ONTOLOGY_MAPPING:
        return BASE_ONTOLOGY_MAPPING[entity_type]

    et = normalize_key(entity_type)
    en = normalize_key(entity_name)

    if any(k in et for k in ["material", "additive", "dopant", "interface_layer", "process"]):
        return "L1_Intervention"
    if any(k in et for k in ["morphology", "structure", "chemical_interaction", "device_design"]):
        return "L2_Structure_Composition"
    if any(k in et for k in ["defect", "carrier", "recombination", "device_function"]):
        return "L3_Mechanism_Physics"
    if any(k in et for k in ["parameter"]):
        return "L4_Device_Parameter"
    if any(k in et for k in ["performance", "outcome"]):
        return "L5_Performance"

    if any(k in en for k in [
        "pce", "efficiency", "stability", "retention", "voc", "jsc", "ff",
        "power conversion efficiency"
    ]):
        return "L5_Performance"

    if any(k in en for k in [
        "resistance", "vbi", "vtfl", "hysteresis", "contact angle", "sheet resistance",
        "fill factor", "open-circuit voltage", "short-circuit current density", "active area"
    ]):
        return "L4_Device_Parameter"

    if any(k in en for k in [
        "recombination", "carrier", "trap", "conductivity", "passivation",
        "charge extraction", "mobility", "lifetime", "light trapping"
    ]):
        return "L3_Mechanism_Physics"

    if any(k in en for k in [
        "formation of", "heterostructure", "morphology", "grain", "texture", "surface",
        "bonding", "void", "layer", "roughness", "hydrophobicity"
    ]):
        return "L2_Structure_Composition"

    if any(k in en for k in [
        "treatment", "post-treatment", "processing", "etching", "annealing",
        "spin-coating", "deposition", "aging", "transfer process", "solvent",
        "mechanical force", "separation of the film"
    ]):
        return "L1_Intervention"

    return None


# ----------------------------------------------------
# 6. 数值签名与概念签名
# ----------------------------------------------------
def concept_signature(name: str, preserve_state_terms: bool = False) -> str:
    name = normalize_key(name)

    replacements = {
        "short circuit": "short-circuit",
        "open circuit": "open-circuit",
        "reflectance": "reflection",
        "power conversion efficiency": "pce",
        "open-circuit voltage": "voc",
        "short-circuit current density": "jsc",
        "fill factor": "ff",
        "η": "efficiency",
    }
    for src, tgt in replacements.items():
        name = name.replace(src, tgt)

    name = re.sub(r"\([^)]*\)", "", name)

    if not preserve_state_terms:
        name = re.sub(
            r"\b(increase|increased|decrease|decreased|improved|improvement|reduced|reduction|"
            r"suppressed|suppression|enhanced|enhancement|loss|gain|retained|retention|stabilized|"
            r"champion|optimized|maintained|negligible)\b",
            "",
            name
        )

    name = re.sub(r"[^a-z0-9\-\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def preserve_direction_for_meta_layer(meta_layer: Any) -> bool:
    return normalize_text(meta_layer) in {
        "L2_Structure_Composition",
        "L3_Mechanism_Physics",
    }


def entity_concept_signature(entity: Dict[str, Any]) -> str:
    return concept_signature(
        entity.get("entity_name", ""),
        preserve_state_terms=preserve_direction_for_meta_layer(entity.get("meta_layer"))
    )


def normalize_metric_unit(value: float, unit: str) -> Tuple[float, str]:
    unit = normalize_text(unit).lower().replace(" ", "")
    if unit == "mv":
        return value / 1000.0, "v"
    if unit in {"v", "%", "ω", "ohm"}:
        if unit == "ω":
            unit = "ohm"
        return value, unit
    if "ma/cm" in unit or "macm" in unit:
        return value, "ma/cm2"
    return value, unit


def extract_final_numeric_signature(text: str) -> Tuple[str, ...]:
    text = normalize_text(text).lower().replace("−", "-")
    if not text:
        return tuple()

    unit_pat = r"(mv|v|%|ω|ohm|ma\s*/?\s*cm(?:\^-?2|−2|-2)?)"

    final_matches = re.findall(
        rf"\bto\s+([-+]?\d+(?:\.\d+)?)\s*(?:±\s*[-+]?\d+(?:\.\d+)?)?\s*{unit_pat}",
        text,
        flags=re.IGNORECASE
    )
    if final_matches:
        out = []
        for val_str, unit in final_matches:
            try:
                val = float(val_str)
                val, unit = normalize_metric_unit(val, unit)
                out.append(f"{val:.6g}|{unit}")
            except Exception:
                pass
        return tuple(sorted(dict.fromkeys(out)))

    value_matches = re.findall(
        rf"\b([-+]?\d+(?:\.\d+)?)\s*(?:±\s*[-+]?\d+(?:\.\d+)?)?\s*{unit_pat}",
        text,
        flags=re.IGNORECASE
    )
    if value_matches:
        out = []
        for val_str, unit in value_matches:
            try:
                val = float(val_str)
                val, unit = normalize_metric_unit(val, unit)
                out.append(f"{val:.6g}|{unit}")
            except Exception:
                pass
        return tuple(sorted(dict.fromkeys(out)))

    nums = re.findall(r"\b[-+]?\d+(?:\.\d+)?\b", text)
    return tuple(sorted(dict.fromkeys(nums)))


def entity_numeric_signature(entity: Dict[str, Any]) -> Tuple[str, ...]:
    texts = [normalize_text(entity.get("entity_name"))]
    aliases = entity.get("aliases", [])
    if isinstance(aliases, list):
        texts.extend([normalize_text(a) for a in aliases if normalize_text(a)])

    all_sigs = []
    for t in texts:
        all_sigs.extend(list(extract_final_numeric_signature(t)))

    return tuple(sorted(dict.fromkeys(all_sigs)))


def metric_signature(entity: Dict[str, Any]) -> str:
    text_parts = [normalize_text(entity.get("entity_name"))]
    aliases = entity.get("aliases", [])
    if isinstance(aliases, list):
        text_parts.extend([normalize_text(a) for a in aliases if normalize_text(a)])

    text = " | ".join(text_parts).lower()

    def has_pat(pattern: str) -> bool:
        return re.search(pattern, text, flags=re.IGNORECASE) is not None

    if (
            has_pat(r"\bpower\s+conversion\s+efficiency\b")
            or has_pat(r"\bpce\b")
            or has_pat(r"\bcell\s+efficiency\b")
            or has_pat(r"\bsolar\s+cell\s+efficiency\b")
            or "η" in text
    ):
        return "pce"

    if has_pat(r"\b(storage|dark\s+storage)\s+stability\b") or has_pat(r"\bretention\b"):
        return "stability_storage"
    if has_pat(r"\bthermal\s+stability\b") or has_pat(r"\bheating\b") or has_pat(r"\b100\s*°?c\b"):
        return "stability_thermal"
    if has_pat(r"\boperational\s+stability\b") or has_pat(r"\bmpp\s+tracking\b"):
        return "stability_operational"
    if has_pat(r"\blifetime\b"):
        return "lifetime"
    if has_pat(r"\bopen[\-\s]?circuit\s+voltage\b") or has_pat(r"\bv[_ ]?oc\b") or has_pat(r"\bvoc\b"):
        return "voc"
    if (
            has_pat(r"\bshort[\-\s]?circuit\s+current\s+density\b")
            or has_pat(r"\bshort[\-\s]?circuit\s+current\b")
            or has_pat(r"\bj[_ ]?sc\b")
            or has_pat(r"\bjsc\b")
            or has_pat(r"\bisc\b")
    ):
        return "jsc"
    if has_pat(r"\bfill\s+factor\b") or has_pat(r"\bff\b"):
        return "ff"
    if has_pat(r"\bsheet\s+resistance\b") or has_pat(r"\brsq\b"):
        return "rsq"
    if has_pat(r"\bseries\s+resistance\b") or has_pat(r"\br[_ ]?s\b") or has_pat(r"\brs\b"):
        return "rs"
    if has_pat(r"\bshunt\s+resistance\b") or has_pat(r"\brsh\b"):
        return "rsh"
    if (
            has_pat(r"\breflectance\b")
            or has_pat(r"\breflection\b")
            or has_pat(r"\breff\b")
            or has_pat(r"\bwar\b")
            or has_pat(r"\bweighted\s+average\s+reflectance\b")
    ):
        return "reflectance"
    if has_pat(r"\bhysteresis\s+index\b") or has_pat(r"\bhi\b"):
        return "hi"
    if has_pat(r"\bhysteresis\b"):
        return "hysteresis"
    if has_pat(r"\bconductivity\b"):
        return "conductivity"
    if has_pat(r"\bbuilt[\-\s]?in\s+potential\b") or has_pat(r"\bvbi\b"):
        return "vbi"
    if has_pat(r"\bemitter\s+saturation\s+current\s+density\b") or has_pat(r"\bj[_ ]?0e\b") or has_pat(r"\bj0e\b"):
        return "j0e"
    if has_pat(r"\bspectral\s+response\b") or has_pat(r"\bblue\s+response\b") or has_pat(r"\biqe\b"):
        return "spectral_response"

    return concept_signature(entity.get("entity_name", "")) or "unknown_metric"


def state_signature(entity: Dict[str, Any]) -> str:
    text_parts = [normalize_text(entity.get("entity_name"))]
    aliases = entity.get("aliases", [])
    if isinstance(aliases, list):
        text_parts.extend([normalize_text(a) for a in aliases if normalize_text(a)])

    text = " | ".join(text_parts).lower()

    if any(k in text for k in ["retained", "retention", "maintained"]):
        return "retention"
    if "stabilized" in text:
        return "stabilized"
    if "champion" in text:
        return "champion"
    if any(k in text for k in [
        "absolute gain", "absolute efficiency gain", "absolute increase", "Δη", "delta eta"
    ]):
        return "absolute_gain"
    if any(k in text for k in [
        "loss", "decreased", "reduced", "drop"
    ]):
        return "loss_or_decrease"
    if any(k in text for k in [
        "increase", "increased", "improved", "boosted", "enhanced"
    ]):
        return "gain_or_increase"
    if entity_numeric_signature(entity):
        return "absolute_value"

    return "generic"


def condition_signature(entity: Dict[str, Any]) -> str:
    text_parts = [normalize_text(entity.get("entity_name"))]
    aliases = entity.get("aliases", [])
    if isinstance(aliases, list):
        text_parts.extend([normalize_text(a) for a in aliases if normalize_text(a)])

    text = " | ".join(text_parts).lower().replace("−", "-")

    tags = []
    for g in re.findall(r"\bgroup\s*\d+\b", text):
        tags.append(g)
    for m in re.findall(r"t[_ ]?[a-z]*\s*=\s*\d+(?:\.\d+)?\s*s", text):
        tags.append(m)
    for m in re.findall(r"\b\d+(?:\.\d+)?\s*(?:ω/sq|ohm/sq|Ω/sq)\b", text):
        tags.append(m)
    for m in re.findall(r"\b\d+(?:\.\d+)?\s*(?:h|hr|hrs|hour|hours)\b", text):
        tags.append(m)
    for m in re.findall(r"\b\d+(?:\.\d+)?\s*(?:°c|c)\b", text):
        tags.append(m)

    keyword_map = {
        "as-diffused": "as_diffused",
        "stabilized": "stabilized_condition",
        "champion": "champion_condition",
        "dark storage": "dark_storage",
        "air": "air_condition",
        "heating": "heating_condition",
        "mpp": "mpp_tracking",
        "seris etch": "seris_etch",
        "rie": "rie",
        "koh": "koh",
        "mae": "mae",
        "nfa": "nfa",
        "tfe-peai": "tfe_peai",
        "group 4": "group_4",
    }

    for k, tag in keyword_map.items():
        if k in text:
            tags.append(tag)

    tags = sorted(dict.fromkeys(tags))
    return "|".join(tags) if tags else "no_condition"


def is_l4_l5_entity(entity: Dict[str, Any]) -> bool:
    return normalize_text(entity.get("meta_layer")) in {
        "L4_Device_Parameter",
        "L5_Performance"
    }


def is_generic_l4_l5(entity: Dict[str, Any]) -> bool:
    if not is_l4_l5_entity(entity):
        return False

    sig = entity_numeric_signature(entity)
    state = state_signature(entity)

    if not sig and state == "generic":
        return True
    return False


def parse_numeric_signature_item(sig: str) -> Tuple[Optional[float], str]:
    try:
        if "|" not in sig:
            return None, ""
        val_str, unit = sig.split("|", 1)
        return float(val_str), unit
    except Exception:
        return None, ""


def numeric_signatures_are_close(sig_a: Tuple[str, ...], sig_b: Tuple[str, ...]) -> bool:
    if not sig_a or not sig_b:
        return False
    if len(sig_a) != 1 or len(sig_b) != 1:
        return False

    va, ua = parse_numeric_signature_item(sig_a[0])
    vb, ub = parse_numeric_signature_item(sig_b[0])

    if va is None or vb is None or not ua or not ub:
        return False
    if ua != ub:
        return False

    diff = abs(va - vb)

    if ua == "%":
        return diff <= 1.0
    if ua == "v":
        return diff <= 0.02
    if ua == "ma/cm2":
        return diff <= 0.3

    return False


def hard_merge_conflict(ent_a: Dict[str, Any], ent_b: Dict[str, Any]) -> Tuple[bool, str]:
    layer_a = normalize_text(ent_a.get("meta_layer"))
    layer_b = normalize_text(ent_b.get("meta_layer"))

    if layer_a in {"L4_Device_Parameter", "L5_Performance"} or layer_b in {"L4_Device_Parameter", "L5_Performance"}:
        sig_a = entity_numeric_signature(ent_a)
        sig_b = entity_numeric_signature(ent_b)

        if sig_a and sig_b and sig_a != sig_b:
            if numeric_signatures_are_close(sig_a, sig_b):
                return False, ""
            return True, "different_final_numeric_values"

    return False, ""


def build_collapse_key(std_name: str, e_type: str, meta_layer: str, raw_name: str = "") -> Tuple:
    base_sig = concept_signature(
        std_name,
        preserve_state_terms=preserve_direction_for_meta_layer(meta_layer)
    ) or normalize_key(std_name)

    if meta_layer in {"L4_Device_Parameter", "L5_Performance"}:
        pseudo_entity = {
            "entity_name": std_name,
            "aliases": [raw_name] if raw_name else []
        }
        num_sig = entity_numeric_signature(pseudo_entity)
        return (base_sig, e_type, meta_layer, num_sig)

    return (base_sig, e_type, meta_layer)


# ----------------------------------------------------
# 7. 上下文构造
# ----------------------------------------------------
def build_entity_contexts(
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]]
) -> Dict[str, List[str]]:
    entity_map = {e.get("entity_id"): e for e in entities if e.get("entity_id")}
    contexts = defaultdict(list)

    for rel in relations:
        sub_id = rel.get("subject_id")
        obj_id = rel.get("object_id")
        raw_pred = normalize_text(rel.get("raw_predicate") or rel.get("predicate"))
        pred_group = normalize_text(rel.get("predicate_group"))
        evidences = rel.get("evidences", [])
        if isinstance(evidences, str):
            evidences = [evidences]

        sub = entity_map.get(sub_id)
        obj = entity_map.get(obj_id)
        if not sub or not obj:
            continue

        sub_name = normalize_text(sub.get("entity_name"))
        obj_name = normalize_text(obj.get("entity_name"))

        if raw_pred:
            ctx = f"OUT: {raw_pred} -> {obj_name}"
            if pred_group:
                ctx += f" [{pred_group}]"
            contexts[sub_id].append(ctx)

            ctx2 = f"IN: {sub_name} -> {raw_pred}"
            if pred_group:
                ctx2 += f" [{pred_group}]"
            contexts[obj_id].append(ctx2)

        for ev in evidences[:1]:
            ev = normalize_text(ev)
            if ev:
                ev = ev[:220]
                contexts[sub_id].append(f"EVIDENCE: {ev}")
                contexts[obj_id].append(f"EVIDENCE: {ev}")

    return contexts


def build_alignment_payload(
        typed_entities: List[Dict[str, Any]],
        entity_contexts: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    payload = []
    seen = set()

    for ent in typed_entities:
        raw_name = normalize_text(ent.get("entity_name"))
        ent_id = ent.get("entity_id")
        if not raw_name or raw_name in seen:
            continue
        seen.add(raw_name)

        ctxs = entity_contexts.get(ent_id, [])
        ctxs = list(dict.fromkeys(ctxs))[:4]

        payload.append({
            "name": raw_name,
            "contexts": ctxs
        })

    return payload


def safe_validate_alignment_output(
        grouped_json: Dict[str, Any],
        allowed_raw_names: List[str]
) -> Dict[str, List[str]]:
    allowed_norm_to_raw = {normalize_key(x): x for x in allowed_raw_names}
    safe_groups = {}

    for std_name, raw_list in grouped_json.items():
        std_name = normalize_text(std_name)
        if not std_name or not isinstance(raw_list, list):
            continue

        clean_raw_list = []
        for raw in raw_list:
            raw_norm = normalize_key(raw)
            if raw_norm in allowed_norm_to_raw:
                clean_raw_list.append(allowed_norm_to_raw[raw_norm])

        clean_raw_list = list(dict.fromkeys(clean_raw_list))
        if clean_raw_list:
            safe_groups[std_name] = clean_raw_list

    return safe_groups


# ----------------------------------------------------
# 7.5 One-shot 全文融合
# ----------------------------------------------------
def build_one_shot_payload(
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    entity_contexts = build_entity_contexts(entities, relations)
    payload = []

    for ent in entities:
        eid = ent.get("entity_id")
        payload.append({
            "entity_id": eid,
            "entity_name": ent.get("entity_name"),
            "entity_type": ent.get("entity_type"),
            "meta_layer": ent.get("meta_layer"),
            "aliases": ent.get("aliases", []),
            "numeric_signature": list(entity_numeric_signature(ent)),
            "concept_signature": entity_concept_signature(ent),
            "metric_signature": metric_signature(ent) if is_l4_l5_entity(ent) else "",
            "state_signature": state_signature(ent) if is_l4_l5_entity(ent) else "",
            "condition_signature": condition_signature(ent) if is_l4_l5_entity(ent) else "",
            "relation_count": relation_count_for_entity(eid, relations),
            "evidence_count": evidence_count_for_entity(eid, relations),
            "contexts": list(dict.fromkeys(entity_contexts.get(eid, [])))[:4],
        })

    return payload


def _collect_entity_aliases_for_cluster(
        cluster_members: List[Dict[str, Any]],
        canonical_name: str,
        extra_aliases: Any
) -> List[str]:
    alias_bucket = []
    for ent in cluster_members:
        alias_bucket.append(ent.get("entity_name"))
        for a in ent.get("aliases", []):
            alias_bucket.append(a)

    if isinstance(extra_aliases, list):
        alias_bucket.extend(extra_aliases)

    return normalize_alias_list(alias_bucket, entity_name=canonical_name)


def apply_one_shot_refinement_plan(
        llm_result: Dict[str, Any],
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    entity_by_id = {e["entity_id"]: dict(e) for e in entities if e.get("entity_id")}

    merge_plan = llm_result.get("merge_plan", [])
    singleton_updates = llm_result.get("singleton_updates", [])
    if not isinstance(merge_plan, list):
        merge_plan = []
    if not isinstance(singleton_updates, list):
        singleton_updates = []

    report = {
        "merge_plan_raw_count": len(merge_plan),
        "singleton_updates_raw_count": len(singleton_updates),
        "accepted_merges": [],
        "blocked_merges": [],
        "applied_singleton_updates": [],
        "notes": llm_result.get("notes", []) if isinstance(llm_result.get("notes", []), list) else []
    }

    singleton_update_map: Dict[str, Dict[str, Any]] = {}
    for item in singleton_updates:
        if not isinstance(item, dict):
            continue
        eid = normalize_text(item.get("entity_id"))
        if eid not in entity_by_id:
            continue
        singleton_update_map[eid] = item

    accepted_merge_specs = []
    used_ids = set()

    for item in merge_plan:
        if not isinstance(item, dict):
            continue

        survivor_id = normalize_text(item.get("survivor_entity_id") or item.get("entity_id"))
        merged_ids = item.get("merged_entity_ids", item.get("member_ids", []))
        if not isinstance(merged_ids, list):
            continue

        merged_ids = [normalize_text(x) for x in merged_ids if normalize_text(x) in entity_by_id]
        merged_ids = list(dict.fromkeys(merged_ids))

        if survivor_id not in entity_by_id or survivor_id not in merged_ids or len(merged_ids) <= 1:
            continue

        overlap = [x for x in merged_ids if x in used_ids]
        if overlap:
            report["blocked_merges"].append({
                "survivor_id": survivor_id,
                "merged_entity_ids": merged_ids,
                "reason": f"overlap_with_previous_clusters: {overlap}"
            })
            continue

        cluster_members = [entity_by_id[mid] for mid in merged_ids]
        hard_block_reason = ""
        for i in range(len(cluster_members)):
            for j in range(i + 1, len(cluster_members)):
                bad, why = hard_merge_conflict(cluster_members[i], cluster_members[j])
                if bad:
                    hard_block_reason = why
                    break
            if hard_block_reason:
                break

        if hard_block_reason:
            report["blocked_merges"].append({
                "survivor_id": survivor_id,
                "merged_entity_ids": merged_ids,
                "reason": hard_block_reason
            })
            continue

        survivor_orig = entity_by_id[survivor_id]
        canonical_name = normalize_text(item.get("canonical_name") or survivor_orig.get("entity_name"))
        if not canonical_name:
            canonical_name = normalize_text(survivor_orig.get("entity_name"))

        entity_type = normalize_entity_type(item.get("entity_type") or survivor_orig.get("entity_type", "Material"))
        meta_layer = normalize_text(item.get("meta_layer"))
        if meta_layer not in VALID_META_LAYERS:
            meta_layer = infer_meta_layer_from_type_and_name(entity_type, canonical_name) or survivor_orig.get(
                "meta_layer") or "Unknown"

        aliases = _collect_entity_aliases_for_cluster(cluster_members, canonical_name, item.get("aliases", []))

        accepted_merge_specs.append({
            "survivor_id": survivor_id,
            "merged_entity_ids": merged_ids,
            "canonical_name": canonical_name,
            "entity_type": entity_type,
            "meta_layer": meta_layer,
            "aliases": aliases,
            "reason": normalize_text(item.get("reason"))
        })
        used_ids.update(merged_ids)

        report["accepted_merges"].append({
            "survivor_id": survivor_id,
            "merged_entity_ids": merged_ids,
            "canonical_name": canonical_name,
            "reason": normalize_text(item.get("reason"))
        })

    for eid, item in singleton_update_map.items():
        ent = entity_by_id[eid]
        entity_name = normalize_text(item.get("entity_name") or ent.get("entity_name"))
        entity_type = normalize_entity_type(item.get("entity_type") or ent.get("entity_type", "Material"))
        meta_layer = normalize_text(item.get("meta_layer"))
        if meta_layer not in VALID_META_LAYERS:
            meta_layer = infer_meta_layer_from_type_and_name(entity_type, entity_name) or ent.get(
                "meta_layer") or "Unknown"

        extra_aliases = item.get("aliases", []) if isinstance(item.get("aliases", []), list) else []
        alias_list = normalize_alias_list(list(ent.get("aliases", [])) + extra_aliases + [ent.get("entity_name")],
                                          entity_name=entity_name)

        ent["entity_name"] = entity_name
        ent["entity_type"] = entity_type
        ent["meta_layer"] = meta_layer
        ent["aliases"] = alias_list
        ent["merged_entity_ids"] = sorted(dict.fromkeys([eid] + list(ent.get("merged_entity_ids", []))))

        report["applied_singleton_updates"].append({
            "entity_id": eid,
            "entity_name": entity_name,
            "reason": normalize_text(item.get("reason"))
        })

    id_mapping: Dict[str, str] = {}
    merged_victim_ids = set()
    for spec in accepted_merge_specs:
        survivor_id = spec["survivor_id"]
        for mid in spec["merged_entity_ids"]:
            id_mapping[mid] = survivor_id
            if mid != survivor_id:
                merged_victim_ids.add(mid)

    final_entities = []
    for eid, ent in entity_by_id.items():
        if eid in merged_victim_ids:
            continue

        new_ent = dict(ent)
        merge_spec = next((x for x in accepted_merge_specs if x["survivor_id"] == eid), None)
        if merge_spec is not None:
            cluster_members = [entity_by_id[mid] for mid in merge_spec["merged_entity_ids"]]
            merged_ids = []
            for m in cluster_members:
                mids = m.get("merged_entity_ids", [])
                if isinstance(mids, list):
                    merged_ids.extend([normalize_text(z) for z in mids if normalize_text(z)])
                merged_ids.append(m.get("entity_id"))
            merged_ids = sorted(dict.fromkeys([z for z in merged_ids if z]))

            new_ent["entity_name"] = merge_spec["canonical_name"]
            new_ent["entity_type"] = merge_spec["entity_type"]
            new_ent["meta_layer"] = merge_spec["meta_layer"]
            new_ent["aliases"] = merge_spec["aliases"]
            new_ent["merged_entity_ids"] = merged_ids
        else:
            new_ent["merged_entity_ids"] = sorted(dict.fromkeys([eid] + list(new_ent.get("merged_entity_ids", []))))
            new_ent["aliases"] = normalize_alias_list(new_ent.get("aliases", []),
                                                      entity_name=new_ent.get("entity_name", ""))

        final_entities.append(new_ent)

    final_entities = sorted(final_entities, key=lambda x: normalize_text(x.get("entity_id")))
    final_relations = rebuild_relations_with_id_mapping(relations, id_mapping)
    return final_entities, final_relations, report


# ----------------------------------------------------
# 8. 关系与实体辅助
# ----------------------------------------------------
def relation_count_for_entity(entity_id: str, relations: List[Dict[str, Any]]) -> int:
    cnt = 0
    for rel in relations:
        if rel.get("subject_id") == entity_id or rel.get("object_id") == entity_id:
            cnt += 1
    return cnt


def evidence_count_for_entity(entity_id: str, relations: List[Dict[str, Any]]) -> int:
    cnt = 0
    for rel in relations:
        if rel.get("subject_id") == entity_id or rel.get("object_id") == entity_id:
            evs = rel.get("evidences", [])
            if isinstance(evs, str):
                evs = [evs]
            cnt += len([x for x in evs if normalize_text(x)])
    return cnt


def infer_preferred_meta_layer(entity_name: str) -> Optional[str]:
    s = normalize_key(entity_name)

    if any(k in s for k in [
        "efficiency", "pce", "power conversion efficiency", "η",
        "stability", "retention", "yield", "capability", "reproducibility"
    ]):
        return "L5_Performance"

    if any(k in s for k in [
        "open-circuit voltage", "voc",
        "short-circuit current density", "jsc", "j_sc",
        "fill factor", "ff",
        "series resistance", "rs",
        "shunt resistance", "rsh",
        "effective reflection", "effective reflectance", "reff",
        "reflectance", "reflection",
        "contact angle", "hysteresis", "vbi", "vtfl", "active area", "yield"
    ]):
        return "L4_Device_Parameter"

    if any(k in s for k in [
        "recombination", "carrier", "trap", "conductivity", "passivation",
        "charge extraction", "mobility", "lifetime", "light trapping"
    ]):
        return "L3_Mechanism_Physics"

    return None


# ----------------------------------------------------
# 9. 材料污染过滤
# ----------------------------------------------------
def infer_material_family(material_system: str, title: str, focus: str) -> Optional[str]:
    text = " ".join([material_system, title, focus]).lower()

    silicon_hit = any(k in text for k in [
        "silicon", "c-si", "crystalline si", "string ribbon", "wafer", "monocrystalline"
    ])
    perovskite_hit = any(k in text for k in [
        "perovskite", "pb-i", "mapbi", "fapbi", "grain boundary", "peai", "tfe-peai"
    ])
    tandem_hit = any(k in text for k in [
        "tandem", "silicon/perovskite", "perovskite/silicon"
    ])

    if tandem_hit:
        return None
    if silicon_hit and not perovskite_hit:
        return "silicon"
    if perovskite_hit and not silicon_hit:
        return "perovskite"
    return None


def forbidden_patterns_for_family(material_family: Optional[str]) -> List[str]:
    if material_family == "silicon":
        return [
            r"\bpb[\s\-]?i\b",
            r"\bpeai\b",
            r"\btfe\b",
            r"\bperovskite\b",
            r"\bgrain boundar",
            r"\bspiro\b",
            r"\bpedot\b",
            r"\bpcbm\b",
            r"\bmapbi",
            r"\bfapbi",
            r"\bhtl\b",
            r"\betl\b",
        ]
    if material_family == "perovskite":
        return [
            r"\bstring ribbon\b",
            r"\bcrystalline silicon\b",
            r"\bmonocrystalline silicon\b",
            r"\bc\-?si\b",
            r"\bwafer\b",
            r"\bmetal assisted etching\b",
            r"\bmace\b",
            r"\bkoh\b",
            r"\bsaw-damage\b",
            r"\binverted pyramid\b",
            r"\balkaline etching\b"
        ]
    return []


def entity_text_blob(entity: Dict[str, Any]) -> str:
    name = normalize_text(entity.get("entity_name"))
    aliases = entity.get("aliases", [])
    if not isinstance(aliases, list):
        aliases = []
    blob = " | ".join([name] + [normalize_text(x) for x in aliases if normalize_text(x)])
    return blob.lower()


def detect_material_inconsistent_entity(
        entity: Dict[str, Any],
        forbidden_patterns: List[str]
) -> Optional[str]:
    blob = entity_text_blob(entity)
    for pat in forbidden_patterns:
        if re.search(pat, blob, flags=re.IGNORECASE):
            return pat
    return None


# ----------------------------------------------------
# 10. 关系重建
# ----------------------------------------------------
def rebuild_relations_with_id_mapping(
        relations: List[Dict[str, Any]],
        id_mapping: Dict[str, str]
) -> List[Dict[str, Any]]:
    relation_map = {}

    for rel in relations:
        sub = id_mapping.get(rel.get("subject_id"), rel.get("subject_id"))
        obj = id_mapping.get(rel.get("object_id"), rel.get("object_id"))

        raw_pred = normalize_text(rel.get("raw_predicate") or rel.get("predicate"))
        pred_group = normalize_predicate_group(rel.get("predicate_group"))

        if not sub or not obj or sub == obj or not raw_pred:
            continue

        evidences = rel.get("evidences", rel.get("evidence", []))
        if isinstance(evidences, str):
            evidences = [evidences]
        clean_evs = [normalize_text(x) for x in evidences if normalize_text(x)]

        sig = (sub, raw_pred, pred_group, obj)
        if sig not in relation_map:
            relation_map[sig] = {
                "subject_id": sub,
                "raw_predicate": raw_pred,
                "predicate_group": pred_group,
                "object_id": obj,
                "evidences": set(clean_evs)
            }
        else:
            relation_map[sig]["evidences"].update(clean_evs)

    final_relations = []
    for rel in relation_map.values():
        rel["evidences"] = sorted(rel["evidences"])
        final_relations.append(rel)

    final_relations = sorted(
        final_relations,
        key=lambda x: (
            normalize_text(x["subject_id"]),
            normalize_text(x["raw_predicate"]),
            normalize_text(x["object_id"])
        )
    )
    return final_relations


def merge_entities_by_id_mapping(
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        extra_id_mapping: Dict[str, str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not extra_id_mapping:
        return entities, rebuild_relations_with_id_mapping(relations, extra_id_mapping)

    entity_by_id = {e["entity_id"]: dict(e) for e in entities if e.get("entity_id")}
    merged_into = defaultdict(list)

    for old_id, new_id in extra_id_mapping.items():
        if old_id != new_id:
            merged_into[new_id].append(old_id)

    for survivor_id, merged_ids in merged_into.items():
        survivor = entity_by_id.get(survivor_id)
        if not survivor:
            continue

        aliases = set(survivor.get("aliases", []))
        all_merged_ids = set(survivor.get("merged_entity_ids", []))
        all_merged_ids.add(survivor_id)

        for mid in merged_ids:
            victim = entity_by_id.get(mid)
            if not victim:
                continue
            aliases.add(normalize_text(victim.get("entity_name")))
            for a in victim.get("aliases", []):
                a = normalize_text(a)
                if a:
                    aliases.add(a)
            for z in victim.get("merged_entity_ids", []):
                if normalize_text(z):
                    all_merged_ids.add(z)

        survivor["aliases"] = sorted([a for a in aliases if a])
        survivor["merged_entity_ids"] = sorted([x for x in all_merged_ids if x])

    kept_ids = set(entity_by_id.keys()) - {k for k, v in extra_id_mapping.items() if k != v}
    final_entities = [entity_by_id[eid] for eid in kept_ids if eid in entity_by_id]
    final_entities = sorted(final_entities, key=lambda x: normalize_text(x.get("entity_id")))

    final_relations = rebuild_relations_with_id_mapping(relations, extra_id_mapping)
    return final_entities, final_relations


# ----------------------------------------------------
# 11. 大模型主导 canonical merge
# ----------------------------------------------------
async def ask_llm_canonical_merge(
        model_api,
        candidate_entities: List[Dict[str, Any]],
        semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    payload = []
    for ent in candidate_entities:
        payload.append({
            "entity_id": ent.get("entity_id"),
            "entity_name": ent.get("entity_name"),
            "entity_type": ent.get("entity_type"),
            "meta_layer": ent.get("meta_layer"),
            "aliases": ent.get("aliases", []),
            "numeric_signature": list(entity_numeric_signature(ent))
        })

    prompt = LOCAL_CANONICAL_MERGE_PROMPT.format(
        entities_json=json.dumps(payload, ensure_ascii=False, indent=2)
    )

    async with semaphore:
        result = await ask_llm_json(model_api, prompt, temperature=0.0)

    if not isinstance(result, dict):
        return {"clusters": [], "keep_separate": []}

    clusters = result.get("clusters", [])
    keep_separate = result.get("keep_separate", [])

    if not isinstance(clusters, list):
        clusters = []
    if not isinstance(keep_separate, list):
        keep_separate = []

    return {
        "clusters": clusters,
        "keep_separate": keep_separate
    }


def canonical_merge_group_key(entity: Dict[str, Any]) -> Tuple[str, ...]:
    if is_l4_l5_entity(entity):
        m_sig = metric_signature(entity)
        if m_sig in {"stability_storage", "stability_thermal", "stability_operational", "lifetime"}:
            return ("L4L5", m_sig)
        return ("L4L5", m_sig)
    return ("DEFAULT", entity_concept_signature(entity) or "unknown_concept")


async def canonical_merge_with_llm_and_rules(
        model_api,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        semaphore: asyncio.Semaphore
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    report = {
        "llm_proposed_clusters": [],
        "accepted_merges": [],
        "blocked_merges": []
    }

    groups = defaultdict(list)
    for ent in entities:
        sig = canonical_merge_group_key(ent)
        groups[sig].append(ent)

    extra_id_mapping = {}
    rename_map = {}

    for sig, group in groups.items():
        if len(group) <= 1:
            continue

        llm_result = await ask_llm_canonical_merge(model_api, group, semaphore)
        report["llm_proposed_clusters"].append({
            "merge_group": str(sig),
            "clusters": llm_result.get("clusters", [])
        })

        entity_by_id = {e["entity_id"]: e for e in group}

        for cluster in llm_result.get("clusters", []):
            member_ids = cluster.get("member_ids", [])
            canonical_name = normalize_text(cluster.get("canonical_name"))
            reason = normalize_text(cluster.get("reason"))

            if not isinstance(member_ids, list) or len(member_ids) <= 1:
                continue

            members = [entity_by_id[mid] for mid in member_ids if mid in entity_by_id]
            if len(members) <= 1:
                continue

            blocked = False
            block_reason = ""
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    bad, why = hard_merge_conflict(members[i], members[j])
                    if bad:
                        blocked = True
                        block_reason = why
                        break
                if blocked:
                    break

            if blocked:
                report["blocked_merges"].append({
                    "member_ids": member_ids,
                    "canonical_name": canonical_name,
                    "llm_reason": reason,
                    "block_reason": block_reason
                })
                continue

            def entity_score(ent: Dict[str, Any]) -> Tuple[int, int, int, int]:
                return (
                    relation_count_for_entity(ent["entity_id"], relations),
                    evidence_count_for_entity(ent["entity_id"], relations),
                    len(entity_numeric_signature(ent)),
                    len(ent.get("aliases", []))
                )

            survivor = sorted(members, key=entity_score, reverse=True)[0]
            survivor_id = survivor["entity_id"]

            for ent in members:
                if ent["entity_id"] != survivor_id:
                    extra_id_mapping[ent["entity_id"]] = survivor_id

            if canonical_name:
                rename_map[survivor_id] = canonical_name

            report["accepted_merges"].append({
                "survivor_id": survivor_id,
                "member_ids": member_ids,
                "canonical_name": canonical_name,
                "reason": reason
            })

    if not extra_id_mapping:
        return entities, relations, report

    final_entities, final_relations = merge_entities_by_id_mapping(
        entities=entities,
        relations=relations,
        extra_id_mapping=extra_id_mapping
    )

    for ent in final_entities:
        eid = ent["entity_id"]
        if eid in rename_map and rename_map[eid]:
            ent["entity_name"] = rename_map[eid]

    return final_entities, final_relations, report


# ----------------------------------------------------
# 12. 跨层冲突解析
# ----------------------------------------------------
def resolve_cross_layer_conflicts(
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    report = {
        "resolved_conflicts": [],
        "unresolved_conflicts": []
    }

    groups = defaultdict(list)
    for ent in entities:
        sig = entity_concept_signature(ent)
        if sig:
            groups[sig].append(ent)

    extra_id_mapping = {}

    for sig, group in groups.items():
        layers = sorted(set(
            normalize_text(e.get("meta_layer")) for e in group if normalize_text(e.get("meta_layer"))
        ))
        if len(group) <= 1 or len(layers) <= 1:
            continue

        group_num_sigs = [entity_numeric_signature(e) for e in group]
        non_empty_num_sigs = sorted(set(sig for sig in group_num_sigs if sig))
        numeric_mismatch = len(non_empty_num_sigs) > 1

        if numeric_mismatch:
            report["unresolved_conflicts"].append({
                "concept": group[0].get("entity_name"),
                "reason": "numeric_signature_mismatch",
                "layers": layers,
                "entity_ids": [e.get("entity_id") for e in group],
                "numeric_signatures": [list(sig) for sig in group_num_sigs]
            })
            continue

        preferred = infer_preferred_meta_layer(group[0].get("entity_name"))
        if not preferred:
            report["unresolved_conflicts"].append({
                "concept": group[0].get("entity_name"),
                "reason": "no_preferred_meta_layer",
                "layers": layers,
                "entity_ids": [e.get("entity_id") for e in group]
            })
            continue

        preferred_candidates = [e for e in group if normalize_text(e.get("meta_layer")) == preferred]
        survivor_pool = preferred_candidates if preferred_candidates else group

        def entity_score(ent: Dict[str, Any]) -> Tuple[int, int, int]:
            eid = ent.get("entity_id")
            return (
                relation_count_for_entity(eid, relations),
                evidence_count_for_entity(eid, relations),
                len(ent.get("aliases", []))
            )

        survivor = sorted(survivor_pool, key=lambda e: entity_score(e), reverse=True)[0]
        survivor_id = survivor["entity_id"]
        survivor["meta_layer"] = preferred

        merged_ids = []
        for ent in group:
            eid = ent.get("entity_id")
            if eid != survivor_id:
                extra_id_mapping[eid] = survivor_id
                merged_ids.append(eid)

        report["resolved_conflicts"].append({
            "concept": group[0].get("entity_name"),
            "preferred_meta_layer": preferred,
            "survivor_id": survivor_id,
            "merged_ids": merged_ids,
            "original_layers": layers,
            "numeric_signature": list(group_num_sigs[0]) if group_num_sigs and group_num_sigs[0] else []
        })

    if not extra_id_mapping:
        return entities, relations, report

    final_entities, final_relations = merge_entities_by_id_mapping(
        entities=entities,
        relations=relations,
        extra_id_mapping=extra_id_mapping
    )

    sig_to_preferred = {}
    for item in report["resolved_conflicts"]:
        sig_to_preferred[concept_signature(item["concept"])] = item["preferred_meta_layer"]

    for ent in final_entities:
        sig = entity_concept_signature(ent)
        if sig in sig_to_preferred:
            ent["meta_layer"] = sig_to_preferred[sig]

    return final_entities, final_relations, report


# ----------------------------------------------------
# 13. 过滤与报告
# ----------------------------------------------------
def remove_material_inconsistent_entities(
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
        material_family: Optional[str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    report = {
        "material_family": material_family,
        "removed_entities": [],
        "removed_entity_ids": [],
        "removed_relations_count": 0
    }

    forbidden_patterns = forbidden_patterns_for_family(material_family)
    if not forbidden_patterns:
        return entities, relations, report

    remove_ids = set()

    for ent in entities:
        matched_pat = detect_material_inconsistent_entity(ent, forbidden_patterns)
        if matched_pat:
            remove_ids.add(ent["entity_id"])
            report["removed_entities"].append({
                "entity_id": ent["entity_id"],
                "entity_name": ent.get("entity_name"),
                "entity_type": ent.get("entity_type"),
                "matched_pattern": matched_pat
            })

    if not remove_ids:
        return entities, relations, report

    kept_entities = [e for e in entities if e.get("entity_id") not in remove_ids]
    kept_relations = []
    removed_rel_cnt = 0

    for rel in relations:
        if rel.get("subject_id") in remove_ids or rel.get("object_id") in remove_ids:
            removed_rel_cnt += 1
            continue
        kept_relations.append(rel)

    report["removed_entity_ids"] = sorted(remove_ids)
    report["removed_relations_count"] = removed_rel_cnt
    return kept_entities, kept_relations, report


def remove_orphan_entities(
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    report = {"flagged_orphans": []}

    used_ids = set()
    for rel in relations:
        if rel.get("subject_id"):
            used_ids.add(rel["subject_id"])
        if rel.get("object_id"):
            used_ids.add(rel["object_id"])

    flagged_entities = []
    for ent in entities:
        eid = ent.get("entity_id")
        ent_copy = dict(ent)
        ent_copy["is_orphan"] = eid not in used_ids
        if ent_copy["is_orphan"]:
            report["flagged_orphans"].append({
                "entity_id": eid,
                "entity_name": ent.get("entity_name"),
                "meta_layer": ent.get("meta_layer")
            })
        flagged_entities.append(ent_copy)

    return flagged_entities, report


def build_focus_alignment_report(
        doc: Dict[str, Any],
        entities: List[Dict[str, Any]]
) -> Dict[str, Any]:
    title = normalize_key(doc.get("article_title_author"))
    focus = normalize_key(doc.get("research_focus"))
    target_problem = normalize_key(doc.get("target_problem"))

    theme_text = " ".join([title, focus, target_problem])

    theme_terms = set()
    for token in re.findall(r"[a-zA-Z][a-zA-Z0-9\-\_]{2,}", theme_text):
        if token not in {
            "this", "work", "introduces", "for", "the", "and", "with", "via", "using",
            "solar", "cells", "cell", "device", "devices", "improvement", "investigating"
        }:
            theme_terms.add(token)

    suspicious = []
    for ent in entities:
        blob = entity_text_blob(ent)
        overlap = sum(1 for t in theme_terms if t in blob)
        if overlap == 0 and len(theme_terms) > 0:
            suspicious.append({
                "entity_id": ent.get("entity_id"),
                "entity_name": ent.get("entity_name"),
                "entity_type": ent.get("entity_type"),
                "meta_layer": ent.get("meta_layer")
            })

    return {
        "theme_terms_sample": sorted(list(theme_terms))[:20],
        "possible_off_topic_entities": suspicious[:50]
    }


# ----------------------------------------------------
# 14. 单篇处理核心逻辑
# ----------------------------------------------------
async def process_single_paper_legacy(api_name: str, model_api: Any, file_path: str,
                                      semaphore: asyncio.Semaphore) -> str:
    filename = os.path.basename(file_path)
    validation_report = {
        "file": filename,
        "api_used": api_name,
        "steps": {}
    }

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception as e:
        return f"[ERROR] [{api_name} | {filename}] 读取失败: {e}"

    if "result" in doc and isinstance(doc["result"], dict):
        doc = doc["result"]

    title = normalize_text(doc.get("article_title_author", "Unknown Title"))
    problem = normalize_text(doc.get("target_problem", "Unknown Problem"))
    material_system = normalize_text(doc.get("material_system", "Unknown"))
    research_focus = normalize_text(doc.get("research_focus", "Unknown"))

    entities = doc.get("entities", [])
    relations = doc.get("relations", [])

    if not isinstance(entities, list) or not entities:
        return f"[WARN] [{api_name} | {filename}] 空实体，跳过"

    if not isinstance(relations, list):
        relations = []

    logging.info(f"[{api_name} | {filename}] 开始独立融合处理...")

    # -------- 步骤 0: 预清洗 --------
    clean_entities = []
    for ent in entities:
        norm_ent = normalize_entity_record(ent)
        if norm_ent is not None:
            clean_entities.append(norm_ent)

    clean_relations = []
    for rel in relations:
        norm_rel = normalize_relation_record(rel)
        if norm_rel is not None:
            clean_relations.append(norm_rel)

    entities = clean_entities
    relations = clean_relations

    if not entities:
        return f"[WARN] [{api_name} | {filename}] 清洗后无有效实体，跳过"

    validation_report["steps"]["preclean"] = {
        "entity_count": len(entities),
        "relation_count": len(relations)
    }

    if not relations:
        validation_report["steps"][
            "fatal_warning"] = "No valid relations survived preclean. Check relation field compatibility."
        doc["entities"] = []
        doc["relations"] = []
        doc["total_entities"] = 0
        doc["total_relations"] = 0
        doc["validation_summary"] = {
            "api_used": api_name,
            "material_family": infer_material_family(material_system, title, research_focus),
            "removed_inconsistent_entities": 0,
            "resolved_cross_layer_conflicts": 0,
            "unresolved_cross_layer_conflicts": 0,
            "flagged_orphans": 0
        }

        save_path = os.path.join(OUTPUT_DIR, filename)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=4)

        if SAVE_VALIDATION_REPORT:
            report_path = os.path.join(REPORT_DIR, f"{safe_filename_stem(filename)}.review.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(validation_report, f, ensure_ascii=False, indent=4)

        return f"[WARN] [{api_name} | {filename}] 预清洗后关系为 0，请检查 schema 兼容。"

    # -------- 步骤 1: 新类别归位 --------
    unknown_types = sorted({
        e["entity_type"] for e in entities
        if e["entity_type"]
           and e["entity_type"] not in BASE_ONTOLOGY_MAPPING
           and infer_meta_layer_from_type_and_name(e["entity_type"], e["entity_name"]) is None
           and e["entity_type"] not in {"Other", "unknown", "None", ""}
    })

    validation_report["steps"]["unknown_types"] = unknown_types
    local_type_mapping = dict(BASE_ONTOLOGY_MAPPING)

    if unknown_types:
        prompt = LOCAL_ONTOLOGY_PROMPT.format(
            title=title,
            new_types=json.dumps(unknown_types, ensure_ascii=False)
        )
        try:
            async with semaphore:
                new_mappings = await ask_llm_json(model_api, prompt, temperature=0.05)

            safe_mappings = {}
            for k, v in new_mappings.items():
                k = normalize_entity_type(k)
                v = normalize_text(v)
                if k in unknown_types and v in VALID_META_LAYERS:
                    safe_mappings[k] = v

            local_type_mapping.update(safe_mappings)
            validation_report["steps"]["new_type_mapping"] = safe_mappings
        except Exception as e:
            logging.warning(f"[WARN] [{api_name} | {filename}] 新类型归层失败，将保留启发式映射/Unknown: {e}")

    for ent in entities:
        ent_type = ent.get("entity_type", "Material")
        if ent_type in local_type_mapping:
            ent["meta_layer"] = local_type_mapping[ent_type]
        elif ent.get("meta_layer") == "Unknown":
            ent["meta_layer"] = infer_meta_layer_from_type_and_name(ent_type, ent.get("entity_name")) or "Unknown"

    # -------- 步骤 2: 分组后做局部实体对齐 --------
    entity_contexts = build_entity_contexts(entities, relations)
    entities_by_group = defaultdict(list)

    for ent in entities:
        e_type = ent.get("entity_type", "Other")
        meta_layer = ent.get("meta_layer", "Unknown")

        if meta_layer in {"L4_Device_Parameter", "L5_Performance"}:
            m_sig = metric_signature(ent)
            s_sig = state_signature(ent)
            c_sig = condition_signature(ent)
            n_sig = entity_numeric_signature(ent)

            if n_sig:
                group_key = ("NUMERIC", e_type, m_sig, s_sig, c_sig, n_sig)
            else:
                group_key = ("GENERIC", e_type, m_sig, s_sig, c_sig)
        else:
            group_key = ("DEFAULT", e_type)

        entities_by_group[group_key].append(ent)

    local_name_mapping: Dict[Tuple[str, str], str] = {}
    alignment_summary = {}

    for group_key, grouped_entities in entities_by_group.items():
        if len(grouped_entities) <= 1:
            continue

        mode = group_key[0]
        e_type = group_key[1]

        payload = build_alignment_payload(grouped_entities, entity_contexts)
        if len(payload) <= 1:
            continue

        raw_names = [item["name"] for item in payload]

        if mode == "DEFAULT":
            group_desc = f"type={e_type}"
        elif mode == "NUMERIC":
            group_desc = (
                f"type={e_type} | metric={group_key[2]} | state={group_key[3]} "
                f"| cond={group_key[4]} | num={group_key[5]}"
            )
        else:
            group_desc = (
                f"type={e_type} | metric={group_key[2]} | state={group_key[3]} "
                f"| cond={group_key[4]}"
            )

        prompt = LOCAL_ALIGNMENT_PROMPT.format(
            title=title,
            problem=problem,
            entity_type=group_desc,
            entities_json=json.dumps(payload, ensure_ascii=False, indent=2)
        )

        try:
            async with semaphore:
                grouped_json = await ask_llm_json(model_api, prompt, temperature=0.05)

            safe_groups = safe_validate_alignment_output(grouped_json, raw_names)
            alignment_summary[str(group_key)] = safe_groups

            for std_name, raw_list in safe_groups.items():
                std_name = normalize_text(std_name)
                for raw in raw_list:
                    local_name_mapping[(normalize_key(raw), e_type)] = std_name

        except Exception as e:
            logging.warning(f"[WARN] [{api_name} | {filename}] 分组 {group_key} 对齐失败，保留原名: {e}")

    validation_report["steps"]["alignment_groups"] = alignment_summary

    # -------- 步骤 3: 初步 collapse --------
    collapsed_entities = {}
    id_mapping = {}

    for ent in entities:
        old_id = ent["entity_id"]
        raw_name = normalize_text(ent["entity_name"])
        e_type = normalize_entity_type(ent.get("entity_type", "Other"))
        std_name = local_name_mapping.get((normalize_key(raw_name), e_type), raw_name)
        meta_layer = ent.get("meta_layer", "Unknown")

        collapse_key = build_collapse_key(
            std_name=std_name,
            e_type=e_type,
            meta_layer=meta_layer,
            raw_name=raw_name
        )

        if collapse_key not in collapsed_entities:
            survivor = dict(ent)
            survivor["entity_name"] = std_name
            survivor["entity_type"] = e_type
            survivor["meta_layer"] = meta_layer
            survivor["aliases"] = [raw_name]
            survivor["merged_entity_ids"] = [old_id]
            collapsed_entities[collapse_key] = survivor
            id_mapping[old_id] = old_id
        else:
            survivor = collapsed_entities[collapse_key]
            survivor_id = survivor["entity_id"]
            id_mapping[old_id] = survivor_id

            aliases = set(survivor.get("aliases", []))
            aliases.add(raw_name)
            survivor["aliases"] = sorted([a for a in aliases if a])

            merged_ids = set(survivor.get("merged_entity_ids", []))
            merged_ids.add(old_id)
            survivor["merged_entity_ids"] = sorted([x for x in merged_ids if x])

    entities = list(collapsed_entities.values())
    relations = rebuild_relations_with_id_mapping(relations, id_mapping)

    validation_report["steps"]["post_collapse"] = {
        "entity_count": len(entities),
        "relation_count": len(relations)
    }

    # -------- 步骤 4: 大模型主导 canonical merge + 规则验收 --------
    entities, relations, llm_merge_report = await canonical_merge_with_llm_and_rules(
        model_api=model_api,
        entities=entities,
        relations=relations,
        semaphore=semaphore
    )
    validation_report["steps"]["llm_canonical_merge"] = llm_merge_report

    # -------- 步骤 5: 材料体系一致性过滤 --------
    material_family = infer_material_family(material_system, title, research_focus)
    entities, relations, material_filter_report = remove_material_inconsistent_entities(
        entities=entities,
        relations=relations,
        material_family=material_family
    )
    validation_report["steps"]["material_filter"] = material_filter_report

    # -------- 步骤 6: 同名跨层冲突解析 --------
    entities, relations, cross_layer_report = resolve_cross_layer_conflicts(
        entities=entities,
        relations=relations
    )
    validation_report["steps"]["cross_layer_resolution"] = cross_layer_report

    # -------- 步骤 7: 孤点打标 --------
    entities, orphan_report = remove_orphan_entities(entities, relations)
    validation_report["steps"]["orphan_annotation"] = orphan_report

    # -------- 步骤 8: 主题对齐软报告 --------
    validation_report["steps"]["focus_alignment_soft_report"] = build_focus_alignment_report(
        doc=doc,
        entities=entities
    )

    # -------- 步骤 9: 最终整理 --------
    doc["entities"] = sorted(entities, key=lambda x: normalize_text(x.get("entity_id")))
    doc["relations"] = sorted(
        relations,
        key=lambda x: (
            normalize_text(x["subject_id"]),
            normalize_text(x["raw_predicate"]),
            normalize_text(x["object_id"])
        )
    )
    doc["total_entities"] = len(doc["entities"])
    doc["total_relations"] = len(doc["relations"])
    doc["validation_summary"] = {
        "api_used": api_name,
        "material_family": material_family,
        "removed_inconsistent_entities": len(material_filter_report.get("removed_entity_ids", [])),
        "resolved_cross_layer_conflicts": len(cross_layer_report.get("resolved_conflicts", [])),
        "unresolved_cross_layer_conflicts": len(cross_layer_report.get("unresolved_conflicts", [])),
        "flagged_orphans": len(orphan_report.get("flagged_orphans", []))
    }

    save_path = os.path.join(OUTPUT_DIR, filename)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=4)

    if SAVE_VALIDATION_REPORT:
        report_path = os.path.join(REPORT_DIR, f"{safe_filename_stem(filename)}.review.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(validation_report, f, ensure_ascii=False, indent=4)

    return (
        f" [{api_name} | {filename}] 融合处理完成"
        f" 实体: {doc['total_entities']}, 关系: {doc['total_relations']},"
        f" 接受: {len(llm_merge_report.get('accepted_merges', []))},"
        f" 拦截: {len(llm_merge_report.get('blocked_merges', []))}"
    )


async def process_single_paper_one_shot(api_name: str, model_api: Any, file_path: str,
                                        semaphore: asyncio.Semaphore) -> str:
    if not USE_ONE_SHOT_REFINEMENT:
        return await process_single_paper_legacy(api_name, model_api, file_path, semaphore)

    filename = os.path.basename(file_path)
    validation_report = {
        "file": filename,
        "api_used": api_name,
        "mode": "one_shot_refinement",
        "steps": {}
    }

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception as e:
        return f"[ERROR] [{api_name} | {filename}] 读取失败: {e}"

    if "result" in doc and isinstance(doc["result"], dict):
        doc = doc["result"]

    title = normalize_text(doc.get("article_title_author", "Unknown Title"))
    problem = normalize_text(doc.get("target_problem", "Unknown Problem"))
    material_system = normalize_text(doc.get("material_system", "Unknown"))
    research_focus = normalize_text(doc.get("research_focus", "Unknown"))

    entities = doc.get("entities", [])
    relations = doc.get("relations", [])

    if not isinstance(entities, list) or not entities:
        return f"[WARN] [{api_name} | {filename}] 空实体，跳过"
    if not isinstance(relations, list):
        relations = []

    logging.info(f" [{api_name} | {filename}] 开始 one-shot 全文融合...")

    clean_entities = []
    for ent in entities:
        norm_ent = normalize_entity_record(ent)
        if norm_ent is not None:
            clean_entities.append(norm_ent)

    clean_relations = []
    for rel in relations:
        norm_rel = normalize_relation_record(rel)
        if norm_rel is not None:
            clean_relations.append(norm_rel)

    entities = clean_entities
    relations = clean_relations

    if not entities:
        return f"[WARN] [{api_name} | {filename}] 清洗后无有效实体，跳过"

    validation_report["steps"]["preclean"] = {
        "entity_count": len(entities),
        "relation_count": len(relations)
    }

    if not relations:
        validation_report["steps"][
            "fatal_warning"] = "No valid relations survived preclean. Check relation field compatibility."
        doc["entities"] = []
        doc["relations"] = []
        doc["total_entities"] = 0
        doc["total_relations"] = 0
        doc["validation_summary"] = {
            "api_used": api_name,
            "material_family": infer_material_family(material_system, title, research_focus),
            "removed_inconsistent_entities": 0,
            "resolved_cross_layer_conflicts": 0,
            "unresolved_cross_layer_conflicts": 0,
            "flagged_orphans": 0
        }

        save_path = os.path.join(OUTPUT_DIR, filename)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=4)

        if SAVE_VALIDATION_REPORT:
            report_path = os.path.join(REPORT_DIR, f"{safe_filename_stem(filename)}.review.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(validation_report, f, ensure_ascii=False, indent=4)

        return f"[WARN] [{api_name} | {filename}] 预清洗后关系为 0，请检查 schema 兼容。"

    payload = build_one_shot_payload(entities, relations)
    validation_report["steps"]["one_shot_payload"] = {
        "entity_payload_count": len(payload),
        "relation_count": len(relations)
    }

    prompt = ONE_SHOT_REFINEMENT_PROMPT.format(
        title=title,
        problem=problem,
        material_system=material_system,
        research_focus=research_focus,
        entities_json=json.dumps(payload, ensure_ascii=False, indent=2)
    )

    try:
        debug_context = {"file_name": filename, "api_name": api_name, "stage": "one_shot_refinement"}
        async with semaphore:
            one_shot_result = await ask_llm_json(model_api, prompt, temperature=ONE_SHOT_TEMPERATURE, debug_context=debug_context)

        entities, relations, one_shot_report = apply_one_shot_refinement_plan(
            one_shot_result,
            entities,
            relations
        )
        validation_report["steps"]["one_shot_refinement"] = one_shot_report
    except Exception as e:
        validation_report["steps"]["one_shot_refinement_error"] = str(e)
        if ONE_SHOT_FALLBACK_TO_LEGACY:
            logging.warning(f"[WARN] [{api_name} | {filename}] one-shot 失败，回退旧流程: {e}")
            return await process_single_paper_legacy(api_name, model_api, file_path, semaphore)
        raise

    material_family = infer_material_family(material_system, title, research_focus)
    entities, relations, material_filter_report = remove_material_inconsistent_entities(
        entities=entities,
        relations=relations,
        material_family=material_family
    )
    validation_report["steps"]["material_filter"] = material_filter_report

    entities, relations, cross_layer_report = resolve_cross_layer_conflicts(
        entities=entities,
        relations=relations
    )
    validation_report["steps"]["cross_layer_resolution"] = cross_layer_report

    entities, orphan_report = remove_orphan_entities(entities, relations)
    validation_report["steps"]["orphan_annotation"] = orphan_report

    validation_report["steps"]["focus_alignment_soft_report"] = build_focus_alignment_report(
        doc=doc,
        entities=entities
    )

    doc["entities"] = sorted(entities, key=lambda x: normalize_text(x.get("entity_id")))
    doc["relations"] = sorted(
        relations,
        key=lambda x: (
            normalize_text(x["subject_id"]),
            normalize_text(x["raw_predicate"]),
            normalize_text(x["object_id"])
        )
    )
    doc["total_entities"] = len(doc["entities"])
    doc["total_relations"] = len(doc["relations"])
    doc["validation_summary"] = {
        "api_used": api_name,
        "mode": "one_shot_refinement",
        "material_family": material_family,
        "accepted_merges": len(validation_report["steps"].get("one_shot_refinement", {}).get("accepted_merges", [])),
        "blocked_merges": len(validation_report["steps"].get("one_shot_refinement", {}).get("blocked_merges", [])),
        "removed_inconsistent_entities": len(material_filter_report.get("removed_entity_ids", [])),
        "resolved_cross_layer_conflicts": len(cross_layer_report.get("resolved_conflicts", [])),
        "unresolved_cross_layer_conflicts": len(cross_layer_report.get("unresolved_conflicts", [])),
        "flagged_orphans": len(orphan_report.get("flagged_orphans", []))
    }

    save_path = os.path.join(OUTPUT_DIR, filename)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=4)

    if SAVE_VALIDATION_REPORT:
        report_path = os.path.join(REPORT_DIR, f"{safe_filename_stem(filename)}.review.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(validation_report, f, ensure_ascii=False, indent=4)

    return (
        f" [{api_name} | {filename}] 单次融合处理完成"
        f" 实体: {doc['total_entities']}, 关系: {doc['total_relations']},"
        f" 接受: {len(validation_report['steps'].get('one_shot_refinement', {}).get('accepted_merges', []))},"
        f" 拦截: {len(validation_report['steps'].get('one_shot_refinement', {}).get('blocked_merges', []))}"
    )


async def process_single_paper_safe(api_name: str, model_api: Any, file_path: str, semaphore: asyncio.Semaphore) -> \
Dict[str, Any]:
    filename = os.path.basename(file_path)
    try:
        message = await process_single_paper_one_shot(api_name, model_api, file_path, semaphore)
        if str(message).startswith("[WARN]"):
            status = "warning"
        else:
            status = "success"
        return {
            "file": filename,
            "api_name": api_name,
            "status": status,
            "message": message
        }
    except Exception as e:
        logging.exception(f"[ERROR] [{api_name} | {filename}] 处理失败")
        err_message = f"[ERROR] [{api_name} | {filename}] 未捕获异常: {e}"
        raw_response_file = ""
        m = re.search(r"raw_response_file=([^|]+)$", str(e))
        if m:
            raw_response_file = m.group(1).strip()
        append_failed_case(
            file_name=filename,
            api_name=api_name,
            stage="one_shot_refinement" if USE_ONE_SHOT_REFINEMENT else "legacy_refinement",
            error_message=str(e),
            retry_count=MAX_RETRIES,
            raw_response_file=raw_response_file,
        )
        return {
            "file": filename,
            "api_name": api_name,
            "status": "failed",
            "message": err_message
        }


# ----------------------------------------------------
# 15. 主循环
# ----------------------------------------------------
async def main():
    print(" 启动【One-shot 全文融合 + 规则验收】批处理引擎...")
    print(f" 已配置 API 池: {API_NAMES}")
    print(f" 任务分配模式: 统一输出，文件级轮询交替 (Round-Robin)")
    print(f" 断点续传: {'开启' if RESUME_MODE and not OVERWRITE_EXISTING else '关闭'}")
    print(f" 主流程跳过历史失败: {'开启' if SKIP_PREVIOUS_FAILED_IN_MAIN else '关闭'}")
    print(f" 最大并发槽位: {MAX_CONCURRENCY}")
    print(
        f" One-shot 模式: {'开启' if USE_ONE_SHOT_REFINEMENT else '关闭'} | 失败回退: {'开启' if ONE_SHOT_FALLBACK_TO_LEGACY else '关闭'}")

    if not os.path.exists(INPUT_DIR):
        print(f"[ERROR] 找不到输入目录: {INPUT_DIR}")
        return

    json_files = sorted([
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.endswith(".json")
    ])

    if not json_files:
        print("[WARN] 输入目录中没有 JSON 文件。")
        return

    # 初始化全部客户端
    api_clients = {}
    for name in API_NAMES:
        try:
            api_clients[name] = get_model_api(name)
        except Exception as e:
            print(f"[WARN] 无法初始化 API [{name}]: {e}")

    valid_api_names = list(api_clients.keys())
    if not valid_api_names:
        print("[ERROR] 没有可用的 API，程序退出。请检查 config。")
        return

    pending_tasks: List[str] = []
    skipped_tasks: List[str] = []
    skipped_failed_tasks: List[str] = []
    previous_failed_files = load_failed_case_files(FAILED_CASES_JSONL) if (RESUME_MODE and SKIP_PREVIOUS_FAILED_IN_MAIN) else set()

    for fp in json_files:
        out_path = os.path.join(OUTPUT_DIR, os.path.basename(fp))
        base_name = os.path.basename(fp)
        if RESUME_MODE and not OVERWRITE_EXISTING and is_effective_output_file(out_path):
            skipped_tasks.append(fp)
        elif RESUME_MODE and SKIP_PREVIOUS_FAILED_IN_MAIN and base_name in previous_failed_files:
            skipped_failed_tasks.append(fp)
        else:
            pending_tasks.append(fp)

    total_tasks = len(pending_tasks) + len(skipped_tasks) + len(skipped_failed_tasks)
    run_status = build_initial_run_status(
        total_files=total_tasks,
        pending_files=pending_tasks,
        skipped_files=skipped_tasks,
        skipped_failed_files=skipped_failed_tasks
    )
    atomic_write_json(RUN_STATUS_FILE, run_status)

    print(
        f" 共生成 {total_tasks} 个处理任务 | 待处理 {len(pending_tasks)} 个 | "
        f"断点续传跳过 {len(skipped_tasks)} 个 | 历史失败跳过 {len(skipped_failed_tasks)} 个\n"
    )

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    start_time = time.time()

    success_count = 0
    warning_count = 0
    failed_count = 0
    skipped_existing_count = len(skipped_tasks)
    skipped_failed_count = len(skipped_failed_tasks)

    # 跟踪各个 API 处理的数量
    api_stats = {name: {"success": 0, "warning": 0, "failed": 0} for name in valid_api_names}

    for fp in skipped_tasks:
        print(f"⏭ [{os.path.basename(fp)}] 已存在有效输出，跳过")
    for fp in skipped_failed_tasks:
        print(f"⏭ [{os.path.basename(fp)}] 历史失败样本，主流程跳过，等待后续专项修复")

    if not pending_tasks:
        print(" 没有需要新处理的任务，本次直接结束。")
        print(f" 输出目录: {OUTPUT_DIR}")
        print(f" 运行状态文件: {RUN_STATUS_FILE}")
        print(f" 失败队列文件: {FAILED_CASES_JSONL}")
        return

    # 任务分配逻辑 (负载均衡模式：文件轮询平分给各个可用模型)
    tasks = []
    for i, fp in enumerate(pending_tasks):
        api_name = valid_api_names[i % len(valid_api_names)]
        client = api_clients[api_name]
        tasks.append(asyncio.create_task(process_single_paper_safe(api_name, client, fp, semaphore)))

    for task in asyncio.as_completed(tasks):
        result = await task
        file_name = result["file"]
        api_name = result["api_name"]
        status = result["status"]
        message = result["message"]

        print(message)

        if status == "success":
            success_count += 1
            api_stats[api_name]["success"] += 1
        elif status == "warning":
            warning_count += 1
            api_stats[api_name]["warning"] += 1
        elif status == "failed":
            failed_count += 1
            api_stats[api_name]["failed"] += 1

        update_run_status(run_status, file_name, api_name, status, message)
        atomic_write_json(RUN_STATUS_FILE, run_status)

        log_progress(
            done=success_count + warning_count + failed_count + skipped_existing_count + skipped_failed_count,
            total=total_tasks,
            success=success_count,
            warning=warning_count,
            failed=failed_count,
            skipped_existing=skipped_existing_count,
            skipped_failed=skipped_failed_count,
            start_time=start_time
        )

    print("\n" + "=" * 60)
    print(" 全部任务处理完成")
    print(f" 输出总目录: {OUTPUT_DIR} (全部文件在同一文件夹下)")
    if SAVE_VALIDATION_REPORT:
        print(f" 审查报告目录: {REPORT_DIR}")
    print(f" 运行状态文件: {RUN_STATUS_FILE}")
    print("\n API 使用统计:")
    for api_name, stats in api_stats.items():
        print(f"  - [{api_name}]: 成功={stats['success']}, 警告={stats['warning']}, 失败={stats['failed']}")
    print(
        f"\n 最终总计 | success={success_count} | warning={warning_count} | "
        f"failed={failed_count} | skipped_existing={skipped_existing_count} | skipped_failed={skipped_failed_count}"
    )
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    asyncio.run(main())
