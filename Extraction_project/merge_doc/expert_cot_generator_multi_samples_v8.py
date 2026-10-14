#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PV-ECoT multi-sample generator V8

核心变化：
- 复用原始生成脚本中的主链过滤、局部链过滤和样本质量过滤逻辑；
- 不再每篇论文只选一个 angle/anchor；
- 每篇论文先构造多个候选机制链 candidate；
- 一次 LLM 调用让模型自己判断生成 0~3 条样本；
- 每条样本必须绑定一个 candidate_id，方便追溯 fact_skeleton / evidence；
- 输出仍然是：
  1) 训练 JSONL：instruction/input/reasoning_text/output/meta_info
  2) debug JSONL：source_file、candidate_id、fact_skeleton、supporting_evidences 等追溯信息

使用方式：
1. 把本脚本放到和原脚本 expert_cot_generator_one_call_prompt_celltype_v6_expert_reasoning.py 同一目录；
2. 修改 USER_CONFIG；
3. 运行：
   python expert_cot_generator_multi_samples_v8.py
"""

import asyncio
import hashlib
import importlib.util
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import aiofiles


# ============================================================
# 运行配置
# ============================================================

USER_CONFIG = {
    # 原始生成脚本路径。建议本脚本和原脚本放同一目录，则保持默认。
    "original_script_path": str(Path(__file__).with_name("expert_cot_generator_one_call_prompt_celltype_v6_expert_reasoning.py")),

    # 输入目录：建议使用 dry-run 输出的 refined_results_generatable；
    # 也可以直接用原始 refined_results。
    "input_dir": os.getenv("PV_REFINED_RESULTS_DIR", str(Path(__file__).resolve().parents[2] / "outputs" / "refined_results")),

    # 输出文件：建议单独输出 V3，不要覆盖旧版。
    "output_jsonl": os.getenv("PV_MULTI_INSTRUCTIONS_JSONL", str(Path(__file__).resolve().parents[2] / "outputs" / "generation" / "multi_instructions.jsonl")),
    "debug_jsonl": os.getenv("PV_MULTI_DEBUG_JSONL", str(Path(__file__).resolve().parents[2] / "outputs" / "generation" / "multi_debug.jsonl")),
    "bad_response_dir": os.getenv("PV_MULTI_BAD_RESP_DIR", str(Path(__file__).resolve().parents[2] / "outputs" / "generation" / "bad_multi_responses")),

    # 断点续传。开启后，debug 中已经出现过的 source_file 会整篇跳过。
    "resume_mode": True,
    "overwrite_output": False,

    # 每篇最多让模型生成几条。建议 3。
    "max_samples_per_paper": 6,

    # 是否允许模型生成 application_answer 长答案样本。
    # 长答案适合“如何构建/如何实现/如何优化/如何设计器件”这类问题。
    "enable_application_answer": True,

    # 每篇最多允许几个 application_answer。
    # 建议 1，避免长答案过多导致模型变啰嗦。
    "max_application_samples_per_paper": 1,

    # 每种类型最多提供多少候选链给模型。建议 2。
    "top_candidates_per_family": 2,

    # 每篇论文最多给模型多少个候选链，避免 prompt 过长。建议 6~8。
    "max_candidates_total": 6,

    # 候选 fact_skeleton 过长时截断的字符数。
    "max_skeleton_chars_per_candidate": 2000,

    # API 设置。为空则沿用原脚本 API_NAMES。
    "api_names": [],

    # 生成参数。为空则沿用原脚本。
    "temperature": 0.0,
    "max_retries": 2,
    "max_model_timeout": 500,
}


# ============================================================
# 加载原始脚本
# ============================================================

def load_original_module(script_path: str):
    script_path = Path(script_path).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"找不到原始生成脚本: {script_path}")

    if str(script_path.parent) not in sys.path:
        sys.path.insert(0, str(script_path.parent))

    spec = importlib.util.spec_from_file_location("pv_ecot_original_generator", str(script_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MOD = load_original_module(USER_CONFIG["original_script_path"])

API_NAMES = USER_CONFIG["api_names"] or MOD.API_NAMES
TEMPERATURE = USER_CONFIG["temperature"]
MAX_RETRIES = USER_CONFIG["max_retries"]
MAX_MODEL_TIMEOUT = USER_CONFIG["max_model_timeout"]
MAX_SAMPLES_PER_PAPER = int(USER_CONFIG["max_samples_per_paper"])
ENABLE_APPLICATION_ANSWER = bool(USER_CONFIG.get("enable_application_answer", True))
MAX_APPLICATION_SAMPLES_PER_PAPER = int(USER_CONFIG.get("max_application_samples_per_paper", 1))
TOP_CANDIDATES_PER_FAMILY = int(USER_CONFIG["top_candidates_per_family"])
MAX_CANDIDATES_TOTAL = int(USER_CONFIG["max_candidates_total"])
MAX_SKELETON_CHARS_PER_CANDIDATE = int(USER_CONFIG["max_skeleton_chars_per_candidate"])

INPUT_DIR = USER_CONFIG["input_dir"]
OUTPUT_JSONL = USER_CONFIG["output_jsonl"]
DEBUG_JSONL = USER_CONFIG["debug_jsonl"]
DEBUG_BAD_RESP_DIR = USER_CONFIG["bad_response_dir"]
RESUME_MODE = bool(USER_CONFIG["resume_mode"])
OVERWRITE_OUTPUT = bool(USER_CONFIG["overwrite_output"])

os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
os.makedirs(os.path.dirname(DEBUG_JSONL), exist_ok=True)
os.makedirs(DEBUG_BAD_RESP_DIR, exist_ok=True)

file_io_lock = asyncio.Lock()


# ============================================================
# 工具函数
# ============================================================

def normalize_text(x: Any) -> str:
    return MOD.normalize_text(x)


def safe_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', '_', name)


async def save_bad_response(file_name: str, attempt: int, raw_response: str):
    base = safe_filename(os.path.splitext(file_name)[0])
    save_path = os.path.join(DEBUG_BAD_RESP_DIR, f"{base}.attempt{attempt}.txt")
    async with aiofiles.open(save_path, "w", encoding="utf-8") as f:
        await f.write(raw_response or "")


def build_graph(doc: Dict[str, Any]):
    entities = {
        e["entity_id"]: e
        for e in doc.get("entities", [])
        if isinstance(e, dict) and e.get("entity_id")
    }
    relations = [
        r for r in doc.get("relations", [])
        if isinstance(r, dict) and r.get("subject_id") and r.get("object_id")
    ]
    out_edges, in_edges = MOD.build_graph_indices(relations)
    return entities, relations, out_edges, in_edges


def trim_text(text: str, max_chars: int) -> str:
    text = normalize_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " ……"


def build_candidate_meta(
    *,
    doc: Dict[str, Any],
    file_name: str,
    angle: str,
    anchor_score: float,
    anchor: Dict[str, Any],
    entities: Dict[str, Dict[str, Any]],
    relations: List[Dict[str, Any]],
    out_edges,
    in_edges,
    candidate_count_map: Dict[str, int],
    raw_score_map: Dict[str, Any],
) -> Tuple[Dict[str, Any] | None, str]:
    """
    复用原脚本 format_fact_skeleton + sample_pass_main_chain_filter。
    """
    anchor_id = anchor.get("entity_id")
    if not anchor_id:
        return None, "empty_anchor_id"

    skeleton, local_rels, support_relations, support_evidences = MOD.format_fact_skeleton(
        anchor_id, entities, relations, out_edges, in_edges
    )

    if not str(skeleton).strip():
        return None, "empty_fact_skeleton"

    anchor_meta = {
        "anchor_entity_id": anchor_id,
        "anchor_entity_name": normalize_text(anchor.get("entity_name")),
        "anchor_meta_layer": normalize_text(anchor.get("meta_layer")),
        "anchor_score": anchor_score,
        "prompt_family": angle,
        "prompt_variant_id": f"{angle}_multi",
        "angle_candidate_count_map": candidate_count_map,
        "angle_raw_score_map": raw_score_map,
        "angle_score_map": raw_score_map,
        "angle_eligible_families": [k for k, v in candidate_count_map.items() if v > 0],
        "angle_selection_mode": "multi_candidate_llm_judgement",
        "angle_weighted_family_weight_map": {},
        "prompt_family_selection_count_before": 0,
        "prompt_family_selection_count": 0,
        "fact_skeleton": skeleton,
        "supporting_relations": support_relations,
        "supporting_evidences": support_evidences,
        "supporting_relation_count": len(local_rels),
    }

    ok, reason = MOD.sample_pass_main_chain_filter(anchor_meta)
    anchor_meta["local_chain_check"] = {"passed": ok, "reason": reason}

    if not ok:
        return None, reason

    return anchor_meta, "ok"


def collect_candidate_chains(doc: Dict[str, Any], file_name: str, paper_check: Dict[str, Any]):
    """
    每篇论文收集多个可用候选链，但仍使用原脚本的候选打分函数和局部链过滤函数。
    """
    entities, relations, out_edges, in_edges = build_graph(doc)

    if not entities:
        return [], {"reason": "empty_entities"}
    if not relations:
        return [], {"reason": "empty_relations"}

    forward_candidates = MOD.score_forward_candidates(entities, out_edges)
    mechanism_candidates = MOD.score_mechanism_candidates(entities, out_edges, in_edges)
    backward_candidates = MOD.score_backward_candidates(entities, in_edges)
    tradeoff_candidates = MOD.score_tradeoff_candidates(entities, out_edges)

    candidate_map = {
        "Forward": forward_candidates,
        "Mechanism": mechanism_candidates,
        "Backward": backward_candidates,
        "Trade-off": tradeoff_candidates,
    }
    candidate_count_map = {k: len(v) for k, v in candidate_map.items()}
    raw_score_map = {
        k: (v[0][0] if v else None)
        for k, v in candidate_map.items()
    }

    candidate_items = []
    fail_reason_counter = {}

    seen = set()
    for angle, candidates in candidate_map.items():
        for score, anchor in candidates[:TOP_CANDIDATES_PER_FAMILY]:
            anchor_id = anchor.get("entity_id")
            sig = (angle, anchor_id)
            if sig in seen:
                continue
            seen.add(sig)

            meta, reason = build_candidate_meta(
                doc=doc,
                file_name=file_name,
                angle=angle,
                anchor_score=score,
                anchor=anchor,
                entities=entities,
                relations=relations,
                out_edges=out_edges,
                in_edges=in_edges,
                candidate_count_map=candidate_count_map,
                raw_score_map=raw_score_map,
            )
            if meta is None:
                fail_reason_counter[reason] = fail_reason_counter.get(reason, 0) + 1
                continue

            meta["paper_main_chain_check"] = paper_check
            candidate_items.append(meta)

    if not candidate_items:
        return [], {
            "reason": "no_valid_local_candidate",
            "candidate_count_map": candidate_count_map,
            "local_fail_reason_counter": fail_reason_counter,
        }

    # 排序：优先关系数多、分数高的候选；Trade-off 不强行优先，交给 LLM 判断。
    candidate_items.sort(
        key=lambda m: (
            m.get("supporting_relation_count", 0),
            m.get("anchor_score", 0),
        ),
        reverse=True
    )

    candidate_items = candidate_items[:MAX_CANDIDATES_TOTAL]

    for idx, meta in enumerate(candidate_items, start=1):
        meta["candidate_id"] = f"C{idx:02d}"

    return candidate_items, {
        "reason": "ok",
        "candidate_count_map": candidate_count_map,
        "candidate_num_after_filter": len(candidate_items),
        "local_fail_reason_counter": fail_reason_counter,
    }


def format_candidates_for_prompt(candidate_items: List[Dict[str, Any]]) -> str:
    blocks = []
    for meta in candidate_items:
        evidences = meta.get("supporting_evidences", [])[:6]
        evidence_text = "\n".join(f"[E{i+1}] {ev}" for i, ev in enumerate(evidences))

        block = f"""
### Candidate {meta["candidate_id"]}
chain_type: {meta.get("prompt_family", "")}
anchor_entity: {meta.get("anchor_entity_name", "")}
anchor_layer: {meta.get("anchor_meta_layer", "")}
supporting_relation_count: {meta.get("supporting_relation_count", 0)}

fact_skeleton:
{trim_text(meta.get("fact_skeleton", ""), MAX_SKELETON_CHARS_PER_CANDIDATE)}

supporting_evidences:
{evidence_text}
""".strip()
        blocks.append(block)

    return "\n\n".join(blocks)


def build_multi_sample_prompt(doc: Dict[str, Any], file_name: str, candidate_items: List[Dict[str, Any]]) -> str:
    material_sys = normalize_text(doc.get("material_system", "photovoltaic materials"))
    target_problem = normalize_text(doc.get("target_problem", "Unknown"))
    research_focus = normalize_text(doc.get("research_focus", "Unknown"))

    max_samples = min(MAX_SAMPLES_PER_PAPER, len(candidate_items))
    candidate_text = format_candidates_for_prompt(candidate_items)

    application_rule = f"""
# Optional application_answer mode
You may generate at most {MAX_APPLICATION_SAMPLES_PER_PAPER} application_answer sample(s) for this paper.

Use application_answer only when the evidence can support a broader technical/research-assistant question, such as:
- how to construct or implement a photovoltaic/semiconductor device;
- how to design a device structure or fabrication route;
- how to realize charge separation, passivation, transport, light management, or stability improvement;
- how to optimize or characterize a device/system.

Do not use application_answer if the evidence only supports a narrow mechanism conclusion.
""" if ENABLE_APPLICATION_ANSWER else """
# Optional application_answer mode
Do not generate application_answer. Use mechanism_answer only.
"""

    prompt = f"""
You are an expert scientific reasoning dataset writer for photovoltaic, solar-cell, and related semiconductor device papers.

Generate 0 to {max_samples} high-quality Chinese SFT samples from ONE paper.
You must decide the number of samples by yourself.

# Paper information
source_file: {file_name}
material/system: {material_sys}
target_problem: {target_problem}
research_focus: {research_focus}

# Candidate evidence pools
Each candidate is a possible mechanism-reasoning chain extracted from the same paper.
You may choose any candidate, but each generated sample must bind to exactly one candidate_id.
Do not generate more than one sample for the same candidate_id unless the two questions are clearly different, and this is usually discouraged.

{candidate_text}

# How many samples to generate
- Generate at most {max_samples} samples.
- If the paper only supports one strong question, output only one sample.
- If evidence is weak or the candidates are redundant, output fewer samples.
- If none of the candidates can support a natural and useful question, output an empty samples array.
- Do not force the output to contain {max_samples} samples.

# Answer style selection
For each sample, choose exactly one answer_style:

1. mechanism_answer
Use this style when the question asks why a material, interface, process, structure, or parameter affects device performance.
The answer should focus on a compact evidence-grounded mechanism chain.

2. application_answer
Use this style when the question asks how to construct, design, implement, optimize, characterize, or experimentally realize a photovoltaic/semiconductor device or mechanism.
The answer should be a complete research-assistant style explanation, not just a short mechanism conclusion.

{application_rule}

# Sample diversity
If multiple samples are generated, they must focus on different scientific angles:
- Forward: intervention/material/process → mechanism → parameter/performance.
- Mechanism: intermediate structure/mechanism/parameter explains performance.
- Backward: observed performance/result → likely upstream causes.
- Trade-off: supported benefit and cost of the same intervention.
- Application: principle + structure/fabrication + mechanism + characterization/optimization, only when supported.

# Strict grounding
- Use only the selected candidate's evidence pool as paper-specific evidence.
- Do not invent paper-specific materials, mechanisms, values, parameters, device structures, or conclusions.
- Do not merge different device groups as if they were the same case.
- If evidence is indirect, use cautious wording such as “更可能”, “倾向于”, “可能与……有关”.
- You may use general domain knowledge only to organize an application_answer, but you must mark it as general principle rather than paper-specific evidence.
- Do not mention candidate, evidence pool, skeleton, graph, relation, hidden context, extraction, dataset, or prompt in instruction/reasoning/output.

# Output JSON schema
Return exactly one JSON object:
{{
  "samples": [
    {{
      "candidate_id": "C01",
      "cell_type": "silicon_solar_cell",
      "answer_style": "mechanism_answer",
      "instruction": "自然中文科研问题",
      "reasoning_text": "公开分析要点或机制判断依据",
      "output": "最终中文专业回答"
    }}
  ]
}}

# cell_type
Choose exactly one:
silicon_solar_cell, perovskite_solar_cell, organic_solar_cell, dye_sensitized_solar_cell,
thin_film_solar_cell, tandem_solar_cell, hybrid_solar_cell, pv_module_or_system, other, uncertain.

# instruction rules

For mechanism_answer:
- Write one natural Chinese research question, 20-100 Chinese characters.
- Focus on mechanism, parameter change, device performance, or diagnostic reasoning.
- Do not reveal the full reasoning chain.

For application_answer:
- Write one natural Chinese research or technical question, 40-180 Chinese characters.
- The question may ask how to construct, design, implement, optimize, characterize, or explain a device/system.
- It may include 2-4 key entities if needed, such as material, device structure, assembly method, charge separation mechanism, or characterization method.

# reasoning_text rules

For mechanism_answer:
- Write one continuous Chinese paragraph, preferably 180-700 Chinese characters.
- Explain the key mechanism chain using evidence-supported materials, parameters, and performance changes.
- Do not use bullets or headings.
- Do not overclaim.

For application_answer:
- Write a concise public answer plan, preferably 250-900 Chinese characters.
- It should explain what the final answer needs to cover, such as:
  device principle, material/structure choice, fabrication route, charge separation mechanism, characterization evidence, optimization factors, and limitations.
- This is not hidden chain-of-thought. It is a readable expert analysis outline.
- Do not invent unsupported procedures, numerical values, mechanisms, or performance claims.
- If the evidence only supports part of the route, clearly state the boundary.

# output rules

For mechanism_answer:
- Write a natural Chinese professional answer, preferably 150-500 Chinese characters.
- Focus on conclusion + key mechanism + key evidence.
- Do not introduce new numbers, parameters, or mechanisms absent from reasoning_text.
- Do not repeat reasoning_text verbatim.

For application_answer:
- Write a complete Chinese research-assistant style answer, preferably 700-1800 Chinese characters.
- Use clear sections when helpful, such as:
  1. 基本原理
  2. 器件结构
  3. 制备路线
  4. 电荷分离机制
  5. 表征与验证
  6. 优化要点与局限
- The answer should be technically useful, not merely a short mechanism conclusion.
- It may use numbered paragraphs or short bullet points.
- It must remain grounded in the selected candidate evidence.
- Do not fabricate exact experimental parameters, concentrations, temperatures, layer thicknesses, or efficiencies unless they appear in the evidence.
- If general domain knowledge is used to organize the answer, mark it as general principle rather than paper-specific result.
- Avoid overclaiming. Use cautious expressions when evidence is indirect.

# forbidden wording
Do not use internal/meta wording:
fact skeleton, meta_layer, supporting relations, graph, prompt, given data,
候选证据, 证据池, 根据给定内容, 根据证据链, 根据图谱, 主链, 锚点, 实体, 关系, 层级, 隐藏证据, 生成任务.

For mechanism_answer, avoid report-style proof phrases:
证据表明, 数据显示, 实验表明, 研究发现, 这表明, 这说明, 由此可见, 综上.

For application_answer, section titles are allowed, but do not use fake certainty.
""".strip()

    return prompt

def parse_multi_sample_response(raw_text: str) -> List[Dict[str, Any]]:
    data = MOD.try_parse_json_object(raw_text)
    if not isinstance(data, dict):
        return []

    samples = data.get("samples", [])
    if isinstance(samples, dict):
        samples = [samples]
    if not isinstance(samples, list):
        return []

    out = []
    for item in samples:
        if not isinstance(item, dict):
            continue
        answer_style = normalize_text(item.get("answer_style") or "mechanism_answer")
        if answer_style not in {"mechanism_answer", "application_answer"}:
            answer_style = "mechanism_answer"
        if not ENABLE_APPLICATION_ANSWER and answer_style == "application_answer":
            answer_style = "mechanism_answer"

        out.append({
            "candidate_id": normalize_text(item.get("candidate_id")),
            "cell_type": normalize_text(item.get("cell_type")),
            "answer_style": answer_style,
            "instruction": MOD.coerce_text_field(item.get("instruction") or item.get("question")),
            "reasoning_text": MOD.coerce_text_field(item.get("reasoning_text") or item.get("reasoning") or item.get("analysis")),
            "output": MOD.coerce_text_field(item.get("output") or item.get("answer") or item.get("final_answer")),
        })
    return out


def generated_application_sample_pass_filter(reasoning_text: str, output: str, anchor_meta: Dict[str, Any]) -> Tuple[bool, str]:
    """
    应用型长答案过滤。
    与原 mechanism filter 的区别：
    - 允许编号、小标题和分节；
    - 允许更长 output；
    - 仍然禁止内部字段泄漏、空字段、明显截断和过短回答。
    """
    reasoning_text = normalize_text(reasoning_text)
    output = normalize_text(output)

    if not reasoning_text or not output:
        return False, "empty_reasoning_or_output"
    if len(reasoning_text) < 180:
        return False, "application_reasoning_too_short"
    if len(reasoning_text) > 3000:
        return False, "application_reasoning_too_long"
    if len(output) < 500:
        return False, "application_output_too_short"
    if len(output) > 7000:
        return False, "application_output_too_long"

    bad_markers = [
        "L1_Intervention", "L2_Structure_Composition", "L3_Mechanism_Physics",
        "L4_Device_Parameter", "L5_Performance", "fact skeleton", "meta_layer",
        "supporting relations", "anchor entity", "prompt", "given data",
        "candidate evidence", "evidence pool", "根据给定内容", "根据给定信息",
        "根据证据链", "根据图谱", "候选证据", "证据池", "图谱", "主链",
        "锚点", "实体层级", "隐藏证据", "生成任务"
    ]
    joined = f"{reasoning_text}\n{output}".lower()
    for marker in bad_markers:
        if marker.lower() in joined:
            return False, f"meta_language_detected:{marker}"

    bad_endings = ["、", "和", "与", "的", "通过", "由于", "以及", "从而", "因此", "例如", "包括", "主要"]
    if any(output.endswith(x) for x in bad_endings):
        return False, "output_truncated"
    if any(reasoning_text.endswith(x) for x in bad_endings):
        return False, "reasoning_truncated"

    # 应用型样本也不能没有基本 evidence 支撑。
    if anchor_meta.get("supporting_relation_count", 0) < 4:
        return False, "insufficient_supporting_relation_count"

    return True, "ok"


def validate_and_build_records(
    *,
    doc: Dict[str, Any],
    file_name: str,
    used_api_name: str,
    raw_samples: List[Dict[str, Any]],
    candidate_items: List[Dict[str, Any]],
    seen_uids: set,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    candidate_by_id = {m["candidate_id"]: m for m in candidate_items}
    sft_rows = []
    debug_rows = []
    invalid_reasons = []
    used_candidate_ids = set()
    application_sample_count = 0

    for sample_idx, sample in enumerate(raw_samples, start=1):
        candidate_id = sample.get("candidate_id", "")
        answer_style = normalize_text(sample.get("answer_style") or "mechanism_answer")
        if answer_style not in {"mechanism_answer", "application_answer"}:
            answer_style = "mechanism_answer"
        if not ENABLE_APPLICATION_ANSWER and answer_style == "application_answer":
            answer_style = "mechanism_answer"

        if answer_style == "application_answer":
            application_sample_count += 1
            if application_sample_count > MAX_APPLICATION_SAMPLES_PER_PAPER:
                invalid_reasons.append(f"sample{sample_idx}:too_many_application_samples")
                continue

        if candidate_id not in candidate_by_id:
            invalid_reasons.append(f"sample{sample_idx}:invalid_candidate_id:{candidate_id}")
            continue

        # 默认不允许同一 candidate 生成多条，避免重复。
        if candidate_id in used_candidate_ids:
            invalid_reasons.append(f"sample{sample_idx}:duplicate_candidate_id:{candidate_id}")
            continue
        used_candidate_ids.add(candidate_id)

        anchor_meta = dict(candidate_by_id[candidate_id])
        angle = anchor_meta.get("prompt_family", "")

        instruction = normalize_text(sample.get("instruction"))
        inst_ok, inst_reason = MOD.validate_natural_instruction(instruction, angle)
        anchor_meta["instruction_validation"] = inst_reason
        if not inst_ok:
            invalid_reasons.append(f"sample{sample_idx}:instruction_invalid:{inst_reason}")
            continue

        reasoning_text = normalize_text(sample.get("reasoning_text"))
        output = normalize_text(sample.get("output"))
        if answer_style == "application_answer":
            ok, reason = generated_application_sample_pass_filter(reasoning_text, output, anchor_meta)
        else:
            ok, reason = MOD.generated_sample_pass_filter(reasoning_text, output, anchor_meta)

        if not ok:
            invalid_reasons.append(f"sample{sample_idx}:generated_filter_failed:{answer_style}:{reason}")
            continue

        sample_uid = hashlib.md5(
            f"{file_name}||{candidate_id}||{angle}||{anchor_meta.get('anchor_entity_id','')}||{instruction}||{anchor_meta.get('fact_skeleton','')}".encode("utf-8")
        ).hexdigest()

        if sample_uid in seen_uids:
            invalid_reasons.append(f"sample{sample_idx}:duplicate_sample_uid")
            continue

        anchor_meta["sample_uid"] = sample_uid

        auto_annotation = MOD.build_auto_annotation(
            doc=doc,
            file_name=file_name,
            angle=angle,
            anchor_meta=anchor_meta,
            thought_core=reasoning_text,
            thought_enhanced="",
            output=output,
        )

        final_cell_type = MOD.normalize_llm_cell_type(
            sample.get("cell_type", ""),
            fallback=auto_annotation["cell_type_auto"]
        )

        sft_data = {
            "instruction": instruction,
            "input": "",
            "reasoning_text": reasoning_text,
            "output": output,
            "meta_info": {
                "chain_type": angle,
                "cell_type": final_cell_type,
                "sample_type": "application_explanation" if answer_style == "application_answer" else auto_annotation["sample_type_auto"],
                "chain_quality": auto_annotation["chain_grade_auto"],
                "grounded_quality": auto_annotation["grounded_grade_auto"]
            }
        }

        debug_data = {
            "sample_uid": sample_uid,
            "source_file": file_name,
            "used_api_name": used_api_name,
            "multi_sample_version": "v8",
            "candidate_id": candidate_id,
            "sample_index_in_response": sample_idx,
            "total_raw_samples_in_response": len(raw_samples),
            "chain_type": angle,
            "answer_style": answer_style,
            "instruction": instruction,
            "instruction_generation_method": "multi_sample_one_call_generation",
            "instruction_style": "generated_with_answer",
            "instruction_generator_api": used_api_name,
            "instruction_validation": inst_reason,
            "material_system": doc.get("material_system", "Unknown"),
            "target_problem": doc.get("target_problem", "Unknown"),
            "research_focus": doc.get("research_focus", "Unknown"),
            "paper_main_chain_check": anchor_meta.get("paper_main_chain_check", {}),
            "anchor_entity_id": anchor_meta.get("anchor_entity_id", ""),
            "anchor_entity_name": anchor_meta.get("anchor_entity_name", ""),
            "anchor_meta_layer": anchor_meta.get("anchor_meta_layer", ""),
            "anchor_score": anchor_meta.get("anchor_score", ""),
            "local_chain_check": anchor_meta.get("local_chain_check", {}),
            "fact_skeleton": anchor_meta.get("fact_skeleton", ""),
            "supporting_relations": anchor_meta.get("supporting_relations", []),
            "supporting_evidences": anchor_meta.get("supporting_evidences", []),
            "supporting_relation_count": anchor_meta.get("supporting_relation_count", 0),
            "prompt_family": angle,
            "prompt_variant_id": anchor_meta.get("prompt_variant_id", ""),
            "angle_candidate_count_map": anchor_meta.get("angle_candidate_count_map", {}),
            "angle_raw_score_map": anchor_meta.get("angle_raw_score_map", {}),
            "angle_score_map": anchor_meta.get("angle_score_map", {}),
            "angle_eligible_families": anchor_meta.get("angle_eligible_families", []),
            "angle_selection_mode": anchor_meta.get("angle_selection_mode", ""),
            "cell_type_auto": auto_annotation["cell_type_auto"],
            "llm_cell_type_raw": sample.get("cell_type", ""),
            "final_cell_type": final_cell_type,
            "sample_type_auto": auto_annotation["sample_type_auto"],
            "chain_grade_auto": auto_annotation["chain_grade_auto"],
            "grounded_grade_auto": auto_annotation["grounded_grade_auto"],
            "annotation": auto_annotation["annotation"],
        }

        sft_rows.append(sft_data)
        debug_rows.append(debug_data)

    return sft_rows, debug_rows, invalid_reasons


async def process_file_for_multi_cot(file_path, model_pool, seen_uids, seen_source_files, progress_tracker):
    file_name = os.path.basename(file_path)

    if file_name in seen_source_files:
        await progress_tracker.log_finish(file_name, "resumed", "已存在 source_file，断点续传跳过")
        return

    await progress_tracker.log_start(file_name)

    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            doc = json.loads(await f.read())
    except Exception as e:
        logging.error(f"[WARN] [{file_name}] 文件读取或解析失败: {e}")
        await progress_tracker.log_finish(file_name, "failed", f"文件读取失败: {e}")
        return

    paper_ok, paper_check = MOD.paper_has_valid_main_chain(doc)
    if not paper_ok:
        logging.warning(f"⏭ [{file_name}] 主链不达标，跳过。原因: {paper_check}")
        await progress_tracker.log_finish(file_name, "skipped", f"主链不达标: {paper_check}")
        return

    candidate_items, candidate_debug = collect_candidate_chains(doc, file_name, paper_check)
    if not candidate_items:
        logging.warning(f"⏭ [{file_name}] 无可用候选链，跳过。原因: {candidate_debug}")
        await progress_tracker.log_finish(file_name, "skipped", f"无可用候选链: {candidate_debug}")
        return

    prompt = build_multi_sample_prompt(doc, file_name, candidate_items)

    logging.info(
        f" [{file_name}] 候选链={len(candidate_items)} | 最多生成={min(MAX_SAMPLES_PER_PAPER, len(candidate_items))} 条 | 单次请求..."
    )

    last_invalid_reasons = []

    for attempt in range(MAX_RETRIES):
        try:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"请基于候选证据池生成 0 到 {min(MAX_SAMPLES_PER_PAPER, len(candidate_items))} 条中文训练样本 JSON。"}
            ]

            used_api_name, raw_response = await model_pool.call(
                messages=messages,
                temperature=TEMPERATURE,
                timeout=MAX_MODEL_TIMEOUT,
            )

            raw_samples = parse_multi_sample_response(raw_response)
            if not raw_samples:
                await save_bad_response(file_name, attempt + 1, raw_response)
                raise ValueError("模型未输出合法 samples 数组或 samples 为空")

            sft_rows, debug_rows, invalid_reasons = validate_and_build_records(
                doc=doc,
                file_name=file_name,
                used_api_name=used_api_name,
                raw_samples=raw_samples,
                candidate_items=candidate_items,
                seen_uids=seen_uids,
            )
            last_invalid_reasons = invalid_reasons

            if not sft_rows:
                await save_bad_response(file_name, attempt + 1, raw_response)
                raise ValueError(f"所有样本均未通过过滤: {invalid_reasons[:5]}")

            async with file_io_lock:
                # 再检查一次，避免并发期间重复写入。
                if file_name in seen_source_files:
                    await progress_tracker.log_finish(file_name, "resumed", "等待期间 source_file 已写入")
                    return

                async with aiofiles.open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
                    for row in sft_rows:
                        await f.write(json.dumps(row, ensure_ascii=False) + "\n")

                async with aiofiles.open(DEBUG_JSONL, "a", encoding="utf-8") as f:
                    for row in debug_rows:
                        await f.write(json.dumps(row, ensure_ascii=False) + "\n")

                for row in debug_rows:
                    seen_uids.add(row["sample_uid"])
                seen_source_files.add(file_name)

            detail = f"API={used_api_name} | valid_samples={len(sft_rows)} | raw_samples={len(raw_samples)}"
            if invalid_reasons:
                detail += f" | invalid={len(invalid_reasons)}"
            logging.info(f" [{file_name}] 成功生成多条 SFT 数据。{detail}")
            await progress_tracker.log_finish(file_name, "success", detail)
            return

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                logging.warning(f"[WARN] [{file_name}] 第 {attempt + 1} 次失败: {e}，准备重试...")
                await asyncio.sleep(2 ** attempt)
            else:
                logging.error(f"[ERROR] [{file_name}] 彻底失败，丢弃该论文: {e}")
                await progress_tracker.log_finish(file_name, "failed", f"最终失败: {e} | invalid={last_invalid_reasons[:3]}")
                return


async def main():
    print("\n 启动【PV-ECoT 多样本生成 V8：机制答案 + 应用长答案】...\n")

    if not os.path.exists(INPUT_DIR):
        print(f"[ERROR] 找不到输入目录: {INPUT_DIR}")
        return

    json_files = sorted(
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.endswith(".json")
    )

    if not json_files:
        print("[WARN] 输入目录中没有 JSON 文件。")
        return

    if not RESUME_MODE and OVERWRITE_OUTPUT:
        if os.path.exists(OUTPUT_JSONL):
            os.remove(OUTPUT_JSONL)
        if os.path.exists(DEBUG_JSONL):
            os.remove(DEBUG_JSONL)

    # 复用原脚本的断点续传扫描逻辑。
    if RESUME_MODE:
        seen_uids, seen_source_files, existing_count = MOD.load_existing_output(OUTPUT_JSONL, DEBUG_JSONL)
    else:
        seen_uids, seen_source_files, existing_count = set(), set(), 0

    pending_files = [fp for fp in json_files if os.path.basename(fp) not in seen_source_files]

    print(f" 发现 {len(json_files)} 篇 refined JSON。")
    print(f"♻ 断点续传模式: {'开启' if RESUME_MODE else '关闭'}")
    print(f" 已有样本数: {existing_count} | 已完成篇数: {len(seen_source_files)} | 待处理篇数: {len(pending_files)}")
    print(f" API 列表: {', '.join(API_NAMES)}")
    print(f" 每篇最多生成: {MAX_SAMPLES_PER_PAPER} 条 | 每篇最多候选链: {MAX_CANDIDATES_TOTAL}")
    print(f" application_answer: {'开启' if ENABLE_APPLICATION_ANSWER else '关闭'} | 每篇最多 {MAX_APPLICATION_SAMPLES_PER_PAPER} 条")

    if not pending_files:
        print(" 没有待处理文件，任务已完成。")
        return

    model_pool = MOD.MultiAPIModelPool(API_NAMES)
    progress_tracker = MOD.ProgressTracker(len(pending_files))

    tasks = [
        asyncio.create_task(process_file_for_multi_cot(fp, model_pool, seen_uids, seen_source_files, progress_tracker))
        for fp in pending_files
    ]
    await asyncio.gather(*tasks)

    print("\n" + "=" * 60)
    print(" 多样本数据集生成完成！")
    print(f" 训练输出文件: {OUTPUT_JSONL}")
    print(f" 调试追溯文件: {DEBUG_JSONL}")
    print(f" 非法输出调试目录: {DEBUG_BAD_RESP_DIR}")
    print(
        f" 本轮待处理: {len(pending_files)} | "
        f"成功={progress_tracker.success} 跳过={progress_tracker.skipped} "
        f"失败={progress_tracker.failed} 续跑跳过={progress_tracker.resumed}"
    )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    asyncio.run(main())
