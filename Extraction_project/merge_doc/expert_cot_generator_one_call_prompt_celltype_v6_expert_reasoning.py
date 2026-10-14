
import asyncio
import logging
import os
import json
import re
import sys
import hashlib

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')
from collections import defaultdict, deque
from typing import Dict, List, Any, Tuple, Optional

import aiofiles

script_dir = os.path.dirname(os.path.abspath(__file__))
# merge_doc -> Extraction_project -> pythonProject2
PROJECT_ROOT = os.path.dirname(os.path.dirname(script_dir))
EXTRACTION_ROOT = os.path.dirname(script_dir)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from Extraction_project.llm_api.model_api_factory import get_model_api
    from Extraction_project.config import API_CONFIG
except ImportError as e:
    print(f"导入失败! 路径: {PROJECT_ROOT} | 错误: {e}")
    raise

INPUT_DIR = os.getenv("PV_REFINED_RESULTS_DIR", os.path.join(PROJECT_ROOT, "outputs", "refined_results"))
OUTPUT_JSONL = os.getenv("PV_BASE_INSTRUCTIONS_JSONL", os.path.join(PROJECT_ROOT, "outputs", "generation", "base_instructions.jsonl"))
DEBUG_JSONL = os.getenv("PV_BASE_DEBUG_JSONL", os.path.join(PROJECT_ROOT, "outputs", "generation", "base_debug.jsonl"))
DEBUG_BAD_RESP_DIR = os.getenv("PV_BAD_RESPONSES_DIR", os.path.join(PROJECT_ROOT, "outputs", "generation", "bad_responses"))

# 支持多个 API 名称，按轮询方式调度；只有一个时也可正常工作
API_NAMES = [
    "silicon_deepseek_r1_0528"
]
MAX_RETRIES = 2
TEMPERATURE = 0.0
MAX_MODEL_TIMEOUT = 500

# 单次调用：一次 LLM 调用同时生成 instruction / reasoning_text / output，避免自然问题生成与回答生成两次串行请求。
USE_ONE_CALL_SAMPLE_GENERATION = True
# 兼容开关：如需退回旧版“两次调用”，可设为 False；默认不建议。
USE_LLM_INSTRUCTION_GENERATOR = False
FALLBACK_TO_TEMPLATE_INSTRUCTION = True
INSTRUCTION_TEMPERATURE = 0.7

# cell_type 由模型输出优先决定；非太阳电池/薄膜电子器件等统一归为 other。
ALLOWED_CELL_TYPES = {
    "silicon_solar_cell",
    "perovskite_solar_cell",
    "organic_solar_cell",
    "dye_sensitized_solar_cell",
    "thin_film_solar_cell",
    "tandem_solar_cell",
    "hybrid_solar_cell",
    "pv_module_or_system",
    "other",
    "uncertain",
}


# 断点续传：True 时保留已有输出并跳过已完成样本；False 时从头开始
RESUME_MODE = True
# 仅在 RESUME_MODE=False 时生效
OVERWRITE_OUTPUT = False

MAX_CONCURRENCY_PER_API = {
    api_name: API_CONFIG.get(api_name, {}).get("concurrency", 3)
    for api_name in API_NAMES
}
MAX_TOTAL_CONCURRENCY = sum(MAX_CONCURRENCY_PER_API.values()) if MAX_CONCURRENCY_PER_API else 1

os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
os.makedirs(DEBUG_BAD_RESP_DIR, exist_ok=True)
file_io_lock = asyncio.Lock()

def safe_filename(name: str) -> str:
    return re.sub(r'[\/:*?"<>|]+', '_', name)


def load_existing_output(output_jsonl: str, debug_jsonl: str = "") -> tuple[set[str], set[str], int]:
    """
    断点续传索引读取函数。
    新版训练文件只保留极简 meta_info，因此 sample_uid/source_file/prompt_family 等追溯字段
    主要从 debug_jsonl 读取；如果 debug_jsonl 不存在，则兼容读取旧版 output_jsonl。
    """
    seen_uids: set[str] = set()
    seen_source_files: set[str] = set()
    line_count = 0

    def _scan(path: str, prefer_debug_payload: bool = False) -> int:
        local_count = 0
        if not path or not os.path.exists(path):
            return 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                local_count += 1

                # debug 文件里字段通常在顶层；旧版训练文件在 meta_info 中。
                meta = obj.get('meta_info', {}) if isinstance(obj.get('meta_info'), dict) else {}
                uid = str(obj.get('sample_uid') or meta.get('sample_uid') or '').strip()
                src = str(obj.get('source_file') or meta.get('source_file') or '').strip()
                fam = str(obj.get('prompt_family') or meta.get('prompt_family') or meta.get('chain_type') or '').strip()

                if uid:
                    seen_uids.add(uid)
                if src:
                    seen_source_files.add(src)
                if fam:
                    ANGLE_SELECTION_COUNTS[fam] += 1
                    FAMILY_VARIANT_COUNTERS[fam] += 1
        return local_count

    # 优先读 debug，因为新版 train 文件不再保存 sample_uid/source_file。
    line_count += _scan(debug_jsonl, prefer_debug_payload=True)

    # 兼容旧版：如果没有 debug，继续从旧 train/output 中恢复索引。
    if not seen_uids and not seen_source_files:
        line_count += _scan(output_jsonl, prefer_debug_payload=False)

    return seen_uids, seen_source_files, line_count


class MultiAPIModelPool:
    def __init__(self, api_names: list[str]):
        if not api_names:
            raise ValueError('API_NAMES cannot be empty')
        self.api_names = api_names
        self.clients = {name: get_model_api(name) for name in api_names}
        self.semaphores = {
            name: asyncio.Semaphore(MAX_CONCURRENCY_PER_API.get(name, 3))
            for name in api_names
        }
        self._rr_index = 0
        self._rr_lock = asyncio.Lock()

    async def _next_api_name(self) -> str:
        async with self._rr_lock:
            name = self.api_names[self._rr_index % len(self.api_names)]
            self._rr_index += 1
            return name

    async def call(self, messages: list[dict[str, str]], temperature: float, timeout: int) -> tuple[str, str]:
        api_name = await self._next_api_name()
        async with self.semaphores[api_name]:
            response = await asyncio.wait_for(
                self.clients[api_name].call(
                    messages=messages,
                    temperature=temperature,
                    timeout=timeout,
                ),
                timeout=timeout + 20,
            )
        return api_name, response


class ProgressTracker:
    def __init__(self, total: int):
        self.total = total
        self.started = 0
        self.finished = 0
        self.success = 0
        self.skipped = 0
        self.failed = 0
        self.resumed = 0
        self._lock = asyncio.Lock()

    async def log_start(self, file_name: str):
        async with self._lock:
            self.started += 1
            logging.info(
                f" [{self.started}/{self.total}] 开始处理: {file_name} | "
                f"已完成={self.finished} 成功={self.success} 跳过={self.skipped} 失败={self.failed} 续跑跳过={self.resumed}"
            )

    async def log_finish(self, file_name: str, status: str, detail: str = ""):
        async with self._lock:
            self.finished += 1
            if status == 'success':
                self.success += 1
                emoji = ''
            elif status == 'failed':
                self.failed += 1
                emoji = '[ERROR]'
            elif status == 'resumed':
                self.resumed += 1
                emoji = '⏭'
            else:
                self.skipped += 1
                emoji = '⏭'
            summary = (
                f"{emoji} 进度 [{self.finished}/{self.total}] {file_name} | "
                f"成功={self.success} 跳过={self.skipped} 失败={self.failed} 续跑跳过={self.resumed}"
            )
            if detail:
                summary += f" | {detail}"
            logging.info(summary)


POS_PREDS = {"improves", "enhances", "increases", "boosts", "raises"}
NEG_PREDS = {"reduces", "decreases", "suppresses", "worsens", "limits", "induces loss"}


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_key(text: Any) -> str:
    return normalize_text(text).lower()


def extract_numeric_values_for_consistency(text: Any) -> set[str]:
    """抽取用于一致性检查的具体数值。
    目标是防止 output 中突然出现 reasoning_text 没有用过的关键数字。
    会跳过 WS2、2D/3D 等化学式或维度标记，避免误伤。
    """
    s = normalize_text(text)
    if not s:
        return set()

    unit_pat = (
        r"%|％|mV|V|mA\s*(?:/|·)?\s*cm(?:\^?-?2|⁻²|-2|²)?|"
        r"A\s*(?:/|·)?\s*cm(?:\^?-?2|⁻²|-2|²)?|Ω|ohm|nm|μm|um|mm|cm|"
        r"ms|s|h|小时|min|分钟|°C|℃|K|cm²/V·s|cm2/Vs|mA·cm⁻²|mA/cm²"
    )
    values: set[str] = set()

    # 带单位或百分号的数字。
    for m in re.finditer(rf"(?<![A-Za-z])([-+]?\d+(?:\.\d+)?)\s*(?:{unit_pat})", s, flags=re.IGNORECASE):
        raw = m.group(1)
        try:
            values.add(f"{float(raw):.8g}")
        except Exception:
            values.add(raw)

    # 裸数字：要求不是化学式/2D/变量名的一部分；主要捕获 16.6、0.80 这类。
    for m in re.finditer(r"(?<![A-Za-z])([-+]?\d+(?:\.\d+)?)(?![A-Za-z])", s):
        start, end = m.span()
        # 跳过已经被带单位规则覆盖的单位后缀附近数字、2D/3D、WS2 等。
        prev_ch = s[start - 1] if start > 0 else ""
        next_ch = s[end] if end < len(s) else ""
        if prev_ch.isalpha() or next_ch.isalpha():
            continue
        raw = m.group(1)
        # 跳过单独的 1/2/3 等常见维度或序号残留；小数和两位以上整数保留。
        if re.fullmatch(r"[+-]?[0-9]", raw):
            continue
        try:
            values.add(f"{float(raw):.8g}")
        except Exception:
            values.add(raw)
    return values


PARAMETER_ALIASES = {
    "pce": ["pce", "效率", "转换效率", "光电转换效率", "power conversion efficiency"],
    "voc": ["voc", "v_oc", "开路电压", "open-circuit voltage"],
    "jsc": ["jsc", "j_sc", "短路电流", "短路电流密度", "current density", "short-circuit current"],
    "ff": ["ff", "填充因子", "fill factor"],
    "stability": ["稳定性", "保持率", "retention", "稳定运行", "storage", "aging"],
    "lifetime": ["寿命", "载流子寿命", "lifetime"],
    "reflectance": ["反射率", "reflectance", "reflection"],
    "mobility": ["迁移率", "mobility"],
    "roughness": ["粗糙度", "roughness", "rq", "rms"],
    "contact_angle": ["接触角", "contact angle"],
    "resistance": ["电阻", "rct", "rs", "resistance"],
    "capacitance": ["电容", "capacitance"],
    "vbi": ["内建电势", "内建电压", "vbi", "built-in potential"],
    "depletion_width": ["耗尽层", "depletion"],
    "j01": ["j01", "饱和电流密度", "saturation current"],
}


def extract_parameter_keys_for_consistency(text: Any) -> set[str]:
    """抽取少量关键器件/材料参数概念，用于避免 output 新增 reasoning_text 未使用的参数。"""
    low = normalize_text(text).lower()
    found: set[str] = set()
    for key, aliases in PARAMETER_ALIASES.items():
        for alias in aliases:
            if alias.lower() in low:
                found.add(key)
                break
    return found


def coerce_text_field(value: Any) -> str:
    """
    把模型输出里的 thought/output 统一收敛成字符串。
    支持:
    - str
    - list[str]
    - list[dict(text/content)]
    - dict(text/content)
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return normalize_text(value)

    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                t = normalize_text(item)
                if t:
                    parts.append(t)
            elif isinstance(item, dict):
                t = normalize_text(item.get("text") or item.get("content") or "")
                if t:
                    parts.append(t)
            else:
                t = normalize_text(item)
                if t:
                    parts.append(t)
        return "\n".join(parts).strip()

    if isinstance(value, dict):
        return normalize_text(value.get("text") or value.get("content") or "")

    return normalize_text(value)


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


def try_parse_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    try:
        clean = extract_json_block(raw_text)
        clean = clean.replace("\r", " ")
        return json.loads(clean, strict=False)
    except Exception:
        return None


def try_recover_thought_output(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text:
        return None
    text = raw_text.strip()

    thought_match = re.search(
        r'"thought"\s*[:：]\s*"(?P<thought>(?:\\.|[^"\\])*)"',
        text,
        flags=re.DOTALL,
    )
    output_match = re.search(
        r'"output"\s*[:：]\s*"(?P<output>(?:\\.|[^"\\])*)"',
        text,
        flags=re.DOTALL,
    )
    if thought_match and output_match:
        thought_raw = thought_match.group("thought")
        output_raw = output_match.group("output")
        try:
            thought = json.loads(f'"{thought_raw}"')
            output = json.loads(f'"{output_raw}"')
        except Exception:
            thought = thought_raw.replace("\\n", "\n").replace('\\"', '"')
            output = output_raw.replace("\\n", "\n").replace('\\"', '"')
        return {"thought": thought, "output": output}

    thought_md = re.search(
        r'(?:thought|思考|推理)\s*[:：]\s*(.+?)(?:\n\s*(?:output|回答|答案|最终回答)\s*[:：])',
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    output_md = re.search(
        r'(?:output|回答|答案|最终回答)\s*[:：]\s*(.+)$',
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if thought_md and output_md:
        return {
            "thought": normalize_text(thought_md.group(1)),
            "output": normalize_text(output_md.group(1)),
        }
    return None


def parse_model_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    解析 teacher 输出。新版优先使用自然推理字段：
    - reasoning_text: 自然语言推理段落，不使用 1/2/3 编号清单
    - output: 最终回答

    兼容旧版字段：thought_core / thought_enhanced / thought / reasoning / analysis / cot。
    这样旧模型偶尔返回旧键名时，不会直接导致整篇失败。
    """
    data = try_parse_json_object(raw_text)
    if not isinstance(data, dict):
        data = try_recover_thought_output(raw_text)
    if not isinstance(data, dict):
        return None

    reasoning_text = coerce_text_field(
        data.get("reasoning_text")
        if data.get("reasoning_text") is not None
        else data.get("reasoning")
        if data.get("reasoning") is not None
        else data.get("analysis")
        if data.get("analysis") is not None
        else data.get("cot")
    )

    # 兼容旧版：如果还返回 thought_core/thought_enhanced，则拼成自然推理文本。
    if not reasoning_text:
        thought_core = coerce_text_field(data.get("thought_core") or data.get("thought"))
        thought_enhanced = coerce_text_field(data.get("thought_enhanced") or data.get("enhanced_thought") or data.get("thought_plus"))
        reasoning_text = normalize_text("\n".join([x for x in [thought_core, thought_enhanced] if x]))

    output = coerce_text_field(
        data.get("output")
        if data.get("output") is not None
        else data.get("answer") or data.get("final_answer") or data.get("response")
    )

    instruction = coerce_text_field(
        data.get("instruction")
        if data.get("instruction") is not None
        else data.get("question") or data.get("user_question")
    )

    cell_type = normalize_text(data.get("cell_type"))

    if not reasoning_text or not output:
        return None

    result = {
        "reasoning_text": reasoning_text,
        "output": output
    }
    if instruction:
        result["instruction"] = instruction
    if cell_type:
        result["cell_type"] = cell_type
    return result


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def infer_cell_type(material_system: str, source_file: str = "") -> str:
    text = f"{material_system} {source_file}".lower()
    if any(k in text for k in ["perovskite", "cspbbr3", "mapbi", "fapbi", "pbbr"]):
        if any(k in text for k in ["bhj", "bulk-heterojunction", "hybrid", "integrated"]):
            return "hybrid_solar_cell"
        return "perovskite_solar_cell"
    if any(k in text for k in ["organic", "polymer", "bhj", "bulk-heterojunction", "it ic", "itc", "j61-itic"]):
        return "organic_solar_cell"
    if any(k in text for k in ["silicon", "si", "mc-si", "pc-si", "μc-si", "microcrystalline", "polycrystalline", "multicrystalline", "heterojunction"]):
        return "silicon_solar_cell"
    if "hybrid" in text:
        return "hybrid_solar_cell"
    return "other_solar_cell"


def normalize_llm_cell_type(cell_type: Any, fallback: str = "uncertain") -> str:
    """把模型返回的 cell_type 收敛到允许集合。
    这里不再新增复杂类型；非太阳电池、薄膜晶体管、薄膜电子器件等统一使用 other。
    """
    ct = normalize_text(cell_type).lower().replace("-", "_").replace(" ", "_")

    alias_map = {
        "other_solar_cell": "other",
        "non_pv_device": "other",
        "non_solar_device": "other",
        "non_solar_thin_film_device": "other",
        "thin_film_electronics": "other",
        "thin_film_electronic_device": "other",
        "thin_film_transistor": "other",
        "tft": "other",
        "unknown": "uncertain",
        "pv_module": "pv_module_or_system",
        "module_system": "pv_module_or_system",
        "pv_system": "pv_module_or_system",
        "dssc": "dye_sensitized_solar_cell",
    }
    ct = alias_map.get(ct, ct)

    if ct in ALLOWED_CELL_TYPES:
        return ct

    fb = normalize_text(fallback).lower().replace("-", "_").replace(" ", "_")
    fb = alias_map.get(fb, fb)
    if fb in ALLOWED_CELL_TYPES:
        return fb
    return "uncertain"


def infer_sample_type(angle: str, source_file: str = "", research_focus: str = "") -> str:
    text = f"{source_file} {research_focus}".lower()
    if any(k in text for k in [
        "modeling", "modelling", "simulation", "simulator", "analysis", "visualization",
        "visualisation", "characterization", "characterisation", "hfss", "fdtd",
        "fem", "fit", "rcwa", "mom", "finite element"
    ]):
        if any(k in text for k in ["characterization", "characterisation", "visualization", "visualisation"]):
            return "characterization_analysis"
        return "modeling_analysis"

    angle_key = normalize_text(angle)
    if angle_key == "Mechanism":
        return "mechanism_explanation"
    if angle_key == "Backward":
        return "mechanism_explanation"
    return "device_improvement"


def infer_chain_grade_auto(
    paper_check: Dict[str, Any],
    local_chain_check: Dict[str, Any],
    supporting_relation_count: int
) -> str:
    if paper_check.get("reason") == "ok" and local_chain_check.get("passed") and supporting_relation_count >= 8:
        return "A"
    if paper_check.get("reason") == "ok" and local_chain_check.get("passed") and supporting_relation_count >= 4:
        return "B"
    return "C"


def infer_grounded_grade_auto(
    support_evidences: List[str],
    supporting_relation_count: int,
    thought: str,
    output: str
) -> str:
    joined = f"{thought}\n{output}".lower()
    bad_markers = ["根据提供的骨架", "fact skeleton", "given data", "prompt", "graph"]
    if any(m.lower() in joined for m in bad_markers):
        return "C"
    if len(support_evidences) >= 3 and supporting_relation_count >= 6:
        return "A"
    if len(support_evidences) >= 1 and supporting_relation_count >= 4:
        return "B"
    return "C"


def build_auto_annotation(
    doc: Dict[str, Any],
    file_name: str,
    angle: str,
    anchor_meta: Dict[str, Any],
    thought_core: str = "",
    thought_enhanced: str = "",
    output: str = ""
) -> Dict[str, Any]:
    cell_type = infer_cell_type(
        normalize_text(doc.get("material_system", "")),
        file_name
    )
    sample_type = infer_sample_type(
        angle=angle,
        source_file=file_name,
        research_focus=normalize_text(doc.get("research_focus", ""))
    )
    chain_grade_auto = infer_chain_grade_auto(
        paper_check=anchor_meta.get("paper_main_chain_check", {}),
        local_chain_check=anchor_meta.get("local_chain_check", {}),
        supporting_relation_count=anchor_meta.get("supporting_relation_count", 0)
    )
    grounded_grade_auto = infer_grounded_grade_auto(
        support_evidences=anchor_meta.get("supporting_evidences", []),
        supporting_relation_count=anchor_meta.get("supporting_relation_count", 0),
        thought=f"{thought_core}\n{thought_enhanced}",
        output=output
    )
    return {
        "cell_type_auto": cell_type,
        "sample_type_auto": sample_type,
        "chain_grade_auto": chain_grade_auto,
        "grounded_grade_auto": grounded_grade_auto,
        "annotation": {
            "cell_type": "",
            "chain_grade": "",
            "grounded_grade": "",
            "sample_type": "",
            "note": ""
        }
    }



def get_relation_predicate(rel: Dict[str, Any]) -> str:
    return normalize_text(rel.get("raw_predicate") or rel.get("predicate"))


def get_relation_evidences(rel: Dict[str, Any]) -> List[str]:
    evidences = rel.get("evidences", rel.get("evidence", []))
    if isinstance(evidences, str):
        evidences = [evidences]
    return [normalize_text(x) for x in evidences if normalize_text(x)]


def relation_is_positive(rel: Dict[str, Any]) -> bool:
    pred = normalize_key(get_relation_predicate(rel))
    group = normalize_key(rel.get("predicate_group"))
    return group in {"positive_effect", "passivation_effect", "transport_effect", "stability_effect"} or pred in POS_PREDS


def relation_is_negative(rel: Dict[str, Any]) -> bool:
    pred = normalize_key(get_relation_predicate(rel))
    group = normalize_key(rel.get("predicate_group"))
    return group == "negative_effect" or pred in NEG_PREDS


def serialize_supporting_relations(local_rels, entities, max_evidences_per_relation=2):
    payload = []
    for rel in local_rels:
        sub = entities.get(rel.get("subject_id"))
        obj = entities.get(rel.get("object_id"))
        if not sub or not obj:
            continue
        payload.append({
            "subject_id": rel.get("subject_id"),
            "subject_name": normalize_text(sub.get("entity_name")),
            "subject_layer": normalize_text(sub.get("meta_layer")),
            "raw_predicate": get_relation_predicate(rel),
            "predicate_group": normalize_text(rel.get("predicate_group")),
            "object_id": rel.get("object_id"),
            "object_name": normalize_text(obj.get("entity_name")),
            "object_layer": normalize_text(obj.get("meta_layer")),
            "evidences": get_relation_evidences(rel)[:max_evidences_per_relation],
        })
    return payload


def collect_supporting_evidences(local_rels, max_items=8):
    seen = set()
    out = []
    for rel in local_rels:
        for ev in get_relation_evidences(rel):
            if ev and ev not in seen:
                seen.add(ev)
                out.append(ev)
            if len(out) >= max_items:
                return out
    return out


def build_graph_indices(relations):
    out_edges = defaultdict(list)
    in_edges = defaultdict(list)
    for r in relations:
        sub = r.get("subject_id")
        obj = r.get("object_id")
        if not sub or not obj:
            continue
        out_edges[sub].append(r)
        in_edges[obj].append(r)
    return out_edges, in_edges


def relation_evidence_snippet(rel, max_len=180):
    evidences = get_relation_evidences(rel)
    if not evidences:
        return ""
    return evidences[0][:max_len]


def has_path_to_meta_layer(start_id, target_layer, entities, out_edges, max_depth=4):
    visited = set()
    queue = deque([(start_id, 0)])
    while queue:
        node_id, depth = queue.popleft()
        if depth > max_depth or node_id in visited:
            continue
        visited.add(node_id)
        if depth > 0:
            ent = entities.get(node_id)
            if ent and ent.get("meta_layer") == target_layer:
                return True
        for rel in out_edges.get(node_id, []):
            nxt = rel.get("object_id")
            if nxt and nxt not in visited:
                queue.append((nxt, depth + 1))
    return False


def count_reachable_meta_layer(start_id, target_layer, entities, out_edges, max_depth=4):
    visited = set()
    queue = deque([(start_id, 0)])
    hits = set()
    while queue:
        node_id, depth = queue.popleft()
        if depth > max_depth or (node_id, depth) in visited:
            continue
        visited.add((node_id, depth))
        if depth > 0:
            ent = entities.get(node_id)
            if ent and ent.get("meta_layer") == target_layer:
                hits.add(node_id)
        for rel in out_edges.get(node_id, []):
            nxt = rel.get("object_id")
            if nxt:
                queue.append((nxt, depth + 1))
    return len(hits)


def has_upstream_intervention(start_id, entities, in_edges, max_depth=4):
    visited = set()
    queue = deque([(start_id, 0)])
    while queue:
        node_id, depth = queue.popleft()
        if depth > max_depth or node_id in visited:
            continue
        visited.add(node_id)
        if depth > 0:
            ent = entities.get(node_id)
            if ent and ent.get("meta_layer") == "L1_Intervention":
                return True
        for rel in in_edges.get(node_id, []):
            prev_node = rel.get("subject_id")
            if prev_node and prev_node not in visited:
                queue.append((prev_node, depth + 1))
    return False


def path_contains_middle_layer(start_id, entities, out_edges, max_depth=4):
    visited = set()
    queue = deque([(start_id, 0, False)])
    while queue:
        node_id, depth, seen_middle = queue.popleft()
        if depth > max_depth:
            continue
        state_key = (node_id, depth, seen_middle)
        if state_key in visited:
            continue
        visited.add(state_key)

        if depth > 0:
            ent = entities.get(node_id)
            if ent:
                layer = normalize_text(ent.get("meta_layer"))
                if layer in {"L2_Structure_Composition", "L3_Mechanism_Physics"}:
                    seen_middle = True
                if layer in {"L4_Device_Parameter", "L5_Performance"} and seen_middle:
                    return True

        for rel in out_edges.get(node_id, []):
            nxt = rel.get("object_id")
            if nxt:
                queue.append((nxt, depth + 1, seen_middle))
    return False


def paper_has_valid_main_chain(doc):
    entities = {
        e["entity_id"]: e for e in doc.get("entities", [])
        if isinstance(e, dict) and e.get("entity_id")
    }
    relations = [
        r for r in doc.get("relations", [])
        if isinstance(r, dict) and r.get("subject_id") and r.get("object_id")
    ]
    if not entities or not relations:
        return False, {"reason": "empty_entities_or_relations"}

    out_edges, _ = build_graph_indices(relations)
    l1_nodes = [eid for eid, ent in entities.items() if normalize_text(ent.get("meta_layer")) == "L1_Intervention"]
    if not l1_nodes:
        return False, {"reason": "no_L1_intervention"}

    direct_pass = False
    middle_pass = False
    for eid in l1_nodes:
        if (
            has_path_to_meta_layer(eid, "L5_Performance", entities, out_edges, max_depth=4)
            or has_path_to_meta_layer(eid, "L4_Device_Parameter", entities, out_edges, max_depth=4)
        ):
            direct_pass = True
        if path_contains_middle_layer(eid, entities, out_edges, max_depth=4):
            middle_pass = True

    passed = direct_pass and middle_pass
    return passed, {
        "reason": "ok" if passed else "main_chain_not_found",
        "l1_count": len(l1_nodes),
        "has_L1_to_L4_or_L5": direct_pass,
        "has_middle_supported_chain": middle_pass,
    }


def collect_local_subgraph_relations(anchor_id, entities, out_edges, in_edges, max_depth=2, max_edges=18):
    visited_nodes = {anchor_id}
    queue = deque([(anchor_id, 0)])
    collected = []

    while queue and len(collected) < max_edges:
        node_id, depth = queue.popleft()
        if depth > max_depth:
            continue

        for rel in out_edges.get(node_id, []):
            collected.append(rel)
            nxt = rel.get("object_id")
            if nxt and nxt not in visited_nodes:
                visited_nodes.add(nxt)
                queue.append((nxt, depth + 1))
            if len(collected) >= max_edges:
                break

        if len(collected) >= max_edges:
            break

        for rel in in_edges.get(node_id, []):
            collected.append(rel)
            prev_node = rel.get("subject_id")
            if prev_node and prev_node not in visited_nodes:
                visited_nodes.add(prev_node)
                queue.append((prev_node, depth + 1))
            if len(collected) >= max_edges:
                break

    seen = set()
    uniq = []
    for rel in collected:
        sig = (rel.get("subject_id"), get_relation_predicate(rel), rel.get("object_id"))
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(rel)
    return uniq[:max_edges]


def format_fact_skeleton(anchor_id, entities, relations, out_edges, in_edges):
    local_rels = collect_local_subgraph_relations(anchor_id, entities, out_edges, in_edges)
    fact_lines = []

    for rel in local_rels:
        sub = entities.get(rel.get("subject_id"))
        obj = entities.get(rel.get("object_id"))
        if not sub or not obj:
            continue
        sub_layer = normalize_text(sub.get("meta_layer", "Unknown"))
        obj_layer = normalize_text(obj.get("meta_layer", "Unknown"))
        sub_name = normalize_text(sub.get("entity_name"))
        obj_name = normalize_text(obj.get("entity_name"))
        predicate = get_relation_predicate(rel)
        ev = relation_evidence_snippet(rel)
        line = f"[{sub_layer}] {sub_name} -> ({predicate}) -> [{obj_layer}] {obj_name}"
        if ev:
            line += f" | evidence: {ev}"
        fact_lines.append(line)

    fact_lines = list(dict.fromkeys(fact_lines))
    support_relations = serialize_supporting_relations(local_rels, entities)
    support_evidences = collect_supporting_evidences(local_rels)
    return "\n".join(fact_lines[:18]), local_rels, support_relations, support_evidences


def sample_pass_main_chain_filter(anchor_meta):
    support_relations = anchor_meta.get("supporting_relations", [])
    support_evidences = anchor_meta.get("supporting_evidences", [])
    fact_skeleton = normalize_text(anchor_meta.get("fact_skeleton"))

    if len(support_relations) < 4:
        return False, "too_few_supporting_relations"
    if len(support_evidences) < 1:
        return False, "no_supporting_evidence"
    if len(fact_skeleton) < 80:
        return False, "fact_skeleton_too_short"

    layers = set()
    for rel in support_relations:
        layers.add(normalize_text(rel.get("subject_layer")))
        layers.add(normalize_text(rel.get("object_layer")))

    has_l1 = "L1_Intervention" in layers
    has_middle = bool({"L2_Structure_Composition", "L3_Mechanism_Physics"} & layers)
    has_result = bool({"L4_Device_Parameter", "L5_Performance"} & layers)

    if not has_l1:
        return False, "no_L1_in_local_chain"
    if not has_middle:
        return False, "no_L2_or_L3_in_local_chain"
    if not has_result:
        return False, "no_L4_or_L5_in_local_chain"

    return True, "ok"


def generated_sample_pass_filter(reasoning_text, output, anchor_meta):
    """过滤自然推理样本，重点避免模板化编号 CoT 和元语言泄漏。"""
    reasoning_text = normalize_text(reasoning_text)
    output = normalize_text(output)

    if not reasoning_text or not output:
        return False, "empty_reasoning_or_output"
    if len(reasoning_text) < 120:
        return False, "reasoning_too_short"
    if len(reasoning_text) > 3500:
        return False, "reasoning_too_long"
    if len(output) < 80:
        return False, "output_too_short"
    if len(output) > 4000:
        return False, "output_too_long"

    bad_markers = [
        # 结构化输入/工程字段泄漏
        "L1_Intervention", "L2_Structure_Composition", "L3_Mechanism_Physics",
        "L4_Device_Parameter", "L5_Performance", "fact skeleton", "meta_layer",
        "supporting relations", "anchor entity", "anchor layer", "prompt", "given data",
        # 中文构造痕迹
        "根据提供的骨架", "证据骨架", "上述骨架", "根据给定内容", "根据给定信息",
        "根据证据链", "论文中的证据链", "基于证据链", "知识图谱显示",
        "从图谱可以看出", "图谱", "主链", "锚点", "实体层级", "关系预览",
        # reasoning_text 常见套路/复述任务开头
        "该问题聚焦于", "这个问题讨论的是", "本文讨论的是", "该研究讨论了",
        "根据相关研究", "根据论文内容", "根据证据", "因果链",
        "这一过程表明", "这些机制共同作用导致了", "综上",
        "因此可以看出", "最终说明","证据中明确提到","可以得到印证","直接证实","进一步佐证","实验数据也证实",
        "充分证明","证据表明","数据显示","证据显示","实验数据显示",
    ]
    joined = f"{reasoning_text}\n{output}".lower()
    for marker in bad_markers:
        if marker.lower() in joined:
            return False, f"meta_language_detected:{marker}"

    if anchor_meta.get("supporting_relation_count", 0) < 3:
        return False, "insufficient_supporting_relation_count"

    # 不希望 reasoning_text 只是最终回答的重复。
    if normalize_key(reasoning_text) == normalize_key(output):
        return False, "reasoning_same_as_output"

    # 明确限制 1/2/3 式或项目符号式推理。
    step_line_pattern = re.compile(r"^\s*(?:\d+[\).．、]|[-*•])\s+", flags=re.MULTILINE)
    if len(step_line_pattern.findall(reasoning_text)) >= 2:
        return False, "reasoning_should_not_be_step_list"
    if len(step_line_pattern.findall(output)) >= 3:
        return False, "output_should_not_be_step_list"

    # 避免模型把内部字段名写进样本。
    if any(k in joined for k in ["thought_core", "thought_enhanced", "reasoning_text"]):
        return False, "field_name_leakage"

    # 限制 reasoning_text 与 output 大段重复：连续中文片段过长会让两者功能混同。
    def _only_cjk(s: str) -> str:
        return "".join(ch for ch in s if "\u4e00" <= ch <= "\u9fff")

    r_cjk = _only_cjk(reasoning_text)
    o_cjk = _only_cjk(output)
    if r_cjk and o_cjk:
        # 用短文本窗口检查，避免 O(n*m) 动态规划造成批处理额外开销。
        for i in range(0, max(len(r_cjk) - 20 + 1, 0)):
            piece = r_cjk[i:i + 20]
            if piece and piece in o_cjk:
                return False, "reasoning_output_overlap_20chars"

    # output 不允许突然引入 reasoning_text 没有用过的具体数值。
    reasoning_nums = extract_numeric_values_for_consistency(reasoning_text)
    output_nums = extract_numeric_values_for_consistency(output)
    new_output_nums = output_nums - reasoning_nums
    if new_output_nums:
        return False, f"output_has_new_numbers_not_in_reasoning:{sorted(new_output_nums)}"

    # output 不允许突然引入 reasoning_text 没有用过的关键参数概念。
    reasoning_params = extract_parameter_keys_for_consistency(reasoning_text)
    output_params = extract_parameter_keys_for_consistency(output)
    new_output_params = output_params - reasoning_params
    if new_output_params:
        return False, f"output_has_new_parameters_not_in_reasoning:{sorted(new_output_params)}"

    return True, "ok"


def score_forward_candidates(entities, out_edges):
    candidates = []
    for ent in entities.values():
        if ent.get("meta_layer") != "L1_Intervention":
            continue
        node_id = ent.get("entity_id")
        if not node_id:
            continue
        if not has_path_to_meta_layer(node_id, "L5_Performance", entities, out_edges, max_depth=4):
            continue
        reach_l5 = count_reachable_meta_layer(node_id, "L5_Performance", entities, out_edges, max_depth=4)
        out_deg = len(out_edges.get(node_id, []))
        score = 10 + 3 * reach_l5 + out_deg
        candidates.append((score, ent))
    return sorted(candidates, key=lambda x: (-x[0], normalize_text(x[1].get("entity_name"))))


def score_mechanism_candidates(entities, out_edges, in_edges):
    candidates = []
    for ent in entities.values():
        if ent.get("meta_layer") != "L4_Device_Parameter":
            continue
        node_id = ent.get("entity_id")
        if not node_id:
            continue
        has_upstream = has_upstream_intervention(node_id, entities, in_edges, max_depth=4)
        has_downstream = has_path_to_meta_layer(node_id, "L5_Performance", entities, out_edges, max_depth=3)
        if not (has_upstream and has_downstream):
            continue
        score = 8 + len(in_edges.get(node_id, [])) + len(out_edges.get(node_id, []))
        candidates.append((score, ent))
    return sorted(candidates, key=lambda x: (-x[0], normalize_text(x[1].get("entity_name"))))


def score_backward_candidates(entities, in_edges):
    candidates = []
    for ent in entities.values():
        if ent.get("meta_layer") != "L5_Performance":
            continue
        node_id = ent.get("entity_id")
        if not node_id:
            continue
        if not has_upstream_intervention(node_id, entities, in_edges, max_depth=4):
            continue
        score = 6 + len(in_edges.get(node_id, []))
        candidates.append((score, ent))
    return sorted(candidates, key=lambda x: (-x[0], normalize_text(x[1].get("entity_name"))))


def score_tradeoff_candidates(entities, out_edges):
    candidates = []
    for ent in entities.values():
        if ent.get("meta_layer") != "L1_Intervention":
            continue
        node_id = ent.get("entity_id")
        if not node_id:
            continue
        rels = out_edges.get(node_id, [])
        if not rels:
            continue
        has_pos = any(relation_is_positive(r) for r in rels)
        has_neg = any(relation_is_negative(r) for r in rels)
        if has_pos and has_neg:
            score = 12 + len(rels)
            candidates.append((score, ent))
    return sorted(candidates, key=lambda x: (-x[0], normalize_text(x[1].get("entity_name"))))



ANGLE_SELECTION_COUNTS = defaultdict(int)
ANGLE_SCORE_TOLERANCE = 8.0
ANGLE_FAMILY_SCORE_BONUS = {
    "Forward": 0.0,
    "Mechanism": 1.0,
    "Backward": 0.5,
    "Trade-off": 0.0,
}


def _deterministic_weighted_choice(options: List[str], weights: Dict[str, float], key: str) -> str:
    total = sum(max(weights.get(opt, 0.0), 0.0) for opt in options)
    if total <= 0:
        return sorted(options)[stable_hash(key) % len(options)]

    threshold = (stable_hash(key) % 1_000_000) / 1_000_000 * total
    acc = 0.0
    for opt in options:
        acc += max(weights.get(opt, 0.0), 0.0)
        if threshold <= acc:
            return opt
    return options[-1]


def choose_best_angle(file_name, forward_candidates, mechanism_candidates, backward_candidates, tradeoff_candidates):
    family_order = ["Forward", "Mechanism", "Backward", "Trade-off"]
    candidate_count_map = {
        "Forward": len(forward_candidates),
        "Mechanism": len(mechanism_candidates),
        "Backward": len(backward_candidates),
        "Trade-off": len(tradeoff_candidates),
    }
    raw_score_map = {
        "Forward": forward_candidates[0][0] if forward_candidates else None,
        "Mechanism": mechanism_candidates[0][0] if mechanism_candidates else None,
        "Backward": backward_candidates[0][0] if backward_candidates else None,
        "Trade-off": (tradeoff_candidates[0][0] - 1) if tradeoff_candidates else None,
    }
    adjusted_score_map = {
        angle: (raw_score_map[angle] + ANGLE_FAMILY_SCORE_BONUS[angle]) if raw_score_map[angle] is not None else None
        for angle in family_order
    }
    selection_count_before = dict(ANGLE_SELECTION_COUNTS)

    if not any(v is not None for v in adjusted_score_map.values()):
        debug_info = {
            "candidate_count_map": candidate_count_map,
            "raw_score_map": raw_score_map,
            "adjusted_score_map": adjusted_score_map,
            "eligible_families": [],
            "selection_mode": "no_candidates",
            "selection_count_before": selection_count_before,
            "selection_count_after": dict(ANGLE_SELECTION_COUNTS),
            "chosen_family": None,
            "weighted_family_weight_map": {},
        }
        return None, debug_info

    # Trade-off 硬触发：只要存在明确正负并存候选，就优先选择
    if tradeoff_candidates:
        chosen = "Trade-off"
        ANGLE_SELECTION_COUNTS[chosen] += 1
        debug_info = {
            "candidate_count_map": candidate_count_map,
            "raw_score_map": raw_score_map,
            "adjusted_score_map": adjusted_score_map,
            "eligible_families": ["Trade-off"],
            "selection_mode": "hard_tradeoff_trigger",
            "selection_count_before": selection_count_before,
            "selection_count_after": dict(ANGLE_SELECTION_COUNTS),
            "chosen_family": chosen,
            "weighted_family_weight_map": {"Trade-off": 1.0},
        }
        return chosen, debug_info

    non_tradeoff = ["Forward", "Mechanism", "Backward"]
    available = [a for a in non_tradeoff if adjusted_score_map[a] is not None]
    if not available:
        debug_info = {
            "candidate_count_map": candidate_count_map,
            "raw_score_map": raw_score_map,
            "adjusted_score_map": adjusted_score_map,
            "eligible_families": [],
            "selection_mode": "no_non_tradeoff_candidates",
            "selection_count_before": selection_count_before,
            "selection_count_after": dict(ANGLE_SELECTION_COUNTS),
            "chosen_family": None,
            "weighted_family_weight_map": {},
        }
        return None, debug_info

    best_adjusted_score = max(adjusted_score_map[a] for a in available)
    eligible_families = [
        a for a in available
        if adjusted_score_map[a] >= best_adjusted_score - ANGLE_SCORE_TOLERANCE
    ]
    if not eligible_families:
        eligible_families = [max(available, key=lambda a: adjusted_score_map[a])]

    weighted_family_weight_map = {}
    for angle in eligible_families:
        score = max(adjusted_score_map[angle], 0.1)
        count_penalty = 1.0 + ANGLE_SELECTION_COUNTS[angle]
        weighted_family_weight_map[angle] = score / count_penalty

    choice_key = f"{file_name}||{'|'.join(eligible_families)}||{selection_count_before}"
    chosen = _deterministic_weighted_choice(eligible_families, weighted_family_weight_map, choice_key)
    ANGLE_SELECTION_COUNTS[chosen] += 1
    debug_info = {
        "candidate_count_map": candidate_count_map,
        "raw_score_map": raw_score_map,
        "adjusted_score_map": adjusted_score_map,
        "eligible_families": eligible_families,
        "selection_mode": "weighted_non_tradeoff_choice",
        "selection_count_before": selection_count_before,
        "selection_count_after": dict(ANGLE_SELECTION_COUNTS),
        "chosen_family": chosen,
        "weighted_family_weight_map": weighted_family_weight_map,
    }
    return chosen, debug_info


FORWARD_TEMPLATES = [
    "作为光伏器件专家，请围绕 {material_sys} 体系中的「{anchor_name}」，系统分析它会如何沿主路径逐步影响微观机制、器件参数与最终性能。请聚焦证据支持最强的链条，不要泛泛罗列。",
    "请从器件工作机理出发，解释 {material_sys} 中「{anchor_name}」为什么会引起后续结构/机制变化，并进一步传导到关键参数和性能结果。请优先讨论主链。",
    "在 {material_sys} 研究中，若采用「{anchor_name}」，最可能先改变哪些局部机制，并最终体现在哪些器件参数和性能上？请基于证据链给出紧凑分析。",
    "请围绕 {material_sys} 体系中的「{anchor_name}」，说明它对器件性能提升的关键贡献路径是什么。要求从上游干预到下游结果逐步展开，避免无关扩展。",
    "请基于证据，分析 {material_sys} 中「{anchor_name}」如何从局部改造逐步演化为宏观性能变化。请强调最可信的主路径，而不是列出所有可能影响。"
]

MECHANISM_TEMPLATES = [
    "请从器件物理角度分析 {material_sys} 中「{anchor_name}」这一关键参数/现象，说明它如何连接上游改造与下游性能结果。要求给出紧凑、可验证的机制链条。",
    "在 {material_sys} 体系里，请围绕「{anchor_name}」解释：它为什么会成为上游材料/界面优化与最终性能变化之间的关键桥梁？",
    "请把「{anchor_name}」视作 {material_sys} 器件中的中介变量，分析其变化背后的主要物理原因，以及它如何进一步影响最终性能。",
    "请围绕 {material_sys} 中的「{anchor_name}」，说明该参数/机制变化如何反映上游结构或工艺优化，并最终传导到器件性能。",
    "从物理机制出发，请解释 {material_sys} 中「{anchor_name}」的变化意味着什么，以及它在整条性能演化链中的作用位置。"
]

BACKWARD_TEMPLATES = [
    "在 {material_sys} 太阳能电池研究中，如果目标是获得「{anchor_name}」，请从结果反推最可能的上游结构、界面或工艺路径，并解释这些路径通过哪些机制实现该结果。",
    "请从 {material_sys} 器件最终获得的「{anchor_name}」出发，重建最可信的因果链条：哪些改造最可能先发生，哪些机制随后起主要作用？",
    "若只知道 {material_sys} 体系取得了「{anchor_name}」，应优先怀疑哪些关键干预和机制变化？请基于证据链倒推主要路线。",
    "请从最终性能/结果「{anchor_name}」出发，分析在 {material_sys} 中最可能支撑这一结果的关键机制路径。要求强调最主要、最可信的链条。",
    "在 {material_sys} 研究里，请围绕结果「{anchor_name}」进行逆向分析：哪些上游改造和中间机制最可能共同导致这一结果？"
]

TRADEOFF_TEMPLATES = [
    "在 {material_sys} 器件中应用「{anchor_name}」时，可能伴随收益与代价并存。请基于证据链分析其正负作用分别通过哪些路径体现，以及最终净影响如何形成。",
    "请讨论 {material_sys} 中「{anchor_name}」带来的 trade-off：它提升了什么，同时又引入了哪些限制？请用主链说明收益与代价如何竞争。",
    "若把「{anchor_name}」视为 {material_sys} 中的双刃剑，请解释它的正面作用和潜在代价分别来源于哪些机制，并说明最终为何表现为当前结果。",
    "请基于证据分析 {material_sys} 中「{anchor_name}」的平衡关系：它在哪些环节带来改善，又在哪些环节产生牺牲？最终净收益由什么决定？",
    "在 {material_sys} 体系里，「{anchor_name}」是否同时带来了促进与抑制效应？请沿因果链说明其收益、代价及最终性能走向。"
]

PROMPT_TEMPLATE_BANK = {
    "Forward": FORWARD_TEMPLATES,
    "Mechanism": MECHANISM_TEMPLATES,
    "Backward": BACKWARD_TEMPLATES,
    "Trade-off": TRADEOFF_TEMPLATES,
}

FAMILY_VARIANT_COUNTERS = defaultdict(int)


def choose_instruction_template(angle: str, file_name: str, anchor_id: str) -> Tuple[str, str]:
    templates = PROMPT_TEMPLATE_BANK.get(angle, [])
    if not templates:
        raise ValueError(f"No templates for angle={angle}")
    idx = FAMILY_VARIANT_COUNTERS[angle] % len(templates)
    FAMILY_VARIANT_COUNTERS[angle] += 1
    variant_id = f"{angle}_{idx+1:02d}"
    return templates[idx], variant_id




def validate_natural_instruction(instruction: str, angle: str) -> Tuple[bool, str]:
    instruction = normalize_text(instruction)
    if not instruction:
        return False, "empty_instruction"
    if len(instruction) < 18:
        return False, "instruction_too_short"
    if len(instruction) > 180:
        return False, "instruction_too_long"

    bad_phrases = [
        "请基于证据链", "基于证据链", "主路径", "主链", "逐步展开",
        "fact skeleton", "证据骨架", "图谱", "根据给定", "给定数据",
        "要求", "不要泛泛罗列", "最可信的主路径"
    ]
    low = instruction.lower()
    for phrase in bad_phrases:
        if phrase.lower() in low:
            return False, f"template_phrase_detected:{phrase}"

    # 问题应像真实提问，通常需要问号或疑问表达；陈述式也允许，但不能像命令模板。
    question_markers = ["？", "?", "为什么", "如何", "怎么", "哪些", "能否", "是不是", "是否", "有什么", "该不该", "应该"]
    if not any(m in instruction for m in question_markers):
        return False, "not_question_like"

    return True, "ok"


async def generate_natural_instruction_with_llm(
        model_pool,
        *,
        angle: str,
        material_sys: str,
        anchor_name: str,
        anchor_layer: str,
        skeleton: str,
        support_relations: List[Dict[str, Any]],
        fallback_instruction: str,
        file_name: str,
) -> Tuple[str, Dict[str, Any]]:
    """在类别约束下生成自然 instruction；失败时回退模板，保证批处理不中断。"""
    fallback_instruction = normalize_text(fallback_instruction)

    if not USE_LLM_INSTRUCTION_GENERATOR:
        return fallback_instruction, {
            "instruction_generation_method": "template_only",
            "instruction_style": "fallback_template",
            "instruction_generator_api": "",
            "instruction_validation": "disabled"
        }

    relation_preview = support_relations[:8] if isinstance(support_relations, list) else []
    prompt = f"""你是光伏领域科研问答数据构建专家。请根据给定论文证据链，生成 1 个自然、真实、不模板化的中文用户问题。

任务类别 chain_type: {angle}
材料/体系: {material_sys}
锚点实体: {anchor_name}
锚点层级: {anchor_layer}

类别含义：
- Forward：从干预、材料、结构或工艺问它会带来什么机制和结果。
- Backward：从性能结果、异常现象或关键参数倒推原因。
- Mechanism：解释某个中间机制、结构变化或器件参数为什么重要。
- Trade-off：分析同一干预的收益与代价。

要求：
1. 问题要像研究生、论文阅读者或光伏研究人员真实会问的话。
2. 不要使用“请基于证据链”“主路径”“逐步展开”“不要泛泛罗列”等数据集模板话术。
3. 不要把完整答案链条塞进问题里；最多暴露 1-2 个关键锚点。
4. 不要引入证据链之外的新材料、新数值、新器件参数。
5. 可以口语化一点，但仍保持科研语境。
6. 只输出 JSON，不要解释。

可用证据摘要：
{skeleton}

关系预览：
{json.dumps(relation_preview, ensure_ascii=False, indent=2)}

返回格式：
{{
  "instruction": "自然用户问题",
  "style": "researcher_question / paper_understanding / diagnostic_question / mechanism_question / tradeoff_question",
  "exposed_anchors": ["锚点1", "锚点2"]
}}
"""

    last_error = ""
    for attempt in range(MAX_RETRIES):
        try:
            used_api_name, raw_response = await model_pool.call(
                messages=[{"role": "user", "content": prompt}],
                temperature=INSTRUCTION_TEMPERATURE,
                timeout=MAX_MODEL_TIMEOUT,
            )
            data = try_parse_json_object(raw_response)
            if not isinstance(data, dict):
                raise ValueError("instruction generator did not return JSON object")

            instruction = coerce_text_field(data.get("instruction") or data.get("question"))
            ok, reason = validate_natural_instruction(instruction, angle)
            if not ok:
                raise ValueError(reason)

            return instruction, {
                "instruction_generation_method": "llm_generated",
                "instruction_style": normalize_text(data.get("style") or "unknown"),
                "instruction_generator_api": used_api_name,
                "instruction_validation": reason,
                "exposed_anchors": data.get("exposed_anchors", []) if isinstance(data.get("exposed_anchors"), list) else []
            }
        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 ** attempt)

    if FALLBACK_TO_TEMPLATE_INSTRUCTION:
        return fallback_instruction, {
            "instruction_generation_method": "fallback_template",
            "instruction_style": "fallback_template",
            "instruction_generator_api": "",
            "instruction_validation": f"fallback_after_error:{last_error}",
            "exposed_anchors": [anchor_name] if anchor_name else []
        }

    raise RuntimeError(f"自然 instruction 生成失败: {last_error}")


def build_instruction_and_anchor(doc, file_name):
    entities = {e["entity_id"]: e for e in doc.get("entities", []) if isinstance(e, dict) and e.get("entity_id")}
    relations = [r for r in doc.get("relations", []) if isinstance(r, dict) and r.get("subject_id") and r.get("object_id")]
    if not entities or not relations:
        return None, None, None, None

    material_sys = normalize_text(doc.get("material_system", "photovoltaic materials"))
    out_edges, in_edges = build_graph_indices(relations)

    forward_candidates = score_forward_candidates(entities, out_edges)
    mechanism_candidates = score_mechanism_candidates(entities, out_edges, in_edges)
    backward_candidates = score_backward_candidates(entities, in_edges)
    tradeoff_candidates = score_tradeoff_candidates(entities, out_edges)

    angle, angle_debug = choose_best_angle(file_name, forward_candidates, mechanism_candidates, backward_candidates, tradeoff_candidates)
    if not angle:
        return None, None, None, None

    if angle == "Forward":
        anchor_score, anchor = forward_candidates[0]
    elif angle == "Mechanism":
        anchor_score, anchor = mechanism_candidates[0]
    elif angle == "Backward":
        anchor_score, anchor = backward_candidates[0]
    elif angle == "Trade-off":
        anchor_score, anchor = tradeoff_candidates[0]
    else:
        return None, None, None, None

    anchor_name = normalize_text(anchor["entity_name"])
    template, variant_id = choose_instruction_template(angle, file_name, anchor["entity_id"])
    instruction = template.format(
        material_sys=material_sys,
        anchor_name=anchor_name
    )

    anchor_id = anchor["entity_id"]
    skeleton, local_rels, support_relations, support_evidences = format_fact_skeleton(anchor_id, entities, relations, out_edges, in_edges)
    if not skeleton.strip():
        return None, None, None, None

    anchor_meta = {
        "anchor_entity_id": anchor_id,
        "anchor_entity_name": normalize_text(anchor.get("entity_name")),
        "anchor_meta_layer": normalize_text(anchor.get("meta_layer")),
        "anchor_score": anchor_score,
        "prompt_family": angle,
        "prompt_variant_id": variant_id,
        "angle_candidate_count_map": angle_debug["candidate_count_map"],
        "angle_raw_score_map": angle_debug["raw_score_map"],
        "angle_score_map": angle_debug["adjusted_score_map"],
        "angle_eligible_families": angle_debug["eligible_families"],
        "angle_selection_mode": angle_debug["selection_mode"],
        "angle_weighted_family_weight_map": angle_debug.get("weighted_family_weight_map", {}),
        "prompt_family_selection_count_before": angle_debug["selection_count_before"].get(angle, 0),
        "prompt_family_selection_count": angle_debug["selection_count_after"].get(angle, 0),
        "fact_skeleton": skeleton,
        "supporting_relations": support_relations,
        "supporting_evidences": support_evidences,
        "supporting_relation_count": len(local_rels),
    }

    ok, reason = sample_pass_main_chain_filter(anchor_meta)
    anchor_meta["local_chain_check"] = {"passed": ok, "reason": reason}
    if not ok:
        return None, None, None, None

    system_prompt = f"""You are an expert scientific reasoning dataset writer for photovoltaic, solar-cell, and related semiconductor device papers.

    Generate ONE complete Chinese SFT sample as a valid JSON object, strictly grounded in the hidden evidence pool.

    # [INPUT CONTEXT]
    Task category: {angle}
    Material/system: {material_sys}
    Anchor entity: {anchor_name}
    Anchor layer: {normalize_text(anchor.get("meta_layer"))}

    Category meaning:
    - Forward: from intervention/material/structure/process to mechanism, parameter, and result.
    - Backward: from performance result or abnormal phenomenon back to likely causes.
    - Mechanism: explain why an intermediate structure, mechanism, or parameter matters.
    - Trade-off: analyze supported benefits and costs of the same intervention.

    # [HIDDEN EVIDENCE POOL]
    ---
    {skeleton}
    ---

    # [GLOBAL PRINCIPLES]
    1. Hidden evidence only:
       Treat the evidence pool as a hidden internal source. The user has not seen it.
       Never mention, imply, or allude to the existence of provided context, extracted evidence, structured data, relations, labels, fields, graphs, or a generation task.

    2. Expert persona:
       Write like a domain expert directly answering a colleague’s question.
       Do not write like an AI model summarizing input data, a reviewer, or a student doing reading comprehension.

    3. Evidence selection:
       The evidence pool is not a fixed answer path.
       Do not mechanically rewrite every relation or force an “A→B→C” chain.
       Select only the most diagnostic evidence needed to answer the question.

    4. Factuality:
       Do not invent materials, mechanisms, thresholds, values, device parameters, or performance results not supported by the evidence pool.
       If evidence is indirect or incomplete, use cautious wording such as “更可能说明”, “倾向于支持”, “可以作为辅助证据”.

    5. Evidence boundary:
       Do not merge results from different device types, experimental groups, material systems, or cell architectures as if they were the same case.
       If a value comes from a related but different device/group, use it only as auxiliary evidence with cautious wording, or do not use it.

    6. Anti-report style:
       Do not use report-style evidence-introduction phrases.
       BAD: “证据表明，Jsc提升了1.4 mA/cm²”
       GOOD: “Jsc提升1.4 mA/cm²更直接指向光吸收增强或载流子收集改善”

    # [OUTPUT JSON SCHEMA]
    Return exactly ONE JSON object with exactly FOUR keys.
    Do not add any other keys such as "input", "meta_info", "analysis", or "confidence".

    {{
      "cell_type": "silicon_solar_cell",
      "instruction": "自然中文科研问题",
      "reasoning_text": "一段自然中文专家判断依据",
      "output": "最终中文专业回答"
    }}

    # [FIELD REQUIREMENTS]

    "cell_type":
    Choose exactly one from:
    silicon_solar_cell, perovskite_solar_cell, organic_solar_cell, dye_sensitized_solar_cell,
    thin_film_solar_cell, tandem_solar_cell, hybrid_solar_cell, pv_module_or_system, other, uncertain.

    Rules:
    - Use "other" for non-solar-cell devices, such as thin-film transistors, source/drain devices, gate/channel devices, mobility-only electronics, or shift-register devices.
    - Use "silicon_solar_cell" for PERC, TOPCon, HJT/HIT, IBC, c-Si, mc-Si, silicon wafer, and boron-oxygen defect solar-cell samples.

    "instruction":
    - Write one natural Chinese question, 20-90 Chinese characters.
    - It should sound like a real graduate student, paper reader, or researcher asking a question.
    - Mention at most 1-2 key scientific entities.
    - Do not reveal the full reasoning chain.
    - Do not use dataset-template wording such as “请基于证据链”, “主路径”, “主链”, “逐步展开”, “根据给定内容”, “根据图谱”.

    "reasoning_text":
    - Write one continuous Chinese paragraph, preferably 160-650 Chinese characters.
    - This is expert diagnostic reasoning, not a relation-chain rewrite.
    - Start directly from the key physical bottleneck or diagnostic issue, such as recombination, optical absorption, passivation, contact resistance, carrier transport, energy-level alignment, stability, manufacturability, or lifetime degradation.
    - Use device parameters diagnostically:
      Jsc usually supports optical absorption or carrier collection;
      Voc often reflects recombination loss, passivation, or energy selectivity;
      FF often reflects contact resistance, series resistance, shunting, or transport quality;
      degradation rate, retention, or operating time reflects durability.
    - Explain why the selected parameters support the mechanism judgment, not merely that they changed.
    - Embed key numerical values from the evidence pool at the corresponding mechanism or parameter position, such as efficiency, Voc, Jsc, FF, degradation rate, lifetime, resistance, thickness, temperature, concentration, stability time, or retention ratio.
    - Do not invent numbers. If no explicit numbers are available, use qualitative wording.
    - Do not use lists, bullets, headings, arrows, or numbered steps.
    - Do not write like a paper abstract, literature review, formal report, or mini discussion section.
    - Do not use report-style transition phrases such as “实验数据也证实了这一点”, “这些机制共同作用”, “更重要的是”, “同时也说明”. Directly connect each parameter to its mechanism meaning.    
    - Avoid overclaiming. Do not turn “one important factor” into “the only determining factor”; do not turn correlation or supporting evidence into certain causality.
    - End naturally at the mechanism judgment, parameter interpretation, evidence boundary, or performance result.

    " factor”; do not turn correlation or supporting evidence into certain causality.
    - End naturally at the mechanism judgment, parameter interpretation, evidence boundary, or performance result.

    "output":
    - Write the final Chinese answer to the user, preferably 100-420 Chinese characters.
    - Structure: conclusion + key mechanism + key quantitative evidence.
    - It should be a professional synthesis of reasoning_text, not an independent new analysis.
    - Do not re-explain the full reasoning process.
    - Do not introduce any new numerical value or new parameter that did not appear in reasoning_text.
    - Preserve the central numerical values from reasoning_text when they are essential to the conclusion.
    - Do not repeat reasoning_text verbatim or reorder the same paragraph.
    - reasoning_text and output must not share identical continuous Chinese text longer than 20 characters.
    - Do not use label-style formatting such as “结论：”, “关键机制：”, “关键证据：”, “原因：”, “主要体现在：”. Write as one natural paragraph.
    -Do not introduce any new mechanism, causal bridge, process detail, or interpretation that did not appear in reasoning_text.

    # [FORBIDDEN EXPRESSIONS]
    Never use the following expressions in any output field:

    Internal/meta leakage:
    L1_Intervention, L2_Structure_Composition, L3_Mechanism_Physics, L4_Device_Parameter, L5_Performance,
    fact skeleton, anchor, meta_layer, supporting relations, graph, prompt, given data,
    根据给定内容, 根据给定信息, 根据证据链, 根据图谱, 根据提供的信息, 论文中的证据链,
    主链, 锚点, 实体, 关系, 层级, 路径,
    从上下文可以看出, 所给材料表明, 上述信息说明, 文本中提到, 提取结果显示, 结构化信息, 内部上下文, 隐藏证据, 生成任务.

    Report-style phrasing:
    证据表明, 数据显示, 结果显示, 实验表明, 研究发现, 已有证据显示, 相关数据说明, 从数据可以看出,
    该问题聚焦于, 这个问题讨论的是, 根据相关研究, 根据论文内容, 根据证据,
    这表明, 这说明, 由此可见, 可以看出,
    验证了该工艺的有效性, 说明该策略有效, 该工艺有效, 该策略具有优势, 具有重要意义,
    因果链, 这一过程表明, 这些机制共同作用导致了, 综上, 综上所述, 因此可以看出, 最终说明.

    Label-style answer formatting:
    结论：, 关键机制：, 关键证据：, 原因：, 主要体现在：.

    Avoid report-style proof phrases such as “被证实”, “直接证据”, “进一步佐证”, “充分证明”. Prefer direct parameter interpretation.

    不要使用“证据显示”“直接证据”“直接印证”“进一步佐证”“根本驱动力”等上述提到的证明式报告腔。应直接把参数变化写成机制判断，例如“J₀e降低和τ_eff提升更直接支持表面复合减少这一判断”。
     
        # [FORMAT EXAMPLE ONLY — DO NOT COPY CONTENT]
    The example below only demonstrates the expected JSON format.
    Do not copy its materials, mechanisms, values, wording, or sentence structure.

    {{
      "cell_type": "silicon_solar_cell",
      "instruction": "某种工艺为什么会改善电池的开路电压和效率？",
      "reasoning_text": "界面复合通常是限制开路电压的关键因素。某参数的提升更直接指向复合损失降低，而另一个电流相关参数的变化则支持光吸收或载流子收集改善的判断。若效率提升同时伴随这些参数变化，更合理的解释是该工艺同时改善了复合控制和电荷收集，但不能把单一参数视为唯一原因。",
      "output": "该工艺的性能提升主要来自复合损失降低和电荷收集改善。开路电压相关参数支持界面或体区复合受到抑制，电流相关参数则指向光吸收或载流子收集增强，因此最终效率得到提升。"
    }}
    """

    # instruction 在单次 LLM 响应中生成；此处返回的 instruction 只是本地兜底模板。
    return instruction, system_prompt, angle, anchor_meta



async def save_bad_response(file_name: str, attempt: int, raw_response: str):
    base = re.sub(r'[\\/:*?"<>|]+', '_', os.path.splitext(file_name)[0])
    save_path = os.path.join(DEBUG_BAD_RESP_DIR, f"{base}.attempt{attempt}.txt")
    async with aiofiles.open(save_path, "w", encoding="utf-8") as f:
        await f.write(raw_response or "")


async def process_file_for_cot(file_path, model_pool, seen_uids, seen_source_files, progress_tracker):
    file_name = os.path.basename(file_path)

    if file_name in seen_source_files:
        await progress_tracker.log_finish(file_name, 'resumed', '已存在 source_file，断点续传跳过')
        return

    await progress_tracker.log_start(file_name)

    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            doc = json.loads(await f.read())
    except Exception as e:
        logging.error(f"[WARN] [{file_name}] 文件读取或解析失败: {e}")
        await progress_tracker.log_finish(file_name, 'failed', f'文件读取失败: {e}')
        return

    paper_ok, paper_check = paper_has_valid_main_chain(doc)
    if not paper_ok:
        logging.warning(f"⏭ [{file_name}] 主链不达标，跳过。原因: {paper_check}")
        await progress_tracker.log_finish(file_name, 'skipped', f'主链不达标: {paper_check}')
        return

    fallback_instruction, system_prompt, angle, anchor_meta = build_instruction_and_anchor(doc, file_name)
    if not fallback_instruction or not system_prompt or not angle or not anchor_meta:
        logging.warning(f"[WARN] [{file_name}] 无法构造高质量样本，跳过。")
        await progress_tracker.log_finish(file_name, 'skipped', '无法构造高质量样本')
        return

    anchor_meta["paper_main_chain_check"] = paper_check

    # 单次调用模式：不再单独请求一次 LLM 生成 instruction，避免每条样本 2 次串行调用。
    instruction_meta = {
        "instruction_generation_method": "one_call_joint_generation",
        "instruction_style": "generated_with_answer",
        "instruction_generator_api": "same_as_teacher",
        "instruction_validation": "pending",
        "exposed_anchors": []
    }
    anchor_meta.update(instruction_meta)
    anchor_meta["fallback_instruction"] = fallback_instruction

    logging.info(
        f" [{file_name}] 视角: {angle} | 锚点: {anchor_meta['anchor_entity_name']} | 单次请求生成 instruction/reasoning/output..."
    )

    for attempt in range(MAX_RETRIES):
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "请生成一个完整的中文训练样本 JSON。"}
            ]
            used_api_name, raw_response = await model_pool.call(
                messages=messages,
                temperature=TEMPERATURE,
                timeout=MAX_MODEL_TIMEOUT,
            )

            response_data = parse_model_json(raw_response)
            if response_data is None:
                await save_bad_response(file_name, attempt + 1, raw_response)
                raise ValueError("模型未输出合法 JSON")

            instruction = coerce_text_field(response_data.get("instruction"))
            if not instruction:
                instruction = fallback_instruction
                anchor_meta["instruction_generation_method"] = "fallback_template_after_missing_instruction"

            inst_ok, inst_reason = validate_natural_instruction(instruction, angle)
            anchor_meta["instruction_validation"] = inst_reason
            if not inst_ok:
                raise ValueError(f"自然 instruction 过滤未通过: {inst_reason}")

            reasoning_text = coerce_text_field(response_data.get("reasoning_text"))
            output = coerce_text_field(response_data.get("output"))
            llm_cell_type_raw = coerce_text_field(response_data.get("cell_type"))

            ok, reason = generated_sample_pass_filter(reasoning_text, output, anchor_meta)
            if not ok:
                raise ValueError(f"生成样本过滤未通过: {reason}")

            sample_uid = hashlib.md5(
                f"{file_name}||{angle}||{anchor_meta['anchor_entity_id']}||{instruction}||{anchor_meta['fact_skeleton']}".encode("utf-8")
            ).hexdigest()
            anchor_meta["sample_uid"] = sample_uid

            async with file_io_lock:
                if sample_uid in seen_uids:
                    logging.info(f"⏭ [{file_name}] 样本已存在，跳过。")
                    seen_source_files.add(file_name)
                    await progress_tracker.log_finish(file_name, 'resumed', 'sample_uid 已存在，断点续传跳过')
                    return

            auto_annotation = build_auto_annotation(
                doc=doc,
                file_name=file_name,
                angle=angle,
                anchor_meta=anchor_meta,
                thought_core=reasoning_text,
                thought_enhanced="",
                output=output
            )

            final_cell_type = normalize_llm_cell_type(
                llm_cell_type_raw,
                fallback=auto_annotation["cell_type_auto"]
            )

            # ===============================
            # 训练文件：自然 instruction + 自然 reasoning_text + 极简元数据
            # ===============================
            sft_data = {
                "instruction": instruction,
                "input": "",
                "reasoning_text": reasoning_text,
                "output": output,
                "meta_info": {
                    "chain_type": angle,
                    "cell_type": final_cell_type,
                    "sample_type": auto_annotation["sample_type_auto"],
                    "chain_quality": auto_annotation["chain_grade_auto"],
                    "grounded_quality": auto_annotation["grounded_grade_auto"]
                }
            }

            # ===============================
            # 调试文件：保留全量追溯信息，不进入训练
            # ===============================
            debug_data = {
                "sample_uid": sample_uid,
                "source_file": file_name,
                "used_api_name": used_api_name,
                "chain_type": angle,
                "instruction": instruction,
                "fallback_instruction": anchor_meta.get("fallback_instruction", ""),
                "instruction_generation_method": anchor_meta.get("instruction_generation_method", ""),
                "instruction_style": anchor_meta.get("instruction_style", ""),
                "instruction_generator_api": used_api_name,
                "instruction_validation": anchor_meta.get("instruction_validation", ""),
                "exposed_anchors": anchor_meta.get("exposed_anchors", []),
                "material_system": doc.get("material_system", "Unknown"),
                "target_problem": doc.get("target_problem", "Unknown"),
                "research_focus": doc.get("research_focus", "Unknown"),
                "paper_main_chain_check": paper_check,
                "anchor_entity_id": anchor_meta["anchor_entity_id"],
                "anchor_entity_name": anchor_meta["anchor_entity_name"],
                "anchor_meta_layer": anchor_meta["anchor_meta_layer"],
                "anchor_score": anchor_meta["anchor_score"],
                "local_chain_check": anchor_meta["local_chain_check"],
                "fact_skeleton": anchor_meta["fact_skeleton"],
                "supporting_relations": anchor_meta["supporting_relations"],
                "supporting_evidences": anchor_meta["supporting_evidences"],
                "supporting_relation_count": anchor_meta["supporting_relation_count"],
                "prompt_family": anchor_meta["prompt_family"],
                "prompt_variant_id": anchor_meta["prompt_variant_id"],
                "prompt_family_selection_count_before": anchor_meta["prompt_family_selection_count_before"],
                "prompt_family_selection_count": anchor_meta["prompt_family_selection_count"],
                "angle_candidate_count_map": anchor_meta["angle_candidate_count_map"],
                "angle_raw_score_map": anchor_meta["angle_raw_score_map"],
                "angle_score_map": anchor_meta["angle_score_map"],
                "angle_eligible_families": anchor_meta["angle_eligible_families"],
                "angle_selection_mode": anchor_meta["angle_selection_mode"],
                "angle_weighted_family_weight_map": anchor_meta["angle_weighted_family_weight_map"],
                "cell_type_auto": auto_annotation["cell_type_auto"],
                "llm_cell_type_raw": llm_cell_type_raw,
                "final_cell_type": final_cell_type,
                "sample_type_auto": auto_annotation["sample_type_auto"],
                "chain_grade_auto": auto_annotation["chain_grade_auto"],
                "grounded_grade_auto": auto_annotation["grounded_grade_auto"],
                "annotation": auto_annotation["annotation"]
            }

            async with file_io_lock:
                if sample_uid in seen_uids:
                    logging.info(f"⏭ [{file_name}] 样本在等待期间已写入，跳过。")
                    seen_source_files.add(file_name)
                    await progress_tracker.log_finish(file_name, 'resumed', '样本在等待期间已写入')
                    return

                async with aiofiles.open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
                    await f.write(json.dumps(sft_data, ensure_ascii=False) + "\n")

                async with aiofiles.open(DEBUG_JSONL, "a", encoding="utf-8") as f:
                    await f.write(json.dumps(debug_data, ensure_ascii=False) + "\n")

                seen_uids.add(sample_uid)
                seen_source_files.add(file_name)

            logging.info(f" [{file_name}] 成功生成单次调用自然化 SFT 数据。 | API={used_api_name}")
            await progress_tracker.log_finish(file_name, 'success', f'API={used_api_name} | family={angle} | one_call')
            return

        except Exception as api_err:
            if attempt < MAX_RETRIES - 1:
                logging.warning(f"[WARN] [{file_name}] 第 {attempt + 1} 次失败: {api_err}，准备重试...")
                await asyncio.sleep(2 ** attempt)
            else:
                logging.error(f"[ERROR] [{file_name}] 彻底失败，丢弃该样本: {api_err}")
                await progress_tracker.log_finish(file_name, 'failed', f'最终失败: {api_err}')
                return


async def main():
    print("\n 启动【自然 instruction + 自然 reasoning_text + 极简元数据 + 双文件输出】数据集生成引擎...\n")

    if not os.path.exists(INPUT_DIR):
        print(f"[ERROR] 找不到输入目录: {INPUT_DIR}")
        return

    json_files = sorted(
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.endswith(".json")
    )

    if not json_files:
        print("[WARN] refined_results 目录中没有 JSON 文件。")
        return

    if not RESUME_MODE and OVERWRITE_OUTPUT:
        if os.path.exists(OUTPUT_JSONL):
            os.remove(OUTPUT_JSONL)
        if os.path.exists(DEBUG_JSONL):
            os.remove(DEBUG_JSONL)

    ANGLE_SELECTION_COUNTS.clear()
    FAMILY_VARIANT_COUNTERS.clear()

    if RESUME_MODE:
        seen_uids, seen_source_files, existing_count = load_existing_output(OUTPUT_JSONL, DEBUG_JSONL)
    else:
        seen_uids, seen_source_files, existing_count = set(), set(), 0

    pending_files = [fp for fp in json_files if os.path.basename(fp) not in seen_source_files]

    print(f" 发现 {len(json_files)} 篇已融合处理图谱。")
    print(f"♻ 断点续传模式: {'开启' if RESUME_MODE else '关闭'}")
    print(f" 已有样本数: {existing_count} | 已完成篇数: {len(seen_source_files)} | 待处理篇数: {len(pending_files)}")
    print(f" API 列表: {', '.join(API_NAMES)}")

    if not pending_files:
        print(" 没有待处理文件，任务已完成。")
        return

    model_pool = MultiAPIModelPool(API_NAMES)
    progress_tracker = ProgressTracker(len(pending_files))

    tasks = [
        asyncio.create_task(process_file_for_cot(fp, model_pool, seen_uids, seen_source_files, progress_tracker))
        for fp in pending_files
    ]
    await asyncio.gather(*tasks)

    print("\n" + "=" * 60)
    print(" 数据集生成完成！")
    print(f" 训练输出文件: {OUTPUT_JSONL}")
    print(f" 调试追溯文件: {DEBUG_JSONL}")
    print(f" 非法输出调试目录: {DEBUG_BAD_RESP_DIR}")
    print(f" 本轮待处理: {len(pending_files)} | 成功={progress_tracker.success} 跳过={progress_tracker.skipped} 失败={progress_tracker.failed} 续跑跳过={progress_tracker.resumed}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    asyncio.run(main())
