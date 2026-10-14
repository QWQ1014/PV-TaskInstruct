import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_PARENT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    PROJECT_PARENT
    / "Extraction_project"
    / "data"
    / "pv_ecot_core_train_v2_safe_no_revision_enriched.jsonl"
)
DEFAULT_TARGET = (
    PROJECT_PARENT
    / "Extraction_project"
    / "data"
    / "pv_attribution_instruction_preview_100.jsonl"
)


if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_output_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if limit is not None and len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[skip] line {line_no}: invalid json: {exc}")
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                print(f"[skip] line {line_no}: not an object")
    return rows


def load_completed_indices(path: Path) -> set[int]:
    completed: set[int] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta = obj.get("meta_info") if isinstance(obj, dict) else None
            if not isinstance(meta, dict):
                continue
            try:
                idx = int(meta.get("source_index"))
            except (TypeError, ValueError):
                continue
            if idx > 0:
                completed.add(idx)
    return completed


def parse_indices(value: str) -> set[int]:
    indices: set[int] = set()
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if start <= 0 or end < start:
                raise ValueError(f"Invalid index range: {chunk}")
            indices.update(range(start, end + 1))
        else:
            index = int(chunk)
            if index <= 0:
                raise ValueError(f"Invalid index: {chunk}")
            indices.add(index)
    return indices


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    else:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            text = match.group(0)
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("teacher output is not a JSON object")
    return obj


def compact_list(items: Any, max_items: int = 8, max_chars: int = 1800) -> str:
    if not isinstance(items, list):
        return normalize_text(items)[:max_chars]
    parts = []
    for item in items[:max_items]:
        text = normalize_text(item)
        if text:
            parts.append(text)
    return "\n".join(parts)[:max_chars]


def get_provenance(item: dict[str, Any]) -> dict[str, Any]:
    provenance = item.get("provenance")
    return provenance if isinstance(provenance, dict) else {}


def choose_input_mode(index: int, input_ratio: float) -> str:
    bucket = (index - 1) % 10
    return "literature_input" if bucket < round(input_ratio * 10) else "no_input_style"


def choose_answer_style(index: int) -> str:
    styles = [
        "conclusion_first",
        "mechanism_flow",
        "contrastive_boundary",
        "parameter_diagnostic",
        "concise_expert",
    ]
    return styles[(index - 1) % len(styles)]


def build_literature_input(item: dict[str, Any]) -> str:
    provenance = get_provenance(item)
    material_system = normalize_text(provenance.get("material_system"))
    research_focus = normalize_text(provenance.get("research_focus"))
    target_problem = normalize_text(provenance.get("target_problem"))
    evidences = compact_list(provenance.get("supporting_evidences"), max_items=8, max_chars=2200)

    parts = []
    if material_system:
        parts.append(f"材料体系：{material_system}")
    if target_problem:
        parts.append(f"研究问题：{target_problem}")
    if research_focus:
        parts.append(f"研究重点：{research_focus}")
    if evidences:
        parts.append(f"文献片段：\n{evidences}")
    return "\n\n".join(parts).strip()


def style_guidance(style: str) -> str:
    if style == "conclusion_first":
        return (
            "采用结论先行写法：首句直接给出主因或核心判断，不完整复述题干；"
            "随后用2-3个短段落展开机制和证据，不要用模板化免责声明收尾。"
        )
    if style == "mechanism_flow":
        return (
            "采用机制链写法：正文要显出推导层次，优先用2-4个短标签或短段落组织，例如“结构变化：...”“输运/复合：...”“性能结果：...”；"
            "不要写成一整块文字墙。"
        )
    if style == "contrastive_boundary":
        return (
            "采用对比排除写法：先给主因，再用简短对比说明“不是单纯因为A，而是B链条”；"
            "对比要服务于机制解释，不要每次都另起一段模板化写“需要指出”。"
        )
    if style == "parameter_diagnostic":
        return (
            "采用参数诊断写法：可以按参数或证据类型分成短标签，例如“Jsc：...”“FF：...”“Voc/PCE：...”；"
            "每个参数都要解释它对应的物理机制，而不是只罗列数值。"
        )
    return (
        "采用简明专家写法：真正简明，约180-550个中文字符，1-2个自然段；"
        "最多保留2个最关键数值或表征，不展开全套实验细节，不写启发，不加泛化式结尾。"
    )


def build_prompt(item: dict[str, Any], index: int, args: argparse.Namespace) -> tuple[str, str, str, str]:
    provenance = get_provenance(item)
    literature_input = build_literature_input(item)
    input_mode = choose_input_mode(index, args.input_ratio)
    answer_style = choose_answer_style(index)
    expected_input = literature_input if input_mode == "literature_input" else ""
    reasoning_text = normalize_text(item.get("reasoning_text"))
    short_answer = normalize_text(item.get("output"))

    if input_mode == "literature_input":
        mode_rule = """
本条样本类型：文献片段输入。
- 训练样本的 input 已由脚本自动填充为文献片段，你不要改写 input。
- output 可以引用 input 中出现的具体实验现象、参数和数值。
- 原始推理草稿和原始短答案只用于理解回答风格与机制方向；如果其中的数字、型号、参数或实验现象没有出现在训练样本 input 中，不得写入 output。
- output 要体现“读文献片段后做性能归因”的能力。
- 如果确实能从文献片段和机制链中自然抽象出启发，可以在结尾补 1 句克制启发；如果不自然就不要写。
- 启发不能写成新材料预测、实验方案或强推荐。
"""
        source_section = f"""
训练样本 input（由脚本自动写入，供你理解证据范围）：
{expected_input}

原始推理草稿（仅供理解机制方向；其中未出现在训练样本 input 里的数字、型号、参数和实验现象不得复制到 output）：
{reasoning_text}

原始短答案（仅供理解回答风格；其中未出现在训练样本 input 里的数字、型号、参数和实验现象不得复制到 output）：
{short_answer}
"""
    else:
        mode_rule = """
本条样本类型：无输入风格归因。
- 训练样本的 input 必须为空。
- output 开头必须先直接回答原问题的核心判断，不要先写大段“常见误区”或论文背景。
- output 仍然可以保留原始推理草稿或原始短答案中已有的关键数值、参数和表征现象，但不得新增。
- 如果保留原问题中没有出现的具体数值、型号、温度、效率或表征结果，必须写成案例支撑，例如使用“例如，某项研究中...”“以一个报道为例...”“有研究报道...”这类表达。
- 案例数字只能服务于机制解释，不能喧宾夺主；不要把案例数字改写成所有体系的普遍规律。
- output 要像一个稳健的领域专家回答，而不是引用文献片段、复述数据字段或假装正在读取某篇论文。
- 可以保留判断依据；不要额外添加泛化免责声明。
- 不写“设计启发”“可迁移启发”“策略启发”等启发类内容，也不要在结尾强行拔高。
"""
        source_section = f"""
原始推理草稿（可保留其中已有的关键数值、参数和表征现象，但不要新增）：
{reasoning_text}

原始短答案（可保留其中已有的关键数值、参数和表征现象，但不要新增）：
{short_answer}
"""

    return f"""输出 JSON schema：
{{
  "output": "重写后的最终回答"
}}

你正在构建“光伏论文结果与机制归因指令数据集”。
请把给定样本重写成用户最终可见的专业回答，而不是隐藏思维链。

总体要求：
- 输出 300-1800 个中文字符为宜，复杂样本最多不要超过 2000 个中文字符。
- 不输出 <think>、</think>、思考过程、步骤编号。
- 不要机械复述原问题。首句不要完整照搬题干中的“在X体系中引入Y为什么/有什么影响”结构；除非会造成指代不清，首句也尽量不要直接以题干中的材料体系或干预名称开头，而应直接进入判断、机制或结论。
- 不要每条都使用同一种开头，不要反复使用“在...体系中...核心在于...”这一类自问自答句式。
- 允许根据体裁使用短标签、短段落或少量分点，让机制链清楚可见；不要把所有内容写成一整块文字墙。
- 不要每条都使用固定标题；表达要多样，像科研人员自然解释问题。
- 第一段必须先回答用户问的“为什么/如何/是否有效”的核心，不要让背景铺垫、误区排除或证据罗列压过主问题。
- 必须有清楚的归因和机制桥梁，不要把证据没有支持的内容说成确定结论。
- 不要在结尾添加泛化免责声明或模板化边界句，尤其不要写“需要指出的是”“需要说明的是”“不同体系或工艺参数下可能存在差异”“不能直接外推为所有体系的普遍结论”等低信息量句子。
- 不要空泛堆叠“改善、提升、降低、促进”，必须说明中间桥梁。
- 不引入原始推理、短答案、文献片段或逻辑链中没有的新材料、新数值或新机制。
- 只输出 JSON，不要解释。

{mode_rule}

回答体裁：
{style_guidance(answer_style)}

原问题：
{normalize_text(item.get("instruction"))}

{source_section}
""", expected_input, input_mode, answer_style


SUPERSCRIPT_TRANS = str.maketrans(
    {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "⁺": "+",
        "⁻": "-",
        "−": "-",
        "–": "-",
        "—": "-",
    }
)
SUBSCRIPT_TRANS = str.maketrans(
    {
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
        "₊": "+",
        "₋": "-",
    }
)
NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<num>[-+]?\d+(?:\.\d+)?)(?:\s*(?P<unit>%|mV|V|mA|cm|nm|eV|Ω|ohm|h|s|K|°C|℃))?",
    flags=re.IGNORECASE,
)
SUPERSCRIPT_RUN_RE = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+")


def normalize_number_text(text: str) -> str:
    text = normalize_text(text)
    text = text.translate(SUBSCRIPT_TRANS)
    # Superscripts in PV text are usually exponents or unit powers (10⁻², cm²);
    # do not treat them as independent experimental numbers.
    text = SUPERSCRIPT_RUN_RE.sub(" ", text)
    return text.translate(SUPERSCRIPT_TRANS).replace(",", "")


def extract_numbers(text: str) -> set[str]:
    nums = set()
    text = normalize_number_text(text)
    for match in NUMBER_RE.finditer(text):
        raw_num = match.group("num")
        unit = match.group("unit") or ""
        try:
            value = abs(float(raw_num))
            if value in {1, 2, 3, 4, 5} and not unit:
                continue
            # Use absolute value so range separators and exponent signs do not create false mismatches.
            nums.add(f"{value:.8g}")
        except ValueError:
            nums.add(raw_num.lstrip("+-"))
    return nums


CASE_MARKERS = [
    "例如",
    "以某",
    "以一个",
    "有研究报道",
    "有报道",
    "某项研究",
    "一项研究",
    "一个案例",
    "作为案例",
    "作为例证",
    "案例中",
    "报道中",
    "研究中",
]

CASE_BOUNDARY_MARKERS = [
    "不能外推",
    "不应外推",
    "不宜外推",
    "不能直接外推",
    "只能说明",
    "只说明",
    "并不意味着",
    "不意味着",
    "不等于",
    "并不代表",
    "不能被理解为",
    "不是普遍",
    "不应被当作",
    "不能概括为",
]


def contains_any(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def validate_sample(
    sample: dict[str, Any],
    source_context: str,
    input_mode: str,
    strict_number_check: bool = False,
) -> tuple[bool, str]:
    output = normalize_text(sample.get("output"))
    if not output:
        return False, "empty_output"
    answer_style = ""
    meta = sample.get("meta_info")
    if isinstance(meta, dict):
        answer_style = normalize_text(meta.get("answer_style"))
    min_output_len = 80 if answer_style == "concise_expert" else 160
    if len(output) < min_output_len:
        return False, "output_too_short"
    if len(output) > 2400:
        return False, "output_too_long"
    if answer_style == "concise_expert" and len(output) > 850:
        return False, "concise_too_long"
    bad_phrases = [
        "<think",
        "</think",
        "思考完成",
        "根据给定证据",
        "根据上述信息",
        "本文提供的证据",
        "需要指出的是",
        "需要说明的是",
        "这一效率是在特定",
        "不同体系或工艺参数下",
        "不能直接外推为所有体系",
        "不能外推为所有体系的普遍结论",
        "原始推理",
        "原始短答案",
        "论文抽取逻辑链",
    ]
    low = output.lower()
    for phrase in bad_phrases:
        if phrase.lower() in low:
            return False, f"bad_phrase:{phrase}"

    output_nums: set[str] = set()
    if strict_number_check:
        source_nums = extract_numbers(source_context)
        output_nums = extract_numbers(output)
        new_nums = output_nums - source_nums
        if new_nums:
            return False, f"new_numbers:{sorted(new_nums)[:5]}"
    if input_mode == "no_input_style":
        no_input_bad = [
            "文献片段",
            "该论文",
            "设计启发",
            "可迁移启发",
            "策略启发",
            "对类似体系",
            "这提示",
            "这启示",
        ]
        for phrase in no_input_bad:
            if phrase in output:
                return False, f"no_input_bad_phrase:{phrase}"
        if strict_number_check:
            instruction_nums = extract_numbers(sample.get("instruction"))
            case_only_nums = output_nums - instruction_nums
            if case_only_nums and not contains_any(output, CASE_MARKERS):
                return False, f"missing_case_marker_for_numbers:{sorted(case_only_nums)[:5]}"
    return True, "ok"


def heuristic_rewrite(
    item: dict[str, Any],
    expected_input: str,
    input_mode: str,
    answer_style: str,
) -> dict[str, Any]:
    instruction = normalize_text(item.get("instruction"))
    reasoning = normalize_text(item.get("reasoning_text"))
    answer = normalize_text(item.get("output"))

    if input_mode == "literature_input":
        output = (
            f"{answer} 更完整地看，{reasoning} "
            "这一回答应以输入中的文献片段为依据，把材料或工艺变化、结构响应、机制变化和性能结果连接起来。"
            "如果某一环节缺少直接表征，就只写到证据能够支撑的机制层级，不要额外添加免责声明式结尾。"
        )
    else:
        output = (
            f"{answer} 从一般机制上看，{reasoning} "
            "没有文献片段作为输入时，回答应聚焦机制层面的稳健解释；涉及具体论文结论时，只保留原始问题或草稿中已有的数值和现象。"
        )
    return {
        "instruction": instruction,
        "input": expected_input,
        "output": output,
        "meta_info": {"input_mode": input_mode, "answer_style": answer_style},
    }


async def call_teacher(model_api: Any, prompt: str, temperature: float, timeout: int) -> str:
    response = model_api.call(
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        timeout=timeout,
    )
    if asyncio.iscoroutine(response):
        response = await response
    return str(response)


class DirectSiliconAPI:
    def __init__(self, api_name: str):
        from Extraction_project.api_all.silicon_api import call_model as silicon_call_model
        from Extraction_project.config import API_CONFIG

        model_config = API_CONFIG.get(api_name)
        if not model_config:
            raise ValueError(f"Silicon config not found: {api_name}")
        self._call_model = silicon_call_model
        self._model_config = model_config

    async def call(
        self,
        prompt: str = None,
        messages: list[dict[str, str]] | None = None,
        stream: bool = False,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> str:
        return await self._call_model(
            prompt=prompt,
            messages=messages,
            stream=stream,
            timeout=timeout,
            model_config=self._model_config,
        )


async def rewrite_one(
    index: int,
    item: dict[str, Any],
    model_api: Any | None,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any] | None:
    instruction = normalize_text(item.get("instruction"))
    if not instruction:
        print(f"[skip] #{index}: missing instruction")
        return None

    prompt, expected_input, input_mode, answer_style = build_prompt(item, index, args)
    if input_mode == "no_input_style":
        source_context = "\n".join(
            [
                instruction,
                normalize_text(item.get("reasoning_text")),
                normalize_text(item.get("output")),
            ]
        )
    else:
        source_context = f"{instruction}\n{expected_input}".strip()

    if args.heuristic:
        sample = heuristic_rewrite(item, expected_input, input_mode, answer_style)
        ok, reason = validate_sample(sample, source_context, input_mode, args.strict_number_check)
        if not ok:
            print(f"[warn] heuristic sample #{index} failed validation: {reason}")
        return sample

    assert model_api is not None
    last_error = ""
    async with semaphore:
        for attempt in range(1, args.max_retries + 1):
            try:
                raw = await call_teacher(model_api, prompt, args.temperature, args.timeout)
                obj = extract_json_object(raw)
                sample = {
                    "instruction": instruction,
                    "input": expected_input,
                    "output": normalize_output_text(obj.get("output")),
                    "meta_info": {
                        "source_index": index,
                        "input_mode": input_mode,
                        "answer_style": answer_style,
                    },
                }
                ok, reason = validate_sample(sample, source_context, input_mode, args.strict_number_check)
                if not ok:
                    raise ValueError(reason)
                return sample
            except Exception as exc:
                last_error = str(exc)
                print(f"[retry] #{index} attempt {attempt}/{args.max_retries}: {last_error}")
                await asyncio.sleep(min(2 * attempt, 8))
    print(f"[fail] #{index}: {last_error}")
    return None


async def run(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.source, limit=args.limit)
    print(f"loaded={len(rows)} source={args.source}")
    indexed_rows = list(enumerate(rows, start=1))
    if args.indices:
        selected_indices = parse_indices(args.indices)
        indexed_rows = [(i, item) for i, item in indexed_rows if i in selected_indices]
        print(f"indices={sorted(selected_indices)} selected={len(indexed_rows)}")

    model_api = None
    if not args.heuristic:
        try:
            from Extraction_project.llm_api.model_api_factory import get_model_api

            model_api = get_model_api(args.api)
        except Exception as exc:
            if not args.api.lower().startswith("silicon"):
                raise
            print(f"[warn] get_model_api failed, fallback to DirectSiliconAPI: {exc}")
            model_api = DirectSiliconAPI(args.api)

    args.target.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)
    completed_indices = load_completed_indices(args.target) if args.resume else set()
    if completed_indices:
        print(f"resume=on completed={len(completed_indices)} target={args.target}")

    done = 0
    failed = 0
    skipped = 0
    mode = "a" if args.resume else "w"
    with args.target.open(mode, encoding="utf-8") as out:
        tasks = [
            rewrite_one(i, item, model_api, args, semaphore)
            for i, item in indexed_rows
            if i not in completed_indices
        ]
        skipped = len(indexed_rows) - len(tasks)
        total_to_run = len(tasks)
        if skipped:
            print(f"skipped_completed={skipped}")
        for coro in asyncio.as_completed(tasks):
            sample = await coro
            if sample is None:
                failed += 1
                continue
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            out.flush()
            done += 1
            if done % 10 == 0 or done == total_to_run:
                print(f"progress done={done} skipped={skipped} failed={failed}")
    print(f"saved={args.target} done={done} skipped={skipped} failed={failed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite PV CoT data into PV result-and-mechanism attribution SFT data."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--indices",
        default="",
        help="Optional comma-separated 1-based source indices or ranges, e.g. 18,42-45.",
    )
    parser.add_argument("--api", default="silicon_deepseek_r1_0528")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=500)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--input-ratio",
        type=float,
        default=0.6,
        help="Fraction of rewritten samples that receive literature evidence in input, repeated every 10 rows.",
    )
    parser.add_argument(
        "--heuristic",
        action="store_true",
        help="Do not call an LLM; use a mechanical preview rewrite for debugging only.",
    )
    parser.add_argument(
        "--strict-number-check",
        action="store_true",
        help="Enable regex-based number consistency checks. Off by default because chemistry/unit notation is noisy.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
