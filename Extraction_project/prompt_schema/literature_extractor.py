import json
import logging
import os
import time
from typing import Optional, Type, List, Tuple, Any
from pydantic import ValidationError, BaseModel
import asyncio
from asyncio import TimeoutError
import re

from Extraction_project.prompt_schema.prompt_templates import LiteratureExtractPrompt
from Extraction_project.prompt_schema.data_schemas import PVLiteratureExtractSchema
from Extraction_project.llm_api.model_api_interface import BaseModelAPI


# ----------------------------------------------------
# Debug 输出目录：保存模型返回的无效 JSON / 空响应 / Schema 错误
# ----------------------------------------------------
def _get_project_root() -> str:
    """
    当前文件通常位于：Extraction_project/prompt_schema/literature_extractor.py
    因此向上 3 层得到项目根目录。
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


PROJECT_ROOT = _get_project_root()
FAILED_RAW_RESPONSE_DIRECTORY = os.path.join(PROJECT_ROOT, "data", "failed_raw_responses")


def _safe_filename(name: str) -> str:
    """Windows/Linux 通用安全文件名。"""
    name = name or "unknown_file"
    return re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", name).strip("._ ") or "unknown_file"


def _save_invalid_debug_artifacts(
        file_name: str,
        raw_response: Any,
        stage: str,
        error_msg: str,
        parsed_obj: Any = None,
        extra_text: Optional[str] = None
) -> None:
    """
    保存失败时的调试材料：
    1. 原始模型输出 .raw_response.txt
    2. 错误说明 .error.txt
    3. 如果已经解析成 dict 但 Schema 失败，则额外保存 .parsed_json.json

    注意：这个函数只保存调试副本，不影响主流程；即使保存失败也不会让抽取程序崩。
    """
    try:
        os.makedirs(FAILED_RAW_RESPONSE_DIRECTORY, exist_ok=True)

        safe_name = _safe_filename(file_name)
        timestamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
        safe_stage = _safe_filename(stage)
        prefix = os.path.join(FAILED_RAW_RESPONSE_DIRECTORY, f"{safe_name}_{timestamp}_{safe_stage}")

        raw_path = prefix + ".raw_response.txt"
        error_path = prefix + ".error.txt"

        with open(raw_path, "w", encoding="utf-8") as f:
            if isinstance(raw_response, str):
                f.write(raw_response)
            else:
                f.write(repr(raw_response))

        with open(error_path, "w", encoding="utf-8") as f:
            f.write(f"file_name: {file_name}\n")
            f.write(f"stage: {stage}\n")
            f.write(f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"error: {error_msg}\n")
            if extra_text:
                f.write("\n--- extra ---\n")
                f.write(extra_text)

        if parsed_obj is not None:
            parsed_path = prefix + ".parsed_json.json"
            with open(parsed_path, "w", encoding="utf-8") as f:
                json.dump(parsed_obj, f, indent=4, ensure_ascii=False, default=str)

        logging.info(f"[{file_name}]  已保存无效模型输出调试文件: {raw_path}")
    except Exception as debug_err:
        logging.warning(f"[{file_name}] [WARN] 保存无效模型输出失败: {debug_err}")


class LiteratureExtractor:
    """
    文献抽取器：支持嵌套结构
    整合 Prompt(含文件名注入)、Schema(含强校验) 和 模型API(含JSON修复)

    新增：
    - API 空响应、JSON 解析失败、Schema 校验失败时，会保存 raw response 到：
      data/failed_raw_responses/
    """

    def __init__(
            self,
            model_api: BaseModelAPI,
            schema_cls: Type[BaseModel] = PVLiteratureExtractSchema
    ):
        self.model_api = model_api
        self.schema_cls = schema_cls
        # 初始化 Prompt 构建器
        self.prompt_builder = LiteratureExtractPrompt(schema_cls=self.schema_cls)

    async def extract(
            self,
            literature_text: str,
            file_name: str = "未知文件",
            timeout: Optional[int] = 60
    ) -> Optional[PVLiteratureExtractSchema]:
        """
        执行单次抽取任务
        """
        # 1. 构建Prompt
        prompt = self.prompt_builder.get_full_prompt(literature_text, file_name=file_name)

        # 2. 调用模型API
        raw_response = None
        try:
            raw_response = await asyncio.wait_for(
                self.model_api.call(
                    prompt=prompt,
                    stream=False,
                    timeout=timeout
                ),
                timeout=timeout + 5
            )

            if not raw_response or not isinstance(raw_response, str):
                logging.warning(f"[{file_name}] API响应为空或类型错误")
                _save_invalid_debug_artifacts(
                    file_name=file_name,
                    raw_response=raw_response,
                    stage="empty_or_invalid_api_response",
                    error_msg=f"API响应为空或类型错误，type={type(raw_response)}"
                )
                return None

        except TimeoutError:
            logging.warning(f"[{file_name}] 模型API调用超时 (>{timeout}s)")
            _save_invalid_debug_artifacts(
                file_name=file_name,
                raw_response=raw_response,
                stage="api_timeout",
                error_msg=f"模型API调用超时 (>{timeout}s)"
            )
            return None
        except Exception as e:
            logging.error(f"[{file_name}] 模型API调用失败：{str(e)}", exc_info=True)
            _save_invalid_debug_artifacts(
                file_name=file_name,
                raw_response=raw_response,
                stage="api_call_failed",
                error_msg=str(e)
            )
            return None

        # 3. JSON 解析与清洗
        response_dict = None

        # 定义一个内部函数来处理反斜杠修复 (化学式/LaTeX保护)
        def robust_json_load(text):
            try:
                # 尝试直接解析
                return json.loads(text)
            except json.JSONDecodeError:
                # 失败则尝试暴力修复反斜杠 (例如 \alpha -> \\alpha)
                # 注意：这对于嵌套结构中的 "mechanism" 字段特别重要
                fixed = text.replace('\\', '\\\\')
                return json.loads(fixed)

        try:
            # A. 基础清洗
            clean_text = raw_response.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()

            # B. 尝试解析
            response_dict = robust_json_load(clean_text)

        except (json.JSONDecodeError, ValueError) as e:
            logging.warning(f"[{file_name}] 初次JSON解析失败: {e} | 尝试正则提取子串...")

            # 备用方案：正则提取最外层 JSON 对象
            # 这对于嵌套结构依然有效，因为最外层依然是 {}
            try:
                match = re.search(r'(\{.*\})', raw_response, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    response_dict = robust_json_load(json_str)
                else:
                    logging.error(f"[{file_name}] [ERROR] 无法解析 JSON。")
                    _save_invalid_debug_artifacts(
                        file_name=file_name,
                        raw_response=raw_response,
                        stage="json_no_object_found",
                        error_msg=f"初次JSON解析失败且未找到最外层 JSON 对象: {e}"
                    )
                    return None
            except Exception as e_inner:
                logging.error(f"[{file_name}] [ERROR] 二次解析失败: {e_inner}")
                _save_invalid_debug_artifacts(
                    file_name=file_name,
                    raw_response=raw_response,
                    stage="json_parse_failed",
                    error_msg=f"初次解析错误: {e}\n二次解析错误: {e_inner}"
                )
                return None

        def robust_json_parse(llm_output: str) -> dict:
            """解析可能包含控制字符或附加文本的模型 JSON 输出"""
            if not llm_output:
                return {}

            # 1. 剥离 Markdown 保护壳 (```json ... ```)
            json_str = llm_output
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            json_str = json_str.strip()

            # 2. 核心清洗：处理未转义的控制字符 (替换真实换行符为空格，因为在evidence里换行没意义)
            # 这步能解决 90% 的 Invalid control character 报错
            json_str = re.sub(r'(?<!\\)\n', ' ', json_str)
            json_str = re.sub(r'(?<!\\)\t', ' ', json_str)

            # 3. 使用 strict=False 解析含控制字符的 JSON
            try:
                return json.loads(json_str, strict=False)
            except json.JSONDecodeError as e:
                # 如果还是失败，可能是模型输出截断或双引号搞乱了，记录错误并返回空字典让外层重试
                print(f"[WARN] JSON 解析失败: {e}\n问题字符串截取: {json_str[:200]}...")
                return {}

        # 4. Schema 校验 (Pydantic)
        try:
            # 关键修改) 强制覆盖/注入文件名
            # 使用当前处理文件名覆盖模型输出中的文件名字段
            # 这一步保证了最终 Knowledge Graph 里溯源的绝对准确性
            if isinstance(response_dict, dict):
                response_dict['extracted_file'] = file_name

            # 实例化 Pydantic 对象
            # 嵌套结构的校验会自动递归进行 (Root -> Molecule -> Group)
            extract_result = self.schema_cls(**response_dict)
            return extract_result

        except ValidationError as e:
            error_msgs = []
            for err in e.errors():
                loc_tuple = err.get('loc', ())
                loc = "->".join([str(x) for x in loc_tuple])
                msg = err.get('msg', '未知错误')
                error_msgs.append(f"[{loc}]: {msg}")

            error_text = '; '.join(error_msgs)
            logging.error(f"[{file_name}] [ERROR] Schema校验失败: {error_text}")
            _save_invalid_debug_artifacts(
                file_name=file_name,
                raw_response=raw_response,
                stage="schema_validation_failed",
                error_msg=error_text,
                parsed_obj=response_dict
            )
            return None

    async def batch_extract(
            self,
            inputs: List[Tuple[str, str]],  # 修改为 [(text, filename), ...]
            timeout: Optional[int] = 60
    ) -> List[Optional[PVLiteratureExtractSchema]]:
        """
        批量接口优化：支持传入文件名列表，方便批量溯源
        """
        tasks = []
        for text, fname in inputs:
            tasks.append(self.extract(text, file_name=fname, timeout=timeout))

        return await asyncio.gather(*tasks, return_exceptions=True)
