import asyncio
import logging
import os
import glob
import json
import time
import uuid
import sqlite3  # 用于轻量级状态账本)
import random  # 用于重试机制的抖动)
from typing import Dict, Any, List, Type
import sys
import signal
import aiofiles
import hashlib  # 哈希分配核心)

# ----------------------------------------------------
# 1. 路径与基础配置
# ----------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(script_dir)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from Extraction_project.llm_api.model_api_factory import get_model_api, BaseModelAPI
    from Extraction_project.config import API_CONFIG, SOURCE_DIRECTORY
    from Extraction_project.prompt_schema.literature_extractor import LiteratureExtractor
    from Extraction_project.prompt_schema.data_schemas import PVLiteratureExtractSchema
    from Extraction_project.prompt_schema.prompt_templates import LiteratureExtractPrompt
except ImportError as e:
    print(f"导入失败! 路径: {PROJECT_ROOT} | 错误: {e}")
    exit(1)

# ----------------------------------------------------
# 2. 核心配置
# ----------------------------------------------------
APIS_TO_RUN_EXTRACTION = ["silicon_deepseek_r1_0528"]
LLM_TIMEOUT = 1000
LITERATURE_FILE_EXTENSIONS = [".md", ".txt"]
OUTPUT_DIRECTORY = os.path.join(PROJECT_ROOT, "data", "extractor_results")

# 数据库路径与重试配置)
DB_PATH = os.path.join(PROJECT_ROOT, "data", "extraction_state.db")
MAX_RETRIES = 3  # 最大重试次数
# ----------------------------------------------------

# 运行状态
current_batch_task = None
progress_counter = 0
total_files_to_process = 0
progress_lock = asyncio.Lock()
file_io_lock = asyncio.Lock()
db_write_lock = asyncio.Lock()  # 保护SQLite并发写入)


def handle_sigint(signum, frame):
    print("\n[WARN]  收到终止信号，正在安全清理任务...")
    if current_batch_task and not current_batch_task.done():
        current_batch_task.cancel()
    print("任务清理完成，即将退出程序")


# ----------------------------------------------------
# 3. SQLite 状态账本管理器
# ----------------------------------------------------
def init_db():
    """初始化 SQLite 数据库，创建状态表"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS file_state
                     (file_path TEXT PRIMARY KEY, status TEXT, api_provider TEXT, update_time TEXT)''')
        conn.commit()


def get_completed_files() -> set:
    """获取已经成功处理或明确跳过的文件集合，用于瞬间完成断点续传"""
    if not os.path.exists(DB_PATH):
        return set()
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT file_path FROM file_state WHERE status IN ('extracted', 'skipped')")
        return set(row[0] for row in c.fetchall())


def _update_file_state_sync(path: str, status: str, api_provider: str):
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        c = conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO file_state (file_path, status, api_provider, update_time) VALUES (?, ?, ?, ?)",
            (path, status, api_provider, time.strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()


async def update_file_state_async(path: str, status: str, api_provider: str):
    """异步包装的写入，防止阻塞事件循环"""
    async with db_write_lock:
        await asyncio.to_thread(_update_file_state_sync, path, status, api_provider)


# ----------------------------------------------------
# 4. 结果写入器
# ----------------------------------------------------
async def save_extraction_to_new_file(
        output_path: str,
        extraction_result: PVLiteratureExtractSchema,
        api_provider: str,
        token_usage: dict = None  # 可选的令牌用量记录
):
    async with file_io_lock:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                output_data = {
                    "extraction_model": api_provider,
                    "extraction_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "token_usage": token_usage or {"prompt_tokens": 0, "completion_tokens": 0},
                    "result": extraction_result.to_dict()
                }
                await f.write(json.dumps(output_data, indent=4, ensure_ascii=False))
        except Exception as e:
            logging.error(f"[ERROR] 异步写入文件失败: {output_path} | 错误: {e}", exc_info=True)
            raise


# ----------------------------------------------------------------
# 5. 核心处理器：指数退避重试
# ----------------------------------------------------------------
async def process_single_extraction(
        path: str,
        file_idx: int,
        total_files: int,
        batch_id: str,
        semaphore: asyncio.Semaphore,
        api_provider: str,
        model_api: BaseModelAPI,
        min_time_per_request: float,
        source_dir_abs: str
):
    file_start_time = time.time()
    file_name = os.path.basename(path)

    try:
        relative_path = os.path.relpath(path, source_dir_abs)
        relative_dir = os.path.dirname(relative_path)
        base_name = os.path.splitext(file_name)[0]
        output_filename = f"{base_name}_extraction_{api_provider}.json"
        output_path = os.path.join(OUTPUT_DIRECTORY, api_provider, relative_dir, output_filename)

        async with aiofiles.open(path, 'r', encoding='utf-8') as f:
            text = await f.read()
        if not text.strip():
            raise ValueError("文件内容为空")

        extractor = LiteratureExtractor(model_api=model_api, schema_cls=PVLiteratureExtractSchema)
        result_schema = None

        # 增加网络重试环)
        for attempt in range(MAX_RETRIES):
            try:
                async with semaphore:
                    call_start_time = time.monotonic()
                    logging.info(
                        f"[Batch:{batch_id}] [File:{file_name}]  API调用中 (尝试 {attempt + 1}/{MAX_RETRIES})...")

                    result_schema = await extractor.extract(
                        literature_text=text,
                        file_name=file_name,
                        timeout=LLM_TIMEOUT
                    )

                    if not result_schema:
                        raise ValueError("模型返回了无效结果 (Result is None)")

                    # 速率限制
                    if min_time_per_request > 0:
                        call_duration = time.monotonic() - call_start_time
                        if call_duration < min_time_per_request:
                            await asyncio.sleep(min_time_per_request - call_duration)

                    break  # 成功跳出重试循环

            except Exception as api_err:
                if attempt < MAX_RETRIES - 1:
                    # 指数退避: 2s, 4s, 8s... 并加上随机抖动防止并发雪崩
                    sleep_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                    logging.warning(f"[WARN] API请求异常: {file_name} | {api_err} | {sleep_time:.1f}s 后重试...")
                    await asyncio.sleep(sleep_time)
                else:
                    raise api_err  # 重试耗尽，抛出异常

        # 保存结果
        await save_extraction_to_new_file(
            output_path=output_path,
            extraction_result=result_schema,
            api_provider=api_provider,
            token_usage={"prompt_tokens": 0, "completion_tokens": 0}
        )

        cost_time = time.time() - file_start_time
        logging.info(f"[Batch:{batch_id}] [File:{file_name}]  抽取成功 | 耗时: {cost_time:.2f}s")

        # 写入成功状态到账本)
        await update_file_state_async(path, "extracted", api_provider)

        return {
            "success": True, "status": "extracted", "original_file_path": path,
            "output_file_path": output_path, "api_provider": api_provider, "cost_time": round(cost_time, 2)
        }

    except Exception as e:
        cost_time = time.time() - file_start_time
        logging.error(f"[Batch:{batch_id}] [File:{file_name}] [ERROR] 文件处理失败 | 错误：{str(e)}", exc_info=True)

        # 写入失败状态到账本)
        await update_file_state_async(path, "failed", api_provider)

        return {
            "success": False, "status": "failed", "original_file_path": path,
            "error": str(e), "cost_time": round(cost_time, 2)
        }


# ----------------------------------------------------------------
# 6. 消费者：处理任务并确认队列完成
# ----------------------------------------------------------------
async def consumer_worker(queue, batch_id, results_list, failed_files_list, api_provider, model_api, semaphore,
                          min_time_per_request, source_dir_abs):
    global progress_counter, total_files_to_process, progress_lock
    while True:
        try:
            path = await queue.get()
            if path is None:
                break

            async with progress_lock:
                progress_counter += 1
                current_index = progress_counter
            total_str = total_files_to_process if total_files_to_process > 0 else '?'

            res = await process_single_extraction(
                path, current_index, total_files_to_process, batch_id,
                semaphore, api_provider, model_api, min_time_per_request, source_dir_abs
            )

            results_list.append(res)
            if not res["success"]:
                failed_files_list.append(res["original_file_path"])

        except Exception as e:
            logging.error(f"消费者任务 ({api_provider}) 失败: {e}", exc_info=True)
        finally:
            queue.task_done()


# ----------------------------------------------------------------
# 7. 生产者：预检任务状态并入队
# ----------------------------------------------------------------
async def producer(api_queues: Dict[str, asyncio.Queue]):
    global total_files_to_process, progress_lock
    total_files_found = 0
    source_dir_abs = os.path.abspath(SOURCE_DIRECTORY)

    # 读取已完成文件集合)
    completed_set = get_completed_files()
    if completed_set:
        logging.info(f"断点续传：跳过 {len(completed_set)} 个已完成文件")

    try:
        top_level_dirs = [d for d in os.listdir(source_dir_abs) if os.path.isdir(os.path.join(source_dir_abs, d))]
    except Exception as e:
        logging.error(f"扫描源目录失败: {e}")
        return 0

    api_list = sorted(list(api_queues.keys()))
    if not top_level_dirs or not api_list:
        return 0

    for dir_name in sorted(top_level_dirs):
        # 哈希分配)
        hash_bytes = hashlib.sha256(dir_name.encode('utf-8')).digest()
        hash_int = int.from_bytes(hash_bytes, 'little')
        api_to_use = api_list[hash_int % len(api_list)]
        dir_path = os.path.join(source_dir_abs, dir_name)

        files_in_this_dir = 0
        for ext in LITERATURE_FILE_EXTENSIONS:
            search_pattern = os.path.join(dir_path, '**', f"*{ext}")
            for path in glob.iglob(search_pattern, recursive=True):
                if os.path.isfile(path):
                    # 入队前查询已完成文件集合，避免重复读取状态文件)
                    if path in completed_set:
                        continue

                    await api_queues[api_to_use].put(path)
                    files_in_this_dir += 1
                    total_files_found += 1

    async with progress_lock:
        total_files_to_process = total_files_found

    logging.info(f"--- 生产者完成：总计入队 {total_files_found} 个待处理文件 ---")
    return total_files_found


# ----------------------------------------------------------------
# 8. 主函数 (Orchestrator)
# ----------------------------------------------------------------
async def main():
    print("\n [DEBUG] 正在运行批量文献抽取... \n")
    global current_batch_task
    signal.signal(signal.SIGINT, handle_sigint)

    source_dir_abs = os.path.abspath(SOURCE_DIRECTORY)
    output_dir_abs = os.path.abspath(OUTPUT_DIRECTORY)
    os.makedirs(output_dir_abs, exist_ok=True)

    # 初始化状态账本)
    init_db()

    start_time = time.time()
    batch_id = str(uuid.uuid4())[:8]

    try:
        model_api_clients = {}
        api_semaphores = {}
        api_min_times = {}
        api_queues = {}

        for api_name in APIS_TO_RUN_EXTRACTION:
            # ... (保持原有的初始化逻辑) ...
            concurrency = API_CONFIG[api_name].get("concurrency", 1)
            rpm_limit = API_CONFIG[api_name].get("rpm_limit")
            model_api_clients[api_name] = get_model_api(api_name)
            api_semaphores[api_name] = asyncio.Semaphore(concurrency)
            api_queues[api_name] = asyncio.Queue()

            if rpm_limit and rpm_limit > 0 and concurrency > 0:
                api_min_times[api_name] = 60.0 / (rpm_limit / concurrency)
            else:
                api_min_times[api_name] = 0.0

        results_list = []
        failed_files_list = []
        all_consumer_tasks = []

        for api_name in APIS_TO_RUN_EXTRACTION:
            concurrency = API_CONFIG[api_name].get("concurrency", 1)
            for _ in range(concurrency):
                all_consumer_tasks.append(
                    asyncio.create_task(consumer_worker(
                        api_queues[api_name], batch_id, results_list, failed_files_list,
                        api_name, model_api_clients[api_name], api_semaphores[api_name],
                        api_min_times[api_name], source_dir_abs
                    ))
                )

        producer_task = asyncio.create_task(producer(api_queues))
        current_batch_task = asyncio.gather(producer_task, *all_consumer_tasks)

        total_files = await producer_task
        if total_files == 0:
            logging.warning("没有需要处理的新文件 (或者全部已通过断点续传跳过)。")
            for task in all_consumer_tasks: task.cancel()
            return

        for api_name in api_queues:
            await api_queues[api_name].join()

        for api_name in APIS_TO_RUN_EXTRACTION:
            concurrency = API_CONFIG[api_name].get("concurrency", 1)
            for _ in range(concurrency):
                await api_queues[api_name].put(None)

        await asyncio.gather(*all_consumer_tasks)

        # 汇总并打印运行报告
        success_count = total_files - len(failed_files_list)
        total_cost_time = time.time() - start_time
        print(
            f"\n 批量抽取完成 | 总耗时：{total_cost_time:.2f} 秒 | 成功: {success_count} | 失败: {len(failed_files_list)}")

    except asyncio.CancelledError:
        print("\n 任务已被手动终止")
    except Exception as e:
        logging.error(f"[ERROR] 运行异常: {e}", exc_info=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] - %(message)s")
    asyncio.run(main())
