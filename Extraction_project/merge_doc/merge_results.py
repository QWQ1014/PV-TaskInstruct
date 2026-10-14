import json
import os
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT_PATH = os.getenv("PV_MERGE_INPUT_DIR", str(REPO_ROOT / "Extraction_project" / "data" / "extractor_results"))
OUTPUT_SAVE_PATH = os.getenv("PV_MERGED_RESULTS_DIR", str(REPO_ROOT / "outputs" / "merged_results"))
OVERRIDE_MAPPING_PATH = Path(__file__).with_name('entity_type_overrides.json')

BASE_ENTITY_TO_META = {
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
    "Optical_Property": "L3_Mechanism_Physics",
    "Spectral_Response": "L4_Device_Parameter",
}

ENTITY_TO_META = dict(BASE_ENTITY_TO_META)
global_unknown_types = set()


def load_entity_to_meta_mapping() -> dict:
    mapping = dict(BASE_ENTITY_TO_META)
    if OVERRIDE_MAPPING_PATH.exists():
        try:
            with open(OVERRIDE_MAPPING_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    if str(k).strip() and str(v).strip():
                        mapping[str(k).strip()] = str(v).strip()
        except Exception:
            pass
    return mapping


def refresh_entity_to_meta_mapping() -> dict:
    global ENTITY_TO_META
    ENTITY_TO_META = load_entity_to_meta_mapping()
    return ENTITY_TO_META


def get_all_json_files_recursively(root_folder):
    json_files = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.json'):
                json_files.append(os.path.join(dirpath, filename))
    return json_files


def select_best_text(text_list):
    valid_texts = [
        t for t in text_list
        if t and t not in ["未明确提及", "未提及", "None", ""] and len(t) > 5
    ]
    if not valid_texts:
        return "未明确提及"
    return max(valid_texts, key=len)


def resolve_vote(vote_list):
    valid_votes = [
        v for v in vote_list
        if v and "无法判断" not in v and v not in ["未明确提及", "未提及", "None", "Other", "other", ""]
    ]
    if not valid_votes:
        return "未明确提及"
    return Counter(valid_votes).most_common(1)[0][0]


def merge_graph_chunks(file_path_list, doc_name):
    refresh_entity_to_meta_mapping()
    candidate_titles = []
    candidate_focuses = []
    candidate_materials = []
    candidate_problems = []

    merged_entities_map = {}
    pe_counter = 1

    # 1. 初始化结果结构，适配双字段关系
    merged_relations_map = defaultdict(lambda: {
        "subject_id": "",
        "raw_predicate": "",
        "predicate_group": "",
        "object_id": "",
        "evidences": set()
    })

    for file_path in file_path_list:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            block = data.get("result", data)
            if not block:
                continue
        except Exception as e:
            print(f"[ERROR] 读取失败: {file_path} | {e}")
            continue

        title = block.get("article_title_author")
        if title:
            candidate_titles.append(title)
        focus = block.get("research_focus")
        if focus:
            candidate_focuses.append(focus)
        material = block.get("material_system")
        if material:
            candidate_materials.append(material)
        problem = block.get("target_problem")
        if problem:
            candidate_problems.append(problem)

        entities = block.get("entities", [])
        relations = block.get("relations", [])
        chunk_id_map = {}

        for ent in entities:
            local_id = ent.get("entity_id")
            raw_name = ent.get("entity_name", "").strip()
            entity_type = ent.get("entity_type", "Other").strip()

            if not local_id or not raw_name or raw_name in ["未明确提及", "未提及"]:
                continue

            norm_name = raw_name.lower()
            if norm_name not in merged_entities_map:
                new_pe_id = f"PE_{pe_counter:03d}"
                pe_counter += 1
                meta_layer = ENTITY_TO_META.get(entity_type)
                if not meta_layer:
                    meta_layer = "Unknown"
                    if entity_type not in ["Other", "unknown", "None", ""]:
                        global_unknown_types.add(entity_type)
                merged_entities_map[norm_name] = {
                    "entity_id": new_pe_id,
                    "entity_name": raw_name,
                    "entity_type": entity_type,
                    "meta_layer": meta_layer
                }
            chunk_id_map[local_id] = merged_entities_map[norm_name]["entity_id"]

        for rel in relations:
            old_sub = rel.get("subject_id")

            # 2. 读取 raw_predicate 和 predicate_group，兼容旧的 predicate
            raw_predicate = rel.get("raw_predicate", rel.get("predicate", "")).strip()
            predicate_group = rel.get("predicate_group", "causal_change").strip()

            old_obj = rel.get("object_id")
            evidence = rel.get("evidence", "").strip()

            new_sub = chunk_id_map.get(old_sub)
            new_obj = chunk_id_map.get(old_obj)
            if new_sub and new_obj:
                # 3. 使用 raw_predicate 作为连线去重的哈希签名
                rel_sig = f"{new_sub}||{raw_predicate}||{new_obj}"
                target = merged_relations_map[rel_sig]
                target["subject_id"] = new_sub
                target["raw_predicate"] = raw_predicate
                target["predicate_group"] = predicate_group
                target["object_id"] = new_obj
                if evidence and evidence not in ["未明确提及", "未提及"]:
                    target["evidences"].add(evidence)

    final_title = select_best_text(candidate_titles)
    if final_title == "未明确提及":
        final_title = doc_name
    final_focus = select_best_text(candidate_focuses)
    final_material = resolve_vote(candidate_materials)
    final_problem = resolve_vote(candidate_problems)

    final_entities = list(merged_entities_map.values())
    final_relations = []
    for rel_data in merged_relations_map.values():
        # 4. 输出时保留双字段结构
        final_relations.append({
            "subject_id": rel_data["subject_id"],
            "raw_predicate": rel_data["raw_predicate"],
            "predicate_group": rel_data["predicate_group"],
            "object_id": rel_data["object_id"],
            "evidences": list(rel_data["evidences"])
        })

    return {
        "filename": doc_name,
        "article_title_author": final_title,
        "material_system": final_material,
        "target_problem": final_problem,
        "research_focus": final_focus,
        "total_entities": len(final_entities),
        "total_relations": len(final_relations),
        "entities": final_entities,
        "relations": final_relations
    }


def process_by_top_level_folder(root_dir, output_save_dir):
    if not os.path.exists(root_dir):
        print(f"[ERROR] 根目录不存在: {root_dir}")
        return
    if not os.path.exists(output_save_dir):
        os.makedirs(output_save_dir)

    items = os.listdir(root_dir)
    paper_folders = [item for item in items if os.path.isdir(os.path.join(root_dir, item))]
    print(f" 开始合并处理 {len(paper_folders)} 个文献文件夹...")

    for folder_name in paper_folders:
        full_folder_path = os.path.join(root_dir, folder_name)
        all_json_paths = get_all_json_files_recursively(full_folder_path)
        if not all_json_paths:
            continue
        merged_data = merge_graph_chunks(all_json_paths, folder_name)
        safe_filename = "".join(
            [c for c in folder_name if c.isalpha() or c.isdigit() or c in (' ', '-', '_', '.')]).rstrip()
        save_path = os.path.join(output_save_dir, f"{safe_filename}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=4)

    print("\n" + "=" * 60)
    print(f" 合并完成！所有单篇文献的最终图谱数据已保存在: \n{output_save_dir}")
    print("=" * 60)
    if global_unknown_types:
        print("\n 发现大模型自创的新实体类别 (Open IE 生效)！ ")
        print("请将以下类别补充到实体类型映射中，并分配 L1~L5 层级：")
        for t in global_unknown_types:
            print(f'  - "{t}": "L?_XXX",')
    else:
        print("\n 没有发现越界的新类别，均符合预设类型体系。")


if __name__ == "__main__":
    process_by_top_level_folder(INPUT_ROOT_PATH, OUTPUT_SAVE_PATH)
