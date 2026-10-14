from pydantic import BaseModel, Field, field_validator
from typing import List

# ==========================================
# 5D 映射字典
# ==========================================
ENTITY_TO_META = {
    # L1
    "Material": "L1_Intervention",
    "Additive_or_Dopant": "L1_Intervention",
    "Interface_Layer": "L1_Intervention",
    "Process_Condition": "L1_Intervention",

    # L2
    "Structural_Morphology_Change": "L2_Structure_Composition",
    "Chemical_Interaction": "L2_Structure_Composition",

    # L3
    "Defect_State": "L3_Mechanism_Physics",
    "Carrier_Dynamics": "L3_Mechanism_Physics",
    "Recombination": "L3_Mechanism_Physics",

    # L4
    "Device_Parameter": "L4_Device_Parameter",

    # L5
    "Performance_Metric": "L5_Performance"
}


# ==========================================
# 实体单元
# ==========================================
class EntityUnit(BaseModel):
    entity_id: str = Field(
        default="E_UNKNOWN",
        description="实体唯一ID，建议格式如 E1, E2"
    )
    entity_name: str = Field(
        default="未明确提及",
        description=(
            "实体名称。"
            "L2/L3 尽量带状态或方向，如 'reduced trap density', 'suppressed nonradiative recombination'。"
            "L4/L5 若论文中明确给出数值，则尽量保留数值，如 'Voc (1.12 V)', 'PCE (20.27%)'。"
            "一个实体尽量只表达一个概念。"
        )
    )
    entity_type: str = Field(
        default="Material",
        description=(
            "优先使用以下类别："
            "Material, Additive_or_Dopant, Interface_Layer, Process_Condition, "
            "Structural_Morphology_Change, Chemical_Interaction, Defect_State, "
            "Carrier_Dynamics, Recombination, Device_Parameter, Performance_Metric。"
            "若确实不适配，可使用 Capitalized_Words 形式新建类别。"
        )
    )

    @field_validator("entity_type", mode="before")
    @classmethod
    def normalize_entity_type(cls, v):
        if not isinstance(v, str):
            return "Material"
        v = v.strip()
        v = "_".join([word.capitalize() for word in v.replace("_", " ").split()])

        if "Performance" in v or "Metric" in v or v in ["Pce", "Voc", "Jsc", "Ff"]:
            return "Performance_Metric"
        if "Parameter" in v:
            return "Device_Parameter"

        return v

    @field_validator("entity_name", mode="before")
    @classmethod
    def enforce_string(cls, v):
        if v is None:
            return "未明确提及"
        v_clean = str(v).strip()
        return v_clean.strip(";")


# ==========================================
# 关系单元：改为“原始关系短语 + 粗粒度关系类别”
# ==========================================
class RelationUnit(BaseModel):
    subject_id: str = Field(default="")
    raw_predicate: str = Field(
        default="induces",
        description=(
            "短的、贴近论文原文的关系短语。"
            "例如：'induces', 'suppresses', 'promotes hole extraction', "
            "'helps maintain', 'reduces trap density', 'enables transfer'。"
            "不要写过长句子。"
        )
    )
    predicate_group: str = Field(
        default="causal_change",
        description=(
            "粗粒度关系类别。建议使用："
            "structural_formation, causal_change, positive_effect, negative_effect, "
            "passivation_effect, transport_effect, stability_effect, correlation, process_enabling"
        )
    )
    object_id: str = Field(default="")
    evidence: str = Field(
        default="未明确提及",
        description="支撑该关系的原文原句，必须是 exact quote"
    )

    @field_validator("raw_predicate", mode="before")
    @classmethod
    def normalize_raw_predicate(cls, v):
        if not isinstance(v, str):
            return "induces"
        v = str(v).strip()
        if not v:
            return "induces"

        # 只做轻量清洗，不做强制压缩到固定词表
        v = v.replace("\n", " ").replace("\t", " ")
        v = " ".join(v.split())
        return v[:120]

    @field_validator("predicate_group", mode="before")
    @classmethod
    def normalize_predicate_group(cls, v):
        allowed = {
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

        if not isinstance(v, str):
            return "causal_change"

        v = v.strip().lower().replace(" ", "_").replace("-", "_")

        # 精细化且中立的容错映射
        if any(k in v for k in ["form", "construct", "recrystall", "morpholog"]):
            return "structural_formation"
        if any(k in v for k in ["passivat", "trap", "defect"]):
            return "passivation_effect"
        if any(k in v for k in ["transport", "extract", "conduct", "mobility"]):
            return "transport_effect"
        if any(k in v for k in ["stability", "retain", "degradation", "lifetime"]):
            return "stability_effect"
        if any(k in v for k in ["reduce", "decrease", "suppress", "lower", "loss", "mitigat"]):
            return "negative_effect"
        if any(k in v for k in ["increase", "enhance", "improve", "boost", "promote"]):
            return "positive_effect"
        if any(k in v for k in ["enable", "allow", "facilitate"]):
            return "process_enabling"
        if any(k in v for k in ["associate", "correlat", "link"]):
            return "correlation"
        if any(k in v for k in ["lead", "caus", "result", "trigger", "induc"]):
            return "causal_change"

        # 兜底为中立的因果改变
        return "causal_change"

    @field_validator("subject_id", "object_id", "evidence", mode="before")
    @classmethod
    def robust_string(cls, v):
        if v is None:
            return "未明确提及"
        return str(v).strip()


# ==========================================
# 根图谱单元
# ==========================================
class PVLiteratureExtractSchema(BaseModel):
    article_title_author: str = Field(default="未明确提及")
    material_system: str = Field(default="未明确提及")
    target_problem: str = Field(default="未明确提及")
    research_focus: str = Field(default="未明确提及")
    entities: List[EntityUnit] = Field(default_factory=list)
    relations: List[RelationUnit] = Field(default_factory=list)
    extracted_file: str = Field(default="未手动填写")

    @field_validator(
        "article_title_author",
        "research_focus",
        "material_system",
        "target_problem",
        "extracted_file",
        mode="before"
    )
    @classmethod
    def handle_null_or_int(cls, v):
        if v is None:
            return "未明确提及"
        if isinstance(v, (int, float)):
            return str(int(v))
        return str(v).strip()

    @field_validator("entities", "relations", mode="before")
    @classmethod
    def robust_root_list(cls, v):
        return v if isinstance(v, list) else []

    def to_dict(self) -> dict:
        return self.model_dump()