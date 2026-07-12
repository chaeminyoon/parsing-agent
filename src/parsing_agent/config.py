from __future__ import annotations

from dataclasses import dataclass, field
import os


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class WorkflowWeights:
    text_coverage: float = 0.35
    normalized_similarity: float = 0.35
    structure_retention: float = 0.15
    table_preservation: float = 0.15


@dataclass(slots=True)
class WorkflowConfig:
    parser_names: list[str] = field(default_factory=lambda: ["opendataloader-pdf", "layout-first-pdf", "text-fallback"])
    ocr_enabled: bool = field(default_factory=lambda: _env_flag("PARSING_AGENT_OCR_ENABLED", False))
    ocr_provider: str = field(default_factory=lambda: os.getenv("PARSING_AGENT_OCR_PROVIDER", "surya"))
    ocr_min_text_characters: int = field(
        default_factory=lambda: int(os.getenv("PARSING_AGENT_OCR_MIN_TEXT_CHARACTERS", "50"))
    )
    ocr_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("PARSING_AGENT_OCR_TIMEOUT_SECONDS", "900"))
    )
    ocr_fail_open: bool = field(default_factory=lambda: _env_flag("PARSING_AGENT_OCR_FAIL_OPEN", True))
    ocr_command: str = field(default_factory=lambda: os.getenv("PARSING_AGENT_OCR_COMMAND", "surya_ocr"))
    max_repair_rounds: int = 3
    output_format: str = "md"
    weights: WorkflowWeights = field(default_factory=WorkflowWeights)
    judge_weight: float = 0.25
    table_repair_gain_weight: float = field(
        default_factory=lambda: float(os.getenv("PARSING_AGENT_TABLE_REPAIR_GAIN_WEIGHT", "0.15"))
    )
    min_total_score: float = 0.7
    min_text_coverage: float = 0.7
    max_hallucination_risk: float | None = None
    judge_prompt_tuning_enabled: bool = field(
        default_factory=lambda: _env_flag("PARSING_AGENT_JUDGE_PROMPT_TUNING_ENABLED", True)
    )
    judge_prompt_version: str | None = field(
        default_factory=lambda: os.getenv("PARSING_AGENT_JUDGE_PROMPT_VERSION")
    )
    judge_system_prompt: str | None = field(
        default_factory=lambda: os.getenv("PARSING_AGENT_JUDGE_SYSTEM_PROMPT")
    )
    judge_feedback_log_path: str = field(
        default_factory=lambda: os.getenv("PARSING_AGENT_JUDGE_FEEDBACK_LOG_PATH", "judge_feedback.jsonl")
    )
    judge_feedback_log_max_records: int = field(
        default_factory=lambda: int(os.getenv("PARSING_AGENT_JUDGE_FEEDBACK_LOG_MAX_RECORDS", "50"))
    )
    judge_multimodal_grounding_enabled: bool = field(
        default_factory=lambda: _env_flag("PARSING_AGENT_JUDGE_MULTIMODAL_GROUNDING_ENABLED", True)
    )
    judge_grounding_max_pages: int = field(
        default_factory=lambda: int(os.getenv("PARSING_AGENT_JUDGE_GROUNDING_MAX_PAGES", "2"))
    )
    judge_model: str | None = field(
        default_factory=lambda: os.getenv("PARSING_AGENT_JUDGE_MODEL", "gpt-4.1-mini")
    )
    judge_base_url: str = field(default_factory=lambda: os.getenv("PARSING_AGENT_JUDGE_BASE_URL", "https://api.openai.com/v1"))
    judge_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    judge_timeout_seconds: float = 60.0
    judge_max_retries: int = field(
        default_factory=lambda: int(os.getenv("PARSING_AGENT_JUDGE_MAX_RETRIES", "2"))
    )
    judge_fail_open: bool = field(
        default_factory=lambda: _env_flag("PARSING_AGENT_JUDGE_FAIL_OPEN", True)
    )
    judge_max_source_characters: int = field(
        default_factory=lambda: int(os.getenv("PARSING_AGENT_JUDGE_MAX_SOURCE_CHARACTERS", "20000"))
    )
    judge_max_candidate_characters: int = field(
        default_factory=lambda: int(os.getenv("PARSING_AGENT_JUDGE_MAX_CANDIDATE_CHARACTERS", "20000"))
    )
    judge_evidence_segments: int = field(
        default_factory=lambda: int(os.getenv("PARSING_AGENT_JUDGE_EVIDENCE_SEGMENTS", "6"))
    )
    judge_table_evidence_limit: int = field(
        default_factory=lambda: int(os.getenv("PARSING_AGENT_JUDGE_TABLE_EVIDENCE_LIMIT", "5"))
    )
    post_loop_normalization_enabled: bool = field(
        default_factory=lambda: _env_flag("PARSING_AGENT_POST_LOOP_NORMALIZATION_ENABLED", True)
    )
    # 구조화 포맷(docx/pptx/csv/html/json/yaml)을 마크다운 형태가 아니라
    # 장식 제거 후 콘텐츠로 평가한다. 끄면 기존 형태 비교로 돌아간다.
    structured_content_evaluation_enabled: bool = field(
        default_factory=lambda: _env_flag("PARSING_AGENT_STRUCTURED_CONTENT_EVALUATION_ENABLED", True)
    )
    table_cell_metric_enabled: bool = field(
        default_factory=lambda: _env_flag("PARSING_AGENT_TABLE_CELL_METRIC_ENABLED", True)
    )
    table_cell_metric_max_pages: int = field(
        default_factory=lambda: int(os.getenv("PARSING_AGENT_TABLE_CELL_METRIC_MAX_PAGES", "40"))
    )
    visual_table_recovery_enabled: bool = field(
        default_factory=lambda: _env_flag("PARSING_AGENT_VISUAL_TABLE_RECOVERY_ENABLED", True)
    )
    visual_table_recovery_model: str | None = field(
        default_factory=lambda: os.getenv("PARSING_AGENT_VISUAL_TABLE_RECOVERY_MODEL")
        or "gpt-5-mini"
    )
    visual_table_recovery_timeout_seconds: float = 90.0
    visual_table_recovery_max_tables: int = field(
        default_factory=lambda: int(os.getenv("PARSING_AGENT_VISUAL_TABLE_RECOVERY_MAX_TABLES", "1"))
    )
    llm_text_repair_enabled: bool = field(
        default_factory=lambda: _env_flag("PARSING_AGENT_LLM_TEXT_REPAIR_ENABLED", True)
    )
    llm_text_repair_model: str | None = field(
        default_factory=lambda: os.getenv("PARSING_AGENT_LLM_TEXT_REPAIR_MODEL")
        or os.getenv("PARSING_AGENT_JUDGE_MODEL", "gpt-4.1-mini")
    )
    llm_text_repair_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("PARSING_AGENT_LLM_TEXT_REPAIR_TIMEOUT_SECONDS", "60"))
    )
    llm_text_repair_max_targets: int = field(
        default_factory=lambda: int(os.getenv("PARSING_AGENT_LLM_TEXT_REPAIR_MAX_TARGETS", "3"))
    )
    llm_text_repair_min_confidence: float = field(
        default_factory=lambda: float(os.getenv("PARSING_AGENT_LLM_TEXT_REPAIR_MIN_CONFIDENCE", "0.6"))
    )
    llm_text_repair_window_lines: int = field(
        default_factory=lambda: int(os.getenv("PARSING_AGENT_LLM_TEXT_REPAIR_WINDOW_LINES", "60"))
    )
    repair_fanout_enabled: bool = field(
        default_factory=lambda: _env_flag("PARSING_AGENT_REPAIR_FANOUT_ENABLED", True)
    )
    repair_fanout_max_tasks: int = field(
        default_factory=lambda: int(os.getenv("PARSING_AGENT_REPAIR_FANOUT_MAX_TASKS", "4"))
    )
    visual_table_detection_provider: str = field(
        default_factory=lambda: os.getenv("PARSING_AGENT_VISUAL_TABLE_DETECTION_PROVIDER", "pymupdf")
    )
    visual_table_crop_padding: float = field(
        default_factory=lambda: float(os.getenv("PARSING_AGENT_VISUAL_TABLE_CROP_PADDING", "8"))
    )
    layout_first_skip_top_margin: float = field(
        default_factory=lambda: float(os.getenv("PARSING_AGENT_LAYOUT_FIRST_SKIP_TOP_MARGIN", "0"))
    )
    layout_first_skip_bottom_margin: float = field(
        default_factory=lambda: float(os.getenv("PARSING_AGENT_LAYOUT_FIRST_SKIP_BOTTOM_MARGIN", "0"))
    )
    layout_first_table_format: str = field(
        default_factory=lambda: os.getenv("PARSING_AGENT_LAYOUT_FIRST_TABLE_FORMAT", "markdown")
    )
    layout_first_merge_multipage_tables: bool = field(
        default_factory=lambda: _env_flag("PARSING_AGENT_LAYOUT_FIRST_MERGE_MULTIPAGE_TABLES", True)
    )
    layout_first_image_captioning_enabled: bool = field(
        default_factory=lambda: _env_flag("PARSING_AGENT_LAYOUT_FIRST_IMAGE_CAPTIONING_ENABLED", bool(os.getenv("OPENAI_API_KEY")))
    )
    layout_first_image_caption_model: str | None = field(
        default_factory=lambda: os.getenv("PARSING_AGENT_LAYOUT_FIRST_IMAGE_CAPTION_MODEL")
        or os.getenv("PARSING_AGENT_JUDGE_MODEL", "gpt-4.1-mini")
    )
    layout_first_image_caption_max_blocks: int = field(
        default_factory=lambda: int(os.getenv("PARSING_AGENT_LAYOUT_FIRST_IMAGE_CAPTION_MAX_BLOCKS", "3"))
    )
    layout_first_image_caption_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("PARSING_AGENT_LAYOUT_FIRST_IMAGE_CAPTION_TIMEOUT_SECONDS", "60"))
    )
    layout_first_image_crop_padding: float = field(
        default_factory=lambda: float(os.getenv("PARSING_AGENT_LAYOUT_FIRST_IMAGE_CROP_PADDING", "4"))
    )
    post_selection_image_captioning_enabled: bool = field(
        default_factory=lambda: _env_flag("PARSING_AGENT_POST_SELECTION_IMAGE_CAPTIONING_ENABLED", True)
    )
    post_selection_image_caption_model: str | None = field(
        default_factory=lambda: os.getenv("PARSING_AGENT_POST_SELECTION_IMAGE_CAPTION_MODEL")
        or os.getenv("PARSING_AGENT_JUDGE_MODEL", "gpt-4.1-mini")
    )
    post_selection_image_caption_max_images: int = field(
        default_factory=lambda: int(os.getenv("PARSING_AGENT_POST_SELECTION_IMAGE_CAPTION_MAX_IMAGES", "3"))
    )
    post_selection_image_caption_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("PARSING_AGENT_POST_SELECTION_IMAGE_CAPTION_TIMEOUT_SECONDS", "60"))
    )
    langsmith_tracing: bool = field(default_factory=lambda: _env_flag("LANGSMITH_TRACING"))
    langsmith_project: str | None = field(default_factory=lambda: os.getenv("LANGSMITH_PROJECT"))
    langsmith_api_key: str | None = field(default_factory=lambda: os.getenv("LANGSMITH_API_KEY"))
    langsmith_endpoint: str | None = field(default_factory=lambda: os.getenv("LANGSMITH_ENDPOINT"))
    langsmith_workspace_id: str | None = field(default_factory=lambda: os.getenv("LANGSMITH_WORKSPACE_ID"))
    langsmith_hide_inputs: bool = field(
        default_factory=lambda: _env_flag("PARSING_AGENT_LANGSMITH_HIDE_INPUTS", True)
    )
    langsmith_hide_outputs: bool = field(
        default_factory=lambda: _env_flag("PARSING_AGENT_LANGSMITH_HIDE_OUTPUTS", True)
    )
