from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class LLMConfig:
    backend: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 512
    max_completion_tokens: Optional[int] = None
    top_p: float = 1.0
    api_key_list: List[str] = field(default_factory=list)
    logprob_model: str = "text-davinci-003"
    auxiliary_openai_model: str = "gpt-4o-mini"
    auxiliary_openai_logprob_model: str = "gpt-3.5-turbo-instruct"
    auxiliary_openai_embedding_model: str = "text-embedding-3-small"
    auxiliary_openai_api_key_list: List[str] = field(default_factory=list)
    auxiliary_openai_api_base: str = "https://api.openai.com/v1"


@dataclass
class DatasetConfig:
    data_dir: str = ""
    dataset_name: str = ""
    mapping_dataset_name: Optional[str] = None
    item_file: Optional[str] = None
    train_file: Optional[str] = None
    domain_label: str = "CD"
    max_candidates: int = 10
    target_item_id: Optional[int] = None
    target_item_ids: List[int] = field(default_factory=list)
    targets_from: Optional[str] = None
    targets_percent: Optional[float] = None
    popular_item_ids: List[int] = field(default_factory=list)


@dataclass
class PromptConfig:
    system_prompt_file: str = "attack_system.txt"
    user_prompt_file: str = "attack_user.txt"
    refine_prompt_file: str = "attack_refine.txt"
    motif_prompt_file: str = "induction_motif.txt"
    expert_prompt_file: str = "induction_expert.txt"
    amateur_prompt_file: str = "induction_amateur.txt"
    naturalness_prompt_file: str = "trigger_naturalness.txt"
    ugc_refine_prompt_file: str = "ugc_refine.txt"


@dataclass
class AttackConfig:
    max_trials: int = 1
    max_words: int = 64
    num_context_users: int = 20
    num_context_items: int = 10
    include_popular_items: bool = True
    save_partial: bool = True
    dry_run: bool = True


@dataclass
class InductionConfig:
    hub_count: int = 10
    motif_top_n: int = 6
    max_words: int = 64
    llm_max_trials: int = 3
    min_token_len: int = 2
    contrastive_lambda: float = 0.7
    mask_prob: float = 0.9
    sampling_temperature: float = 1.0
    pad_token: str = "[PAD]"
    save_contrastive_details: bool = False
    refine_ugc: bool = False
    refine_max_trials: int = 3


@dataclass
class TriggerConfig:
    num_anchors: int = 2
    max_bridge_candidates: int = 200
    reuse_bridges: bool = True
    include_target: bool = True
    use_hub_items: bool = True
    cost_alpha: float = 0.5
    embedding_model: str = "text-embedding-3-small"
    use_llm_naturalness: bool = False
    naturalness_max_trials: int = 3


@dataclass
class AgentPoisoningConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    induction: InductionConfig = field(default_factory=InductionConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    output_dir: str = "outputs"
    random_seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "AgentPoisoningConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentPoisoningConfig":
        def build(dataclass_type, key: str):
            payload = data.get(key, {})
            if not isinstance(payload, dict):
                return dataclass_type()
            allowed = {field_info.name for field_info in fields(dataclass_type)}
            clean = {k: v for k, v in payload.items() if k in allowed}
            return dataclass_type(**clean)

        attack = build(AttackConfig, "attack")
        induction = build(InductionConfig, "induction")
        trigger = build(TriggerConfig, "trigger")

        if "induction" not in data:
            induction = InductionConfig(
                hub_count=attack.num_context_items or induction.hub_count,
                motif_top_n=induction.motif_top_n,
                max_words=attack.max_words or induction.max_words,
                llm_max_trials=attack.max_trials or induction.llm_max_trials,
                min_token_len=induction.min_token_len,
            )

        if "trigger" not in data:
            trigger = TriggerConfig(
                num_anchors=trigger.num_anchors,
                max_bridge_candidates=trigger.max_bridge_candidates,
                reuse_bridges=trigger.reuse_bridges,
                include_target=trigger.include_target,
                use_hub_items=attack.include_popular_items,
            )

        return cls(
            dataset=build(DatasetConfig, "dataset"),
            attack=attack,
            induction=induction,
            trigger=trigger,
            prompts=build(PromptConfig, "prompts"),
            llm=build(LLMConfig, "llm"),
            output_dir=data.get("output_dir", "outputs"),
            random_seed=data.get("random_seed", 42),
        )
