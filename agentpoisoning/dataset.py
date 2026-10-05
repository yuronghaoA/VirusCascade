from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class ItemRecord:
    item_id: int
    asin: str
    title: str
    category: str

    def describe(self, domain_label: str) -> str:
        label = domain_label.strip() or "CD"
        title = self.title.replace("'", "\\'")
        category = self.category.replace("'", "\\'")
        return f"The {label} is called '{title}'. The category of this {label} is: '{category}'."

    @property
    def default_memory(self) -> str:
        return self.describe("CD")


@dataclass
class UserRecord:
    user_id: int
    interactions: List[int]


class Dataset:
    def __init__(
        self,
        data_dir: Path,
        dataset_name: str,
        item_file: Optional[str] = None,
        train_file: Optional[str] = None,
    ) -> None:
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.item_file = Path(item_file) if item_file else data_dir / f"{dataset_name}.item"
        self.train_file = Path(train_file) if train_file else data_dir / f"{dataset_name}.train.inter"
        self.meta_file = data_dir / "meta_data.json"
        self.train_json = data_dir / "train.json"
        self.smap_file = data_dir / "smap.json"
        self.umap_file = data_dir / "umap.json"
        self.items: Dict[int, ItemRecord] = {}
        self.users: Dict[int, UserRecord] = {}

    def load(self) -> None:
        if self.meta_file.exists() and self.train_json.exists() and self.smap_file.exists():
            self.items = self._load_items_recformer(self.meta_file, self.smap_file)
            self.users = self._load_train_recformer(self.train_json)
            return
        self.items = self._load_items(self.item_file)
        self.users = self._load_train_interactions(self.train_file)

    def _load_items(self, path: Path) -> Dict[int, ItemRecord]:
        records: Dict[int, ItemRecord] = {}
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter="\t")
            next(reader, None)
            for row in reader:
                if not row:
                    continue
                try:
                    item_id = int(row[0])
                except ValueError:
                    continue
                asin = str(item_id)
                title = row[1] if len(row) > 1 else ""
                category = row[2] if len(row) > 2 else ""
                records[item_id] = ItemRecord(item_id=item_id, asin=asin, title=title, category=category)
        return records

    def _load_train_interactions(self, path: Path) -> Dict[int, UserRecord]:
        users: Dict[int, UserRecord] = {}
        with path.open("r", encoding="utf-8") as handle:
            header = handle.readline().strip().split("\t")
            if not header or header[0] == "":
                return users
            idx_user = header.index("user_id:token") if "user_id:token" in header else 0
            idx_item = header.index("item_id:token") if "item_id:token" in header else -1
            idx_seq = header.index("item_id_list:token_seq") if "item_id_list:token_seq" in header else -1
            for line in handle:
                parts = line.strip().split("\t")
                if len(parts) <= max(idx_user, idx_item, idx_seq):
                    continue
                try:
                    user_id = int(parts[idx_user])
                except ValueError:
                    continue
                interactions: List[int] = []
                if idx_seq >= 0 and parts[idx_seq]:
                    interactions.extend(int(token) for token in parts[idx_seq].split(" ") if token)
                if idx_item >= 0 and parts[idx_item]:
                    try:
                        interactions.append(int(parts[idx_item]))
                    except ValueError:
                        pass
                if user_id not in users:
                    users[user_id] = UserRecord(user_id=user_id, interactions=interactions)
                else:
                    users[user_id].interactions.extend(interactions)
        return users

    def _load_items_recformer(self, meta_path: Path, smap_path: Path) -> Dict[int, ItemRecord]:
        meta_raw = json.loads(meta_path.read_text(encoding="utf-8"))
        smap = json.loads(smap_path.read_text(encoding="utf-8"))
        records: Dict[int, ItemRecord] = {}
        for asin, idx in smap.items():
            meta = meta_raw.get(asin, {})
            title = meta.get("title", "") if isinstance(meta, dict) else ""
            category = meta.get("category", "") if isinstance(meta, dict) else ""
            records[int(idx)] = ItemRecord(
                item_id=int(idx),
                asin=str(asin),
                title=str(title),
                category=str(category),
            )
        return records

    def _load_train_recformer(self, train_path: Path) -> Dict[int, UserRecord]:
        data = json.loads(train_path.read_text(encoding="utf-8"))
        users: Dict[int, UserRecord] = {}
        for raw_user_id, items in data.items():
            try:
                user_id = int(raw_user_id)
            except ValueError:
                continue
            users[user_id] = UserRecord(user_id=user_id, interactions=[int(item_id) for item_id in items])
        return users

    def popularity(self) -> List[Tuple[int, int]]:
        counts: Dict[int, int] = {}
        for user in self.users.values():
            for item_id in user.interactions:
                counts[item_id] = counts.get(item_id, 0) + 1
        return sorted(counts.items(), key=lambda pair: pair[1], reverse=True)

    def describe_item(self, item_id: int, domain_label: str) -> str:
                # --- load pretrained_item descriptions once (dataset-V1) ---
        if not hasattr(self, "_pretrained_item_descriptions"):
            from pathlib import Path
            self._pretrained_item_descriptions = {}
            data_dir = getattr(self, "data_dir", None)
            domain = str(domain_label) if domain_label is not None else ""
            if data_dir and domain:
                root = Path(data_dir).parent
                path = root / "dataset-V1" / f"{domain}-100-user-sparse" / f"{domain}.pretrained_item"
                if path.exists():
                    try:
                        with path.open("r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                parts = line.split("\t", 1)
                                if len(parts) < 2:
                                    continue
                                item_id_str, desc = parts[0].strip(), parts[1].strip()
                                if not item_id_str.isdigit():
                                    continue
                                if desc.startswith('"') and desc.endswith('"') and len(desc) >= 2:
                                    desc = desc[1:-1]
                                self._pretrained_item_descriptions[int(item_id_str)] = desc
                    except Exception:
                        pass
        desc = self._pretrained_item_descriptions.get(item_id)
        if desc:
            return desc
        return self.items[item_id].describe(domain_label)

    def candidate_block(self, item_ids: List[int], domain_label: str) -> str:
        label = domain_label.strip() or "CD"
        lines: List[str] = []
        for index, item_id in enumerate(item_ids, start=1):
            item = self.items.get(item_id)
            if item is None:
                continue
            title = item.title or f"{label}-{item_id}"
            category = item.category or "unknown"
            lines.append(f"{index}. {title} | category: {category}")
        return "\n".join(lines)

    def get_user_memory(self, user_id: int, domain_label: str) -> str:
        return f"I enjoy {domain_label} very much."
