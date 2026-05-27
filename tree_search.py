import json, os, re


class TreeIndex:
    def __init__(self, data_dir="data"):
        self.nodes = {}
        self.children = {}
        self.texts = []
        self.id_map = {}
        self.lookup = {}

        tree_path = os.path.join(data_dir, "tree_index.json")
        delta_path = os.path.join(data_dir, "delta_tree_index.json")

        uid = 0
        max_orig_main = -1
        if os.path.exists(tree_path):
            uid, max_orig_main = self._load(tree_path, start_uid=uid, id_shift=0)

        if os.path.exists(delta_path):
            uid, _ = self._load(delta_path, start_uid=uid, id_shift=max_orig_main + 1)

        self._build_lookup()

    def _build_lookup(self):
        for uid, node in self.nodes.items():
            file = node.get("file", "")
            name = node.get("name", "")
            if file and name:
                self.lookup[(file, name)] = uid

    def match_node(self, text: str) -> dict | None:
        m = re.match(r'^\S+\s+(\S+)\s+(\S+)', text)
        if not m:
            return None
        file = m.group(1)
        name = m.group(2)
        uid = self.lookup.get((file, name))
        if uid is not None:
            return self.nodes.get(uid)
        return None

    def annotate(self, hits: list[dict]) -> list[dict]:
        for h in hits:
            n = self.match_node(h.get("text", ""))
            if not n:
                continue
            uid = n["_uid"]
            h["context"] = {
                "children": [
                    {"name": c["name"], "type": c["type"], "file": c["file"],
                     "lines": f"{c['start_line']}-{c['end_line']}"}
                    for c in self.get_children(uid)
                ],
                "parent": None,
                "siblings": [
                    {"name": s["name"], "type": s["type"]}
                    for s in self.get_siblings(uid)
                ],
            }
            p = self.get_parent(uid)
            if p:
                h["context"]["parent"] = {"name": p["name"], "type": p["type"]}
        return hits

    def _load(self, tree_path, start_uid=0, id_shift=0):
        with open(tree_path) as f:
            data = json.load(f)

        uid = start_uid
        max_original_id_without_shift = -1
        for n in data["nodes"]:
            original_id = n["id"]
            if original_id > max_original_id_without_shift:
                max_original_id_without_shift = original_id
            shifted_id = original_id + id_shift
            original_parent_id = n.get("parent_id", -1)
            n["_shifted_parent_id"] = original_parent_id + id_shift if original_parent_id != -1 else -1
            n.pop("id", None)
            n.pop("parent_id", None)
            n["_uid"] = uid
            self.nodes[uid] = n
            self.id_map[shifted_id] = uid
            uid += 1

        self.texts.extend(data["texts"])

        for n in self.nodes.values():
            shifted_parent_id = n.get("_shifted_parent_id", -1)
            if shifted_parent_id == -1:
                n["parent_id"] = -1
            else:
                n["parent_id"] = self.id_map.get(shifted_parent_id, -1)
            n.pop("_shifted_parent_id", None)

        self.children = {}
        for nid, n in self.nodes.items():
            parent_uid = n.get("parent_id", -1)
            if parent_uid == -1:
                continue
            if parent_uid not in self.children:
                self.children[parent_uid] = []
            self.children[parent_uid].append(nid)

        return uid, max_original_id_without_shift

    def get_node(self, node_id: int) -> dict | None:
        return self.nodes.get(node_id)

    def get_children(self, node_id: int) -> list[dict]:
        return [self.nodes[cid] for cid in self.children.get(node_id, []) if cid in self.nodes]

    def get_parent(self, node_id: int) -> dict | None:
        n = self.nodes.get(node_id)
        if n and n["parent_id"] >= 0:
            return self.nodes.get(n["parent_id"])
        return None

    def get_siblings(self, node_id: int) -> list[dict]:
        n = self.nodes.get(node_id)
        if not n or n["parent_id"] < 0:
            return []
        return [self.nodes[cid] for cid in self.children.get(n["parent_id"], [])
                if cid in self.nodes and cid != node_id]
