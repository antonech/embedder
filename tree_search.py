import json, os


class TreeIndex:
    def __init__(self, data_dir="data"):
        self.nodes = {}
        self.children = {}
        self.texts = []

        tree_path = os.path.join(data_dir, "tree_index.json")
        if os.path.exists(tree_path):
            self._load(tree_path)

        delta_path = os.path.join(data_dir, "delta_tree_index.json")
        if os.path.exists(delta_path):
            self._load(delta_path)

    def _load(self, tree_path):
        with open(tree_path) as f:
            data = json.load(f)

        uid = 0
        id_map = {}
        for n in data["nodes"]:
            old_id = n["id"]
            n["_uid"] = uid
            n.pop("id", None)
            self.nodes[uid] = n
            id_map[old_id] = uid
            uid += 1

        for n in self.nodes.values():
            pid = n.get("parent_id", -1)
            mapped = id_map.get(pid, -1)
            n["parent_id"] = mapped
            nid = n["_uid"]
            if mapped not in self.children:
                self.children[mapped] = []
            self.children[mapped].append(nid)

        self.texts = data["texts"]

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

    def _match_node(self, text: str) -> dict | None:
        import re
        m = re.search(r'\(([^)]+)\)\s*$', text)
        if not m:
            return None
        file = m.group(1)
        parts = [p.strip() for p in text.split('|')]
        name = parts[1] if len(parts) > 1 else ''
        for n in self.nodes.values():
            nf = n["file"].lstrip("./")
            if nf == file and n["name"] == name:
                return n
        return None

    def search_with_context(self, store, query_vec, top_k: int = 5) -> list[dict]:
        hits = store.search(query_vec, top_k=top_k)
        for h in hits:
            n = self._match_node(h["text"])
            if n:
                h["context"] = {
                    "children": [
                        {"name": c["name"], "type": c["type"], "file": c["file"],
                         "lines": f"{c['start_line']}-{c['end_line']}"}
                        for c in self.get_children(n["_uid"])
                    ],
                    "parent": None,
                    "siblings": [
                        {"name": s["name"], "type": s["type"]}
                        for s in self.get_siblings(n["_uid"])
                    ],
                }
                p = self.get_parent(n["_uid"])
                if p:
                    h["context"]["parent"] = {"name": p["name"], "type": p["type"]}
        return hits
