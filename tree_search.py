import json, os


class TreeIndex:
    def __init__(self, data_dir="data"):
        self.nodes = {}          # Maps _uid to node dict
        self.children = {}       # Maps parent _uid to list of child _uids
        self.texts = []          # Parallel array of texts (same order as nodes by _uid)
        self.id_map = {}         # Maps identifier (original id + shift) to _uid

        tree_path = os.path.join(data_dir, "tree_index.json")
        delta_path = os.path.join(data_dir, "delta_tree_index.json")

        uid = 0
        max_orig_main = -1
        if os.path.exists(tree_path):
            uid, max_orig_main = self._load(tree_path, start_uid=uid, id_shift=0)

        if os.path.exists(delta_path):
            # Shift the delta by max_orig_main + 1 to avoid id overlap
            uid, _ = self._load(delta_path, start_uid=uid, id_shift=max_orig_main + 1)

    def _load(self, tree_path, start_uid=0, id_shift=0):
        """Load nodes from a tree index file.

        Args:
            tree_path: Path to the JSON file.
            start_uid: The starting _uid to assign to nodes from this file.
            id_shift: The value to add to the original 'id' and 'parent_id' to get the identifier used in id_map.

        Returns:
            (next_uid, max_original_id) where next_uid is the next available _uid after loading,
            and max_original_id is the maximum original 'id' found in this file (before shifting).
        """
        with open(tree_path) as f:
            data = json.load(f)

        uid = start_uid
        max_original_id_without_shift = -1
        # First pass: assign _uid and store mapping from (original_id + id_shift) to _uid
        for n in data["nodes"]:
            original_id = n["id"]
            if original_id > max_original_id_without_shift:
                max_original_id_without_shift = original_id
            shifted_id = original_id + id_shift
            original_parent_id = n.get("parent_id", -1)
            # We'll store the shifted parent id temporarily in the node (to be resolved later)
            n["_shifted_parent_id"] = original_parent_id + id_shift if original_parent_id != -1 else -1
            # Remove the original id field
            n.pop("id", None)
            # Remove the original parent_id field (we'll use _shifted_parent_id for now)
            n.pop("parent_id", None)
            # Assign sequential _uid
            n["_uid"] = uid
            self.nodes[uid] = n
            # Map the identifier (shifted_id) to _uid
            self.id_map[shifted_id] = uid
            uid += 1

        # Extend the texts
        self.texts.extend(data["texts"])

        # Second pass: resolve parent_id to _uid using the id_map
        for n in self.nodes.values():
            shifted_parent_id = n.get("_shifted_parent_id", -1)
            if shifted_parent_id == -1:
                n["parent_id"] = -1
            else:
                # Look up the _uid of the parent using the identifier (shifted_parent_id)
                n["parent_id"] = self.id_map.get(shifted_parent_id, -1)
            # Remove the temporary field
            n.pop("_shifted_parent_id", None)

        # Rebuild the entire children mapping using the current id_map (which maps identifier to _uid)
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

    def _match_node(self, text: str) -> dict | None:
        # 1. Clean up the search text: remove potential [LINE] or [FILE] prefixes
        import re
        cleaned_text = re.sub(r'^\[(?:LINE|FILE|CLASS|FUNC)\]\s*', '', text).strip()
        
        # 2. Try exact match on the stored text
        for n in self.nodes.values():
            if n["text"] == cleaned_text or n["text"] == text:
                return n
        
        # 3. Fallback to parsing: extract file path and name
        # The expected format is: "filepath type name (signature) | docstring"
        
        # Remove docstring part if present
        base_part = cleaned_text.split(' | ')[0].strip()
        tokens = base_part.split()
        
        if len(tokens) >= 3:
            # Format: filepath type name...
            file_path = tokens[0].lstrip("./")
            # The name might contain spaces or signatures
            name_part = ' '.join(tokens[2:])
            
            # Strip signature if present: "name (arg1, arg2)" -> "name"
            # We use rsplit to handle cases where name itself might have parentheses
            pure_name = name_part.rsplit(' (', 1)[0] if ' (' in name_part else name_part
            
            for n in self.nodes.values():
                nf = n["file"].lstrip("./")
                if nf == file_path and n["name"] == pure_name:
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
