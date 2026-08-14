import os, re, sys

from common import iter_tree_index


class TreeNode:
    """One AST node, kept as slots instead of a dict.

    The parsed JSON node carries eleven keys of which only these are ever read;
    holding the raw dicts cost ~270 MB for a 100k-node index against ~20 MB of
    actual character data. A node's chunk text stays available as
    `TreeIndex.texts[uid]`.
    """

    __slots__ = ("uid", "name", "type", "file", "start_line", "end_line", "parent_id")
    _ALIASES = {"_uid": "uid"}

    def __init__(self, uid: int, name: str, type: str, file: str,
                 start_line: int, end_line: int, parent_id: int = -1):
        self.uid = uid
        self.name = name
        self.type = type
        self.file = file
        self.start_line = start_line
        self.end_line = end_line
        self.parent_id = parent_id

    # Mapping access so callers can keep using node["name"] / node.get("parent_id", -1).
    def __getitem__(self, key: str):
        try:
            return getattr(self, self._ALIASES.get(key, key))
        except AttributeError:
            raise KeyError(key) from None

    def __contains__(self, key: str) -> bool:
        return self._ALIASES.get(key, key) in self.__slots__

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __repr__(self) -> str:
        return f"TreeNode(uid={self.uid}, name={self.name!r}, file={self.file!r})"


class TreeIndex:
    def __init__(self, data_dir="data"):
        self.nodes = {}
        self.children = {}
        self.texts = []
        self.id_map = {}
        self.lookup = {}
        self.name_uids = {}

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
                self.name_uids.setdefault((file, name), []).append(uid)

    def _owner_class(self, uid: int) -> str:
        parent = self.get_parent(uid)
        if parent and "class" in parent.get("type", ""):
            return parent.get("name", "")
        return ""

    def _pick_uid(self, uids: list[int], text: str) -> int:
        """Disambiguate same-named nodes in one file by enclosing class.

        A method and a module-level function can share a name (a class method and
        its thin module-level wrapper), so the chunk's `In class:` marker decides
        which node the chunk belongs to.
        """
        if len(uids) == 1:
            return uids[0]
        m = re.search(r'In class:\s*([A-Za-z_]\w*)', text)
        wanted = m.group(1) if m else ""
        for uid in uids:
            if self._owner_class(uid) == wanted:
                return uid
        return uids[0]

    def match_node(self, text: str) -> "TreeNode | None":
        m = re.match(r'^\S+\s+(\S+)\s+(\S+)', text)
        if not m:
            return None
        file = m.group(1)
        name = m.group(2).rstrip(".,;:!?(){}[]")
        uids = self.name_uids.get((file, name))
        if not uids:
            return None
        return self.nodes.get(self._pick_uid(uids, text))

    def annotate(self, hits: list[dict]) -> list[dict]:
        for h in hits:
            nid = h.get("node_id")
            n = self.get_node(nid) if nid is not None else self.match_node(h.get("text", ""))
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
        uid = start_uid
        max_original_id_without_shift = -1
        # Only the nodes from this file get their parents resolved below. The
        # previous version re-walked every loaded node on the second call, by
        # which point main-index nodes had lost their shifted parent id, so
        # loading a delta reset the whole main index to parent_id -1.
        pending: list[tuple[TreeNode, int]] = []
        for n, text in iter_tree_index(tree_path):
            original_id = n["id"]
            if original_id > max_original_id_without_shift:
                max_original_id_without_shift = original_id
            original_parent_id = n.get("parent_id", -1)
            node = TreeNode(
                uid,
                n.get("name", ""),
                sys.intern(n.get("type", "")),
                sys.intern(n.get("file", "")),
                n.get("start_line", 0),
                n.get("end_line", 0),
            )
            self.nodes[uid] = node
            self.id_map[original_id + id_shift] = uid
            self.texts.append(text)
            pending.append(
                (node, original_parent_id + id_shift if original_parent_id != -1 else -1)
            )
            uid += 1

        for node, shifted_parent_id in pending:
            if shifted_parent_id == -1:
                continue
            node.parent_id = self.id_map.get(shifted_parent_id, -1)
            if node.parent_id != -1:
                self.children.setdefault(node.parent_id, []).append(node.uid)

        return uid, max_original_id_without_shift

    def get_node(self, node_id: int) -> "TreeNode | None":
        return self.nodes.get(node_id)

    def get_children(self, node_id: int) -> "list[TreeNode]":
        return [self.nodes[cid] for cid in self.children.get(node_id, []) if cid in self.nodes]

    def get_parent(self, node_id: int) -> "TreeNode | None":
        n = self.nodes.get(node_id)
        if n and n.get("parent_id", -1) >= 0:
            return self.nodes.get(n["parent_id"])
        return None

    def get_siblings(self, node_id: int) -> "list[TreeNode]":
        n = self.nodes.get(node_id)
        if not n or n.get("parent_id", -1) < 0:
            return []
        return [self.nodes[cid] for cid in self.children.get(n["parent_id"], [])
                if cid in self.nodes and cid != node_id]
