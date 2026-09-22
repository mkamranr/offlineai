"""Dependency graph (section 58).

Shows what a package is made of and where each piece came from. The value is
in the second half: on an air-gapped host, "where did this come from" is not
answerable by looking it up.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from offlineai.schema.manifest import ArtifactEntry, ArtifactType, Manifest
from offlineai.utils.sizes import format_bytes

__all__ = ["GraphNode", "build_graph", "to_ascii", "to_dot", "to_json"]

_GROUP_LABEL = {
    ArtifactType.OCI_IMAGE: "containers",
    ArtifactType.MODEL: "models",
    ArtifactType.PYTHON_WHEEL: "python dependencies",
    ArtifactType.PYTHON_SDIST: "python source distributions",
    ArtifactType.SYSTEM_PACKAGE: "system packages",
    ArtifactType.CONFIG: "configuration",
    ArtifactType.SCRIPT: "scripts",
    ArtifactType.MISC: "other",
}


@dataclass(slots=True)
class GraphNode:
    name: str
    kind: str
    size: int = 0
    detail: str | None = None
    children: list[GraphNode] = field(default_factory=list)

    @property
    def total_size(self) -> int:
        return self.size + sum(c.total_size for c in self.children)


def build_graph(manifest: Manifest) -> GraphNode:
    """Build a tree rooted at the package."""
    root = GraphNode(name=f"{manifest.package.name} {manifest.package.version}", kind="package")

    for artifact_type in ArtifactType:
        artifacts = manifest.artifacts_of_type(artifact_type)
        if not artifacts:
            continue

        group = GraphNode(name=_GROUP_LABEL[artifact_type], kind="group")

        if artifact_type is ArtifactType.MODEL:
            # Collapse a sharded checkpoint into one node: a tree with eight
            # identical-looking safetensors lines is noise, not information.
            by_model: dict[str, list[ArtifactEntry]] = {}
            for artifact in artifacts:
                key = str(artifact.metadata.get("model") or _model_dir(artifact.path))
                by_model.setdefault(key, []).append(artifact)
            for name, members in sorted(by_model.items()):
                total = sum(m.size for m in members)
                source = next((m.source for m in members if m.source), None)
                group.children.append(
                    GraphNode(
                        name=name,
                        kind="model",
                        size=total,
                        detail=f"{len(members)} file(s)"
                        + (f", from {_origin(source)}" if source else ""),
                    )
                )
        else:
            for artifact in sorted(artifacts, key=lambda a: a.path):
                group.children.append(
                    GraphNode(
                        name=_leaf_name(artifact),
                        kind=artifact_type.value,
                        size=artifact.size,
                        detail=_leaf_detail(artifact),
                    )
                )

        root.children.append(group)

    return root


def _model_dir(path: str) -> str:
    parts = path.split("/")
    return parts[2] if len(parts) > 2 else path


def _leaf_name(artifact: object) -> str:
    metadata = getattr(artifact, "metadata", {}) or {}
    if package := metadata.get("package"):
        version = metadata.get("version")
        return f"{package} {version}" if version else str(package)
    if image := metadata.get("image"):
        tag = metadata.get("tag")
        return f"{image}:{tag}" if tag else str(image)
    return str(getattr(artifact, "path", "")).rsplit("/", 1)[-1]


def _leaf_detail(artifact: object) -> str | None:
    digest = getattr(artifact, "digest", None)
    if digest:
        return f"digest {digest[:19]}…"
    metadata = getattr(artifact, "metadata", {}) or {}
    if tag := metadata.get("platform_tag"):
        return str(tag)
    return None


def _origin(source: str | None) -> str:
    """Reduce an artifact source to the thing it came from, not the file.

    A model's source is per-file (hf://repo@rev/config.json); the useful answer
    for a collapsed model node is the repository and revision.
    """
    if not source:
        return "unknown"
    if source.startswith("hf://"):
        body = source.removeprefix("hf://")
        repo_and_revision = body.split("/", 2)
        if len(repo_and_revision) >= 2 and "@" in repo_and_revision[1]:
            return "hf://" + "/".join(repo_and_revision[:2])
        return "hf://" + "/".join(repo_and_revision[:2])
    if "/resolve/" in source:
        return source.split("/resolve/")[0]
    return source


def to_ascii(node: GraphNode, *, show_sizes: bool = True) -> str:
    lines: list[str] = []

    def walk(current: GraphNode, prefix: str, is_last: bool, is_root: bool) -> None:
        if is_root:
            lines.append(current.name)
            child_prefix = ""
        else:
            connector = "└── " if is_last else "├── "
            label = current.name
            if show_sizes and current.total_size:
                label += f"  ({format_bytes(current.total_size)})"
            if current.detail:
                label += f"  [{current.detail}]"
            lines.append(f"{prefix}{connector}{label}")
            child_prefix = prefix + ("    " if is_last else "│   ")

        for index, child in enumerate(current.children):
            walk(child, child_prefix, index == len(current.children) - 1, False)

    walk(node, "", True, True)
    return "\n".join(lines)


def to_json(node: GraphNode) -> dict[str, object]:
    return {
        "name": node.name,
        "kind": node.kind,
        "size": node.size,
        "total_size": node.total_size,
        "detail": node.detail,
        "children": [to_json(c) for c in node.children],
    }


def to_dot(node: GraphNode) -> str:
    lines = ["digraph offlineai {", "  rankdir=LR;", '  node [shape=box, fontname="sans"];']
    counter = {"n": 0}
    ids: dict[int, str] = {}

    def register(current: GraphNode) -> str:
        counter["n"] += 1
        node_id = f"n{counter['n']}"
        ids[id(current)] = node_id
        label = current.name.replace('"', '\\"')
        if current.total_size:
            label += f"\\n{format_bytes(current.total_size)}"
        shape = "ellipse" if current.kind == "package" else "box"
        lines.append(f'  {node_id} [label="{label}", shape={shape}];')
        return node_id

    def walk(current: GraphNode) -> None:
        parent_id = ids[id(current)]
        for child in current.children:
            child_id = register(child)
            lines.append(f"  {parent_id} -> {child_id};")
            walk(child)

    register(node)
    walk(node)
    lines.append("}")
    return "\n".join(lines)


def graph_json_text(manifest: Manifest) -> str:
    return json.dumps(to_json(build_graph(manifest)), indent=2)
