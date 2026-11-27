"""Notebook 生成器：把「源码 cell 列表」编译成带真实执行输出的 .ipynb。

设计要点
--------
1. **输入是 Python 源码，不是手写 JSON**。每个 notebook 用一个 ``.py`` 描述
   文件（markdown 用 ``# %% [markdown]`` 分隔），本脚本负责执行并落盘。
   好处：notebook 内容可以像普通代码一样被 review、被 git diff。
2. **输出必须真实**。用 ``nbclient`` 真正起内核执行，禁止手填输出——
   这是本仓库 notebook 的一贯做法（旧 notebook 都带 execution_count 与
   iopub 元数据）。
3. **失败即中止**。任一 cell 报错就让整个生成失败，不留半成品。

用法::

    python docs/_make_examples.py            # 生成全部
    python docs/_make_examples.py 01 05      # 只生成指定编号
"""

from __future__ import annotations

import hashlib
import pathlib
import sys
import textwrap

import nbformat
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "docs" / "_example_sources"
OUTPUT_DIR = ROOT / "examples"

MARKDOWN_MARKER = "# %% [markdown]"


def parse_source(path: pathlib.Path) -> list[tuple[str, str]]:
    """把描述文件切成 ``[(cell_type, source), ...]``。

    ``# %% [markdown]`` 之后到下一个分隔符之前是 markdown cell；
    其余按代码 cell 处理。行首的 ``# `` 会被去掉（便于在 .py 里写注释式文档）。
    """
    cells: list[tuple[str, str]] = []
    current_kind = "code"
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        text = "\n".join(buffer).strip("\n")
        if not text.strip():
            return
        if current_kind == "markdown":
            lines = []
            for line in text.splitlines():
                if line.startswith("# "):
                    lines.append(line[2:])
                elif line.strip() == "#":
                    lines.append("")
                else:
                    lines.append(line)
            text = "\n".join(lines)
        cells.append((current_kind, text))

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line.strip() == MARKDOWN_MARKER:
            flush()
            buffer = []
            current_kind = "markdown"
            continue
        if raw_line.strip() == "# %%":
            flush()
            buffer = []
            current_kind = "code"
            continue
        buffer.append(raw_line)
    flush()
    return cells


def build_notebook(name: str, cells: list[tuple[str, str]]) -> nbformat.NotebookNode:
    """组装 notebook 节点。

    cell 的 ``id`` 用「序号 + 内容」的哈希派生，而不是让 nbformat 随机生成。
    否则每次重新生成都会换一批 id，git 里表现为全文件重写，看不出真实改动。
    """
    nb = nbformat.v4.new_notebook()
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        }
    )
    for index, (kind, source) in enumerate(cells):
        digest = hashlib.sha1(f"{name}:{index}:{kind}:{source}".encode()).hexdigest()
        cell_id = digest[:8]
        if kind == "markdown":
            cell = nbformat.v4.new_markdown_cell(source)
        else:
            cell = nbformat.v4.new_code_cell(source)
        cell["id"] = cell_id
        nb.cells.append(cell)
    return nb


def _merge_stream_outputs(cell: nbformat.NotebookNode) -> None:
    """把 cell 内连续的同类 stream 输出合并成一段。

    nbclient 写入输出时，同一段 ``print`` 可能因为内核缓冲被切成多个
    ``stream`` 块（例如 ``['abc']`` 与 ``['abc\\n', 'def\\n']``）。切法取决于
    执行时机，于是同样的代码两次生成会得到字节不同的 notebook。
    合并后输出内容不变，但字节稳定。
    """
    outputs = cell.get("outputs")
    if not outputs:
        return

    merged: list[nbformat.NotebookNode] = []
    for output in outputs:
        previous = merged[-1] if merged else None
        same_stream = (
            previous is not None
            and output.get("output_type") == "stream"
            and previous.get("output_type") == "stream"
            and previous.get("name") == output.get("name")
        )
        if not same_stream:
            merged.append(output)
            continue
        prev_text = previous["text"]
        cur_text = output["text"]
        prev_joined = "".join(prev_text) if isinstance(prev_text, list) else prev_text
        cur_joined = "".join(cur_text) if isinstance(cur_text, list) else cur_text
        previous["text"] = prev_joined + cur_joined

    cell["outputs"] = merged


def generate(number: str, *, timeout: int = 600) -> pathlib.Path:
    """生成单个 notebook：解析 -> 执行 -> 落盘。"""
    matches = sorted(SOURCE_DIR.glob(f"{number}_*.py"))
    if not matches:
        msg = f"找不到编号为 {number} 的示例源文件"
        raise FileNotFoundError(msg)
    if len(matches) > 1:
        msg = f"编号 {number} 对应多个源文件: {matches}"
        raise RuntimeError(msg)

    source = matches[0]
    cells = parse_source(source)
    nb = build_notebook(source.stem, cells)

    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        allow_errors=False,
    )
    client.execute()

    # nbclient 会给每个 code cell 写 metadata.execution（含真实时间戳）。
    # 时间戳对读者没有价值，却让「同内容重新生成」产生字节差异，
    # git 里看不出真实改动。这里统一剥掉。
    #
    # 同时把同一 cell 内**连续的 stdout 片段合并**：nbclient 何时切段取决于
    # 内核缓冲时机，同样的输出可能被切成 1 段或 2 段，导致字节不一致。
    for cell in nb.cells:
        metadata = cell.get("metadata", {})
        metadata.pop("execution", None)
        _merge_stream_outputs(cell)

    OUTPUT_DIR.mkdir(exist_ok=True)
    target = OUTPUT_DIR / f"{source.stem}.ipynb"
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        nbformat.write(nb, handle)
        handle.write("\n")

    executed = sum(
        1 for cell in nb.cells if cell.cell_type == "code" and cell.get("outputs")
    )
    print(
        f"  {target.relative_to(ROOT)}  "
        f"({len(nb.cells)} cells, {executed} 个含输出)"
    )
    return target


def main(argv: list[str]) -> int:
    if not SOURCE_DIR.is_dir():
        print(f"缺少示例源目录: {SOURCE_DIR}", file=sys.stderr)
        return 1

    numbers = argv[1:]
    if not numbers:
        numbers = sorted(
            path.name.split("_", 1)[0] for path in SOURCE_DIR.glob("[0-9][0-9]_*.py")
        )

    print(f"生成 {len(numbers)} 个 notebook -> {OUTPUT_DIR.relative_to(ROOT)}/")
    failures = []
    for number in numbers:
        try:
            generate(number)
        except Exception as exc:  # noqa: BLE001 - 需要汇总所有失败后统一报告
            failures.append((number, exc))
            print(f"  [失败] {number}: {exc}", file=sys.stderr)

    if failures:
        print(f"\n{len(failures)} 个 notebook 生成失败", file=sys.stderr)
        for number, exc in failures:
            detail = textwrap.indent(str(exc), "      ")
            print(f"  - {number}\n{detail}", file=sys.stderr)
        return 1

    print("全部成功")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
