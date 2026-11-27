"""把 pandas 长表解析成“超边分组”的内部工具。

``Hypergraph`` 与四种高阶网络的 ``from_dataframe`` 共用同一套解析规则，
集中在这里实现，避免各子包重复代码。

约定：输入是长表，一行表示“某个结点属于某条超边 / 单形 / 事件”；
``key_cols`` 给出分组身份列，例如 ``("edge",)``、``("layer", "edge")``、
``("time", "edge")``，同一个键的行合并成一条超边。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from honet._types import Node

if TYPE_CHECKING:
    import pandas as pd


def group_rows(
    frame: pd.DataFrame,
    *,
    key_cols: tuple[str, ...],
    node_col: str,
    weight_col: str | None = None,
    label_col: str | None = None,
) -> list[tuple[tuple[Any, ...], list[Node], float, str | None]]:
    """按 ``key_cols`` 分组，返回 ``(身份键, 成员列表, 权重, 标签)``。

    成员按行序保留（有向变体依赖这个顺序），重复结点由调用方去重；
    同一组内出现的权重或标签必须一致，否则说明表格本身有冲突。

    Args:
        frame: pandas DataFrame，按列名读取。
        key_cols: 分组身份列，至少一列；取值为空视为脏数据。
        node_col: 结点列。
        weight_col: 可选权重列，按 ``float`` 解析；缺省时权重恒为 1.0。
        label_col: 可选标签列，非字符串用 ``str()`` 转换，缺失视为无标签。

    Returns:
        分组列表，顺序为各键首次出现的顺序。

    Raises:
        KeyError: 缺少指定列。
        ValueError: 身份列或结点为空、权重不是实数、同组内权重/标签冲突。
    """
    columns = [*key_cols, node_col]
    if weight_col is not None:
        columns.append(weight_col)
    if label_col is not None:
        columns.append(label_col)

    absent = [name for name in columns if name not in frame.columns]
    if absent:
        raise KeyError(f"DataFrame 缺少列 {absent}；实际列为 {list(frame.columns)}")

    def missing(value: object) -> bool:
        """识别 None / NaN / NaT / pandas.NA。"""
        if value is None:
            return True
        # pd.NA / pd.NaT 做布尔比较会抛错，按类型名识别，避免为可选依赖 import pandas
        if type(value).__name__ in {"NAType", "NaTType"}:
            return True
        try:
            return bool(value != value)
        except (TypeError, ValueError):
            return False

    groups: dict[tuple[Any, ...], tuple[list[Node], float, str | None]] = {}
    key_count = len(key_cols)

    for row_no, row in enumerate(frame[columns].itertuples(index=False, name=None)):
        key = tuple(row[:key_count])
        for position, value in enumerate(key):
            if missing(value):
                raise ValueError(f"第 {row_no} 行的 {key_cols[position]!r} 为空")

        node = row[key_count]
        if missing(node):
            raise ValueError(f"第 {row_no} 行的 {node_col!r} 为空")

        offset = key_count + 1
        weight = 1.0
        if weight_col is not None:
            raw_weight = row[offset]
            offset += 1
            if missing(raw_weight):
                raise ValueError(f"第 {row_no} 行的 {weight_col!r} 为空")
            try:
                weight = float(raw_weight)
            except (TypeError, ValueError) as exc:
                message = f"第 {row_no} 行的 {weight_col!r} 不是实数：{raw_weight!r}"
                raise ValueError(message) from exc

        label: str | None = None
        if label_col is not None:
            raw_label = row[offset]
            if not missing(raw_label):
                label = raw_label if isinstance(raw_label, str) else str(raw_label)

        if key in groups:
            members, known_weight, known_label = groups[key]
            if weight != known_weight:
                raise ValueError(f"超边 {key!r} 的权重不一致：{known_weight} 与 {weight}")
            if label != known_label:
                raise ValueError(f"超边 {key!r} 的标签不一致：{known_label!r} 与 {label!r}")
        else:
            members = []
            groups[key] = (members, weight, label)
        members.append(node)

    return [(key, members, weight, label) for key, (members, weight, label) in groups.items()]
