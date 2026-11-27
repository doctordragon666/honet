"""honet.plot —— 高阶网络分析结果的绘图接口（基于 matplotlib）。

本子包把 ``hypertrade.draw`` 的 15 个绘图函数重构为「一个文件一个函数」的
形态，覆盖四类图形：

* 传播动力学曲线 —— :func:`draw_sis`、:func:`draw_sir`、:func:`draw_sis_phase`、
  :func:`draw_sis_init`、:func:`draw_sis_init_beta`、:func:`draw_sis_community`；
* 社团与控制节点分布 —— :func:`draw_community_partition`、
  :func:`draw_control_partition`、:func:`draw_communities_onmap`；
* 整体指标与同配性 —— :func:`draw_hnxs_global`、:func:`draw_test_assortativity`；
* 攻击鲁棒性 —— :func:`draw_attack_results`。

本子包的全部符号都是惰性导出的：只有真正访问某个符号时才会 import
对应的实现模块，因此 ``import honet`` 不会因此拉入 matplotlib。
缺依赖时的提示指向 ``honet[viz]``；:func:`draw_communities_onmap` 额外需要
geopandas（同样属于 ``viz`` 组）。

所有绘图函数都接受可选的 ``ax`` 参数：传入时只在该坐标轴上绘制、不显示图像，
便于多图排版与无界面（Agg 后端）测试；不传时新建画布并显示。
"""

from __future__ import annotations

from honet._lazy import lazy_getattr

_EXPORTS = {
    "_sub_draw_sir_community": "honet.plot._sub_draw_sir_community:_sub_draw_sir_community",
    "_sub_draw_sis_community": "honet.plot._sub_draw_sis_community:_sub_draw_sis_community",
    "draw_attack_results": "honet.plot.draw_attack_results:draw_attack_results",
    "draw_communities_onmap": "honet.plot.draw_communities_onmap:draw_communities_onmap",
    "draw_community_partition": "honet.plot.draw_community_partition:draw_community_partition",
    "draw_control_partition": "honet.plot.draw_control_partition:draw_control_partition",
    "draw_hnxs_global": "honet.plot.draw_hnxs_global:draw_hnxs_global",
    "draw_sir": "honet.plot.draw_sir:draw_sir",
    "draw_sis": "honet.plot.draw_sis:draw_sis",
    # 旧名别名：旧库把 community 拼成了 conmmunity，保留以免破坏既有调用
    "draw_sis_conmmunity": "honet.plot.draw_sis_community:draw_sis_community",
    "draw_sis_community": "honet.plot.draw_sis_community:draw_sis_community",
    "draw_sis_init": "honet.plot.draw_sis_init:draw_sis_init",
    "draw_sis_init_beta": "honet.plot.draw_sis_init_beta:draw_sis_init_beta",
    "draw_sis_phase": "honet.plot.draw_sis_phase:draw_sis_phase",
    "draw_test_assortativity": "honet.plot.draw_test_assortativity:draw_test_assortativity",
}

#: 旧名别名 → 规范名。旧库把 community 拼成了 conmmunity。
_ALIASES = {"draw_sis_conmmunity": "draw_sis_community"}

#: 两个下划线开头的辅助函数同样可惰性访问（``honet.plot._sub_draw_sis_community``），
#: 但不列入 ``__all__``，避免 ``from honet.plot import *`` 污染调用方命名空间。
__all__ = sorted(name for name in _EXPORTS if not name.startswith("_"))


def __getattr__(name: str):
    """按需加载实现模块（PEP 562）。

    别名先解析成规范名再加载：``draw_sis_conmmunity`` 与
    ``draw_sis_community`` 指向同一个实现模块，若直接用别名去
    :func:`honet._lazy.lazy_getattr`，import 该模块时 Python 会把模块对象
    写回包的同名属性 ``draw_sis_community``，反而遮蔽掉惰性导出的函数。
    """
    canonical = _ALIASES.get(name, name)
    value = lazy_getattr(globals(), __name__, _EXPORTS, canonical, "viz")
    globals()[name] = value
    return value
