# API 参考

```{toctree}
:maxdepth: 2

core
networks
analysis
```

| 页面 | 内容 |
| --- | --- |
| [核心容器](core.md) | `Hypergraph` / `Hyperedge` / `Node` 与三个工具函数 |
| [网络形态](networks.md) | `HigherOrderNetwork` 基类、mixin 与 16 个具体类 |
| [分析子包](analysis.md) | `io` / `convert` / `build` / `metrics` / `community` / `contagion` / `attack` / `control` / `plot` / `export` |

```{note}
API 页面由 `sphinx.ext.autodoc` 从源码 docstring 生成，**签名与说明以源码为准**。
若某页与正文表述不一致，说明源码变了、文档没跟上——请提 issue。
```
