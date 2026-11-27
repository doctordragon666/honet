"""通用绘图接口

为其他模块的结果提供绘图功能，所有函数默认不返回图对象，需要手动调用 plt.show() 来显示图像，除非传入了 ax 参数。
"""
from . import np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 关闭unicode负号转义，避免负号显示为方块
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.grid'] = True
plt.rcParams['pdf.fonttype'] = 42
legend_font = {"family": "SimSun", "size": 14}
plt.rcParams['xtick.direction'] = 'in'   # x 轴刻度朝内
plt.rcParams['ytick.direction'] = 'in'   # y 轴刻度朝内
plt.rcParams['axes.grid']      = True    # 对应第一个参数 True（是否显示网格）
plt.rcParams['grid.linestyle'] = '--'    # 对应 linestyle='--'
plt.rcParams['grid.alpha']     = 0.6     # 对应 alpha=0.6


def draw_sis(S, I, t, title="", ax=None, y_end=-0.2):
    """绘制SIS模型结果的函数

    Parameters
    -----------
    S : numpy.ndarray
        易感者数量历史
    I : numpy.ndarray
        感染者数量历史
    t : numpy.ndarray
        时间点列表
    title : str, optional
        图表标题，默认为空字符串
    ax : matplotlib.axes.Axes, optional
        如果ax参数为None，说明是单图绘制，直接显示图表；如果传入了ax参数，说明是多图绘制，不调用plt.show()，由外部统一显示
    """
    if ax is None:
        plt.figure(figsize=(6, 5))
    else:
        plt.sca(ax)

    # # 将S,I的值转化为百分比
    # sum_country = S[0] + I[0]
    # S = S / sum_country
    # I = I / sum_country

    # plt.plot(t, S, 'g-', label='恢复')
    plt.plot(t, S, 'g-', label=r'$S$')
    plt.plot(t, I, 'r-', label=r'$I$')
    # plt.plot(t, I, 'r-', label='感染')
    # plt.xlabel("时间", fontsize=14, fontname='SimSun')
    plt.xlabel(r"$t$", fontsize=14, fontname='SimSun')
    # plt.ylabel("国家(地区)数量", fontsize=14, fontname='SimSun')
    plt.ylabel(r"$V$", fontsize=14, fontname='SimSun')
    I_mean = np.mean(I)
    plt.axhline(y=I_mean, color='b', linestyle='--',
                label=rf'$\overline{{I}}:\ {I_mean:.2f}$')
    # label=f'感染均值: {I_mean:.2f}')
    plt.legend(fontsize=14, prop=legend_font)

    if title:
        plt.title(title, fontsize=14, y=y_end, fontname='SimSun')

    if ax is None:
        plt.show()


def draw_sir(S, I, R, t, title="", ax=None):
    """绘制SIR模型结果的函数

    可能存在多个交点，后续会添加标签，显示每个交点的时间和对应的S、I、R值

    Parameters
    -----------
    S : numpy.ndarray
        易感者数量历史
    I : numpy.ndarray
        感染者数量历史
    R : numpy.ndarray
        康复者数量历史
    t : numpy.ndarray
        时间点列表
    title : str, optional
        图表标题，默认为空字符串
    ax : matplotlib.axes.Axes, optional
        如果ax参数为None，说明是单图绘制，直接显示图表；如果传入了ax参数，说明是多图绘制，不调用plt.show()，由外部统一显示
    """
    if ax is None:
        plt.figure(figsize=(8, 5))
    else:
        plt.sca(ax)

    plt.plot(t, S, 'g-', label='易感')
    plt.plot(t, I, 'r-', label='感染')
    plt.plot(t, R, 'b-', label='康复')
    plt.xlabel("时间", fontsize=14, fontname='SimSun')
    plt.ylabel("国家(地区)数量", fontsize=14, fontname='SimSun')
    I_mean = np.mean(I)
    plt.axhline(y=I_mean, color='b', linestyle='--',
                label=f'感染均值: {I_mean:.2f}')
    plt.legend(fontsize=14, prop=legend_font)

    if title:
        plt.title(title, fontsize=14, y=-0.2, fontname='SimSun')

    if ax is None:
        plt.show()


# def draw_sis_phase(phase_map: np.ndarray, beta2_border: tuple, beta_higher_border: tuple, ax=None):
#     """绘制SIS模型的相图

#     Parameters
#     ----------
#     phase_map : np.ndarray
#         相图数据
#     beta2_border : tuple
#         beta2的边界
#     beta_higher_border : tuple
#         beta_{m>=3}的边界
#     ax : matplotlib.axes.Axes, optional
#         如果ax参数为None，说明是单图绘制，直接显示图表；如果传入了ax参数，说明是多图绘制，不调用plt.show()，由外部统一显示
#     """
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(8.5, 6.5))
#     else:
#         fig = ax.figure

#     cmap = ListedColormap(["#fb6a4a", "#fde0dd"])
#     norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

#     extent = [
#         float(beta2_border[0]),
#         float(beta2_border[1]),
#         float(beta_higher_border[0]),
#         float(beta_higher_border[1]),
#     ]

#     im = ax.imshow(
#         phase_map,
#         origin='lower',
#         extent=extent,
#         aspect='auto',
#         cmap=cmap,
#         norm=norm,
#         # rasterized=True,
#         interpolation='nearest',
#     )

#     ax.set_xlabel(r'$\beta_2$', fontsize=14)
#     ax.set_ylabel(r'$\beta_{m\geq 3}$', fontsize=14)

#     cbar = fig.colorbar(im, ax=ax, ticks=[0, 1], shrink=0.85)
#     cbar.ax.set_yticklabels([
#         '爆发感染',
#         '无感染',
#     ])

#     if ax is None:
#         plt.show()

def draw_sis_phase(phase_map: np.ndarray, beta2_border: tuple, beta_higher_border: tuple, ax=None):
    """绘制SIS模型的相图

    Parameters
    ----------
    phase_map : np.ndarray
        相图数据
    beta2_border : tuple
        beta2的边界
    beta_higher_border : tuple
        beta_{m>=3}的边界
    ax : matplotlib.axes.Axes, optional
        如果ax参数为None，说明是单图绘制，直接显示图表；如果传入了ax参数，说明是多图绘制，不调用plt.show()，由外部统一显示
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    cmap = ListedColormap(["#fb6a4a", "#fde0dd"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    extent = [
        float(beta2_border[0]),
        float(beta2_border[1]),
        float(beta_higher_border[0]),
        float(beta_higher_border[1]),
    ]

    im = ax.imshow(
        phase_map,
        origin='lower',
        extent=extent,
        aspect='auto',
        cmap=cmap,
        norm=norm,
        interpolation='none',
    )

    ax.set_xlabel(r'$\beta_2$', fontsize=14)
    ax.set_ylabel(r'$\beta_{m\geq 3}$', fontsize=14)
    
    ny, nx = phase_map.shape
    xs = np.linspace(extent[0], extent[1], nx)
    ys = np.linspace(extent[2], extent[3], ny)
    X, Y = np.meshgrid(xs, ys)

    for val, txt in zip((0, 1), ('爆发感染', '无感染')):
        mask = (phase_map == val)
        if not mask.any():
            continue

        cx, cy = X[mask].mean(), Y[mask].mean()
        d2 = (X[mask] - cx) ** 2 + (Y[mask] - cy) ** 2
        k = np.argmin(d2)
        px, py = X[mask][k], Y[mask][k]

        ax.text(
            px, py, txt,
            ha='center', va='center',
            fontsize=14,
        )

    return fig, ax


def draw_sis_init(result: dict, ax=None, title=""):
    """绘制SIS模型的初始感染节点数量对最终感染节点数量的影响

    Parameters
    ----------
    result : dict
        SIS模型的结果字典，键为初始感染节点数量，值为对应的(S, I, t)元组
    ax : matplotlib.axes.Axes, optional
        如果ax参数为None，说明是单图绘制，直接显示图表；如果传入了ax参数，说明是多图绘制，不调用plt.show()，由外部统一显示
    title : str, optional
        图表标题，默认为空字符串
    """
    if ax is None:
        plt.figure(figsize=(8, 6))
    else:
        plt.sca(ax)
    for label, (S, I, t) in result.items():
        plt.plot(t, I, label=label)
    plt.xlabel("时间步")
    plt.ylabel("感染节点数量")
    plt.legend()
    plt.grid(True)
    if title:
        plt.title(title, fontsize=14, y=-0.2, fontname='SimSun')

    if ax is None:
        plt.show()


def draw_sis_init_beta(result: dict, ax=None, title=""):
    """绘制SIS模型的初始感染节点数量对最终感染节点数量的影响

    result结构为表格
    指标，峰值，时间，P

    指标做图例，p是比例做横坐标，峰值/时间做纵坐标，折线图

    一共画两个图，一个是峰值，一个是时间

    不同的图例，用不同的标记表示，把折线图最上面的点标记出来，标记的颜色和折线图的颜色一致
    """


def _sub_draw_sis_community(result, community, ax=None, title=""):
    """绘制SIS模型在不同社团的感染节点数量随时间变化的函数

    Parameters
    ----------
    result : dict
        SIS模型的结果字典，键为时间步，值为对应的事件列表
    community : list
        社团列表，每个元素为该社团的节点列表
    ax : matplotlib.axes.Axes, optional
        如果ax参数为None，说明是单图绘制，直接显示图表；如果传入了ax参数，说明是多图绘制，不调用plt.show()，由外部统一显示
    title : str, optional
        图表标题，默认为空字符串
    """
    if isinstance(result, (list, tuple)) and len(result) == 3:
        _, _, I_arr = result
        I_arr = np.array(I_arr)
        if I_arr.dtype.kind in 'if':
            raise RuntimeError(
                "传入的 I 序列为数值统计，无法按节点类别拆分；请调用 SIS(..., return_all_data=True) 并将返回的事件字典传入本函数")
    event_dict = result.copy()
    t_sorted = sorted(event_dict.keys())
    t_arr = np.array(t_sorted)

    nodes = set()
    for evts in event_dict.values():
        for ev in evts:
            if len(ev) >= 2:
                nodes.add(ev[1])

    if not nodes:
        for cls_nodes in community:
            nodes.update(cls_nodes)

    state = {n: 'S' for n in nodes}
    cls_counts = {cls: [] for cls in range(len(community))}

    for tt in t_sorted:
        evts = event_dict.get(tt, [])
        for ev in evts:
            if not ev:
                continue
            action = ev[0]
            node = ev[1] if len(ev) > 1 else None
            if node is None:
                continue
            if action == 'I':
                state[node] = 'I'
            elif action == 'S':
                state[node] = 'S'

        for cls, cls_nodes in enumerate(community):
            cnt = sum(1 for n in cls_nodes if state.get(n) == 'I')
            cls_counts[cls].append(cnt)

    t_arr = np.array(t_sorted)

    if ax is None:
        plt.figure(figsize=(8, 5))
    else:
        plt.sca(ax)

    colors = plt.rcParams.get(
        'axes.prop_cycle').by_key().get('color', None)
    for idx, (cls, counts) in enumerate(cls_counts.items()):
        c = colors[idx % len(colors)] if colors else None
        plt.plot(t_arr, np.array(counts), '-', label=f'社团{str(cls)}', color=c)

    plt.xlabel("时间", fontsize=14, fontname='SimSun')
    plt.ylabel("感染节点数量", fontsize=14, fontname='SimSun')
    plt.legend(fontsize=12)
    if title:
        plt.title(title, fontsize=14, fontname='SimSun', y=-0.3)
    if ax is None:
        plt.show()


def _sub_draw_sir_community(result, community, ax=None, title=""):
    """绘制SIR模型在不同社团的感染节点数量随时间变化的函数

    Parameters
    ----------
    result : dict
        SIR模型的结果字典，键为时间步，值为对应的事件列表
    community : list
        社团列表，每个元素为该社团的节点列表
    ax : matplotlib.axes.Axes, optional
        如果ax参数为None，说明是单图绘制，直接显示图表；如果传入了ax参数，说明是多图绘制，不调用plt.show()，由外部统一显示
    title : str, optional
        图表标题，默认为空字符串
    """
    if isinstance(result, (list, tuple)) and len(result) == 3:
        _, _, I_arr, _ = result
        I_arr = np.array(I_arr)
        if I_arr.dtype.kind in 'if':
            raise RuntimeError(
                "传入的 I 序列为数值统计，无法按节点类别拆分；请调用 SIR(..., return_all_data=True) 并将返回的事件字典传入本函数")
    event_dict = result.copy()
    t_sorted = sorted(event_dict.keys())
    t_arr = np.array(t_sorted)

    nodes = set()
    for evts in event_dict.values():
        for ev in evts:
            if len(ev) >= 2:
                nodes.add(ev[1])

    if not nodes:
        for cls_nodes in community:
            nodes.update(cls_nodes)

    state = {n: 'S' for n in nodes}
    cls_counts = {cls: [] for cls in range(len(community))}

    for tt in t_sorted:
        evts = event_dict.get(tt, [])
        for ev in evts:
            if not ev:
                continue
            action = ev[0]
            node = ev[1] if len(ev) > 1 else None
            if node is None:
                continue
            if action == 'I':
                state[node] = 'I'
            elif action == 'S':
                state[node] = 'S'
            elif action == 'R':
                state[node] = 'R'

        for cls, cls_nodes in enumerate(community):
            cnt = sum(1 for n in cls_nodes if state.get(n) == 'I')
            cls_counts[cls].append(cnt)

    t_arr = np.array(t_sorted)

    if ax is None:
        plt.figure(figsize=(8, 5))
    else:
        plt.sca(ax)

    colors = plt.rcParams.get(
        'axes.prop_cycle').by_key().get('color', None)
    for idx, (cls, counts) in enumerate(cls_counts.items()):
        c = colors[idx % len(colors)] if colors else None
        plt.plot(t_arr, np.array(counts), '-', label=f'社团{str(cls)}', color=c)

    plt.xlabel("时间", fontsize=14, fontname='SimSun')
    plt.ylabel("感染节点数量", fontsize=14, fontname='SimSun')
    plt.legend(fontsize=12)
    if title:
        plt.title(title, fontsize=14, fontname='SimSun', y=-0.3)
    if ax is None:
        plt.show()


def draw_sis_conmmunity(result: dict, communities: list, title=""):
    """绘制SIS模型在不同社团的感染节点数量随时间变化的函数

    Parameters
    ----------
    result : dict
        SIS模型的结果字典，键为时间步，值为对应的事件列表
    communities : list
        社团列表，每个元素为该社团的节点列表
    title : str, optional
        图表标题，默认为空字符串
    """
    ax = plt.subplots(2, 2, figsize=(10, 8))[1].flatten()
    alphabeta = 'abcd'
    for index in range(len(communities)):
        _sub_draw_sis_community(
            result[index], title=f"({alphabeta[index]}) {title} (社团{index})", community=communities, ax=ax[index])


def draw_community_partition(nodemap, title="", ax=None):
    """绘制社团划分中各社团的国家数量分布的饼图

    Parameters
    ----------
    nodemap : list
        社团划分结果，每个元素为一个社团的节点列表
    title : str, optional
        图表标题，默认为空字符串
    ax : matplotlib.axes.Axes, optional
        如果ax参数为None，说明是单图绘制，直接显示图表；如果传入了ax参数，说明是多图绘制，不调用plt.show()，由外部统一显示
    """
    community_sizes = {community_id: len(
        community) for community_id, community in enumerate(nodemap)}

    labels_communities = [
        f"社团{community_id}" for community_id in community_sizes.keys()]
    sizes_communities = list(community_sizes.values())
    ax.pie(sizes_communities, labels=labels_communities, autopct='%1.1f%%',
           startangle=0, textprops={'fontsize': 15, 'family': 'SimSun'})
    ax.set_title(title, fontsize=20, y=-0.1, fontname='SimSun')
    ax.axis('equal')  # 保持饼图为圆形
    ax.tick_params(axis='both', labelfontfamily='Times New Roman')
    if ax is None:
        plt.show()


def draw_communities_onmap(nodemap: dict, world, title=""):
    """在世界地图上绘制全球高阶贸易社团的分布情况。

    该函数通过合并国家边界地理数据与社团划分数据，使用不同的颜色在地图上为属于
    不同社团的国家着色，未分配社团的国家以灰色显示。最终生成带有图例、经纬网格
    和标题的全球社团分布可视化地图。

    Parameters:
        nodemap (dict): 国家到社团的映射字典。键为国家的ISO_A3_EH代码（如'CHN', 'USA'），
                        值为该国家所属的社团编号（整数）。
        tag (str, optional): 地图标题的附加标签，用于区分不同的社团划分结果。
                             如果为空字符串，则不添加标题。默认为 ""。

    Returns:
        pandas.DataFrame: 包含国家边界地理信息以及对应社团归属信息的数据框。
                          该数据框已过滤掉南极洲，并按国家代码合并了nodemap中的数据。
    """
    import pandas as pd
    # 1. 加载世界国家边界数据（Cartopy内置）
    communities = pd.DataFrame(
        list(nodemap.items()), columns=['country', 'community']
    )
    communities_len = len(set(communities['community']))

    # 3. 合并数据（按国家名）
    world = world.merge(communities, left_on='ISO_A3_EH',
                        right_on='country', how='left')
    world = world[world['NAME'] != 'Antarctica']

    # 4. 设置10种颜色，实际取余数选颜色
    colors = ['#66B2FF', '#99FF99', '#FFCC99', '#FF99CC',
              '#FFFF99', '#99FFCC', '#CC99FF', '#FF99FF', '#FFCCFF']  # 自定义RGB
    cmap = ListedColormap(colors[:communities_len])

    # 5. 绘图
    import cartopy.feature as cfeature
    import cartopy.crs as ccrs
    from matplotlib.patches import Patch
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(
        1, 1, 1, projection=ccrs.PlateCarree(central_longitude=148))
    # 1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor='none',
                   edgecolor='black', linewidth=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

    # 按社团编号着色国家
    world.plot(
        column='community',
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        legend=False,
        missing_kwds={'color': 'lightgray'},  # 无数据国家灰显
        edgecolor='black',
        linewidth=0.2
    )

    # 添加经纬线网格
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)

    legend_elements = [
        Patch(facecolor=colors[i], edgecolor='black', label=f'社团 {i}')
        for i in range(communities_len)
    ]

    # 添加图例
    ax.legend(handles=legend_elements, loc='lower left', fontsize=12)

    # 添加标题
    if title != "":
        plt.title(title, fontsize=18, fontweight='bold')

    plt.tight_layout()
    plt.show()

    return world


def draw_control_partition(drivers, nodemap, title="", ax=None):
    """绘制控制节点在各社团的分布的饼图

    Parameters
    ----------
    drivers : list
        控制节点列表
    nodemap : list
        社团划分结果，每个元素为一个社团的节点列表
    title : str, optional
        图表标题，默认为空字符串
    ax : matplotlib.axes.Axes, optional
        如果ax参数为None，说明是单图绘制，直接显示图表；如果传入了ax参数，说明是多图绘制，不调用plt.show()，由外部统一显示
    """
    community_counts = {}
    for community_id, community in enumerate(nodemap):
        community_counts[community_id] = 0
        for driver in drivers:
            if driver in community:
                community_counts[community_id] += 1
    labels_drivers = [
        f"社团{community_id}" for community_id in community_counts.keys()]
    sizes_drivers = list(community_counts.values())
    ax.pie(sizes_drivers, labels=labels_drivers, autopct='%1.1f%%',
           startangle=0, textprops={'fontsize': 15, 'family': 'SimSun'})
    ax.set_title(title,
                 fontsize=20, y=-0.1, fontname='SimSun')
    ax.axis('equal')  # 保持饼图为圆形
    ax.tick_params(axis='both', labelfontfamily='Times New Roman')


def draw_hnxs_global(global_metrics, title="", ax=None):
    """绘制高阶网络的整体性质

    Parameters
    ----------
    global_metrics : pd.DataFrame
        通过`honmetric.hnxs_global`接口生成的高阶网络整体性质数据。
    title : str, optional
        图表标题，默认为空字符串
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        目标坐标轴对象，默认为None
    """
    if ax is None:
        plt.figure(figsize=(10, 6))
    else:
        plt.sca(ax)
    # 将每个列的值缩放到0-1范围内，便于比较
    global_metrics_scaled = (global_metrics - global_metrics.min()) / \
        (global_metrics.max() - global_metrics.min())
    # 最大最小相等的情况处理，避免除以0
    global_metrics_scaled = global_metrics_scaled.fillna(1)

    global_metrics_scaled.plot(kind='line', figsize=(10, 6))
    if title:
        plt.title(title, fontsize=14, fontname='SimSun', y=-0.3)
    else:
        plt.title('超网络特征变化趋势')
    plt.xlabel('年份')
    plt.ylabel('值')
    plt.xticks(rotation=0)
    plt.legend(loc='upper right')


def draw_test_assortativity(result, ax=None, scatter_kwargs=None):
    """根据 test_assortativity 的返回结果绘制度-余平均度散点图。

    Parameters
    ----------
    result : dict
        test_assortativity 的返回值。
    ax : matplotlib.axes.Axes, optional
        目标坐标轴对象。
    scatter_kwargs : dict, optional
        传给 scatter 的额外参数。
    show : bool, default True
        是否立即显示图像。
    """
    degrees = list(result.keys())
    excesses = list(result.values())

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    else:
        plt.sca(ax)

    scatter_options = {
        's': 45,
        'alpha': 0.8,
        'color': '#1f77b4',
    }
    if scatter_kwargs:
        scatter_options.update(scatter_kwargs)
    plt.scatter(degrees, excesses, **scatter_options)

    plt.xlabel('度')
    plt.ylabel('余平均度')
    plt.title('度与余平均度散点图')
    plt.grid(True, linestyle='--', alpha=0.3)


def draw_attack_results(results, title="", ax=None):
    """绘制攻击结果的函数

    Parameters
    ----------
    results : dict
        攻击结果字典，键为策略名称，值为对应的攻击结果
    title : str, optional
        图表标题，默认为空字符串
    ax : matplotlib.axes.Axes, optional
        如果ax参数为None，说明是单图绘制，直接显示图表；如果传入了ax参数，说明是多图绘制，不调用plt.show()，由外部统一显示
    """
    if ax is None:
        plt.figure(figsize=(8, 5))
    else:
        plt.sca(ax)

    markers = ['o', 's', '^', 'D', 'v', '*', 'P', 'X']
    for idx, (strategy_name, attack_result) in enumerate(results.items()):
        steps = attack_result['steps']
        nc_vals = attack_result['nc_vals']
        plt.plot(steps, nc_vals, label=strategy_name,
                 marker=markers[idx % len(markers)], markersize=5)

    plt.xlabel("攻击比例")
    plt.ylabel("自然连通度")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)

    if ax is None:
        plt.show()
