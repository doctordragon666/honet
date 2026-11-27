"""导出高阶网络数据

对应的是表格导出，内部数据导入和导出，小数只保留3位
"""
from numbers import Number
from pathlib import Path

from pandas import DataFrame, isna, set_option, read_csv, read_excel
set_option('display.float_format', lambda x: '%.3f' % x)


def _format_latex_value(value: float | int | str) -> str:
    """格式化值为LaTeX表格中的字符串"""
    if isna(value):
        return ''
    if isinstance(value, Number) and not isinstance(value, bool):
        numeric_value = float(value)
        if numeric_value.is_integer():
            return str(int(numeric_value))
        return f'{numeric_value:.3f}'
    return str(value)


def to_latex(
    df: DataFrame,
    caption_cn: str | None = None,
    caption_en: str | None = None,
    label: str | None = None,
) -> str:
    """将DataFrame导出为完整的latex表格代码

    Parameters
    ----------
    df : DataFrame
        要导出的DataFrame对象
    caption_cn : str, optional
        中文标题, 默认为None
    caption_en : str, optional
        英文标题, 默认为None
    label : str, optional
        标签, 默认为None

    Returns
    -------
    str
        返回完整的latex表格代码
    """
    columns = [str(column) for column in df.columns]
    column_format = 'l' if len(
        columns) == 1 else f'l*{{{len(columns) - 1}}}{{c}}'

    lines = ['\\begin{center}']
    if caption_cn or caption_en:
        lines.append(
            f'\t\\bicaption[table]{{{caption_cn or ""}}}{{{caption_en or ""}}}'
        )
    lines.append('\t{\\small')
    lines.append('\t\t\\setlength{\\tabcolsep}{4pt}')
    lines.append('\t\t\\renewcommand{\\arraystretch}{1.2}')
    lines.append(f'\t\t\\begin{{tabular}}{{{column_format}}}')
    lines.append('\t\t\t\\hline')
    lines.append('\t\t\t' + ' & '.join(columns) + ' \\\\')
    lines.append('\t\t\t\\hline')

    for row in df.itertuples(index=False, name=None):
        row_text = ' & '.join(_format_latex_value(value) for value in row)
        lines.append(f'\t\t\t{row_text} \\\\')

    lines.append('\t\t\t\\hline')
    lines.append('\t\t\\end{tabular}')
    lines.append('\t}')
    if label:
        lines.append(f'\t\\label{{{label}}}')
    lines.append('\\end{center}')

    text = '\n'.join(lines) + '\n'
    return text


def beautify_global_metrics(global_metrics: DataFrame, path: str, ext: str = 'csv', index: bool = False):
    """美化全局指标导出的样式

    Parameters
    ----------
    global_metrics : DataFrame
        全局指标的表格
    path : str
        导出的文件地址
    ext : str, optional
        扩展名, 默认为'csv'
    index : bool, optional
        是否导出索引, 默认为False
    """
    if ext == 'csv':
        global_metrics.to_csv(path+'.csv', index=index)
    elif ext == 'excel':
        global_metrics.to_excel(path+'.xlsx', index=index)
    elif ext == 'latex':
        text = to_latex(global_metrics)
        Path(path+'.tex').write_text(text, encoding='utf-8')


def beautify_honcommunity(nodemap: dict, path: str, ext: str = 'csv') -> DataFrame:
    """美化高阶网络的社区划分导出的样式

    Parameters
    ----------
    nodemap : dict
        节点到社团的映射字典
    path : str
        导出的文件地址
    ext : str, optional
        扩展名, 默认为'csv'
        可选值为'csv', 'excel', 'latex'

    Returns
    -------
    DataFrame
        返回美化后的社团划分表格
    """
    if not nodemap:
        raise ValueError('nodemap cannot be empty')

    community_to_countries = {}
    for country, community in nodemap.items():
        community_to_countries.setdefault(community, []).append(str(country))

    records = [
        {
            '社团编号': community,
            '国家列表': ','.join(sorted(countries))
        }
        for community, countries in sorted(community_to_countries.items(), key=lambda item: item[0])
    ]

    result_df = DataFrame(records)
    beautify_global_metrics(result_df, path, ext)
    return result_df


def load_honcommunity(path: str, ext: str) -> dict:
    """加载高阶网络的社区划分数据

    Parameters
    ----------
    path : str
        导入的文件地址
    ext : str
        扩展名, 可选值为'csv', 'excel', 'latex'

    Returns
    -------
    dict
        返回节点到社团的映射字典
    """
    if ext == 'csv':
        d = read_csv(path+'.csv', index_col=None)
    elif ext == 'excel':
        d = read_excel(path+'.xlsx', index_col=None)
    elif ext == 'latex':
        raise NotImplementedError('Loading from latex is not implemented yet')
    d = d.dropna(subset=['社团编号', '国家列表'])
    nodemap = {}
    for _, row in d.iterrows():
        community = row['社团编号']
        countries = row['国家列表'].split(',')
        for country in countries:
            nodemap[country] = community
    return nodemap


def load_global_metrics(path: str, ext: str) -> DataFrame:
    """加载高阶网络的整体性质数据

    Parameters
    ----------
    path : str
        导入的文件地址
    ext : str
        扩展名, 可选值为'csv', 'excel', 'latex'

    Returns
    -------
    DataFrame
        返回多个超图的全局指标表格
    """
    if ext == 'csv':
        return DataFrame.from_csv(path+'.csv', index_col=None)
    elif ext == 'excel':
        return DataFrame.from_excel(path+'.xlsx', index_col=None)
    elif ext == 'latex':
        raise NotImplementedError('Loading from latex is not implemented yet')
