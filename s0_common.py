# -*- coding: utf-8 -*-
"""
s0_common.py — 全部分析的公共模块
数据目录默认为脚本同级的 data/, 输出目录默认为 result/;
可用环境变量 RECS_DATA / RECS_RESULT 覆盖
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(os.environ.get('RECS_DATA',
    Path(__file__).resolve().parent / 'data'))
RESULT_DIR = Path(os.environ.get('RECS_RESULT',
    Path(__file__).resolve().parent / 'result'))
RESULT_DIR.mkdir(parents=True, exist_ok=True)

CSV2020 = DATA_DIR / 'recs2020_public_v7.csv'
CODEBOOK2020 = DATA_DIR / 'RECS_2020_Codebook_v7.xlsx'
CSV2015 = DATA_DIR / 'recs2015_public_v4.csv'

# ---------------- 2020: 权重与口径 ----------------
REPS = [f'NWEIGHT{i}' for i in range(1, 61)]     # 60 个 JK1 重复权重
JK_COEF = 59 / 60                                 # EIA 规定系数 (R-1)/R
INCOME_MID = {1:2.5,2:6.25,3:8.75,4:11.25,5:13.75,6:17.5,7:22.5,8:27.5,
              9:32.5,10:37.5,11:45,12:55,13:67.5,14:87.5,15:125,16:200}   # $k
SRC_ELEC = 2.80                                   # source energy 电力折算

# ---------------- 2015: 权重与口径 (仅 s6 疫情稳健性用) ----------------
BRR = [f'BRRWT{i}' for i in range(1, 97)]         # 96 个 Fay-BRR 权重
FAY_FACTOR = 1 / (96 * (1 - 0.5) ** 2)            # rho=0.5 -> 1/24
INCOME_MID15 = {1:10,2:30,3:50,4:70,5:90,6:110,7:130,8:170}

def load_2020():
    df = pd.read_csv(CSV2020)
    from validate_codes import validate_2020
    validate_2020(df)                       # 取值域自动核验(失败即报错)
    new = pd.DataFrame({
        'income_k': df['MONEYPY'].map(INCOME_MID),
        'insecure': (df['SCALEB'].isin([1,2,3]) | df['SCALEG'].isin([1,2,3])
                     | df['SCALEE'].isin([1,2,3])).astype(int),
        'severe':   (df['SCALEB'].eq(1) | df['SCALEG'].eq(1) | df['SCALEE'].eq(1)).astype(int),
        # v3(P1-01): 维度变量显式化。dim_temp_any=任何温度困难; dim_fin_any=任何财务困难;
        # grp_* 为互斥三分组(与secure=insecure==0构成四分). 旧scaleg_only已删除:
        # 它在全样本回归中的参照组混入财务困难户, 不是所称的"vs 安全家庭"
        'dim_fin_any':  (df['SCALEB'].isin([1,2,3]) | df['SCALEE'].isin([1,2,3])).astype(int),
        'dim_temp_any': df['SCALEG'].isin([1,2,3]).astype(int),
        'grp_fin_only': ((df['SCALEB'].isin([1,2,3]) | df['SCALEE'].isin([1,2,3]))
                         & ~df['SCALEG'].isin([1,2,3])).astype(int),
        'grp_temp_only': (df['SCALEG'].isin([1,2,3])
                          & ~(df['SCALEB'].isin([1,2,3]) | df['SCALEE'].isin([1,2,3]))).astype(int),
        'grp_both':     (df['SCALEG'].isin([1,2,3])
                         & (df['SCALEB'].isin([1,2,3]) | df['SCALEE'].isin([1,2,3]))).astype(int),
        'bps':    df['TOTALBTU'] / df['SQFTEST'],
        'bps_en': df['TOTALBTU'] / df['TOTSQFT_EN'],
        'bps_el': df['BTUEL'] / df['SQFTEST'],
        'dps':    df['TOTALDOL'] / df['SQFTEST'],
        'src_btu': SRC_ELEC*df['BTUEL'] + df['TOTALBTU'] - df['BTUEL'],
    })
    return pd.concat([df, new], axis=1)

def load_2015():
    df = pd.read_csv(CSV2015)
    from validate_codes import validate_2015
    validate_2015(df)
    new = pd.DataFrame({
        'income_k': df['MONEYPY'].map(INCOME_MID15),
        'insecure': (df['SCALEB'].isin([1,2,3]) | df['SCALEG'].isin([1,2,3])
                     | df['SCALEE'].isin([1,2,3])).astype(int),
        'severe':   (df['SCALEB'].eq(1) | df['SCALEG'].eq(1) | df['SCALEE'].eq(1)).astype(int),
        # v3(P1-01): 与2020同构的维度变量(见load_2020注释)
        'dim_fin_any':  (df['SCALEB'].isin([1,2,3]) | df['SCALEE'].isin([1,2,3])).astype(int),
        'dim_temp_any': df['SCALEG'].isin([1,2,3]).astype(int),
        'grp_fin_only': ((df['SCALEB'].isin([1,2,3]) | df['SCALEE'].isin([1,2,3]))
                         & ~df['SCALEG'].isin([1,2,3])).astype(int),
        'grp_temp_only': (df['SCALEG'].isin([1,2,3])
                          & ~(df['SCALEB'].isin([1,2,3]) | df['SCALEE'].isin([1,2,3]))).astype(int),
        'grp_both':     (df['SCALEG'].isin([1,2,3])
                         & (df['SCALEB'].isin([1,2,3]) | df['SCALEE'].isin([1,2,3]))).astype(int),
        'bps': df['TOTALBTU'] / df['TOTSQFT_EN'],
        'bps_el': df['BTUEL'] / df['TOTSQFT_EN'],
    })
    return pd.concat([df, new], axis=1)

def trim_threshold(df, ycol):
    """全局截尾阈值: 在'全体正消费/正支出'人群上定义一次, 子样本分析必须沿用"""
    base = df[(df[ycol] > 0) & (df['TOTALDOL'] > 0)]
    return base[ycol].quantile(0.995)

def analysis_sample(df, ycol='bps', threshold=None):
    """正消费/正支出, 剔除强度上0.5%; threshold传入全局阈值时子样本沿用"""
    d = df[(df[ycol] > 0) & (df['TOTALDOL'] > 0)]
    thr = threshold if threshold is not None else d[ycol].quantile(0.995)
    return d[d[ycol] < thr].copy()

def add_cells(d):
    """反事实/双重负担配对单元
    v3(P2-04) 命名纪律: inc3/vint3 是【固定分档】不是三等分位(terciles)——
      收入: <$25k / $25k-<$75k / >=$75k (加权份额约20/41/39%)
      年代: pre-1970 / 1970-1999 / 2000+ (约37/42/21%)
    描述时须用 income bands / vintage categories 并列出阈值;
    房型合并: 独栋+联排 -> sf, 两类公寓 -> apt"""
    d['inc3'] = pd.cut(d['MONEYPY'], [0,7,13,16], labels=['low','mid','high'])
    d['vint3'] = pd.cut(d['YEARMADERANGE'], [0,3,6,9], labels=['pre70','70-99','00+'])
    d['typc'] = d['TYPEHUQ'].map({1:'mobile',2:'sf',3:'sf',4:'apt',5:'apt'})
    return d

def wquantile(x, w, q=0.5):
    """加权分位数(项18): 排序累计权重, side='left', 端点钳制
    v3.1(复查#4): 显式拒绝空输入/总权重非正/非法q(单个零权重合法)"""
    x = np.asarray(x, dtype=float); w = np.asarray(w, dtype=float)
    if len(x) == 0: raise ValueError('wquantile: 空输入')
    if not (0.0 <= q <= 1.0): raise ValueError(f'wquantile: 非法分位数 q={q}')
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError('wquantile: x/w 含非有限值或负权重')
    if w.sum() <= 0: raise ValueError('wquantile: 总权重非正')
    i = np.argsort(x); x, w = x[i], w[i]
    c = np.cumsum(w)
    idx = np.searchsorted(c, q * c[-1], side='left')
    return x[min(idx, len(x)-1)]

def wshare(g, mask, wcol='NWEIGHT'):
    """加权占比(%), 空组保护(项19)"""
    if len(g) == 0: 
        import warnings; warnings.warn('wshare: 空组, 返回NaN')
        return float('nan')
    return float(np.average(np.asarray(mask, dtype=float), weights=g[wcol]) * 100)

def flag_above_and_db(d):
    """单一定义源(项16/17; v3/P1-03 改NA语义):
    above: 强度高于所在单元(收入3档×气候×房型合并×房龄3档)安全户加权中位数;
           单元内安全户<10时基准不可定义 -> above=NaN(不可分类), 不再默认False
    db:    (above==1) ∧ SCALEG∈{1,2,3}; 基准不可定义 -> db=NaN
    v2 把"无法判断"编码为False, 使老移动住宅等无基准单元显示为0%双重负担——
    0是算法默认值而非观测结果(P1-03), v3 起调用方必须区分 ==1 / ==0 / isna
    返回 (d含above/db/cell_defined列, coverage: 不安全户中基准可定义的加权占比%)"""
    d = add_cells(d)
    d['above'] = np.nan; d['cell_defined'] = False
    for _, cell in d.groupby(['inc3','BA_climate','typc','vint3'], observed=True):
        sec = cell[cell['insecure']==0]
        if len(sec) < 10: continue
        med = wquantile(sec['bps'], sec['NWEIGHT'], 0.5)
        d.loc[cell.index, 'cell_defined'] = True
        d.loc[cell.index, 'above'] = (cell['bps'] > med).astype(float)
    d['db'] = np.where(d['cell_defined'],
                       ((d['above']==1) & d['SCALEG'].isin([1,2,3])).astype(float), np.nan)
    ins = d[d['insecure']==1]
    coverage = float(np.average(ins['cell_defined'], weights=ins['NWEIGHT'])*100)
    return d, coverage

def wmedian(x, w):
    return wquantile(x, w, 0.5)

_RANK_WARNED = set()

def wls_beta(Xd, y, w, check=True):
    X = np.column_stack([np.ones(len(Xd))] + [Xd[c].values for c in Xd.columns])
    if check:
        if not np.all(np.isfinite(X)): 
            bad=[c for c in Xd.columns if not np.all(np.isfinite(Xd[c].values))]
            raise ValueError(f'设计矩阵含NaN/Inf: {bad}')
        if not np.all(np.isfinite(y)): raise ValueError('因变量含NaN/Inf')
        if not np.all(np.isfinite(w)) or np.any(w<0): raise ValueError('权重非法')
    sw = np.sqrt(w)
    beta, _res, rank, _sv = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    if check and rank < X.shape[1]:
        key = (rank, X.shape[1])
        if key not in _RANK_WARNED:
            _RANK_WARNED.add(key)
            import warnings
            warnings.warn(f'设计矩阵秩亏: rank={rank} < 列数{X.shape[1]} (完全共线, lstsq最小范数解; 同形态仅警告一次)')
    return beta

def jackknife(est_fn, data):
    """2020 方差: est_fn(data,wcol)->标量; JK1"""
    th = est_fn(data, 'NWEIGHT')
    reps = np.array([est_fn(data, r) for r in REPS])
    return th, np.sqrt(JK_COEF * np.sum((reps - th) ** 2))

def brr_se(est_fn, data):
    """2015 方差: Fay-BRR"""
    th = est_fn(data, 'NWEIGHT')
    reps = np.array([est_fn(data, r) for r in BRR])
    return th, np.sqrt(FAY_FACTOR * np.sum((reps - th) ** 2))

def build_X(data, spec, ins_col='insecure', use_state=False, sqcol='SQFTEST'):
    """2020 逐步控制设计矩阵; 第一列恒为不安全指示
    spec: 0无 | 1+气候地理 | 2+收入住房燃料租购 | 3+住房质量 | 4+温控行为"""
    X = {'insecure': data[ins_col].astype(float).values}
    if spec >= 1:
        X['hdd'] = data['HDD65'].values/1000; X['cdd'] = data['CDD65'].values/1000
        X['urban'] = (data['UATYP10']=='U').astype(float).values
        X['rural'] = (data['UATYP10']=='R').astype(float).values
        if use_state:
            for s in sorted(data['state_postal'].unique())[1:]:
                X[f'st_{s}'] = (data['state_postal']==s).astype(float).values
        else:
            for cz in ['Cold','Hot-Humid','Hot-Dry','Marine','Mixed-Dry','Very-Cold','Subarctic']:
                X[f'cz_{cz}'] = (data['BA_climate']==cz).astype(float).values
    if spec >= 2:
        X['log_inc'] = np.log(data['income_k'].values)
        X['log_sqft'] = np.log(data[sqcol].values)
        X['nmem'] = data['NHSLDMEM'].values
        # v3(P2-01): 产权三分类。码本 KOWNRENT: 1=自有(参照), 2=租住, 3=不付租金居住;
        # v2 把 3 并入参照组并称其为"owner-occupied", 名不副实(分析样本178户/约1.22M加权)
        X['rent'] = (data['KOWNRENT']==2).astype(float).values
        X['occ_free'] = (data['KOWNRENT']==3).astype(float).values
        for t in [1,3,4,5]: X[f'type_{t}'] = (data['TYPEHUQ']==t).astype(float).values
        for y in range(1,9): X[f'vint_{y}'] = (data['YEARMADERANGE']==y).astype(float).values
        for f in [1,2,3,7,99]: X[f'fuel_{f}'] = (data['FUELHEAT']==f).astype(float).values
        X['no_heat'] = (data['FUELHEAT']==-2).astype(float).values   # 审计表: 751户无采暖, 独立指示
    if spec >= 3:
        X['poor_insul'] = (data['ADQINSUL']>=3).astype(float).values
        X['drafty'] = (data['DRAFTY'].isin([1,2])).astype(float).values
        X['single_pane'] = (data['TYPEGLASS']==1).astype(float).values
        X['old_equip'] = (data['EQUIPAGE']>=5).astype(float).values
    if spec >= 4:
        # 插补口径(审计表): 仅-2为缺失(无采暖/无制冷), 用NWEIGHT加权均值插补(固定预处理);
        # 缺失指示由 no_heat(spec2) / no_ac(下方) 承担, 防止插补值携带虚假信息
        wts = data['NWEIGHT'].values
        for v, n in [('TEMPHOME','temphome'), ('TEMPHOMEAC','tempac')]:
            x = data[v].values.astype(float); m = x < 0
            x[m] = np.average(x[~m], weights=wts[~m]); X[n] = x
        X['no_ac'] = (data['TEMPHOMEAC']==-2).astype(float).values   # 审计表: 2325户无制冷, 独立指示
        X['athome5'] = (data['ATHOME']==5).astype(float).values
    return pd.DataFrame(X)

def build_X15(data, spec, ins_col='insecure'):
    """2015 设计矩阵 (面积用TOTSQFT_EN, 气候5区, 收入8组, 燃料码21=other)"""
    X = {'insecure': data[ins_col].astype(float).values}
    if spec >= 1:
        X['hdd'] = data['HDD65'].values/1000; X['cdd'] = data['CDD65'].values/1000
        X['urban'] = (data['UATYP10']=='U').astype(float).values
        X['rural'] = (data['UATYP10']=='R').astype(float).values
        for cz in ['Hot-Humid','Hot-Dry/Mixed-Dry','Marine','Mixed-Humid']:
            X[f'cz_{cz}'] = (data['CLIMATE_REGION_PUB']==cz).astype(float).values
    if spec >= 2:
        X['log_inc'] = np.log(data['income_k'].values)
        X['log_sqft'] = np.log(data['TOTSQFT_EN'].values)
        X['nmem'] = data['NHSLDMEM'].values
        # v3(P2-01): 与2020一致, 产权三分类(2015码本同为1自有/2租住/3免租居住)
        X['rent'] = (data['KOWNRENT']==2).astype(float).values
        X['occ_free'] = (data['KOWNRENT']==3).astype(float).values
        for t in [1,3,4,5]: X[f'type_{t}'] = (data['TYPEHUQ']==t).astype(float).values
        for y in range(1,8): X[f'vint_{y}'] = (data['YEARMADERANGE']==y).astype(float).values
        for f in [1,2,3,7,21]: X[f'fuel_{f}'] = (data['FUELHEAT']==f).astype(float).values
        X['no_heat'] = (data['FUELHEAT']==-2).astype(float).values   # 2015: 258户无采暖
    if spec >= 3:
        X['poor_insul'] = (data['ADQINSUL']>=3).astype(float).values
        X['drafty'] = (data['DRAFTY'].isin([1,2])).astype(float).values
        X['single_pane'] = (data['TYPEGLASS']==1).astype(float).values
        X['old_equip'] = (data['EQUIPAGE'].isin([42,5])).astype(float).values  # 2015码: 42=15-19yr, 5=20+yr(码本核实)
    if spec >= 4:
        wts = data['NWEIGHT'].values
        for v, n in [('TEMPHOME','temphome'), ('TEMPHOMEAC','tempac')]:
            x = data[v].values.astype(float); m = x < 0
            x[m] = np.average(x[~m], weights=wts[~m]); X[n] = x
        X['no_ac'] = (data['TEMPHOMEAC']==-2).astype(float).values   # 2015: 737户无制冷
        X['athome5'] = (data['ATHOME']==5).astype(float).values
    return pd.DataFrame(X)

def premium_coef(data, wcol, spec=2, ycol='bps', ins_col='insecure',
                 use_state=False, sqcol='SQFTEST', builder=build_X):
    Xd = builder(data, spec, ins_col, use_state, sqcol) if builder is build_X \
         else builder(data, spec, ins_col)
    return wls_beta(Xd, np.log(data[ycol].values), data[wcol].values)[1]

def pct(b): return 100*(np.exp(b)-1)

def save_csv(df, path, index=True):
    """CSV写入保护: Windows下文件被Excel/WPS打开会加独占锁 -> 重试一次并给出明确指引"""
    import time
    for attempt in range(2):
        try:
            df.to_csv(path, index=index); return
        except PermissionError:
            if attempt == 0:
                print(f'[提示] {getattr(path,"name",path)} 被其他程序占用(通常是Excel/WPS正打开该文件), 2秒后重试...')
                time.sleep(2)
    raise PermissionError(f'无法写入 {path}: 请关闭 Excel/WPS 中打开的该文件后重新运行本脚本')

# ---------------- 图形样式 (Nature/Science 风格 token) ----------------
STYLE = dict(INK='#0b0b0b', SEC='#52514e', MUT='#898781', GRID='#e1e0d9',
             BASE='#c3c2b7', BLUE='#2a78d6', ORANGE='#eb6834', RED='#e34948',
             NEUT='#b5b3ac', DARK='#52514e',
             SEQ={100:'#cde2fb',150:'#b7d3f6',200:'#9ec5f4',300:'#6da7ec',
                  450:'#2a78d6',550:'#1c5cab',700:'#0d366b'})

def pub_font_family():
    """平台自适应字体: Windows 用 Arial(显式注册, 免疫缓存问题), 其他用 Liberation Sans"""
    import platform
    if platform.system() == 'Windows':
        from matplotlib import font_manager
        for f in (r'C:\Windows\Fonts\arial.ttf', r'C:\Windows\Fonts\arialbd.ttf'):
            try: font_manager.fontManager.addfont(f)
            except Exception: pass
        return ['Arial', 'DejaVu Sans']
    return ['Liberation Sans', 'DejaVu Sans']

def set_pub_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
      'font.family':pub_font_family(),'font.size':7,
      'axes.linewidth':0.6,'xtick.major.width':0.6,'ytick.major.width':0.6,
      'xtick.major.size':2.5,'ytick.major.size':2.5,'xtick.labelsize':6.4,
      'ytick.labelsize':6.6,'xtick.color':STYLE['MUT'],'ytick.color':STYLE['MUT'],
      'axes.edgecolor':STYLE['BASE'],'text.color':STYLE['INK'],
      'axes.labelcolor':STYLE['SEC'],'pdf.fonttype':42})
