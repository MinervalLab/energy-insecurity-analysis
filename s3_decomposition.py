# -*- coding: utf-8 -*-
"""s3 — Oaxaca-Blinder 分解 (table3 / fig2 数据)
主报告参考系: pooled coefficients WITH group indicator (Jann 2008, Stata oaxaca pooled 定义式)
  该参考系的已知恒等性质: unexplained = 合并回归中组指示变量系数 (程序内数值验证)
补充: secure-β / insecure-β / Neumark pooled(无指示) / threefold(E,C,I) 参考系对照
基准不变性: 分类变量完备时家族层级贡献对虚拟变量基准选择不变(Σ Δp_k=0), 程序内数值验证
输出: table3_oaxaca.csv, table3b_altrefs.csv  |  耗时 2-4 分钟"""
import numpy as np, pandas as pd
from s0_common import save_csv
from s0_common import (load_2020, analysis_sample, build_X, wls_beta,
                       REPS, JK_COEF, RESULT_DIR)

df = load_2020(); d = analysis_sample(df)
BLOCKS = {
 'geo_climate': ['hdd','cdd','urban','rural']+[f'cz_{c}' for c in
     ['Cold','Hot-Humid','Hot-Dry','Marine','Mixed-Dry','Very-Cold','Subarctic']],
 'structure':   ['log_sqft']+[f'type_{t}' for t in [1,3,4,5]]+[f'vint_{y}' for y in range(1,9)],
 'fuel':        [f'fuel_{f}' for f in [1,2,3,7,99]]+['no_heat'],
 'econ_demo':   ['log_inc','nmem','rent','occ_free'],   # v3(P2-01): 三分类新增列必须入块, 否则explained漏项破坏恒等式
 'quality':     ['poor_insul','drafty','single_pane','old_equip'],
 'behavior':    ['temphome','tempac','no_ac','athome5']}

def _prep(data):
    Xd = build_X(data, 4)
    ins = data['insecure'].values.astype(bool)
    y = np.log(data['bps'].values)
    return Xd, ins, y

def _means(Xd, ins, w):
    mi = {c: np.average(Xd[c].values[ins], weights=w[ins]) for c in Xd.columns}
    ms = {c: np.average(Xd[c].values[~ins], weights=w[~ins]) for c in Xd.columns}
    return mi, ms

def ob_pooled_indicator(data, wcol):
    """主参考系: pooled WITH indicator"""
    Xd, ins, y = _prep(data); w = data[wcol].values
    beta = wls_beta(Xd, y, w, check=False)          # Xd第一列即insecure指示
    bmap = dict(zip(Xd.columns, beta[1:]))
    mi, ms = _means(Xd, ins, w)
    gap = np.average(y[ins],weights=w[ins]) - np.average(y[~ins],weights=w[~ins])
    out = {}; expl = 0.0
    for b, cols in BLOCKS.items():
        c_ = sum((mi[c]-ms[c])*bmap[c] for c in cols)
        out[f'ex_{b}'] = c_; expl += c_
    out.update(gap=gap, explained=expl, unexplained=gap-expl, indicator_coef=bmap['insecure'])
    return out

# ---------- 主结果 + JK ----------
th = ob_pooled_indicator(d,'NWEIGHT')
reps = [ob_pooled_indicator(d,r) for r in REPS]
rows = []
print('=== 主参考系: pooled with group indicator (JK se) ===')
for k in [k for k in th if k!='indicator_coef']:
    rv = np.array([r[k] for r in reps])
    se = float(np.sqrt(JK_COEF*np.sum((rv-th[k])**2)))
    # v3(P1-08): 份额(占gap%)的SE按"每重复权重同时重算分子与分母"的完整比率JK,
    # 不再用 se/gap 缩放(那把随机的gap当固定分母, 结构份额SE被低估 5.76pp->9.39pp)
    ratio = 100*th[k]/th['gap']
    rr = np.array([100*r[k]/r['gap'] for r in reps])
    ratio_se = float(np.sqrt(JK_COEF*np.sum((rr-ratio)**2)))
    rows.append([k, th[k], se, ratio, ratio_se])
    print(f'{k:15s}: {th[k]:+.4f} (se={se:.4f}) [{ratio:+.1f}% of gap, ratio-JK se={ratio_se:.2f}pp]')
print(f'恒等性质验证: unexplained={th["unexplained"]:+.6f} vs 指示变量系数={th["indicator_coef"]:+.6f} '
      f'(差异 {abs(th["unexplained"]-th["indicator_coef"]):.2e}, 应≈0为实现正确的证据)')
# 性质校验: explained与unexplained份额之和恒为100% -> 两者份额SE必须相等
_ex = 100*np.array([r['explained']/r['gap'] for r in reps]); _th_ex = 100*th['explained']/th['gap']
_un = 100*np.array([r['unexplained']/r['gap'] for r in reps]); _th_un = 100*th['unexplained']/th['gap']
_se_ex = np.sqrt(JK_COEF*np.sum((_ex-_th_ex)**2)); _se_un = np.sqrt(JK_COEF*np.sum((_un-_th_un)**2))
assert abs(_se_ex-_se_un) < 1e-9, 'explained/unexplained 份额SE应相等(和恒为100%)'
save_csv(pd.DataFrame(rows, columns=['component','coef','se','pct_of_gap','pct_of_gap_se']).round(4), RESULT_DIR/'table3_oaxaca.csv', index=False)

# ---------- v3(P2-10): structure块分项(面积/房型/年代), 101%不能笼统归"低效建筑" ----------
def structure_split(data, wcol):
    Xd, ins, y = _prep(data); w = data[wcol].values
    beta = wls_beta(Xd, y, w, check=False)
    bmap = dict(zip(Xd.columns, beta[1:]))
    mi, ms = _means(Xd, ins, w)
    parts = {'sqft': ['log_sqft'],
             'type': [f'type_{t}' for t in [1,3,4,5]],
             'vintage': [f'vint_{y_}' for y_ in range(1,9)]}
    return {p: sum((mi[c]-ms[c])*bmap[c] for c in cols) for p, cols in parts.items()}
sp = structure_split(d,'NWEIGHT')
sp_r = [structure_split(d,r) for r in REPS]
srows = []
print('=== structure块分项 (v3/P2-10) ===')
for p in ['sqft','type','vintage']:
    rv = np.array([r[p] for r in sp_r])
    se = float(np.sqrt(JK_COEF*np.sum((rv-sp[p])**2)))
    srows.append([f'ex_structure_{p}', sp[p], se, 100*sp[p]/th['gap']])
    print(f'ex_structure_{p:8s}: {sp[p]:+.4f} (se={se:.4f}) [{100*sp[p]/th["gap"]:+.0f}% of gap]')
assert abs(sum(sp.values()) - th['ex_structure']) < 1e-10, '分项之和应等于structure块总贡献'
save_csv(pd.DataFrame(srows, columns=['component','coef','se','pct_of_gap']).round(4),
         RESULT_DIR/'table3c_structure_split.csv', index=False)

# ---------- 补充: 其他参考系 (点估计) ----------
def ob_ref(data, wcol, ref):
    """twofold, 参考系数 ref ∈ {secure, insecure, neumark}"""
    Xd, ins, y = _prep(data); w = data[wcol].values
    Xc = Xd.drop(columns=['insecure'])
    if ref=='secure':   beta = wls_beta(Xc[~pd.Series(ins).values], y[~ins], w[~ins], check=False)
    elif ref=='insecure': beta = wls_beta(Xc[pd.Series(ins).values], y[ins], w[ins], check=False)
    else:               beta = wls_beta(Xc, y, w, check=False)      # Neumark: pooled无指示
    bmap = dict(zip(Xc.columns, beta[1:]))
    mi, ms = _means(Xd, ins, w)
    gap = np.average(y[ins],weights=w[ins]) - np.average(y[~ins],weights=w[~ins])
    expl = sum((mi[c]-ms[c])*bmap[c] for c in Xc.columns)
    return gap, expl, gap-expl

def ob_threefold(data, wcol):
    """threefold(以secure为基): E=ΔX'β_s, C=X̄_s'Δβ+Δα, I=ΔX'Δβ"""
    Xd, ins, y = _prep(data); w = data[wcol].values
    Xc = Xd.drop(columns=['insecure'])
    bs = wls_beta(Xc[~pd.Series(ins).values], y[~ins], w[~ins], check=False)
    bi = wls_beta(Xc[pd.Series(ins).values], y[ins], w[ins], check=False)
    mi, ms = _means(Xd, ins, w)
    E = sum((mi[c]-ms[c])*bs[j+1] for j,c in enumerate(Xc.columns))
    C = (bi[0]-bs[0]) + sum(ms[c]*(bi[j+1]-bs[j+1]) for j,c in enumerate(Xc.columns))
    I = sum((mi[c]-ms[c])*(bi[j+1]-bs[j+1]) for j,c in enumerate(Xc.columns))
    return E, C, I

# 表自包含: 首行放主参考系(与 table3 一致), 便于跨参考系比较; interaction 列仅 threefold 行使用
alt = [['twofold ref=pooled w/ indicator (primary, = table3_oaxaca)',
        th['gap'], th['explained'], th['unexplained'], np.nan]]
for ref in ['secure','insecure','neumark']:
    g,e,u = ob_ref(d,'NWEIGHT',ref)
    alt.append([f'twofold ref={ref}', g, e, u, np.nan])
    print(f'twofold ref={ref}: explained={e:+.4f} ({100*e/g:+.0f}%), unexplained={u:+.4f}')
E,C,I = ob_threefold(d,'NWEIGHT')
alt.append(['threefold (base=secure)', E+C+I, E, C, I])
print(f'threefold: E={E:+.4f}, C={C:+.4f}, I={I:+.4f} (E+C+I={E+C+I:+.4f} 应=gap)')
assert abs(E+C+I - th['gap']) < 1e-10, 'threefold 三项之和应精确等于 gap'
save_csv(pd.DataFrame(alt, columns=['reference','gap_or_sum','explained_or_E','unexplained_or_C','interaction_I']).round(4), RESULT_DIR/'table3b_altrefs.csv', index=False)

# ---------- 基准不变性数值验证 ----------
def block_totals_with_base(data, type_base):
    """把房型基准从2换为type_base后重算structure块贡献, 应与主结果一致"""
    Xd, ins, y = _prep(data); w = data['NWEIGHT'].values
    types = [1,2,3,4,5]
    Xd2 = Xd.drop(columns=[f'type_{t}' for t in [1,3,4,5]])
    for t in types:
        if t != type_base:
            Xd2[f'type_{t}'] = (data['TYPEHUQ']==t).astype(float).values
    beta = wls_beta(Xd2, y, w, check=False)
    bmap = dict(zip(Xd2.columns, beta[1:]))
    mi = {c: np.average(Xd2[c].values[ins], weights=w[ins]) for c in Xd2.columns}
    ms = {c: np.average(Xd2[c].values[~ins], weights=w[~ins]) for c in Xd2.columns}
    cols = ['log_sqft']+[f'type_{t}' for t in types if t!=type_base]+[f'vint_{y_}' for y_ in range(1,9)]
    return sum((mi[c]-ms[c])*bmap[c] for c in cols)
b2 = block_totals_with_base(d, 2); b5 = block_totals_with_base(d, 5)
print(f'基准不变性: structure块贡献 base=独栋 {b2:+.6f} vs base=公寓5+ {b5:+.6f} (差异 {abs(b2-b5):.2e}, 应≈0)')
print('已存 table3_oaxaca.csv / table3b_altrefs.csv ->', RESULT_DIR)
