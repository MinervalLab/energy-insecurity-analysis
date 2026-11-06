# -*- coding: utf-8 -*-
"""s8 — 分用途分解: 条件强度差按用途拆分
设计:
  - 用途划分采用 EIA 校准分用途变量的完备五分: TOTALBTUSPH(空间采暖, 全燃料) +
    BTUELCOL(制冷, 纯电) + TOTALBTUWTH(热水, 全燃料) + BTUELRFG(冰箱) +
    TOTALBTUOTH(其他) = TOTALBTU (本脚本开头对全样本断言, 容差为校准舍入级)
  - 呈现四组: 采暖 / 制冷 / 热水 / 其他基荷(冰箱并入其他)
  - 结果变量用水平强度(kBtu/ft^2)而非对数: (i)制冷12.6%、采暖4.7%为真实零值户,
    取对数需任意处理; (ii)WLS对y线性 -> 四组系数之和精确等于总强度水平回归系数
    (程序内断言), 分解可对账
  - 控制集与主模型完全一致(M2/M4), 样本=主分析样本(同一批家庭, 不按用途重截尾)
  - caveat: 分用途数值是 EIA 的工程-统计校准分摊(calibrated end-use estimates),
    非分项计量, 解读时须注明
输出: table6_enduse.csv, fig_enduse.json  |  耗时 ~4-6 分钟(JK x 10个回归)"""
import numpy as np, pandas as pd, json
from s0_common import save_csv
from s0_common import (load_2020, analysis_sample, build_X, wls_beta, pct,
                       REPS, JK_COEF, RESULT_DIR)

df = load_2020()

# ---- 用途划分与闭合断言(全样本, 未截尾) ----
EU = {'sph': ['TOTALBTUSPH'],
      'col': ['BTUELCOL'],
      'wth': ['TOTALBTUWTH'],
      'oth': ['BTUELRFG', 'TOTALBTUOTH']}   # 冰箱并入其他基荷
_sum = sum(df[c] for cols in EU.values() for c in cols)
_gap = (_sum - df['TOTALBTU']).abs().max()
assert _gap < 10, f'用途闭合失败: max|diff|={_gap}'   # 校准舍入级(观测值~7 kBtu)
print(f'用途闭合: SPH+COL+WTH+RFG+OTH = TOTALBTU, max|diff| = {_gap:.2f} kBtu (舍入级)')

d = analysis_sample(df)          # 主分析样本, 与主模型同一批家庭
for k, cols in EU.items():
    d[f'y_{k}'] = sum(d[c] for c in cols) / d['SQFTEST']
d['y_tot'] = d['bps']            # 总强度(水平), 闭合对账用

NAMES = {'sph': 'Space heating', 'col': 'Space cooling',
         'wth': 'Water heating', 'oth': 'Other end uses (incl. refrigeration)'}

def coef_level(dd, w, ycol, spec):
    Xd = build_X(dd, spec)
    return wls_beta(Xd, dd[ycol].values, dd[w].values, check=False)[1]

rows = []; fig = {}
W = 'NWEIGHT'
sec = d[d['insecure']==0]; ins = d[d['insecure']==1]
for spec, mlab in [(2, 'M2'), (4, 'M4')]:
    print(f'=== {mlab} 控制集: 分用途条件强度差 (kBtu/ft^2, JK se) ===')
    csum = 0.0; csum_r = np.zeros(len(REPS))
    for k in ['sph', 'col', 'wth', 'oth']:
        th = coef_level(d, W, f'y_{k}', spec)
        rv = np.array([coef_level(d, r, f'y_{k}', spec) for r in REPS])
        se = float(np.sqrt(JK_COEF*np.sum((rv-th)**2)))
        csum += th; csum_r += rv
        m_sec = float(np.average(sec[f'y_{k}'], weights=sec[W]))   # 安全组均值(语境用)
        rows.append([f'{mlab}: {NAMES[k]}', th, se, th/se, m_sec, 100*th/m_sec])
        print(f'  {NAMES[k]:38s}: {th:+.3f} (se={se:.3f}, t={th/se:.1f}) | 安全组均值 {m_sec:.2f} -> 相对差 {100*th/m_sec:+.1f}%')
        if spec == 4:
            fig[k] = {'name': NAMES[k], 'coef': th, 'se': se, 'sec_mean': m_sec}
    # 闭合对账: WLS对y线性 -> 四组系数之和精确等于"分量之和强度"的回归系数;
    # 对 bps(=TOTALBTU/SQFTEST) 的系数差仅来自校准变量的户级舍入(<=7.24 kBtu), 单独报告
    d['y_sum'] = d[[f'y_{k}' for k in EU]].sum(axis=1)
    th_sum = coef_level(d, W, 'y_sum', spec)
    assert abs(csum - th_sum) < 1e-9, f'{mlab} 系数闭合失败: {csum} vs {th_sum}'
    th_tot = coef_level(d, W, 'y_tot', spec)
    rv_tot = np.array([coef_level(d, r, 'y_tot', spec) for r in REPS])
    se_tot = float(np.sqrt(JK_COEF*np.sum((rv_tot-th_tot)**2)))
    m_sec_t = float(np.average(sec['y_tot'], weights=sec[W]))
    rows.append([f'{mlab}: Total (level; sum of end uses, identity)', th_tot, se_tot,
                 th_tot/se_tot, m_sec_t, 100*th_tot/m_sec_t])
    print(f'  闭合: Σ用途系数={csum:+.6f} ≡ 分量和强度系数(差 {abs(csum-th_sum):.1e}); '
          f'vs TOTALBTU口径系数 {th_tot:+.6f} (差 {abs(csum-th_tot):.1e}, 源于校准舍入)')

save_csv(pd.DataFrame(rows, columns=['estimate','coef_kbtu_ft2','se','t',
         'secure_mean_kbtu_ft2','pct_of_secure_mean']).round(4),
         RESULT_DIR/'table6_enduse.csv', index=False)
json.dump(fig, open(RESULT_DIR/'fig_enduse.json','w'))
print('已存 table6_enduse.csv / fig_enduse.json ->', RESULT_DIR)

# ---- 敏感性: 面积五分位虚拟变量替代 log 面积(非线性面积控制) ----
# 动机: 基荷含按户近恒定的固定负荷, 其强度与面积近似成反比(~1/A),
# log面积线性项可能吸收不全 -> 用五分位虚拟变量放开函数形式, 检验分用途结论是否依赖线性设定
print('=== 敏感性: M4, log面积 -> 加权面积五分位虚拟变量 ===')
from s0_common import wquantile
qs = [wquantile(d['SQFTEST'].values, d[W].values, q) for q in (0.2, 0.4, 0.6, 0.8)]
d['_abin'] = np.digitize(d['SQFTEST'], qs)          # 0..4
def coef_bins(dd, w, ycol):
    Xd = build_X(dd, 4).drop(columns=['log_sqft'])
    for b in range(1, 5):
        Xd[f'abin_{b}'] = (dd['_abin']==b).astype(float).values
    return wls_beta(Xd, dd[ycol].values, dd[w].values, check=False)[1]
rows2 = []
for k in ['sph', 'col', 'wth', 'oth']:
    th = coef_bins(d, W, f'y_{k}')
    rv = np.array([coef_bins(d, r, f'y_{k}') for r in REPS])
    se = float(np.sqrt(JK_COEF*np.sum((rv-th)**2)))
    m_sec = float(np.average(sec[f'y_{k}'], weights=sec[W]))
    rows2.append([f'M4 area-quintile bins: {NAMES[k]}', th, se, th/se, m_sec, 100*th/m_sec])
    print(f'  {NAMES[k]:38s}: {th:+.3f} (se={se:.3f}, t={th/se:.1f}) -> 相对差 {100*th/m_sec:+.1f}%')
save_csv(pd.DataFrame(rows2, columns=['estimate','coef_kbtu_ft2','se','t',
         'secure_mean_kbtu_ft2','pct_of_secure_mean']).round(4),
         RESULT_DIR/'table6b_enduse_areabins.csv', index=False)
print('已存 table6b_enduse_areabins.csv ->', RESULT_DIR)

# ---- 备忘(M4): 冰箱单列 vs 其他(剔冰箱) ----
# 两系数之和应精确等于主表 oth 系数(线性恒等)
d['y_rfg'] = d['BTUELRFG']/d['SQFTEST']
d['y_oth_norfg'] = d['TOTALBTUOTH']/d['SQFTEST']
memo = []
for ycol, nm in [('y_rfg','M4 memo: Refrigeration (separate)'),
                 ('y_oth_norfg','M4 memo: Other excl. refrigeration')]:
    th = coef_level(d, W, ycol, 4)
    rv = np.array([coef_level(d, r, ycol, 4) for r in REPS])
    se = float(np.sqrt(JK_COEF*np.sum((rv-th)**2)))
    m_sec = float(np.average(sec[ycol], weights=sec[W]))
    memo.append([nm, th, se, th/se, m_sec, 100*th/m_sec])
    print(f'{nm}: {th:+.4f} (se={se:.4f}, t={th/se:.1f}) -> 相对差 {100*th/m_sec:+.1f}%')
assert abs(memo[0][1]+memo[1][1] - [r for r in rows if r[0]=='M4: Other end uses (incl. refrigeration)'][0][1]) < 1e-9, '冰箱拆分闭合失败'
save_csv(pd.DataFrame(memo, columns=['estimate','coef_kbtu_ft2','se','t',
         'secure_mean_kbtu_ft2','pct_of_secure_mean']).round(4),
         RESULT_DIR/'table6c_refrig_memo.csv', index=False)
print('已存 table6c_refrig_memo.csv ->', RESULT_DIR)
