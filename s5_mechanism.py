# -*- coding: utf-8 -*-
"""s5 — 机制定位: 电力/气候/电暖器/设备中介/援助 (table4b / fig4 数据)
输出: result/table4b_mechanism.csv, result/fig3_mech.json  |  耗时 4-7 分钟"""
import numpy as np, pandas as pd, json
from s0_common import save_csv
from s0_common import (load_2020, analysis_sample, build_X, wls_beta, jackknife,
                       pct, REPS, JK_COEF, RESULT_DIR)

raw = load_2020(); rows = []; f3 = {}

print('=== A. 分燃料定位 (全局截尾阈值; 子群体估计为条件性对比, 见措辞纪律) ===')
def coef_fn(dd, w):
    return wls_beta(build_X(dd,2), np.log(dd['y'].values), dd[w].values, check=False)[1]
# 项6a修正: 截尾阈值在全体正消费人群上定义一次, 子群体沿用同一阈值
full_el = raw.assign(y=raw['BTUEL']/raw['SQFTEST'])
thr_el = full_el[(full_el['y']>0)&(full_el['TOTALDOL']>0)]['y'].quantile(0.995)
full_ng = raw.assign(y=raw['BTUNG']/raw['SQFTEST'])
thr_ng = full_ng[(full_ng['y']>0)&(full_ng['TOTALDOL']>0)]['y'].quantile(0.995)
def samp(base, num, thr):
    b = base.assign(y=num/base['SQFTEST'])
    d_ = b[(b['y']>0)&(b['TOTALDOL']>0)]
    return d_[d_['y']<thr].copy()
cases = [('Electricity intensity, all households', samp(raw, raw['BTUEL'], thr_el)),
         ('Gas intensity, conditional on gas use (selected subsample)',
          samp(raw[raw['BTUNG']>0], raw.loc[raw['BTUNG']>0,'BTUNG'], thr_ng)),
         ('Electricity intensity, conditional on gas main heat (selected subsample)',
          samp(raw[raw['FUELHEAT']==1], raw.loc[raw['FUELHEAT']==1,'BTUEL'], thr_el))]
f3['fuel'] = []
for name, dd in cases:
    th, se = jackknife(coef_fn, dd)
    print(f'{name} (n={len(dd)}): {th:+.4f} (se={se:.4f}, t={th/se:.1f}) -> {pct(th):+.1f}%')
    rows.append([name, th, se, th/se]); f3['fuel'].append([name, th, se])

d = analysis_sample(raw)   # 主分析样本(A2/B/C/D共用)
print('=== A2. 两部模型参与边际 (项6b: 无样本筛选的补充) ===')
def gas_participation(dd, w):
    # LPM: P(任何燃气使用) ~ insecure + 控制(不含燃料虚拟变量, 因其与因变量重合)
    Xd = build_X(dd, 2)
    Xd = Xd.drop(columns=[c for c in Xd.columns if c.startswith('fuel_') or c=='no_heat'])
    yy = (dd['BTUNG']>0).astype(float).values
    return wls_beta(Xd, yy, dd[w].values, check=False)[1]
th, se = jackknife(gas_participation, d)
print(f'P(用燃气) 中 insecure 系数(LPM): {th:+.4f} (se={se:.4f}, t={th/se:.1f})')
rows.append(['Any-gas-use participation margin (LPM, main analysis sample)', th, se, th/se])
# v3(P2-05): 补对称检验 —— 气采暖户子样本(6.9%电力行)对应的参与边际是 P(主采暖=燃气),
# v2 只检验了任何用气一个边际却为两个条件子样本背书。零结果不排除未观测选择。
def gasheat_participation(dd, w):
    Xd = build_X(dd, 2)
    Xd = Xd.drop(columns=[c for c in Xd.columns if c.startswith('fuel_') or c=='no_heat'])
    yy = (dd['FUELHEAT']==1).astype(float).values
    return wls_beta(Xd, yy, dd[w].values, check=False)[1]
th, se = jackknife(gasheat_participation, d)
print(f'P(气采暖) 中 insecure 系数(LPM): {th:+.4f} (se={se:.4f}, t={th/se:.1f})')
rows.append(['Gas-main-heat participation margin (LPM, main analysis sample)', th, se, th/se])

# 参数化说明(项10): 主效应未中心化+交互项中心化, 与两者皆中心化仅差截距重参数化;
# 交互系数不变, insecure主效应=均值气候处的溢价, 为报告口径
d = analysis_sample(raw)
def clim_int(dd, w, var):
    Xd = build_X(dd, 2)
    c = dd[var].values/1000 - np.average(dd[var].values/1000, weights=dd[w].values)
    Xd['ix'] = Xd['insecure']*c
    b = wls_beta(Xd, np.log(dd['bps'].values), dd[w].values)
    return b[1], b[len(Xd.columns)]
f3['climate'] = {}
for var in ['HDD65','CDD65']:
    th = clim_int(d,'NWEIGHT',var)
    reps = np.array([clim_int(d,r,var) for r in REPS])
    se = np.sqrt(JK_COEF*np.sum((reps[:,1]-th[1])**2))
    print(f'insecure x {var}: {th[1]:+.4f}/千度日 (se={se:.4f}, t={th[1]/se:.1f})')
    rows.append([f'insecure x {var} (per 1000 deg-days)', th[1], se, th[1]/se])
    f3['climate'][var] = {'main': th[0], 'inter': th[1], 'se': float(se),
                          'mean': float(np.average(d[var]/1000, weights=d['NWEIGHT']))}
    # 隐含溢价网格 (p10/p50/p90 处, 完整JK)
    # v3(P2-08): 分位点改用调查权重加权分位数(全国分布), v2 的 np.percentile 是
    # 未加权样本分位数(如CDD P10/50/90: 476/1180/3358 vs 加权 623/1305/3601 度日)
    from s0_common import wquantile
    grid = [float(wquantile(d[var].values/1000, d['NWEIGHT'].values, q)) for q in (0.10, 0.50, 0.90)]
    def prem_at(dd, w, x, v=var):
        b1, b3 = clim_int(dd, w, v)
        m = np.average(dd[v].values/1000, weights=dd[w].values)
        return b1 + b3*(x - m)
    pts = []
    for x in grid:
        t0 = prem_at(d, 'NWEIGHT', x)
        rp = np.array([prem_at(d, r, x) for r in REPS])
        pts.append([x, t0, float(np.sqrt(JK_COEF*np.sum((rp-t0)**2)))])
    f3['climate'][var]['grid'] = pts

print('=== C. 便携电暖器使用 (加权占比) ===')
# v3(P2-12): 码本 NUMPORTEL = "Number of portable electric heaters USED" —— 变量测的是
# 使用数量而非拥有数量; 行名统一为 use, 不得写 own/ownership
W='NWEIGHT'; f3['heaters'] = {}
for grp, mask in [('Insecure', d['insecure']==1), ('Secure', d['insecure']==0)]:
    g = d[mask]
    shares = [float(np.average((g['NUMPORTEL'].clip(lower=0)==0), weights=g[W])*100),
              float(np.average((g['NUMPORTEL']==1), weights=g[W])*100),
              float(np.average((g['NUMPORTEL']>=2), weights=g[W])*100)]
    f3['heaters'][grp] = shares
    print(f'{grp}: 使用0台 {shares[0]:.1f}% / 1台 {shares[1]:.1f}% / 2+台 {shares[2]:.1f}%')
    rows.append([f'{grp}: uses >=1 portable heater %', 100-shares[0], np.nan, np.nan])

print('=== D. 设备存量中介 + 援助交互 ===')
def coef_dev(dd, w):
    Xd = build_X(dd, 3)
    Xd['portel'] = dd['NUMPORTEL'].clip(lower=0).values
    Xd['aux'] = (dd['EQUIPAUX']==1).astype(float).values
    Xd['frig2'] = (dd['NUMFRIG']>=2).astype(float).values
    Xd['oldfrig'] = (dd['AGERFRI1']>=5).astype(float).values
    return wls_beta(Xd, np.log(dd['bps'].values), dd[w].values)[1]
th, se = jackknife(coef_dev, d)
print(f'M3+额外设备控制(非正式中介检验): {th:+.4f} (se={se:.4f}) -> {pct(th):+.1f}%')
rows.append(['Premium after additional equipment controls (not a formal mediation test)', th, se, th/se])
def assist(dd, w):
    Xd = build_X(dd, 2)
    a = (dd['ENERGYASST']==1).astype(float).values
    Xd['asst'] = a; Xd['ix'] = Xd['insecure']*a
    b = wls_beta(Xd, np.log(dd['bps'].values), dd[w].values)
    return b[len(Xd.columns)]
th, se = jackknife(assist, d)
print(f'insecure x 曾获援助: {th:+.4f} (se={se:.4f}, t={th/se:.1f})')
rows.append(['insecure x ever-received-assistance', th, se, th/se])

save_csv(pd.DataFrame(rows, columns=['estimate','coef','se','t']).round(4), RESULT_DIR/'table4b_mechanism.csv', index=False)
json.dump(f3, open(RESULT_DIR/'fig3_mech.json','w'))
print('已存 table4b_mechanism.csv / fig3_mech.json ->', RESULT_DIR)
