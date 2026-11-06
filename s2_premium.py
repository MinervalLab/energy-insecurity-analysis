# -*- coding: utf-8 -*-
"""s2 — 原始缺口 + 条件溢价 (table2 / fig1 数据)
输出: result/table2_premium.csv, result/fig1a_specs.json  |  耗时 3-6 分钟"""
import numpy as np, pandas as pd, json
from s0_common import save_csv
from s0_common import (load_2020, analysis_sample, jackknife, premium_coef, pct, RESULT_DIR)

df = load_2020(); d = analysis_sample(df)
rows = []; specs_json = []

print('=== 原始缺口 (JK se) ===')
def gap_fn(dd, w, col='bps'):
    a, b = dd[dd['insecure']==1], dd[dd['insecure']==0]
    return np.average(a[col], weights=a[w]) - np.average(b[col], weights=b[w])
th, se = jackknife(gap_fn, d)
print(f'全体 kBtu/sqft: {th:.2f} (se={se:.2f}, t={th/se:.1f})')
rows.append(['raw gap kBtu/sqft', th, se, th/se])
for lo, hi, name in [(1,7,'<$25k'), (8,13,'$25-75k'), (14,16,'>$75k')]:
    sub = d[d['MONEYPY'].between(lo,hi)]
    th, se = jackknife(gap_fn, sub)
    rows.append([f'raw gap, income {name}', th, se, th/se])
th, se = jackknife(lambda dd,w: gap_fn(dd,w,'dps'), d)
rows.append(['raw gap $/sqft', th, se, th/se])

print('=== 逐步控制 + 州FE + 口径 (JK se) ===')
LBL = {0:'M0 Unadjusted (log scale)',1:'M1 +Climate/geography',2:'M2 +Income/dwelling/fuel/tenure',
       3:'M3 +Reported housing quality',4:'M4 +Thermostat behaviour'}
for spec in range(5):
    th, se = jackknife(lambda dd,w,s=spec: premium_coef(dd,w,s), d)
    print(f'{LBL[spec]}: {th:+.4f} (se={se:.4f}) -> {pct(th):+.1f}%')
    rows.append([LBL[spec], th, se, th/se]); specs_json.append([LBL[spec], th, se])
th, se = jackknife(lambda dd,w: premium_coef(dd,w,4,use_state=True), d)
print(f'State fixed effects: {th:+.4f} (se={se:.4f}) -> {pct(th):+.1f}%')
rows.append(['State fixed effects', th, se, th/se]); specs_json.append(['State fixed effects', th, se])
# source折算比为外部参数(项22): 主值2.80(EIA site-source比), 敏感性2.5/3.0
for fac in [2.80, 2.50, 3.00]:
    src = fac*df['BTUEL'] + df['TOTALBTU'] - df['BTUEL']
    d_src = analysis_sample(df.assign(bps=src/df['SQFTEST']))
    th, se = jackknife(lambda dd,w: premium_coef(dd,w,2), d_src)
    lab = f'Source-energy accounting (elec x {fac:.2f})'
    print(f'{lab}: {th:+.4f} (se={se:.4f}) -> {pct(th):+.1f}%')
    rows.append([lab, th, se, th/se])
    if fac == 2.80: specs_json.append(['Source-energy accounting', th, se])
d_en = analysis_sample(df, 'bps_en')
th, se = jackknife(lambda dd,w: premium_coef(dd,w,2,ycol='bps_en',sqcol='TOTSQFT_EN'), d_en)
rows.append(['TOTSQFT_EN floor-area basis (own trim)', th, se, th/se])

print('=== v3(P2-03): 共同样本口径对比 —— 在主分析样本上重估, 分离口径效应与样本构成 ===')
# v2 的 source/EN 行各自重新截尾, 两口径样本对称差72户(EN)/162户(source), 相同N≠相同家庭;
# 下面两行固定主分析样本成员不变, 只换因变量口径, 是纯口径效应
th, se = jackknife(lambda dd,w: premium_coef(dd,w,2,ycol='src_bps'),
                   d.assign(src_bps=d['src_btu']/d['SQFTEST']))
rows.append(['Source-energy (elec x 2.80), same households as main sample', th, se, th/se])
print(f'source同样本: {th:+.4f} (se={se:.4f}) -> {pct(th):+.1f}%')
th, se = jackknife(lambda dd,w: premium_coef(dd,w,2,ycol='bps_en',sqcol='TOTSQFT_EN'), d)
rows.append(['TOTSQFT_EN basis, same households as main sample', th, se, th/se])
print(f'EN同样本: {th:+.4f} (se={se:.4f}) -> {pct(th):+.1f}%')

print('=== v3(P2-02): 截尾敏感性 + 阈值透明化 ===')
base = df[(df['bps']>0)&(df['TOTALDOL']>0)]
thr = base['bps'].quantile(0.995)
print(f'未加权99.5%截尾阈值 = {thr:.3f} kBtu/ft^2 (删1户非正支出 + {int((base["bps"]>=thr).sum())}户上端)')
rows.append(['[info] unweighted 99.5% trim threshold (kBtu/sqft)', thr, np.nan, np.nan])
th, se = jackknife(lambda dd,w: premium_coef(dd,w,2), base)           # 不截尾
rows.append(['M2, no intensity trim (sensitivity)', th, se, th/se])
print(f'不截尾M2: {th:+.4f} (se={se:.4f}) -> {pct(th):+.1f}%')
from s0_common import wquantile
thr_w = wquantile(base['bps'].values, base['NWEIGHT'].values, 0.995)  # 加权阈值
dw = base[base['bps'] < thr_w]
th, se = jackknife(lambda dd,w: premium_coef(dd,w,2), dw)
rows.append(['M2, weighted 99.5% trim (sensitivity)', th, se, th/se])
print(f'加权截尾M2: {th:+.4f} (se={se:.4f}) -> {pct(th):+.1f}%')

print('=== v3(P1-04): 跨年图数据导出 (2020 TOTSQFT_EN 同口径序列, 字段完备) ===')
# fig5 只能读本文件的同口径序列, 不得再混用 fig1a_specs.json(SQFTEST口径)
xy = []
for spec in [0,1,2,4]:
    th, se = jackknife(lambda dd,w,s=spec: premium_coef(dd,w,s,ycol='bps_en',sqcol='TOTSQFT_EN'), d_en)
    xy.append({'wave':2020,'area_basis':'TOTSQFT_EN','spec':f'M{spec}','n':int(len(d_en)),
               'coef':float(th),'se':float(se)})
    print(f"2020 EN M{spec}: {th:+.4f} (se={se:.4f}) -> {pct(th):+.1f}%")
json.dump(xy, open(RESULT_DIR/'fig5_crossyear_2020.json','w'))

save_csv(pd.DataFrame(rows, columns=['estimate','coef','se','t']).round(4), RESULT_DIR/'table2_premium.csv', index=False)
json.dump(specs_json, open(RESULT_DIR/'fig1a_specs.json','w'))
print('已存 table2_premium.csv / fig1a_specs.json / fig5_crossyear_2020.json ->', RESULT_DIR)
