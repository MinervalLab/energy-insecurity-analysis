# -*- coding: utf-8 -*-
"""s6 — 疫情稳健性: 2015 复制 (核心集合 + 补充) (table5 / fig5 数据)
输出: result/table5_2015.csv, result/fig3_2015.json  |  耗时 2-4 分钟"""
import numpy as np, pandas as pd, json
from s0_common import save_csv
from s0_common import (load_2015, build_X15, wls_beta, brr_se, pct, RESULT_DIR)

df = load_2015()
d = df[(df['bps']>0)&(df['TOTALDOL']>0)]; d = d[d['bps']<d['bps'].quantile(0.995)].copy()
print(f'2015 n={len(d)}, 加权 {d["NWEIGHT"].sum()/1e6:.1f}M, any不安全率 '
      f'{np.average(df["insecure"],weights=df["NWEIGHT"])*100:.1f}%')
rows = []; f3d = []; xy15 = []   # xy15: v3(P1-04) 跨年图同口径序列(与s2的2020导出字段一致)

def coef(dd, w, spec=2, ycol='bps', ins='insecure'):
    # 子样本内退化列剔除: 如燃气采暖子样本中 fuel_1≡1(与常数重复)、其余燃料列≡0,
    # 6列共线致 rank=25<31; insecure 不在共线span内, 剔除不改变其估计(见验证输出)
    Xd = build_X15(dd, spec, ins); w_ = dd[w].values
    m = w_ > 0
    keep = [c for c in Xd.columns if Xd[c].values[m].std() > 0 or c=='insecure']
    return wls_beta(Xd[keep], np.log(dd[ycol].values), w_)[1]

print('=== 核心集合: 溢价 + 维度结构 (BRR se) ===')
# v3(P1-04): 'M2 Full controls' 是误称(M2无质量/行为块), 改为与2020同名的层级标签
for spec, name in [(0,'M0 Unadjusted'),(1,'M1 +Climate/geography'),
                   (2,'M2 +Income/dwelling/fuel/tenure')]:
    th, se = brr_se(lambda dd,w,s=spec: coef(dd,w,s), d)
    print(f'2015 {name}: {th:+.4f} (se={se:.4f}, t={th/se:.1f}) -> {pct(th):+.1f}%')
    rows.append([f'2015 {name}', th, se, th/se]); f3d.append([name, th, se])
    xy15.append({'wave':2015,'area_basis':'TOTSQFT_EN','spec':f'M{spec}','n':int(len(d)),
                 'coef':float(th),'se':float(se)})
# v3(P1-01/02): 与2020 s4同构的参照组修正 —— 全部以完全安全为参照
th, se = brr_se(lambda dd,w: coef(dd,w,2,ins='severe'),
                d[(d['insecure']==0)|(d['severe']==1)])
print(f'2015 Severe vs secure: {th:+.4f} (se={se:.4f}, t={th/se:.1f})')
rows.append(['2015 Severe insecurity vs secure', th, se, th/se])
th, se = brr_se(lambda dd,w: coef(dd,w,2,ins='dim_temp_any'),
                d[(d['insecure']==0)|(d['dim_temp_any']==1)])
print(f'2015 Any temperature difficulty vs secure: {th:+.4f} (se={se:.4f}, t={th/se:.1f})')
rows.append(['2015 Any temperature difficulty vs secure', th, se, th/se])
def excl15(dd, w):
    Xd = build_X15(dd, 2, 'grp_fin_only')
    Xd['grp_temp_only'] = dd['grp_temp_only'].astype(float).values
    Xd['grp_both'] = dd['grp_both'].astype(float).values
    # 注意: build_X15 首列固定命名'insecure'(承载ins_col传入的指示, 此处即grp_fin_only)
    w_ = dd[w].values; m = w_ > 0
    keep = [c for c in Xd.columns if Xd[c].values[m].std() > 0
            or c in ('insecure','grp_temp_only','grp_both')]
    b = wls_beta(Xd[keep], np.log(dd['bps'].values), w_, check=False)
    k = list(Xd[keep].columns)
    return np.array([b[1], b[1+k.index('grp_temp_only')], b[1+k.index('grp_both')]])
th0 = excl15(d,'NWEIGHT'); r_ = np.array([excl15(d,w) for w in
      [f'BRRWT{i}' for i in range(1,97)]])
for i, nm in enumerate(['2015 Financial difficulty only vs secure (mutually exclusive)',
                        '2015 Temperature difficulty only vs secure (mutually exclusive)',
                        '2015 Financial and temperature difficulty vs secure (mutually exclusive)']):
    se = np.sqrt((1/24)*np.sum((r_[:,i]-th0[i])**2))
    print(f'{nm}: {th0[i]:+.4f} (se={se:.4f})')
    rows.append([nm, th0[i], se, th0[i]/se])

print('=== 补充: M3/M4 复制 (项25: 2015 具备质量与行为变量) ===')
for spec, name in [(3,'2015 M3 +housing quality'),(4,'2015 M4 +thermostat behaviour')]:
    th, se = brr_se(lambda dd,w,s_=spec: coef(dd,w,s_), d)
    print(f'{name}: {th:+.4f} (se={se:.4f}, t={th/se:.1f})')
    rows.append([name, th, se, th/se])
    if spec == 4:
        xy15.append({'wave':2015,'area_basis':'TOTSQFT_EN','spec':'M4','n':int(len(d)),
                     'coef':float(th),'se':float(se)})

print('=== 跨期同口径对比说明 (项23) ===')
print('table5 的 2020 对照行应使用 TOTSQFT_EN 面积口径 (s2 输出中的 TOTSQFT_EN 行),')
print('与 2015 的 TOTSQFT_EN 口径构成同口径比较')
rows.append(['NOTE: compare vs 2020 TOTSQFT_EN-basis estimate from table2 (same floor-area basis)', np.nan, np.nan, np.nan])

print('=== 补充: 机制方向核对 ===')
# v3.1(复查#2): 全体电力行此前沿用总能耗截尾样本 d(与结果变量口径不符, 对称差44户),
# 改为按电力强度自身口径截尾(与2020 s5同构: 阈值在全体正结果人群定义)
from s0_common import analysis_sample as _asamp
d_el = _asamp(df, ycol='bps_el')
th, se = brr_se(lambda dd,w: coef(dd,w,2,ycol='bps_el'), d_el)
print(f'2015 电力强度(电力口径截尾, n={len(d_el)}): {th:+.4f} (se={se:.4f}) -> {pct(th):+.1f}%')
rows.append(['2015 electricity intensity premium (appendix; electricity-basis trim)', th, se, th/se])
# v3(P2-06): 截尾规则与2020 s5统一 —— 阈值在全体正结果人群上定义一次, 子样本沿用,
# 不再在子样本内部取0.995分位(v2的2015燃料模型与2020方法不一致)
full_el = df.assign(y=df['BTUEL']/df['TOTSQFT_EN'])
thr_el = full_el[(full_el['y']>0)&(full_el['TOTALDOL']>0)]['y'].quantile(0.995)
full_ng = df.assign(y=df['BTUNG']/df['TOTSQFT_EN'])
thr_ng = full_ng[(full_ng['y']>0)&(full_ng['TOTALDOL']>0)]['y'].quantile(0.995)
g2 = df[df['FUELHEAT']==1].assign(y=lambda x: x['BTUEL']/x['TOTSQFT_EN'])
d2 = g2[(g2['y']>0)&(g2['TOTALDOL']>0)]; d2 = d2[d2['y']<thr_el].copy()
th, se = brr_se(lambda dd,w: coef(dd,w,2,ycol='y'), d2)
rows.append(['2015 electricity premium, gas-main-heat homes (appendix; global trim)', th, se, th/se])
g3 = df[df['BTUNG']>0].assign(y=lambda x: x['BTUNG']/x['TOTSQFT_EN'])
d3 = g3[(g3['y']>0)&(g3['TOTALDOL']>0)]; d3 = d3[d3['y']<thr_ng].copy()
th, se = brr_se(lambda dd,w: coef(dd,w,2,ycol='y'), d3)
rows.append(['2015 gas intensity premium, gas users (appendix; global trim; NOTE: significant in 2015)', th, se, th/se])

save_csv(pd.DataFrame(rows, columns=['estimate','coef','se','t']).round(4), RESULT_DIR/'table5_2015.csv', index=False)
json.dump(f3d, open(RESULT_DIR/'fig3_2015.json','w'))
json.dump(xy15, open(RESULT_DIR/'fig5_crossyear_2015.json','w'))
print('已存 table5_2015.csv / fig3_2015.json / fig5_crossyear_2015.json ->', RESULT_DIR)
