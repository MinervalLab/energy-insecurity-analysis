# -*- coding: utf-8 -*-
"""s9 — 补充稳健性分析 (模块 A-E)
模块A: 面积函数形式 + 面积分层
设计(检验 premium 是否为"面积分母"的机械产物):
  (0) 恒等式演示: 以 ln(总能耗 TOTALBTU) 为因变量、同样控制 ln A 的 M2 回归,
      其不安全系数与主模型(ln 强度)完全相等(程序内断言, 差<1e-9) ——
      premium = 同等面积下总能耗的条件比例差, 非"小面积除出的高比率"
  (1) 面积函数形式: M2 的 ln A 线性项依次放开为 二次/三次多项式(中心化)、
      加权五分位/十分位虚拟变量, 报告各设定下的 premium (JK se)
  (2) 面积分层: 按加权四分位切 SQFTEST, 层内分别估计 M2 premium ——
      层内面积差异已很小, 若"规模伪象"成立, 层内 premium 应消失
各模块仅新增输出文件, 不改动其他脚本的输出
输出: table7_area_robustness.csv  |  耗时 ~3-5 分钟
"""
import numpy as np, pandas as pd
from s0_common import save_csv
from s0_common import (load_2020, analysis_sample, build_X, wls_beta, wquantile,
                       premium_coef, pct, REPS, JK_COEF, RESULT_DIR)

df = load_2020(); d = analysis_sample(df)
W = 'NWEIGHT'
rows = []

# ---- (0) 恒等式演示: ln(E) 与 ln(E/A) 在含 ln A 控制下 β 恒等 ----
th_int = premium_coef(d, W, 2)                       # 主模型: ln(bps), M2
y_tot = np.log(d['TOTALBTU'].values)                 # ln(总能耗)
Xd = build_X(d, 2)
th_lnE = wls_beta(Xd, y_tot, d[W].values, check=False)[1]
assert abs(th_int - th_lnE) < 1e-9, f'恒等式失败: {th_int} vs {th_lnE}'
print(f'恒等式: ln(E/A)模型 β={th_int:+.6f} ≡ ln(E)模型 β={th_lnE:+.6f} '
      f'(差 {abs(th_int-th_lnE):.1e}) —— premium=同面积下总能耗条件差')
rows.append(['M2 baseline: ln intensity, linear ln(A) control', th_int, np.nan, np.nan,
             len(d), pct(th_int)])
rows.append(['M2 identity check: ln total energy, linear ln(A) control', th_lnE,
             np.nan, np.nan, len(d), pct(th_lnE)])

# ---- (1) 面积函数形式 ----
lm = float(np.average(np.log(d['SQFTEST'].values), weights=d[W].values))  # 中心化点
qs5 = [wquantile(d['SQFTEST'].values, d[W].values, q) for q in (0.2,0.4,0.6,0.8)]
qs10 = [wquantile(d['SQFTEST'].values, d[W].values, q/10) for q in range(1,10)]
d['_ab5'] = np.digitize(d['SQFTEST'], qs5)      # 0..4
d['_ab10'] = np.digitize(d['SQFTEST'], qs10)    # 0..9

def coef_form(dd, w, form):
    Xd = build_X(dd, 2)
    la = np.log(dd['SQFTEST'].values) - lm
    if form in ('quad','cubic'):
        Xd['log_sqft'] = la          # 重参数化(中心化), 不改 β_insecure
        Xd['la2'] = la**2
        if form == 'cubic': Xd['la3'] = la**3
    elif form in ('bins5','bins10'):
        Xd = Xd.drop(columns=['log_sqft'])
        col, nb = ('_ab5',5) if form=='bins5' else ('_ab10',10)
        for b in range(1, nb):
            Xd[f'ab_{b}'] = (dd[col]==b).astype(float).values
    return wls_beta(Xd, np.log(dd['bps'].values), dd[w].values, check=False)[1]

FORMS = [('quad',  'M2, quadratic ln(A) control'),
         ('cubic', 'M2, cubic ln(A) control'),
         ('bins5', 'M2, floor-area quintile indicators'),
         ('bins10','M2, floor-area decile indicators')]
print('=== 面积函数形式 (M2, 因变量 ln 强度) ===')
for form, nm in FORMS:
    th = coef_form(d, W, form)
    rv = np.array([coef_form(d, r, form) for r in REPS])
    se = float(np.sqrt(JK_COEF*np.sum((rv-th)**2)))
    rows.append([nm, th, se, th/se, len(d), pct(th)])
    print(f'  {nm:42s}: {th:+.4f} (se={se:.4f}, t={th/se:.1f}) -> {pct(th):+.1f}%')

# ---- (2) 面积四分位分层 ----
qc = [wquantile(d['SQFTEST'].values, d[W].values, q) for q in (0.25,0.5,0.75)]
d['_aq'] = np.digitize(d['SQFTEST'], qc)        # 0..3
def sub_coef(data, wcol):
    # 子样本回归(同 s4 口径): 按有效权重行剔除零方差列, insecure 恒保留
    Xd = build_X(data, 2); w = data[wcol].values
    m = w > 0
    keep = [c for c in Xd.columns if Xd[c].values[m].std() > 0 or c=='insecure']
    return wls_beta(Xd[keep], np.log(data['bps'].values), w, check=False)[1]
print('=== 面积四分位分层 (M2, 层内估计) ===')
QL = ['Q1 (smallest)','Q2','Q3','Q4 (largest)']
for q in range(4):
    sub = d[d['_aq']==q]
    med = wquantile(sub['SQFTEST'].values, sub[W].values, 0.5)
    th = sub_coef(sub, W)
    rv = np.array([sub_coef(sub, r) for r in REPS])
    se = float(np.sqrt(JK_COEF*np.sum((rv-th)**2)))
    nm = f'M2, floor-area quartile {QL[q]}, median {med:,.0f} sq ft'
    rows.append([nm, th, se, th/se, len(sub), pct(th)])
    print(f'  {nm:58s}: n={len(sub):5d}, {th:+.4f} (se={se:.4f}, t={th/se:.1f}) -> {pct(th):+.1f}%')

save_csv(pd.DataFrame(rows, columns=['estimate','coef','se','t','n','premium_pct'])
         .round(4), RESULT_DIR/'table7_area_robustness.csv', index=False)
print('已存 table7_area_robustness.csv ->', RESULT_DIR)

# ======================================================================
# 模块B: 中位数封顶核算情景的基准敏感性
# 设计: 主口径 = p50 封顶 / 单元 收入×气候×房型×年代 / 最少同伴 10 (与 s7 相同算法).
# 逐一变动三个维度: 封顶分位 p50/p60/p75; 单元定义 粗化(收入×气候)/现行/
# 细化(+租购二分); 最少安全同伴数 5/10/20. 报告全国点量级(户数/TBtu/$B/覆盖率).
# 推断口径: 主口径的完整管线JK在 s7; 本敏感性网格报告点估计(量级比较用).
# 对既有结果影响: 零(仅新增 table8 输出; 模块A的 table7 不变)
# ======================================================================
from s0_common import add_cells
print('=== 模块B: 基准敏感性 (点估计网格; 主口径JK见s7) ===')
CELLS0 = ['inc3','BA_climate','typc','vint3']
db = add_cells(analysis_sample(df))
db['ten2'] = np.where(db['KOWNRENT']==2, 'rent', 'own')   # 细化: +租购二分

def bench(q, cells, min_n):
    hh = tb = bil = 0.0; cov_w = 0.0
    w = db['NWEIGHT']
    for _, cell in db.groupby(cells, observed=True):
        sec_ = cell[cell['insecure']==0]
        if len(sec_) < min_n: continue
        med = wquantile(sec_['bps'], sec_['NWEIGHT'], q)
        cov_w += cell.loc[cell['insecure']==1, 'NWEIGHT'].sum()
        ia = cell[(cell['insecure']==1) & (cell['bps'] > med)]
        sk = (ia['bps']-med)*ia['SQFTEST']
        hh  += ia['NWEIGHT'].sum()/1e6
        tb  += (sk*ia['NWEIGHT']).sum()/1e9
        bil += (sk*(ia['TOTALDOL']/ia['TOTALBTU'])*ia['NWEIGHT']).sum()/1e9
    ins_w = db.loc[db['insecure']==1, 'NWEIGHT'].sum()
    return hh, tb, bil, 100*cov_w/ins_w

GRID = [('Primary: p50 cap, income x climate x type x vintage, min 10 peers', 0.50, CELLS0, 10),
        ('Cap at p60 of secure peers',                                        0.60, CELLS0, 10),
        ('Cap at p75 of secure peers',                                        0.75, CELLS0, 10),
        ('Coarser cells: income x climate',                                   0.50, ['inc3','BA_climate'], 10),
        ('Finer cells: + owner/renter',                                       0.50, CELLS0+['ten2'], 10),
        ('Minimum 5 secure peers per cell',                                   0.50, CELLS0, 5),
        ('Minimum 20 secure peers per cell',                                  0.50, CELLS0, 20)]
rows_b = []
for nm, q, cells, mn in GRID:
    hh, tb, bil, cov = bench(q, cells, mn)
    rows_b.append([nm, hh, tb, bil, cov])
    print(f'  {nm:62s}: {hh:5.2f}M 户, {tb:6.1f} TBtu, ${bil:5.2f}B, 覆盖 {cov:.1f}%')

# 交叉校验: 主口径应与 s7 的 counterfactual_summary.csv 点估计一致(若该文件存在)
try:
    cs = pd.read_csv(RESULT_DIR/'counterfactual_summary.csv', index_col=0)
    ref = [float(cs.loc['above-benchmark insecure households (M; among benchmarkable)','value']),
           float(cs.loc["median-capping scenario: energy (TBtu/yr; accounting quantity, NOT identified savings)",'value']),
           float(cs.loc['median-capping scenario: bills ($B/yr; household-average unit costs)','value'])]
    dif = [abs(rows_b[0][1]-ref[0]), abs(rows_b[0][2]-ref[1]), abs(rows_b[0][3]-ref[2])]
    assert max(dif) < 0.01, f'主口径与 s7 不一致: {dif}'
    print(f'  交叉校验: 主口径与 s7 counterfactual_summary 一致 (18.24M/500.98/12.13)')
except FileNotFoundError:
    print('  (未找到 counterfactual_summary.csv, 跳过交叉校验)')

save_csv(pd.DataFrame(rows_b, columns=['scenario','above_hh_M','energy_TBtu',
         'bills_B','coverage_pct_of_insecure']).round(2),
         RESULT_DIR/'table8_benchmark_sensitivity.csv', index=False)
print('已存 table8_benchmark_sensitivity.csv ->', RESULT_DIR)

# ======================================================================
# 模块C: 有效同伴样本量 (主口径单元, p50/min 10)
# 报告: 有效单元数; 每单元安全同伴数分布(中位/最大; 最小=10为构造约束);
#       不安全户视角的加权分布(典型不安全户面对多少安全同伴; >=20/>=50 占比)
# 对既有结果影响: 零(仅新增 table8b 输出)
# ======================================================================
print('=== 模块C: 有效同伴样本量 ===')
stats = []
for _, cell in db.groupby(CELLS0, observed=True):
    ns = int((cell['insecure']==0).sum())
    if ns < 10: continue
    stats.append((ns, cell.loc[cell['insecure']==1, 'NWEIGHT'].sum()))
ns_arr = np.array([s_[0] for s_ in stats], dtype=float)
wins   = np.array([s_[1] for s_ in stats], dtype=float)
order  = np.argsort(ns_arr)
cw     = np.cumsum(wins[order])
wmed   = ns_arr[order][np.searchsorted(cw, 0.5*cw[-1], side='left')]
rows_c = [
 ['defined benchmark cells (>=10 secure peers)', len(ns_arr)],
 ['median secure peers per defined cell', float(np.median(ns_arr))],
 ['min secure peers per defined cell (by construction >=10)', float(ns_arr.min())],
 ['max secure peers per defined cell', float(ns_arr.max())],
 ['weighted median peers faced by an insecure household', float(wmed)],
 ['share of insecure households in cells with >=20 peers (%)', float(100*wins[ns_arr>=20].sum()/wins.sum())],
 ['share of insecure households in cells with >=50 peers (%)', float(100*wins[ns_arr>=50].sum()/wins.sum())]]
for nm, v in rows_c: print(f'  {nm:58s}: {v:,.1f}' if isinstance(v,float) else f'  {nm:58s}: {v}')
save_csv(pd.DataFrame(rows_c, columns=['statistic','value']).round(1),
         RESULT_DIR/'table8b_peer_counts.csv', index=False)
print('已存 table8b_peer_counts.csv ->', RESULT_DIR)

# ======================================================================
# 模块D: 上尾截尾敏感性补充档 —— 非加权 99% 截尾
# 主口径 = 非加权 99.5% 截尾(analysis_sample); table2 已有 不截尾(+6.2%) 与
# 加权 99.5%(+6.1%) 两档; 此处补收紧一档(99%), 同 M2 控制集与 JK 程序
# 对既有结果影响: 零(仅新增 table9 输出)
# ======================================================================
print('=== 模块D: 非加权 99% 截尾敏感性 ===')
d_pos = df[(df['bps'] > 0) & (df['TOTALDOL'] > 0)]
thr99 = d_pos['bps'].quantile(0.99)
d99 = d_pos[d_pos['bps'] < thr99].copy()
th99 = premium_coef(d99, 'NWEIGHT', 2)
rv99 = np.array([premium_coef(d99, r, 2) for r in REPS])
se99 = float(np.sqrt(JK_COEF*np.sum((rv99-th99)**2)))
print(f'  M2, 非加权99%截尾 (阈值 {thr99:.2f} kBtu/ft^2, n={len(d99)}): '
      f'{th99:+.4f} (se={se99:.4f}, t={th99/se99:.1f}) -> {pct(th99):+.1f}%')
save_csv(pd.DataFrame([[f'M2, unweighted 99% trim (threshold {thr99:.4f} kBtu/ft2)',
         th99, se99, th99/se99, len(d99), pct(th99)]],
         columns=['estimate','coef','se','t','n','premium_pct']).round(4),
         RESULT_DIR/'table9_trim99.csv', index=False)
print('已存 table9_trim99.csv ->', RESULT_DIR)

# ======================================================================
# 模块E: 三项定义 vs EIA 宽口径(含设备/燃料故障)
# 三项 = SCALEB/SCALEG/SCALEE 任一∈{1,2,3} (s0 的 insecure, 主定义)
# 宽口径 = 三项 或 过去一年因下列原因无法使用采暖/制冷设备(任一=1):
#   NOHEATBROKE(设备损坏且无钱修) / NOHEATEL / NOHEATNG / NOHEATBULK(付不起
#   电/天然气/散装燃料) / NOACBROKE / NOACEL —— 与 EIA 2020 宽口径一致
# 输出: 人口对账(加权户数/比例/仅故障户) + 宽口径 M2 premium(JK, 参照=宽口径安全)
# 对既有结果影响: 零(本地定义, 不改 s0/s4; 仅新增 table10 输出)
# ======================================================================
print('=== 模块E: 宽口径定义对账 ===')
EQUIP = ['NOHEATBROKE','NOHEATEL','NOHEATNG','NOHEATBULK','NOACBROKE','NOACEL']
d_e = analysis_sample(df)
equip_any = (df.loc[d_e.index, EQUIP] == 1).any(axis=1)
d_e['insecure_broad'] = ((d_e['insecure']==1) | equip_any).astype(int)
tot_e = d_e['NWEIGHT'].sum()/1e6
m3  = d_e.loc[d_e['insecure']==1, 'NWEIGHT'].sum()/1e6
mb  = d_e.loc[d_e['insecure_broad']==1, 'NWEIGHT'].sum()/1e6
mon = d_e.loc[(d_e['insecure']==0)&(d_e['insecure_broad']==1), 'NWEIGHT'].sum()/1e6
assert abs(m3 + mon - mb) < 1e-9, '宽口径人口对账失败'
th_b = premium_coef(d_e, 'NWEIGHT', 2, ins_col='insecure_broad')
rv_b = np.array([premium_coef(d_e, r, 2, ins_col='insecure_broad') for r in REPS])
se_b = float(np.sqrt(JK_COEF*np.sum((rv_b-th_b)**2)))
rows_e = [
 ['three-item insecure (M; % of analysis sample)', m3, 100*m3/tot_e],
 ['broad-definition insecure (M; %)', mb, 100*mb/tot_e],
 ['equipment/fuel-failure only (M; %; = broad minus three-item)', mon, 100*mon/tot_e],
 ['M2 premium, broad definition (coef; %)', th_b, pct(th_b)],
 ['M2 premium, broad definition (se; t)', se_b, th_b/se_b]]
# 全样本(未截尾, 调查全域123.5M)对账: 与 EIA 公布数比对
# (EIA Today in Energy #51979: 34M/27%; 无法采暖~5M; 无法制冷~6M)
Wf = df['NWEIGHT']
broad_full = (df['insecure']==1) | (df[EQUIP]==1).any(axis=1)
noheat_f = (df[EQUIP[:4]]==1).any(axis=1)
noac_f   = (df[EQUIP[4:]]==1).any(axis=1)
rows_e += [
 ['full-sample broad insecure (M; %; EIA publishes 34M/27%)',
  Wf[broad_full].sum()/1e6, 100*Wf[broad_full].sum()/Wf.sum()],
 ['full-sample unable to use heating, any NOHEAT* (M; EIA ~5M)',
  Wf[noheat_f].sum()/1e6, np.nan],
 ['full-sample unable to use AC, any NOAC* (M; EIA ~6M)',
  Wf[noac_f].sum()/1e6, np.nan]]
print(f'  全样本对账: 宽口径 {Wf[broad_full].sum()/1e6:.2f}M ({100*Wf[broad_full].sum()/Wf.sum():.1f}%) vs EIA 34M/27%; '
      f'无法采暖 {Wf[noheat_f].sum()/1e6:.2f}M vs ~5M; 无法制冷 {Wf[noac_f].sum()/1e6:.2f}M vs ~6M')
print(f'  三项口径: {m3:.2f}M ({100*m3/tot_e:.1f}%) | 宽口径: {mb:.2f}M ({100*mb/tot_e:.1f}%) | '
      f'仅设备/燃料故障: {mon:.2f}M ({100*mon/tot_e:.1f}%)')
print(f'  宽口径 M2 premium: {th_b:+.4f} (se={se_b:.4f}, t={th_b/se_b:.1f}) -> {pct(th_b):+.1f}% '
      f'(三项口径基线 +6.1%)')
save_csv(pd.DataFrame(rows_e, columns=['item','value_1','value_2']).round(4),
         RESULT_DIR/'table10_broad_definition.csv', index=False)
print('已存 table10_broad_definition.csv ->', RESULT_DIR)
