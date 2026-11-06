# -*- coding: utf-8 -*-
"""s7 — 中位数封顶核算情景 + 双重负担 + 州级 (fig6 / fig7 数据)
v3 修订要点 (P1-03/05/06/07, P2-09):
  - 措辞重定位: 输出为 median-capping accounting scenario(会计情景), 不是"由回归溢价
    换算的总量", 也不是已识别的可实现节能量; 行名统一用该措辞(P1-05)
  - 安全组诊断(描述性): 同一算法应用于安全家庭自身, 展示指标吸收组内正常
    离散度的幅度; 另报两组"正差占组总消费比"作可比率(P1-05; 非正式安慰剂检验)
  - 系数缩放情景: (1-exp(-beta_M4)) × 不安全组总消费/总支出, 与约6%条件关联自洽的
    假设情景(比例缩减+账单同比假设), 全管线JK(P1-05; v3.1 复查#1 明确非因果性质)
  - 实现率行改称 assumed scaling(外加情景系数, 非校准区间)(P1-06)
  - 州级完整JK: 每个重复权重下重估全部单元中位数后再汇总州值, 与全国同一程序;
    v2 的固定中位数部分JK低估SE(TX 30.9->44.8, CA 25.2->36.8)(P1-07)
  - 双重负担规模附完整JK标准误(v2未输出)(P1-07)
  - 无基准单元: above/db=NaN(s0 v3), 计数均限定"可建立基准的家庭", 未覆盖户
    单独报告规模(P1-03)
  - 10%负担阈值比例改精确加权计数, 插值CDF仅用于画曲线(P2-09)
输出: counterfactual_summary.csv, state_cf.json, state_table_appendix.csv,
      fig2_data.json  |  耗时 ~8-12 分钟(全管线JK×2)"""
import numpy as np, pandas as pd, json
from s0_common import save_csv
from s0_common import (load_2020, analysis_sample, add_cells, flag_above_and_db,
                       wquantile, wshare, premium_coef, REPS, JK_COEF, RESULT_DIR)

df = load_2020()
CELLS = ['inc3','BA_climate','typc','vint3']

def full_pipeline(wcol):
    """一个权重下的完整重算(单元中位数->户级正差->全国+州+db), 供JK整体重估.
    返回 (全国: above户M, TBtu, $B, dbM), 州字典 {st: (per_hh, total_M, ins_hh_M)}"""
    d = add_cells(analysis_sample(df))
    d['save_kbtu'] = 0.0; d['save_dol'] = 0.0
    d['cell_defined'] = False; d['above'] = np.nan
    for _, cell in d.groupby(CELLS, observed=True):
        sec = cell[cell['insecure']==0]
        if len(sec) < 10: continue
        med = wquantile(sec['bps'], sec[wcol], 0.5)
        d.loc[cell.index, 'cell_defined'] = True
        d.loc[cell.index, 'above'] = (cell['bps'] > med).astype(float)
        ia = cell[(cell['insecure']==1) & (cell['bps'] > med)]
        sk = (ia['bps']-med)*ia['SQFTEST']
        d.loc[ia.index, 'save_kbtu'] = sk
        d.loc[ia.index, 'save_dol'] = sk*(ia['TOTALDOL']/ia['TOTALBTU'])
    db1 = d['cell_defined'] & (d['above']==1) & d['SCALEG'].isin([1,2,3])
    d['db'] = np.where(d['cell_defined'], db1.astype(float), np.nan)
    ab = d[(d['insecure']==1) & (d['above']==1)]
    nat = (ab[wcol].sum()/1e6,
           (ab['save_kbtu']*ab[wcol]).sum()/1e9,
           (ab['save_dol']*ab[wcol]).sum()/1e9,
           d.loc[db1, wcol].sum()/1e6)
    st = {}
    for s, g in d.groupby('state_postal'):
        gi = g[g['insecure']==1]; wsum = gi[wcol].sum()
        tot = (gi['save_dol']*gi[wcol]).sum()
        st[s] = (tot/wsum if wsum>0 else np.nan, tot/1e6, wsum/1e6)
    return np.array(nat), st, d

print('=== 全国+州+db: 完整管线 JK(每重复权重重估全部单元中位数) ===')
th, st_main, d_main = full_pipeline('NWEIGHT')
rep_nat = []; rep_st = {s: [] for s in st_main}
for r in REPS:
    natr, str_, _ = full_pipeline(r)
    rep_nat.append(natr)
    for s in st_main: rep_st[s].append(str_[s])
rep_nat = np.array(rep_nat)
ses = np.sqrt(JK_COEF*np.sum((rep_nat-th)**2, axis=0))
for v, se, nm in zip(th, ses, ['基准以上不安全户(M)','情景能量(TBtu/yr)','情景金额($B/yr)','双重负担户(M)']):
    print(f'{nm}: {v:.2f} (se={se:.2f})')

# ---- 覆盖率与未覆盖规模(P1-03) ----
W = 'NWEIGHT'
ins = d_main[d_main['insecure']==1]
coverage = float(np.average(ins['cell_defined'], weights=ins[W])*100)
nobench_M = ins.loc[~ins['cell_defined'], W].sum()/1e6
print(f'不安全户基准覆盖率: {coverage:.2f}% | 无基准不安全户: {nobench_M:.2f}M (不可分类, 不计入以上各数)')

# ---- 安慰剂: 同一算法用于安全家庭(P1-05) ----
sec = d_main[d_main['insecure']==0]
sec_ab = sec[sec['above']==1]
# 安全户正差需重算(save_*只对不安全户存了): 对安全above户逐单元重取中位数
d_p = d_main.copy(); d_p['p_kbtu'] = 0.0; d_p['p_dol'] = 0.0
for _, cell in d_p.groupby(CELLS, observed=True):
    s_ = cell[cell['insecure']==0]
    if len(s_) < 10: continue
    med = wquantile(s_['bps'], s_[W], 0.5)
    sa = cell[(cell['insecure']==0) & (cell['bps'] > med)]
    pk = (sa['bps']-med)*sa['SQFTEST']
    d_p.loc[sa.index, 'p_kbtu'] = pk
    d_p.loc[sa.index, 'p_dol'] = pk*(sa['TOTALDOL']/sa['TOTALBTU'])
pl = d_p[(d_p['insecure']==0) & (d_p['above']==1)]
pl_hh  = pl[W].sum()/1e6
pl_tb  = (pl['p_kbtu']*pl[W]).sum()/1e9
pl_bil = (pl['p_dol']*pl[W]).sum()/1e9
# 可比率: 正差占组总消费比(绝对总量受组规模影响, 不可直接比较)
ins_cons = (ins['TOTALBTU']*ins[W]).sum()/1e9
sec_cons = (sec['TOTALBTU']*sec[W]).sum()/1e9
rate_ins = 100*th[1]/ins_cons; rate_sec = 100*pl_tb/sec_cons
print(f'安全组诊断(同算法对自身中位数; 描述性, 非正式安慰剂检验): {pl_hh:.2f}M 户, {pl_tb:.2f} TBtu, ${pl_bil:.2f}B')
print(f'正差占组总消费比: 不安全 {rate_ins:.1f}% vs 安全 {rate_sec:.1f}% (差 {rate_ins-rate_sec:.1f} 个百分点)')

# ---- 系数缩放情景(P1-05; v3.1复查#1改名): (1-exp(-beta_M4)) × 不安全组总消费/总支出
# 性质声明: 这是"保持面积与其他条件不变、假设每户观测消费按同一比例缩减"的会计情景,
# 不是横截面关联自动识别的可实现节能量; 金额行额外假设账单与消费同比例变化
# (固定费用与燃料结构变化未建模); SE 仅反映该情景统计量的抽样变动 ----
def anchored(wcol):
    da = analysis_sample(df)
    b = premium_coef(da, wcol, spec=4)          # M4 全控制系数(与 table2 同一估计)
    f = 1 - np.exp(-b)
    gi = da[da['insecure']==1]
    return np.array([f*(gi['TOTALBTU']*gi[wcol]).sum()/1e9,
                     f*(gi['TOTALDOL']*gi[wcol]).sum()/1e9])
anc = anchored('NWEIGHT')
anc_r = np.array([anchored(r) for r in REPS])
anc_se = np.sqrt(JK_COEF*np.sum((anc_r-anc)**2, axis=0))
print(f'系数缩放情景(M4): {anc[0]:.1f} TBtu (se={anc_se[0]:.1f}), ${anc[1]:.2f}B (se={anc_se[1]:.2f})')

# ---- 未匹配单元回退敏感性(粗化到 收入×气候) ----
extra = 0.0
und = d_main[(~d_main['cell_defined']) & (d_main['insecure']==1)]
if len(und):
    for _, cell in d_main.groupby(['inc3','BA_climate'], observed=True):
        s_ = cell[cell['insecure']==0]
        tgt = cell.loc[cell.index.isin(und.index)]
        if len(s_) < 10 or len(tgt) == 0: continue
        med = wquantile(s_['bps'], s_[W], 0.5)
        above = tgt[tgt['bps'] > med]
        extra += ((above['bps']-med)*above['SQFTEST']*above[W]).sum()/1e9
print(f'回退敏感性: 无基准户改用粗单元(收入×气候)后额外正差 {extra:.1f} TBtu')

summary = {
 'above-benchmark insecure households (M; among benchmarkable)': [th[0], ses[0]],
 'median-capping scenario: energy (TBtu/yr; accounting quantity, NOT identified savings)': [th[1], ses[1]],
 'median-capping scenario: bills ($B/yr; household-average unit costs)': [th[2], ses[2]],
 'secure-group diagnostic: households above own-cell median (M)': [pl_hh, np.nan],
 'secure-group diagnostic: positive gap (TBtu/yr; descriptive, shows within-group dispersion)': [pl_tb, np.nan],
 'secure-group diagnostic: positive gap ($B/yr)': [pl_bil, np.nan],
 'positive gap as % of group consumption: insecure': [rate_ins, np.nan],
 'positive gap as % of group consumption: secure (diagnostic)': [rate_sec, np.nan],
 'coefficient-scaled scenario (M4 beta): energy (TBtu/yr)': [anc[0], anc_se[0]],
 'coefficient-scaled scenario (M4 beta): bills ($B/yr; assumes proportional bill changes)': [anc[1], anc_se[1]],
 # 外加情景系数(P1-06): 非同一估计量的校准, 不构成可实现节能的置信区间
 'median-cap bills at assumed 51% scaling, Christensen et al. 2023 ($B/yr)': [0.51*th[2], 0.51*ses[2]],
 'median-cap bills at assumed 40% scaling, Fowlie et al. 2018 ($B/yr)': [0.40*th[2], 0.40*ses[2]],
 'coefficient-scaled bills at assumed 51% scaling ($B/yr)': [0.51*anc[1], 0.51*anc_se[1]],
 'coefficient-scaled bills at assumed 40% scaling ($B/yr)': [0.40*anc[1], 0.40*anc_se[1]],
 'identified double-burden households (M; among benchmarkable)': [th[3], ses[3]],
 'insecure households without benchmark (M; unclassifiable)': [nobench_M, np.nan],
 'cell coverage of insecure households (%)': [coverage, np.nan],
 'fallback (coarse-cell) additional TBtu': [extra, np.nan]}
save_csv(pd.DataFrame(summary, index=['value','jk_se']).T.round(2), RESULT_DIR/'counterfactual_summary.csv')

# ---- 州级(完整JK) ----
res = {}
for s, (per, tot, hh) in st_main.items():
    rv = np.array([r[0] for r in rep_st[s]], dtype=float)
    rt = np.array([r[1] for r in rep_st[s]], dtype=float)
    # v3.1(复查#7): 重复估计必须全部有限; 无效值应显式失败而非被nansum静默跳过
    assert np.all(np.isfinite(rv)) and np.all(np.isfinite(rt)), f'{s}: 州级重复估计含非有限值'
    se_per = float(np.sqrt(JK_COEF*np.sum((rv-per)**2)))
    se_tot = float(np.sqrt(JK_COEF*np.sum((rt-tot)**2)))
    g = d_main[d_main['state_postal']==s]
    res[s] = {'save_per_ins_hh': float(per) if np.isfinite(per) else None,
              'save_per_ins_hh_se_fullJK': se_per,
              'total_save_M': float(tot), 'total_save_M_se_fullJK': se_tot,
              'ins_hh_M': float(hh),
              'ins_rate': wshare(g, g['insecure']==1),
              'n_ins': int(len(g[g['insecure']==1]))}
json.dump(res, open(RESULT_DIR/'state_cf.json','w'), indent=0)
save_csv(pd.DataFrame(res).T.round(2), RESULT_DIR/'state_table_appendix.csv')

# ---- fig6(双重负担图) 数据 ----
d = d_main
out = {'funnel': {
    # v3: 首级改为分析样本加权总量(122.9M); 调查全域为123.5M(此前口径混用)
    'all': float(d[W].sum()/1e6),
    'insecure': float(d.loc[d['insecure']==1, W].sum()/1e6),
    'above': float(d.loc[(d['insecure']==1)&(d['above']==1), W].sum()/1e6),
    'db': float(d.loc[d['db']==1, W].sum()/1e6),
    'insecure_no_benchmark': float(nobench_M)}}
chars = [('Income below $25k', lambda g: g['MONEYPY']<=7),
 ('Renter', lambda g: g['KOWNRENT']==2),
 ('Poorly/not insulated (reported)', lambda g: g['ADQINSUL']>=3),
 ('Built before 1970', lambda g: g['YEARMADERANGE']<=3),
 ('Children in household', lambda g: g['NUMCHILD']>=1),
 ('Black householder', lambda g: g['HOUSEHOLDER_RACE']==2),
 ('Hispanic or Latino', lambda g: g['SDESCENT']==1),
 ('Ever received energy assistance', lambda g: g['ENERGYASST']==1),
 ('Householder 65+', lambda g: g['HHAGE']>=65),
 ('Electric main heat', lambda g: g['FUELHEAT']==5)]
db_g = d[d['db']==1]
out['chars'] = [{'label': l, 'db': wshare(db_g, f(db_g)), 'nat': wshare(d, f(d))} for l, f in chars]
d['typ4'] = d['TYPEHUQ'].map({1:'mobile',2:'sf',3:'sf',4:'apt24',5:'apt5'})
out['heat'] = {}
for t in ['mobile','sf','apt24','apt5']:
    for v in ['pre70','70-99','00+']:
        g = d[(d['typ4']==t)&(d['vint3']==v)]
        gc = g[g['cell_defined']]          # 只有基准可定义的户才能判定db
        key = f'{t}|{v}'
        cov = wshare(g, g['cell_defined']) if len(g)>0 else 0.0
        if len(gc) == 0:
            # v3(P1-03): 无任何可分类户 -> NA(不可估计), 不再显示0%
            out['heat'][key] = {'share': None, 'n_total': int(len(g)), 'n_covered': 0,
                                'coverage_pct': cov}
        else:
            # v3.1(复查#5): share 分母 = 可建立基准的家庭; n_covered 供小样本标记使用
            out['heat'][key] = {'share': wshare(gc, gc['db']==1), 'n_total': int(len(g)),
                                'n_covered': int(len(gc)), 'coverage_pct': cov}
d['burden'] = d['TOTALDOL']/(d['income_k']*1000)*100
xs = np.arange(0, 25.01, 0.25); cdf = {}
for name, m in [('Double burden', d['db']==1),
                ('Other insecure', (d['insecure']==1)&(d['db']!=1)),
                ('Secure', d['insecure']==0)]:
    g = d[m].sort_values('burden')
    b = g['burden'].values; w = g[W].values
    c = np.clip(np.cumsum(w)/w.sum()*100, 0, 100)
    cdf[name] = {'x': xs.tolist(),
                 'y': [float(np.clip(np.interp(x, b, c, left=0.0, right=100.0),0,100)) for x in xs],
                 'median': float(wquantile(b, w, 0.5)),
                 # v3(P2-09): 阈值比例用精确加权计数, 插值CDF仅供画曲线
                 'share10': float(np.average((b > 10).astype(float), weights=w)*100)}
out['cdf'] = cdf
out['notes'] = {'Other insecure': 'insecure & not identified double burden; includes '
                'unclassifiable insecure households (no benchmark cell)',
                'funnel.all': 'analysis sample (122.9M); survey universe 123.5M in caption'}
json.dump(out, open(RESULT_DIR/'fig2_data.json', 'w'))
print('已存 counterfactual_summary.csv / state_cf.json / state_table_appendix.csv / fig2_data.json ->', RESULT_DIR)
