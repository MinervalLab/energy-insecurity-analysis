# -*- coding: utf-8 -*-
"""verify — 复现校验: 对照 expected_results.json 检查关键结果(带容差)
用法: 全部脚本跑完后运行 python verify.py; 预期值与分析代码完全隔离(项1/21)"""
import json, sys
import pandas as pd
from s0_common import RESULT_DIR
from pathlib import Path

EXP = json.load(open(Path(__file__).parent/'expected_results.json'))
fails = []
def check(name, got, want, tol):
    ok = abs(got-want) <= tol
    print(f'{"PASS" if ok else "FAIL"}  {name}: got {got:.4f}, expected {want:.4f} (±{tol})')
    if not ok: fails.append(name)

t2 = pd.read_csv(RESULT_DIR/'table2_premium.csv').set_index('estimate')
for k, (want, tol) in EXP['table2'].items():
    check(f'table2/{k}', float(t2.loc[k,'coef']), want, tol)
t3 = pd.read_csv(RESULT_DIR/'table3_oaxaca.csv').set_index('component')
for k, (want, tol) in EXP['table3'].items():
    check(f'table3/{k}', float(t3.loc[k,'coef']), want, tol)
t4 = pd.read_csv(RESULT_DIR/'table4_heterogeneity.csv').set_index('estimate')
for k, (want, tol) in EXP['table4'].items():
    check(f'table4/{k}', float(t4.loc[k,'coef']), want, tol)
t5 = pd.read_csv(RESULT_DIR/'table5_2015.csv').set_index('estimate')
for k, (want, tol) in EXP['table5'].items():
    check(f'table5/{k}', float(t5.loc[k,'coef']), want, tol)
cf = pd.read_csv(RESULT_DIR/'counterfactual_summary.csv', index_col=0)
for k, (want, tol) in EXP['counterfactual'].items():
    check(f'cf/{k}', float(cf.loc[k,'value']), want, tol)

# ---------------- v3(P3-01): 语义检查 —— 数值复现≠科学正确, 以下检查估计量的构造本身 ----------------
import json as _json, numpy as _np
def sem(name, ok):
    print(f'{"PASS" if ok else "FAIL"}  [语义] {name}')
    if not ok: fails.append('[语义] '+name)

# 1) 分解恒等: explained+unexplained=gap; 份额SE对称(和恒为100%)
# CSV为4位舍入值, 两项相加的最大舍入残差为1.5e-4; 全精度恒等由s3内部assert保证(1e-15级)
sem('分解: explained+unexplained==gap (4位舍入容差1.5e-4)',
    abs(float(t3.loc['explained','coef'])+float(t3.loc['unexplained','coef'])-float(t3.loc['gap','coef']))<1.5e-4)
sem('分解: explained/unexplained 份额SE相等(比率JK性质)',
    abs(float(t3.loc['explained','pct_of_gap_se'])-float(t3.loc['unexplained','pct_of_gap_se']))<0.01)
# 2) threefold E+C+I=gap (table3b 自包含行)
t3b = pd.read_csv(RESULT_DIR/'table3b_altrefs.csv')
r3 = t3b[t3b['reference'].str.startswith('threefold')].iloc[0]
sem('threefold: E+C+I==gap (1e-3, 4位舍入)',
    abs(r3['explained_or_E']+r3['unexplained_or_C']+r3['interaction_I']-r3['gap_or_sum'])<1e-3)
# 3) 跨年图数据: 两年同spec集合且均为TOTSQFT_EN口径
xy20 = _json.load(open(RESULT_DIR/'fig5_crossyear_2020.json'))
xy15 = _json.load(open(RESULT_DIR/'fig5_crossyear_2015.json'))
sem('跨年图: spec集合一致且面积口径均为TOTSQFT_EN',
    {r['spec'] for r in xy20}=={r['spec'] for r in xy15}
    and all(r['area_basis']=='TOTSQFT_EN' for r in xy20+xy15))
# 4) 无基准单元: mobile|pre70 必须为 NA(不可为0), 且每格附覆盖率
fg = _json.load(open(RESULT_DIR/'fig2_data.json'))
sem('热图: mobile|pre70 share 为 NA 而非 0',
    fg['heat']['mobile|pre70']['share'] is None)
sem('热图: 所有格均报告 coverage_pct',
    all('coverage_pct' in c for c in fg['heat'].values()))
# 5) 反事实: 安全组诊断行与系数缩放情景行存在(解释重定位的载体)
sem('反事实: 含 secure-group diagnostic 行与 coefficient-scaled 行',
    any('diagnostic' in i for i in cf.index) and any('coefficient-scaled' in i for i in cf.index))
# 6) 数据层构造检查(载入微数据, ~20s)
# v3.1(复查#3): 此前的产权/严重度检查是恒真表达式(筛KOWNRENT==3再查==3、计数自比较),
# 改为检查【实际进入回归的设计矩阵列】与【实际估计子样本】
from s0_common import load_2020, analysis_sample, build_X
_d = analysis_sample(load_2020())
sem('互斥分组+secure 恰好划分全样本',
    bool((_d['grp_fin_only']+_d['grp_temp_only']+_d['grp_both']+(1-_d['insecure'])).eq(1).all()))
_X = build_X(_d, 2)
sem('build_X: rent 列精确等于 KOWNRENT==2',
    _np.array_equal(_X['rent'].to_numpy(), (_d['KOWNRENT']==2).astype(float).to_numpy()))
sem('build_X: occ_free 列精确等于 KOWNRENT==3',
    _np.array_equal(_X['occ_free'].to_numpy(), (_d['KOWNRENT']==3).astype(float).to_numpy()))
sem('build_X: rent=occ_free=0 恰为 KOWNRENT==1(owner参照纯净)',
    _np.array_equal(((_X['rent']==0)&(_X['occ_free']==0)).to_numpy(),
                    (_d['KOWNRENT']==1).to_numpy()))
_sub_sev = _d[(_d['insecure']==0)|(_d['severe']==1)]
sem('严重度子样本: severe==0 组全为完全安全户, 且处理组非空',
    bool(_sub_sev.loc[_sub_sev['severe']==0,'insecure'].eq(0).all())
    and int(_sub_sev['severe'].sum()) > 0)
_sub_tmp = _d[(_d['insecure']==0)|(_d['dim_temp_any']==1)]
sem('温度子样本: dim_temp_any==0 组全为完全安全户, 且处理组非空',
    bool(_sub_tmp.loc[_sub_tmp['dim_temp_any']==0,'insecure'].eq(0).all())
    and int(_sub_tmp['dim_temp_any'].sum()) > 0)
from s0_common import flag_above_and_db
_df, _cov = flag_above_and_db(_d)
sem('flag_above_and_db: 无基准户 above/db 为 NaN(不可分类)',
    bool(_df.loc[~_df['cell_defined'],'above'].isna().all()) and
    bool(_df.loc[~_df['cell_defined'],'db'].isna().all()))
sem('table4: 严重度/非严重行存在(vs secure 标签)',
    'Severe insecurity vs secure' in t4.index and 'Non-severe insecurity vs secure' in t4.index)
# 7) 热图: 每格含 n_total/n_covered(小样本标记依据可分类样本量)
sem('热图: 每格含 n_total 与 n_covered',
    all(('n_total' in c and 'n_covered' in c) for c in fg['heat'].values()))

print()
print('全部通过 ✓' if not fails else f'失败 {len(fails)} 项: {fails}')
sys.exit(1 if fails else 0)
