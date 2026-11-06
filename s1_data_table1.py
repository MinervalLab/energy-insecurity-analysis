# -*- coding: utf-8 -*-
"""s1 — 数据自检 + 分组描述统计
v2: 预期值移至 expected_results.json (verify.py 校验); above/db 用 s0 单一定义源
输出: result/table1_descriptives.csv"""
import numpy as np, pandas as pd
from s0_common import save_csv
from s0_common import (load_2020, analysis_sample, flag_above_and_db, wshare, RESULT_DIR)

df = load_2020()          # 载入即自动做取值域核验
print('数据维度:', df.shape, '| 加权户数(M):', round(df['NWEIGHT'].sum()/1e6,1))
print('any不安全率(%):', round(np.average(df['insecure'],weights=df['NWEIGHT'])*100,1))
d, coverage = flag_above_and_db(analysis_sample(df))
print('分析样本 n =', len(d), '| 单元覆盖率(%):', round(coverage,1))

# v3(P1-03): db 现为 NaN/0/1 三值, 掩码必须显式 ==1 (NaN即无基准, 不计入双重负担列)
groups = {'Secure': d['insecure']==0, 'Insecure (any)': d['insecure']==1, 'Double burden': d['db']==1}
rows = []; W='NWEIGHT'
def wm(g, col): return np.average(g[col], weights=g[W]) if len(g) else float('nan')
for name, mask in groups.items():
    g = d[mask]
    rows.append({'group': name, 'n (unweighted)': len(g), 'households (M)': g[W].sum()/1e6,
      'site kBtu/sqft': wm(g,'bps'), 'total kBtu': wm(g,'TOTALBTU'), 'annual $': wm(g,'TOTALDOL'),
      'sqft': wm(g,'SQFTEST'), 'income $k (midpoint)': wm(g,'income_k'),
      'HDD65': wm(g,'HDD65'), 'CDD65': wm(g,'CDD65'),
      'renter %': wshare(g, g['KOWNRENT']==2), 'mobile home %': wshare(g, g['TYPEHUQ']==1),
      'apartment %': wshare(g, g['TYPEHUQ'].isin([4,5])), 'pre-1970 %': wshare(g, g['YEARMADERANGE']<=3),
      'poorly insulated %': wshare(g, g['ADQINSUL']>=3), 'electric heat %': wshare(g, g['FUELHEAT']==5),
      'no heating equipment %': wshare(g, g['FUELHEAT']==-2), 'household size': wm(g,'NHSLDMEM')})
t1 = pd.DataFrame(rows).set_index('group').T.round(2)
save_csv(t1, RESULT_DIR/'table1_descriptives.csv')
print(t1.to_string())
