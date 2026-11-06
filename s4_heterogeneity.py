# -*- coding: utf-8 -*-
"""s4 — 维度异质性 + 敏感性 + 公平性 (table4 / fig3 数据)
v2: 房型异质性改为全样本交互模型(主口径, 共同支撑), 另存子样本版;
    插补排除明确针对处理变量(不安全状态), 另设更严格版(加收入/面积);
    交互标签修正为无歧义表述
输出: table4_heterogeneity.csv, table4b_subsample_appendix.csv  |  耗时 4-6 分钟"""
import numpy as np, pandas as pd
from s0_common import save_csv
from s0_common import (load_2020, analysis_sample, build_X, wls_beta, jackknife,
                       premium_coef, pct, REPS, JK_COEF, RESULT_DIR)

df = load_2020(); d = analysis_sample(df)
rows = []
def run(name, data, **kw):
    th, se = jackknife(lambda dd,w: premium_coef(dd,w,2,**kw), data)
    print(f'{name}: {th:+.4f} (se={se:.4f}, t={th/se:.1f}) -> {pct(th):+.1f}%')
    rows.append([name, th, se, th/se])

print('=== 定义 (v3/P1-01,P1-02: 全部以完全安全家庭为参照; 明确样本构成) ===')
# 基线: 全样本, insecure vs 其余(其余即insecure==0=完全安全) —— 参照组本来就纯净, 不变
run('Any insecurity vs secure (baseline)', d)
# 严重度: v2 的 severe-vs-非severe 把轻度不安全户留在参照组(17,258户中2,988户不安全,
# 加权19.60%), 不是"vs 安全家庭"。v3 改为分层子样本: 参照组只含 insecure==0
run('Severe insecurity vs secure', d[(d['insecure']==0)|(d['severe']==1)], ins_col='severe')
run('Non-severe insecurity vs secure', d[(d['insecure']==0)|((d['insecure']==1)&(d['severe']==0))])
# 温度维度: v2 的 scaleg_only 在全样本上回归, 参照组混入2,545户仅财务困难家庭 -> 已废弃
run('Any temperature difficulty vs secure', d[(d['insecure']==0)|(d['dim_temp_any']==1)],
    ins_col='dim_temp_any')
# 互斥分类(单一回归, 参照=完全安全): fin_only / temp_only / both 三个指示同入模型
print('=== 定义: 互斥维度分组 (单回归, 参照=完全安全) ===')
def excl_groups(data, wcol):
    Xd = build_X(data, 2, ins_col='grp_fin_only')
    Xd['grp_temp_only'] = data['grp_temp_only'].astype(float).values
    Xd['grp_both'] = data['grp_both'].astype(float).values
    b = wls_beta(Xd, np.log(data['bps'].values), data[wcol].values, check=False)
    cols = list(Xd.columns)
    return (b[1], b[1+cols.index('grp_temp_only')], b[1+cols.index('grp_both')])
th3 = excl_groups(d,'NWEIGHT'); reps3 = np.array([excl_groups(d,r) for r in REPS])
for i, nm in enumerate(['Financial difficulty only vs secure (mutually exclusive)',
                        'Temperature difficulty only vs secure (mutually exclusive)',
                        'Financial and temperature difficulty vs secure (mutually exclusive)']):
    se = np.sqrt(JK_COEF*np.sum((reps3[:,i]-th3[i])**2))
    print(f'{nm}: {th3[i]:+.4f} (se={se:.4f}, t={th3[i]/se:.1f}) -> {pct(th3[i]):+.1f}%')
    rows.append([nm, th3[i], se, th3[i]/se])
# 结构断言(P3-01方向): 三组与secure构成分析样本的划分, 且各"vs secure"子样本参照组纯净
assert (d['grp_fin_only']+d['grp_temp_only']+d['grp_both']+(1-d['insecure'])).eq(1).all(), \
    '互斥分组+secure 必须恰好划分全样本'
assert (d.loc[(d['insecure']==0)|(d['severe']==1),'insecure']
        .loc[lambda s: s==0].eq(0).all()), '严重度子样本参照组必须只含insecure==0'
print('=== 样本(插补敏感性, 针对处理变量测量) ===')
run('Excl. imputed insecurity status (ZSCALE*=0)',
    d[(d['ZSCALEB']==0)&(d['ZSCALEG']==0)&(d['ZSCALEE']==0)])
strict = d[(d['ZSCALEB']==0)&(d['ZSCALEG']==0)&(d['ZSCALEE']==0)&(d['ZMONEYPY']==0)]
if 'ZSQFTEST' in d.columns: strict = strict[strict['ZSQFTEST']==0]
run(f'... also excl. imputed income/floor-area (n={len(strict)})', strict)
run('Bill-paying households only (ELPAY=1)', d[d['ELPAY']==1])

print('=== 房型异质性: 全样本交互模型(主口径) ===')
def type_premiums(data, wcol):
    # v3(P2-07): 返回全部五类房型的组内溢价 + 相对独栋的交互差异(townhouse不再省略)
    Xd = build_X(data, 2)
    for t in [1,3,4,5]:
        Xd[f'ix_{t}'] = Xd['insecure']*Xd[f'type_{t}']
    b = wls_beta(Xd, np.log(data['bps'].values), data[wcol].values, check=False)
    cols = list(Xd.columns); base = b[1]
    dif = {t: b[1+cols.index(f'ix_{t}')] for t in [1,3,4,5]}
    return (base, base+dif[1], base+dif[3], base+dif[4], base+dif[5],
            dif[1], dif[3], dif[4], dif[5])
th = type_premiums(d,'NWEIGHT'); reps = np.array([type_premiums(d,r) for r in REPS])
for i, name in enumerate(['Premium, single-family detached (interaction ref.)',
                          'Premium, mobile homes (interaction)',
                          'Premium, single-family attached/townhouse (interaction)',
                          'Premium, apartments 2-4 units (interaction)',
                          'Premium, apartments 5+ units (interaction)',
                          'Difference vs detached: mobile homes',
                          'Difference vs detached: attached/townhouse',
                          'Difference vs detached: apartments 2-4 units',
                          'Difference vs detached: apartments 5+ units']):
    se = np.sqrt(JK_COEF*np.sum((reps[:,i]-th[i])**2))
    print(f'{name}: {th[i]:+.4f} (se={se:.4f}) -> {pct(th[i]):+.1f}%')
    rows.append([name, th[i], se, th[i]/se])

print('=== 公平性: 组别溢价(交互模型, 标签无歧义) ===')
def group_premiums(data, wcol, gfn, grp_in_controls=False):
    # grp_in_controls=True: 组指示已在控制集中(如rent), 不再添加重复列以避免完全共线(v2秩检查捕获的v1隐患)
    Xd = build_X(data, 2); g = gfn(data).astype(float).values
    if not grp_in_controls: Xd['grp'] = g
    Xd['ix'] = Xd['insecure']*g
    b = wls_beta(Xd, np.log(data['bps'].values), data[wcol].values, check=False)
    return b[1], b[1]+b[len(Xd.columns)], b[len(Xd.columns)]
# v3(P2-01): 产权改三分类交互。rent与occ_free均已在控制集(build_X spec>=2),
# 交互对两类分别加入, 使owner参照纯净(KOWNRENT==1), 不再把免租户(178户)并入"owner"
def tenure_premiums(data, wcol):
    Xd = build_X(data, 2)
    Xd['ix_rent'] = Xd['insecure']*Xd['rent']
    Xd['ix_free'] = Xd['insecure']*Xd['occ_free']
    b = wls_beta(Xd, np.log(data['bps'].values), data[wcol].values, check=False)
    cols = list(Xd.columns)
    br, bf = b[1+cols.index('ix_rent')], b[1+cols.index('ix_free')]
    return (b[1], b[1]+br, br, b[1]+bf)   # owner / renter / 交互(rent) / free
tht = tenure_premiums(d,'NWEIGHT'); rept = np.array([tenure_premiums(d,r) for r in REPS])
for i, nm in enumerate(['Premium, owner-occupied (KOWNRENT=1, interaction ref.)',
                        'Premium, renter-occupied',
                        'Interaction: insecure x renter-occupied',
                        'Premium, rent-free occupants (KOWNRENT=3; small group)']):
    se = np.sqrt(JK_COEF*np.sum((rept[:,i]-tht[i])**2))
    print(f'{nm}: {tht[i]:+.4f} (se={se:.4f})')
    rows.append([nm, tht[i], se, tht[i]/se])

for g1name, g0name, gfn, in_ctrl in [
        ('Black householder','non-Black householder', lambda x:(x['HOUSEHOLDER_RACE']==2), False),
        ('Hispanic/Latino','non-Hispanic', lambda x:(x['SDESCENT']==1), False),
        ('income < $25k','income >= $25k', lambda x:(x['MONEYPY']<=7), False),
        # v4: 家庭构成交互 -- 户规模/儿童/老年户主。
        # 注: nmem(连续)在控制集中, 单人户虚拟变量与其不共线, grp照常加入
        ('single-person household','multi-person household', lambda x:(x['NHSLDMEM']==1), False),
        ('household with children','household without children', lambda x:(x['NUMCHILD']>=1), False),
        ('householder aged 65+','householder under 65', lambda x:(x['HHAGE']>=65), False)]:
    th = group_premiums(d,'NWEIGHT',gfn,in_ctrl)
    reps = np.array([group_premiums(d,r,gfn,in_ctrl) for r in REPS])
    for i, nm in enumerate([f'Premium, {g0name}', f'Premium, {g1name}', f'Interaction: insecure x {g1name}']):
        se = np.sqrt(JK_COEF*np.sum((reps[:,i]-th[i])**2))
        rows.append([nm, th[i], se, th[i]/se])
    print(f'{g0name} {pct(th[0]):+.1f}% | {g1name} {pct(th[1]):+.1f}% | 交互 t={th[2]/np.sqrt(JK_COEF*np.sum((reps[:,2]-th[2])**2)):.1f}')

save_csv(pd.DataFrame(rows, columns=['estimate','coef','se','t']).round(4), RESULT_DIR/'table4_heterogeneity.csv', index=False)

# 补充: 子样本版(与交互版对照, 说明支撑域差异)
def sub_coef(data, wcol):
    # 子样本回归: 按有效权重行(w>0)剔除零方差列 —— JK重复权重会置零某层, 
    # 导致部分虚拟变量在有效样本内变常数; insecure不在共线span内, 剔除仅为满秩整洁
    Xd = build_X(data, 2); w = data[wcol].values
    m = w > 0
    keep = [c for c in Xd.columns if Xd[c].values[m].std() > 0 or c=='insecure']
    return wls_beta(Xd[keep], np.log(data['bps'].values), w)[1]
app = []
for t,name in [(1,'mobile homes'),(2,'single-family detached'),(5,'apartments 5+')]:
    th, se = jackknife(sub_coef, d[d['TYPEHUQ']==t])
    app.append([f'Subsample: {name}', th, se, th/se])
save_csv(pd.DataFrame(app, columns=['estimate','coef','se','t']).round(4), RESULT_DIR/'table4b_subsample_appendix.csv', index=False)
print('已存 table4_heterogeneity.csv / table4b_subsample_appendix.csv ->', RESULT_DIR)
