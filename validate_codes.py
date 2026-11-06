# -*- coding: utf-8 -*-
"""validate_codes — 取值域自动核验: 观测值 ⊄ 预期码集 时立即报错
被 s0_common.load_2020 / load_2015 自动调用; 也可单独运行打印核验报告"""
import numpy as np
import pandas as pd

EXPECTED_2020 = {
 'SCALEB':{0,1,2,3},'SCALEG':{0,1,2,3},'SCALEE':{0,1,2,3},
 'MONEYPY':set(range(1,17)),'KOWNRENT':{1,2,3},'TYPEHUQ':{1,2,3,4,5},
 'YEARMADERANGE':set(range(1,10)),
 'FUELHEAT':{-2,1,2,3,5,7,99},'EQUIPAGE':{-2,1,2,3,4,5,6},
 'ADQINSUL':{1,2,3,4},'DRAFTY':{1,2,3,4},'TYPEGLASS':{1,2,3},
 'EQUIPAUX':{-2,0,1},'ELPAY':{1,2,3,99},'ENERGYASST':{0,1},   # ELPAY 99=Other(n=5), 码本确认
 'HOUSEHOLDER_RACE':{1,2,3,4,5,6},'SDESCENT':{0,1},'ATHOME':{0,1,2,3,4,5},
 'AGERFRI1':{-2,1,2,3,4,5,6},
}
RANGE_2020 = {  # 连续变量: (最小允许, 最大允许, 允许的例外负码)
 'TEMPHOME':(50,90,{-2}),'TEMPHOMEAC':(50,90,{-2}),'NUMPORTEL':(1,9,{-2}),
 'HDD65':(0,20000,set()),'CDD65':(0,7000,set()),'SQFTEST':(1,20000,set()),
 'TOTSQFT_EN':(1,20000,set()),'NHSLDMEM':(1,20,set()),'NUMFRIG':(0,10,set()),
 'HHAGE':(18,110,set()),'NUMCHILD':(0,15,set()),
}
EXPECTED_2015 = {
 'SCALEB':{0,1,2,3},'SCALEG':{0,1,2,3},'SCALEE':{0,1,2,3},
 'MONEYPY':set(range(1,9)),'KOWNRENT':{1,2,3},'TYPEHUQ':{1,2,3,4,5},
 'YEARMADERANGE':set(range(1,9)),'FUELHEAT':{-2,1,2,3,5,7,21},'ELPAY':{1,2,3,9},
 'ADQINSUL':{1,2,3,4},'DRAFTY':{1,2,3,4},'TYPEGLASS':{1,2,3},
 'EQUIPAGE':{-2,1,2,3,5,41,42},'ATHOME':{0,1,2,3,4,5},
}
RANGE_2015 = {'TEMPHOME':(50,90,{-2}),'TEMPHOMEAC':(50,90,{-2})}
CAT_2020 = {'UATYP10':{'U','C','R'},
 'BA_climate':{'Cold','Very-Cold','Hot-Humid','Hot-Dry','Mixed-Humid','Mixed-Dry','Marine','Subarctic'}}
CAT_2015 = {'UATYP10':{'U','C','R'},
 'CLIMATE_REGION_PUB':{'Cold/Very Cold','Hot-Dry/Mixed-Dry','Hot-Humid','Marine','Mixed-Humid'}}

# v3(P3-02)/v3.1(复查#4): 关键列必需非缺失 + 权重有效性
# 清单覆盖【全部实际使用变量】(含 BTUNG/ATHOME/NUMPORTEL 等 v3 遗漏项);
# 权重按【完整预定列集合】核对(2020: NWEIGHT+60个重复; 2015: NWEIGHT+96个BRR),
# 且要求有限(拒绝±Inf)、非负、总和为正 —— v2/v3 注入 +Inf 或删除 NWEIGHT60 均不报错
REQUIRED_NONMISSING_2020 = ['SCALEB','SCALEG','SCALEE','MONEYPY','KOWNRENT','TYPEHUQ',
    'YEARMADERANGE','FUELHEAT','SQFTEST','TOTSQFT_EN','TOTALBTU','TOTALDOL','BTUEL',
    'BTUNG','HDD65','CDD65','NHSLDMEM','ADQINSUL','DRAFTY','TYPEGLASS','EQUIPAGE',
    'TEMPHOME','TEMPHOMEAC','NUMPORTEL','ATHOME','EQUIPAUX','ELPAY','ENERGYASST',
    'HOUSEHOLDER_RACE','SDESCENT','HHAGE','NUMCHILD','NUMFRIG','AGERFRI1',
    'UATYP10','BA_climate','state_postal','ZSCALEB','ZSCALEG','ZSCALEE','ZMONEYPY']
REQUIRED_NONMISSING_2015 = ['SCALEB','SCALEG','SCALEE','MONEYPY','KOWNRENT','TYPEHUQ',
    'YEARMADERANGE','FUELHEAT','TOTSQFT_EN','TOTALBTU','TOTALDOL','BTUEL','BTUNG',
    'HDD65','CDD65','NHSLDMEM','ADQINSUL','DRAFTY','TYPEGLASS','EQUIPAGE',
    'TEMPHOME','TEMPHOMEAC','ATHOME','UATYP10','CLIMATE_REGION_PUB']
WEIGHT_COLS_2020 = ['NWEIGHT'] + [f'NWEIGHT{i}' for i in range(1, 61)]
WEIGHT_COLS_2015 = ['NWEIGHT'] + [f'BRRWT{i}' for i in range(1, 97)]

def _check_missing(df, cols, wcols, wave, errors):
    for v in cols:
        if v not in df.columns:
            errors.append(f'[{wave}] 缺少必需变量 {v}'); continue
        n_na = int(df[v].isna().sum())
        if n_na: errors.append(f'[{wave}] {v} 存在缺失值 {n_na} 条(关键列必须非缺失)')
    for w in wcols:                                  # 完整预定权重列集合
        if w not in df.columns:
            errors.append(f'[{wave}] 缺少权重列 {w}'); continue
        x = df[w].to_numpy(dtype=float)
        if not np.all(np.isfinite(x)):
            errors.append(f'[{wave}] 权重 {w} 含缺失/±Inf')
        elif np.any(x < 0):
            errors.append(f'[{wave}] 权重 {w} 含负值')
        elif x.sum() <= 0:
            errors.append(f'[{wave}] 权重 {w} 总和非正')

def _check(df, expected, ranges, cats, wave):
    errors = []
    _check_missing(df, REQUIRED_NONMISSING_2020 if wave=='2020' else REQUIRED_NONMISSING_2015,
                   WEIGHT_COLS_2020 if wave=='2020' else WEIGHT_COLS_2015, wave, errors)
    for v, allowed in expected.items():
        if v not in df.columns: errors.append(f'[{wave}] 缺少变量 {v}'); continue
        obs = set(pd.unique(df[v].dropna()))
        extra = {x for x in obs if x not in allowed}
        if extra: errors.append(f'[{wave}] {v} 出现预期外取值: {sorted(extra)[:8]}')
    for v, (lo, hi, exc) in ranges.items():
        if v not in df.columns: continue
        x = df[v].dropna()
        bad = x[~(((x >= lo) & (x <= hi)) | x.isin(list(exc)))]
        if len(bad): errors.append(f'[{wave}] {v} 超出范围[{lo},{hi}]∪{exc}: 如 {sorted(set(bad))[:5]}')
    for v, allowed in cats.items():
        if v not in df.columns: continue
        # v3.1(复查#4): 类别列不再先dropna —— 缺失即违规(下游会落入全零虚拟变量模式)
        if df[v].isna().any():
            errors.append(f'[{wave}] {v} 含缺失类别 {int(df[v].isna().sum())} 条')
        extra = set(df[v].dropna().unique()) - allowed
        if extra: errors.append(f'[{wave}] {v} 出现预期外类别: {extra}')
    if errors:
        raise ValueError('取值域核验失败:\n' + '\n'.join(errors))
    return True

def validate_2020(df): return _check(df, EXPECTED_2020, RANGE_2020, CAT_2020, '2020')
def validate_2015(df): return _check(df, EXPECTED_2015, RANGE_2015, CAT_2015, '2015')

if __name__ == '__main__':
    from s0_common import CSV2020, CSV2015
    validate_2020(pd.read_csv(CSV2020)); print('2020 取值域核验: 全部通过')
    validate_2015(pd.read_csv(CSV2015)); print('2015 取值域核验: 全部通过')
