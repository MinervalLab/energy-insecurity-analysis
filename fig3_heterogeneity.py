# -*- coding: utf-8 -*-
"""fig3 — premium 的稳健性与异质性森林图 (独立整图)
数据驱动: table4_heterogeneity.csv (s4 生成); 含 COMPOSITION 组 (家庭构成交互, 6 行)
输出: fig3_heterogeneity.png (600dpi) + .pdf"""
import numpy as np
import pandas as pd
import matplotlib
from s0_common import RESULT_DIR as OUT
from s0_common import pub_font_family
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK='#0b0b0b'; SEC='#52514e'; MUT='#898781'; GRID='#e1e0d9'; BASE='#c3c2b7'
BLUE='#2a78d6'; RED='#e34948'
plt.rcParams.update({
 'font.family':pub_font_family(),'font.size':7,'axes.linewidth':0.6,
 'xtick.major.width':0.6,'ytick.major.width':0.6,'xtick.major.size':2.5,'ytick.major.size':2.5,
 'xtick.color':MUT,'ytick.color':MUT,'axes.edgecolor':BASE,'xtick.labelsize':6.4,
 'text.color':INK,'axes.labelcolor':SEC,'pdf.fonttype':42})

def pct(b): return 100*(np.exp(b)-1)
def ci(b,se): return pct(b-1.96*se), pct(b+1.96*se)

_t4=pd.read_csv(OUT/'table4_heterogeneity.csv').set_index('estimate')
def _g4(k): r=_t4.loc[k]; return float(r['coef']), float(r['se'])
_c=[('DEFINITION (all vs fully secure)',None,None,'head'),
 ('Any insecurity (baseline)',)+_g4('Any insecurity vs secure (baseline)')+('filled',),
 ('Severe insecurity',)+_g4('Severe insecurity vs secure')+('filled',),
 ('Non-severe insecurity',)+_g4('Non-severe insecurity vs secure')+('filled',),
 ('Any temperature difficulty',)+_g4('Any temperature difficulty vs secure')+('filled',),
 ('DIMENSIONS (mutually exclusive)',None,None,'head'),
 ('Financial difficulty only',)+_g4('Financial difficulty only vs secure (mutually exclusive)')+('filled',),
 ('Temperature difficulty only',)+_g4('Temperature difficulty only vs secure (mutually exclusive)')+('open',),
 ('Financial & temperature',)+_g4('Financial and temperature difficulty vs secure (mutually exclusive)')+('filled',),
 ('SAMPLE',None,None,'head'),
 ('Excl. imputed insecurity status',)+_g4('Excl. imputed insecurity status (ZSCALE*=0)')+('filled',),
 ('Bill-paying households only',)+_g4('Bill-paying households only (ELPAY=1)')+('filled',),
 ('HOUSING TYPE (interaction model)',None,None,'head'),
 ('Single-family detached (ref.)',)+_g4('Premium, single-family detached (interaction ref.)')+('filled',),
 ('Mobile homes',)+_g4('Premium, mobile homes (interaction)')+('filled',),
 ('Attached / townhouse',)+_g4('Premium, single-family attached/townhouse (interaction)')+('filled',),
 ('Apartments 2–4 units',)+_g4('Premium, apartments 2-4 units (interaction)')+('filled',),
 ('Apartments 5+ units',)+_g4('Premium, apartments 5+ units (interaction)')+('filled',),
 ('TENURE (interaction model)',None,None,'head'),
 ('Owner-occupied (ref.)',)+_g4('Premium, owner-occupied (KOWNRENT=1, interaction ref.)')+('filled',),
 ('Renter-occupied',)+_g4('Premium, renter-occupied')+('filled',),
 ('COMPOSITION (interaction models)',None,None,'head'),
 ('Multi-person household (ref.)',)+_g4('Premium, multi-person household')+('filled',),
 ('Single-person household',)+_g4('Premium, single-person household')+('filled',),
 ('Without children (ref.)',)+_g4('Premium, household without children')+('filled',),
 ('With children',)+_g4('Premium, household with children')+('filled',),
 ('Householder under 65 (ref.)',)+_g4('Premium, householder under 65')+('filled',),
 ('Householder aged 65+',)+_g4('Premium, householder aged 65+')+('open',)]
rows=_c

fig, ax = plt.subplots(figsize=(7.2, 6.3))
fig.subplots_adjust(left=0.205, right=0.98, top=0.925, bottom=0.065)
ys=np.arange(len(rows))[::-1]
ax.axvline(0,color=BASE,lw=0.8,zorder=1)
for (lab,b,se,kind),y in zip(rows,ys):
    if kind=='head': continue
    lo,hi=ci(b,se)
    c=BLUE if kind=='filled' else RED
    ax.plot([lo,hi],[y,y],lw=1.6,color=c,solid_capstyle='round',zorder=3)
    ax.plot(pct(b),y,'o',ms=4.6,mfc=c if kind=='filled' else 'white',
            mec='white' if kind=='filled' else c,mew=0.9 if kind=='filled' else 1.1,zorder=4)
    ax.text(hi+0.7,y,f'{pct(b):+.1f}%',va='center',fontsize=6.2,color=SEC)
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in rows],fontsize=6.6)
for tl,(lab,_,_,kind) in zip(ax.get_yticklabels(),rows):
    if kind=='head': tl.set_fontsize(5.7); tl.set_color(MUT); tl.set_fontweight('bold')
    else: tl.set_color(INK)
ax.set_xlim(-10.5,23); ax.set_ylim(-0.7,len(rows)-0.3)
ax.set_xticks([-10,-5,0,5,10,15,20])
ax.set_xlabel('Energy-intensity premium vs secure households (%)',fontsize=6.7)
ax.xaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True)
for s in ['top','right','left']: ax.spines[s].set_visible(False)
ax.tick_params(left=False)
ax.set_title('Premium attaches to financial strain; none detected for temperature-only difficulty',
             fontsize=7.5,fontweight='bold',loc='left',pad=8)
fig.savefig(OUT/'fig3_heterogeneity.png',dpi=600,facecolor='white')
fig.savefig(OUT/'fig3_heterogeneity.pdf',facecolor='white')
print('saved fig3_heterogeneity ->', OUT)
