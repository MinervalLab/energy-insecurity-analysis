# -*- coding: utf-8 -*-
"""fig2 — 原始强度差距的 OB 分解贡献 (独立整图)
数据驱动: table3_oaxaca.csv (s3 生成)
输出: fig2_decomposition.png (600dpi) + .pdf"""
import numpy as np
import pandas as pd
import matplotlib
from s0_common import RESULT_DIR as OUT
from s0_common import pub_font_family
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

INK='#0b0b0b'; SEC='#52514e'; MUT='#898781'; GRID='#e1e0d9'; BASE='#c3c2b7'
BLUE='#2a78d6'; RED='#e34948'; NEUT='#b5b3ac'; DARK='#52514e'
plt.rcParams.update({
 'font.family':pub_font_family(),'font.size':7,'axes.linewidth':0.6,
 'xtick.major.width':0.6,'ytick.major.width':0.6,'xtick.major.size':2.5,'ytick.major.size':2.5,
 'xtick.color':MUT,'ytick.color':MUT,'axes.edgecolor':BASE,'xtick.labelsize':6.4,
 'text.color':INK,'axes.labelcolor':SEC,'pdf.fonttype':42})

def rounded_hbar(ax,y,w,h,color,zorder=3):
    r=min(abs(w)*0.9,h*0.45); s=1 if w>0 else -1
    x1=w
    verts=[(0,y-h/2),(x1-s*r,y-h/2),(x1,y-h/2),(x1,y-h/2+r),(x1,y+h/2-r),(x1,y+h/2),(x1-s*r,y+h/2),(0,y+h/2),(0,y-h/2)]
    codes=[Path.MOVETO,Path.LINETO,Path.CURVE3,Path.CURVE3,Path.LINETO,Path.CURVE3,Path.CURVE3,Path.LINETO,Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts,codes),facecolor=color,edgecolor='none',zorder=zorder))

_t3 = pd.read_csv(OUT/'table3_oaxaca.csv').set_index('component')
def _g3(k): r=_t3.loc[k]; return float(r['pct_of_gap']), float(r['pct_of_gap_se'])
_names={'ex_structure':'Housing structure\n(chiefly floor area)','ex_quality':'Reported housing quality',
 'ex_behavior':'Thermostat behaviour','ex_econ_demo':'Income, demographics',
 'ex_geo_climate':'Climate, geography','ex_fuel':'Heating-fuel mix'}
_comp=sorted(_names, key=lambda k:-_g3(k)[0])
rows=[(_names[k],)+_g3(k)+('comp',) for k in _comp]+[None,
      ('Explained',)+_g3('explained')+('sum1',),
      ('Unexplained',)+_g3('unexplained')+('sum2',)]

fig, ax = plt.subplots(figsize=(3.9, 3.0))
fig.subplots_adjust(left=0.335, right=0.965, top=0.86, bottom=0.16)
ys=np.arange(len(rows))[::-1]
ax.axvline(0,color=BASE,lw=0.8,zorder=2)
for r,y in zip(rows,ys):
    if r is None: continue
    lab,v,vse,kind=r; err=1.96*vse
    c={'comp':BLUE if v>0 else RED,'sum1':NEUT,'sum2':DARK}[kind]
    rounded_hbar(ax,y,v,0.62,c)
    ax.plot([v-err,v+err],[y,y],lw=1.0,color=INK,alpha=0.5,solid_capstyle='round',zorder=4)
    xt=v+err+4 if v>0 else v-err-4
    ax.text(xt,y,f'{v:+.0f}%',va='center',ha='left' if v>0 else 'right',fontsize=6.2,color=SEC)
ax.set_yticks([y for r,y in zip(rows,ys) if r])
ax.set_yticklabels([r[0] for r in rows if r],fontsize=6.6,color=INK)
ax.set_xlim(-82,148); ax.set_ylim(-0.7,len(rows)-0.3)
ax.set_xticks([-50,0,50,100])
ax.set_xlabel('Contribution to raw intensity gap (% of gap)',fontsize=6.7)
ax.xaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True)
for s in ['top','right','left']: ax.spines[s].set_visible(False)
ax.tick_params(left=False)
ax.set_title('Structure (chiefly floor area)\nexplains the gap',
             fontsize=7.5,fontweight='bold',loc='left',pad=7)
fig.savefig(OUT/'fig2_decomposition.png',dpi=600,facecolor='white')
fig.savefig(OUT/'fig2_decomposition.pdf',facecolor='white')
print('saved fig2_decomposition ->', OUT)
