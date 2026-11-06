# -*- coding: utf-8 -*-
"""fig5 — 2015/2020 跨年复制 (独立整图)
数据驱动: fig5_crossyear_2020.json / fig5_crossyear_2015.json (s2/s6 生成,
两年均为 TOTSQFT_EN 同口径)
输出: fig5_replication2015.png (600dpi) + .pdf"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from s0_common import RESULT_DIR, STYLE, set_pub_style

S = STYLE; set_pub_style()
INK,SEC,MUT,GRID,BASE = S['INK'],S['SEC'],S['MUT'],S['GRID'],S['BASE']
BLUE,ORANGE = S['BLUE'],S['ORANGE']

def pct(b): return 100*(np.exp(b)-1)
def ci(b, se): return pct(b-1.96*se), pct(b+1.96*se)

_xy20 = {r['spec']: (r['coef'], r['se']) for r in json.load(open(RESULT_DIR/'fig5_crossyear_2020.json'))}
_xy15 = {r['spec']: (r['coef'], r['se']) for r in json.load(open(RESULT_DIR/'fig5_crossyear_2015.json'))}
assert set(_xy20) == set(_xy15) == {'M0','M1','M2','M4'}, '跨年序列必须同spec集合'
SPECS = [('M0','M0: Unadjusted'),('M1','M1: M0 + Climate'),
         ('M2','M2: M1 + Economic\n       & structural'),('M4','M4: Full controls')]

fig, ax = plt.subplots(figsize=(3.7, 2.9))
fig.subplots_adjust(left=0.295, right=0.96, top=0.86, bottom=0.165)
ys = np.arange(len(SPECS))[::-1]
ax.axvline(0,color=BASE,lw=0.8,zorder=1)
off = 0.17
for (k,lab),y in zip(SPECS,ys):
    for (bb,ss),c,dy in [(_xy20[k],BLUE,off),(_xy15[k],ORANGE,-off)]:
        lo,hi = ci(bb,ss)
        ax.plot([lo,hi],[y+dy,y+dy],lw=1.6,color=c,solid_capstyle='round',zorder=3)
        ax.plot(pct(bb),y+dy,'o',ms=4.4,mfc=c,mec='white',mew=0.9,zorder=4)
        ax.text(hi+0.6,y+dy,f'{pct(bb):+.1f}%',va='center',fontsize=5.9,color=SEC)
ax.set_yticks(ys); ax.set_yticklabels([lab for _,lab in SPECS],fontsize=6.6,color=INK)
ax.set_xlim(-2,27); ax.set_xticks([0,5,10,15,20,25])
ax.set_xlabel('Premium, TOTSQFT_EN floor-area basis (%)',fontsize=6.7)
ax.xaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True)
for sp in ['top','right','left']: ax.spines[sp].set_visible(False)
ax.tick_params(left=False)
ax.legend(handles=[Line2D([],[],marker='o',ls='',mfc=BLUE,mec='white',ms=4.4,label='2020'),
                   Line2D([],[],marker='o',ls='',mfc=ORANGE,mec='white',ms=4.4,label='2015 (pre-pandemic)')],
          loc='lower right',frameon=False,fontsize=6.0,handletextpad=0.2,borderaxespad=0)
ax.set_title('Not a pandemic artifact:\nthe premium predates COVID-19',
             fontsize=7.5,fontweight='bold',loc='left',pad=7)
fig.savefig(RESULT_DIR/'fig5_replication2015.png', dpi=600, facecolor='white')
fig.savefig(RESULT_DIR/'fig5_replication2015.pdf', facecolor='white')
print('saved fig5_replication2015 ->', RESULT_DIR)
