# -*- coding: utf-8 -*-
"""fig4 — 燃料/用途/气候/电暖器 2x2 整图
面板: (a) 分燃料溢价  (b) 分用途条件强度差  (c) 气候梯度  (d) 便携电暖器
数据驱动: fig3_mech.json (s5) + fig_enduse.json (s8)
输出: fig4_fuel_enduse_climate.png (600dpi) + .pdf (矢量, 供PDF排版做字体微调)"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch
from s0_common import RESULT_DIR, STYLE, set_pub_style

S = STYLE; set_pub_style()
INK,SEC,MUT,GRID,BASE = S['INK'],S['SEC'],S['MUT'],S['GRID'],S['BASE']
BLUE,ORANGE,NEUT = S['BLUE'],S['ORANGE'],S['NEUT']
mech = json.load(open(RESULT_DIR/'fig3_mech.json'))
eu   = json.load(open(RESULT_DIR/'fig_enduse.json'))

def pct(b): return 100*(np.exp(b)-1)
def ci(b, se): return pct(b-1.96*se), pct(b+1.96*se)
def style(ax):
    for sp in ['top','right','left']: ax.spines[sp].set_visible(False)
def letter(ax, s, x=-0.13):
    ax.text(x, 1.09, s, transform=ax.transAxes, fontsize=8.5,
            fontweight='bold', ha='right', va='bottom')
def rounded_vbar(ax, x, h, w, color, zorder=3):
    r = min(h*0.9, w*0.9)/2
    verts=[(x-w/2,0),(x-w/2,h-r),(x-w/2,h),(x-w/2+r,h),(x+w/2-r,h),(x+w/2,h),(x+w/2,h-r),(x+w/2,0),(x-w/2,0)]
    codes=[MPath.MOVETO,MPath.LINETO,MPath.CURVE3,MPath.CURVE3,MPath.LINETO,MPath.CURVE3,MPath.CURVE3,MPath.LINETO,MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts,codes),facecolor=color,edgecolor='none',zorder=zorder))

fig = plt.figure(figsize=(6.9, 4.9))
gs = fig.add_gridspec(2, 2, left=0.14, right=0.975, top=0.93, bottom=0.085,
                      hspace=0.62, wspace=0.46)

# ---------- (a) 分燃料 ----------
ax = fig.add_subplot(gs[0,0]); letter(ax, 'a')
labels = ['Electricity,\nall households', 'Natural gas,\ngas users only', 'Electricity, gas-\nheated homes only']
# 行序断言: 显示标签按位置映射到 mech['fuel'], 锁死与 s5 输出行名的对应关系
_expect = ['Electricity intensity, all households',
           'Gas intensity, conditional on gas use (selected subsample)',
           'Electricity intensity, conditional on gas main heat (selected subsample)']
assert [r[0] for r in mech['fuel']] == _expect, 'fig3_mech.json fuel 行序与标签不一致'
ys = np.arange(3)[::-1]
ax.axvline(0, color=BASE, lw=0.8, zorder=1)
for (name, b, se), y, lab in zip(mech['fuel'], ys, labels):
    lo, hi = ci(b, se); sig = abs(b/se) > 1.96
    ax.plot([lo,hi],[y,y],lw=1.6,color=BLUE,solid_capstyle='round',zorder=3)
    ax.plot(pct(b),y,'o',ms=4.8,mfc=BLUE if sig else 'white',
            mec='white' if sig else BLUE,mew=0.9 if sig else 1.2,zorder=4)
    ax.text(hi+0.7,y,f'{pct(b):+.1f}%'+('' if sig else ' (n.s.)'),va='center',fontsize=6.3,color=SEC)
ax.set_yticks(ys); ax.set_yticklabels(labels,fontsize=6.6,color=INK)
ax.set_xlim(-3,15); ax.set_ylim(-0.6,2.6); ax.set_xticks([0,5,10])
ax.set_xlabel('Energy-intensity premium (%)',fontsize=6.7)
ax.xaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True)
style(ax); ax.tick_params(left=False)
ax.set_title('The premium lives in electricity —\neven where gas heats the home',
             fontsize=7.5,fontweight='bold',loc='left',pad=7)

# ---------- (b) 分用途 (M4, 水平值口径) ----------
ax = fig.add_subplot(gs[0,1]); letter(ax, 'b')
ORDER = ['sph','col','wth','oth']
EULAB = {'sph':'Space heating','col':'Space cooling','wth':'Water heating',
         'oth':'Other end uses\n(incl. refrig.)'}
ys = np.arange(len(ORDER))[::-1]
ax.axvline(0,color=BASE,lw=0.8,zorder=1)
for k,y in zip(ORDER,ys):
    b,se,m = eu[k]['coef'],eu[k]['se'],eu[k]['sec_mean']
    lo,hi = b-1.96*se, b+1.96*se
    ax.plot([lo,hi],[y,y],lw=1.6,color=BLUE,solid_capstyle='round',zorder=3)
    ax.plot(b,y,'o',ms=4.8,mfc=BLUE,mec='white',mew=0.9,zorder=4)
    ax.text(hi+0.08,y,f'+{b:.2f} ({100*b/m:.1f}%)',va='center',fontsize=6.3,color=SEC)  # 轴量(kBtu/ft2)+括号相对安全组均值
ax.set_yticks(ys); ax.set_yticklabels([EULAB[k] for k in ORDER],fontsize=6.6,color=INK)
ax.set_xlim(-0.35,3.05); ax.set_ylim(-0.6,3.6); ax.set_xticks([0,1,2,3])
ax.set_xlabel('Conditional difference (kBtu/ft²)',fontsize=6.7)
ax.xaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True)
style(ax); ax.tick_params(left=False)
ax.set_title('The premium spans all end uses;\nlargest share in base load',
             fontsize=7.5,fontweight='bold',loc='left',pad=7)

# ---------- (c) 气候梯度 ----------
ax = fig.add_subplot(gs[1,0]); letter(ax, 'c')
colors = {'HDD65': S['SEQ'][550], 'CDD65': ORANGE}
names = {'HDD65':'Heating degree-days','CDD65':'Cooling degree-days'}
for var in ['HDD65','CDD65']:
    g = np.array(mech['climate'][var]['grid'])   # [x(千度日,加权分位点), coef, se]
    xn = np.array([0,1,2], dtype=float)
    ylo = [ci(b,s)[0] for b,s in g[:,1:]]; yhi = [ci(b,s)[1] for b,s in g[:,1:]]
    ax.fill_between(xn,ylo,yhi,color=colors[var],alpha=0.13,lw=0)
    ax.plot(xn,[pct(b) for b in g[:,1]],lw=2,color=colors[var],solid_capstyle='round')
    dyv = 7.0 if var=='HDD65' else -10.5
    for xi,(x_,b,s) in zip(xn,g):
        ax.plot(xi,pct(b),'o',ms=4.2,mfc=colors[var],mec='white',mew=0.9,zorder=4)
        ax.annotate(f'{x_*1000:,.0f}',(xi,pct(b)),textcoords='offset points',
                    xytext=(0,dyv),ha='center',fontsize=5.0,color=colors[var])
ax.axhline(0,color=BASE,lw=0.8)
ax.set_xticks([0,1,2]); ax.set_xticklabels(['P10','P50','P90'],fontsize=6.4)
ax.set_xlim(-0.25,2.25)
ax.set_xlabel('Survey-weighted degree-day percentile (values: deg-days)',fontsize=6.7)
ax.set_ylabel('Implied premium (%)',fontsize=6.7)
ax.yaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True)
style(ax); ax.tick_params(left=False)
ax.legend(handles=[Line2D([],[],lw=2,color=colors[v],label=names[v]) for v in ['HDD65','CDD65']],
          loc='upper left',frameon=False,fontsize=6.0,handlelength=1.3,borderaxespad=0.1)
ax.set_title('Larger where cooling dominates,\nsmaller where heating dominates',
             fontsize=7.5,fontweight='bold',loc='left',pad=7)

# ---------- (d) 便携电暖器 ----------
ax = fig.add_subplot(gs[1,1]); letter(ax, 'd')
cats = ['One portable\nelectric heater','Two or more']
xpos = np.array([0,1]); bw = 0.32
for k,(grp,c) in enumerate([('Insecure',BLUE),('Secure',NEUT)]):
    vals = mech['heaters'][grp][1:]
    for x,v in zip(xpos,vals):
        rounded_vbar(ax, x+(k-0.5)*(bw+0.06), v, bw, c)
        ax.text(x+(k-0.5)*(bw+0.06), v+0.5, f'{v:.1f}%', ha='center', fontsize=6.3, color=SEC)
ax.set_xticks(xpos); ax.set_xticklabels(cats,fontsize=6.6,color=INK)
ax.set_xlim(-0.65,1.65); ax.set_ylim(0,18); ax.set_yticks([0,5,10,15])
ax.set_ylabel('Share of households (%)',fontsize=6.7)
ax.yaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True)
style(ax); ax.tick_params(left=False, bottom=False)
ax.legend(handles=[Line2D([],[],marker='s',ls='',mfc=c,mec='none',ms=6,label=n)
                   for n,c in [('Insecure',BLUE),('Secure',NEUT)]],
          loc='upper right',frameon=False,fontsize=6.0,handletextpad=0.3,borderaxespad=0)
ax.set_title('The space-heater trace: inefficient\nplug-in heating rises with insecurity',
             fontsize=7.5,fontweight='bold',loc='left',pad=7)

fig.savefig(RESULT_DIR/'fig4_fuel_enduse_climate.png', dpi=600, facecolor='white')
fig.savefig(RESULT_DIR/'fig4_fuel_enduse_climate.pdf', facecolor='white')
print('saved fig4_fuel_enduse_climate (a燃料/b分用途/c气候/d电暖器)')
