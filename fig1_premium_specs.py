# -*- coding: utf-8 -*-
"""fig1 — premium 随控制集递进 (独立整图)
数据驱动: fig1a_specs.json (s2 生成)
输出: fig1_premium_specs.png (600dpi) + .pdf"""
import json
import numpy as np
import matplotlib
from s0_common import RESULT_DIR as OUT
from s0_common import pub_font_family
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK='#0b0b0b'; SEC='#52514e'; MUT='#898781'; GRID='#e1e0d9'; BASE='#c3c2b7'
BLUE='#2a78d6'
plt.rcParams.update({
 'font.family':pub_font_family(),'font.size':7,'axes.linewidth':0.6,
 'xtick.major.width':0.6,'ytick.major.width':0.6,'xtick.major.size':2.5,'ytick.major.size':2.5,
 'xtick.color':MUT,'ytick.color':MUT,'axes.edgecolor':BASE,'xtick.labelsize':6.4,
 'text.color':INK,'axes.labelcolor':SEC,'pdf.fonttype':42})

def pct(b): return 100*(np.exp(b)-1)
def ci(b,se): return pct(b-1.96*se), pct(b+1.96*se)

_specs_raw = json.load(open(OUT/'fig1a_specs.json'))
_LABELS = {'M0 Unadjusted (log scale)':'Unadjusted (log scale)',
 'M1 +Climate/geography':'+ Climate & geography',
 'M2 +Income/dwelling/fuel/tenure':'+ Income, dwelling,\n   fuel, tenure',
 'M3 +Reported housing quality':'+ Reported housing\n   quality',
 'M4 +Thermostat behaviour':'+ Thermostat\n   behaviour',
 'State fixed effects':'State fixed effects',
 'Source-energy accounting':'Source-energy accounting\n(electricity × 2.80)'}
specs=[(_LABELS.get(l,l), b, se) for l, b, se in _specs_raw]

fig, ax = plt.subplots(figsize=(3.7, 2.9))
fig.subplots_adjust(left=0.40, right=0.965, top=0.90, bottom=0.165)
ys=np.arange(len(specs))[::-1]
ax.axvline(0,color=BASE,lw=0.8,zorder=1)
for (lab,b,se),y in zip(specs,ys):
    lo,hi=ci(b,se)
    ax.plot([lo,hi],[y,y],lw=1.6,color=BLUE,solid_capstyle='round',zorder=3)
    ax.plot(pct(b),y,'o',ms=4.6,mfc=BLUE,mec='white',mew=0.9,zorder=4)
    ax.text(hi+0.7,y,f'+{pct(b):.1f}%',va='center',ha='left',fontsize=6.2,color=SEC)
ax.set_yticks(ys); ax.set_yticklabels([s[0] for s in specs],fontsize=6.6,color=INK)
ax.set_xlim(-2.5,19); ax.set_ylim(-0.7,len(specs)-0.3)
ax.set_xticks([0,5,10,15])
ax.set_xlabel('Energy-intensity premium of insecure households (%)',fontsize=6.7)
ax.xaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True)
for s in ['top','right','left']: ax.spines[s].set_visible(False)
ax.tick_params(left=False)
ax.set_title('Premium survives progressively\nricher controls',
             fontsize=7.5,fontweight='bold',loc='left',pad=7)
fig.savefig(OUT/'fig1_premium_specs.png',dpi=600,facecolor='white')
fig.savefig(OUT/'fig1_premium_specs.pdf',facecolor='white')
print('saved fig1_premium_specs ->', OUT)
