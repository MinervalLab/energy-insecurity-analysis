# -*- coding: utf-8 -*-
"""fig6 — 双重负担画像 (需先跑 s7; 输入 fig2_data.json 为内部历史命名)
输出: result/fig6_double_burden.png + .pdf"""
import json
import numpy as np
import matplotlib
from s0_common import RESULT_DIR as OUT
from s0_common import pub_font_family
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.lines import Line2D

INK='#0b0b0b'; SEC='#52514e'; MUT='#898781'; GRID='#e1e0d9'; BASE='#c3c2b7'; NEUT='#b5b3ac'
BLUE='#2a78d6'; ORANGE='#eb6834'
SEQ={100:'#cde2fb',150:'#b7d3f6',200:'#9ec5f4',300:'#6da7ec',450:'#2a78d6',550:'#1c5cab',700:'#0d366b'}
plt.rcParams.update({'font.family':pub_font_family(),'font.size':7,'axes.linewidth':0.6,
 'xtick.major.width':0.6,'ytick.major.width':0.6,'xtick.major.size':2.5,'ytick.major.size':2.5,
 'xtick.color':MUT,'ytick.color':MUT,'axes.edgecolor':BASE,'xtick.labelsize':6.4,'ytick.labelsize':6.6,
 'text.color':INK,'axes.labelcolor':SEC,'pdf.fonttype':42})
D=json.load(open(OUT/'fig2_data.json'))

def rounded_hbar(ax,y,w,h,color,zorder=3):
    r=min(abs(w)*0.9,h*0.45)
    verts=[(0,y-h/2),(w-r,y-h/2),(w,y-h/2),(w,y-h/2+r),(w,y+h/2-r),(w,y+h/2),(w-r,y+h/2),(0,y+h/2),(0,y-h/2)]
    codes=[Path.MOVETO,Path.LINETO,Path.CURVE3,Path.CURVE3,Path.LINETO,Path.CURVE3,Path.CURVE3,Path.LINETO,Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts,codes),facecolor=color,edgecolor='none',zorder=zorder))
def style(ax):
    for s in ['top','right','left']: ax.spines[s].set_visible(False)
def letter(ax,s,x=-0.04):
    ax.text(x,1.07,s,transform=ax.transAxes,fontsize=8.5,fontweight='bold',ha='right',va='bottom')

fig=plt.figure(figsize=(7.2,6.7))
gs=fig.add_gridspec(2,2,left=0.20,right=0.975,top=0.94,bottom=0.065,hspace=0.55,wspace=0.50,
                    height_ratios=[0.78,1])

# ---------- a. funnel ----------
ax=fig.add_subplot(gs[0,0]); letter(ax,'a',-0.44)
f=D['funnel']
# v3(P1-03/P2-02): 首级=分析样本(122.9M; 调查全域为123.5M); 后两级限定可建立基准的家庭
steps=[('US households\n(analysis sample)',f['all'],SEQ[150],''),
       ('Energy insecure\n(any dimension)',f['insecure'],SEQ[300],''),
       ('… and above secure-peer\nmedian (benchmarkable)',f['above'],SEQ[450],''),
       ('… and unhealthy indoor\ntemperature (double burden)',f['db'],SEQ[700],'')]
ys=np.arange(len(steps))[::-1]
for (lab,v,c,note),y in zip(steps,ys):
    rounded_hbar(ax,y,v,0.66,c)
    inside = v>40
    if inside:
        ax.text(v-3,y,f'{v:.1f}M',va='center',ha='right',fontsize=6.6,color='white',fontweight='bold')
    else:
        ax.text(v+3,y,f'{v:.1f}M',va='center',ha='left',fontsize=6.6,color=SEC,fontweight='bold')
ax.set_yticks(ys); ax.set_yticklabels([s[0] for s in steps],fontsize=6.6,color=INK)
ax.set_xlim(0,132); ax.set_ylim(-0.65,len(steps)-0.35)
ax.set_xticks([0,50,100]); ax.set_xlabel('Households (millions)',fontsize=6.7)
ax.xaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True)
style(ax); ax.tick_params(left=False)
ax.set_title('6.5 million households pay more\nand are still too cold or too hot',fontsize=7.5,fontweight='bold',loc='left',pad=8)

# ---------- b. dumbbell ----------
ax=fig.add_subplot(gs[0,1]); letter(ax,'b',-0.42)
chars=sorted(D['chars'],key=lambda c: c['db']-c['nat'],reverse=True)
ys=np.arange(len(chars))[::-1]
for c,y in zip(chars,ys):
    lo,hi=sorted([c['nat'],c['db']])
    ax.plot([lo,hi],[y,y],lw=1.2,color=GRID,zorder=2)
    ax.plot(c['nat'],y,'o',ms=4.2,mfc=NEUT,mec='white',mew=0.8,zorder=3)
    ax.plot(c['db'],y,'o',ms=4.6,mfc=BLUE,mec='white',mew=0.8,zorder=4)
    ax.text(hi+1.6,y,f"{c['db']:.0f}% vs {c['nat']:.0f}%",va='center',fontsize=5.9,color=SEC)
labs={'Poorly/not insulated (reported)':'Poorly/not insulated\n(reported)',
      'Ever received energy assistance':'Ever received\nenergy assistance'}
ax.set_yticks(ys); ax.set_yticklabels([labs.get(c['label'],c['label']) for c in chars],fontsize=6.4,color=INK)
ax.set_xlim(0,80); ax.set_ylim(-0.7,len(chars)-0.3)
ax.set_xticks([0,20,40,60])
ax.set_xlabel('Share of households (%)',fontsize=6.7)
ax.xaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True)
style(ax); ax.tick_params(left=False)
ax.legend(handles=[Line2D([],[],marker='o',ls='',mfc=BLUE,mec='white',ms=4.6,label='Double burden'),
                   Line2D([],[],marker='o',ls='',mfc=NEUT,mec='white',ms=4.2,label='All households')],
          loc='lower right',frameon=False,fontsize=6.0,handletextpad=0.2,borderaxespad=0.2)
ax.set_title('Renters, poorly insulated homes and\nminority householders are over-represented',fontsize=7.5,fontweight='bold',loc='left',pad=8)

# ---------- c. heatmap ----------
ax=fig.add_subplot(gs[1,0]); letter(ax,'c',-0.44)
types=[('mobile','Mobile home'),('apt24','Apartment 2–4 units'),('apt5','Apartment 5+ units'),('sf','Single-family')]
vints=[('pre70','Pre-1970'),('70-99','1970–1999'),('00+','2000–2020')]
# v3(P1-03): heat 单元为 {share, n, coverage_pct}; share=None 表示无可用基准(NA),
# 画灰格标注"no benchmark", 不再显示会被误读为观测零发生率的 0%
vmax=13.0
def seqcol(v):
    ramp=[SEQ[100],SEQ[150],SEQ[200],SEQ[300],SEQ[450],SEQ[550],SEQ[700]]
    idx=int(min(v/vmax,0.999)*len(ramp))
    return ramp[idx]
def lum(h):
    r,g,b=[int(h[i:i+2],16)/255 for i in (1,3,5)]
    return 0.2126*r+0.7152*g+0.0722*b
for i,(t,tl) in enumerate(types):
    for j,(v,vl) in enumerate(vints):
        # v3.1(复查#5): 小样本标记依据可分类样本量 n_covered(share的实际分母), 非整组n
        cell=D['heat'][f'{t}|{v}']; val=cell['share']; n=cell['n_covered']
        if val is None:
            ax.add_patch(plt.Rectangle((j,-i),0.94,0.94,facecolor='#f1f0ec',
                         edgecolor='white',lw=1.5,hatch='///'))
            ax.text(j+0.47,-i+0.47,'NA',ha='center',va='center',fontsize=6.4,color=MUT)
        else:
            c=seqcol(val)
            ax.add_patch(plt.Rectangle((j,-i),0.94,0.94,facecolor=c,edgecolor='white',lw=1.5))
            txt=f'{val:.0f}%' + ('†' if n<100 else '')
            ax.text(j+0.47,-i+0.47,txt,ha='center',va='center',fontsize=6.6,
                    color='white' if lum(c)<0.55 else INK,fontweight='bold')
ax.set_xlim(-0.06,3.2); ax.set_ylim(-3.55,1.15); ax.set_aspect('equal'); ax.axis('off')
for j,(v,vl) in enumerate(vints):
    ax.text(j+0.47,1.02,vl,ha='center',va='bottom',fontsize=6.5,color=SEC)
for i,(t,tl) in enumerate(types):
    ax.text(-0.12,-i+0.47,tl,ha='right',va='center',fontsize=6.5,color=INK)
ax.text(0,-3.22,'Rates are among households with a defined benchmark; \u2020 fewer than 100 such households',fontsize=5.8,color=MUT)
ax.text(0,-3.50,'NA: no benchmark cell (fewer than 10 secure peers); share is undefined, not zero',
        fontsize=5.8,color=MUT)
ax.set_title('Higher observed double-burden rates\nin small apartment buildings',fontsize=7.5,fontweight='bold',loc='left',pad=10)

# ---------- d. burden survival curves ----------
ax=fig.add_subplot(gs[1,1]); letter(ax,'d',-0.42)
series=[('Double burden',BLUE),('Other insecure',ORANGE),('Secure',NEUT)]
for name,c in series:
    s=D['cdf'][name]
    x=np.array(s['x']); y=100-np.array(s['y'])
    ax.plot(x,y,lw=2,color=c,solid_capstyle='round',zorder=3)
ax.axvline(10,color=BASE,lw=0.8,ls=(0,(2,2)),zorder=1)
ax.text(10.5,93,'10% of income',fontsize=5.9,color=MUT)
s=D['cdf']
ax.annotate(f"{s['Double burden']['share10']:.0f}%",xy=(10,s['Double burden']['share10']),xytext=(14.2,40),
            fontsize=6.3,color=SEC,arrowprops=dict(arrowstyle='-',lw=0.6,color=MUT))
ax.annotate(f"{s['Secure']['share10']:.0f}%",xy=(10,s['Secure']['share10']),xytext=(7.0,2.5),
            fontsize=6.3,color=SEC,arrowprops=dict(arrowstyle='-',lw=0.6,color=MUT))
ax.set_xlim(0,25); ax.set_ylim(0,100)
ax.set_xticks([0,5,10,15,20,25]); ax.set_yticks([0,25,50,75,100])
ax.set_xlabel('Energy burden threshold (% of household income)',fontsize=6.7)
ax.set_ylabel('Share of households above threshold (%)',fontsize=6.7)
ax.yaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True)
style(ax); ax.tick_params(left=False)
ax.legend(handles=[Line2D([],[],lw=2,color=c,label=n) for n,c in series],
          loc='upper right',frameon=False,fontsize=6.0,handlelength=1.4,borderaxespad=0)
ax.set_title('Triple the exposure to severe\nenergy burdens',fontsize=7.5,fontweight='bold',loc='left',pad=8)

fig.savefig(OUT/'fig6_double_burden.png',dpi=600,facecolor='white')
fig.savefig(OUT/'fig6_double_burden.pdf',facecolor='white')
print('saved')
