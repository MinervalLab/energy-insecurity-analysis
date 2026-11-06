# -*- coding: utf-8 -*-
"""fig7 — 超基准消费的州级地理 (需先跑 s7)
输出: result/fig7_state_map.png (600dpi) + .pdf"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch
from s0_common import RESULT_DIR, STYLE, set_pub_style

S = STYLE; set_pub_style()
INK,SEC,MUT,GRID,BASE,BLUE = S['INK'],S['SEC'],S['MUT'],S['GRID'],S['BASE'],S['BLUE']
SEQ = S['SEQ']
data = json.load(open(RESULT_DIR/'state_cf.json'))

GRIDMAP={'AK':(0,0),'ME':(11,0),'VT':(10,1),'NH':(11,1),
'WA':(1,2),'ID':(2,2),'MT':(3,2),'ND':(4,2),'MN':(5,2),'WI':(6,2),'MI':(8,2),'NY':(9,2),'MA':(10,2),'RI':(11,2),
'OR':(1,3),'NV':(2,3),'WY':(3,3),'SD':(4,3),'IA':(5,3),'IL':(6,3),'IN':(7,3),'OH':(8,3),'PA':(9,3),'NJ':(10,3),'CT':(11,3),
'CA':(1,4),'UT':(2,4),'CO':(3,4),'NE':(4,4),'MO':(5,4),'KY':(6,4),'WV':(7,4),'VA':(8,4),'MD':(9,4),'DE':(10,4),
'AZ':(2,5),'NM':(3,5),'KS':(4,5),'AR':(5,5),'TN':(6,5),'NC':(7,5),'SC':(8,5),'DC':(9,5),
'OK':(4,6),'LA':(5,6),'MS':(6,6),'AL':(7,6),'GA':(8,6),
'HI':(0,7),'TX':(3,7),'FL':(8,7)}
BINS=[250,350,425,500]; COLS=[SEQ[150],SEQ[300],SEQ[450],SEQ[550],SEQ[700]]
def col(v):
    for b,c in zip(BINS,COLS):
        if v<b: return c
    return COLS[-1]
def lum(h):
    r,g,b=[int(h[i:i+2],16)/255 for i in (1,3,5)]
    return 0.2126*r+0.7152*g+0.0722*b
def style(ax):
    for sp in ['top','right','left']: ax.spines[sp].set_visible(False)
def letter(ax,s,x=-0.04):
    ax.text(x,1.05,s,transform=ax.transAxes,fontsize=8.5,fontweight='bold',ha='right',va='bottom')
def rounded_hbar(ax,y,w,h,color,zorder=3):
    r=min(abs(w)*0.9,h*0.45)
    verts=[(0,y-h/2),(w-r,y-h/2),(w,y-h/2),(w,y-h/2+r),(w,y+h/2-r),(w,y+h/2),(w-r,y+h/2),(0,y+h/2),(0,y-h/2)]
    codes=[MPath.MOVETO,MPath.LINETO,MPath.CURVE3,MPath.CURVE3,MPath.LINETO,MPath.CURVE3,MPath.CURVE3,MPath.LINETO,MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts,codes),facecolor=color,edgecolor='none',zorder=zorder))

fig=plt.figure(figsize=(7.2,5.4))
gs=fig.add_gridspec(2,2,left=0.055,right=0.975,top=0.92,bottom=0.09,
                    hspace=0.55,wspace=0.28,width_ratios=[1.45,1],height_ratios=[1,1])

# ---------- a. 瓦片地图 (占左侧整列) ----------
ax=fig.add_subplot(gs[:,0]); letter(ax,'a',-0.02)
for st,(cx,cy) in GRIDMAP.items():
    v=data[st]['save_per_ins_hh']; c=col(v)
    tc='white' if lum(c)<0.55 else INK
    ax.add_patch(plt.Rectangle((cx,-cy*1.06),0.94,0.94,facecolor=c,edgecolor='white',lw=1.2))
    ax.text(cx+0.47,-cy*1.06+0.60,st,ha='center',va='center',fontsize=5.8,color=tc,fontweight='bold')
    ax.text(cx+0.47,-cy*1.06+0.28,f"{v:.0f}",ha='center',va='center',fontsize=4.9,color=tc)
ax.set_xlim(-0.4,12.4); ax.set_ylim(-10.2,2.1); ax.set_aspect('equal'); ax.axis('off')
labels=['<$250','250–350','350–425','425–500','>$500']
for i,(c,lb) in enumerate(zip(COLS,labels)):
    ax.add_patch(plt.Rectangle((0.2+i*2.45,1.45),0.46,0.34,facecolor=c,edgecolor='none'))
    ax.text(0.2+i*2.45+0.58,1.62,lb,fontsize=5.8,color=SEC,va='center')
# v3(P1-05/06): 措辞改 median-capping scenario(会计情景), 不称 efficiency-gap/savings
_min_n = min(data[s_]['n_ins'] for s_ in data)   # 最小州样本量取自数据(P3-04), 不再硬编码
ax.set_title('Median-capping scenario value per insecure household (US$ per year)',
             fontsize=7.5,fontweight='bold',loc='left',pad=13)
ax.text(-0.4,-9.0,'Tile value: median-capping accounting scenario per insecure household, benchmarked within\n'
        f'income band \u00d7 climate \u00d7 housing type \u00d7 vintage cells. Smallest state cell: n = {_min_n} insecure households\n'
        '(full-pipeline jackknife SEs for all states in the replication materials). Panel c applies assumed 40\u201351% scaling;\n'
        'scenario quantity, not identified realizable savings.',
        fontsize=5.4,color=MUT)

# ---------- b. 优先级象限散点 ----------
ax=fig.add_subplot(gs[0,1]); letter(ax,'b',-0.16)
xs=[data[s_]['ins_rate'] for s_ in data]; ys=[data[s_]['save_per_ins_hh'] for s_ in data]
tot=[data[s_]['total_save_M'] for s_ in data]; hh=[data[s_]['ins_hh_M'] for s_ in data]
mx,my=np.median(xs),np.median(ys)
ax.axvline(mx,color=GRID,lw=0.8); ax.axhline(my,color=GRID,lw=0.8)
for s_,x,y,h in zip(data,xs,ys,hh):
    ax.scatter(x,y,s=18+h*90,facecolor=col(y),edgecolor='white',linewidth=0.8,zorder=3)
top_by_total=sorted(data,key=lambda s_: data[s_]['total_save_M'],reverse=True)[:5]
offsets={'TX':(1.1,-26),'CA':(0.8,-32),'FL':(-3.4,-36),'NY':(-3.4,14),'MI':(0.8,14),'GA':(1.7,4)}
for s_ in top_by_total:
    dx,dy=offsets.get(s_,(0.5,8))
    ax.text(data[s_]['ins_rate']+dx,data[s_]['save_per_ins_hh']+dy,s_,
            fontsize=5.9,color=SEC,fontweight='bold')
ax.text(0.97,0.95,'high need ×\nhigh potential',transform=ax.transAxes,ha='right',va='top',
        fontsize=5.6,color=MUT,style='italic')
ax.set_xlabel('Energy-insecurity rate (%)',fontsize=6.7)
ax.set_ylabel('Scenario value per insecure household ($/yr)',fontsize=6.7)
ax.set_xlim(12,42); ax.set_ylim(80,600)
style(ax); ax.tick_params(left=False)
ax.yaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True)
ax.text(0.03,0.03,'bubble area increases with insecure households',transform=ax.transAxes,fontsize=5.4,color=MUT,style='italic')
ax.set_title('Where need and potential coincide',fontsize=7.5,fontweight='bold',loc='left',pad=6)

# ---------- c. 州总额 Top 10: 核算上限 + 实证实现率区间(40-51%) ----------
# 区间依据: 40% = Fowlie et al. (QJE 2018); 51% = Christensen et al. (REStat 2023)
ax=fig.add_subplot(gs[1,1]); letter(ax,'c',-0.16)
R_LO, R_HI = 0.40, 0.51
top10=sorted(data,key=lambda s_: data[s_]['total_save_M'],reverse=True)[:10]
ys_=np.arange(10)[::-1]
from matplotlib.lines import Line2D
C_UB, C_BAND = SEQ[150], SEQ[550]     # 统一色: 浅蓝=核算上限, 深蓝=实证实现率段(与图例一致)
for s_,y in zip(top10,ys_):
    v=data[s_]['total_save_M']
    rounded_hbar(ax,y,v,0.60,C_UB)
    ax.add_patch(plt.Rectangle((R_LO*v,y-0.30),(R_HI-R_LO)*v,0.60,
                 facecolor=C_BAND,edgecolor='white',lw=0.8,zorder=4))
    ax.text(v+18,y,f'${R_LO*v:,.0f}\u2013{R_HI*v:,.0f}M',va='center',fontsize=5.8,color=SEC)
ax.set_yticks(ys_); ax.set_yticklabels(top10,fontsize=6.4,color=INK)
ax.set_xlim(0,max(data[s_]['total_save_M'] for s_ in top10)*1.34)
ax.set_ylim(-0.6,9.9)
ax.set_xlabel('Annual bill value ($ million)',fontsize=6.7)
ax.xaxis.grid(True,color=GRID,lw=0.5); ax.set_axisbelow(True)
style(ax); ax.tick_params(left=False)
ax.legend(handles=[
    Line2D([],[],marker='s',ls='',mfc=SEQ[150],mec='none',ms=6,label='Median-capping scenario (accounting)'),
    Line2D([],[],marker='s',ls='',mfc=SEQ[550],mec='none',ms=6,label='At assumed 40\u201351% scaling')],
    loc='lower right',frameon=False,fontsize=5.7,handletextpad=0.3,borderaxespad=0)
ax.set_title('Largest state totals, with assumed\n40\u201351% scaling band',fontsize=7.5,fontweight='bold',loc='left',pad=6)

fig.savefig(RESULT_DIR/'fig7_state_map.png',dpi=600,facecolor='white')
fig.savefig(RESULT_DIR/'fig7_state_map.pdf',facecolor='white')
print('saved fig7_state_map ->',RESULT_DIR)
