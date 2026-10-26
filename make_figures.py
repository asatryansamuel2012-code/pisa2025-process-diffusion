import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9})
# Fig 1 schematic
fig,ax=plt.subplots(figsize=(6.3,3.0)); ax.axis('off'); ax.set_xlim(0,10); ax.set_ylim(0,5)
def box(x,y,w,h,t,c='#eef2f7'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.05',fc=c,ec='#333',lw=0.8)); ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=7.5)
def arr(x1,y1,x2,y2,t=''):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',lw=0.8)); 
    if t: ax.text((x1+x2)/2,(y1+y2)/2+0.12,t,fontsize=6.5,ha='center')
box(0.1,2.1,1.5,0.8,'Item onset')
box(2.2,3.3,2.4,1.1,'Engaged (1-$\\pi$)\nEncoding stage\n$F\\sim$LogNormal($\\epsilon_i-\\omega_p,\\sigma_F$)')
box(2.2,0.4,2.4,1.1,'Disengaged ($\\pi$)\nrapid guess\n$T\\sim$LogNormal, $X\\sim$Bern($g$)','#fdf1e6')
box(5.2,3.3,2.3,1.1,'Decision stage\nWiener diffusion on $T-F$\ndrift $v$, boundary $a$')
box(5.2,1.9,2.3,0.9,'Persistence clock\n$T_c\\sim$LogNormal','#eef7ee')
box(8.1,2.6,1.8,1.1,'Observed\n$(X,T,F,$ action\nstatus$)$','#f3f3f3')
arr(1.6,2.7,2.2,3.7); arr(1.6,2.3,2.2,1.0); arr(4.6,3.85,5.2,3.85); arr(7.5,3.85,8.1,3.3); arr(7.5,2.35,8.1,2.9); ax.text(7.8,4.0,'boundary hit',fontsize=6.3,ha='center'); ax.text(7.8,1.85,'censored: guess',fontsize=6.3,ha='center')
arr(4.6,0.95,8.1,2.6)
ax.text(5.0,4.75,'race: response = first of {boundary hit, persistence time}',fontsize=6.8,ha='center',style='italic')
plt.tight_layout(); plt.savefig('fig/fig1.png',dpi=300); plt.close()
# Fig 2 Armenia vs others (mathematics)
C=['ARM','KAZ','MDA','LVA','LTU','EST','JPN','ARE','SGP']
rap=[.0948,.0417,.0322,.0525,.0368,.0232,.0354,.0716,.0309]; noa=[.1233,.0130,.0243,.0188,.0154,.0122,.0182,.0186,.0075]; nr=[.0486,.0103,.0134,.0087,.0067,.0119,.0093,.0089,.0069]; acc=[.3199,.3912,.3894,.4811,.5098,.5634,.6004,.4508,.6509]
fig,axs=plt.subplots(1,2,figsize=(6.3,2.6)); x=np.arange(9); w=0.27
axs[0].bar(x-w,np.array(rap)*100,w,label='rapid guessing',color='#4c72b0'); axs[0].bar(x,np.array(noa)*100,w,label='no action',color='#dd8452'); axs[0].bar(x+w,np.array(nr)*100,w,label='not reached',color='#55a868')
axs[0].set_xticks(x); axs[0].set_xticklabels(C,rotation=45); axs[0].set_ylabel('% of item responses'); axs[0].legend(fontsize=6.5,frameon=False); axs[0].set_title('(a) disengagement indicators',fontsize=8)
axs[1].bar(x,acc,color=['#c44e52']+['#8c8c8c']*8); axs[1].set_xticks(x); axs[1].set_xticklabels(C,rotation=45); axs[1].set_ylabel('proportion correct'); axs[1].set_title('(b) accuracy',fontsize=8)
plt.tight_layout(); plt.savefig('fig/fig2.png',dpi=300); plt.close()
# Fig 3 model comparison
M=['M6 (proposed)','M6b','M6a','M3 Wang & Xu','M5 Ranger et al.','M1 van der Linden','M4 diffusion IRT']; dE=[0,-7641.2,-8560.7,-9598.8,-16398.5,-37253.5,-44432.8]; se=[0,217.3,686.2,689.2,694.8,775.5,755.2]
fig,ax=plt.subplots(figsize=(6.3,2.4)); y=np.arange(7)[::-1]
ax.errorbar(np.array(dE)/1000,y,xerr=2*np.array(se)/1000,fmt='o',color='#333',capsize=2,ms=4); ax.set_yticks(y); ax.set_yticklabels(M); ax.axvline(0,lw=0.6,ls='--',color='grey')
ax.set_xlabel('$\\Delta$ELPD$_{loo}$ relative to M6 (thousands; bars = $\\pm$2 SE)'); plt.tight_layout(); plt.savefig('fig/fig3.png',dpi=300); plt.close()
# Fig 4 forest plot (standardized beta, 95% CI)
O=['accuracy','rapid-guess rate','no-action rate','not-reached rate','log time to first action','log total time']
D={('math','pooled'):[(-.0141,-2.4142),(-.0168,-2.7033),(-.0019,-.3080),(-.0118,-1.6830),(.0139,1.8879),(.0171,2.8173)],
   ('math','ARM'):[(-.0081,-.2079),(.0235,.6924),(.0359,1.1258),(.0769,1.4158),(.0242,.6635),(.0351,.8405)],
   ('read','pooled'):[(.0143,2.3707),(-.0281,-4.8674),(-.0067,-.9262),(.0018,.3686),(.0298,3.0699),(.0239,3.3844)],
   ('read','ARM'):[(.0655,1.8229),(-.0072,-.1943),(-.0024,-.0527),(.0160,.5331),(.0008,.0198),(.0008,.0200)]}
fig,axs=plt.subplots(1,2,figsize=(6.3,2.8),sharey=True); y=np.arange(6)[::-1]
for a,dom,tt in [(axs[0],'math','(a) mathematics'),(axs[1],'read','(b) reading')]:
    for k,(sc,off,col) in enumerate([('pooled',0.15,'#333'),('ARM',-0.15,'#c44e52')]):
        b=np.array([v[0] for v in D[(dom,sc)]]); z=np.array([v[1] for v in D[(dom,sc)]]); s=np.abs(b/z)
        a.errorbar(b,y+off,xerr=1.96*s,fmt='o' if sc=='pooled' else 's',color=col,ms=3.5,capsize=2,label='9 countries' if sc=='pooled' else 'Armenia')
    a.axvline(0,lw=0.6,ls='--',color='grey'); a.set_title(tt,fontsize=8)
axs[0].set_yticks(y); axs[0].set_yticklabels(O); axs[1].legend(fontsize=6.5,frameon=False); fig.supxlabel('standardized $\\beta$ of AIUSESCH (95% BRR CI)',fontsize=8)
plt.tight_layout(); plt.savefig('fig/fig4.png',dpi=300); plt.close()
print('ok')
