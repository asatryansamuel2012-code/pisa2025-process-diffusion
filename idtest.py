import sys; import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); from core import *
rng=np.random.default_rng(1); nC,nPc,nI,J=9,60,40,20; nP=nC*nPc
pp=np.repeat(np.arange(nP),J); ii=np.concatenate([rng.choice(nI,J,replace=False) for _ in range(nP)]); cp=np.repeat(np.arange(nC),nPc)
ai=rng.normal(size=nP); aiobs=rng.random(nP)>0.3
d=dict(ii=jnp.array(ii),pp=jnp.array(pp),cp=jnp.array(cp),ai=jnp.array(np.where(aiobs,ai,0.),jnp.float32),aiobs=jnp.array(aiobs),esc=jnp.array(rng.normal(size=nP),jnp.float32),fem=jnp.array((rng.random(nP)<.5)*1.,jnp.float32),
       mint=jnp.full(nI,0.5),mindec=jnp.full(nI,0.2))
G=np.zeros((5,3)); G[:,0]=[-0.15,0.08,0.15,0.25,0.08]
Lt=np.linalg.cholesky(np.array([[1,.3,-.2,-.3,.2],[.3,1,.1,0,0],[-.2,.1,1,-.3,0],[-.3,0,-.3,1,0],[.2,0,0,0,1.]]))
pm=dict(mu_free=rng.normal(0,.2,(nC-1,5)),Lcorr=Lt,sd=np.array([.5,.3,.6,.8,.5]),z=rng.normal(size=(nP,5)),b=rng.normal(0,.7,nI),la=rng.normal(1,.2,nI),r=np.zeros(nI),r0=-np.ones(nI),
        oc=np.full(nI,2.),eps=rng.normal(.7,.2,nI),nu=np.full(nI,-3.),la_b=np.zeros(nI),lb_b=np.zeros(nI),sc=.5,se=.5,g=.25,kap0=-3.,mur_raw=1.,sr_raw=0.,lam_na=1.,
        Gam=G,mai=np.zeros(nC),sai=1.,ai_z=np.where(aiobs,0.,ai),zeta=np.zeros(nC),zw=np.array([-.5,0.]))
x,t,f,E=simulate(pm,jax.random.PRNGKey(0),d)
print('acc',x.mean().round(3),'median t (s)',np.median(t)*10,'hasF',np.mean(~np.isnan(f)).round(3))
dd=design_update(d,ii,x,t,f)
s,Ls,rt=fit(dd,nP,nI,nC,int(sys.argv[1]),100)
sd,C,Gs=derived(s); Eh=np.asarray(jax.lax.map(lambda j: eta_of(pick(s,j),dd),jnp.arange(100))).mean(0)
print('time',round(rt),'loss',np.nanmean(Ls[-200:]).round(1))
print('sd true',pm['sd'],'est',sd.mean(0).round(3))
print('Gam AI true',G[:,0],'est',s['Gam'][:,:,0].mean(0).round(3),'q05',np.quantile(s['Gam'][:,:,0],.05,0).round(3),'q95',np.quantile(s['Gam'][:,:,0],.95,0).round(3))
print('person r',[round(np.corrcoef(E[:,k],Eh[:,k])[0,1],3) for k in range(5)])
print('item r b,la,eps',[round(np.corrcoef(pm[k],s[k].mean(0))[0,1],3) for k in ['b','la','eps']])
print('zw',s['zw'].mean(0).round(3),'g',s['g'].mean().round(3),'rapid s',(10*np.exp(np.log(BMAX)-np.log1p(np.exp(s['mur_raw'])))).mean().round(2),'kap0',s['kap0'].mean().round(2))
