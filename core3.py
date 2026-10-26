import os, time, json, numpy as np, pandas as pd, jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist, optax
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoLowRankMultivariateNormal, AutoNormal
EPS=0.01; L10=float(np.log(10.)); K5=3; PN=['drift','log_caution','encoding_speed']; BMAX=3.0   # rapid-guess mean <= 30 s
def wfpt(t,v,a,w=0.5):
    u=t/a**2; k=jnp.arange(1,21.)
    sl=jnp.sum(k*jnp.sin(k*jnp.pi*w)*jnp.exp(-(k**2-1)*jnp.pi**2*u[...,None]/2),-1)
    large=jnp.log(jnp.pi)+jnp.log(jnp.maximum(sl,1e-20))-jnp.pi**2*u/2
    ww=w+2*jnp.arange(-5,6.); ss=jnp.sum(ww*jnp.exp(-(ww**2-w**2)/(2*u[...,None])),-1)
    small=-0.5*jnp.log(2*jnp.pi*u**3)+jnp.log(jnp.maximum(ss,1e-20))-w**2/(2*u)
    return jnp.where(u<0.5,small,large)-2*jnp.log(a)-v*a*w-v**2*t/2
def wsurv(t,v,a):
    k=jnp.arange(1,151.); sk=k*jnp.sin(k*jnp.pi/2); lam=v[...,None]**2/2+k**2*jnp.pi**2/(2*a[...,None]**2)
    s=jnp.sum(sk/lam*jnp.exp(-(lam-lam[...,:1])*t[...,None]),-1)
    ls=jnp.log(2*jnp.pi/a**2)+(jnp.abs(v*a/2)+jnp.log1p(jnp.exp(-jnp.abs(v*a)))-jnp.log(2.))+jnp.log(jnp.maximum(s,1e-30))-lam[...,0]*t
    return jnp.where(t/a**2<0.01,0.,jnp.minimum(ls,0.))
def Mmat(p): return p['sd'][:,None]*p['Lcorr']
def eta_of(p,d):
    mu=jnp.concatenate([jnp.zeros((1,K5)),p['mu_free']],0)[d['cp']]
    aif=jnp.where(d['aiobs'],d['ai'],p['mai'][d['cp']]+p['sai']*p['ai_z'])
    return mu+jnp.stack([aif,d['esc'],d['fem']],1)@p['Gam'].T+p['z']@Mmat(p).T
def parts(p,d):
    e=eta_of(p,d); i,q=d['ii'],d['pp']
    v=jnp.clip(e[q,0]-p['b'][i],-8.,8.); a=jnp.exp(jnp.clip(p['la'][i]+e[q,1],-3.,4.))
    ter0=d['mindec'][i]*jax.nn.sigmoid(p['r0'][i])*0.99; teri=d['mint'][i]*jax.nn.sigmoid(p['r'][i])*0.99
    mur=jnp.log(BMAX)-jax.nn.softplus(p['mur_raw']); sr=0.1+0.9*jax.nn.sigmoid(p['sr_raw'])
    return e,v,a,ter0,teri,mur,sr
def LL(p,d):
    e,v,a,ter0,teri,mur,sr=parts(p,d); i,q=d['ii'],d['pp']; t=d['ts']; hf=d['hasf']
    td=jnp.where(hf,jnp.maximum(t-d['fs']-ter0,1e-4),jnp.maximum(t-teri,1e-4))
    lw=jnp.where(d['x']>0.5,wfpt(td,-v,a),wfpt(td,v,a))
    zc=(d['lts']-L10-p['pc'][d['cp'][q]]-p['oc'][i])/p['sc']; lfc=dist.Normal(0.,1.).log_prob(zc)-jnp.log(p['sc'])-(d['lts']-L10); lSc=jax.scipy.special.log_ndtr(-zc)
    lg=dist.Bernoulli(probs=p['g']).log_prob(d['x'])
    race=jnp.logaddexp(lw+lSc,lfc+wsurv(td,v,a)+lg)
    race=jnp.logaddexp(jnp.log(1-EPS)+race,jnp.log(EPS)+jnp.log(0.5)-jnp.log(d['tmax'][i]))-L10
    lenc=dist.LogNormal(p['eps'][i]-e[q,2],p['se']).log_prob(jnp.maximum(d['fs'],1e-4))-L10
    fgen=dist.Beta(jnp.exp(p['la_b'][i]),jnp.exp(p['lb_b'][i])).log_prob(d['r'])-d['lts']
    rg=lg+dist.LogNormal(mur,sr).log_prob(t)-L10; k=p['kap0']+jnp.concatenate([jnp.zeros(1),p['kc_free']])[d['cp'][q]]
    mix=jnp.logaddexp(jax.nn.log_sigmoid(-k)+race+jnp.where(hf,lenc,0.),jax.nn.log_sigmoid(k)+rg+jnp.where(hf,fgen,0.))
    return mix+dist.Bernoulli(logits=p['nu'][i]+jnp.concatenate([jnp.zeros(1),p['nc_free']])[d['cp'][q]]).log_prob(d['noact'])
def make_model(nP,nI,nC):
    def model(d):
        S=numpyro.sample; p={}
        p['mu_free']=S('mu_free',dist.Normal(0.,1.).expand([nC-1,K5])); p['Lcorr']=S('Lcorr',dist.LKJCholesky(K5,4.))
        p['sd']=S('sd',dist.HalfNormal(1.).expand([K5])); p['z']=S('z',dist.Normal(0.,1.).expand([nP,K5]))
        for k,pr in [('b',dist.Normal(0.,2.)),('la',dist.Normal(1.,1.)),('r',dist.Normal(0.,1.5)),('r0',dist.Normal(-1.,1.5)),('oc',dist.Normal(1.5,1.)),
                     ('eps',dist.Normal(1.,1.)),('nu',dist.Normal(-2.,2.)),('la_b',dist.Normal(0.,1.)),('lb_b',dist.Normal(0.,1.))]: p[k]=S(k,pr.expand([nI]))
        p['sc']=S('sc',dist.HalfNormal(1.)); p['se']=S('se',dist.HalfNormal(1.)); p['g']=S('g',dist.Beta(2.,6.))
        p['kap0']=S('kap0',dist.Normal(-3.,1.)); p['mur_raw']=S('mur_raw',dist.Normal(1.,1.)); p['sr_raw']=S('sr_raw',dist.Normal(0.,1.))
        p['pc']=S('pc',dist.Normal(0.,1.).expand([nC])); p['kc_free']=S('kc_free',dist.Normal(0.,1.).expand([nC-1])); p['nc_free']=S('nc_free',dist.Normal(0.,1.).expand([nC-1])); p['Gam']=S('Gam',dist.Normal(0.,1.).expand([K5,3]))
        p['mai']=S('mai',dist.Normal(0.,1.).expand([nC])); p['sai']=S('sai',dist.HalfNormal(1.)); p['ai_z']=S('ai_z',dist.Normal(0.,1.).expand([nP]))
        p['zeta']=S('zeta',dist.Normal(0.,2.).expand([nC])); p['zw']=S('zw',dist.Normal(0.,1.).expand([2]))
        numpyro.factor('ai_obs',jnp.where(d['aiobs'],dist.Normal(p['mai'][d['cp']],p['sai']).log_prob(d['ai']),0.).sum())
        e=eta_of(p,d); numpyro.factor('ai_miss',dist.Bernoulli(logits=p['zeta'][d['cp']]+p['zw'][0]*e[:,1]+p['zw'][1]*e[:,0]).log_prob((~d['aiobs']).astype(jnp.float32)).sum())
        numpyro.factor('obs',LL(p,d).sum())
    return model
import pickle
def fit(dd,nP,nI,nC,steps,ndraw,seed=0,lr=0.005,rank=10,ckpt=None,chunk=500):
    """SVI with a low-rank multivariate-normal guide; checkpoints to Drive every `chunk` steps and resumes automatically."""
    model=make_model(nP,nI,nC); guide=(AutoNormal(model,init_scale=0.05) if rank==0 else AutoLowRankMultivariateNormal(model,rank=rank,init_scale=0.05))
    opt=numpyro.optim.optax_to_numpyro(optax.chain(optax.clip_by_global_norm(10.),optax.adam(optax.exponential_decay(lr,steps//4,0.5,end_value=lr/20))))
    svi=SVI(model,guide,opt,Trace_ELBO()); state=svi.init(jax.random.PRNGKey(seed),dd); done=0; Ls=[]
    if ckpt and os.path.exists(ckpt):
        st,done,Ls=pickle.load(open(ckpt,'rb')); state=jax.tree_util.tree_map(jnp.asarray,st); print('resumed at step',done)
    @jax.jit
    def run_chunk(state):
        def body(c,_):
            st,l=svi.stable_update(c,dd); return st,l
        return jax.lax.scan(body,state,None,length=chunk)
    t0=time.time()
    while done<steps:
        state,l=run_chunk(state); l=np.asarray(l); Ls+=l.tolist(); done+=chunk
        el=time.time()-t0; print(f'step {done}/{steps}  loss {np.nanmean(l):.1f}  elapsed {el/60:.1f} min',flush=True)
        if ckpt: pickle.dump((jax.device_get(state),done,Ls),open(ckpt+'.tmp','wb')); os.replace(ckpt+'.tmp',ckpt)
    params=svi.get_params(state); Ls=np.asarray(Ls)
    s=Predictive(guide,params=params,num_samples=ndraw)(jax.random.PRNGKey(seed+1),dd)
    s={k:np.asarray(v) for k,v in s.items() if k not in('obs','ai_obs','ai_miss')}
    return s,Ls,time.time()-t0
pick=lambda s,j: {k:jnp.asarray(v)[j] for k,v in s.items()}
def derived(s):   # person-level SDs, correlations, standardized AI effects (per SD of each person parameter)
    sd=s['sd']; L=s['Lcorr']; C=np.einsum('nij,nkj->nik',L,L)
    Gs=s['Gam'][:,:,0]/sd; return sd,C,Gs
def simulate(pm,key,d,dt=0.01,T=60.):
    p={k:jnp.asarray(v) for k,v in pm.items()}; e,v,a,ter0,teri,mur,sr=parts(p,d); i,q=d['ii'],d['pp']; N=len(i); ks=jax.random.split(key,8)
    dis=jax.random.uniform(ks[0],(N,))<jax.nn.sigmoid(p['kap0']+jnp.concatenate([jnp.zeros(1),p['kc_free']])[d['cp'][q]])
    act=jax.random.uniform(ks[1],(N,))>jax.nn.sigmoid(p['nu'][i]+jnp.concatenate([jnp.zeros(1),p['nc_free']])[d['cp'][q]])
    f=jnp.exp(p['eps'][i]-e[q,2]+p['se']*jax.random.normal(ks[2],(N,)))
    def step(c,kk):
        X,done,tt,up=c; X=X+jnp.where(done,0.,v*dt+jnp.sqrt(dt)*jax.random.normal(kk,(N,)))
        hit=(~done)&((X>=a)|(X<=0)); return (X,done|hit,jnp.where(hit,tt,tt+dt*(~done)),jnp.where(hit,X>=a,up)),None
    (X,done,tdd,up),_=jax.lax.scan(step,(a/2,jnp.zeros(N,bool),jnp.zeros(N),jnp.zeros(N,bool)),jax.random.split(ks[3],int(T/dt)))
    tdd=jnp.where(done,tdd,T); tdec=jnp.where(act,f+ter0,teri)+tdd
    Tc=jnp.exp(p['pc'][d['cp'][q]]+p['oc'][i]+p['sc']*jax.random.normal(ks[4],(N,))); gx=(jax.random.uniform(ks[5],(N,))<p['g']).astype(jnp.float32)
    cens=Tc<tdec; t_eng=jnp.where(cens,Tc,tdec); x_eng=jnp.where(cens,gx,up.astype(jnp.float32))
    t_dis=jnp.exp(mur+sr*jax.random.normal(ks[6],(N,))); rr=jax.random.beta(ks[7],jnp.exp(p['la_b'][i]),jnp.exp(p['lb_b'][i]))
    t=jnp.clip(jnp.where(dis,t_dis,t_eng),1e-3,None); x=jnp.where(dis,gx,x_eng); fsim=jnp.where(dis,rr*t,f); act=act&(fsim<t)
    return np.asarray(x),np.asarray(t),np.asarray(jnp.where(act,fsim,np.nan)),np.asarray(e)
def design_update(d,ii,x,t,f):   # rebuild data-dependent arrays (t,f in 10-s units)
    hs=~np.isnan(f); fs=np.where(hs,f,0.); g=pd.Series(t).groupby(ii); dec=np.where(hs,t-fs,np.nan)
    dd=dict(d); dd.update(x=jnp.array(x,jnp.float32),ts=jnp.array(t,jnp.float32),lts=jnp.array(np.log(t*10),jnp.float32),
        mint=jnp.array(g.min().values,jnp.float32),tmax=jnp.array(g.max().values,jnp.float32),mindec=jnp.array(pd.Series(dec).groupby(ii).min().fillna(0).values,jnp.float32),
        fs=jnp.array(fs,jnp.float32),hasf=jnp.array(hs),noact=jnp.array(~hs,jnp.float32),r=jnp.array(np.clip(np.where(hs,fs/t,.5),1e-4,1-1e-4),jnp.float32))
    return dd
