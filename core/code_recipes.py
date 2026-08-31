"""Laboratuvar sekmeleri için bağımsız Python ve Colab kod çıktıları."""

from __future__ import annotations

from dataclasses import dataclass
from json import dumps
from textwrap import dedent


@dataclass(frozen=True)
class CodeRecipe:
    topic_key: str
    section_key: str
    section_title: str
    python_code: str
    filename_stem: str


TOPIC_SECTIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "konu01": (("mekanizma", "Mekanizma"), ("geometri", "OLS geometrisi"), ("veri", "Veri laboratuvarı"), ("cikti", "Çıktı okuma")),
    "konu02": (("fwl", "FWL"), ("cikarim", "Güvenilir çıkarım"), ("bicim", "Fonksiyonel biçim"), ("etkili", "Etkili gözlem")),
    "konu03": (("zincir", "Tanımlama zinciri"), ("icsellik", "İçsellik"), ("atama", "Rastgele atama"), ("cikti", "Çıktı okuma")),
    "konu04": (("gecerlilik", "Geçerlilik"), ("wald", "Wald oranı"), ("iki_asama", "2SLS zinciri"), ("zayif", "Zayıf araç")),
    "konu05": (("egriler", "Olasılık eğrileri"), ("etkiler", "Marjinal etkiler"), ("veri", "CPS laboratuvarı"), ("secim", "Model seçimi")),
    "konu06": (("mekanizma", "Veri mekanizması"), ("hedefler", "Tobit tahmin hedefleri"), ("veri", "Veri laboratuvarı"), ("secim", "Örneklem seçimi")),
    "konu07": (("kayip", "Kantil kaybı"), ("hedef", "Aynı veri, farklı hedef"), ("profil", "Katsayı profili"), ("veri", "CPS laboratuvarı")),
    "konu08": (("agirlik", "Çekirdek ağırlıkları"), ("yerel", "Yerel regresyon"), ("dogrulama", "Çapraz doğrulama ve artıklaştırma"), ("veri", "DDK laboratuvarı")),
    "konu09": (("kesin", "Kesin RDD"), ("duyarlilik", "Bant genişliği duyarlılığı"), ("bulanik", "Kesin ve bulanık RDD"), ("tanilar", "Tanılar ve LM2007")),
    "konu10": (("ornekleme", "Yeniden örnekleme"), ("dagilim", "Yeniden örnekleme dağılımı"), ("aralik", "Güven aralıkları"), ("veri", "CPS laboratuvarı")),
    "konu11": (("yanlilik", "Yanlılık-varyans"), ("ceza", "Ridge ve Lasso yolu"), ("dogrulama", "Çapraz doğrulama ve Post-Lasso"), ("veri", "CPS laboratuvarı")),
    "konu12": (("secim", "Seçim stratejileri"), ("capraz", "Çapraz uyarlama"), ("duyarlilik", "Bölünme duyarlılığı"), ("veri", "DDK ve araştırma akışı")),
}


COMMON = '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

TOHUM = 807
rng = np.random.default_rng(TOHUM)
'''


TOPIC_PROGRAMS: dict[str, str] = {
    "konu01": COMMON + r'''
def veri_uret(n=800):
    egitim = rng.normal(13, 2.2, n)
    yetenek = rng.normal(size=n)
    deneyim = np.maximum(rng.normal(15, 7, n), 0)
    log_ucret = 1.4 + .08*egitim + .025*deneyim + .35*yetenek + rng.normal(0, .3, n)
    return pd.DataFrame({"log_ucret": log_ucret, "egitim": egitim, "deneyim": deneyim, "yetenek": yetenek})

def mekanizma():
    d = veri_uret(); x = np.linspace(7, 20, 100)
    m = sm.OLS(d.log_ucret, sm.add_constant(d[["egitim"]])).fit()
    plt.scatter(d.egitim, d.log_ucret, s=9, alpha=.25); plt.plot(x, m.params.iloc[0] + m.params.iloc[1]*x, color="crimson")
    plt.xlabel("Eğitim"); plt.ylabel("Log ücret"); plt.show(); print(m.params)

def geometri():
    d = veri_uret(); m = sm.OLS(d.log_ucret, sm.add_constant(d[["egitim", "deneyim"]])).fit()
    print("Artık toplamı:", m.resid.sum()); print("X ile artık iç çarpımları:\n", d[["egitim", "deneyim"]].T @ m.resid)

def veri():
    yol = Path("cps09mar_ikt807.csv")
    d = pd.read_csv(yol) if yol.exists() else veri_uret()
    sonuc = "lwage" if "lwage" in d else "log_ucret"; aciklayici = [x for x in ("education", "experience", "egitim", "deneyim") if x in d]
    print(sm.OLS(d[sonuc], sm.add_constant(d[aciklayici])).fit(cov_type="HC1").summary())

def cikti():
    d = veri_uret(); m = sm.OLS(d.log_ucret, sm.add_constant(d[["egitim", "deneyim"]])).fit(cov_type="HC1")
    print(pd.DataFrame({"katsayi": m.params, "standart_hata": m.bse, "p_degeri": m.pvalues, "ga_alt": m.conf_int()[0], "ga_ust": m.conf_int()[1]}))

SECTIONS = {"mekanizma": mekanizma, "geometri": geometri, "veri": veri, "cikti": cikti}
''',
    "konu02": COMMON + r'''
def veri_uret(n=700):
    okul = rng.integers(0, 35, n); egitim = rng.normal(13, 2, n); deneyim = rng.normal(15, 6, n); kadin = rng.binomial(1, .48, n)
    hata = rng.normal(0, .18 + .02*np.abs(egitim-13), n) + rng.normal(0, .12, 35)[okul]
    y = 1.5 + .08*egitim + .025*deneyim - .12*kadin + hata
    return pd.DataFrame({"y":y,"egitim":egitim,"deneyim":deneyim,"kadin":kadin,"okul":okul})

def fwl():
    d=veri_uret(); y_art=sm.OLS(d.y,sm.add_constant(d[["deneyim","kadin"]])).fit().resid; x_art=sm.OLS(d.egitim,sm.add_constant(d[["deneyim","kadin"]])).fit().resid
    tam=sm.OLS(d.y,sm.add_constant(d[["egitim","deneyim","kadin"]])).fit().params["egitim"]; kismi=sm.OLS(y_art,x_art).fit().params.iloc[0]
    print({"çoklu_OLS":tam,"FWL":kismi})

def cikarim():
    d=veri_uret(); x=sm.add_constant(d[["egitim","deneyim","kadin"]]); m=sm.OLS(d.y,x).fit()
    print(pd.DataFrame({"klasik":m.bse,"HC1":m.get_robustcov_results("HC1").bse,"kümeli":m.get_robustcov_results("cluster",groups=d.okul).bse},index=m.params.index))

def bicim():
    d=veri_uret(); d["egitim2"]=d.egitim**2; d["egitim_kadin"]=d.egitim*d.kadin
    print(sm.OLS(d.y,sm.add_constant(d[["egitim","egitim2","kadin","egitim_kadin","deneyim"]])).fit(cov_type="HC1").summary())

def etkili():
    d=veri_uret(); m=sm.OLS(d.y,sm.add_constant(d[["egitim","deneyim","kadin"]])).fit(); t=m.get_influence().summary_frame()
    print(t.nlargest(10,"cooks_d")[["hat_diag","student_resid","cooks_d"]])

SECTIONS={"fwl":fwl,"cikarim":cikarim,"bicim":bicim,"etkili":etkili}
''',
    "konu03": COMMON + r'''
def zincir():
    adimlar=[("Tahmin hedefi","Anakütlede öğrenilecek büyüklük"),("Tanımlama","Hedefi gözlenen dağılıma bağlayan varsayımlar"),("Tahmin edici","Veriden hesaplama kuralı"),("Tahmin","Örneklemde bulunan sayı")]
    print(pd.DataFrame(adimlar,columns=["Aşama","Soru"]))

def icsellik():
    for n in (300,1000,5000):
        u=rng.normal(size=n); x=.7*u+rng.normal(size=n); y=2+1.5*x+u
        print(n, sm.OLS(y,sm.add_constant(x)).fit().params[1])

def atama():
    okul=np.repeat(np.arange(60),20); tedavi=np.repeat(rng.binomial(1,.5,60),20); taban=rng.normal(size=len(okul)); y=.3*tedavi+.5*taban+rng.normal(size=len(okul))+rng.normal(0,.3,60)[okul]
    m=sm.OLS(y,sm.add_constant(np.column_stack([tedavi,taban]))).fit(cov_type="cluster",cov_kwds={"groups":okul})
    print("Tedavi etkisi ve okul-kümeli SH:",m.params[1],m.bse[1])

def cikti():
    print("Okuma sırası: tahmin hedefi -> atama/seçim -> tanımlama -> çıkarım birimi -> duyarlılık")
    atama()

SECTIONS={"zincir":zincir,"icsellik":icsellik,"atama":atama,"cikti":cikti}
''',
    "konu04": COMMON + r'''
def veri_uret(n=1200,guc=.45):
    z=rng.binomial(1,.5,n); x=guc*z+rng.normal(size=n); u=rng.normal(size=n); d=.8*x+.7*u+rng.normal(size=n); y=1.2*d+.5*x+u
    return pd.DataFrame({"y":y,"d":d,"z":z,"x":x})

def gecerlilik():
    print(pd.DataFrame({"Koşul":["Uygunluk","Dışsallık","Dışlama"],"Soru":["Z, D'yi değiştiriyor mu?","Z, yapısal hatadan bağımsız mı?","Z, Y'yi yalnız D üzerinden mi etkiliyor?"]}))

def wald():
    d=veri_uret(); rf=d.groupby("z").y.mean().diff().iloc[-1]; fs=d.groupby("z").d.mean().diff().iloc[-1]
    print({"indirgenmiş_biçim":rf,"ilk_aşama":fs,"Wald":rf/fs})

def iki_asama():
    d=veri_uret(); ilk=sm.OLS(d.d,sm.add_constant(d[["z","x"]])).fit(); d_hat=ilk.fittedvalues
    ikinci=sm.OLS(d.y,sm.add_constant(pd.DataFrame({"d_hat":d_hat,"x":d.x}))).fit(cov_type="HC1")
    print(ilk.summary()); print(ikinci.summary())

def zayif():
    for guc in (.05,.15,.45):
        tahmin=[]
        for _ in range(100):
            d=veri_uret(400,guc); tahmin.append(d.groupby("z").y.mean().diff().iloc[-1]/d.groupby("z").d.mean().diff().iloc[-1])
        print(guc,np.nanmedian(tahmin),np.nanquantile(tahmin,[.05,.95]))

SECTIONS={"gecerlilik":gecerlilik,"wald":wald,"iki_asama":iki_asama,"zayif":zayif}
''',
    "konu05": COMMON + r'''
from scipy.special import expit
def veri_uret(n=1000):
    yas=rng.normal(35,9,n); kadin=rng.binomial(1,.5,n); p=expit(-2+.055*yas+.45*kadin); y=rng.binomial(1,p,n)
    return pd.DataFrame({"y":y,"yas":yas,"kadin":kadin})

def egriler():
    d=veri_uret(); x=sm.add_constant(d[["yas","kadin"]]); lpm=sm.OLS(d.y,x).fit(); logit=sm.Logit(d.y,x).fit(disp=False); grid=np.linspace(18,60,100)
    plt.plot(grid,lpm.predict(sm.add_constant(pd.DataFrame({"yas":grid,"kadin":0}),has_constant="add")),label="Doğrusal"); plt.plot(grid,logit.predict(sm.add_constant(pd.DataFrame({"yas":grid,"kadin":0}),has_constant="add")),label="Logit"); plt.legend(); plt.show()

def etkiler():
    d=veri_uret(); m=sm.Logit(d.y,sm.add_constant(d[["yas","kadin"]])).fit(disp=False); print(m.get_margeff(at="overall").summary())

def veri():
    yol=Path("cps09mar_ikt807.csv"); d=pd.read_csv(yol) if yol.exists() else veri_uret(); print(d.head()); etkiler() if "y" in d else print("CPS sütunlarını sonuç modelinize göre seçin.")

def secim():
    d=veri_uret(); x=sm.add_constant(d[["yas","kadin"]]); modeller={"LPM":sm.OLS(d.y,x).fit(),"Logit":sm.Logit(d.y,x).fit(disp=False),"Probit":sm.Probit(d.y,x).fit(disp=False)}
    print(pd.DataFrame({k:{"AIC":v.aic,"BIC":v.bic} for k,v in modeller.items()}).T)

SECTIONS={"egriler":egriler,"etkiler":etkiler,"veri":veri,"secim":secim}
''',
    "konu06": COMMON + r'''
from scipy.stats import norm
def veri_uret(n=1000):
    x=rng.normal(size=n); gizli=-.4+1.2*x+rng.normal(size=n); gozlenen=np.maximum(gizli,0)
    return pd.DataFrame({"x":x,"gizli":gizli,"gozlenen":gozlenen})

def mekanizma():
    d=veri_uret(); print("Sıfır oranı:",(d.gozlenen==0).mean()); plt.scatter(d.x,d.gozlenen,s=8,alpha=.3); plt.show()

def hedefler():
    beta0,beta1,sigma=-.4,1.2,1.; x=np.linspace(-2,2,9); z=(beta0+beta1*x)/sigma
    print(pd.DataFrame({"x":x,"gizli_ortalama":beta0+beta1*x,"pozitif_olasilik":norm.cdf(z),"gozlenen_ortalama":norm.cdf(z)*(beta0+beta1*x)+sigma*norm.pdf(z)}))

def veri():
    d=veri_uret(); print(sm.OLS(d.gozlenen,sm.add_constant(d[["x"]])).fit(cov_type="HC1").summary()); print("Sansürlü OLS gizli denklem eğimini hedeflemez.")

def secim():
    n=1500; x=rng.normal(size=n); z=rng.normal(size=n); sec=(.7*z+.4*x+rng.normal(size=n)>0); y=1+1.1*x+rng.normal(size=n); goz=pd.DataFrame({"y":y[sec],"x":x[sec],"z":z[sec]})
    print("Seçilme oranı:",sec.mean()); print(sm.OLS(goz.y,sm.add_constant(goz[["x"]])).fit(cov_type="HC1").summary())

SECTIONS={"mekanizma":mekanizma,"hedefler":hedefler,"veri":veri,"secim":secim}
''',
    "konu07": COMMON + r'''
def veri_uret(n=900):
    x=rng.uniform(0,5,n); y=1+.6*x+(0.3+.25*x)*rng.standard_t(5,n)
    return pd.DataFrame({"y":y,"x":x})

def kayip():
    u=np.linspace(-2,2,200); plt.plot(u,u*(.25-(u<0)),label="tau=.25"); plt.plot(u,u*(.75-(u<0)),label="tau=.75"); plt.legend(); plt.show()

def hedef():
    d=veri_uret(); ols=sm.OLS(d.y,sm.add_constant(d[["x"]])).fit(); qr=sm.QuantReg(d.y,sm.add_constant(d[["x"]])).fit(q=.5)
    print(pd.DataFrame({"OLS":ols.params,"Medyan":qr.params}))

def profil():
    d=veri_uret(); tau=np.arange(.1,1,.1); b=[sm.QuantReg(d.y,sm.add_constant(d[["x"]])).fit(q=t).params["x"] for t in tau]
    plt.plot(tau,b,marker="o"); plt.xlabel("Kantil düzeyi"); plt.ylabel("x katsayısı"); plt.show()

def veri():
    yol=Path("cps09mar_ikt807.csv"); d=pd.read_csv(yol) if yol.exists() else veri_uret(); print(d.head()); profil() if "y" in d else print("CPS için lwage ve eğitim değişkenlerini seçin.")

SECTIONS={"kayip":kayip,"hedef":hedef,"profil":profil,"veri":veri}
''',
    "konu08": COMMON + r'''
def veri_uret(n=600):
    x=rng.uniform(-3,3,n); y=np.sin(1.4*x)+.25*x+rng.normal(0,.35,n); return x,y
def agirliklar(x,x0,h):
    w=np.exp(-.5*((x-x0)/h)**2); return w/w.sum()
def yerel_tahmin(x,y,grid,h):
    return np.array([agirliklar(x,g,h)@y for g in grid])

def agirlik():
    x,y=veri_uret(); w=agirliklar(x,0,.7); plt.scatter(x,w,s=10); plt.xlabel("x"); plt.ylabel("Normalize çekirdek ağırlığı"); plt.show(); print("Toplam:",w.sum())

def yerel():
    x,y=veri_uret(); g=np.linspace(-3,3,150); plt.scatter(x,y,s=8,alpha=.2); plt.plot(g,yerel_tahmin(x,y,g,.7),color="crimson"); plt.show()

def dogrulama():
    x,y=veri_uret(); kat=np.arange(len(x))%5; hs=np.linspace(.2,1.5,14); kayip=[]
    for h in hs:
        hata=[]
        for k in range(5):
            tr=kat!=k; hata.extend((y[~tr]-yerel_tahmin(x[tr],y[tr],x[~tr],h))**2)
        kayip.append(np.mean(hata))
    print("Seçilen bant genişliği:",hs[np.argmin(kayip)]); plt.plot(hs,kayip); plt.show()

def veri():
    yol=Path("DDK2011_ikt807.csv"); print(pd.read_csv(yol).head() if yol.exists() else "DDK dosyası yok; kontrollü örnek çalıştırılıyor."); yerel()

SECTIONS={"agirlik":agirlik,"yerel":yerel,"dogrulama":dogrulama,"veri":veri}
''',
    "konu09": COMMON + r'''
def veri_uret(n=1600,etki=1.4):
    x=rng.uniform(-10,10,n); d=(x>=0).astype(float); y=.3*x+.04*x**2+etki*d+rng.normal(size=n); uyum=rng.binomial(1,np.where(x>=0,.75,.25),n); yf=.3*x+etki*uyum+rng.normal(size=n)
    return pd.DataFrame({"x":x,"d":d,"y":y,"uyum":uyum,"yf":yf})
def sicrama(y,x,h=4):
    sol=(x<0)&(x>=-h); sag=(x>=0)&(x<=h); return np.polyfit(x[sag],y[sag],1)[1]-np.polyfit(x[sol],y[sol],1)[1]

def kesin():
    d=veri_uret(); print("Kesin RDD sıçraması:",sicrama(d.y.to_numpy(),d.x.to_numpy())); plt.scatter(d.x,d.y,s=6,alpha=.2); plt.axvline(0,color="crimson"); plt.show()

def duyarlilik():
    d=veri_uret(); hs=np.arange(2,9); b=[sicrama(d.y.to_numpy(),d.x.to_numpy(),h) for h in hs]; print(pd.DataFrame({"bant_genisligi":hs,"tahmin":b})); plt.plot(hs,b,marker="o"); plt.show()

def bulanik():
    d=veri_uret(); rf=sicrama(d.yf.to_numpy(),d.x.to_numpy()); fs=sicrama(d.uyum.to_numpy(),d.x.to_numpy()); print({"indirgenmiş_biçim":rf,"ilk_aşama":fs,"yerel_Wald":rf/fs})

def tanilar():
    d=veri_uret(); print("Eşik çevresi sağ/sol sayısı:",((d.x>=0)&(d.x<1)).sum(),((d.x<0)&(d.x>-1)).sum()); print("Placebo -5/+5:",sicrama(d.y.to_numpy(),(d.x+5).to_numpy(),2),sicrama(d.y.to_numpy(),(d.x-5).to_numpy(),2))

SECTIONS={"kesin":kesin,"duyarlilik":duyarlilik,"bulanik":bulanik,"tanilar":tanilar}
''',
    "konu10": COMMON + r'''
def veri_uret(n=700):
    x=rng.normal(size=n); y=1+.8*x+rng.normal(0,.5+.3*np.abs(x),n); return pd.DataFrame({"y":y,"x":x})
def cekimler(d,b=500):
    sonuc=[]
    for _ in range(b):
        s=d.iloc[rng.integers(0,len(d),len(d))]; sonuc.append(sm.OLS(s.y,sm.add_constant(s[["x"]])).fit().params["x"])
    return np.asarray(sonuc)

def ornekleme():
    d=veri_uret(); idx=rng.integers(0,len(d),len(d)); frekans=np.bincount(idx,minlength=len(d)); print(pd.Series(frekans).value_counts().sort_index())

def dagilim():
    b=cekimler(veri_uret()); plt.hist(b,bins=30); plt.xlabel("Yeniden örnekleme katsayısı"); plt.show(); print("Standart hata:",b.std(ddof=1))

def aralik():
    d=veri_uret(); m=sm.OLS(d.y,sm.add_constant(d[["x"]])).fit(cov_type="HC1"); b=cekimler(d); print({"HC1":m.conf_int().loc["x"].tolist(),"yüzdelik":np.quantile(b,[.025,.975]).tolist()})

def veri():
    yol=Path("cps09mar_ikt807.csv"); print(pd.read_csv(yol).head() if yol.exists() else "CPS dosyası yok; kontrollü örnek çalıştırılıyor."); aralik()

SECTIONS={"ornekleme":ornekleme,"dagilim":dagilim,"aralik":aralik,"veri":veri}
''',
    "konu11": COMMON + r'''
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
def veri_uret(n=800,p=30):
    x=rng.normal(size=(n,p)); beta=np.zeros(p); beta[:6]=[2,-1.7,1.4,-1.1,.8,-.6]; return x,x@beta+rng.normal(size=n)

def yanlilik():
    x=rng.uniform(-2.5,2.5,400)[:,None]; y=np.sin(1.6*x[:,0])+rng.normal(0,.45,400); tr,te=train_test_split(np.arange(400),test_size=.35,random_state=TOHUM); rows=[]
    for d in range(1,13):
        m=make_pipeline(PolynomialFeatures(d),LinearRegression()).fit(x[tr],y[tr]); rows.append((d,mean_squared_error(y[tr],m.predict(x[tr])),mean_squared_error(y[te],m.predict(x[te]))))
    z=pd.DataFrame(rows,columns=["derece","eğitim_MSE","sınama_MSE"]); print(z); z.plot(x="derece"); plt.show()

def ceza():
    x,y=veri_uret(); z=StandardScaler().fit_transform(x); a=np.logspace(-3,1,35); l=np.array([Lasso(alpha=v,max_iter=20000).fit(z,y).coef_ for v in a]); r=np.array([Ridge(alpha=v).fit(z,y).coef_ for v in a]); plt.semilogx(a,l[:,:8]); plt.title("Lasso yolu"); plt.show(); plt.semilogx(a,r[:,:8]); plt.title("Ridge yolu"); plt.show()

def dogrulama():
    x,y=veri_uret(); tr,te=train_test_split(np.arange(len(y)),test_size=.25,random_state=TOHUM); a=np.logspace(-3,0,25); k=KFold(5,shuffle=True,random_state=TOHUM); kayip=[]
    for v in a:
        e=[]
        for i,j in k.split(x[tr]):
            m=make_pipeline(StandardScaler(),Lasso(alpha=v,max_iter=20000)).fit(x[tr][i],y[tr][i]); e.append(mean_squared_error(y[tr][j],m.predict(x[tr][j])))
        kayip.append(np.mean(e))
    sec=a[np.argmin(kayip)]; m=make_pipeline(StandardScaler(),Lasso(alpha=sec,max_iter=20000)).fit(x[tr],y[tr]); print("Seçilen lambda ve sınama MSE:",sec,mean_squared_error(y[te],m.predict(x[te])))

def veri():
    yol=Path("cps09mar_ikt807.csv"); print(pd.read_csv(yol).head() if yol.exists() else "CPS dosyası yok; seyrek kontrollü örnek çalıştırılıyor."); dogrulama()

SECTIONS={"yanlilik":yanlilik,"ceza":ceza,"dogrulama":dogrulama,"veri":veri}
''',
    "konu12": COMMON + r'''
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
def veri_uret(n=800,p=24,g=40):
    x=rng.normal(size=(n,p)); grup=np.arange(n)%g; rng.shuffle(grup); d=.7*x[:,0]-.5*x[:,4]+rng.normal(size=n); y=1.5*d+1.2*x[:,0]+np.sin(x[:,1])+rng.normal(size=n)+rng.normal(0,.3,g)[grup]; return y,d,x,grup

def secim():
    y,d,x,_=veri_uret(); my=make_pipeline(StandardScaler(),LassoCV(cv=5)).fit(x,y); md=make_pipeline(StandardScaler(),LassoCV(cv=5)).fit(x,d); sy=np.flatnonzero(my[-1].coef_); sd=np.flatnonzero(md[-1].coef_); union=np.union1d(sy,sd)
    print({"sonuç_seçimi":sy.tolist(),"tedavi_seçimi":sd.tolist(),"birleşim":union.tolist()}); print(sm.OLS(y,sm.add_constant(np.column_stack([d,x[:,union]]))).fit(cov_type="HC1").params[1])

def dml(tohum=812):
    y,d,x,g=veri_uret(); yh=np.empty(len(y)); dh=np.empty(len(y)); kat=np.empty(len(y),int)
    for k,(tr,te) in enumerate(GroupKFold(5,shuffle=True,random_state=tohum).split(x,groups=g)):
        my=make_pipeline(StandardScaler(),RidgeCV()).fit(x[tr],y[tr]); md=make_pipeline(StandardScaler(),RidgeCV()).fit(x[tr],d[tr]); yh[te]=my.predict(x[te]); dh[te]=md.predict(x[te]); kat[te]=k
    m=sm.OLS(y-yh,d-dh).fit(cov_type="cluster",cov_kwds={"groups":g}); print("theta, kümeli SH:",m.params[0],m.bse[0]); return m.params[0],kat

def capraz():
    theta,kat=dml(); print("Kat gözlem sayıları:",np.bincount(kat),"theta:",theta)

def duyarlilik():
    print(pd.DataFrame({"başlangıç_değeri":[812,919,1207],"theta":[dml(s)[0] for s in (812,919,1207)]}))

def veri():
    yol=Path("DDK2011_ikt807.csv"); print(pd.read_csv(yol).head() if yol.exists() else "DDK dosyası yok; okul-kümeli kontrollü örnek çalıştırılıyor."); print("Akış: hedef -> tanımlama -> veri -> yardımcı modeller -> kat dışı tahmin -> duyarlılık -> kayıt"); dml()

SECTIONS={"secim":secim,"capraz":capraz,"duyarlilik":duyarlilik,"veri":veri}
''',
}


def list_topic_sections(topic_key: str) -> tuple[tuple[str, str], ...]:
    try:
        return TOPIC_SECTIONS[topic_key]
    except KeyError as error:
        raise ValueError(f"Kod tarifi olmayan konu: {topic_key}") from error


def build_python_recipe(topic_key: str, section_key: str) -> CodeRecipe:
    sections = dict(list_topic_sections(topic_key))
    if section_key not in sections:
        raise ValueError(f"Kod tarifi olmayan bölüm: {topic_key}/{section_key}")
    program = dedent(TOPIC_PROGRAMS[topic_key]).strip()
    code = (
        '"""IKT 807 bağımsız laboratuvar kodu.\n\n'
        "Kurulum: pip install numpy pandas matplotlib scipy statsmodels scikit-learn\n"
        '"""\n\n'
        f'ACTIVE_SECTION = "{section_key}"\n\n'
        f"{program}\n\n"
        'if __name__ == "__main__":\n'
        '    print(f"Çalıştırılan bölüm: {ACTIVE_SECTION}")\n'
        '    SECTIONS[ACTIVE_SECTION]()\n'
    )
    return CodeRecipe(
        topic_key=topic_key,
        section_key=section_key,
        section_title=sections[section_key],
        python_code=code,
        filename_stem=f"ikt807_{topic_key}_{section_key}",
    )


def build_colab_notebook(recipe: CodeRecipe) -> str:
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"name": f"{recipe.filename_stem}.ipynb", "provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# IKT 807 - {recipe.section_title}\n",
                    "Bu defter, Streamlit laboratuvarındaki bölümü bağımsız olarak yeniden üretir.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["%pip install -q numpy pandas matplotlib scipy statsmodels scikit-learn\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": recipe.python_code.splitlines(keepends=True),
            },
        ],
    }
    return dumps(notebook, ensure_ascii=False, indent=2)
