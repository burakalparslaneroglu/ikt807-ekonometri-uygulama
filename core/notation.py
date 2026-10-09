"""Ders notlarındaki Hansen convention'ının arayüzden bağımsız açıklaması."""

COMMON_NOTATION = r"""
Hansen (§1.3, §3.10–3.11) büyük italik harfleri rassal skalerler ve rassal
vektörler için kullanır. **Kalın olmamak skaler olmak anlamına gelmez.** Küçük
$x,y$ gerçekleşmeleri gösterir; hata $e$ geleneksel küçük harf istisnasıdır.
Boyut, sembolün tanımından anlaşılır. Tek gözlem modeli

$$Y=X'\beta+e,\qquad Y,e\in\mathbb R,\quad X,\beta\in\mathbb R^k.$$

Gözlem indeksi eklenince $Y_i=X_i'\beta+e_i$ olur. $n$ gözlemin yığıldığı
örneklemde kalın italik yazım kullanıyoruz:

$$\boldsymbol Y=\boldsymbol X\beta+\boldsymbol e,\qquad\boldsymbol Y,\boldsymbol e\in\mathbb R^{n\times1}.$$

$$\boldsymbol X=\begin{bmatrix}X_1'\\\vdots\\X_n'\end{bmatrix}\in\mathbb R^{n\times k}.$$

Burada $\boldsymbol X$ **transpoze edilmez**. $k$ sabit terimi de içerir;
skaler açıklayıcı değişkenlerle açılmış denklemlerde $\beta_0$ sabit terimdir.
Sezgi deneylerinde $X$ açıkça tek açıklayıcı değişken olarak tanımlanıyorsa
skalerdir; regresör vektörü $(1,X)'$ olur. Yunan harfli parametre vektörleri
Hansen gibi normal italik yazılır.

$\widehat{\boldsymbol V}_{\hat\beta}$, $\hat\beta$'nın tahmini kovaryansıdır.
$\boldsymbol V_\beta$, $\sqrt n(\hat\beta-\beta)$'nın limit kovaryansıdır.

İkisi farklı ölçeklerdir: $\widehat{\boldsymbol V}_\beta=n\widehat{\boldsymbol V}_{\hat\beta}$.
Skaler hedefte $V_{\hat\theta}$ varyanstır ve $SH(\hat\theta)=\sqrt{\widehat V_{\hat\theta}}$.
Hata ile artık ayrılır: $e_i$ hata, $\hat e_i=Y_i-X_i'\hat\beta$ artık.
"""

TOPIC_NOTATION = {
    "konu01": r"""$m(x)=\mathbb E[Y\mid X=x]$ koşullu ortalamadır.
$\beta=(\mathbb E[XX'])^{-1}\mathbb E[XY]$ anakütle projeksiyon katsayısı;
$\hat\beta=(\boldsymbol X'\boldsymbol X)^{-1}\boldsymbol X'\boldsymbol Y$ örneklem tahmin edicisidir.
$0_k$ normal denklemlerdeki $k\times1$ sıfır vektörüdür.""",
    "konu02": r"""$\boldsymbol a$ doğrusal birleşimin ağırlık vektörüdür.
İki katsayı seçilirse $\boldsymbol a=(1,1)'$ ve ilgili $2\times2$ kovaryans
alt matrisi kullanılır; tam katsayı vektöründe diğer ağırlıklar sıfırdır.
$H_0:\beta_j=b_0$ testinde $b_0$ varsayılan değerdir; bir soruda $\beta_0$
ile gösterilen null değer, o soruya özgüdür. $g$ küme, $i$ küme içindeki
gözlem indeksidir; $X_g$ bütün kümenin paylaştığı rassal skaler regresördür.""",
    "konu03": r"""$Y_1,Y_0$ tedavi altında ve tedavisiz potansiyel sonuçlardır;
$Y_{1i},Y_{0i}$ gözlem $i$ için karşılıklarıdır. $\theta=Y_1-Y_0$ bireysel
etki, $ATE=\mathbb E[\theta]$, $ATT=\mathbb E[\theta\mid D=1]$;
$D\in\{0,1\}$ tedavi göstergesidir. Sabit etkili simülasyonda $\theta$ tüm
bireylerde aynıdır. $\Delta_{obs}$ gözlenen grup ortalamaları farkını gösterir.
Ölçüm hatası örneğinde $Z$ gerçek, $X=Z+u$ ölçülen skaler değişkendir.""",
    "konu04": r"""IV denklemlerinde $X$ skaler içsel değişken, $W$ dışsal
kontrol vektörü, $Z$ dışlanmış araç vektörüdür; $e$ yapısal, $v$ ilk aşama,
$\eta$ indirgenmiş biçim hatasıdır. Matris formunda $\boldsymbol X$
kontrolleri de içeren $n\times k$, $\boldsymbol Z$ bütün araçları içeren
$n\times\ell$ matristir; $\boldsymbol P_Z=\boldsymbol Z(\boldsymbol Z'\boldsymbol Z)^{-1}\boldsymbol Z'$
$n\times n$ projeksiyon matrisi, $\widehat{\boldsymbol X}=\boldsymbol P_Z\boldsymbol X$.
LATE örneğinde $D_1,D_0$ araç durumlarına göre potansiyel tedaviler,
$\theta_c$ uyumluların ortalama etkisidir; alt indis $c$ uyumlu grubu belirtir.
$D_{1i},D_{0i}$ gözlem $i$ için potansiyel tedavilerdir; $D_i=Z_iD_{1i}+(1-Z_i)D_{0i}$.""",
    "konu05": r"""$G$ indeksin dağılım fonksiyonu, $g=G'$ yoğunluk;
Logit için $\Lambda$, Probit için $\Phi$ kullanılır. $\boldsymbol Q$
beklenen negatif Hessian, $\boldsymbol\Omega$ gözlem skorunun kovaryansıdır.
$\boldsymbol V_\beta=\boldsymbol Q^{-1}\boldsymbol\Omega\boldsymbol Q^{-1}$
limit kovaryansıdır; tahmin edicinin kovaryansı için $1/n$ ölçeği gerekir.
Tek parametreli soruda $Q,\Omega$ bunların skaler karşılıklarıdır.""",
    "konu06": r"""$Y^*$ gizli sonuç, $S$ seçilme göstergesi;
$\lambda(v)=\phi(v)/\Phi(v)$ ters Mills oranıdır. $\sigma_{21}=\operatorname{Cov}(e,u)$,
$\operatorname{Var}(u)=1$ normalizasyonunda düzeltme katsayısıdır;
$\lambda$'nın kendisi bu katsayı değildir. $\rho=\operatorname{Corr}(e,u)$
ve $\sigma_{21}=\rho\sigma_e$. Seçim simülasyonunda $X,Z$ skaler,
seçim indeksi $0{,}2+0{,}8X+\gamma_ZZ$ ve $\sigma_e=1$ olduğu için $\sigma_{21}=\rho$.""",
    "konu07": r"""$\tau\in(0,1)$ kantil düzeyidir;
$Q_\tau[Y\mid X]=q_\tau(X)=X'\beta_\tau$. $\beta_\tau$ vektör,
$\beta_{j,\tau}$ onun $j$'nci bileşenidir; eğitim eğimi için $j=1$.
$\rho_\tau(u)=u(\tau-\mathbf1\{u<0\})$ kontrol kaybıdır.
HK hesabındaki ek sembol $\boldsymbol H_\tau=\sum_i\hat f_iX_iX_i'$
$k\times k$ yoğunluk ağırlıklı matristir; $\hat f_i$ ilgili kantildeki
tahmini koşullu yoğunluktur. Kovaryans formülleri bu toplam ve örneklem
matrisiyle yazıldığında doğrudan $\hat\beta_\tau$'nın kovaryansını verir.""",
    "konu08": r"""$K$ çekirdek fonksiyonu, $h$ bant genişliği;
$K_i=K((X_i-x)/h)$ yerel ağırlıktır. Seri regresyonunda ayrıca belirtilen
$K$ baz fonksiyonu sayısıdır. Kısmen doğrusal $Y=D\theta+g(X)+e$ modeli,
notların $Y=m(X)+Z'\beta+e$ gösteriminin $Z=D$, $\beta=\theta$, $m=g$
özel durumudur. $m_Y(X)=\mathbb E[Y\mid X]=\theta m_D(X)+g(X)$ ve
$m_D(X)=\mathbb E[D\mid X]$; yapısal $g$, $m_Y$ ile aynı fonksiyon değildir.
$v=D-m_D(X)$ bu bölümde açıkça tanımlanan skaler artıktır.""",
    "konu09": r"""$X$ skaler eşik değişkeni, $c$ eşik,
$R_i=X_i-c$ merkezlenmiş değişken, $D_i=\mathbf1\{X_i\ge c\}$ keskin
tasarımın tedavi göstergesidir. Yerel denklemlerde $\theta=\theta(c)$
eşikteki etkidir. Bulanık RDD'de $\Delta_Y,\Delta_D$ sonuç ve tedavi
olasılığının sıçramaları, $\theta_{FRD}=\Delta_Y/\Delta_D$ hedef orandır.
$n_h$ pozitif çekirdek ağırlığı alan gözlem sayısıdır.""",
    "konu10": r"""Üst simge $*$ yeniden örnekleme dünyasını gösterir;
$(Y_i^*,X_i^*)$ yeniden çekilen gözlem çiftleri, $\boldsymbol Y^*,\boldsymbol X^*$
yığılmış örneklemdir. $B$ bootstrap tekrar sayısı; $\xi_i^*$ wild bootstrap
çarpanıdır. Birinci deneyde $X_i$ skaler olduğundan uyum değeri
$\hat\beta_0+\hat\beta_1X_i$ olarak açılır. $g$ küme indeksidir.""",
    "konu11": r"""Ceza formüllerinde $\boldsymbol X,\boldsymbol Y$ eğitim
örnekleminde merkezlenmiş matris ve vektördür; sabit terim cezalandırılmaz.
$p$ cezalandırılan eğim sayısı, $\boldsymbol I_p$ birim matris,
$r_j$ $\boldsymbol X'\boldsymbol X$'in özdeğeridir.
$\lambda$ Hansen SSE ölçeğindeki, $\lambda_y$ yazılım ölçeğindeki cezadır:
Lasso için $\lambda=2n\lambda_y$, Ridge için $\lambda=n\lambda_y$.
Hansen Elastic Net'te $\alpha$ L2 ağırlığıdır; yazılımın L1 oranı $r=1-\alpha$.
Doğrudan SSE kullanan scikit-learn Ridge parametresi $\lambda$'dır.""",
    "konu12": r"""$D,Y$ skaler, $X$ kontrol vektörü; doğrusal indirgenmiş
biçimler $D=X'\gamma+V$, $Y=X'\eta+U$ ve $\eta=\beta+\gamma\theta$.
$m_Y,m_D$ koşullu ortalamalar, $g=m_Y-\theta m_D$ yapısal fonksiyondur.
$A_k$ tutulmuş kat; $i\in A_k$ için $\hat m^{(-k)}$ bu kat dışındaki veriden
öğrenilir. $\hat U_i=Y_i-\hat m_Y^{(-k)}(X_i)$,
$\hat V_i=D_i-\hat m_D^{(-k)}(X_i)$ skaler artıklardır.
$\psi_i=V_i(U_i-\theta V_i)$ gözlem skoru; normalize edilmiş etki fonksiyonu
notlarda $\varphi$ ile gösterilir. Yazılımdaki Lasso cezası $\lambda_y$'dir.
$\widehat V_{\hat\theta,s}=SH_s^2$ bölme $s$'nin tahmini varyansı,
$C_s=\widehat V_{\hat\theta,s}+(\hat\theta_s-\hat\theta_{med})^2$
medyan birleştirmesine giren terimdir. $\|f\|_{P,2}=(\mathbb E[f(X)^2])^{1/2}$
anakütle L2 normudur.""",
}
