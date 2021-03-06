# 第二部分 深层网络：现代实践

##第六章 深度前馈网络

###6.5 反向传播和其它微分算法

####6.5.6 一般化的反向传播

*P.184*

$\because \boldsymbol{a}^{(k)} = \boldsymbol{b}^{(k)}  + \boldsymbol{W}^{(k)}\boldsymbol{h}^{(k-1)}$

$\nabla_{\boldsymbol{b}^{(k)}}J = \nabla_{\boldsymbol{a}^{(k)}} L(\widehat{y},y) \cdot \frac{\partial\boldsymbol{a}^{(k)}}{\partial \boldsymbol{b}^{(k)}} + \lambda \cdot \nabla_{\boldsymbol{b}^{(k)}} \Omega (\theta)$

$\because \frac{\partial\boldsymbol{a}^{(k)}}{\partial \boldsymbol{b}^{(k)}}=1$

$\therefore \nabla_{\boldsymbol{b}^{(k)}}J = \nabla_{\boldsymbol{a}^{(k)}} L(\widehat{y},y) + \lambda \cdot \nabla_{\boldsymbol{b}^{(k)}} \Omega (\theta) = \boldsymbol{g} + \lambda \cdot \nabla_{\boldsymbol{b}^{(k)}} \Omega (\theta)$

对$\boldsymbol{W}^{(k)}$以及$\boldsymbol{h}^{(t-1)}$的梯度同样,前者参考《The Matrix Cookbook》2.4.1中式(71)，将$\boldsymbol{a}^\intercal$换成单位向量即可得用其梯度为$\boldsymbol{h}^{(k-1)\intercal}$，后者参考式(69)，将$\boldsymbol{a}^\intercal$换为$\boldsymbol{W^{(k)}}$可解。

*P.185*

[矩阵求导](https://zhuanlan.zhihu.com/p/24709748)

先写出全微分形式：

$$
\begin{align}
dz &= tr[(\frac{\partial z}{\partial \boldsymbol{C}})^\intercal d\boldsymbol{C}] = tr[(\frac{\partial z}{\partial \boldsymbol{C}})^\intercal d\boldsymbol{AB}] \\
   &= tr[\boldsymbol{G}^\intercal (d\boldsymbol{A}\cdot \boldsymbol{B}+\boldsymbol{A}d\boldsymbol{B})]\\
   &= tr(\boldsymbol{BG}^\intercal d\boldsymbol{A}) + tr(\boldsymbol{G}^\intercal \boldsymbol{A} d\boldsymbol{B})
\end{align}
$$

故：

$$\frac{\partial z}{\partial \boldsymbol{A}} = (\boldsymbol{BG}^\intercal)^\intercal = \boldsymbol{GB}^\intercal$$

$$\frac{\partial z}{\partial \boldsymbol{B}} = (\boldsymbol{G}^\intercal \boldsymbol{A})^\intercal = \boldsymbol{A}^\intercal \boldsymbol{G}$$

##第七章 深度学习中的正则化

###7.1 参数范数惩罚

####7.1.1 $L^{2}$参数正则化

*P.199*

式7.6为$\widehat{J}(\boldsymbol{\theta})$在$\boldsymbol{\omega}^*$处的泰勒展开式，且根据$\boldsymbol{\omega}^* = argmin_{\boldsymbol{\omega}}J(\boldsymbol{\omega})$知$\widehat{J}(\boldsymbol{\theta})$在$\boldsymbol{\omega}^*$处的Jacobian为0，且$(\boldsymbol{\omega}-\boldsymbol{\omega}^*)^\intercal \boldsymbol{H} (\boldsymbol{\omega}-\boldsymbol{\omega}^*) \ge 0$。

加入权重衰减的梯度相当于被优化目标方程改写为：$\widehat{J}(\boldsymbol{\theta}) = J(\boldsymbol{\omega}^*) + \frac{1}{2}(\boldsymbol{\omega}-\boldsymbol{\omega}^*)^\intercal \boldsymbol{H} (\boldsymbol{\omega}-\boldsymbol{\omega}^*) + \frac{1}{2}\boldsymbol{\omega}^\intercal \boldsymbol{\omega}$

*P.200*

等高线图正好与空间拉伸图趋势相反，容易混淆：

图2.3为空间拉伸示意图，为作等高线图使$\lambda_1 v_1 = \lambda_2 v_2$，因为$\lambda_1 > \lambda_2$，易得$v_1 < v_2$

*P.201*

目标函数应为：

$f = (\boldsymbol{X\omega}-\boldsymbol{y})^\intercal(\boldsymbol{X\omega}-\boldsymbol{y})+\alpha\boldsymbol{\omega}^\intercal \boldsymbol{\omega}$

将式7.17按前述方法进行SVD后得到第i项经正则化后其因子缩放倍率为:
$\frac{\lambda_i}{\lambda_i^{2} + \alpha}$

####7.1.2 $L^{1}$参数正则化

*p.202*

与$L^{2}$正则化相类似，不考虑偏置的情形，我们将$L^{1}$正则化项写成一个自定义超参数$\alpha$与权重$\boldsymbol{\omega}$的一范数相乘的一项，则有：

$$\widetilde{J}(\boldsymbol{\omega; X, y})= J(\boldsymbol{\omega; X, y}) + \alpha\Vert\boldsymbol{\omega}\Vert_1 = J(\boldsymbol{\omega; X, y}) + \sum_{i}\alpha|\omega_i| \tag{1}$$

为求近似，我们保留上式中的第二项，将上式中的前一项在$\boldsymbol{\omega^*}$处进行泰勒展开，根跟定义可知$\boldsymbol{J}$在$\boldsymbol{\omega^*}$处的Jacobian为0，则有：

$$\widetilde{J}(\boldsymbol{\omega; X, y})= J(\boldsymbol{\omega^*; X, y}) + \frac{1}{2}(\boldsymbol{\omega}-\boldsymbol{\omega}^*)^\intercal \boldsymbol{H} (\boldsymbol{\omega}-\boldsymbol{\omega}^*) + \sum_{i}\alpha|\omega_i| \tag{2}$$

可以在局部小的区域作二次近似（严格的凸函数），不妨假设上式在$\boldsymbol{\omega^*}$处取得唯一最小值，那么$\boldsymbol{H}$一定为正定矩阵，满秩且可逆，因此我们可以通过数据预处理将$\boldsymbol{X}$变换为合适的$\boldsymbol{X'}$后，实现对原Hessian的对角化：

$$\widetilde{J}(\boldsymbol{\omega; X', y})= J(\boldsymbol{\omega^*; X', y}) + \frac{1}{2}(\boldsymbol{\omega}-\boldsymbol{\omega}^*)^\intercal \boldsymbol{\Lambda} (\boldsymbol{\omega}-\boldsymbol{\omega}^*) + \sum_{i}\alpha|\omega_i| \tag{3}$$

因为严格凸性的前提，可知$\Lambda$中所有元素均大于0，若定义$tr(\Lambda) = \sum_{i}H_{i,i}$，就可以把式(3)还原成书中式(7.22)的形式：

$$\widehat{J}(\boldsymbol{\omega; X', y})= J(\boldsymbol{\omega^*; X', y}) + \sum_{i}[\frac{1}{2}H_{i,i}(\omega_i-\omega_i^{*})^2+\alpha|\omega_i|] \tag{4}$$

$\alpha|\omega_i|$的一阶偏导为符号函数，在0点不可导，除0点外，我们对式(4)向$\omega_i$求二阶偏导可知其结果正好为$H_{i,i} > 0$，经处理后的Hessian阵为对角矩阵，意味着所有的$\omega_i$均线性无关，也就是说，分别调整每一维的$\omega_i$使$\omega_i = argmin_{\omega_i}J(\boldsymbol{\omega; X', y})$后（一阶偏导数为0），$\boldsymbol{J}$也就取得了全局最小值。

式(4)对$\omega_i$求一阶偏导数后有：

$$\frac{\partial \widehat{J}(\boldsymbol{\omega; X', y})}{\partial \omega_i}= H_{i,i}(\omega_i-\omega_i^{*})+\alpha \cdot sign(\omega_i) \tag{5}$$

使上式为0，有：

$$\omega_i = \omega_i^*-\frac{\alpha}{H_{i,i}}sign(\omega_i) = |\omega_i^*|\cdot sign(\omega_i^*)-\frac{\alpha}{H_{i,i}}sign(\omega_i) \tag{6}$$

我们前面已做过假设使其在$\boldsymbol{\omega^*}$处取得唯一最小值，观察式(4)，在$|\omega_i|$不变的情况下，为使其取得最小值，$\omega_i$应与$\omega_i^*$同号，即$sign(\omega_i) = sign(\omega_i^*)$，将其代入式(6)中有：

$$\omega_i = |\omega_i^*|\cdot sign(\omega_i^*)-\frac{\alpha}{H_{i,i}}sign(\omega_i^*) = sign(\omega_i^*)(|\omega_i^*|-\frac{\alpha}{H_{i,i}}) \tag{7}$$

通过上式我们就可以明显看出$L^{1}$正则化与$L^{2}$相比的最大不同：对于给定的超参数$\alpha$，其中任意一个维度i的参数$\omega_i$的$L^{1}$方法是从$\omega_i^*$向0的方向行进$(\alpha/H_{i,i})$的步长，但同时还必须满足$sign(\omega_i) = sign(\omega_i^*)$的约束，因此当$|\omega_i^*|\leq|\frac{\alpha}{H_{i,i}}|$时，$\omega_i$只能取到0，因此其最优解也就写成了书中式(7.23)的形式。

通过增大$\alpha$的值，$L^{1}$方法可以使更多的$\omega_i$变为0，以此获得更为稀疏的解，这是其与$L^{2}$方法相比最本质的不同。这种特性使其广泛地被用于机器学习中的特征选择，从而达到简化机器学习的问题的目的。

*P.203*

结合式(5.79)发现，其中的后一项$log\,p(\boldsymbol{\omega})$在当$p(\boldsymbol{\omega}) \sim N(\boldsymbol{\omega};\boldsymbol{0},\frac{1}{\alpha^2}\boldsymbol{I}^2)$且$\Sigma = -\frac{1}{\alpha}\boldsymbol{I}$时，$log\,p(\boldsymbol{\omega})\propto \alpha \boldsymbol{\omega}^\intercal \boldsymbol{\omega}$ 正好对应$L^2$正则化；当$p(\boldsymbol{\omega}) \sim La(\boldsymbol{\omega};\boldsymbol{0},-\frac{1}{\alpha})$时，$log\,p(\boldsymbol{\omega})\propto \alpha |\boldsymbol{\omega}| = \alpha \sum_{i}|\omega_i|$正好对应于$L^1$正则化

原问题$f$为带不等式约束的凸优化问题，引入广义拉格朗日函数后，将其转换为对偶问题$q$，根据弱对偶定理，使$q$取得最大值的$\widehat{q}$与$f$取得最小值的$\widehat{f}$总存在如下关系：

$$\widehat{q} \leq \widehat{f} \tag{1}$$

若原问题中的$f$是可微的凸函数，则根据强对偶定理，使得式(1)中等式成立的拉格朗日乘数$\alpha$总是存在，因此有：

$$\boldsymbol{\theta}^* = \mathop{\arg\max}_{\alpha,\alpha \ge 0} \mathop{\min}_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}, \alpha) = \mathop{\arg\min}_{\boldsymbol{\theta}} \mathop{\max}_{\alpha,\alpha \ge 0} \mathcal{L}(\boldsymbol{\theta}, \alpha) \tag{2}$$

对$\alpha$的求算需用到SMO算法。

*P.208*

对于一个标准的$\,l\,$层MLP，假设最简单的情况（没有偏置项且各层之间都是线性激活函数）下该模型可写成：

$$\widehat{\boldsymbol{y}} = \boldsymbol{Wx} \tag{1}$$

向网络权重中添加随机扰动$\epsilon_w \sim \mathcal{N}(\boldsymbol{\epsilon},0,\eta\boldsymbol{I})$之后有：

$$\widehat{\boldsymbol{y}}_{\boldsymbol{\epsilon_{\boldsymbol{\omega}}}} = (\boldsymbol{W}+\boldsymbol{\epsilon_{\omega}})\boldsymbol{x} = \widehat{\boldsymbol{y}} + \boldsymbol{\epsilon_{\omega}}\boldsymbol{x} \tag{2}$$

将注入噪声后的目标函数(书中式7.31)完全展开：

$$\widetilde{\boldsymbol{J}}_{\boldsymbol{\omega}} = \mathbb{E}_{p(x,y,\epsilon_{\omega})}[(\widehat{y}_{\epsilon_{\omega}}-y)^2] = \mathbb{E}_{p(x,y,\epsilon_{\omega})}[(\widehat{y}-y+\epsilon_{\omega}x)^2]= \mathbb{E}_{p(x,y,\epsilon_{\omega})}[(\widehat{y}-y)^2 + (\epsilon_{\omega}x)^2 + 2\epsilon_{\omega}x(\widehat{y}-y)] \tag{3}$$

因为变量$\epsilon_{\omega}$与$x$，$y$，$\widehat{y}$相互独立，所以$p(x,y,\epsilon_{\omega}) = p(x,y)p(\epsilon_{\omega})$，代入上式的期望求算后有：

$$\widetilde{\boldsymbol{J}}_{\boldsymbol{\omega}} = \mathbb{E}_{p(x,y)}[(\widehat{y}-y)^2] + \mathbb{E}_{p(x,y)}[x^2]\mathbb{E}_{p(\epsilon_{\omega})}[\epsilon_{\omega}^2] + 2\mathbb{E}_{p(x,y)}[x(\widehat{y}-y)]\mathbb{E}_{p(\epsilon_{\omega})}[\epsilon_{\omega}] \tag{4}$$

因为$\epsilon_w \sim \mathcal{N}(\boldsymbol{\epsilon},0,\eta\boldsymbol{I})$，所以$\mathbb{E}_{p(\epsilon_{\omega})}[\epsilon_{\omega}]=0$，$\mathbb{E}_{p(\epsilon_{\omega})}[\epsilon_{\omega}^2] = \mathbb{D}_{p(\epsilon_{\omega})}[\epsilon_{\omega}]=\eta$，代入上式后得到：

$$\widetilde{\boldsymbol{J}}_{\boldsymbol{\omega}} = \mathbb{E}_{p(x,y)}[(\widehat{y}-y)^2] + \eta\mathbb{E}_{p(x,y)}[x^2] \tag{5}$$

此外$\nabla_{\boldsymbol{W}}\widehat{\boldsymbol{y}}=\frac{\partial \boldsymbol{Wx}}{\partial \boldsymbol{W}} = \boldsymbol{x}^\intercal$，故$\mathbb{E}_{p(x,y)}[\Vert \nabla_{\boldsymbol{\omega}}\widehat{\boldsymbol{y}} \Vert^2] = \mathbb{E}_{p(x,y)}[\boldsymbol{x}^\intercal\boldsymbol{x}] = \mathbb{E}_{p(x,y)}[x^2]$。代入上式后有：

$$\widetilde{\boldsymbol{J}}_{\boldsymbol{\omega}} = \mathbb{E}_{p(x,y)}[(\widehat{y}-y)^2] + \eta\mathbb{E}_{p(x,y)}[\Vert \nabla_{\boldsymbol{\omega}}\widehat{\boldsymbol{y}} \Vert^2] \tag{6}$$

因此对权重添加扰动等同于向代价函数中添加带权重噪声的正则化项，它引导$\widehat{y}$向$W$梯度小的方向变化，这便是权重的噪声扰动对增加训练模型鲁棒性的贡献。

###7.8 提前终止

*P.215*

$$\boldsymbol{Q}^\intercal \widetilde{\boldsymbol{\omega}} = (\boldsymbol{\Lambda} + \alpha \boldsymbol{I})^{-1}\boldsymbol{\Lambda Q}^\intercal \boldsymbol{\omega}^* = (\boldsymbol{\Lambda} + \alpha \boldsymbol{I})^{-1}(\boldsymbol{\Lambda} + \alpha\boldsymbol{I} - \alpha\boldsymbol{I})\boldsymbol{Q}^\intercal \boldsymbol{\omega}^* = [\boldsymbol{I} - (\boldsymbol{\Lambda} + \alpha \boldsymbol{I})^{-1}\alpha]\boldsymbol{Q}^\intercal \boldsymbol{\omega}^*$$

*P.216*

展开后有：

$$\tau\sum_{n=1}^{\infty}\frac{(-\boldsymbol{I})^{n+1}}{n}(-\epsilon\boldsymbol{\Lambda})^n = -\sum_{n=1}^{\infty}\frac{(-\boldsymbol{I})^{n+1}}{n}(\frac{\boldsymbol{\Lambda}}{\alpha})^n$$

逐项用等式可式(7.44)与式(7.45)的结论

*P.220*

参考周志华《机器学习》2.2评估方法中提及的“自助法”采样，对于m个样品中的任意一个样品，在其m次反复未被抽中的概率为：

$$(1-\frac{1}{m})^m$$

取其极限有：

$$\lim_{m\to\infty}(1-\frac{1}{m})^m = \frac{1}{e}$$

则自助法采样出的数据集含有原始数据集中实例的期望为$1-(1/e)$约为三分之二。

*P.225*

翻译有遗漏，原文：*the model with all units, but with the weights going out of unit i multiplied by the probability of including unit i.*

go out可表熄灭的意思，这里的意思说的是，假设有一个模型包含所有单元，但有的单元是未被激活的（以此实现dropout），那么在评价这个模型时我们就直接乘以“熄灭单元被抽中的概率”并以此作为模型权重。

*P.226*

对于任意的掩码的反码 $\boldsymbol{d}$， 总存在它的补集 $\boldsymbol{d'}$使得 $\boldsymbol{d}+\boldsymbol{d'}$全为所有元素全为1，按组来分，则这样的 $\boldsymbol{d}+\boldsymbol{d'}$将会有$2^{n-1}$组，因此不妨将式(7.65)重写为：

$$
\begin{align}
\widetilde{P}_{ensemble}(\tt{y}=\mathcal{y}|\boldsymbol{v})
&\propto exp(\frac{1}{2^n}\sum_{\boldsymbol{d, d'} \in \{0,1\}^n}\boldsymbol{W}_{y,:}^\intercal ((\boldsymbol{d}+\boldsymbol{d'})\odot \boldsymbol{v} + \boldsymbol{b}_{\mathcal{y}}))\\
 &= exp(\frac{1}{2^n}\frac{2^n}{2}\boldsymbol{W}_{y,:}^\intercal \boldsymbol{v} + \boldsymbol{b}_{\mathcal{y}}) = exp(\frac{1}{2}\boldsymbol{W}_{y,:}^\intercal \boldsymbol{v} + \boldsymbol{b}_{\mathcal{y}})
\end{align}
$$

*P.228*

在噪声鲁棒性一节中我们已经推导过向权重引入随机干扰后的代价函数满足：

$$\widetilde{\boldsymbol{J}}_{\boldsymbol{\omega}} = \mathbb{E}_{p(x,y)}[(\widehat{y}-y)^2] + \eta\mathbb{E}_{p(x,y)}[\Vert \nabla_{\boldsymbol{\omega}}\widehat{\boldsymbol{y}} \Vert^2]$$

后一项可视为一个正则项，这表明权重的随机扰动一定具有正则化的效果，而Bagging是通过对同一数据应用不同的模型进行投票后的结果。这样看来，虽然Dropout是随机作用于隐层(并非权重)，但随机屏蔽的特点让我们并不能将其简点理解为系统带来的随机噪声。相反，理解为“随机屏蔽部分节点后形成的子网络”之间“按权重进行投票”的结果则更为恰当一些。

##第八章 深度模型中的优化

###8.1 学习和纯优化有什么不同

*P.235*

式(8.2)中的 $f(\boldsymbol{x}; \boldsymbol{\theta})$为决策函数，通常将决策函数与真实值之间差异的量度为损失函数，风险就是将决策函数在特定的概率分布上求期望。在训练集上的经验分布上求期望时，这种风险对应为经验风险；在数据的后验分布上求期望时，这种风险对应为贝叶斯风险。

###8.2 神经网络优化中的挑战

####8.2.2 局部极小值

*P.242*

权重空间的对称性，以及权重在不同层间的增益都导致了已训练出来的权重都是通解中的一组特解，这样的特性导致了模型是不可辨识性的（模型参数是不唯一的），模型循路径落入的位置是局部极小值。如果局部极小值与全局最小值之间相差过大，那么这样的优化是不成功的。

*P.247*

[幂方法](https://baike.baidu.com/item/幂法求矩阵特征值/3082486)

对于n阶方阵$A$给定一系列n维向量 $x_i$并构建递推关系：$x_i = Ax_{i-1}$，则 $x_n = A^{n}x_0$，设 $r_i$与 $\alpha_i$分别为 $A$的一组特征值与特征向量，则 $x_0$可表示为 $x_0 = \sum_{i=1}^{n}a_i \alpha_i$，$a_i$为 $x_0$对应的系数，代入 $x_n$并考虑 $A\alpha_i = r_i \alpha_i$有 $x_n = A^n\sum_{i=1}^{n}a_i \alpha_i = \sum_{i=1}^{n}r_i^n a_i \alpha_i$。设 $max(r_i) = r_1$，则有 $x_n = r_1^n ( a_1 \alpha_1 + \sum_{i=2}^{n}(\frac{r_i}{r_1})^n a_i \alpha_i)$。取极限后求和符号里所有项均变为无穷小量。因此各时间步重复乘以相同的矩阵相当于保留特征值最大的主成份。

###8.3 基本算法

####8.3.1 随机梯度下降

*P.251*

按概率论的观点来看，从训练集中在线学习一个小批量的样本，批量梯度作为当前梯度的估计量。因为小批量样本依概率收敛于总体，因此批量梯度也依概率收敛于总体梯度。这意味着，while经循环多次后的确能使梯度达到极小点的作用。但因为每一次采样的容量是有限的，除非m趋于无穷使批量梯度依分布收敛于总体梯度，否则这个估计量一定是有偏的，这意味着批量梯度达到最小点之后目标函数并不会平滑，而是反复小幅振动。

####8.3.2 动量

*P.253*

相比于传统SGD，使用动量的SGD多了个动量参数，将计算更新速度按递推关系改写为：

$$\boldsymbol{v_{k+1}} = \alpha \boldsymbol{v_k} - \epsilon \boldsymbol{g_k} = \alpha^{k} \boldsymbol{v_1} - \epsilon\sum_{i=0}^{k-1} \alpha^i \boldsymbol{g_{k-i}}$$

可以看出更新的步长是所有梯度按动量参数的幂相乘后的线性加和形式，因此$\alpha$一定是一个介于0到1之间的数，否则随步长增加更新会发散。

另外用动量进行更新，不妨把$\boldsymbol{g_{k-i}}$写成两个分量$\boldsymbol{m_{i-1}}$与$\boldsymbol{n_{i-1}}$的形式，前者是重直于优化的目标方向，后者平行于优化的目标方向（参考图4.6），则上式又可写为：

$$\boldsymbol{v_{k+1}} = \alpha^{k} \boldsymbol{v_1} - \epsilon\sum_{i=0}^{k-1} \alpha^i \boldsymbol{m_{k-i}} - \epsilon\sum_{i=0}^{k-1} \alpha^i \boldsymbol{n_{k-i}}$$

对于前项，在病态条件下总有$sign(\boldsymbol{m_i}) = -sign(\boldsymbol{m_{i-1}})$，因而在k足够大时，梯度在重直优化方向上的震荡会趋向于0，而后一项的分量基本同向，把$\boldsymbol{n_i}$视为随机变量：在k足够大时直观上看就是平行于优化目标的平均梯度分量$\overline{\boldsymbol{n}}_i$以$\alpha$的幂的级数情形得以保留，即

$$- \epsilon\sum_{i=0}^{k-1} \alpha^i \boldsymbol{n_{k-i}} \approx - \epsilon \overline{\boldsymbol{n}}_i \sum_{i=0}^{k-1} \alpha^i$$

特别地，当$\alpha = 0.5$时，因为$\sum_{i=1}^{\infty}(1/2)^i = 1$，梯度更新在平行于优化目标的方向也仅以平均梯度分量$\overline{\boldsymbol{n}}_i$进行，通常不会导致结果发散。而一般情况下，因为在$0<\alpha < 1$时，级数$\sum_{i=1}^{\infty}\alpha^i$收敛且和为$\frac{1}{1-\alpha}$，因此其梯度更新将保持在步长为$\frac{\epsilon \overline{\boldsymbol{n}}_i}{1-\alpha}$的情况下进行，这正对应于书中式(8.17)给出的结论。

###8.4 参数初始化策略

*P.257*

由Glorot提议使用的标准初始化方法的基本思想是，设第i-1层与第i层分别有$m_{i-1}$与$m_i$个单元，初始化使两层间权重满足均值为0，方差为$\frac{2}{m_{i-1}+m_i}$的分布，因而：

$$W_{i-1, i} \sim U(w_{i-1,i}; -\sqrt{\frac{6}{m_{i-1}+m_i}},\sqrt{\frac{6}{m_{i-1}+m_i}})\; or\; \mathcal{N}(w_{i-1,i}; 0, \sqrt{\frac{2}{m_{i-1}+m_i}})$$

*P.258*

以缩放准则作为权重的初始化策略是有缺点的，原文举的例子传达的意思是，层的规模本身也会影响到权重初始值的范数，不管是LeCun的初始化策略将权重初始为均值为0，方差为$\frac{1}{m}$的分布，还是Glorot的初始化策略将权重初始为均值为0，方差为$\frac{2}{m_{i-1}+m_i}$的分布，只要其分母m（层的规模，单元数）趋向于无穷，权重也将收敛于$\mathcal{N}(0,0)$。

关于Martens提出的稀疏初始化利弊的讨论结合之前病态Hessian条件（图8.5，图4.6），当固定选择每个单元恰好有k个非零权重时，一旦那些为零的权重更新了一个非常小的量（一个稀疏矩阵的突然更新），可以等效理解为此时的Hessian阵相较于初始时拥有了更大的条件数，结合图4.6不难发现，此时作随机梯度下降的话，路径很容易在这些新生成的具有很大特征值的维度上反复探索而偏离了最初始的优化问题（可以简单理解为运用这种策略后，程序一开始就会先制造一个比原问题更棘手的问题，然后SGD会在优化原问题的同时反复在制造出来的这部分困难上进行探索，因而等到去缩小那些理应被缩小的“大值”时实际已经耗费了过长的时间）。

*P.259*

偏置的设置，第一点说明比较晦涩。举一个简单的例子，当输入为一个设计矩阵X，第一列为均值为166cm的身高，第二列为均值为47kg的体重，那么可以认为该分布是高度偏态的，如果未设置任何偏置，系统从0学习到166和47这两个数字将会是十分漫长的过程。从这个事例可以看出，如果一开始将偏置设置在(166,47)附近有助于系统学习到更符合事实的解(不一定是全局最优)，而从(0,0)到(166,47)的过程里很有可能陷入局部最优而得不到正确的结果。

对于这种问题，一个更简单的方法是将输入进行预处理，例如使所有边缘分布标准化为$\mathcal{N}(0,1)$分布，将这种输入在无偏置设置的结构里学习，和将原输入在设置合适的偏置中学习实际上是等效的。

###8.5 自适应学习算法

*P.260*

以动量更新参数会导致额处的超参数（动量参数）$\alpha$

Delta-bar-delta算法也是源于动量更新的思想：如之前分析动量SGD，将动量写成两个分量的形式。在病态条件下，SGD总向着特征值较大的方向复反探索（第一个动量分量符号总是来回震荡），不过这是基于全局的。如果用小批量来进行更新的话，因为取样的随机性，不一定能得到相同的结果。

*P.261*

AdaGrad算法按照定义来说是用以下方式进行更新：

$$\Delta \boldsymbol{\theta}_i^{(t)} = -\frac{\epsilon}{\sqrt{\sum_{s=1}^{t}(\boldsymbol{g}_i^{(s)})^2}}\boldsymbol{g}_i^{(t)}  \tag{1}$$

其中i为梯度向量对应着的第i个分量。这里的$\sum_{s=1}^{t}(\boldsymbol{g}_i^{(s)})^2$对应着算法8.4中的 $\boldsymbol{r}$，为了防止算法一开始，某些分量的 $\boldsymbol{r}$值过小导致数值上溢，须在分母上加上一个小分量 $\delta$对数值加以稳定，书中建议大约设置在$10^{-7}$。

综合来看，AdaGrad在初始梯度较大的分量上，其梯度更新会由缩放的控制慢慢减小，而在较小初始梯度的分量上，会进行更大的步长。与动量法相比，它不会导致学习到合适的结果后因学习过程中的动量累积不为零而停不下来的效果。但相反，作为一种阻力，它往往是过大的，主要表现在AdaGrad对学习率和初始权重十分敏感，即使初始梯度很大的权重也常常在更新到最佳状态前因缩放致过小导致更新困难。

RMSProp可以看成AdaGrad加上指数形式的权重衰减。其中RMS指均方根(root mean square)，与AdaGrad最大的区别在于使用以下更新参量：

$$\Delta \boldsymbol{\theta}_i^{(t)} = -\frac{\epsilon}{\sqrt{v_{i,t}+\delta}}\boldsymbol{g}_i^{(t)} = -\frac{\epsilon}{RMS[\boldsymbol{g}_i^{(t)}]}\boldsymbol{g}_i^{(t)} \tag{2}$$

其中$\delta$是为防止数值上溢加的一个小量，在泷雅人的深度学习讲义中建议值在$10^{-6}$左右；$v_{i,t}$更新规则为：

$$v_{i,t} = \rho v_{i, t-1}+(1-\rho)(\boldsymbol{g}_i^{(t)})^2$$

这与动量表示类似，根据$v_{i,t}$的递推规则，较早的$(\boldsymbol{g}_i^{(t)})^2$受$(1-\rho)$控制会有指数级的衰减。因此与AdaGrad算法对初始权重十分敏感相比，RMSProp算法在这一点上做出了很好的自适应；但这只解决了AdaGrad算法初始权重过大导致更新困难的问题，事实上RMSProp算法与AdaGrad算法一样，仍是对全体学习率十分敏感的算法类型。

关于AdaGrad算法和RMSProp算法对学习率敏感的事实，无论是式(1)还是式(2)，分别考虑$\Delta\boldsymbol{\theta}_i^{(t)}$和$\boldsymbol{g}_i^{(t)}$的物理意义：前者为系统更新权重的梯度，具有一定量纲，而后者为代价函数对权重的计算梯度，因为代价函数本身是一个无量纲量，则它对权重梯度的量纲正好为权重梯度量纲的倒数。所以严格上说式(1)和式(2)的等式两边由于不同的测度，虽然感觉上很符合直觉，但其实并不匹配。考察下列等式：

$$\Delta \boldsymbol{\theta} = \boldsymbol{H}^{-1} \nabla J(\boldsymbol{\theta}) \sim \frac{1}{\frac{\partial^2 J}{\partial {\theta}^2}} \frac{\partial J}{\partial \theta}$$

$\boldsymbol{H}$是代价函数$J$对权重的Hessian矩阵，因为${\partial^2 J}/{\partial {\theta}^2}$具有权重梯度量纲的倒数平方的量纲，所以等式左右是具有相同量纲的量，这便是下文提到的牛顿参数更新规则(式(8.27))。

但现实中，对Hessian阵求逆的计算开销太大。我们重新观察式(2)，可以将其重新写成：

$$\frac{\Delta \boldsymbol{\theta}_i^{(t)}}{\epsilon} = -\frac{\boldsymbol{g}_i^{(t)}}{RMS[\boldsymbol{g}_i^{(t)}]}$$

尽管$\boldsymbol{g}_i^{(t)}$是有量纲量，但等式右边整体是无量纲量。为了使等式成立，将等式左边的分母用上一步的$RMS[\Delta \boldsymbol{\theta}_i^{(t-1)}]$近似，使全体学习率也使用自适应算法：

$$\epsilon = RMS[\Delta \boldsymbol{\theta}_i^{(t-1)}] = \sqrt{u_{i, t-1} + \delta} \quad (其中 \: u_{i, t-1} = \rho u_{i, t-1} +(1-\rho)(\Delta \boldsymbol{\theta}_i^{(t-1)})^2)$$

再写成式(2)的形式：

$$\Delta \boldsymbol{\theta}_i^{(t)} = -\frac{RMS[\Delta \boldsymbol{\theta}_i^{(t-1)}]}{RMS[\boldsymbol{g}_i^{(t)}]}\boldsymbol{g}_i^{(t)} \tag{3}$$

基于式(3)的这种参数更新方式就是基于RMSProp改进的AdaDelta方法。衰减率$\rho$作者推荐设置为0.95。

Adam更新法是另一种基于RMSProp法的改进。与RMSProp法不同，Adam先直接将式(2)里的$\boldsymbol{g}_i^{(t)}$换成了移动平均梯度$m_{i,t} = \eta m_{i, t-1} +(1-\eta)\boldsymbol{g}_i^{(t)}$，于是式(2)变为：

$$\Delta \boldsymbol{\theta}_i^{(t)} = -\frac{\epsilon}{\sqrt{v_{i,t}+\delta}}m_{i,t} \tag{4}$$

因为一开始不具有任何的动量，可以让初值$m_{i,0}$与$v_{i,0}$都取为0。但即便如此，我们发现基于移动平均的定义式写成的统计量并不是原参数的一个无偏估计，以梯度的移动平均为例：

$$m_{i,t} = \eta m_{i, t-1} +(1-\eta)\boldsymbol{g}_i^{(t)} = \eta^t m_{i, 0} + (1-\eta)\sum_{k=0}^{t-1}\eta^k \boldsymbol{g}_i^{(t-k)} = (1-\eta)\sum_{k=0}^{t-1}\eta^k \boldsymbol{g}_i^{(t-k)} \tag{5}$$

注意这里的$\boldsymbol{g}_i^{(t-k)}$并不是$t-k$次幂，而是第$t-k$次求得的计算梯度。将其视为随机变量，假设该随机变量i.i.d，则有$\forall t-k, \: \mathbb{E}[\boldsymbol{g}_i^{(t-k)}] \equiv \mathbb{E}[\boldsymbol{g}_i^{(1)}]$。式(5)两边取期望后：

$$\mathbb{E}[m_{i,t}] = (1-\eta)\mathbb{E}[\boldsymbol{g}_i^{(1)}]\sum_{k=0}^{t-1}\eta^k = (1-\eta^t)\mathbb{E}[\boldsymbol{g}_i^{(1)}] \neq \mathbb{E}[\boldsymbol{g}_i^{(1)}]$$

将该估计量替换为$\widehat{m}_{i,t} = m_{i,t}/(1-\eta^t)$后有$\mathbb{E}[\widehat{m}_{i,t}] = \mathbb{E}[\boldsymbol{g}_i^{(1)}]$，分母上的$v_{i,t}$也用相同的无偏估计后有：$\widehat{v}_{i,t} = v_{i,t}/(1-\rho^t)$。将新的估计量代入式(4)后有：

$$\Delta \boldsymbol{\theta}_i^{(t)} = -\frac{\epsilon}{\sqrt{\widehat{v}_{i,t}+\delta}}\widehat{m}_{i,t} \tag{6}$$

基于式(6)的更新方式便是Adam法的核心步骤。原论文中的参数建议设置为：

$$\epsilon = 0.001, \quad \eta = 0.9, \quad \rho = 0.999, \quad \delta = 10^{-8}$$

###8.6 二阶近似方法

####8.6.2 共轭梯度

*P.267*

二阶近似方法主要对应着非线性最优化的知识点。原书中有一处表述不当：式(8.29)中的$\nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta})$前应有负号(因为$\boldsymbol{d}_t$更新的方向应靠近负梯度的方向)。

之所以提到二阶近似的方法，需要明确它与一阶方法最主要的两点区别：一、在一阶方法如AdaGrad，RMSProp和Adam，学习率的选取往往是主观的、经验性的；二、一阶方法的迭代次数是没有保证的，往往设计成一个比较大的值，而二阶方法无论是共轭梯度，还是拟牛顿法，都具备在有限步骤(对探索的向量维度相同的量级，如最优化一个N维严格凸函数，共轭梯度所需要的迭代步数法理论上不超过N)达到最优解的特性。

在线性共轭梯度法中，若$\boldsymbol{x}^{t+1} = \boldsymbol{x}^{t} + \alpha_t \boldsymbol{d}_t$中的步长因子$\alpha_t = (\nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta_t}))^\intercal \nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta_t}) /[{(\boldsymbol{d}^t)^\intercal \boldsymbol{A} \boldsymbol{d}^t}]$很难用显示表示出来，则需要用精确(每步都需要求$(\nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta_t}))$与$\boldsymbol{d}^t$的夹角，计算代价较大)或非精确(设置满足的Wolfe条件的步长)的线搜索方式来确定，这种方式叫非线性共轭梯度法。非线性共轭梯度法主要分为两种：在其更新步骤$\boldsymbol{d}_{t} = -\nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta_t}) + \beta_t \boldsymbol{d}_{t-1}$中，若$\beta_t$恒用$\beta_t = (\nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta_t}))^\intercal \nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta_t})/[(\nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta_{t-1}}))^\intercal \nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta_{t-1}})]$的更新方式则称为FR方式，记为$\beta_t^{FR}$；若恒用$\beta_t = [(\nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta_t})-(\nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta_{t-1}}))^\intercal \nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta_t})]/[(\nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta_{t-1}}))^\intercal \nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta_{t-1}})]$则称为PRP方式，记为$\beta_t^{PRP}$。两种更新方式的收敛性在《数学规划》(黄选红，韩继业，2006)中有详细证明过程。注意到当$\nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta_t}) \to \nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta_{t-1}})$且非常小时，$\beta_t^{FR} \to 1$而$\beta_t^{PRP} \to 0$，所以对于FR方式来说，在新的梯度方向上几乎很难更新，因此需要设置在每进行一定的步长后重启线性搜索(即$\beta_t^{FR} $置零)；而对PRP方式而言相当于当梯度足够小时，会自动重启线性搜索过程，因而它在数值计算方面表现较好。

以上都是在严格凸函数的前提下的结论，特别地，若$\boldsymbol{d}_{t} = -\nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta_t}) + \beta_t^{PRP} \boldsymbol{d}_{t-1}$中将$\beta_t^{PRP}$换为$\beta_t^{PRP+}$使$\beta_t^{PRP+} = ReLU(\beta_t^{PRP})$，那么这个方法对非凸函数仍然适用。

*P.269*

牛顿法最大优点是利用了函数的曲率信息(Hessian矩阵)使得收敛速度很快。但Hessian矩阵的计算开销太大，拟牛顿法的核心思想是利用一阶信息去构造一个与原Hessian矩阵近似的矩阵，并由此产生新的探索方向。

设代价函数在$\boldsymbol{\theta}^{t+1}$二阶可微，则对其在$\boldsymbol{\theta}^{t+1}$进行一阶泰勒展开：

$$\nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta}) = \nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta}^{t+1}) + \nabla_{\boldsymbol{\theta}}^2 J(\boldsymbol{\theta}^{t+1})(\boldsymbol{\theta}-\boldsymbol{\theta}^{t+1})$$

将$\boldsymbol{\theta} = \boldsymbol{\theta}^t$代入上式后有：

$$\nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta}^{t+1}) - \nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta})^t = \nabla_{\boldsymbol{\theta}}^2 J(\boldsymbol{\theta}^{t+1})(\boldsymbol{\theta}^{t+1}-\boldsymbol{\theta}^t)$$

使$\nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta}^{t+1}) - \nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta}^t) = \boldsymbol{y}^t$且$\boldsymbol{\theta}^{t+1}-\boldsymbol{\theta}^t = \boldsymbol{s}^t$上式变成$\nabla_{\boldsymbol{\theta}}^2 J(\boldsymbol{\theta}^{t+1}) \boldsymbol{s}^t = \boldsymbol{y}^t$或$\boldsymbol{s}^t = \nabla_{\boldsymbol{\theta}}^2 J(\boldsymbol{\theta}^{t+1})^{-1} \boldsymbol{y}^t$。设$\boldsymbol{B}_{t+1}$与$\boldsymbol{H}_{t+1}$分别为$\nabla_{\boldsymbol{\theta}}^2 J(\boldsymbol{\theta}^{t+1})$和$\nabla_{\boldsymbol{\theta}}^2 J(\boldsymbol{\theta}^{t+1})^{-1}$的近似矩阵，则$\boldsymbol{B}_{t+1} \boldsymbol{s}^t = \boldsymbol{y}^t$和$\boldsymbol{s}^t = \boldsymbol{H}_{t+1} \boldsymbol{y}^t$称为拟牛顿条件。

牛顿法的线搜索过程$\boldsymbol{x}^{t+1} = \boldsymbol{x}^{t} + \alpha_t \boldsymbol{d}_t$与共轭梯度法中的方法一致，而梯度的更新规则为：$\boldsymbol{d}^t = -\boldsymbol{H}_t \nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta}^t)$，在计算下一步的$\boldsymbol{H}_{t+1}$时，不直接进行计算，而是利用$\boldsymbol{g}^{t+1} = \nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta}^{t+1})$对矩阵进行校正(以满足拟牛顿条件)。

校正方式分为两类：对称秩一校正(SR1校正)和对称秩二校正(SR2校正)。

所谓秩一校正，就是在原矩阵上增加一个秩为1的新矩阵，通常为$\boldsymbol{uv}^\intercal$，$\boldsymbol{u}$和$\boldsymbol{v}^\intercal$可以任取使其满足拟牛顿条件：$\boldsymbol{H}_{t+1}\boldsymbol{y}^t = (\boldsymbol{H}_t + \boldsymbol{uv}^\intercal) \boldsymbol{y}^t = \boldsymbol{s}^t$，$\therefore (\boldsymbol{v}^\intercal \boldsymbol{y}^t) \boldsymbol{u} = \boldsymbol{s}^t - \boldsymbol{H}_t \boldsymbol{y}^t$。将$\boldsymbol{u}$代入$\boldsymbol{H}_{t+1} = \boldsymbol{H}_t + \boldsymbol{uv}^\intercal$中有：

$$\boldsymbol{H}_{t+1} = \boldsymbol{H}_t + \frac{(\boldsymbol{s}^t - \boldsymbol{H}_t \boldsymbol{y}^t)\boldsymbol{v}^\intercal}{\boldsymbol{v}^\intercal \boldsymbol{y}^t}$$

已知$\boldsymbol{H}_{t}$对称(严格凸函数前提)，为满足$\boldsymbol{H}_{t+1}^\intercal = \boldsymbol{H}_{t+1}$(对称性)，则可以使$\boldsymbol{v} = \boldsymbol{s}^t - \boldsymbol{H}_t \boldsymbol{y}^t$后有：

$$\boldsymbol{H}_{t+1} = \boldsymbol{H}_t + \frac{(\boldsymbol{s}^t - \boldsymbol{H}_t \boldsymbol{y}^t)\boldsymbol{(\boldsymbol{s}^t - \boldsymbol{H}_t \boldsymbol{y}^t)}^\intercal}{\boldsymbol{(\boldsymbol{s}^t - \boldsymbol{H}_t \boldsymbol{y}^t)}^\intercal \boldsymbol{y}^t}$$

用Sherman-Morrison定理可推出：

$$\boldsymbol{B}_{t+1} = \boldsymbol{B}_t + \frac{(\boldsymbol{y}^t - \boldsymbol{B}_t \boldsymbol{s}^t)\boldsymbol{(\boldsymbol{y}^t - \boldsymbol{B}_t \boldsymbol{s}^t)}^\intercal}{\boldsymbol{(\boldsymbol{y}^t - \boldsymbol{B}_t \boldsymbol{s}^t)}^\intercal \boldsymbol{s}^t}$$

关于SR2校正，李航的《统计学习方法》里的介绍比较容易理解。每次更新矩阵时，都是在原矩阵上附加两个待定矩阵构成：$\boldsymbol{G}_{t+1} = \boldsymbol{G}_t + \boldsymbol{P}_t + \boldsymbol{Q}_t$。

当待更新矩阵为$\boldsymbol{G}_{t+1} = \boldsymbol{H}_{t+1}$时，对应着DFP算法(应用的拟牛顿条件为$\boldsymbol{s}^t = \boldsymbol{H}_{t+1} \boldsymbol{y}^t$):

$$\boldsymbol{H}_{t+1} \boldsymbol{y}^t = \boldsymbol{H}_t \boldsymbol{y}^t + \boldsymbol{P}_t \boldsymbol{y}^t + \boldsymbol{Q}_t \boldsymbol{y}^t$$

使$\boldsymbol{P}_t \boldsymbol{y}^t = \boldsymbol{s}^t$以及$\boldsymbol{Q}_t \boldsymbol{y}^t = - \boldsymbol{H}_t \boldsymbol{y}^t$使拟牛顿条件成立，不难发现可以取：

$$\boldsymbol{P}_t = \frac{\boldsymbol{s}^t\boldsymbol{s}^{t\intercal}}{\boldsymbol{s}^{t\intercal}\boldsymbol{y}^t}; \quad \boldsymbol{Q}_t = -\frac{\boldsymbol{H}_t \boldsymbol{y}^t \boldsymbol{y}^{t\intercal}\boldsymbol{H}_t}{\boldsymbol{y}^{t\intercal}\boldsymbol{H}_t \boldsymbol{y}^t}$$

则$\boldsymbol{H}_{t+1}$校正公式为：

$$\boldsymbol{H}_{t+1}^{DFP} = \boldsymbol{H}_t - \frac{\boldsymbol{H}_t \boldsymbol{y}^t \boldsymbol{y}^{t\intercal}\boldsymbol{H}_t}{\boldsymbol{y}^{t\intercal}\boldsymbol{H}_t \boldsymbol{y}^t} + \frac{\boldsymbol{s}^t\boldsymbol{s}^{t\intercal}}{\boldsymbol{s}^{t\intercal}\boldsymbol{y}^t}$$

与之对应的$\boldsymbol{B}_{t+1}$用Sherman-Morrison定理可推出：

$$\boldsymbol{B}_{t+1}^{DFP} = (\boldsymbol{I} - \frac{\boldsymbol{y}^t\boldsymbol{s}^{t\intercal}}{\boldsymbol{y}^{t\intercal}\boldsymbol{s}^t}) \boldsymbol{B}_t (\boldsymbol{I} - \frac{\boldsymbol{y}^t\boldsymbol{s}^{t\intercal}}{\boldsymbol{y}^{t\intercal}\boldsymbol{s}^t})^\intercal + \frac{\boldsymbol{y}^t\boldsymbol{y}^{t\intercal}}{\boldsymbol{y}^{t\intercal}\boldsymbol{s}^t}$$

当待更新矩阵为$\boldsymbol{G}_{t+1} = \boldsymbol{B}_{t+1}$时，对应着BFGS算法(应用的拟牛顿条件为$\boldsymbol{B}_{t+1} \boldsymbol{s}^t = \boldsymbol{y}^t$)。类似上述求解过程可求得：

$$\boldsymbol{B}_{t+1}^{BFGS} = \boldsymbol{B}_t - \frac{\boldsymbol{B}_t \boldsymbol{s}^t \boldsymbol{s}^{t\intercal}\boldsymbol{B}_t}{\boldsymbol{s}^{t\intercal}\boldsymbol{B}_t \boldsymbol{s}^t} + \frac{\boldsymbol{y}^t\boldsymbol{y}^{t\intercal}}{\boldsymbol{y}^{t\intercal}\boldsymbol{s}^t}$$

$$\boldsymbol{H}_{t+1}^{BFGS} = (\boldsymbol{I} - \frac{\boldsymbol{s}^t\boldsymbol{y}^{t\intercal}}{\boldsymbol{s}^{t\intercal}\boldsymbol{y}^t}) \boldsymbol{H}_t (\boldsymbol{I} - \frac{\boldsymbol{s}^t\boldsymbol{y}^{t\intercal}}{\boldsymbol{s}^{t\intercal}\boldsymbol{y}^t})^\intercal + \frac{\boldsymbol{s}^t\boldsymbol{s}^{t\intercal}}{\boldsymbol{s}^{t\intercal}\boldsymbol{y}^t}$$

在数值优化中计算性能比较好的，最流行的拟牛顿方法是BFGS方法，但观察上式的递推关系，其计算复杂度为$O(t^2)$，不适用于深度学习的架构。从BFGS中改进的[L-BFGS](https://zhuanlan.zhihu.com/p/29672873)只保存最近m次的$\boldsymbol{s}^t$和$\boldsymbol{y}^t$，自动删除输早的项$\boldsymbol{s}^{t-m}$和$\boldsymbol{y}^{t-m}$并保持$\boldsymbol{s}^t = \boldsymbol{x}^{t+1} - \boldsymbol{x}^t$和$\boldsymbol{y}^t = \nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta}^{t+1}) - \nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta}^t)$，从而使计算复杂度降到$O(t)$。

###8.7 优化策略和元算法

####8.7.1 批标准化

*P.271*

关于反向传播批标准化的操作：对于某一个神经层而言，起重要作用的并不是其权重真实值的绝对大小，而是该权重与单元其它权重相比的相对大小(或强弱)，真实值的大小只是这种性质的具体表现。如果每次迭代都采用显式，使用固定的学习率，在经过有限步更新之后权重的真实值会下降至非常小(而造成梯度消失)。而每次将更新后的权重重新标准化之后，既保证了权重之间的相对强弱关系，不会因为迭代而改变某一层权重的尺度。

####8.7.6 延拓法和课程学习

*P.278*

本质上来看，之前提到的代价函数都是显式定义的，是具体的。而“模糊”原来的代价函数可以理解为通过原来在一个“代价函数族”中选取，优化。例如先选择了$\boldsymbol{J^{(i)}}(\boldsymbol{\theta})$优化至$\boldsymbol{\theta}^{(k)}$点，那么对于$\boldsymbol{J^{(i)}}(\boldsymbol{\theta})$的描述而言，$\boldsymbol{\theta}^{(k)}$已经落入一个局部最优的状态，无法继续以$\boldsymbol{J^{(i)}}(\boldsymbol{\theta})$的方式进行更新。因此重新选择一个$\boldsymbol{J^{(i+1)}}(\boldsymbol{\theta})$，这相当于改变了代价函数的形态，对于$\boldsymbol{J^{(i+1)}}(\boldsymbol{\theta})$而言，上一步优化得到的$\boldsymbol{\theta}^{(k)}$点并不一定是最优状态，因而具有继续优化下去的驱动力。

##第九章 卷积网络

###9.1 卷积运算

*P.282*

使式(9.4)中的$i - m = m'$以及$j - n = n'$代回原式即可得到“翻转”后的式(9.5)。同样，如果把$m' = -m''$以及$n' = -n''$代回式(9.5)将会得到：

$$S(i,j) = (K \ast I)(i,j)=\sum_{m’’} \sum_{n''} I(i+m'',j+n'')K(-m'',-n'')$$

这同样也是由式(9.4)经“翻转”而来，核与输入的索引之间此消彼长的性质依旧不变。若不对核进行翻转，我们用$m = -m''$以及$n = -n''$仅代回上式的核函数K，并将输入里的$m''$和$n''$都写为$m$和$n$的形式就得到了不具备翻转性质的式(9.6)，可看到输入与核函数的索引同增同减的性质。

###9.2 动机

*P.284*

在稀疏连接中，假设输入和输出都为n，且不存在激活函数的情况下，假设对于任意一个输出，其接受域为3，那么对于其权重矩阵$\boldsymbol{W} \in \mathbb{R}^{n \times n}$，$\forall |i-j| > 2, \quad w_{ij} = 0$，$\boldsymbol{W}$为三对角矩阵，非零元素大量集中在对角线附近，其排列是稀疏的。

*P.285*

等变性质：设$I(i,j)$经平移$(a,b)$后变为$I(i'-a,j'-b)$，所以$I(i'-a,j'-b) = I(i,j)$，根据式(9.6)：

$$S(i'-a,j'-b) = \sum_{m} \sum_{n} I(i'-a+m,j'-b+n)K(m,n) = \sum_{m} \sum_{n} I(i+m,j+n)K(m,n) = S(i,j)$$

$S(i'-a,j'-b) = S(i,j)$，所以原图先经平移再变换后的结果，和原图直接变换后的结果，仍具有平移性质。

*P.287*

把核函数看成一个filter，参数共享决定了我们不用针对输入图片像素的幂次级别的参数数进行逐个学习，而是可以仅靠习得filter内有限像素的合适参数，获得一个满足任务要求的核。如果我们用的是同一个核，在输入的不同位置如果参数不一样，那么该核对模式的识别能力也将因位置而发生变化，这是机器学习任务所不希望看到的。参数共享能很好地解决这个问题。以图9.5的上图为例，参数共享决定了$s_2$对$x_1$、$x_2$及$x_3$的提取方式与$s_3$对$x_2$、$x_3$及$x_4$的提取方式一致，不因位置的关系而产生差异(如$s_2 = a \cdot x_1 + b \cdot x_2 + c \cdot x_3$则一定$s_3 = a \cdot x_2 + b \cdot x_3 + c \cdot x_4$)。如此看来无论输入有多大，卷积层能够实现”在有限接受域内“提取特征的功能的同时，又保持了这种特性的平移等变性。

###9.5 基本卷积函数的变体

*P.296*

带”步幅“(stride)的卷积计算公式有误，应为：

$$Z_{i,j,k} = c(\boldsymbol{K},\boldsymbol{V},s)_{i,j,k} = \sum_{l,m,n}[V_{l,js+m-1,ks+n-1}K_{i,l,m,n}]$$

*P.300*

标准卷积是对一种模式的提取。在图像中，我们几乎可以认为在数个像素点的变化中图像是基本不变的。可以看出，平铺卷积的意义在于，在针对局部的“基本不变”的输入，仅使用一次卷积的代价，同时对多种特征进行提取。

*P.301*

带”步幅“(stride)的反向传播公式有误，注意i为输入的层索引，j为输出的层索引，k和l对应了输入中的绝对位置(宽和高的像素坐标)，式(9.11)对应着式(9.8)应写为：

$$
\begin{align}
g(\boldsymbol{G},\boldsymbol{V},s)_{i,l,m,n}
&= \frac{\partial}{\partial K_{i,l,m,n}}J(\boldsymbol{V},\boldsymbol{K}) = \sum_{j,k} \frac{\partial J(\boldsymbol{V},\boldsymbol{K})}{\partial Z_{i,j,k}} \frac{\partial Z_{i,j,k}}{\partial K_{i,l,m,n}} \\
&= \sum_{j,k}[G_{i,j,k}V_{l,js+m-1,ks+n-1}]
\end{align}
$$

这个结果是对应着代价函数对卷积层权重的梯度，更一般地，我们希望得到代价函数对被卷积层的梯度，则有：

$$\frac{\partial}{\partial V_{l,a,b}}J(\boldsymbol{V},\boldsymbol{K}) = \sum_{i,j,k} \frac{\partial J(\boldsymbol{V},\boldsymbol{K})}{\partial Z_{i,j,k}} \frac{\partial Z_{i,j,k}}{\partial V_{l,a,b}}
= \sum_{i,j,k}[G_{i,j,k} \sum_{m\; s.t.\\m=a+1-js} \sum_{n\; s.t.\\n=b+1-ks} K_{i,l,m,n}]$$

上式的各下标与之前的推导保持一致，而且比式(9.13)容易理解得多。按定义式来看该运算涉及大量递归的线性运算的过程。

虽然卷积运算定义式支持完整的反向传播运算，但通过上式也看出，对完整的$V_{l,a,b}$求和后涉及到对两张量的共计八次求和操作，其运算代价相当大，很难适用大规模的训练模型。如9.9节中提到，卷积网络常用来无监督地提取样本特征，然后再用相对低的代价进行训练。

##第十章 序列建模：循环和递归网络

###10.8 回声状态网络

*P.345*

实对称矩阵四大性质：1.不同特征值对应的特征向量都是正交的；2.特征值都是实数，特征向量都是实向量；3.必可对角化，且对角元素为特征值；4.如果存在k重特征值，则必有k个线性无关的特征向量。

这里关于复数系数幅值的讨论，可以分别以```matrix(c(1,1,-1,1),2,2,T)```和 ```matrix(c(0.1,0.1,-0.1,0.1),2,2,T)```(R语言)为例，反复乘以任一个二维向量并观察其变化。

###10.10 长短期记忆和其它门控RNN

####10.10.1 LSTM

*P.349*

*P.399*

$$\sum_{i=1}^{|\mathbb{V}|}P(i|C)\frac{\partial a_i}{\partial \theta} = \sum_{i=1}^{|\mathbb{V}|} \frac{P(i|C)}{Q(i|C)}\frac{\partial a_i}{\partial \theta}Q(i|C) = \sum_{i=1}^{|\mathbb{V}|}\omega_i \frac{\partial a_i}{\partial \theta}Q(i|C) = E_Q[\omega_i \frac{\partial a_i}{\partial \theta}] \approx \frac{1}{m}\sum_{i=1}^{|\mathbb{V}|} \omega_i \frac{\partial a_i}{\partial \theta}$$

#第三部分

##第十三章 线性因子模型

###13.2 独立成份分析

*P.419*

式3.47描述的模型用公式表示：$p_{\boldsymbol{h}}({\boldsymbol{h}}) = p_{\boldsymbol{x}}(\boldsymbol{W}^{-1}\boldsymbol{x})|det(\frac{\partial{\boldsymbol{W}^{-1}\boldsymbol{x}}}{\partial{\boldsymbol{x}}})| = p_{\boldsymbol{x}}(\boldsymbol{W}^{-1}\boldsymbol{x})|det((\boldsymbol{W^{-1}})^{\intercal})|$。所谓代价很高且不稳定的操作，一定程度上是指设计矩阵$\boldsymbol{W}$存在唯一解且可逆在真实情况中是十分少见的。如果不对其施加一定的约束，则通过计算式很难得到$p_{\boldsymbol{h}}({\boldsymbol{h}})$的估计。

ICA的所有变种之所以要求$p(\boldsymbol{h})$是非高斯的，因为ICA本身就是一种线性模型。$\boldsymbol{x} = \boldsymbol{Wh}$就是最典型的线性变换，因而$p(\boldsymbol{h})$一定也是对应维度的线性平面。如果$p(\boldsymbol{h})$是高斯的，那么一定会具有$p(\boldsymbol{h}) \sim \exp[\boldsymbol{-\frac{1}{2} \cdot (Ax^2+Bx)}]$这样的结构(非线性)，这是很难用$\boldsymbol{W}$来进行直接表述的。

关于非高斯分布在0附近有比高斯分布更高的峰值。可以用R的mvtnorm包进行验证，大多数情况下，如果均值向量偏离原点$\boldsymbol{0}$向量过多，而参数$\Sigma$的范数又不是非常大的情况下，用```rmvnorm()```去生成高维随机向量时，$\boldsymbol{0}$点附近的值几乎很难被取到。

###13.3 慢特征分析

SFA在深度非线性条件下预测学习特征，必须知道关于配置空间的环境动力。这种算法更像是一种对已有特理模型抓取的一个辅助参数，而却很难保证其自身的独立工作。“代价函数高度依赖于特定像素值”，简而言之就是图像中的对象(object)大多数情况下都是非线性的（比如书中所提到的斑马，在视频中需要获取其轮廓就是一个非线性对象）。先运用其它的图形技术将每一帧的轮廓选取出来，再用SFA算法训练来抓取慢性特征是可行的（运动规律，或者3D渲染环境某些程度上都会有一定的线性准则），然而直接对各帧的各像素点用SFA的话，很有可能每一帧都得到奇形怪状的轮廓，所以输出难以保证。

###13.4 稀疏编码

*P.423*

Laplace分布是指数分布向负数域扩展的形式，有位置参数$\mu$和尺度参数$\lambda$。其概率密度的完全形式为$p(h_i) = Laplace(h_i; \mu, \lambda) = \frac{1}{2\lambda} \exp(-\frac{|h_i - \mu|}{\lambda})$，分布密度图像具有相当的尖度（考虑指数分布0的左极限的情况下导数并不为0）。用Laplace先验可以产生稀疏解。

t分布有三个参数：$\nu > 0$为自由度；$\mu \in \boldsymbol{R}^1$为位置参数；$\sigma > 0$为尺度参数。其概率密度完全形式为：

$$p(h_i; \nu, \mu, \sigma^2) = \frac{\Gamma(\frac{\nu+1}{2})}{\Gamma(\frac{\nu}{2})\sqrt{\nu \pi} \sigma}[1 + \frac{1}{\nu} (\frac{h_i - \mu}{\sigma})^2]^{-\frac{\nu+1}{2}}$$

其中$[1 + \frac{1}{\nu} (\frac{h_i - \mu}{\sigma})^2]^{-\frac{\nu+1}{2}}$是分布的核。取$\mu = 0; \sigma = 1$即可获得如式(13.14)的结果。

带有激活值的特征字典：一个字典结构，记录了所有特征的激活值($x_i^\prime = ReLU(x_i - T_{x_i})$ for all $i$)。相当于进行模型训练前的一个筛选器，用MAP推断时，生成的值本身就服从一个连续的概率分布，不需要人为地对输入数据进行挑选，因此激活值设置为0。

###13.5 PCA的流形解释

*P.426*

线性编码器和解码器背后所对应的模型就是PCA模型。通俗来说，它将高维数据投向一个各变量相互间关联度最小的方向（该步骤对应着奇异值分解），找到最大的特征值，其对应的特征向量$\boldsymbol x$存在一个生成子空间$\boldsymbol{xx}^\intercal$，能够以最小的代价来最大限度地重构观察数据。

##第十四章 自编码器

###14.2 正则自编码器

如第七章所示，根据所添加的正则化项的不同，学习到的模型（这里指编码器和解码器）也会具有不同的特性。例如R语言中构建模型的过程就是这种构建自编码器的过程。

####14.2.1 稀疏自编码器

*P.432*

关于Student-t先验诱导稀疏性的原理：观察t分布的核$[1 + \frac{1}{\nu} (\frac{h_i - \mu}{\sigma})^2]^{-\frac{\nu+1}{2}}$取$\mu = 0; \sigma = 1$时会有$p(h_i) \sim (1+\frac{h_i^2}{\nu})^{-\frac{\nu + 1}{2}}$，观察其负对数似然：

$$-log\ p_{model}(\boldsymbol{h})=\frac{\nu + 1}{2} \sum_i log (1+ \frac{h_i^2}{\nu})$$

其中每一项$log (1+ \frac{h_i^2}{\nu})$都是在$[1,+\infty)$间的单调增函数，当具仅当$h_i = 0$时，$log (1+ \frac{h_i^2}{\nu})$取到最小值0。因此最小化$-log\ p_{model}(\boldsymbol{h}, \boldsymbol{x})$的过程中，针对$-log\ p_{model}(\boldsymbol{h})$一项的优化会倾向于选择含有更多0成份的$\boldsymbol{h}$。

###14.5 去噪自编码器

####14.5.2 历史展望

*P.439*

DAE动机是允许学习容量很高，也就是具有更好泛化性能的AE，而不是仅在当前数据集中表现良好但却没有任何提取的抽象特征（无用的恒等函数）。

###14.6 使用自编码器学习流形

*P.440*

一维流形就是一条处处可微的曲线（每指定一个自变量，因变量也就因此确定），其中每一点的导数（切线）就是这样的该点的“切平面”。同理，二维流形可简单看作空间曲面，每指定两个自变量，因变量也会因此确定，三维流形同理。

理解这一点对理解图像识别的工作原理有很重要的意义。如图14.6所示，很多图像其实都是高维空间的向量。通过对一维流形的学习，我们能够获得图像的平移不变性等特征；通过二维流形的学习，如图14.8所示，能够学习到高维空间数据（图像）在一个低维曲面上的各种投影。例如我们点头或者摇头，都是在一个二维的流形曲面运动所产生的变化。

###14.6 使用自编码器学习流形

*P.442*

最邻近图的非参方法之所以有效，是因为流形在局部上与欧氏空间同胚的性质。但也有其局限性，即当涉及到相当复杂的流形结构的时候，需要大量的样本点作为代价。

###14.7 收缩自编码器

*P.444*

将Jacobian的Frobenius norm作为惩罚项，鼓励学习到更平稳的流形变化，而这种变流形往往具备更好的泛化性能。

考虑空间某一点$\boldsymbol x$，设其邻域内有另一点$\boldsymbol x^\prime$，因为流形在局部上与欧氏空间的同胚性质，我们可以得出$\boldsymbol x^\prime = \boldsymbol{x} + \boldsymbol{J}(\boldsymbol{x^\prime - x})$（线性理论），如果$\boldsymbol J$是收缩的，则模型鼓励在$\boldsymbol x$邻域内的点不要有太大梯度。

关于术语“收缩”的理解，可以类比到一维情形：一个具有扰动的时序数据，例如股价走势图，基斜率大体是按照“正负”的情形不断重复排列的，因此原始数据相邻两点的梯度变化也是相当大的。收缩自编码器的过程，相当于对这样的原始数据进行平滑(smoothing)的处理操作。利用二范数正则项（也就是一维情况下的最小二乘，高维情况下还可能利用其它更复杂的计算方式），学习到的平滑后曲线（一维流形）往往居于不断扰动的数据中心。原始数据的每个点向平滑曲线作垂线的过程就像是原始数据向曲线“收缩”的过程。高维情况下也是同理，原始数据点是十分离散的，利用收缩自编码器，可以平滑出一个较为“平稳”的高维曲面，而如果将所有数据点对该曲线作垂线，而其直观感受也如同数据“收缩”至目标流形的过程。

*P.445*

图14.10：CAE利用不同位置的参数共享，从而能够从有限数据集中学习到更准确的估计，而对于局部主成份分析（local PCA），分析区域每变换一个位置，其主成份都是不一样的，因此虽然物体位于流形的同一位置，图形的每个位置都需要执行一次PCA算法，这显然是低效的。

###14.7 预测稀疏分解

*P.446*

式（14.19）可以看出，中间项$\lambda|\boldsymbol h|_1$利用L1正则化的方式决定着稀疏的特征的选择，而$||\boldsymbol x - g(\boldsymbol h)||^2$与$||\boldsymbol h - f(\boldsymbol x)||^2$利用L2正则项的方式分别优化学习编码器和解码器。

##第十五章 表示学习

###15.1 贪心逐层无监督预训练

*P.450* & *目录 xxi*

目录中的第二项$f \circ g$原文是composition of function。这里应为复合函数的意思。

算法15.1解析：该算法可以简单想象为一个MLP优化器。第一步$f$ <- Identity function对应于初始化的函数，$\widetilde{\boldsymbol X} = \boldsymbol X$将原始数据送入第一层。在第一层for循环中，循环内的第一步通过$\mathcal L$学习到当前输入状态的表示，并将当前层中，输入状态到这种表示的映射函数保存为$f^{(k)}$。循环内第二步将复合函数$f^{(k)}(f)$保存为函数$f$，第三步设定当前层的输入为下一层的输入，不断重复上述过程来进行优化。不求整体最优，而是逐层做到最优以达到整体最优的一种近似，就是贪心算法的思想。

在执行精调的情况下，完成上述过程后，还整体考虑$f,\boldsymbol X, \boldsymbol Y$重新调整$f$使整体具有更优化的解。这里需要注意，如果没有for循环的情况下执行精调，最终得到的结果高度依赖于初始化的函数$f$，也就是说很容易落入局部最优。

####15.1.1 何时以及为何无监督预训练有效？

*P.454*

原文：Except in the field of natural language processing, where the natural representation of words as one-hot vectors conveys no similarity information and where very large unlabeled sets are available.

除了应用在自然语言处理领域，因为在该领域，需要处理的数据是经过one-hot稀疏编码的各种具有互斥特征的单词，而且这样的单词大多数未经标注。

###15.3 半监督解释因果关系

*P.462*

式15.3中，每一个$\boldsymbol x$都是一个样例，每一个$\boldsymbol y$都是一个标签或者标注，它们都是容易显而易见的(explicit)。分子上是先验分布与先验下的条件分布，也是可以通过简单的统计手段获取。而唯有$p(\boldsymbol x)$是描述观察样本$\boldsymbol x$内在分布规律的一种隐性表述，它可能来自某一个概率分布。因此对数据集建模（统计模型）的准确性与否，直接影响着模型预测的准确性。

再回到上文的例子，如果$\boldsymbol x$本来就倾向于某一些取值，如果不加判断冒然将$p(\boldsymbol x)$设置成均匀分布，相当对原始的概率分布进行人为地修饰，对某些高度可能的取值视而不见，最终导致模型失真。

*P.464*

生成对抗网络相当于在原有生成器的基础上，利用固有模式去纠正那些容易被原算法忽略的细节。这相当于模型生成中的“错题集”，不断让模型对判断失误的地方加深记忆，从而提升模型容量和准确性。

###15.4 分布式表式

*P.468*

分布式表示的优势在于，它学习的目标或标签并没有被定义为最终的任务目标，相反却是一些简单的基础概念。各种复杂的最终任务目标，都可以通过排列组合这些简单的基础概念来进行提取。比如文中对红色汽车，绿色卡车的举例就是很好的例子。让机器识别红色汽车可能是困难的，但大而化小，让机器识别颜色，车型这两种属性相对而言容易得多（训练中也是如此，同样的一张训练图像，可以因不同的学习任务而被反复调用从而节省实验样本量）。当所有的标注都完成后，需要找到“红色汽车”的对象时，只需要对标签为“红色”以及对象为“汽车”的样本集取交集就可以输出最终结果。

##第十六章 深度学习中的结构化概率模型

###16.2 使用图描述模型结构

####16.2.2 无向模型

*P.483*

图16.3中，$\{h_r, h_y\}$与$\{h_y, h_c\}$都是团，但$\{h_r, h_y, h_c\}$不是，因为$h_r$与$h_c$之间缺少连接。

按概率论的观点来看，每一个团都适用于一个多维正态分布。这也是团包含了全连接隐藏条件的原因。例子中的团都只含有两个元素。但倘若含有三个或三个以上的元素的情况下，用均值和协方差矩阵的描述各元素的数据分布状态显然是合乎逻辑的（任意单一元素都有方差，任意一对不同元素对之间也都因为全连接百都存在不为0的协方差）。

从这一点上看，无向概率图模型最核心的精神，是把针对一个复杂情况的建模，拆解为一系列简单的概率模型耦合的情况。

####16.2.4 基于能量的模型

*P.485*

统计力学中，系统中粒子在各种可能的微观态的概率分布有如下形式：

$$F(state) \propto e^{-\frac{E}{kT}}$$

其概率分布可表达为：

$$p_i = \frac{e^{-\epsilon_i/kT}}{\sum_{j=1}^{M}e^{-\epsilon_j/kT}} = \frac{e^{-\beta\epsilon_i}}{\sum_{j=1}^{M}e^{-\beta\epsilon_j}}$$

其中$\epsilon_i$是量子态$i$的能量，$k$是波尔兹曼常数，$T$是系统温度且$M$为系统具有的量子态数目。

观察softmax函数的形式，就不难理解为何很多应用在机器学习里的模型都可以被称为“波尔兹曼机”。另外，回到式10.18中的笔记，很容易理解为什么要构建负对数似然的损失函数后再进行softmax形式的归一化处理。所以在下文中以$-log\ \tilde{p}_{model}(\boldsymbol x)$的提法也理所应当。

*P.489*

d-分离可以看成条件推断的阻塞。在给定$c$的情况下，明确$p(a,b|c) \neq p(a|c)p(b|c)$；因此$a$和$b$不是d-分离的。同理$p(a,b|d) = p(d)p(c|d)p(a,b|c) \neq p(d)p^2(c|d)p(a|c)p(b|c) = p(a|d)p(b|d)$，因此在给定$d$的情况下，$a$和$b$也不是d-分离的。

*P.492*

图16.12中将无向图转为有向图之后，$p(b,d|a,c) = p(d|a,c)p(b|a,c) = p(d|a,c)p(b|a)$，$p(a,c|b,d) = p(a|b,d)p(c|b,d,a) = \{[p(b,d|a)p(a)]/p(b,d)\}p(c|b,a)$。所以当已知$a$和$c$时，$b$和$d$是d-分离的；而当已知$b$和$d$时，$a$和$c$不是d-分离的。

###16.3 从图模型中采样

*P.494*

成本很高的多次迭代过程，这里的成本涉及两个方面：第一是生成样本的选择问题，为了使生成样本更接近于原来的概率分布，必然有一部分不满足条件的样本被丢弃，可以借鉴Metropolis-Hastings算法中的思想；另外一个成体是其于多维变量而言的，当其它维度确定的时候，当前维度所服从条件分布（满条件分布）的推断问题。

Gibbs采样是基于多元高斯分布的随机生成方法。之所以说理论上Gibbs采样是最简单的方法，是因为其具备两个特点：一、它是特化的，接受率为1的Metropolis-Hastings算法，因此不存在舍弃样本的问题；二、它的所有满条件分布都会退化成某一个特定的一元正态分布，并且参数十分都是很容易计算的，因此只要应用正态分布的随机生成器就可以进行更新参数的过程。

注：$p(x_i|x_{-i})$所表示的就是$x_i$的满条件分布。

###16.7 结构化概率模型的深度学习方法

*P.499*

注意，$\boldsymbol{v}^\intercal \boldsymbol{W}_{:,i}$中的$\boldsymbol{W}_{:,i}$是一个具体的向量，与$\boldsymbol{v}$的内积为一标量。

$\frac{\partial}{\partial \boldsymbol W} E(\boldsymbol{v}, \boldsymbol h) = -\boldsymbol{vh}^\intercal$，因此$\frac{\partial}{\partial W_{i,j}} E(\boldsymbol{v}, \boldsymbol h) = - v_i h_j$。

##第十七章 蒙特卡罗方法

###17.2 重要采样

*P.503*

式17.8的意义在于，$p(\boldsymbol x)$并不一定是一个很标准的多维随机分布，因而很难从$p(\boldsymbol x)$生成随机数来对$f(\boldsymbol x)$作出估计。而改写成：

$$p(\boldsymbol x)f(\boldsymbol x) = q(\boldsymbol x)\frac{p(\boldsymbol x)f(\boldsymbol x)}{q(\boldsymbol x)} = q(\boldsymbol x) f^\prime (\boldsymbol x)$$

后，$q(\boldsymbol x)$可以来自于一个简单的多维分布，例如多元高斯分布。用这种情况下生成的随机数去估计$f^\prime$，与用$p(\boldsymbol x)$（样本分布）的随机数来估计$f$是等效的。

###17.3 马尔可夫链蒙特卡罗方法

*P.508*

尽量消除样本间的相关性，是基于“独立同分布”的假设。一般作ACF图诊断，随着间隔样本数的增加，样本间的自相关性也呈下降的趋势。但由于间隔样本数的增加，以马尔可夫链为基础生成随机数的方法需要舍弃大量样本，因此计算代价也非常高。

##第十八章 直面配分函数

###18.1 对数似然梯度

*P.516*

式18.14来自于莱布尼兹变限积分法则（[Leibniz integral rule](https://en.wikipedia.org/wiki/Leibniz_integral_rule)）。其中关于第一个条件的限制，是由于未归一化分布$\tilde p$可能是很复杂的情况且并不一定是黎曼可积的，因此用勒贝格可积加以限制。第二，三个条件相当于对被积函数的连续性等性质加以限制。

###18.2 随机最大似然和对比散度

*P.518*

两个for循环里的内容相当于体系随机游走了k步，这里用了吉布斯算法针对每一个样本来进行更新。“设吉布斯步数k大到足以让从p采样的马尔可夫链磨合”。MCMC方法会存在初始值和预烧期(burn-in period)，根据初始值的设定的不同，预烧期所持续的时间也可能会不一样。当采样过了预烧期过后，马氏链才会趋于稳定。

对比算法18.1~18.3，其区别在于：第一个使用随机初始值，第二个使用样本初始值，第三个针对样本使用随机初始化（while循环中不会涉及重置马氏链，经过一定次数的循环后基本能保证马氏链走出预烧期）。

###18.4 得分匹配和比率匹配

*P.525*

式18.22原书中写法为$L(\boldsymbol x, \boldsymbol{\theta}) = 0.5 ||\nabla_{\boldsymbol x}log\ p_{model}(\boldsymbol x; \boldsymbol{\theta}), -\nabla_{\boldsymbol x}log\ p_{data}(\boldsymbol x)||_2^2$，意为从$\nabla_{\boldsymbol x}log\ p_{model}(\boldsymbol x; \boldsymbol{\theta})$到$-\nabla_{\boldsymbol x}log\ p_{data}(\boldsymbol x)$的欧氏距离的平方。式18.25根据此结果推断出来。但根据原文*"the expected squared diﬀerence between the derivatives of themodel’s log density with respect to the input and the derivatives of the data’s logdensity with respect to the input"*推测式18.22其中的二范数应为$\nabla_{\boldsymbol x}log\ p_{model}(\boldsymbol x; \boldsymbol{\theta}) -\nabla_{\boldsymbol x}log\ p_{data}(\boldsymbol x)$。

在Hyvarinen的论文里，得分函数被定义为（原论文中变量用$\boldsymbol{\xi}$表示，数据用$\boldsymbol x$表示）：

$$
\boldsymbol{\psi}(\boldsymbol{x}; \boldsymbol{\theta}) =
\begin{bmatrix}
    \frac{\partial log\ p(\boldsymbol{x}; \boldsymbol{\theta})}{\partial x_1} \\
    \vdots \\
    \frac{\partial log\ p(\boldsymbol{x}; \boldsymbol{\theta})}{\partial x_n}
\end{bmatrix} =
\begin{bmatrix}
    \psi_1(\boldsymbol{x}; \boldsymbol{\theta})\\
    \vdots \\
    \psi_n(\boldsymbol{x}; \boldsymbol{\theta})
\end{bmatrix} = \nabla_{\boldsymbol{x}}log\ p(\boldsymbol x; \boldsymbol{\theta})
$$

所以式18.22也可以写为:

$$
\begin{align}
L(\boldsymbol x, \boldsymbol{\theta}) = &\frac{1}{2}||\boldsymbol{\psi}(\boldsymbol x, \boldsymbol{\theta})-\boldsymbol{\psi_{data(x)}}(\boldsymbol x)||_2^2
\end{align}
$$

式18.25的由来需要$\boldsymbol{\psi}$可微，并且附加一些弱正则化条件。

##第十九章 近似推断

###19.3 最大后验推断和稀疏编码

*P.541*

当$\boldsymbol h \equiv \boldsymbol{\mu}$时，$\int q(\boldsymbol h | \boldsymbol v) = \int \delta (\boldsymbol h - \boldsymbol{\mu}) = 1$。

###19.4 变分推断和变分学习

####19.4.1 离散型潜变量

*P.547*

因为$\boldsymbol h$是贝努利的，所以根据sigmoid函数的对称性，式19.19隐含着条件：$p(h_i = 0) = 1 - \sigma(b_i) = \sigma(-b_i)$，所以：

$$
\begin{align}
&\mathbb{E}_{\boldsymbol{h} \sim q}[\sum_{i=1}^m \log p(\boldsymbol{h}_i) - \sum_{i=1}^m \log q(h_i | \boldsymbol{v})]\\
= &\sum_{i=1}^m[\sigma(b_i)[\log \sigma(b_i) - \log \hat{h}_i] + \sigma(-b_i)[\log \sigma(-b_i) - \log(1-\hat{h}_i)]]\\
= &\sum_{i=1}^m[\hat{h}_i[\log \sigma(b_i) - \log \hat{h}_i] + (1 - \hat{h}_i)[\log \sigma(-b_i) - \log(1-\hat{h}_i)]]
\end{align}
$$

另，原书和翻译中的式19.34中中括号里的值展开后为：

$$\frac{1}{2}\sum_{i=1}^{n}[\log\frac{\beta_i}{2\pi}-\beta_i(v_i^2 - 2v_i\boldsymbol{W_{i,:}\boldsymbol{h}}+\sum_j[W_{i,j}^2h_j^2 + 2 \sum_{k \neq j}W_{i,j}h_j W_{i,k}h_k])]$$

其实容易理解，$\boldsymbol{W}_{i,:}$和$\boldsymbol{h}$都是向量，最后一项为向量内积的平方。考虑两向量$\boldsymbol x = (x_1, x_2, x_3)$和$\boldsymbol y = (y_1, y_2, y_3)$，两向量内积为$\boldsymbol{xy} = x_1 y_1 + x_2 y_2 + x_3 y_3$，内积的平方一定为$(x_1 y_1)^2 + (x_2 y_2)^2 + (x_3 y_3)^2 + 2x_1 y_1 x_2 y_2 + 2x_2 y_2 x_3 y_3 + 2 x_1 y_1 x_3 y_3$


式19.34到式19.36的变换如下所示：

$$
\begin{align}
&\mathbb{E}_{\boldsymbol{h} \sim q}[\sum_{i=1}^n \log \sqrt{\frac{\beta_i}{2\pi}}\exp(-\frac{\beta_i}{2}(v_i - \boldsymbol{W}_{i,:}\boldsymbol{h})^2)] \\
= \frac{1}{2} &\mathbb{E}_{\boldsymbol{h} \sim q}[\sum_{i=1}^n \log \frac{\beta_i}{2\pi}-\beta_i(v_i^2 - 2v_i\sum_j W_{i,j} h_j + \sum_j[W_{i,j}^2h_j^2 + 2 \sum_{k \neq j} W_{i,j} h_j W_{i,k} h_k])]
\end{align}
$$

因为$h_i$是贝努利的，所以$\mathbb{E}_{h_i}[h_i] = \hat{h}_i$且$\mathbb{E}_{h_i}[h_i^2] = \mathbb{D}_{h_i}[h_i] + (\mathbb{E}_{h_i}[h_i])^2 = \hat{h}_i(1-\hat{h_i}) + \hat{h}_i^2 = \hat{h}_i$，由于做过均值场近似，因此上式又写作：

$$
\begin{align}
&\frac{1}{2} [\sum_{i=1}^n \log \frac{\beta_i}{2\pi}-\beta_i(v_i^2 - 2v_i\sum_j W_{i,j} \mathbb{E}_{h_j}[h_j] + \sum_j[W_{i,j}^2 \mathbb{E}_{h_j} [h_j^2] + 2 \sum_{k \neq j} W_{i,j} \mathbb{E}_{h_j} [h_j] W_{i,k} \mathbb{E}_{h_k} [h_k]])]\\
= &\frac{1}{2} [\sum_{i=1}^n \log \frac{\beta_i}{2\pi}-\beta_i(v_i^2 - 2v_i\sum_j W_{i,j} \hat{h}_j + \sum_j[W_{i,j}^2 \hat{h}_j + 2 \sum_{k \neq j} W_{i,j} \hat{h}_j W_{i,k} \hat{h}_k])]\\
= &\frac{1}{2} \sum_{i=1}^n [\log \frac{\beta_i}{2\pi}-\beta_i(v_i^2 - 2v_i\boldsymbol{W}_{i,:}\hat{\boldsymbol h} + \sum_j[W_{i,j}^2 \hat{h}_j + 2 \sum_{k \neq j} W_{i,j} \hat{h}_j W_{i,k} \hat{h}_k])]
\end{align}
$$

因此式19.36以及下面的一系列计算中，最后一项都缺了一个系数2。（但式19.43及以后却又是正确的结果）。

####19.4.2 变分法

*P.552*

拉格朗日泛函（式19.50）从$\lambda_1$到$\lambda_3$分别添加的限制条件为：$p(x)$是归一化的概率分布，均值为$\mu$，方差固定为$\sigma^2$。

$$
\begin{align}
\frac{\delta}{\delta p(x)} \mathcal{L} &= \frac{\delta}{\delta p(x)} [\int g_1(p(x), x)dx + C] = \frac{\delta}{\delta p(x)} g_1(p(x), x)\\
 &= \lambda_1 + \lambda_2 x + \lambda_3 (x - \mu)^2 - \log p(x) - 1
\end{align}
$$

式19.53中$\lambda$的取值只需要将该式因子化：

$$
\begin{align}
p(x) &= \exp(\lambda_1 + \lambda_2 x + \lambda_3 (x - \mu)^2 - 1) \\
&= \exp[\lambda_1 + \lambda_2 \mu - \frac{\lambda_2^2}{3\lambda_3} - 1]\exp[-\frac{1}{2(-2\lambda_3)^{-1}}[x - ( \mu - \frac{\lambda_2}{2\lambda_3})]^2]
\end{align}
$$

所以$-2\lambda_3 = (\sigma^2)^{-1}$则有$\lambda_3 = 1/(2\sigma^2)$；$(\mu - \lambda_2/(2\lambda_3)) = \mu$则有$\lambda_2 = 0$；因此$\exp[\lambda_1 + \lambda_2 \mu - \lambda_2^2/(3\lambda_3) - 1] = \exp[\lambda_1 - 1] = 1/(\sqrt{2\pi}\sigma)$所以$\lambda_1 = 1 - \log \sigma \sqrt{2\pi}$。

*P.554*

只要在$p(x)$的核中存在如$\exp[-0.5(Ax^2 + Bx)]$的形式，则$p(x)$必满足某一高斯分布，多元的情况下把括号内内容改成二次型和内积形式。具体推导过程参考[这里](https://github.com/CubicZebra/MVNBayesian/blob/master/Introduction/Introduction%20to%20MVNBayesian.pdf)。

##第二十章 深度生成模型

###20.2 受限玻尔兹曼机

*P.560*

因为是受限玻尔兹曼机，$\boldsymbol h$与$\boldsymbol h$之间，$\boldsymbol v$与$\boldsymbol v$之间都不存在任何连接，式20.3中的$-\boldsymbol{v^{\intercal}R}v$与$-\boldsymbol{h^{\intercal}S}h$都为0，变成式20.5的形式。

####20.2.1 条件分布

*P.562*

式20.10有问题（原书中也有误），正确形式应为：

$$\frac{1}{Z^\prime}\exp\{\sum_{j=1}^{n_h}c_j h_j + \sum_{j=1}^{n_h}\boldsymbol{v}^\intercal\boldsymbol{W}_{:,j}h_j\}$$

从式20.11完全导出式20.15的过程推导如下（必须用原书中的形式）：

$$
\begin{align}
P(\boldsymbol h | \boldsymbol v) &= \frac{P(\boldsymbol h, \boldsymbol v)}{\sum_{\boldsymbol h} P(\boldsymbol h, \boldsymbol v)} = \frac{\exp[\boldsymbol c^\intercal \boldsymbol h + \boldsymbol v^\intercal \boldsymbol{Wh}]}{\sum_{\boldsymbol h \in \{0,1\}^j}\exp[\boldsymbol c^\intercal \boldsymbol h + \boldsymbol v^\intercal \boldsymbol{Wh}]}\\
&= \frac{\prod_{j}\exp[c_j h_j + \boldsymbol v^\intercal \boldsymbol W_{:,j} h_j]}{\prod_{j} \sum_{h_j \in \{0,1\}} \exp[c_j h_j + \boldsymbol v^\intercal \boldsymbol W_{:,j} h_j]}\\
&= \prod_{j} \frac{\exp[(c_j + \boldsymbol v^\intercal \boldsymbol W_{:,j}) h_j]}{\exp[0] + \exp[c_j + \boldsymbol v^\intercal \boldsymbol W_{:,j}]}\\
&= \prod_{j} \frac{\exp[T_j h_j]}{\exp[T_j(1 - h_j)]\exp[T_j(h_j - 1)](1 + \exp[T_j])}\\
&= \prod_{j} \frac{\exp[T_j (2h_j - 1)]\exp[T_j (1 - h_j)]}{\exp[T_j(1 - h_j)]\{\exp[T_j(h_j - 1)] + \exp[T_j h_j]\}}\\
&= \prod_{j} \frac{\exp[T_j (2h_j - 1)]}{\exp[T_j(h_j - 1)] + \exp[T_j h_j]}\\
\end{align}
$$

因为$h_j$是二值的，当$h_j$分别为$0$或$1$时，上式分别为$\exp[-T_j]/(\exp[-T_j] + 1)$和$T_j/(1 + \exp[T_j])$，这两者仍为$\sigma$函数，所以可写作$(2\boldsymbol h - 1)$与$(\boldsymbol c + \boldsymbol W^\intercal \boldsymbol v)$的Hadamard乘积的$\sigma$函数其实是等效的。

$\boldsymbol v \in \{0,1\}^i$也是二值的，式20.16也同理。

###20.4 深度玻尔兹曼机

*P.567*

深度信念网络（DBN）和深度玻尔兹曼机（DBM）的一大不同：前者的计算是逐层的，我们可以利用简单的正向计算与反向求梯度进行两个方向上的传播与更新，但这个计算过程在当前层未完成时，下一层的计算是无法开启的；而后者由于条件独立的情况，计算都是并行展开的，无论设置的层数如何，两步内都可以实现整个网络完全的更新。

*P.572*

计算玻尔兹曼机期望值的困难在于，各单元的随机变量之间存在各种各样复杂的关联，易辛模型（ising model）给出的物理解释是，在微观粒子自旋时会存在相互作用，于是就变成了复杂的多体问题。用来提示这些相互作用的，正是哈密顿能量函数（Hamiltonian）中不同单元之间乘以其混合权重的项：

$$-\sum_{i,j} w_{i,j} x_i x_j \tag{1}$$

究其原因，是考虑当所有权重都变为$0$时，玻尔兹曼分布将会变成：

$$P(\boldsymbol x | \boldsymbol{\theta}) = \frac{1}{Z(\boldsymbol{\theta})}e^{\sum_i b_i x_i} = \prod_i \frac{e^{b_i x_i}}{1 + e^{b_i}} \tag{2}$$

这时就变为了独立的单变量分布之间乘积这样简单的问题了。也就是说这时的玻尔兹曼机的计算会退化成简单的一体问题。暂先不管这个使期望值计算极度简化的玻尔兹曼机，我们先准备一个各个随机变量相互独立的概率分布模型（试验分布，test distribution）集合。然后我们从这样的集合当中探索出最接近玻尔兹曼机的一个概率分布出来。当然$w_{i,j}=0$的分布由于过于简单必然不是问题的最优解。那么我们怎样去找出这样一种概率分布呢？

首先，由于试验分布所有变量相互独立，我们可以写出其分布的一般情形：

$$Q(\boldsymbol x) = \prod_i Q_i(x_i) \tag{3}$$

这样一来每一个$Q_i(x_i)$都对应着一个单变量的概率分布，只是$Q_i(x_i)$的具体形式我们现在还无从知晓罢了。我们从这个分布函数$Q(\boldsymbol x)$来针对接近玻尔兹曼机的条件来进行优化。首先考虑$Q(\boldsymbol x)$是我们的试验分布，$P(\boldsymbol x | \boldsymbol{\theta})$是目标玻尔兹曼分布，考虑KL散度是两分布间的距离，要让它们更为相似，则使KL散度最小即可。试验分布$Q$对玻尔兹曼机$P$的KL散度为：

$$D_{KL}(Q||P) = \sum_{\boldsymbol x} Q(\boldsymbol x) \log \frac{Q(\boldsymbol x)}{P(\boldsymbol x | \boldsymbol{\theta})} \tag{4}$$

但需要注意的是，这并不是一个无约束的目标函数，因为对于任意的$i$都需要满足以下的等式约束：

$$\sum_{x_i = 0,1} Q_i(x_i) = 1 \tag{5}$$

由于$Q_i(x_i)$不是随机变量而是某个函数，因而我们可以写出式(4)满足式(5)约束条件的拉格朗日泛函：

$$\mathcal{L}[Q_i; \Lambda_i] = D_{KL}(Q||P) + \sum_i \Lambda_i (\sum_{x_i = 0,1} Q_i(x_i) - 1) \tag{6}$$

这里$\Lambda_i$是$\mathcal L$的拉格朗日系数。用$\mathcal L$对$\Lambda_i$求变分，其结果一定满足以下约束条件：

$$0 = \frac{\partial \mathcal L}{\partial \Lambda_i} = \sum_{x_i = 0,1} Q_i (x_i) - 1 \tag{7}$$

然后再用$\mathcal L$对$Q_i(x_i)$求变分。因为：

$$
\begin{align}
\mathcal L &= \sum_{\boldsymbol x} Q(\boldsymbol x)\log\frac{Q(\boldsymbol x)}{P(\boldsymbol x | \boldsymbol{\theta})} + \sum_i \Lambda_i (\sum_{x_i = 0,1} Q_i(x_i) - 1)\\
&= \sum_{\boldsymbol x} \prod_i Q_i(x_i)[\sum_i\log Q_i(x_i) - \log P(\boldsymbol x | \boldsymbol{\theta})]\\
&=  \sum_{x_i} Q_i(x_i)\{\prod_{j(\neq i)} Q_j(x_j)[\sum_i\log Q_i(x_i) - \log P(\boldsymbol x | \boldsymbol{\theta})]\}\\
&\quad+ \sum_{\boldsymbol{x}_{-i}} \prod_{j(\neq i)} Q_j(x_j)[Q_i (x_i)][\log Q_i (x_i) + \sum_{j(\neq i)}\log Q_j(x_j) - \log P(\boldsymbol x | \boldsymbol{\theta})]
\tag{8}
\end{align}
$$

针对前一项

$$
\sum_{x_i} Q_i(x_i)\{\prod_{j(\neq i)} Q_j(x_j)[\sum_i\log Q_i(x_i) - \log P(\boldsymbol x | \boldsymbol{\theta})]\} \\
= \sum_{x_i} Q_i(x_i) T_i = E_{x_i \sim Q_i} [T_i] \tag{9}$$

可以直接将其视为不变量，因此有：

$$
\begin{align}
0 = \frac{\delta \mathcal L}{\delta Q_i(x_i)}
&= \sum_{\boldsymbol{x}_{-i}} \prod_{j(\neq i)} Q_j(x_j)[\log Q_i (x_i) + \sum_{j(\neq i)}\log Q_j(x_j) \\
&\quad- \log P(\boldsymbol x | \boldsymbol{\theta}) + Q_i(x_i)\frac{1}{Q_i(x_i)}] + \Lambda_i\\
&= \sum_{\boldsymbol{x}_{-i}} \prod_{j(\neq i)} Q_j(x_j)[\sum_{k}\log Q_k(x_k) - \log P(\boldsymbol x | \boldsymbol{\theta}) + 1] + \Lambda_i
\tag{10}
\end{align}
$$

对于式（10）中括号最左边的一项而言，将其中的第$i$项乘出来以后，联立式（7）的结论有有：

$$\sum_{\boldsymbol{x}_{-i}} \prod_{j(\neq i)} Q_j(x_j) \log Q_i(x_i) = \log Q_i(x_i) \prod_{j(\neq i)} (\sum_{x_j} Q_j (x_j)) = \log Q_i(x_i) \tag{11}$$

对于那些$k \neq i$的部分，由于与$i$不直接关联，将这部分值视为常数即可。然后对式（10）中中括号内的第二项，乘出来并代入玻尔兹曼机的一般形，有：

$$
\sum_{\boldsymbol{x}_{-i}} \prod_{j(\neq i)} Q_j(x_j) P(\boldsymbol x | \boldsymbol{\theta}) = \sum_{\boldsymbol{x}_{-i}} \prod_{j(\neq i)} Q_j(x_j)[\sum_k b_k x_k + \sum_{j,k} w_{j,k} x_j x_k - \log Z]\\
= \sum_{\boldsymbol{x}_{-i}} \prod_{j(\neq i)} Q_j(x_j)[b_i x_i + \sum_{k (\neq i)} b_k x_k + \sum_{j(\neq i)} w_{j,i} x_j x_i + \sum_{j(\neq k)} \sum_{k(\neq i)} w_{j,k} x_j x_k - \log Z]\\
\tag{12}
$$

对于所有$j$，将试验分布中任意的$x_j$的期望引入（也就是所谓的平均场，mean field），也就是：

$$\mu_j = \sum_{\boldsymbol x} x_j Q(\boldsymbol x) = \sum_{x_j = 0,1} x_j Q_j(x_j) \tag{13}$$

可以将式（12）中中括号里逐项分别乘出来并化简，第一项与最后一项的操作过程如式（11）所示不再赘述，其余化简过程分别如下:

$$
\sum_{\boldsymbol{x}_{-i}} \prod_{j(\neq i)} Q_j(x_j) \sum_{k(\neq i)} b_k x_k = \sum_{x_j} \sum_{\boldsymbol{x}_{-i,j}} \prod_{m(\neq i,j)} Q_m(x_m) Q_j(x_j) \sum_{j(\neq i)} b_j x_j\\
= \sum_{j(\neq i)} b_j \sum_{x_j} Q_j(x_j) x_j \cdot [\prod_{m(\neq i,j)} (\sum_{x_m} Q_m(x_m))] = \sum_{j(\neq i)} b_j \mu_j
$$

第三项同理：

$$
\sum_{\boldsymbol{x}_{-i}} \prod_{j(\neq i)} Q_j(x_j) \sum_{j(\neq i)} w_{j,i} x_j x_i = x_i \sum_{x_j} \sum_{\boldsymbol{x}_{-i,j}} \prod_{m(\neq i,j)} Q_m(x_m) w_{i,j} \sum_{j(\neq i)} Q_j(x_j) x_j\\
= x_i \sum_{j(\neq i)} w_{i,j} \sum_{x_j} Q_j(x_j) x_j \cdot [\prod_{m(\neq i,j)} (\sum_{x_m} Q_m(x_m))] = x_i \sum_{j(\neq i)} w_{i,j} \mu_j
$$

第四项略复杂：

$$
\sum_{\boldsymbol{x}_{-i}} \prod_{j(\neq i)} Q_j(x_j) \sum_{j(\neq k)} \sum_{k(\neq i)} w_{j,k} x_j x_k\\
= \sum_{x_j} \sum_{x_k} \sum_{\boldsymbol{x}_{-i,j,k}} \prod_{m(\neq i,j,k)} Q_m(x_m) \sum_{j(\neq k)} \sum_{k(\neq i)} w_{j,k} x_j x_k Q_j(x_j) Q_k(x_k)\\
= \sum_{j(\neq k)} \sum_{k(\neq i)} w_{j,k} \sum_{x_j} Q_j(x_j)x_j \sum_{x_k} Q_k(x_k)x_k[\prod_{m(\neq i,j,k)} (\sum_{x_m} Q_m(x_m))]\\
= \sum_{j,k(\neq i)} w_{j,k} \mu_j \mu_k
$$

值得注意的是，第二项，第四项和第五项由于不存在与$i$相关的变量，因此在计算过程中可以并入一个常数项，所以式（10）又可以写成：

$$
0 = \log Q_i(x_i) - (b_i + \sum_{j \in \mathcal{N}_i} w_{i,j} \mu_j)x_i + C + \Lambda_i \tag{14}
$$

其中C是一个与$i$无关的常数，$\mathcal N_i$是与$x_i$相邻接的所有节点的集合（只有满足这种情况时$w_{i,j}$不为$0$）。解式（14）可知，$Q_i(x_i)$一定具有如下形式的核函数：

$$Q_i (x_i) \propto e^{-\Lambda_i}e^{(b_i + \sum_{j \in \mathcal{N}_i} w_{i,j} \mu_j)x_i} \tag{15}$$

另外，由于对拉格朗日系数$\Lambda_i$变分的限制条件存在，所以：

$$\sum_{x_i} Q_i(x_i) = 1 \tag{16}$$

由于$x_i$是二值的，不妨设$Q_i (x_i) = Ce^{-\Lambda_i}e^{(b_i + \sum_{j \in \mathcal{N}_i} w_{i,j} \mu_j)x_i} = Ce^{-\Lambda_i}e^{T_i x_i}$，联立式（15）和式（16）后显然有：

$$
\begin{align}
1 &= Ce^{-\Lambda_i} + Ce^{-\Lambda_i}e^{T_i}\\
e^{-\Lambda_i} &= C^{-1}(1 + e^{T_i})^{-1}\\
\therefore Q_i(x_i) &= C\cdot[C^{-1}(1 + e^{T_i})^{-1}]e^{T_i x_i}\\
&= \frac{e^{T_i x_i}}{1 + e^{T_i}} = \frac{e^{(b_i + \sum_{j \in \mathcal{N}_i} w_{i,j} \mu_j)x_i}}{1 + e^{b_i + \sum_{j \in \mathcal{N}_i} w_{i,j} \mu_j}}\\
\tag{17}
\end{align}
$$

对比$x_i$的满条件分布$\phi$：

$$
\begin{align}
P(x_i | \boldsymbol x_{-i}, \boldsymbol{\theta}) &= \frac{P(\boldsymbol x, \boldsymbol{\theta})}{\sum_{x_i = 0, 1} P(\boldsymbol x, \boldsymbol{\theta})} = \frac{e^{- \boldsymbol \phi (\boldsymbol x, \boldsymbol \theta)}}{\sum_{x_i = 0,1} e^{- \boldsymbol \phi (\boldsymbol x, \boldsymbol \theta)}}\\
&= \frac{e^{(b_i + \sum_{j \in \mathcal{N}_i} w_{i,j} x_j)x_i}}{1 + e^{b_i + \sum_{j \in \mathcal{N}_i} w_{i,j} x_j}}
\tag{18}
\end{align}
$$

比较式（18）与式（17）不难发现，我们所说的“平均场近似”名字的由来，正是基于“在讨论变量$i$时，将其它无关的随机变量$x_j$全部换成其平均值（期望值）$\mu_j$”这样的想法。在统计物理里，为了使近似的意思更加明确，会引入更严密的推导过程，这里不作展开。现在虽然已经写出了$Q_i(x_i)$的具体表达式，但如果涉及具体计算，就需要知道所有与$x_i$临接的$x_j$的期望值，如果想要知道它们的期望值，就需要去计算每一个$x_j$的概率分布，这样一来计算会以一个节点展来没完没了，为了使全体定义不相矛盾，可以作出假设对任意的$i$都有$Q_i(x_i = 1) = \mu_i$，因此有：

$$\mu_i = \frac{e^{b_i + \sum_{j \in \mathcal{N}_i} w_{i,j} \mu_j}}{1 + e^{b_i + \sum_{j \in \mathcal{N}_i} w_{i,j} \mu_j}} = \sigma(b_i + \sum_{j \in \mathcal{N}_i} w_{i,j} \mu_j) \tag{19}$$

这正是自洽场方程式。物理上易辛模型使用的自洽场方程是双曲正切函数$\tanh(\cdot)$，这里使用的$\sigma$函数是由于对$x_i$作了二值假设$x_i \in \{0,1\}$，实质上两者并无本质区别。

因此自洽场方程，就是求出平均场值$\mu_i$的决定性方程，解开了它那么原概率分布的具体参数也就能够确定下来了。但由于自洽场方程组是联立的非线性方程，很难求出其解析解的形式，所以通常采用数值方法求算。一般地，我们给定$\boldsymbol \mu (0)$一组随机初始值：

$$\mu_1(0), \mu_2(0), ..., \mu_M(0) \tag{20}$$

然后利用对于每一步$t=1,2,3,...$，依次迭代计算：

$$\mu_i(t) = \sigma(b_i + \sum_{j \in \mathcal{N}_i} w_{i,j} \mu_j(t-1)) \tag{21}$$

收敛后的结果一般都可以作为平均场方程组的数值解。

不难看出，算法20.1在吉布斯采样开始前计算的$\Delta_{\boldsymbol{W}^{(1)}}$与$\Delta_{\boldsymbol{W}^{(2)}}$都是在计算当前式（20.35）对权重矩阵的梯度。经一组吉布斯采样后，由于$\boldsymbol{V}$，$\boldsymbol{H^{(1)}}$，$\boldsymbol{H^{(2)}}$都发生了变化，于是式（20.35）对权重产生了新的梯度，两梯度相减得出梯度的变化，乘以学习率之后加在原权重上使权重向“使深度玻尔兹曼机整体能量更小的方向”产生变化。

####20.4.4 逐层预训练

*P.571*

预训练是为了获得更好的初始值。具体的操作是将DBM看成不同的RBM串联的形式，自上而下或者自下而上地训练。如果写出每一层的自洽方程式可知，如果训练保持单向进行，则从上到下，或者从下到上的第二层的自洽方程里，$\sigma$函数涉及来源于自下而上或者自上而下第三层单元值的内容，从而导致训练无法继续。预训练的想法是，把这一项换成顶部或者底部的副本（相当底层与第三层对第二层的影响，近似于两倍来自底层的影响），这样单向的训练就可以进行。

只不过要注意的是，经过这样处理后除顶层和底层的数据外，其它层相当于经过了两倍的增益，因此连接这些层的权重得到的最终结果，都将会是我们目标值的两倍。

###20.5 实值数据上的玻尔兹曼机

*P.578*

它不能被全部包括在内，也就是仅包含从$\boldsymbol v$到$\boldsymbol{h}^{(1)}$的权重部分，这是由受限玻尔兹曼机的结构所决定的。另外，“如果包含这些项，我们将得到一个线性模型，而不是受限玻尔兹曼机”是因为假设每一层的权重分别记为$\boldsymbol{W}^{(1)}$，$\boldsymbol{W}^{(2)}$，$\boldsymbol{W}^{(3)}$...，包含所有权重的模型经向前传播会产生这样的效果：$\boldsymbol{h}^{(n)} = \boldsymbol{W}^{(n)} \cdots \boldsymbol{W}^{(2)} \boldsymbol{W}^{(1)} \boldsymbol v = \boldsymbol{W}^\prime \boldsymbol v$，相当于对原始输入数据作了仅一次线性变换。

对于$\frac{1}{2}\sum_j \beta_j W_{j,i}^2 h_j h_i$，如果$h_j \neq h_i$则其中必有一个为0，则$i$与$j$不同的项一定为0，另外$h_i^2 = h_i$，所以可以把式(20.40)化成关于式(20.41)中不同$i$成份相加的形式。

####20.5.2 条件协方差的无向模型

*P.579*

由式(20.43)~式(20.46)向式(20.47)的推导：

$$
\begin{align}
p_{mc}(\boldsymbol x, \boldsymbol h^{(m)}, \boldsymbol h^{(c)}) &= \frac{1}{Z}\exp\{-E_{mc}(\boldsymbol x, \boldsymbol h^{(m)}, \boldsymbol h^{(c)})\}\\
&= \frac{1}{Z}\exp\{-\frac{1}{2}\boldsymbol x^\intercal \boldsymbol x + \sum_j \boldsymbol x^\intercal \boldsymbol W_{:,j} \boldsymbol h_j^{(m)} - \frac{1}{2}\sum_j h_j^{(c)} (\boldsymbol x^\intercal \boldsymbol r^{(j)})^2 + C\}\\
&\propto \exp\{-\frac{1}{2}[\boldsymbol x^\intercal (\boldsymbol I + \sum_j h_j^{(c)} r^{(j)} r^{(j)\intercal}) \boldsymbol x - 2 \boldsymbol x^\intercal \sum_j \boldsymbol W_{:,j} h_j^{(m)}]\}
\end{align}
$$

参考[这里](https://github.com/CubicZebra/MVNBayesian/blob/master/Introduction/Introduction%20to%20MVNBayesian.pdf)的推导过程，写上如上形式之后，多元高斯分布的均值与方差协方差矩阵也就一目了然了。
