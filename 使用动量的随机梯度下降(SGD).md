#使用动量的随机梯度下降(SGD)

相比于传统SGD，使用动量的SGD多了个动量参数，将计算更新速度按递推关系改写为：

$$\boldsymbol{v_{k+1}} = \alpha \boldsymbol{v_k} - \epsilon \boldsymbol{g_k} = \alpha^{k} \boldsymbol{v_1} - \epsilon\sum_{i=0}^{k-1} \alpha^i \boldsymbol{g_{k-i}}$$

可以看出更新的步长是所有梯度按动量参数的幂相乘后的线性加和形式，因此$\alpha$一定是一个介于0到1之间的数，否则随步长增加更新会发散。

另外用动量进行更新，不妨把$\boldsymbol{g_{k-i}}$写成两个分量$\boldsymbol{m_{i-1}}$与$\boldsymbol{n_{i-1}}$的形式，前者是重直于优化的目标方向，后者平行于优化的目标方向（参考图4.6），则上式又可写为：

$$\boldsymbol{v_{k+1}} = \alpha^{k} \boldsymbol{v_1} - \epsilon\sum_{i=0}^{k-1} \alpha^i \boldsymbol{m_{k-i}} - \epsilon\sum_{i=0}^{k-1} \alpha^i \boldsymbol{n_{k-i}}$$

对于前项，在病态条件下总有$sign(\boldsymbol{m_i}) = -sign(\boldsymbol{m_{i-1}})$，因而在k足够大时，梯度在重直优化方向上的震荡会趋向于0，而后一项的分量基本同向，把$\boldsymbol{n_i}$视为随机变量：在k足够大时直观上看就是平行于优化目标的平均梯度分量$\overline{\boldsymbol{n}}_i$以$\alpha$的幂级数和的情形得以保留，即

$$- \epsilon\sum_{i=0}^{k-1} \alpha^i \boldsymbol{n_{k-i}} \approx - \epsilon \overline{\boldsymbol{n}}_i \sum_{i=0}^{k-1} \alpha^i$$

特别地，当$\alpha = 0.5$时，因为$\sum_{i=1}^{\infty}(1/2)^i = 1$，梯度更新在平行于优化目标的方向也仅以平均梯度分量$\overline{\boldsymbol{n}}_i$进行，通常不会导致结果发散。而一般情况下，因为在$0<\alpha < 1$时，级数$\sum_{i=1}^{\infty}\alpha^i$收敛且和为$\frac{1}{1-\alpha}$，因此其梯度更新将保持在步长为$\frac{\overline{\boldsymbol{n}}_i}{1-\alpha}$的情况下进行，这正对应于书中式(8.17)给出的结论。