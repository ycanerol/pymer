"""

"""
from sklearn.cross_decomposition import CCA
from omb import OMB


exp, stim_nr = '20180710', 8
n_components = 2



st = OMB(exp, stim_nr)

spikes = st.allspikes()
stimulus = st.bgsteps

cca = CCA(n_components=n_components,
          scale=True, max_iter=500, tol=1e-06, copy=True)

cca.fit(spikes.T, stimulus.T)

x, y = cca.transform(spikes.T, stimulus.T)
x, y = x.T, y.T


#%%
import matplotlib.pyplot as plt

plt.plot(x[:, :200].T)
plt.plot(spikes[:2, :200].T);
plt.show()


plt.plot(y[:, :200].T)
plt.plot(stimulus[:, :200].T);

