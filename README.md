# nssc-tools
Tools for using the Non Stationary Structured Coalescent model

*Remark:* The scripts work fine for computing the IICR. However, for some values, the computation of the cdf and the pdf under the proposed model has numerical instability. A modification solving this will be commited soon.

Here is an example for plotting the IICR corresponding of a scenario specified by using the text format:

```python
import numpy as np
import matplotlib.pyplot as plt
from functions import readScenario
from model import NSSC
d = readScenario("./scenarios/HumansNeandScenario_samplingHumans.txt")
model = NSSC(d)
t = np.arange(0.00001, 50, 0.1)
IICR = [model.evaluateIICR(i) for i in t]
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
line, = ax.plot(t, IICR, color='blue', lw=2)
ax.set_xscale('log')
plt.plot(t, IICR)
plt.xlim((1, 50))
plt.show()
```

