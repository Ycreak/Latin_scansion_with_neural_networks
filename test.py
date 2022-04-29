

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
# importing module
from pandas import *
 
from cltk.prosody.lat.metrical_validator import MetricalValidator
result = MetricalValidator().is_valid_hexameter("-UU---UU---UU-U")
print(result)
exit(0)

# reading CSV file
data = read_csv("test.csv")
 
# converting column data to list
x = data['size'].tolist()

ivv = data['ivv'].tolist()
ov = data['ov'].tolist()
lucr = data['lucr'].tolist()
verg = data['verg'].tolist()


# exit(0)

# x = [0,5,9,10,15]
# y = [0,1,2,3,4]
plt.plot(x,ivv, label='Iuvenal')
plt.plot(x,ov, label='Ovid')
plt.plot(x,lucr, label='Lucretius')
plt.plot(x,verg, label='Virgil')

# set number of bins
plt.locator_params(nbins=8)
plt.ylim(ymin=0)  # this line
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
plt.title('Effect of different training set sizes')
plt.legend(loc='lower right')

plt.show()
