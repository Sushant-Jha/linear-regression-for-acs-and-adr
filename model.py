import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression  

df = pd.read_csv("TR.csv")
df.head()

y=df.ACS
x=df.ADR.values.reshape(-1,1)

print(x.shape,y.shape)



model = LinearRegression().fit(x,y)



r_sq = model.score(x,y)
print(r_sq)   #check_score
intercept = model.intercept_
slope = model.coef_



y_pred = intercept + slope*x  #LinearRegression equation



fig,ax = plt.subplots(figsize=(18,18))
plt.scatter(x,y)

plt.plot(x,y_pred,c='red',linestyle='--',dashes=(5,5)) #line


fig.set_facecolor('#f3edd3')
ax.patch.set_facecolor('#f3edd3')
ax.grid(ls='dotted',lw=.8,color='lightgrey',axis='y',zorder=1)
#annotate teams
ax.annotate(xy=(175.4,281.3),text='vakk',fontsize=20)
ax.annotate(xy=(70,121),text='Zehradieux ',fontsize=20)
#analysis 
ax.annotate(xy=(80,270),text=f'R-Squared = {round(r_sq,2)}\nThe regression equation: y = {intercept} + {slope} * x ',fontname='Andale Mono',fontsize=20)

plt.xlabel('Average Damange per round',fontsize=18,fontname='Druk')
plt.ylabel('Average Combact Score',fontsize=18,fontname='Druk')
plt.title('Evaluating the Relationship Between ADR  and ACS',fontsize=24,)

plt.savefig('foof.png',dpi=300,bbox_inches = 'tight',facecolor='#f3edd3')
