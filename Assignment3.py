#Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import scipy.stats as stats
import pylab
from random import uniform
from matplotlib.pyplot import figure

#Import the data set
df = pd.read_csv("HY_Universe_corporate bond.csv")

#Get an idea of the size and shape of the data set.
print(df.shape)

#Get number of numeric attributes in df
num_cols = df._get_numeric_data().columns

#Drop row with more than 20% of data missing
df.replace(["Nan","WR","NR","WD"],np.nan,inplace=True)
df = df.dropna(thresh=30)
# New number of row after clean up
print(df.shape[0])
df.head()

#Find col index of categorical data
cate_col_index = np.zeros(shape=(1,df.shape[1]-len(num_cols)))
count = 0
for i in range(0,df.shape[1]):
    if isinstance(df.iat[23,i],str):
        cate_col_index[0,count] = i
        count += 1
        
print(cate_col_index)

# Count of the unique categories in each categorical attribute.
# Summary statistic for the numerical attributes.
df.describe(include='all')

#Visualize outliers using Quantile-Quantile plot for coupon
coup = (df.loc[:,['Coupon']])
coup = np.array(coup)

# This is only to remove the large outlier of Coupon.(900)
coup_temp = np.zeros(shape=(1,2178))
x = 0
for i in range(0,len(coup)):
    if coup[i] > 800:
        x = i
    else:
        coup_temp[0,i] =  coup[i]

# Plot with the very large outlier
stats.probplot(np.sort(coup,axis=None),dist="norm",plot=pylab)
plot.show()
# Plot without the very large outlier
stats.probplot(np.sort(coup_temp,axis=None),dist="norm",plot=pylab)

num_col_index = np.array([9,10,13,15,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36])
plot.figure(figsize=(10,10))
for i in range(0,2178):
    if df.iat[i,19] == "Yes":
        pcolor = "red"
    else:
        pcolor = "blue"
    
    dataRow = df.iloc[i,num_col_index]
    dataRow.plot(color=pcolor)
plot.xlabel("Attribute index")
plot.ylabel("Attribute values")
plot.show()

# Scatter plot of LIQ Score V.S. number of trades
# We can see a certain correlation between the two attributes
attribute_1 = df.iloc[:,20]
attribute_2 = df.iloc[:,21]
plot.scatter(attribute_1,attribute_2)
plot.xlabel("LIQ Score")
plot.ylabel("number of trades")
plot.show()

# Scatter plot of Coupon V.S. Maturity at issue months
# Use of Coupon attribute with very large outlier removed.
# We cannot see a certain correlation between the two attributes
attribute_1 = coup_temp
attribute_2 = df.iloc[:,13]
plot.scatter(attribute_1,attribute_2)
plot.xlabel("Coupon")
plot.ylabel("Maturity at issue months")
plot.show()

target = []
for i in range(0,2178):
    if df.iat[i,19] == 'Yes':
        target.append(1.0 + uniform(-0.1,0.1))
    else:
        target.append(0.0 + uniform(-0.1,0.1))
data_row = df.iloc[:,20]
#data_row = coup_temp
plot.scatter(data_row,target,alpha=0.1,s=120)
plot.xlabel('Attribute Value(LIQ Score)')
plot.ylabel('Target Value(In ETF)')

# We check the correlation between the attributes that we used for the scatter plots.
# The correlation scores confirm our assumptions from the scatter plot.
print("Correlation between Liquidity score and number of trades: " + str(df['LIQ SCORE'].corr(df['n_trades'])))
print("Correlation between Coupon score and Maturity at issue month: " + str(df['Coupon'].corr(df['Maturity At Issue months'])))

# Correlation using a heat map
corMat = pd.DataFrame(df.corr())
plot.pcolor(corMat)
plot.show()

print("My name is Timothee Becker")
print("My NetID is: tbecker5")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
