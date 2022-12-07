import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import datasets, linear_model
import sklearn.metrics as metrics
import numpy

df= pd.read_csv(r"tmdb_movies_data.csv")
colors = list("rgbcmyk")

df['budget'] = pd.to_numeric(df['budget'],errors='coerce')
df['popularity'] = pd.to_numeric(df['popularity'],errors='coerce')
df['revenue'] = pd.to_numeric(df['revenue'],errors='coerce')

df_revenue = df[['revenue']].values.tolist()
df_budget = df['budget'].values.tolist()
df_popularity = df['popularity'].values.tolist()

# clear zero data
BudgetList = []
PopularityList = []
revenueList = []
for ind in range(len(df_budget)):
    if df_budget[ind] != 0:
        BudgetList.append(df_budget[ind])
        PopularityList.append(df_popularity[ind])
        revenueList.append(df_revenue[ind][0])
    
# STATISTICAL INFO #

print("---- popularity ----")
#Average
def Average(lt):
    return sum(lt) / len(lt)
Average_popularity = Average(PopularityList)
Average_popularity = round(Average_popularity, 2)
#Median
PopularityList.sort()
midPoint = len(PopularityList) // 2
Median_popularity = (PopularityList[midPoint] + PopularityList[~midPoint]) / 2
Median_popularity = round(Median_popularity, 2)
#Range
minR = min(PopularityList)
maxR = max(PopularityList)
Range_popularity = maxR-minR
Range_popularity = round(Range_popularity, 2)
#Standard Deviation
variance = sum([((x - Average_popularity) ** 2) for x in PopularityList]) / len(PopularityList)
Standard_Deviation_popularity = variance ** 0.5
Standard_Deviation_popularity = round(Standard_Deviation_popularity, 2)
#Print Results
print("Mean: " + str(Average_popularity))
print("Median: " + str(Median_popularity))
print("Range: " + str(Range_popularity))
print("Standard Deviation: " + str(Standard_Deviation_popularity))

print("---- budget ----")
#Average
Average_budget = Average(BudgetList)
Average_budget = round(Average_budget, 2)
#Median
BudgetList.sort()
midPoint = len(BudgetList) // 2
Median_budget = (BudgetList[midPoint] + BudgetList[~midPoint]) / 2
Median_budget = round(Median_budget, 2)
#Range
minR = min(BudgetList)
maxR = max(BudgetList)
Range_budget = maxR-minR
Range_budget = round(Range_budget, 2)
#Standard Deviation
variance = sum([((x - Average_budget) ** 2) for x in BudgetList]) / len(BudgetList)
Standard_Deviation_budget = variance ** 0.5
Standard_Deviation_budget = round(Standard_Deviation_budget, 2)
#Print Results
print("Mean: " + str(Average_budget))
print("Median: " + str(Median_budget))
print("Range: " + str(Range_budget))
print("Standard Deviation: " + str(Standard_Deviation_budget))

print("---- revenue ----")
#Average
Average_revenue = Average(revenueList)
Average_revenue = round(Average_revenue, 2)
#Median
revenueList.sort()
midPoint = len(revenueList) // 2
Median_revenue = (revenueList[midPoint] + revenueList[~midPoint]) / 2
Median_revenue = round(Median_revenue, 2)
#Range
minR = min(revenueList)
maxR = max(revenueList)
Range_revenue = maxR-minR
Range_revenue = round(Range_revenue, 2)
#Standard Deviation
variance = sum([((x - Average_popularity) ** 2) for x in revenueList]) / len(revenueList)
Standard_Deviation_revenue = variance ** 0.5
Standard_Deviation_revenue = round(Standard_Deviation_revenue, 2)
#Print Results
print("Mean: " + str(Average_revenue))
print("Median: " + str(Median_revenue))
print("Range: " + str(Range_revenue))
print("Standard Deviation: " + str(Standard_Deviation_revenue))

x = BudgetList
y = PopularityList
plt.scatter(x,y,color=colors.pop())
plt.legend(BudgetList)
plt.title("Movie Budget vs Movie popularity")
plt.ylabel("Popularity")
plt.xlabel("Budget ($USD)")
plt.show()

### DAY 2 ###
filename = "cleanedData.csv"
f = open(filename, "w+")
f.close()

dataToCSV = {'budget': BudgetList,'revenue': revenueList}
df2 = pd.DataFrame.from_dict(dataToCSV)
df2.to_csv('cleanedData.csv')
# ///////////////////////////////////// #
df2 = pd.read_csv("cleanedData.csv")

df2['budget'] = pd.to_numeric(df2['budget'],errors='coerce')
df2['revenue'] = pd.to_numeric(df2['revenue'],errors='coerce')

Y = df2['budget']
X = df2['revenue']


# find 1/4 spot
one_fourth_pos = len(PopularityList) / 2
one_fourth_pos = int(one_fourth_pos)

# split --> training/testing
X_test = X[-one_fourth_pos:]
Y_test = Y[-one_fourth_pos:]

# Plot outputs
plt.scatter(X_test, Y_test,  color='black')
plt.title('Test Data')
plt.ylabel('Revenue ($USD)')
plt.xlabel('Budget ($USD)')
# clear units
plt.xticks(())
plt.yticks(())

X=X.values.reshape(len(X),1)
Y=Y.values.reshape(len(Y),1)

X_train = X[:-one_fourth_pos]
Y_train = Y[:-one_fourth_pos]

regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, Y_train)

#convert 1D array to 2D array:
X_test2 = []
for each in X_test:
    arr = []
    arr.append(each)
    X_test2.append(arr)
# Plot outputs
plt.plot(X_test2, regr.predict(X_test2), color='red',linewidth=2)

plt.show()

# how did it do?
mse = metrics.mean_squared_error(Y_test, Y_train)
rmse = np.sqrt(mse) 
r2 = metrics.r2_score(Y_test,Y_train)

print("RMSE -- >  " + str(rmse))
print("R^2 -- >  " + str(r2))