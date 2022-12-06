import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
import collections
#import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import scipy

df_site = pd.read_csv(r"SITE_INFO.csv")
df_waterlevel= pd.read_csv(r"WATERLEVEL.csv",dtype={"Comment": "string", "Original Direction": "string"})

# Water level relative to NAVD88
df_waterlevel['Water level in feet relative to NAVD88'] = pd.to_numeric(df_waterlevel['Water level in feet relative to NAVD88'],errors='coerce')
# Depth to water below land surface in ft
df_waterlevel['Depth to Water Below Land Surface in ft.'] = pd.to_numeric(df_waterlevel['Depth to Water Below Land Surface in ft.'],errors='coerce')

df_waterlevel2 = df_waterlevel[['Water level in feet relative to NAVD88']].values.tolist()
df_date = df_waterlevel[['Time']].values.tolist()
df_SiteNo = df_waterlevel[['SiteNo']].values.tolist()
Dates = []
SiteNos = []
waterlevels = []
for i in df_waterlevel2:
    for j in i:
        waterlevels.append(j)
for i in df_date:
    for j in i:
        Dates.append(j)
for i in df_SiteNo:
    for j in i:
        SiteNos.append(j)
diff = {}
RY = -1
RM = -1
RD = -1
SiteNosAlready = []
origLevel = -1
DiffLevel = -1
for i in range(len(waterlevels)):
    if SiteNos[i] in SiteNosAlready:
        RY = Dates[i].split('-')[0]
        RM = Dates[i].split('-')[1]
        RD = Dates[i].split('-')[2]
        RD = RD.split('T')[0]
        diffLevel = int(waterlevels[i]) - origLevel
    if SiteNos[i] not in SiteNosAlready:
        SiteNosAlready.append(SiteNos[i])
        origLevel = int(waterlevels[i])
        diff[SiteNos[i]] = 0;

# keep only waterlevel
waterlevel_list = []
for l in df_waterlevel2:
    for each in l:
        waterlevel_list.append(each)
counted_waterlevel = collections.Counter(waterlevel_list)

# "How many variables are available?"
#print(len(df_waterlevel.columns))
# "What are their names?" (Columns)
# print(df_waterlevel.columns)

print("---- Water Level ----")
#Average
def Average(lt):
    return sum(lt) / len(lt)
Average_waterLevel = Average(waterlevel_list)
Average_waterLevel = round(Average_waterLevel, 2)
#Median
waterlevel_list.sort()
midPoint = len(waterlevel_list) // 2
Median_waterLevel = (waterlevel_list[midPoint] + waterlevel_list[~midPoint]) / 2
Median_waterLevel = round(Median_waterLevel, 2)
#Range
minR = min(waterlevel_list)
maxR = max(waterlevel_list)
Range_waterLevel = maxR-minR
Range_waterLevel = round(Range_waterLevel, 2)
#Standard Deviation
variance = sum([((x - Average_waterLevel) ** 2) for x in waterlevel_list]) / len(waterlevel_list)
Standard_Deviation_waterLevel = variance ** 0.5
Standard_Deviation_waterLevel = round(Standard_Deviation_waterLevel, 2)
#Print Results
print("Mean: " + str(Average_waterLevel))
print("Median: " + str(Median_waterLevel))
print("Range: " + str(Range_waterLevel))
print("Standard Deviation: " + str(Standard_Deviation_waterLevel))

# TEST
#for each in counted_waterlevel:
#    if counted_waterlevel[each] > 1:
#        print(each)
#        print(counted_waterlevel[each])
#        print("--------")
#print(counted_waterlevel)

# Plot waterlevel
colors = list("rgbcmyk")
x = counted_waterlevel.keys()
y = counted_waterlevel.values()
plt.scatter(x,y,color=colors.pop())
plt.legend(counted_waterlevel.keys())
plt.title("Water Level Distribution")
plt.ylabel("Frequency")
plt.xlabel("Water level in feet relative to NAVD88")
plt.show()

# keep only siteNo
SiteNo2 = df_waterlevel[['SiteNo']].values.tolist()
SiteNo_list = []
for l in SiteNo2:
    for each in l:
        SiteNo_list.append(each)
counted_siteNo = collections.Counter(SiteNo_list)
#Plot SiteNo
x_Site = counted_siteNo.keys()
y_Site = counted_siteNo.values()
plt.scatter(x_Site,y_Site,color=colors.pop())
plt.legend(counted_siteNo.keys())
plt.title("PlotNo Distribution")
plt.ylabel("Frequency")
plt.xlabel("Site (xVal^14)")
plt.show()

print("---- SiteNo ----")
#Mode
siteNo_Mode = -1
siteNo_Name = ""
for k in counted_siteNo:
    if counted_siteNo[k] > siteNo_Mode:
        siteNo_Mode = counted_siteNo[k]
        siteNo_Name = str(k);
print("Mode--> SiteNo: " + siteNo_Name + " Frequency: " + str(siteNo_Mode))

# keep only Date
Time2 = df_waterlevel[['Time']].values.tolist()
Time_list = []
for l in Time2:
    for each in l:
        # Only get year
        each = each.split('-')[0]
        Time_list.append(int(each))
counted_Time = collections.Counter(Time_list)
#Plot Time
x_Time = counted_Time.keys()
y_Time = counted_Time.values()
plt.scatter(x_Time,y_Time,color=colors.pop())
plt.legend(counted_Time.keys())
plt.title("Time Distribution")
plt.ylabel("Frequency")
plt.xlabel("Year")
plt.show()

print("---- Time ----")
#Mode
Time_Mode = -1;
Time_Name = ""
for k in counted_Time:
    if counted_Time[k] > Time_Mode:
        Time_Mode = counted_Time[k]
        Time_Name = str(k)
#Range
minR = min(Time_list)
maxR = max(Time_list)
Range_time = maxR-minR
Range_time = round(Range_time, 2)
print("Mode--> Time: " + Time_Name + " Frequency: " + str(Time_Mode))
print("Range: " + str(Range_time))

### DAY 2 ###
filename = "WaterLevelFrequency.csv"
f = open(filename, "w+")
f.close()

dataToCSV = {'WaterLevel': x,'Frequency': y}
df = pd.DataFrame.from_dict(dataToCSV)
df.to_csv('WaterLevelFrequency.csv')
# ///////////////////////////////////// #
df = pd.read_csv("WaterLevelFrequency.csv")

Y = df['WaterLevel']
X = df['Frequency']

#X=X.reshape(len(X),1)
#Y=Y.reshape(len(Y),1)

# find 1/4 spot
one_fourth_pos = len(waterlevel_list) / 4
one_fourth_pos = int(one_fourth_pos)

# split --> training/testing
X_train = X[:-one_fourth_pos]
X_test = X[-one_fourth_pos:]

Y_train = Y[:-one_fourth_pos]
Y_test = Y[-one_fourth_pos:]

# Plot outputs
plt.scatter(X_test, Y_test,  color='black')
plt.title('Test Data')
plt.xlabel('Water Level (Feet)')
plt.ylabel('')
plt.xticks(())
plt.yticks(())

plt.show()

# Plot outputs