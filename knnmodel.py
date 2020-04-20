#import packages
import pandas as pd
import numpy as np
import datetime
import calendar
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#to plot within notebook
import matplotlib.pyplot as plt
pd.plotting.register_matplotlib_converters() #restore pandas timestamp compatibility issue with matplotlib

#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#read the file
df = pd.read_csv("tataglobal.csv")

#setting index as date
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

#plot graph of original data set
plt.figure(figsize=(16,8))
#plt.plot(df['Close'], label='Close Price history')
#plt.show()

print('\n Shape of the data:')
print(df.shape) #(1235,8)

#creating dataframe with date and the target variable ONLY
# sets index as date values
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
for i in range(0,len(data)):
     new_data['Date'][i] = data['Date'][i]
     new_data['Close'][i] = data['Close'][i]

# Adds new columns to dataset
list_of_year = []
list_of_months = []
list_of_weeks = []
list_of_days = []
list_of_day_of_week = []
list_of_Dayofyear = []
list_of_Is_month_end = []

for i in new_data["Date"]:
    date = (i.date())
    date = str(date)
    year = int(date[:4])
    month = int(date[5:7])
    day = int(date[8:10])
    week = datetime.date(year, month, day).isocalendar()[1]
    date_in_number = datetime.datetime.strptime(date, '%Y-%m-%d').weekday()
    day_of_year = datetime.date(year,month,day).timetuple().tm_yday
    last_day_of_month = calendar.monthrange(year, month)[1] # last day of the month

    list_of_weeks.append(week)
    list_of_year.append(year)
    list_of_months.append(month)
    list_of_days.append(day)
    list_of_day_of_week.append(date_in_number)
    list_of_Dayofyear.append(day_of_year)
    if day == last_day_of_month:
        list_of_Is_month_end.append(True)
    else:
        list_of_Is_month_end.append(False)

new_data['Year'] = list_of_year
new_data['Month'] = list_of_months
new_data['Week'] = list_of_weeks
new_data['Day'] = list_of_days
new_data['Dayofweek'] = list_of_day_of_week
new_data['Dayofyear'] = list_of_Dayofyear
new_data['Is_month_end'] = list_of_Is_month_end

# Adds new column Is_month_start to dataset
list_of_Is_month_start = []
for i in new_data["Date"]:
    date = (i.date())
    date = str(date)
    day = date[8:10]
    if day == "01":
        list_of_Is_month_start.append(True)
    else:
        list_of_Is_month_start.append(False)
new_data['Is_month_start'] = list_of_Is_month_start

# Adds new column Is_quarter_end to dataset
list_of_Is_quarter_end = []
for i in new_data["Date"]:
    date = (i.date())
    date = str(date)
    month_day = date[5:10]
    if month_day == "03-31" or month_day == "06-30" or month_day == "09-30" or month_day == "12-31":
        list_of_Is_quarter_end.append(True)
    else:
        list_of_Is_quarter_end.append(False)
new_data['Is_quarter_end'] = list_of_Is_quarter_end

# Adds new column Is_quarter_start to dataset
list_of_Is_quarter_start = []
for i in new_data["Date"]:
    date = (i.date())
    date = str(date)
    month_day = date[5:10]
    if month_day == "01-01" or month_day == "04-01" or month_day == "07-01" or month_day == "10-01":
        list_of_Is_quarter_start.append(True)
    else:
        list_of_Is_quarter_start.append(False)
new_data['Is_quarter_start'] = list_of_Is_quarter_start

# Adds new column Is_year_end to dataset
list_of_Is_year_end = []
for i in new_data["Date"]:
    date = (i.date())
    date = str(date)
    month_day = date[5:10]
    if month_day == "12-31":
        list_of_Is_year_end.append(True)
    else:
        list_of_Is_year_end.append(False)
new_data['Is_year_end'] = list_of_Is_year_end

# Adds new column Is_year_start to dataset
list_of_Is_year_start = []
for i in new_data["Date"]:
    date = (i.date())
    date = str(date)
    month_day = date[5:10]
    if month_day == "01-01":
        list_of_Is_year_start.append(True)
    else:
        list_of_Is_year_start.append(False)
new_data['Is_year_start'] = list_of_Is_year_start

# Adds new column mon_fri to dataset
list_mon_fri = []
for i in new_data["Dayofweek"]:
    if (i == 0 or i == 4):
        list_mon_fri.append(1)
    else:
        list_mon_fri.append(0)
new_data["mon_fri"] = list_mon_fri

#split into train and validation
train = new_data[:987]
valid = new_data[987:]
print(train.head())

x_train = train.drop('Close', axis=1)
x_train = x_train.drop('Date', axis=1)
y_train = train['Close']
x_valid = valid.drop('Close', axis=1)
x_valid = x_valid.drop('Date', axis=1)
y_valid = valid['Close']

#scaling data
x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)
x_valid_scaled = scaler.fit_transform(x_valid)
x_valid = pd.DataFrame(x_valid_scaled)

#using gridsearch to find the best parameter
params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)

#fit the model and make predictions
model.fit(x_train,y_train)
preds = model.predict(x_valid)

#rmse
rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
print(rms)

#plot
valid['Predictions'] = 0
valid['Predictions'] = preds
plt.plot(valid[['Close', 'Predictions']])
plt.plot(train['Close'])
plt.show()
