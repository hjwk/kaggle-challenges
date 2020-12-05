import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

data_train = pd.read_csv('data/train.csv')
data_test = pd.read_csv('data/test.csv')

data = data_train.append(data_test, ignore_index=True)

# Cleaning and feature engineering
def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

data = simplify_ages(data)

def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

data = simplify_fares(data)

def create_family(df):
    df['Family'] = df.Parch + df.SibSp + 1
    return df

family = create_family(data).Family

# Selecting features
sex = pd.Series( np.where( data.Sex == 'male', 1 , 0 ) , name = 'Sex' )
embarked = pd.get_dummies( data.Embarked, prefix='Embarked' )
pclass = pd.get_dummies( data.Pclass, prefix='Pclass' )
fare = pd.get_dummies( data.Fare, prefix='Fare' )
age = pd.get_dummies( data.Age, prefix='Age' )

X_full = pd.concat([sex, embarked, pclass, fare, age, family], axis=1)

print(X_full.head())

# Splitting the data
surv = data.Survived
X_train = X_full.values[:712]
y_train = surv.values[:712]
X_val = X_full.values[713:890]
y_val = surv.values[713:890]
X_test = X_full.values[891:]

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.hdf5', 
                               verbose=1, save_best_only=True)

model = Sequential()
model.add(Dense(6, activation='relu', input_dim = 21))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.fit(X_train, y_train,
          validation_data=(X_val, y_val), 
          epochs=120, batch_size=32,
          callbacks=[checkpointer])

model.load_weights('saved_models/weights.best.hdf5')
output = model.predict(X_test)

f = open('results.csv', 'w')
f.write('PassengerID,Survived\n')
idNum = 892
for o in output:
    f.write(str(idNum))
    f.write(',')
    if o >= 0.5:
        f.write('1\n')
    else:
        f.write('0\n')
    idNum += 1
f.close()
