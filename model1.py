import pandas as pd
import numpy as np
import pickle

df=pd.read_csv("fakeaccount_bot.csv")

#droping the unnecesary columns
df = df.drop('location',axis=1)
df = df.drop('lang',axis=1)
df = df.drop('created_at',axis=1)
df = df.drop('id',axis=1)
df =df.drop('id_str',axis=1)
df = df.drop('status',axis=1)
df = df.drop('url',axis=1)

df=df.dropna()

#using label encoder for string values
from sklearn import preprocessing
#Label Encoding
LE= preprocessing.LabelEncoder()
# Fitting it to our dataset
df.verified = LE.fit_transform(df.verified)
df.default_profile = LE.fit_transform(df.default_profile)
df.default_profile_image = LE.fit_transform(df.default_profile_image)
df.has_extended_profile = LE.fit_transform(df.has_extended_profile)

bag_of_words_bot = r'bot|b0t|cannabis|tweet me|mishear|follow me|updates every|gorilla|yes_ofc|forget' \
                   r'expos|kill|clit|bbb|butt|fuck|XXX|sex|truthe|fake|anony|free|virus|funky|RNA|kuck|jargon' \
                   r'nerd|swag|jack|bang|bonsai|chick|prison|paper|pokem|xx|freak|ffd|dunia|clone|genie|bbb' \
                   r'ffd|onlyman|emoji|joke|troll|droop|free|every|wow|cheese|yeah|bio|magic|wizard|face'

df['screen_name'] = df.screen_name.str.contains(bag_of_words_bot, case=False, na=False)
df['name'] = df.name.str.contains(bag_of_words_bot, case=False, na=False)
df['description'] = df.description.str.contains(bag_of_words_bot, case=False, na=False)

from sklearn import preprocessing
#Label Encoding
LE= preprocessing.LabelEncoder()
# Fitting it to our dataset
df.screen_name = LE.fit_transform(df.screen_name)
df.description = LE.fit_transform(df.description)
df.name = LE.fit_transform(df.name)

#standardizing the values
from sklearn.preprocessing import StandardScaler
sst = StandardScaler()
data_scaled=df.iloc[:,:-1].values
data_scaled=sst.fit_transform(data_scaled)
data_scaled=pd.DataFrame(data_scaled)
data_scaled.columns=['screen_name','description','followers_count','friends_count','listed_count','favourites_count','verified','statuses_count','default_profile','default_profile_image','has_extended_profile','name']
data_scaled['fake_account_bot'] = df.fake_account_bot
data_scaled=data_scaled.dropna()

x=data_scaled.iloc[:,:-1].values
y=data_scaled.iloc[:,-1].values

#spliting the data  into training and testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#TRAINING THE MODEL
from sklearn.ensemble import RandomForestClassifier
modelrf=RandomForestClassifier()
modelrf.fit(x_train,y_train)

y_pred=modelrf.predict(x_test)
print(y_pred)

pickle.dump(modelrf,open('model1.pkl','wb')) #we are Serializing our model by creating model.pkl and writing into it by 'wb'
model=pickle.load(open('model1.pkl','rb')) #Deserializing - reading the file - "rb"
print("Sucess loaded")
