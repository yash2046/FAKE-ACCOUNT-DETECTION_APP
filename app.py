import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

df=pd.read_csv("fakeaccount_bot.csv")
app = Flask(__name__)

#Deserialize
model = pickle.load(open('model1.pkl','rb'))


def generate_substrings(test_str):
    # printing original string
    print("The original string is : " + str(test_str))
    # Get all substrings of string
    # Using list comprehension + string slicing
    res = [test_str[i: j] for i in range(len(test_str))
           for j in range(i + 1, len(test_str) + 1)]
    # printing result
    return res

def convert(string1):
    bag_of_words_bot = r'bot|b0t|cannabis|tweet me|mishear|follow me|updates every|gorilla|yes_ofc|forget' \
                       r'expos|kill|clit|bbb|butt|fuck|XXX|sex|truthe|fake|anony|free|virus|funky|RNA|kuck|jargon' \
                       r'nerd|swag|jack|bang|bonsai|chick|prison|paper|pokem|xx|freak|ffd|dunia|clone|genie|bbb' \
                       r'ffd|onlyman|emoji|joke|troll|droop|free|every|wow|cheese|yeah|bio|magic|wizard|face'
    bad_words_l1 = bag_of_words_bot.split("|")
    string1 = string1.lower()
    print(generate_substrings(string1))
    res = generate_substrings(string1)
    for i in res:
        if i in bad_words_l1:
            print(i)
            print("True")
            return 1
    else:
        print("False")
        return 0


@app.route('/')
def index():
    return render_template("index.html") #due to this function we are able to send our webpage to client(browser) - GET


@app.route('/predict',methods=['POST','GET'])  #gets inputs data from client(browser) to Flask Server - to give to ml model
def predict():
    features = [(x) for x in request.form.values()]
   # print(features)
    final=features
    #our model was trained on Normalized(scaled) data
    df = pd.read_csv("fakeaccount_bot.csv")
    df = df.drop('location', axis=1)
    df = df.drop('lang', axis=1)
    df = df.drop('created_at', axis=1)
    df = df.drop('id', axis=1)
    df = df.drop('id_str', axis=1)
    df = df.drop('status', axis=1)
    df = df.drop('url', axis=1)
    from sklearn import preprocessing
    # Label Encoding
    LE = preprocessing.LabelEncoder()
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
    # Label Encoding
    LE = preprocessing.LabelEncoder()
    # Fitting it to our dataset
    df.screen_name = LE.fit_transform(df.screen_name)
    df.description = LE.fit_transform(df.description)
    df.name = LE.fit_transform(df.name)

    some_va = convert(final[0])
    some_va1 = convert(final[1])
    some_va2 = convert(final[11])
    final[0] = some_va
    final[1] = some_va1
    final[11] = some_va2

    for i in range(0, 11):
        if final[i] == 'False':
            final[i] = 0
        elif final[i] == 'True':
            final[i] = 1
    for i in range(0, 11):
        final[i] = int(final[i])

    X = df.iloc[:, 0:12].values
    print(len(X[0]))
    sst=StandardScaler().fit(X)
    final = [np.array(final)]
    print(len(final[0]))
    print(final)
    prediction = model.predict_proba(sst.transform(final))
    output = '{0:.{1}f}'.format(prediction[0][1], 2)
    print(output)
    return render_template('index.html',
                           pred='Probability of being fake account is {}'.format(output))


'''if output > str(0.5):
        return render_template('index.html',
                               pred='fake account Detected\nProbability of being bot is {}'.format(output))
    else:
        return render_template('index.html',
                               pred='Not a TwitterBot.\n Probability of being bot is {}'.format(output))'''
if __name__ == '__main__':
    app.run(debug=True)