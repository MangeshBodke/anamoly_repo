# import json
import pickle

from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

app = Flask(__name__)
isf = pickle.load(open('IsolationForest.pkl', 'rb'))

# @app.route('/')
# def home():
#     return render_template('home.html')


# @app.route('/',methods=['POST'])

drop_columns = {'BMI', 'Id', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32',
                'Medical_Keyword_1', 'Medical_Keyword_10', 'Medical_Keyword_11', 'Medical_Keyword_12',
                'Medical_Keyword_13', 'Medical_Keyword_14', 'Medical_Keyword_15', 'Medical_Keyword_16',
                'Medical_Keyword_17', 'Medical_Keyword_18', 'Medical_Keyword_19', 'Medical_Keyword_2',
                'Medical_Keyword_20', 'Medical_Keyword_21', 'Medical_Keyword_22', 'Medical_Keyword_23',
                'Medical_Keyword_24', 'Medical_Keyword_25', 'Medical_Keyword_26', 'Medical_Keyword_27',
                'Medical_Keyword_28', 'Medical_Keyword_29', 'Medical_Keyword_3', 'Medical_Keyword_30',
                'Medical_Keyword_31', 'Medical_Keyword_32', 'Medical_Keyword_33', 'Medical_Keyword_34',
                'Medical_Keyword_35', 'Medical_Keyword_36', 'Medical_Keyword_37', 'Medical_Keyword_38',
                'Medical_Keyword_39', 'Medical_Keyword_4', 'Medical_Keyword_40', 'Medical_Keyword_41',
                'Medical_Keyword_42', 'Medical_Keyword_43', 'Medical_Keyword_44', 'Medical_Keyword_45',
                'Medical_Keyword_46', 'Medical_Keyword_47', 'Medical_Keyword_48', 'Medical_Keyword_5',
                'Medical_Keyword_6', 'Medical_Keyword_7', 'Medical_Keyword_8', 'Medical_Keyword_9',
                'Product_Info_2'}

selected_features = {'Employment_Info_3', 'Medical_History_16', 'Medical_History_28', 'Medical_History_4',
                     'Employment_Info_2', 'InsuredInfo_5', 'Medical_History_40', 'Medical_History_33', 'Wt',
                     'InsuredInfo_6', 'Medical_History_23', 'Family_Hist_1', 'Medical_History_13',
                     'Insurance_History_2', 'InsuredInfo_1', 'Product_Info_4', 'Ins_Age', 'Medical_History_30',
                     'Medical_History_39', 'Ht', 'Medical_History_6', 'Family_Hist_4', 'InsuredInfo_7'}

outliers = {'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5', 'Insurance_History_5',
            'Medical_History_1', 'Medical_History_10', 'Medical_History_15', 'Medical_History_24',
            'Medical_History_32', 'Product_Info_2_code', 'Product_Info_3', 'Product_Info_5'}

data = pd.read_csv('test.csv')
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(data['Product_Info_2'])
data['Product_Info_2_code'] = le.transform(data['Product_Info_2'])

data.drop(drop_columns, axis=1, inplace=True)
# new_data = data[selected_features]
# print(new_data)

anomaly = outliers.intersection(selected_features)
# print(anomaly)

df_outlier = data[~data['Family_Hist_4'].isna()].copy()
# print(df_outlier)

isf = pickle.load(open('IsolationForest.pkl', 'rb'))

df_outlier['scores'] = isf.decision_function(df_outlier[['Family_Hist_4']])
df_outlier['anomaly'] = isf.predict(df_outlier[['Family_Hist_4']])

fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
sns.countplot(ax=ax[0], x='anomaly', data=df_outlier)
sns.scatterplot(ax=ax[1], y='scores', x='Family_Hist_4', hue='anomaly', data=df_outlier)
plt.show()
fig.savefig('Anamoly_detection.png')

if __name__=="__main__":
    app.run(debug=True)
