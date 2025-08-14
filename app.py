# #!/usr/bin/env python
# # coding: utf-8
#
# # In[3]:
#
#
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
# from sklearn.preprocessing import LabelEncoder
#
# import zipfile
# import glob
#
# with zipfile.ZipFile('dataset.zip', 'r') as zip_ref:
#     zip_ref.extractall('new dataset')  # Folder where files will be extracted
#
#
# # In[4]:
#
#
# df = pd.concat([pd.read_pickle(f) for f in glob.glob("new dataset/data/*.pkl")], ignore_index=True)
#
# # Show first few rows
# print(df.head(5))
#
#
# # In[5]:
#
#
# df.info()
#
#
# # In[6]:
#
#
# df.shape
#
#
# # In[7]:
#
#
# df = df[["TRANSACTION_ID","TX_DATETIME","CUSTOMER_ID","TERMINAL_ID","TX_AMOUNT","TX_FRAUD"]]
#
#
# # In[8]:
#
#
# df.head()
#
#
# # In[9]:
#
#
# df.isnull().sum()
#
#
# # In[10]:
#
#
# fraud_distribution = df['TX_FRAUD'].value_counts(normalize=True)*100
# print(fraud_distribution.rename({0:'Legit',1:'Fraud'}).round(2).astype(str)+'%')
#
#
# # In[11]:
#
#
# #parse DATETIME
# df['TX_DATETIME'] =pd.to_datetime(df['TX_DATETIME'])
# #sort data
# df = df.sort_values('TX_DATETIME').reset_index(drop =True)
#
# #extract time features
# df['TX_DATE'] = df['TX_DATETIME'].dt.date
# df['TX_HOUR'] = df['TX_DATETIME'].dt.hour
#
#
# # In[12]:
#
#
# #condition 1 TX_AMOUNT >220
# df['Rule1_fraud'] = (df['TX_AMOUNT']>220).astype(int)
# #--------------------------------------------------------------------------------------------------------------------
# #condition 2
# terminal_fraud_dict = {}
# unique_dates = df['TX_DATE'].unique()
#
# for i in range(len(unique_dates)):
#     if i >=len(unique_dates):break
#     day = unique_dates[i]
#     next_28_days = unique_dates[i:i+28]
#     sampled_terminals = df[df['TX_DATE'] == day]['TERMINAL_ID'].sample(2,random_state = i).tolist()
#     for t in sampled_terminals:
#         for d in next_28_days:
#             terminal_fraud_dict.setdefault((t,d),1)
# df['Rule2_fraud'] = df.apply(lambda x: terminal_fraud_dict.get((x['TERMINAL_ID'],x['TX_DATE']),0),axis = 1)
# #--------------------------------------------------------------------------------------------------------------------
# #condition 3 -> 3 random customers per day -> 1/3 of next 14 days *5 amount
# df['Rule3_fraud'] = 0
# customer_fraud_dict = {}
#
# for i in range(len(unique_dates)):
#     day = unique_dates[i]
#     next_14_days = unique_dates[i:i+14]
#     sampled_customers = df[df['TX_DATE'] == day]['CUSTOMER_ID'].drop_duplicates().sample(3,random_state = i)
#     for cust in sampled_customers:
#         customer_fraud_dict.setdefault(cust,[])
#         for d in next_14_days:
#             customer_fraud_dict[cust].append(d)
#
# for cust, days in customer_fraud_dict.items():
#     mask = (df['CUSTOMER_ID'] == cust) & (df['TX_DATE'].isin(days))
#     eligible_indices = df[mask].sample(frac = 1/3, random_state = 42).index
#
# #--------------------------------------------------------------------------------------------------------------------
# # final label
# df['TX_FRAUD'] = df[['Rule1_fraud','Rule2_fraud','Rule3_fraud']].max(axis=1)
#
#
# # In[13]:
#
#
# #EDA
# #pie chart for LEGIT AND FRAUD
# plt.figure(figsize=(6,6))
# df['TX_FRAUD'].value_counts().plot.pie(autopct='%1.1f%%',labels =['Legit','Fraud'],colors =['green','red'])
# plt.title('Fraud Vs Legit Transaction')
# plt.show()
#
#
# # In[14]:
#
#
# #kde plot to visualize amount difference
# plt.figure(figsize=(8,5))
# sns.kdeplot(df[df['TX_FRAUD'] == 0]['TX_AMOUNT'],label='Legit',shade = True)
# sns.kdeplot(df[df['TX_FRAUD'] == 0]['TX_AMOUNT'],label ='Fraud',color = 'red')
# plt.title('Transaction Amount By Fraud Status')
# plt.legend()
# plt.show()
#
#
# # In[15]:
#
#
# #to check feature correlation
# plt.figure(figsize =(8,5))
# sns.heatmap(df[['TX_AMOUNT','TX_HOUR','TX_FRAUD']].corr(),annot = True, cmap = 'viridis')
# plt.title("Feature Correlation")
# plt.show()
#
#
# # In[16]:
#
#
# #feature Engineering
# features = ['TX_AMOUNT','TX_HOUR']
# x= df[features]
# y = df['TX_FRAUD']
#
# #train test split
# X_train,X_test, y_train,y_test = train_test_split(x, y,test_size = 0.2,random_state = 42)
#
#
# # In[17]:
#
#
# X_train
#
#
# # In[18]:
#
#
# model = RandomForestClassifier(n_estimators=20,max_depth=10,random_state=42)
# x_small = X_train.sample(1000,random_state=42)
# y_small = y_train.loc[x_small.index]
# model.fit(x_small,y_small)
#
#
# # In[19]:
#
#
# # prediction and evaluation
# y_pred = model.predict(X_test)
# print("classification report/n",classification_report(y_test,y_pred))
#
#
# # In[20]:
#
#
# import joblib
#
# # Assuming your trained model is named `model`
# joblib.dump(model, "model.pkl")
# print("✅ Model saved as model.pkl")
#
#
# # In[ ]:
#
#
#
#


import gradio as gr
import joblib
import numpy as np

# Load model
model = joblib.load("model.pkl")

# Prediction function
def predict_fraud(amount, transaction_type, old_balance, new_balance):
    X = np.array([[amount, transaction_type, old_balance, new_balance]])
    prediction = model.predict(X)[0]
    return "🚨 Fraud Detected" if prediction == 1 else "✅ Legitimate Transaction"

# Gradio app
app = gr.Interface(
    fn=predict_fraud,
    inputs=[
        gr.Number(label="Transaction Amount"),
        gr.Number(label="Transaction Type"),
        gr.Number(label="Old Balance"),
        gr.Number(label="New Balance")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Fraud Detection System",
    description="Enter transaction details to check if they are fraudulent."
)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=8080)

