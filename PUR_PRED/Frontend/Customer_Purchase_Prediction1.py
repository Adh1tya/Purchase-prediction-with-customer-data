import streamlit as st
import pandas as pd
from PIL import Image
import pickle
from sklearn import preprocessing

model = pickle.load(open('../Model/ML_Model_Customer_Purchase_Prediction1.pkl', 'rb'))

def run():
    img1 = Image.open('bank.png')
    img1 = img1.resize((156,145))
    st.image(img1,use_column_width=False)
    st.title("Customer Purchasing Option Prediction")

    ## Gender of the Customer
    gen_display = ('Female','Male')
    gen_options = list(range(len(gen_display)))
    gen = st.selectbox("Gender",gen_options, format_func=lambda x: gen_display[x])

    ## Age of the Customer
    age = st.number_input('Age of the Customer', value=0)

    ## Estimated Salary of the Customer
    est_sal = st.number_input('Estimated Salary of the Customer', value=0)

    #--------Normalization of Customer Data ----------------------
    columnNames = ['gen', 'age', 'est_sal']
    lst=[gen, age, est_sal]
    df=pd.DataFrame(lst)
    min_max_scaler_object = preprocessing.MinMaxScaler()
    df1 = min_max_scaler_object.fit_transform(df)
    df1 = pd.DataFrame(df1 , columns = columnNames)
    #gen1 = min_max_scaler_object.fit_transform(gen)
    #age1 = min_max_scaler_object.fit_transform(age)
    #est_sal1 = min_max_scaler_object.fit_transform(est_sal)
    #----------------------------------------------------------------
    #lst=[gen1, age1, est_sal1]
    #df=pd.DataFrame(lst)
    #gen2= gen1.reshape(-1,1)
    #age2= age1.reshape(-1,1)
    #est_sal2= est_sal1.reshape(-1,1)
    if st.button("Submit"):

        #features = [[gen2, age2, est_sal2]] 
        #features1 = features.reshape(1,-1)
        #features = [gen1, age1, est_sal1] 
        #print(features)
        prediction = model.predict(df1)
        print(prediction)
        lc = [str(i) for i in prediction]
        #lc1=int(lc)
        #print(lc)
        ans = int("".join(lc))
        #print(ans)
        if ans == 0:
            st.error(
                 'According to ML prediction, the Customer will not make a Purchase'
            )
        else:
            st.success(
                 'According to ML Prediction, the Customer will make a Purchase'
            )

run()