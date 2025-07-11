import streamlit as st
from PIL import Image
import pickle


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

    
    if st.button("Submit"):

        features = [[gen, age, est_sal]] 
 
        print(features)
        prediction = model.predict(features)
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