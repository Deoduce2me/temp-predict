import pandas as pd
import seaborn as sns
import streamlit as st
# import plotly.express as px
# import shap
st.set_option('deprecation.showPyplotGlobalUse', False)
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import string
import altair as alt
import pickle
st.title('Temperature Prediction')
st.write('This app predicts Temperature value from other Parameters ')

abid=pd.read_csv(r'C:\Users\Adeola\Desktop\RESEARCH\DATA\croschek\Sortdate\Abid Combine.csv')
abid['Month'] = pd.to_datetime(abid['Month'], format='%m').dt.month_name().str.slice(stop=3)
xx=abid.drop(['Month','Temperature (degree celsius)'],axis=1)
#xx
yy= abid[['Temperature (degree celsius)']]

# abid
# st.table(abid.plot(x="Month", kind="line",figsize=(18,5)))

st.table(abid)
# chart = alt.Chart(abid).mark_line().encode(
#   x=alt.X('Month'),
#   y=alt.Y('value:Q')
# ).properties(title="Hello World")
# st.altair_chart(chart, use_container_width=True)

st.line_chart(abid.drop(['Month'],axis=1))
if st.checkbox("Show Correlation Plot"):
    st.write("### Heatmap")
    fig, ax = plt.subplots()
    st.write(sns.heatmap(abid.drop(['Month'],axis=1).corr(), annot=True, linewidths=0.5))
    st.pyplot()

# figs = plt.figure()
# ax = figs.add_subplot(1,1,1)
#
# plt.scatter(abid)
#
# st.write(fig)
# xopt=['Month']
# yopt=["Temperature (degree celsius)", "Relative Humidity(%)","Pressure(hPa)","Wind speed(m/s)","Wind direction(deg)","Rainfall(mm)"]
# figs= px.scatter(abid,x='Month',y=yopt)
#
# st.plotly_chart(figs)



#loading our model
model1 = pickle.load(open('Best.pkl','rb'))
r_sq = model1.score(xx, yy)
print('coefficient of determination:', r_sq)
# st.write("### r_sq")

def main():
  st.markdown("<h1 style='text-align: center; color: White;background-color:#e84343'>TEMPERATURE PREDICTOR</h1>", unsafe_allow_html=True)
  # st.markdown("<h3 style='text-align: center; color: Black;'>Drop in The required Inputs and we will do  the rest.</h3>", unsafe_allow_html=True)
  st.markdown("<h4 style='text-align: center; color: Black;'>Submission for ADS PROJECT</h4>", unsafe_allow_html=True)
  st.sidebar.header("What is this Project about?")
  st.sidebar.text("It is a Web app that would help predict Temperature.")
  st.sidebar.header("What tools were used to make this?")
  st.sidebar.text("The Model was made using a dataset from http://www./soda-pro.com/web-services/radiation/helioclim-1 along with using jupyter notebook to train the model. We made use of Sci-Kit learn in order to make our Linear Regression Model.")

Rel_Hum = st.slider("Input your Relative Humidity(%)",0.000,100.000)
press= st.slider("Input your Pressure(hPa)",0.000,1500.000)
wind_speed = st.slider("Input your Wind speed(m/s)",0.000,10.000)
wind_direction = st.slider("Input your Wind direction(deg)",0.000,500.000)
rainfall = st.slider("Input your Rainfall(mm)",0.000,500.000)

inputs = [[Rel_Hum,press,wind_speed,wind_direction,rainfall]] #our inputs

if st.button('Predict'): #making and printing our prediction
    result = model1.predict(inputs)
    updated_res = result.flatten().astype(float)
    st.success('The Temperature is {}degree celsius'.format(updated_res))


if __name__ =='__main__':
  main() #calling the main method