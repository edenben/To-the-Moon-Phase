#!/usr/bin/env python
# coding: utf-8

# # To the Moon [Phase]! 🚀🌕

# ## 📑 Import Libraries

# In[1]:


# !pip install pyorbital
# !pip install https://github.com/pytroll/pyorbital/archive/feature-moon-phase.zip
# !pip install streamlit


# In[2]:


from requests import Request, Session
import json
from datetime import datetime
import pandas as pd
from pyorbital.moon_phase import moon_phase
import numpy as np
import streamlit as st
import altair as alt


# ## 💲 Bitcoin Prices Dataset

# In[3]:


end = datetime.now()


# In[4]:


end_str = str(end.year) + '-' + str(end.month).zfill(2) + '-' + str(end.day).zfill(2)


# In[5]:


start_str = '2013-09-01'


# In[6]:


url =  f'https://api.coindesk.com/v1/bpi/historical/close.json?start={start_str}&end={end_str}'


# In[7]:


session = Session()


# In[8]:


response = session.get(url)


# In[9]:


response.json()


# In[10]:


df_btc = pd.json_normalize(response.json()['bpi']).transpose()


# In[11]:


df_btc = df_btc.reset_index()


# In[12]:


# df_btc.columns


# In[13]:


df_btc.rename(columns={'index': 'date', 0: 'price'}, inplace=True)


# In[14]:


df_btc.head()


# ## 🌙 Moon Phases Dataset

# In[15]:


time_t = np.arange(start_str, end_str, dtype='datetime64[D]')


# In[16]:


df_moon = pd.DataFrame({
    'date':np.arange(start_str, end_str, 1 ,dtype='datetime64[D]'),
    'minutes':(moon_phase(time_t))
})


# In[17]:


# full moon ~ 1
# new moon  ~ 0
df_moon.head(40)


# In[18]:


df_moon['increasing'] = df_moon['minutes'] > df_moon['minutes'].shift(1)


# In[19]:


tmp_val = 0
new_moon = []
for id_row, row in df_moon.iterrows():
  x = row['increasing']*(1+tmp_val)
  tmp_val = x
  new_moon.append(x)
df_moon['sum_of_increasing'] = new_moon


# In[20]:


df_moon.head(20)


# In[21]:


df_moon['phase'] = ''


# In[22]:


for i, row in df_moon.iterrows():
    if i==0:
        continue
    if ((df_moon.at[i-1, 'sum_of_increasing'] == 0) and (df_moon.at[i, 'sum_of_increasing'] == 1)):
        df_moon.at[i-1, 'phase'] = 'new'
    elif ((df_moon.at[i-1, 'sum_of_increasing'] > 0) and (df_moon.at[i, 'sum_of_increasing'] == 0)):
        df_moon.at[i-1, 'phase'] = 'full'
    else:
        df_moon.at[i-1, 'phase'] = ''


# In[23]:


df_moon.head(55)


# ## 📦 Merging Dataframes

# In[24]:


df_all = pd.concat([df_btc, df_moon], axis=1, join='inner')


# In[25]:


df_all.head()


# In[26]:


df_all = df_all.reset_index()
df_all.columns.values[3] = "date2"
df_all = df_all.drop(columns=['date2', 'index'], index=1)


# In[27]:


df_small = df_all[df_all['phase']!='']


# In[28]:


df_small.head(20)


# In[29]:


df_small['difference'] = df_small['price'].diff()


# In[30]:


df_small['percent_change'] = df_small['price'].pct_change()


# In[31]:


# df_small.drop('')


# In[32]:


df_small.head()


# ## 🔢 Investment Calculator

# In[33]:


df_full = df_small[['date', 'price']][df_small['phase']=='full']
df_new = df_small[['date', 'price', 'percent_change']][df_small['phase']=='new']


# In[34]:


df_new.head()


# In[35]:


df_new = df_new.iloc[1:]
df_new.head()


# In[36]:


# investment = 1

# for i, row in df_full.iterrows():
#     if i==0:
#         continue        
        
#     df_full['btc_owned'] = investment/df_full['price']
#     investment = df_full['btc_owned']


# # 🔀 Pivot Tables

# In[37]:


df_new = df_new.reset_index()


# In[38]:


df_full = df_full.reset_index()


# In[39]:


df_stream = df_new[['date','price']]
df_stream = df_stream.rename(columns = {'date':'date_new', 'price':'price_new'})
df_stream.head()


# In[40]:


df_stream = pd.concat([df_full, df_stream], axis=1)
df_stream = df_stream.rename(columns = {'date':'date_full', 'price':'price_full'})
df_stream.head()


# In[41]:


df_stream = df_stream.drop(columns=['index'])


# In[42]:


# Buy on Full, sell on New
df_stream['difference'] = df_stream['price_new'] - df_stream['price_full']


# In[43]:


df_stream['returns'] = ''


# In[44]:


df_stream.tail()


# ## 📈 Streamlit Dashboard

# ### Version 1️⃣

# In[45]:


# # Page setup
# st.set_page_config(page_icon=':chart_with_upwards_trend:',
#                    layout='wide', # either 'centered' or 'wide'
#                    initial_sidebar_state='expanded')


# In[46]:


# # Defining containers
# header = st.container()
# sidebar = st.container()
# description = st.container()
# calculations = st.container()
# visualization = st.container()


# In[47]:


# # Sidebar section
# with sidebar:
#     st.sidebar.header('Input Values')
#     initial_investment = st.sidebar.number_input('Investment Amount', value=100.0, min_value=0.01)

#     invest_date = st.sidebar.date_input('Buy Date',  value=pd.to_datetime('2021-09-01'))
#     sell_date = st.sidebar.date_input('Sell Date', value=pd.to_datetime('today'), min_value=invest_date)

# # Create a date range section
# #     invest_sell_date = st.sidebar.date_input(
# #         'Buy Date',  
# #         value=[pd.to_datetime('2021-07-01'), pd.to_datetime(df_all.max()['date'])],
# #         min_value=pd.to_datetime(df_all.min()['date']),
# #         max_value=pd.to_datetime(df_all.max()['date'])
# #     )

# #     if len(tuple(invest_sell_date)) == 1:
# #         st.sidebar.error('Error: Please choose date range.')
# #         invest_date, sell_date = pd.to_datetime('2021-07-01'), pd.to_datetime(df_all.max()['date'])
# #     else:
# #         invest_date, sell_date = invest_sell_date


# In[48]:


# # Error message on dates
# def get_range(df, start, end):
#     # Filtered DataFrame based on inputs
#     return df[
#         (pd.to_datetime(df['date'])>=pd.to_datetime(start))
#         &
#         (pd.to_datetime(df['date'])<=pd.to_datetime(end))][['date','price','phase']]

# def correct_moon_order(df):
#     moon_count = df[df['phase'].isin(['new','full'])]
#     if moon_count.shape[0] > 2:
#         return True
#     elif moon_count.shape[0] == 2 and moon_count.iloc[0]['phase'] == 'full':
#         return True
#     elif moon_count.shape[0] == 2 and moon_count.iloc[0]['phase'] == 'new':
#         return False
#     else:
#         return False

# if not correct_moon_order(get_range(df_all, invest_date, sell_date)):
#     st.sidebar.error('Error: Please choose a wider date range.')
#     invest_date, sell_date = pd.to_datetime('2021-07-01'), pd.to_datetime(df_all.max()['date'])


# In[49]:


# # Calculate returns column based on inputs from the Sidebar
# def calc_returns(df, invest_date, sell_date):
#     df_returns = df.copy()
#     df_returns = df_returns[
#         (pd.to_datetime(df_returns['date_full']) >= pd.to_datetime(invest_date))
#         &
#         (pd.to_datetime(df_returns['date_new']) <= pd.to_datetime(sell_date))
#     ]
#     init_inv = float(initial_investment)
#     for id_row, row in df_returns.iterrows():
#         x = ((init_inv / row['price_full']) * row['price_new'])
#         init_inv = x
#         df_returns.at[id_row, 'returns'] = x
#     return df_returns

# df_returns = calc_returns(df_stream, invest_date, sell_date)


# In[50]:


# # Header section
# with header:
#     st.title('🚀 To the Moon [Phase]!')
#     # st.subheader('Examining whether buying and selling ₿itcoin based on the moon phase is a good investment strategy 🌙')
#     st.subheader('Lunar investing 🌙')


# In[51]:


# # Description section
# with description:
#     row1_1, row1_2, row1_3 = st.columns(3)
#     with row1_1.expander('Goal', expanded=True):
#         st.write('''**What is the goal of this app?**
#                 The goal of this app is to examine whether buying and selling Bitcoin based on the moon phase is a good investment strategy.''')

#     with row1_2.expander('Description & Definition', expanded=False):
#         st.write('''
#                 **What is this app?**
#                 This app is a Bitcoin backtesting simulator for the moon phase trading strategy.
                
#                 **What is the moon phase trading strategy?** The moon phase trading strategy is:
#                 - Buy on the full moon 🌕
#                 - Sell on the new moon 🌑
                
#                 This app calculates the ROI of using this trading stratefy between the chosen historical date range.
#                 ''')

#     with row1_3.expander('Instructions', expanded=False):
#         st.write('''**How do I use it?**
#                 In the sidebar, enter your investment amount and historical date range you'd like to review.
#                 Your first purchase will be made on the first Full Moon after your selected Buy Date.''')


# In[52]:


# # Filtered DataFrame based on inputs
# df_viz = df_all[
#         (pd.to_datetime(df_all['date'])>=pd.to_datetime(invest_date))
#         &
#         (pd.to_datetime(df_all['date'])<=pd.to_datetime(sell_date))][['date','price','phase']]


# In[53]:


# # Dynamic calculations section
# with calculations:
    
#     # Calculate ROI total
#     return_on_investment = (df_returns[
#                         (pd.to_datetime(df_returns['date_full']) >= pd.to_datetime(invest_date))
#                         &
#                         (pd.to_datetime(df_returns['date_new']) <= pd.to_datetime(sell_date))]['returns'].dropna().values[-1])
#     roi_total = '${:,.2f}'.format(return_on_investment)
    
#     # Calculate ROI percent
#     percent_return = (((return_on_investment - initial_investment) / initial_investment))
#     roi_percentage = '{:,.2%}'.format(percent_return)
    
#     # Count number of Moon Cycles
#     count_moon_cycle = df_viz[df_viz['phase'].isin(['new','full'])].shape[0]/2
    
#     # Show calcuations of ROIs
#     row2_1, row2_2, row2_3, row2_4, row2_5 = st.columns(5)

#     with row2_1:
#         st.metric(label='Current Value', value=roi_total)

#     with row2_3:
#         st.metric(label='Percentage Total Gain', value=roi_percentage)
        
#     with row2_5:
#         st.metric(label='Total Moon Cycles', value=count_moon_cycle)

# # Dynamic viz section
# with visualization:

#     # Base line chart with Bitcoin prices
#     chart_price = alt.Chart(df_viz, height=500).mark_line().encode(
#                 x=alt.X('date:T', axis=alt.Axis(labelOverlap="parity", grid=False), title="Date"),
#                 y=alt.Y('price:Q', title="Price in USD"),
#                 color=alt.value('#FAA307'),
#                 tooltip=['date:T','price:Q']).interactive()

#     # DataFrames based on df_viz separated by the phase of the moon
#     df_viz_fullmoon = df_viz[df_viz['phase'] == 'full']
#     df_viz_newmoon = df_viz[df_viz['phase'] == 'new']
    
#     #DataFrames for first buy and last sell
#     df_viz_first_fullmoon = pd.DataFrame(df_viz_fullmoon.iloc[0]).T
#     df_viz_last_newmoon = pd.DataFrame(df_viz_newmoon.iloc[-1]).T

#     # Chart for all the New Moons
#     chart_newmoon = alt.Chart(df_viz_newmoon).mark_text(text='🌑', fontSize=18, dy=-20).encode(
#         x='date:T',
#         y='price:Q',
#         tooltip=alt.Tooltip(['phase:N', 'date:T','price:Q'])).interactive()

#     # Chart for all the Full Moons
#     chart_fullmoon = alt.Chart(df_viz_fullmoon).mark_text(text='🌕', fontSize=18, dy=20).encode(
#         x='date:T',
#         y='price:Q',
#         tooltip=alt.Tooltip(['phase:N', 'date:T','price:Q'])).interactive()
    
#     # Chart for first buy
#     chart_first_buy = alt.Chart(df_viz_first_fullmoon).mark_text(text='🏳️', dy=25, dx=13).encode(
#         x='date:T',
#         y='price:Q')
    
#     # Chart for last sell
#     chart_last_sell = alt.Chart(df_viz_last_newmoon).mark_text(text='🏁', dy=-15, dx=13).encode(
#         x='date:T',
#         y='price:Q')
    
#     # Layer the three charts into one
#     chart_layered = alt.layer(
#         chart_price,
#         chart_fullmoon,
#         chart_newmoon,
#         chart_first_buy,
#         chart_last_sell
#     ).configure_axis(labelFontSize=14, titleFontSize=14)

#     # Visualize the layered chart in the webapp
#     st.altair_chart(chart_layered, use_container_width=True)


# ### Version 2️⃣

# In[54]:


# Page setup
st.set_page_config(page_icon=':chart_with_upwards_trend:',
                   layout='wide', # either 'centered' or 'wide'
                   initial_sidebar_state='expanded')


# In[55]:


# Defining containers
header = st.container()
sidebar = st.container()
instructions = st.container()
inputs = st.container()
calculations = st.container()
visualization = st.container()


# In[56]:


# Header section
with header:
    st.title('To the Moon [Phase]! 🚀')
    st.subheader('Examining whether buying and selling ₿itcoin based on the moon phase* is a good investment strategy')
    st.write('*Buy on the full moon, sell on the new moon')


# In[57]:


# # Inputs section

# with inputs:
#     row1_1, row1_2, row1_3 = st.columns(3)
#     initial_investment = row1_1.number_input('Investment Amount', value=100.0, min_value=0.01)
#     invest_date = row1_2.date_input('Buy Date',  value=pd.to_datetime('2021-09-01'))
#     sell_date = row1_3.date_input('Sell Date', value=pd.to_datetime('today'), min_value=invest_date)

# # Create a date range section
#     invest_sell_date = st.sidebar.date_input(
#         'Buy Date',  
#         value=[pd.to_datetime('2021-07-01'), pd.to_datetime(df_all.max()['date'])],
#         min_value=pd.to_datetime(df_all.min()['date']),
#         max_value=pd.to_datetime(df_all.max()['date'])
#     )

#     if len(tuple(invest_sell_date)) == 1:
#         st.sidebar.error('Error: Please choose date range.')
#         invest_date, sell_date = pd.to_datetime('2021-07-01'), pd.to_datetime(df_all.max()['date'])
#     else:
#         invest_date, sell_date = invest_sell_date


# In[58]:


# Sidebar section

with sidebar:
    st.sidebar.header('Lunar investing 🌙')
    
    description_expander = st.sidebar.expander('About', expanded=False)
    with description_expander:
        st.write('''
                This app is a Bitcoin backtesting simulator for the moon phase trading strategy.

                The moon phase trading strategy is:
                - Buy on the Full Moon 🌕
                - Sell on the New Moon 🌑
                ###
                The simulator calculates the ROI of using this trading strategy between the chosen
                historical date range, reinvesting gains and losses between each moon cycle.
                The first buy and last sell are indicated by ❤️.

                ''')
    # Inputs section
    st.sidebar.write('''
        ###
        **Enter an investment amount and choose a historical date range:**
    ''')
    initial_investment = st.sidebar.number_input('Investment Amount', value=100.0, min_value=0.01)
    invest_date = st.sidebar.date_input('Buy Date',  value=pd.to_datetime('2021-09-01'))
    sell_date = st.sidebar.date_input('Sell Date', value=pd.to_datetime('today'), min_value=invest_date)


# In[59]:


# Error message on dates
def get_range(df, start, end):
    # Filtered DataFrame based on inputs
    return df[
        (pd.to_datetime(df['date'])>=pd.to_datetime(start))
        &
        (pd.to_datetime(df['date'])<=pd.to_datetime(end))][['date','price','phase']]

def correct_moon_order(df):
    moon_count = df[df['phase'].isin(['new','full'])]
    if moon_count.shape[0] > 2:
        return True
    elif moon_count.shape[0] == 2 and moon_count.iloc[0]['phase'] == 'full':
        return True
    elif moon_count.shape[0] == 2 and moon_count.iloc[0]['phase'] == 'new':
        return False
    else:
        return False

if not correct_moon_order(get_range(df_all, invest_date, sell_date)):
    st.sidebar.error('Error: Please choose a wider date range.')
    invest_date, sell_date = pd.to_datetime('2021-07-01'), pd.to_datetime(df_all.max()['date'])


# In[60]:


# Calculate returns column based on inputs
def calc_returns(df, invest_date, sell_date):
    df_returns = df.copy()
    df_returns = df_returns[
        (pd.to_datetime(df_returns['date_full']) >= pd.to_datetime(invest_date))
        &
        (pd.to_datetime(df_returns['date_new']) <= pd.to_datetime(sell_date))
    ]
    init_inv = float(initial_investment)
    for id_row, row in df_returns.iterrows():
        x = ((init_inv / row['price_full']) * row['price_new'])
        init_inv = x
        df_returns.at[id_row, 'returns'] = x
    return df_returns

df_returns = calc_returns(df_stream, invest_date, sell_date)


# In[61]:


# Filtered DataFrame based on inputs
df_viz = df_all[
        (pd.to_datetime(df_all['date'])>=pd.to_datetime(invest_date))
        &
        (pd.to_datetime(df_all['date'])<=pd.to_datetime(sell_date))][['date','price','phase']]

df_viz['date'] =  pd.to_datetime(df_viz['date'], infer_datetime_format=True).dt.tz_localize('EST')


# In[62]:


# Dynamic calculations section
with calculations:
    
    # Calculate ROI total
    return_on_investment = (df_returns[
                        (pd.to_datetime(df_returns['date_full']) >= pd.to_datetime(invest_date))
                        &
                        (pd.to_datetime(df_returns['date_new']) <= pd.to_datetime(sell_date))]['returns'].dropna().values[-1])
    roi_total = '${:,.2f}'.format(return_on_investment)
    
    # Calculate ROI percent
    percent_return = (((return_on_investment - initial_investment) / initial_investment))
    roi_percentage = '{:,.2%}'.format(percent_return)
    
    # Count number of Moon Cycles
    count_moon_cycle = df_viz[df_viz['phase'].isin(['new','full'])].shape[0]/2
    
    # Show calcuations of ROIs
    row2_1, row2_2, row2_3 = st.columns(3)

    with row2_1:
        st.metric(label='Current Value', value=roi_total)

    with row2_2:
        st.metric(label='Percentage Total Gain', value=roi_percentage)
        
    with row2_3:
        st.metric(label='Total Moon Cycles', value=count_moon_cycle)

# Dynamic viz section
with visualization:

    # Base line chart with Bitcoin prices
    price_min = df_viz['price'].min() - 3000
    price_max = df_viz['price'].max() + 300
    chart_price = alt.Chart(df_viz, height=500).mark_line().encode(
                x=alt.X('date:T', axis=alt.Axis(labelOverlap="parity", grid=False), title="Date"),
                y=alt.Y('price:Q', title="Price in USD", scale=alt.Scale(domain=[price_min,price_max])),
                color=alt.value('#FAA307'),
                tooltip=['date:T','price:Q']).interactive()

    # DataFrames based on df_viz separated by the phase of the moon
    df_viz_fullmoon = df_viz[df_viz['phase'] == 'full']
    df_viz_newmoon = df_viz[df_viz['phase'] == 'new']
    
    #DataFrames for first buy and last sell
    df_viz_first_fullmoon = pd.DataFrame(df_viz_fullmoon.iloc[0]).T
    df_viz_last_newmoon = pd.DataFrame(df_viz_newmoon.iloc[-1]).T

    # Chart for all the New Moons
    chart_newmoon = alt.Chart(df_viz_newmoon).mark_text(text='🌑', fontSize=18, dy=-20).encode(
        x='date:T',
        y='price:Q',
        tooltip=alt.Tooltip(['phase:N', 'date:T','price:Q'])).interactive()

    # Chart for all the Full Moons
    chart_fullmoon = alt.Chart(df_viz_fullmoon).mark_text(text='🌕', fontSize=18, dy=20).encode(
        x='date:T',
        y='price:Q',
        tooltip=alt.Tooltip(['phase:N', 'date:T','price:Q'])).interactive()
    
    # Chart for first buy
    chart_first_buy = alt.Chart(df_viz_first_fullmoon).mark_text(text='❤️', dy=25, dx=8).encode(
        x='date:T',
        y='price:Q')
    
    # Chart for last sell
    chart_last_sell = alt.Chart(df_viz_last_newmoon).mark_text(text='❤️', dy=-15, dx=8).encode(
        x='date:T',
        y='price:Q')
    
    # Layer the three charts into one
    chart_layered = alt.layer(
        chart_price,
        chart_fullmoon,
        chart_newmoon,
        chart_first_buy,
        chart_last_sell
    ).configure_axis(labelFontSize=14, titleFontSize=14)

    # Visualize the layered chart in the webapp
    st.altair_chart(chart_layered, use_container_width=True)


# In[63]:


df_viz_fullmoon.head()


# ## ✔️ Don't Need

# In[64]:


# chart_returns = alt.Chart(df_stream.dropna(), height=700).mark_line().encode(
#             x=alt.X('date_new', axis=alt.Axis(labelOverlap="greedy",grid=False)),
#             y=alt.Y('returns'),
#             tooltip=['date_new','returns']).interactive()
# st.altair_chart(chart_returns, use_container_width=True)


# In[65]:


#!jupyter nbconvert --to script Bitcoin_Moon_Phase.ipynb

# !streamlit run Bitcoin_Moon_Phase.py

# Go to C:\Users\Eden\Documents\Projects> and run:
# jupyter nbconvert --to script Bitcoin_Moon_Phase.ipynb ; streamlit run .\Bitcoin_Moon_Phase.py


# # 📥 Export Dataframe

# In[66]:


#df_all.to_csv('btc_moon_all.csv', index=False)


# In[67]:


#df_small.to_csv('btc_moon.csv', index=False)


# In[68]:


#df_new.to_csv('btc_new.csv', index=False)

