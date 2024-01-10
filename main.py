import pymysql
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime, timedelta
from streamlit_tree_select import tree_select
import matplotlib.dates as mdates
from sshtunnel import SSHTunnelForwarder

st.set_page_config(
		page_title= "H2 Data Center", # String or None. Strings get appended with "• Streamlit".
		 layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
		 #initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
		 #page_icon=None,  # String, anything supported by st.image, or None.
)

@st.cache_resource
def tunnel_connection():
    return SSHTunnelForwarder((st.secrets["server"], st.secrets["port"]),
                       ssh_username=st.secrets["ssh_username"],
                       ssh_password=st.secrets["ssh_password"],
                       remote_bind_address=(st.secrets["remote_bind_address"], st.secrets["remote_bind_port"]))
tunnel=tunnel_connection()
tunnel.start()


# @st.cache_resource
# def init_connection():
#     return pymysql.connect(
#             host='127.0.0.1', #(local_host)
#             user='ns0331',
#             passwd='wlsdn1469!!',
#             db='hmcportal',
#             charset='utf8',
#             port=tunnel.local_bind_port)
# conn = init_connection()

st.write(tunnel.local_bind_port)

@st.cache_resource
def init_connection():
    return st.experimental_connection('hmc_db',type="sql")
conn=init_connection()

hrs=pd.read_csv('./data/hrs.csv',header=None)
# hrs=pd.read_csv('C:\\Users\\researcher\\Desktop\\hrs.csv',header=None)
hrs.columns=['Location','Address']
hrs['Last Connected Time']='Disconnected'

# @st.cache_data(ttl=600)
def streamlit_init(hrs):
    for idx,i in enumerate(hrs['Location']):
        query1 = (f"SELECT Time ,Tag ,Value FROM RawData"
                  f" where Tag like '%{i}%온도%' order by Time desc LIMIT 1;")
        # qry= pd.read_sql(query1, conn)
        qry=conn.query(query1)
        qry=pd.DataFrame(qry)
        try:
            hrs.loc[idx,'Last Connected Time']=qry.loc[0,'Time'].strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            print(e)
    return hrs

@st.cache_data(ttl=600)
def runqry(date_i,loc_i):
    query = "SELECT Time, Tag, Value FROM RawData where Time > '" + date_i.strftime("%Y-%m-%d") + " 07:00:00' and Time < '" + \
            date_i.strftime("%Y-%m-%d") + " 21:00:00' and tag like '%" + loc_i + "%' order by Time asc;"
    x = conn.query(query)
    x = pd.DataFrame(x)
    return x

if 'key' not in st.session_state:
    st.session_state.key = False
if 'plot' not in st.session_state:
    st.session_state['plot'] = False
if 'update' not in st.session_state:
    st.session_state['update'] = True

col1, col2 = st.columns(2)
with col1:
    hometab, tab2  = st.tabs(["📋 Board", "📊 Operation"])

with hometab:
    hrs_update = pd.read_csv('./data/hrs_update.csv')
    hometab.table(hrs_update[['Location','Last Connected Time','Address']])
    if st.button(label="Update", use_container_width=True):
        # hrs = streamlit_init(hrs)
        streamlit_init(hrs).to_csv('./data/hrs_update.csv', index=False, encoding='utf-8')
        # hrs_update = pd.read_csv('./data/hrs_update.csv')
        # st.session_state['update'] = True

    # if st.session_state['update']:
    #     hrs_update = pd.read_csv('./data/hrs_update.csv')
        # hometab.table(hrs_update[['Location','Last Connected Time','Address']])


with col2:
    tab1, tab3  = st.tabs(["📈 Chart","❗ Alarm"])

with st.sidebar:
    st.sidebar.markdown("<h1 style='text-align: center;"
                        " color: #235191;'>H2 Data Center</h1>", unsafe_allow_html=True)
    date_i = st.sidebar.date_input(label="Select Time")
    loc_i = st.sidebar.selectbox("H2 Refueling Station",list(hrs.iloc[:,0]),index=2)

    nodes=[{"label": "Query 버튼을 클릭하세요", "value": 0}]

    if st.button(label="Query", use_container_width=True):
        # st.session_state.key = False
        runqry(date_i, loc_i).to_csv('./data/loc_i.csv', index=False, encoding='utf-8')
        st.session_state.key = True
        st.session_state['plot'] = True
    st.markdown("---")

    if st.session_state.key:
        x= pd.read_csv('./data/loc_i.csv')
        y = pd.concat([x["Time"], x["Tag"].str.extract(r'(\w+)-(\w+)-(\w-\w+)-(.+)'),
                       x["Value"]], axis=1)
        y.columns = ["Time", "Location", "Attribute", "Serial", "Tag", "Value"]
        x['Legend'] = y['Tag']
        z = y["Tag"]
        z = pd.concat([z, z], axis=1)
        z.drop_duplicates(inplace=True)
        z.columns = ["label", "value"]
        z = z.to_dict('records')

        srl = y['Serial'].drop_duplicates()
        srl_trd = []
        for idx_i,i in enumerate(srl):
            atr = y.loc[y['Serial'] == i, 'Attribute'].drop_duplicates()
            atr_trd = []
            for idx_j,j in enumerate(atr):
                tag = y.loc[(y['Serial'] == i) & (y['Attribute'] == j), 'Tag'].drop_duplicates()
                tag_trd = []
                for k in tag:
                    tag_trd.append({'label': k, 'value': j+'-'+i+'-'+k})
                atr_trd.append({'label': j, 'value': j+'-'+i, 'children': tag_trd})
            srl_trd.append({'label': i, 'value': i, 'children': atr_trd})
        root=[{'label': loc_i, 'value': loc_i, 'children': srl_trd}]
        return_select = tree_select(root, checked=[root[0]['children'][0]['children'][0]['children'][0]['value']],
                                    expanded=[root[0]['value'], root[0]['children'][0]['value'],
                                              root[0]['children'][0]['children'][0]['value']])
        opr=y[(y["Attribute"]=='STS')]
        opr['Value']=opr['Value'].astype(bool)
        opr.set_index("Time",drop=True,inplace=True)
        alm=y[(y["Attribute"]=='ALM')]
        alm['Value']=alm['Value'].astype(bool)
        alm.set_index("Time",drop=True,inplace=True)
        tab2.dataframe(opr)
        tab3.dataframe(alm)

if st.session_state['plot']:
    if return_select["checked"]:
        y2=x.loc[x["Tag"].str.contains('|'.join(return_select["checked"])), ["Time", "Tag", "Value","Legend"]]
        try:
            tab1.line_chart(y2,x='Time',y='Value',color='Legend')
        except Exception as e:
            tab1.line_chart(y2,x='Time',y='Value')
