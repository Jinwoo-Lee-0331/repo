import pymssql
import pyodbc
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np
import datetime
from streamlit_tree_select import tree_select
import matplotlib.dates as mdates
# from matplotlib import font_manager, rc
# font_path = "C:/Windows/Fonts/batang.ttf"
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)

@st.cache_resource
def init_connection():
    return pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};SERVER="
        + st.secrets["server"]
        + ";DATABASE="
        + st.secrets["database"]
        + ";UID="
        + st.secrets["username"]
        + ";PWD="
        + st.secrets["password"]
    )
conn = init_connection()

@st.cache_data(ttl=600)
def runqry(date_i,loc_i):
    # conn = pymssql.connect(server='192.168.210.14', user='hmcportal', password='qwer1234!', database='hmcportal',
    #                        charset='utf8')
    cursor = conn.cursor()
    cursor.execute("SELECT Time, TAG, Value FROM [HMCPISVR].[piarchive]..[PICOMP2] WHERE Time > '" + date_i.strftime(
        "%Y-%m-%d") + " 07:00:00' and Time < '" + date_i.strftime("%Y-%m-%d") + " 21:00:00' and tag like '%" + loc_i + "%' order by Time asc")
    x = pd.DataFrame(cursor, columns=["Time", "TAG", "VAL"])
    x = x[x["TAG"].str.contains(r'(OPC UA.(\w+).2.Tags.\w+.\w+.(\w+)-(\w-\d\w)-(.+))')]
    y = pd.concat([pd.to_datetime(x["Time"]), x["TAG"].str.extract(r'OPC UA.(\w+).2.Tags.\w+.\w+.(\w+)-(\w-\d\w)-(.+)'), x["VAL"].astype('float')],
                  axis=1)
    y.columns = ["Time", "Location", "Attribute", "Serial", "TAG", "Value"]
    # 연결 끊기
    conn.close()
    return y

if 'key' not in st.session_state:
    st.session_state.key = False

with st.sidebar:
    date_i = st.sidebar.date_input(label="Select Time")
    loc_i = st.sidebar.selectbox("H2 Refueling Station",['강서충전소', '괴산개미충전소', '서산수소충전소', '성남E1수소충전소', '안성아트센터수소충전소', '양산차고지수소충전소', '전주삼천수소충전소', '제천삼보수소충전소', '청주문의수소충전소', '평택버스충전소', '화성체육시설충전소'],index=2)

    nodes=[{"label": "Query 버튼을 클릭하세요", "value": 0}]

    if st.button(label="Query"):
        st.session_state.key = True

    if st.session_state.key:
        y = runqry(date_i, loc_i)
        z = y.loc[y["Attribute"] == "TAG", "TAG"]
        z = pd.concat([z, z], axis=1)
        z.drop_duplicates(inplace=True)
        z.columns = ["label", "value"]
        z = z.to_dict('records')

        nodes = [{"label": "TAG", "value": "TAG", "children": z}]
        return_select = tree_select(nodes)
        # st.write(return_select)

if st.sidebar.button(label="Plot"):
    # st.write('|'.join(return_select["checked"]))
    y2=y.loc[y["TAG"].str.contains('|'.join(return_select["checked"])), ["Time", "TAG", "Value"]]
    # y2["Time"].astype("")
    # y2["Time"]=y2["Time"].strftime("%H:%M")
    grouped_y = y2.groupby("TAG")
    plt.rc('font', family='Malgun Gothic')
    plt.xticks(np.arange(min(y["Time"]), max(y["Time"]),datetime.timedelta(hours=2)))
    plt.yticks(np.arange(min(y["Value"]), min(y["Value"]),0.2*(max(y["Value"])-min(y["Value"]))))
    fig, ax = plt.subplots()
    z=[""]
    for group_name, group_data in grouped_y:
        ax.plot(group_data["Time"],group_data["Value"])
        z.append(group_name)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
    z.remove("")
    ax.legend(z)
    # y2.set_index('Time')
    # st.line_chart(y2.groupby('TAG')['Value'])
    yy=y
    yy.set_index("Time",drop=True,inplace=True)
    st.pyplot(fig)
    st.table(yy[yy["Attribute"]=="ALM"])



