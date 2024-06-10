# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:29:05 2024

@author: 213
"""

import streamlit as st
from streamlit_option_menu import option_menu
from viz import showViz
from statistic import showStat
from map import showMap

def home():
    st.markdown("### Visualization 개요 \n"
                "자치구별 인구수와 CCTV 설치 대수의 관계 \n")
    st.markdown("### Statistics 개요 \n")
    st.markdown("### Map 개요 \n")

def run_eda_home(total_df):
    st.markdown("### 탐색적 자료 분석 \n")
    
    selected = option_menu(None, ['Home', 'Visualization', 'Statistics', 'Map'],
                           icons=['house', 'bar-chart', 'file-spreadsheet', 'map'],
                           menu_icon='cast', default_index=0, orientation='horizontal',
                           styles={
                               'container': {
                                               'padding': '0!important',
                                               'background-color': '#808080'},
                               'icon':      {
                                               'color': 'orange',
                                               'font-size': '25px'},
                               'nav-link':  {
                                               'font-size': '15px',
                                               'text-align': 'left',
                                               'margin': '0px',
                                               '--hover-color': '#eee'},
                               'nav-link-selected': {
                                               'background-color': 'green'}
                               })
    
    st.write(f'Selected menu: {selected}')  # 디버깅을 위해 선택된 메뉴 값을 출력
    
    if selected == 'Home':
        home()
    elif selected == 'Visualization':
        showViz(total_df)
    elif selected == 'Statistics':
        showStat(total_df)
    elif selected == 'Map':
        showMap(*total_df)  # total_df를 개별 인수로 분리하여 전달
    else:
        st.warning(f'Unknown menu selection: {selected}')
