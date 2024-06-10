# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 15:27:57 2024

@author: 109-2
"""
import streamlit as st
from streamlit_option_menu import option_menu
from home import home
from eda import run_eda_home
from m1 import run_ml_home
from utils import load_data

def main():
    st.title("서울시 자치구별 CCTV와 범죄율 분석")
    total_df = load_data()
    
    with st.sidebar:
        selected = option_menu('데시보드 메뉴', ['홈', '탐색적 자료분석', '기계학습 예측'],
                              icons=['house', 'file-bar-graph', 'robot'], menu_icon='cast', default_index=0)
        
    if selected == '홈':
        home()
    elif selected == '탐색적 자료분석':
        run_eda_home(total_df)
    elif selected == '기계학습 예측':
        run_ml_home(total_df)
    else:
        st.warning('메뉴 선택 오류')

if __name__ == "__main__":
    main()
