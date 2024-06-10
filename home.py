# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:05:39 2024

@author: 109-2
"""

import streamlit as st

def home():
    st.markdown("""
    
    ## 소개 및 목적
    - CCTV가 범죄 예방에 실질적인 효과가 있는지 분석
    - 다른 변수들과의 관계 이해

    ## 데이터 소개
    - 서울시 자치구별 CCTV 설치 현황 데이터
    - 서울시 자치구별 범죄 발생 현황 데이터

    ## 분석 방법론
    1. CCTV 개수와 범죄율 간의 상관 관계 분석
    2. 인구수와 범죄율 간의 상관 관계 분석
    3. Prophet 모델을 사용한 범죄율 예측
    4. 다른 변수와의 관계 탐색
    """)
