# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:37:49 2024

@author: 213
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from scipy.stats import pearsonr, ttest_ind
from prophet import Prophet
import matplotlib.font_manager as fm
import os


def load_font():
    try:
        path = os.path.join('fonts', 'H2MJRE.ttf')  # 폰트 파일 경로 설정
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Font file not found at {path}")
        return fm.FontProperties(fname=path, size=12)
    except Exception as e:
        st.error(f"Error loading font: {e}")
        return None

def calculate_statistics(cctv_data, crime_data):
    # 데이터 병합 및 '범죄율' 계산 (절도와 강간·추행만 포함)
    merged_data = pd.merge(cctv_data, crime_data, left_on='자치구', right_on='구분')
    merged_data['절도 강간·추행 범죄율'] = (merged_data['절도 발생'] + merged_data['강간·강제추행 발생']) / merged_data['인구수'] * 1000

    # CCTV 개수와 절도 강간·추행 범죄율 간의 상관 계수 및 p-값 계산
    cctv_count = merged_data['총 계']
    crime_rate = merged_data['절도 강간·추행 범죄율']

    corr, p_value = pearsonr(cctv_count, crime_rate)

    # t-검정 수행
    median_cctv = merged_data['총 계'].median()
    high_cctv_group = merged_data[merged_data['총 계'] > median_cctv]['절도 강간·추행 범죄율']
    low_cctv_group = merged_data[merged_data['총 계'] <= median_cctv]['절도 강간·추행 범죄율']

    t_stat, t_p_value = ttest_ind(high_cctv_group, low_cctv_group)

    return corr, p_value, t_stat, t_p_value, merged_data

def explore_data(cctv_data, crime_data):
    # 데이터 병합 및 '범죄율' 계산 (절도와 강간·추행만 포함)
    merged_data = pd.merge(cctv_data, crime_data, left_on='자치구', right_on='구분')
    merged_data['절도 강간·추행 범죄율'] = (merged_data['절도 발생'] + merged_data['강간·강제추행 발생']) / merged_data['인구수'] * 1000

    fontprop = load_font()
    if fontprop is None:
        fontprop = fm.FontProperties(size=12)  # 기본 폰트 사용

    # CCTV 개수와 절도 강간·추행 범죄율 간의 관계 시각화
    fig, ax = plt.subplots()
    sns.scatterplot(x=merged_data['총 계'], y=merged_data['절도 강간·추행 범죄율'], ax=ax)
    sns.regplot(x='총 계', y='절도 강간·추행 범죄율', data=merged_data, ax=ax, scatter_kws={'s':50}, line_kws={'color':'red'})
    ax.set_xlabel('CCTV 개수', fontproperties=fontprop)
    ax.set_ylabel('절도 및 강간·추행 범죄율 (per 1000명)', fontproperties=fontprop)
    ax.set_title('CCTV 개수와 절도 및 강간·추행 범죄율 간의 관계', fontproperties=fontprop)
    st.pyplot(fig)

    # 상관 계수와 p-값 출력
    corr, p_value, t_stat, t_p_value, _ = calculate_statistics(cctv_data, crime_data)
    st.markdown(f"### 상관 계수: {corr}")
    st.markdown(f"### p-값: {p_value}")
    if p_value < 0.05:
        st.markdown("상관 계수가 통계적으로 유의미합니다.")
    else:
        st.markdown("상관 계수가 통계적으로 유의미하지 않습니다.")

def explore_other_variables(cctv_data, crime_data):
    # 데이터 병합 및 '범죄율' 계산 (모든 범죄 유형 포함)
    merged_data = pd.merge(cctv_data, crime_data, left_on='자치구', right_on='구분')
    merged_data['전체 범죄율'] = (merged_data['살인 발생'] + merged_data['강도 발생'] +
                             merged_data['강간·강제추행 발생'] + merged_data['절도 발생'] +
                             merged_data['폭력 발생']) / merged_data['인구수'] * 1000

    fontprop = load_font()
    if fontprop is None:
        fontprop = fm.FontProperties(size=12)  # 기본 폰트 사용

    # 인구수와 전체 범죄율 간의 관계 시각화
    fig, ax = plt.subplots()
    sns.scatterplot(x=merged_data['인구수'], y=merged_data['전체 범죄율'], ax=ax)
    sns.regplot(x='인구수', y='전체 범죄율', data=merged_data, ax=ax, scatter_kws={'s':50}, line_kws={'color':'red'})
    ax.set_xlabel('인구수', fontproperties=fontprop)
    ax.set_ylabel('전체 범죄율 (per 1000명)', fontproperties=fontprop)
    ax.set_title('인구수와 전체 범죄율 간의 관계', fontproperties=fontprop)
    st.pyplot(fig)

    # 상관 계수와 p-값 출력
    corr, p_value = pearsonr(merged_data['인구수'], merged_data['전체 범죄율'])
    st.markdown(f"### 상관 계수: {corr}")
    st.markdown(f"### p-값: {p_value}")
    if p_value < 0.05:
        st.markdown("상관 계수가 통계적으로 유의미합니다.")
    else:
        st.markdown("상관 계수가 통계적으로 유의미하지 않습니다.")

def run_prophet_forecast(cctv_data, crime_data):
    st.markdown("### Prophet을 이용한 범죄율 예측")

    # cctv_data와 crime_data 병합
    merged_data = pd.merge(cctv_data, crime_data, left_on='자치구', right_on='구분')

    # 발생 년도를 2022년으로 가정하여 날짜 데이터 생성
    merged_data['ds'] = pd.date_range(start='2022-01-01', periods=len(merged_data), freq='M')
    merged_data['y'] = (merged_data['절도 발생'] + merged_data['강간·강제추행 발생']) / merged_data['인구수'] * 1000

    # Prophet 모델 학습
    model = Prophet()
    model.fit(merged_data[['ds', 'y']])

    # 미래 데이터프레임 생성 및 예측
    future = model.make_future_dataframe(periods=12, freq='M')
    forecast = model.predict(future)

    # 예측 결과 시각화
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)
def home():
    st.markdown("### 기계학습 예측 개요 \n"
                "기계학습 예측 페이지 입니다."
                "사용한 알고리즘 : Meta의 Prophet")


def run_ml_home(total_df):
    selected = option_menu(None, ['Home', 'CCTV와 범죄율 관계', '다른 변수 탐색', 'Prophet 예측 모델', '보고서'],
                           icons=['house', 'search', 'bar-chart', 'calendar', 'file-spreadsheet'],
                           menu_icon='cast', default_index=0, orientation='horizontal',
                           styles={
                               'container': {
                                   'padding': '0!important',
                                   'background-color': '#808080'
                               },
                               'icon': {
                                   'color': 'orange',
                                   'font-size': '25px'
                               },
                               'nav-link': {
                                   'font-size': '15px',
                                   'text-align': 'left',
                                   'margin': '0px',
                                   '--hover-color': '#eee'
                               },
                               'nav-link-selected': {
                                   'background-color': 'green'
                               }
                           })
    
    if selected == 'Home':
        home()
    elif selected == 'CCTV와 범죄율 관계':
        explore_data(*total_df)
    elif selected == '다른 변수 탐색':
        explore_other_variables(*total_df)
    elif selected == 'Prophet 예측 모델':
        run_prophet_forecast(*total_df)  # cctv_data와 crime_data 모두 전달
    elif selected == '보고서':
        generate_report(*total_df)
    else:
        st.warning(f'Unknown menu selection: {selected}')

def generate_report(cctv_data, crime_data):
    st.markdown("# 분석 보고서")
    
    st.markdown("## CCTV와 범죄율 관계")
    st.markdown("""
    본 분석에서는 자치구별 CCTV 개수와 범죄율(절도 및 강간·추행) 간의 상관 관계를 조사했습니다.
    분석 결과, CCTV 개수와 범죄율 간의 상관 계수는 -0.103으로, 매우 낮은 음의 상관 관계를 보였습니다.
    또한, p-값이 0.633으로, 이 상관 관계는 통계적으로 유의미하지 않았습니다.
    따라서, CCTV 개수와 범죄율 간에 명확한 상관 관계가 없음을 알 수 있습니다.
    """)

    st.markdown("## 인구수와 범죄율 간의 관계")
    st.markdown("""
    추가적으로 인구수와 범죄율 간의 관계를 탐색했습니다.
    분석 결과, 인구수와 절도 및 강간·추행 범죄율 간의 상관 계수는 -0.462로, 중간 정도의 음의 상관 관계를 보였습니다.
    또한, p-값이 0.023으로, 이 상관 관계는 통계적으로 유의미하였습니다.
    이는 인구수가 증가할수록 절도 및 강간·추행 범죄율이 감소하는 경향이 있음을 의미합니다.
    """)

    st.markdown("## 왜 인구가 많은 지역에서 범죄율이 적어지는가")
    st.markdown("""
   
    1. **경찰력 및 보안 자원 배치**
       - 자원 배분: 인구가 많은 지역일수록 경찰력 및 보안 자원 배치가 더 집중될 가능성이 있습니다. 이에 따라 인구가 적은 지역에서는 상대적으로 보안 자원이 부족할 수 있으며, 이로 인해 범죄 발생을 억제하는 능력이 떨어질 수 있습니다.

    2. **사회적 연결망**
       - 사회적 연결망: 인구가 적은 지역에서는 주민들 간의 사회적 연결망이 약할 수 있습니다. 이는 범죄 예방 및 대응에 부정적인 영향을 미칠 수 있습니다. 반면, 인구가 많은 지역에서는 주민들 간의 상호작용이 많아 범죄 발생 시 협력하여 대처하는 경우가 많을 수 있습니다.

    3. **경제적 요인**
       - 경제적 불균형: 인구가 적은 지역에서는 경제적 불균형이 클 수 있습니다. 이러한 지역에서는 실업률이 높거나 경제적으로 어려운 상황에 처한 주민이 많을 수 있으며, 이는 범죄 발생의 요인이 될 수 있습니다.

    4. **환경적 요인**
       - 주거 환경: 인구가 적은 지역일수록 주거 환경이 열악할 가능성이 있습니다. 이는 범죄 발생의 요인으로 작용할 수 있습니다. 또한, 이러한 지역에서는 공공시설이 부족하여 범죄 예방 및 대응이 어려울 수 있습니다.

    5. **범죄 신고율**
       - 범죄 신고율: 인구가 적은 지역에서는 범죄가 발생해도 신고율이 낮을 수 있습니다. 이는 범죄율을 과소평가하게 만들 수 있으며, 실제 범죄 발생률은 통계적으로 나타난 것보다 높을 수 있습니다.

    ### 데이터 분석 관점에서:
    - **데이터 해석**: 인구수가 적은 지역에서의 높은 범죄율은 단순히 인구 대비 범죄 발생 건수가 많다는 것을 의미합니다. 이는 절대적인 범죄 발생 건수는 적더라도 인구가 적기 때문에 범죄율(인구 1000명당 범죄 건수)이 상대적으로 높게 나타나는 경우입니다.
    """)


    st.markdown("## 결론")
    st.markdown("""
    이번 분석을 통해 CCTV 개수와 범죄율 간에 명확한 상관 관계가 없음을 확인했습니다. 
    이는 단순히 CCTV 개수만으로는 범죄율을 예측하거나 감소시키는 데 한계가 있음을 시사합니다.
    
    그러나, 인구수와 범죄율 간에는 통계적으로 유의미한 음의 상관 관계가 있음을 확인했습니다. 
    이는 인구수가 증가할수록 절도 및 강간·추행 범죄율이 감소하는 경향이 있음을 의미합니다.
    
    추가적인 변수(예: 경제적 요인, 사회적 요인)를 고려한 종합적인 접근이 필요합니다.

    """)

