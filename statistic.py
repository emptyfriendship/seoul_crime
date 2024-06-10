# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:25:28 2024

@author: 213
"""
import pandas as pd
from scipy.stats import pearsonr, ttest_ind
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import os

def load_font():
    try:
        path = os.path.join('fonts', 'H2MJRE.ttf')
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Font file not found at {path}")
        return fm.FontProperties(fname=path, size=12)
    except Exception as e:
        st.error(f"Error loading font: {e}")
        return None

def set_korean_font():
    font_path = os.path.join('fonts', 'H2MJRE.ttf')
    if os.path.isfile(font_path):
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rcParams['font.family'] = font_name
        plt.rcParams['axes.unicode_minus'] = False  # 마이너스 폰트 깨짐 방지
    else:
        st.error(f"Font file not found at {font_path}")

def calculate_statistics(cctv_data, crime_data):
    fontprop = load_font()
    if fontprop is None:
        fontprop = fm.FontProperties(size=12)  # 기본 폰트 사용
    
    # cctv_data의 '인구수' 열을 사용하여 범죄율 계산
    crime_data = pd.merge(crime_data, cctv_data[['자치구', '인구수']], left_on='구분', right_on='자치구')
    crime_data['범죄율'] = (crime_data[['살인 발생', '강도 발생', '강간·강제추행 발생', '절도 발생', '폭력 발생']].sum(axis=1) /
                         crime_data['인구수']) * 1000

    # 검거율 계산 (검거 수 / 발생 수 * 100)
    crime_data['검거율'] = (crime_data[['살인 검거', '강도 검거', '강간·강제추행 검거', '절도 검거', '폭력 검거']].sum(axis=1) /
                         crime_data[['살인 발생', '강도 발생', '강간·강제추행 발생', '절도 발생', '폭력 발생']].sum(axis=1)) * 100

    # 데이터 병합
    merged_data = pd.merge(cctv_data, crime_data, left_on='자치구', right_on='구분')

    # CCTV 개수와 범죄율 간의 상관 계수 및 p-값 계산
    cctv_count = merged_data['총 계']
    crime_rate = merged_data['범죄율']

    corr_crime, p_value_crime = pearsonr(cctv_count, crime_rate)

    # CCTV 개수와 검거율 간의 상관 계수 및 p-값 계산
    arrest_rate = merged_data['검거율']
    corr_arrest, p_value_arrest = pearsonr(cctv_count, arrest_rate)

    # t-검정 수행
    median_cctv = merged_data['총 계'].median()
    high_cctv_group = merged_data[merged_data['총 계'] > median_cctv]['범죄율']
    low_cctv_group = merged_data[merged_data['총 계'] <= median_cctv]['범죄율']

    t_stat, t_p_value = ttest_ind(high_cctv_group, low_cctv_group)

    return corr_crime, p_value_crime, t_stat, t_p_value, corr_arrest, p_value_arrest, merged_data

def showStat(total_df):
    set_korean_font()  # 한글 폰트 설정

    cctv_data, crime_data = total_df

    corr_crime, p_value_crime, t_stat, t_p_value, corr_arrest, p_value_arrest, merged_data = calculate_statistics(cctv_data, crime_data)

    st.markdown("### CCTV 개수와 범죄율 및 검거율 간의 상관 관계 및 t-검정")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x='총 계', y='범죄율', data=merged_data, ax=ax)
    ax.set_xlabel('CCTV 개수')
    ax.set_ylabel('범죄율')
    ax.set_title('CCTV 개수와 범죄율 간의 관계')
    st.pyplot(fig)

    st.write(f'CCTV 개수와 범죄율 간의 상관 계수: {corr_crime:.4f}')
    st.write(f'p-값: {p_value_crime:.4f}')

    if p_value_crime > 0.05:
        st.markdown('상관 계수의 p-값이 0.05보다 크므로, CCTV 개수와 범죄율 간의 상관 관계는 통계적으로 유의미하지 않습니다.')
    else:
        st.markdown('상관 계수의 p-값이 0.05보다 작으므로, CCTV 개수와 범죄율 간의 상관 관계는 통계적으로 유의미합니다.')

    st.write(f'고 CCTV 그룹과 저 CCTV 그룹 간의 범죄율 t-검정 통계량: {t_stat:.4f}')
    st.write(f'p-값: {t_p_value:.4f}')

    if t_p_value > 0.05:
        st.markdown('t-검정의 p-값이 0.05보다 크므로, 고 CCTV 그룹과 저 CCTV 그룹 간의 범죄율 차이는 통계적으로 유의미하지 않습니다.')
    else:
        st.markdown('t-검정의 p-값이 0.05보다 작으므로, 고 CCTV 그룹과 저 CCTV 그룹 간의 범죄율 차이는 통계적으로 유의미합니다.')

    if corr_crime > 0:
        st.markdown('상관 계수가 양수이므로, CCTV 개수가 증가할수록 범죄율이 증가하는 경향이 있습니다.')
    else:
        st.markdown('상관 계수가 음수이므로, CCTV 개수가 증가할수록 범죄율이 감소하는 경향이 있습니다.')

    # CCTV 개수와 검거율 간의 관계 그래프
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x='총 계', y='검거율', data=merged_data, ax=ax)
    ax.set_xlabel('CCTV 개수')
    ax.set_ylabel('검거율')
    ax.set_title('CCTV 개수와 검거율 간의 관계')
    st.pyplot(fig)

    st.write(f'CCTV 개수와 검거율 간의 상관 계수: {corr_arrest:.4f}')
    st.write(f'p-값: {p_value_arrest:.4f}')

    if p_value_arrest > 0.05:
        st.markdown('상관 계수의 p-값이 0.05보다 크므로, CCTV 개수와 검거율 간의 상관 관계는 통계적으로 유의미하지 않습니다.')
    else:
        st.markdown('상관 계수의 p-값이 0.05보다 작으므로, CCTV 개수와 검거율 간의 상관 관계는 통계적으로 유의미합니다.')

    if corr_arrest > 0:
        st.markdown('상관 계수가 양수이므로, CCTV 개수가 증가할수록 검거율이 증가하는 경향이 있습니다.')
    else:
        st.markdown('상관 계수가 음수이므로, CCTV 개수가 증가할수록 검거율이 감소하는 경향이 있습니다.')

