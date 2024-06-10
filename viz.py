# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:57:57 2024

@author: 213
"""

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import streamlit as st
import matplotlib.font_manager as fm
import os
import seaborn as sns

def load_font():
    try:
        path = os.path.join('fonts', 'H2MJRE.ttf')
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Font file not found at {path}")
        return fm.FontProperties(fname=path, size=12)
    except Exception as e:
        st.error(f"Error loading font: {e}")
        return None

# colormap을 사용자 정의로 세팅
color_step = ['#00FA9A', '#7FFFD4', '#00BFFF', '#87CEFA', '#F0E68C', '#FFFF00']
my_cmap = ListedColormap(color_step)

def draw_scatter_plot(seoul_data, df_sort_f, df_sort_t, fx, f1):
    fontprop = load_font()
    if fontprop is None:
        fontprop = fm.FontProperties(size=12)  # 기본 폰트 사용
    
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(seoul_data['인구수'], seoul_data['총 계'], s=50, c=seoul_data['오차'], cmap=my_cmap)
    plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')

    for n in range(5):
        # 상위 5개
        plt.text(df_sort_f['인구수'].iloc[n] * 1.02,  # x좌표
                 df_sort_f['총 계'].iloc[n] * 0.979,  # y좌표
                 df_sort_f['자치구'].iloc[n],  # title
                 fontsize=10, fontproperties=fontprop)
        # 하위 5개
        plt.text(df_sort_t['인구수'].iloc[n] * 1.02,  # x좌표
                 df_sort_t['총 계'].iloc[n] * 0.98,  # y좌표
                 df_sort_t['자치구'].iloc[n],  # title
                 fontsize=10, fontproperties=fontprop)

    plt.xlabel('인구수', fontproperties=fontprop)
    plt.ylabel('CCTV', fontproperties=fontprop)
    plt.grid()
    plt.colorbar(scatter, label='오차')

    # x축 레이블 회전만 적용
    plt.xticks(rotation=90, fontproperties=fontprop)
    st.pyplot(plt)

def show_additional_visualizations(cctv_data, crime_data):
    
    
    fontprop = load_font()
    if fontprop is None:
        fontprop = fm.FontProperties(size=12)  # 기본 폰트 사용
    
    # CCTV 개수와 범죄 발생 건수 간의 관계
    fig, ax = plt.subplots()
    total_crime = crime_data['살인 발생'] + crime_data['강도 발생'] + crime_data['강간·강제추행 발생'] + crime_data['절도 발생'] + crime_data['폭력 발생']
    sns.scatterplot(x=cctv_data['총 계'], y=total_crime, ax=ax)
    ax.set_xlabel('CCTV 개수', fontproperties=fontprop)
    ax.set_ylabel('범죄 발생 건수', fontproperties=fontprop)
    ax.set_title('CCTV 개수와 범죄 발생 건수 간의 관계', fontproperties=fontprop)
    st.pyplot(fig)
    
    # 자치구별 CCTV 개수 분포
    fig, ax = plt.subplots()
    sns.histplot(cctv_data['총 계'], kde=True, ax=ax)
    ax.set_xlabel('CCTV 개수', fontproperties=fontprop)
    ax.set_ylabel('빈도', fontproperties=fontprop)
    ax.set_title('자치구별 CCTV 개수 분포', fontproperties=fontprop)
    st.pyplot(fig)
    
    # 자치구별 범죄 발생 유형 분포
    fig, ax = plt.subplots(figsize=(12, 6))
    crime_data_melted = crime_data.melt(id_vars='구분', value_vars=['살인 발생', '강도 발생', '강간·강제추행 발생', '절도 발생', '폭력 발생'])
    sns.barplot(x='variable', y='value', hue='구분', data=crime_data_melted, ax=ax)
    ax.set_xlabel('범죄 유형', fontproperties=fontprop)
    ax.set_ylabel('발생 건수', fontproperties=fontprop)
    ax.set_title('자치구별 범죄 발생 유형 분포', fontproperties=fontprop)
    st.pyplot(fig)


def showViz(total_df):
    fontprop = load_font()
    if fontprop is None:
        fontprop = fm.FontProperties(size=12)  # 기본 폰트 사용
    
    seoul_cctv, seoul_crime = total_df

    # 데이터 전처리 및 정렬
    df_sort_f = seoul_cctv.sort_values(by='총 계', ascending=False).head(5)
    df_sort_t = seoul_cctv.sort_values(by='총 계', ascending=True).head(5)

    # 회귀선 계산
    fp1 = np.polyfit(seoul_cctv['인구수'], seoul_cctv['총 계'], 1)
    f1 = np.poly1d(fp1)

    fx = np.linspace(100000, 700000, 100)

    st.markdown("### 자치구별 인구수와 CCTV 설치 대수의 관계")


    draw_scatter_plot(seoul_cctv, df_sort_f, df_sort_t, fx, f1)

    # 추가 시각화 표시
    show_additional_visualizations(seoul_cctv, seoul_crime)



