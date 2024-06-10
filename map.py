# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 17:34:15 2024

@author: 109-2
"""

import pandas as pd
import geopandas as gpd
import streamlit as st
import matplotlib.pyplot as plt
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
        return fm.FontProperties(size=12)

def mapMatplotlib(merge_df):
    fontprop = load_font()

    # 서브플롯 생성
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 데이터 플로팅
    merge_df.plot(ax=ax, column='범죄율', cmap='Reds', legend=True, alpha=0.9, edgecolor='gray')

    # 컬러바
    patch_col = ax.collections[0]
    cb = fig.colorbar(patch_col, ax=ax, shrink=0.5)

    # 지도 주석
    for i, row in merge_df.iterrows():
        ax.annotate(row['SIG_KOR_NM'], xy=(row['lon'], row['lat']), xytext=(-7, 2),
                    textcoords='offset points', fontsize=8, color='black', fontproperties=fontprop)

    # 제목 설정
    ax.set_title('자치구별 범죄율 (범죄 발생 수 / 인구 수 * 1000)', fontproperties=fontprop)
    
    # 축 제거
    ax.set_axis_off()
    
    # 플롯 표시
    st.pyplot(fig)

def showMap(cctv_data, crime_data):
    st.markdown("### 자치구별 범죄율 지도")

    # 지리 데이터 로드 및 준비
    seoul_gpd = gpd.read_file("seoul_sig.geojson.gpkg")
    seoul_gpd = seoul_gpd.set_crs(epsg='5178', allow_override=True)
    seoul_gpd['center_point'] = seoul_gpd['geometry'].centroid
    seoul_gpd['geometry'] = seoul_gpd['geometry'].to_crs(epsg=4326)
    seoul_gpd['center_point'] = seoul_gpd['center_point'].to_crs(epsg=4326)
    seoul_gpd['lon'] = seoul_gpd['center_point'].map(lambda x: x.xy[0][0])
    seoul_gpd['lat'] = seoul_gpd['center_point'].map(lambda x: x.xy[1][0])
    seoul_gpd = seoul_gpd.rename(columns={"SIG_CD": "SGG_CD"})
    
    # 범죄 데이터 준비
    crime_data = pd.merge(crime_data, cctv_data[['자치구', '인구수']], left_on='구분', right_on='자치구')
    crime_data['범죄율'] = (crime_data[['살인 발생', '강도 발생', '강간·강제추행 발생', '절도 발생', '폭력 발생']].sum(axis=1) /
                           crime_data['인구수']) * 1000

    # 범죄 데이터 요약
    summary_df = crime_data.groupby(['구분'])['범죄율'].mean().reset_index()
    summary_df = summary_df.rename(columns={'구분': 'SIG_KOR_NM'})
    
    # 지리 데이터와 범죄 데이터 병합
    merge_df = seoul_gpd.merge(summary_df, on='SIG_KOR_NM')

    # 병합 데이터 샘플 표시
    st.write(merge_df[['SIG_KOR_NM', 'geometry', '범죄율']].head(3))
    
    # 지도 생성 및 표시
    mapMatplotlib(merge_df)

if __name__ == "__main__":
    cctv_data = pd.read_csv('01_Seoul_CCTV_Data.csv')
    crime_data = pd.read_csv('seoul_crime_with_detection_rates_cp949 (2).csv', encoding='cp949')
    showMap(cctv_data, crime_data)


