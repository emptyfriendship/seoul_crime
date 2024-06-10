# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:04:47 2024

@author: 109-2
"""

import pandas as pd

def load_data():
    file1 = '01_Seoul_CCTV_Data.csv'
    file2 = 'seoul_crime_with_detection_rates_cp949 (2).csv'
    data1 = pd.read_csv(file1, encoding='utf-8')
    data2 = pd.read_csv(file2, encoding='cp949')
    return data1, data2

