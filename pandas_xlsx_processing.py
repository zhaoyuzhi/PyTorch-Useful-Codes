# -*- coding: utf-8 -*-
import os
import pandas as pd

excel_path = '123.xlsx'

def read_excel(excel_path):
    # 读取excel
    excel_pd = pd.read_excel(excel_path, sheet_name = 'Sheet1')
    # 读取一列的数据
    excel_pd_column_a = excel_pd['a']
    # 读取多列的数据
    excel_pd_column_a_and_b = excel_pd[['a', 'b']]
    # 将一列的数据转为list
    excel_pd_column_a_list = excel_pd_column_a.values.tolist()
    # 读取一行的数据
    excel_pd_row_2 = excel_pd[2]
    # 读取多行的数据
    excel_pd_row_1_to_4 = excel_pd[1:4]

def read_excel_with_try(excel_path):
    # 读取excel
    try:
        excel_pd = pd.read_excel(excel_path, sheet_name = 'Sheet1')
        # 读取一列的数据
        try:
            excel_pd_column_a = excel_pd['a']
        except KeyError:
            pass
    except ValueError:
        pass

def save_excel(df1, df2, save_path = 'output.xlsx'):
    with pd.ExcelWriter(save_path) as writer:
        df1.to_excel(writer, 'sheet1')
        df2.to_excel(writer, 'sheet2')
