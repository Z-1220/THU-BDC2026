"""
标注沪深300股票行业分类
"""

import pandas as pd

# 读取行业分类数据
industry_df = pd.read_csv('doc/行业分类.csv', encoding='utf-8')

# 创建证券代码到行业分类的映射字典（使用一级行业分类简称）
industry_dict = dict(zip(industry_df['证券代码'].astype(str).str.zfill(6), industry_df['中证一级行业分类简称']))

# 读取HS300股票列表
stock_df = pd.read_csv('data/hs300_stock_list.csv')

# 提取股票代码（去掉前缀sh.或sz.）
stock_df['code_clean'] = stock_df['code'].str.split('.').str[1]

# 根据证券代码添加行业分类
stock_df['行业'] = stock_df['code_clean'].map(industry_dict)

# 保存标注后的数据到新文件
stock_df.to_csv('data/hs300_stock_list_annotated.csv', index=False, encoding='utf-8-sig')

print("标注完成，结果保存到 data/hs300_stock_list_annotated.csv")