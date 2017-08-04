import pandas as pd
import numpy as np


# 读取算法所需的列
def get_data():
    df = pd.read_csv('./data/GZ_traffic.csv', usecols=[1, 5, 9, 11, 12, 13, 
                                                14,15, 16, 17, 18, 
                                                19, 20,21, 22, 23, 
                                                24, 25, 26,27, 28])
    return df


# 对数据进行归一化处理
def normalization():
    dataframe = get_data()
    data_sorted = dataframe.sort_values('sjxh', ascending=True)
    pieces1 = data_sorted.ix[:, 0:3]
    pieces2 = data_sorted.ix[:, 3:]
    new_pieces2 = pieces2.apply(lambda x: (x-x.min())/(x.max()-x.min()), axis=0)
    dn = pd.concat([pieces1, new_pieces2], axis=1)
    return dn


# 求车辆数和车速的平均值并按时间序号和车道号分组
def group_by_time_series():
    data_norm = normalization()
    vehicles = data_norm.ix[:, 3:11]
    velocity = data_norm.ix[:, 11:]
    left_cols = data_norm.ix[:, 0:3]
    vehicles['vehicles'] = vehicles.apply(lambda x: x.mean(), axis=1)
    velocity['velocity'] = velocity.apply(lambda x: x.mean(), axis=1)
    new_cols = pd.concat([left_cols, vehicles.loc[:,'vehicles'], velocity.loc[:,'velocity']], axis=1) 
    grouped = new_cols[['vehicles', 'velocity']].groupby([new_cols['gcrq'], new_cols['sjxh'], new_cols['cdh']]).mean()
    return grouped 


# 处理缺失值并转换成新的数据文件
def convert_to_norm_csv():
    dataset = group_by_time_series()
    norm_data = dataset.fillna(dataset.mean())
    norm_data.to_csv('norm_traffic.csv')
    return norm_data


#data = convert_to_norm_csv()
#print data
