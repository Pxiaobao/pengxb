常用函数及使用方法：

- 去重：
	按全量字段去重, 保留第一个(默认)
	DataFrame.drop_duplicates()
	按指定字段去重, 保留第一个
	DataFrame.drop_duplicates(subset=['colA', 'colB'], keep='first')
	按全量字段去重, 保留最后一个
	DataFrame.drop_duplicates(keep='last')
- 取唯一值
	DataFrame['A'].unique()
- 按列取值
	直接选择想要的列
		df2 = df \[['A', 'D', 'E']\]
	方法一：按照列的位置删除
	df2 = df.drop(df.columns[1,2], axis=1)
	方法二：直接按列名删除
	df2 = df.drop(['B', 'C'], axis=1)
	正则
	df2 = df.filter(regex="\[^BD\]")

指定数据类型
      pd.read_csv(file,converters={'A1': str})

自定义列名：
	pd.read_csv('./t.csv',header=None,names=['工单编号','工单类型'])

取已知index的某一行数据：
	df.loc[a]
取未知index某一行的数据:
	df[1:2]#括号下包含，如取第二行数据则为应为[1:2]
取未知index某N行的数据:
	df[0:10]
给单元格赋值
	语句：df.loc[行号，'列名']= value
	例：df.loc[0,'name']='Elle'     给第一行name列赋值
语句：
	df.loc[df["STATIONID"] == station_id,"flag"] = 0  
	给列“STATIONID”等于station_id的行的“flag”赋值为0；

新增列：
	pandas 的 insert 方法，第一个参数指定插入列的位置，第二个参数指定插入列的列名，第三个参数指定插入列的数据
	data.insert(data.shape[1], 'd', 0)
	直接对 DataFrame 直接赋值即可
	data['d'] = 0

使用 reindex 函数，还可以指定缺失值填充的值，不过缺点是要把原有的列名和新列名都加上
	data = data.reindex(columns=['a', 'b', 'c', 'd'], fill_value=0)

concat 方法是用来拼接数据的，在这里是利用拼接过程中新建一个包含新列名的空DataFrame，好处是可以同时新增多个列名
	data = pd.concat([data, pd.DataFrame(columns=['d'])], sort=False)

修改列名：
	df.rename(columns={"A": "a", "B": "c"})

GroupBy：
	 df.groupby('columname',as_index=False).mean()

根据一列的值取另一列的第一个值：

data[data['id']==val]['describe'].values[0]

#取某一列等于某些值的行

ss[ss['org_name'].isin(['上海大众燃气有限公司','徐汇站','长宁站','静安站'])]

日期操作

date.strftime('%Y')

date.strftime('%m')  提取pandas日期类型中的完整月份

DataFrame的遍历：

iterrows():将DataFrame迭代为(insex, Series)对。

itertuples(): 将DataFrame迭代为元祖。

iteritems():将DataFrame迭代为(列名, Series)对。

dataframe对nan的处理：

# 如果A列不为nan，就将 B 列的值更新为 A 列的值

df.loc[df['A'].notnull(), 'B'] = df['A']

# 删除包含任何缺失值的行  
df.dropna(axis=0, how='any', inplace=True)  
# 删除所有值都缺失的列  
df.dropna(axis=1, how='all', inplace=True)

# 用均值填充缺失值  
df.fillna(df.mean(), inplace=True)  
# 用指定值填充缺失值  
df.fillna(value=0, inplace=True)

# 线性插值填充  
df.interpolate(method='linear', inplace=True)

# 更改列 B 的数据类型为整数型，并用 0 填充缺失值  
df['B'] = df['B'].astype(int)  
df.fillna(value=0, inplace=True)