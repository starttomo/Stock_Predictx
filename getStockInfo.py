import baostock as bs
import pandas as pd
import mysql.connector


def get_baostock_data():
    try:
        # 登录系统
        lg = bs.login()
        print('baostock登录响应:', lg.error_msg)

        # 获取沪深300指数数据
        rs = bs.query_history_k_data_plus("sh.000300",
                                          "date,open,high,low,close,volume,amount",
                                          start_date='2015-01-01', end_date='2025-10-30',
                                          frequency="d", adjustflag="3")

        print('查询历史K线数据响应:', rs.error_msg)

        # 转换成DataFrame
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())

        df = pd.DataFrame(data_list, columns=rs.fields)

        # 数据类型转换
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        # 登出系统
        bs.logout()

        return df
    except Exception as e:
        print(f"baostock获取失败: {e}")
        return pd.DataFrame()


# 安装baostock
# pip install baostock

df = get_baostock_data()
if len(df) > 0:
    print(f"baostock获取 {len(df)} 条数据")
if len(df) > 0:
    print(f"共获取 {len(df)} 条数据")
    print(df.head())

    # 连接数据库
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='zhang728',
        database='stock_system'
    )
    cursor = conn.cursor()

    # 清空表
    cursor.execute("TRUNCATE TABLE stocks")
    print("已清空 stocks 表")

    # 插入数据
    inserted = 0
    for _, row in df.iterrows():
        try:
            cursor.execute('''
                INSERT INTO stocks (date, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s)
            ''', (row['date'], row['open'], row['high'], row['low'], row['close'], row['volume']))
            inserted += 1
        except mysql.connector.Error as e:
            if e.errno == 1062:
                print(f"跳过重复日期: {row['date']}")
            else:
                print(f"插入错误 {row['date']}: {e}")

    print(f"尝试插入 {len(df)} 条，成功 {inserted} 条")

    # 提交
    conn.commit()
    print("数据已提交！")

    # 验证
    cursor.execute("SELECT COUNT(*) FROM stocks")
    count = cursor.fetchone()[0]
    print(f"数据库当前总条数: {count}")

    cursor.close()
    conn.close()
else:
    print("未能获取到数据")