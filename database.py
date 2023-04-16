import utils
import pymysql
import pandas as pd
from sqlalchemy import create_engine

is_translate = True

db = pymysql.connect(
    host="localhost",
    port=3306,
    user="root",
    password="616131",
    database="swim",
    charset='utf8'
)

def csv2sql(db, path,table_name):
    '''将csv文件转化sql文件'''
    mysql_setting = {
        'host': 'localhost',
        'port': 3306,
        'user': 'root',
        'passwd': '616131',
        # 数据库名称
        'db': db,
        'charset': 'utf8'
    }
    path = path  # csv路径
    table_name = table_name  # mysql表
    df = pd.read_csv(path, encoding='utf-8')
    engine = create_engine("mysql+pymysql://{user}:{passwd}@{host}:{port}/{db}".format(**mysql_setting), max_overflow=5)
    df.to_sql(table_name, engine, index=False, if_exists='replace')
    print("ok")


def sql2df(db):
    '''从mysql中读取数据存入DataFrame'''
    mysql_setting = {
        'host': 'localhost',
        'port': 3306,
        'user': 'root',
        'passwd': '616131',
        # 数据库名称
        'db': db,
        'charset': 'utf8'
    }
    engine = create_engine("mysql+pymysql://{user}:{passwd}@{host}:{port}/{db}".format(**mysql_setting), max_overflow=5)
    sql = '''select * from matches'''  # sql查询语句
    df = pd.read_sql(sql, engine)
    return df

if is_translate:
    csv2sql('swim','weather_.csv',"weather")