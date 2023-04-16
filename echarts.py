import utils
import numpy as np
import random
import data_process
from pyecharts.charts import Bar, Pie, Page, Line, Map, Radar, Geo
import pyecharts.options as opts
from pyecharts.globals import ThemeType
from pyecharts.components import Table
from pyecharts.options import ComponentTitleOpts

# echarts滑动
# dataZoom: [{
#       type: 'slider',
#       show: true,
#       yAxisIndex: [0],
#       right: '0%',
#       bottom: 70,
#       start: 10,
#       end: 15 //初始化滚动条
#   }],

# # 折线图滑动
# dataZoom: [{
#         type: 'slider',
#         show: true,
#         xAxisIndex: [0],
#         left: '9%',
#         bottom: 0,
#         start: 10,
#         end: 1 //初始化滚动条
#     }],

df = utils.csv2df('weather_.csv')
df_chongqing = utils.csv2df('chongqing_.csv')
name2index = {'times': 0, 'city': 1, 'date': 2, 'date_': 3, 'quality': 4, 'aqi': 5, 'aqi_rank': 6, 'pm25': 7, 'pm10': 8,
              'so2': 9, 'no2': 10, 'co': 11, 'o3': 12, 'situation': 13, 'wind': 14, 'temp': 15, 'min_temp': 16,
              'max_temp': 17}
name2index_chongqing = {'city': 0, 'date': 1, 'date_': 2, 'quality': 3, 'aqi': 4, 'aqi_rank': 5, 'pm25': 6, 'pm10': 7,
                        'so2': 8, 'no2': 9, 'co': 10, 'o3': 11, 'situation': 12, 'wind': 13, 'temp': 14, 'times': 15,
                        'min_temp': 16, 'max_temp': 17}
city_names = data_process.get_city_names(df)

## 获取全部城市的最高最低天气
min_temp_city_list_10 = np.array(sorted([[city_names[i],
                                          round(data_process.get_data_by_city(df, city_name=city_names[i])[
                                              name2index['min_temp']].values.mean(), 3)] for i in
                                         range(len(city_names))],
                                        key=lambda x: x[1]))
max_temp_city_list_10 = np.array(sorted([[city_names[i],
                                          round(data_process.get_data_by_city(df, city_name=city_names[i])[
                                              name2index['max_temp']].values.mean(), 3)] for i in
                                         range(len(city_names))],
                                        key=lambda x: x[1], reverse=True))

temp_bar = (
    Bar(init_opts=opts.InitOpts(theme=ThemeType.WHITE, width="400px", bg_color='rgba(6,48,109,.2)', height="400px",
                                chart_id='temp_bar'))
    .add_xaxis(list(min_temp_city_list_10[:, 0]))
    .add_xaxis(list(max_temp_city_list_10[:, 0]))
    .add_yaxis('平均最低温度', list(min_temp_city_list_10[:, 1]))
    .add_yaxis('平均最高温度', list(max_temp_city_list_10[:, 1]))
    .reversal_axis()
    .set_series_opts(label_opts=opts.LabelOpts(position='down'))
    .set_global_opts(title_opts=opts.TitleOpts(title="最高最低温度"))
)

## 重庆天气质量优良差占比
quality = df_chongqing[name2index['quality']].value_counts()
quality_degree = ['严重污染', '中度污染', '优', '良', '轻度污染', '重度污染']
quality_index = list(np.array(quality.index))
quality_names = [quality_degree[index] for index in quality_index]
quality_values = list(quality.values)
quality_pairs = [[quality_names[i], int(
    quality_values[i])] for i in range(len(quality_names))]

quality_pie = (
    Pie(init_opts=opts.InitOpts(theme=ThemeType.WHITE, width="450px", bg_color='rgba(6,48,109,.2)', height="350px",
                                chart_id='quality_bar'))
    .add(series_name="重庆空气质量",  # 系列名称
         data_pair=quality_pairs,
         rosetype="radius",  # 是否展示成南丁格尔图
         radius=["30%", "55%"],  # 扇区圆心角展现数据的百分比，半径展现数据的大小
         )  # 加入数据
    .set_global_opts(  # 全局设置项
        title_opts=opts.TitleOpts(title="重庆空气质量", pos_left='left'),  # 标题
        legend_opts=opts.LegendOpts(
            pos_left='right', orient='vertical')  # 图例设置项,靠右,竖向排列
    )
    .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}")))  # 样式设置项

## 重庆风力柱状情况
wind = df_chongqing[name2index['wind']].value_counts(sort=False)
wind_names = list(np.array(wind.index))
wind_values = [int(value) for value in wind.values]

wind_bar = (
    Bar(init_opts=opts.InitOpts(theme=ThemeType.WHITE, width="400px", bg_color='rgba(6,48,109,.2)', height="400px",
                                chart_id='wind_bar'))
    .add_xaxis(wind_names)
    .add_yaxis('风力情况', wind_values)
    .set_series_opts(label_opts=opts.LabelOpts(position='down'))
    .set_global_opts(title_opts=opts.TitleOpts(title="重庆风力"), xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=90)))
)

## 重庆成都北京上海深圳的平均空气指标
chongqing = data_process.get_data_by_city(df, '重庆')
chengdu = data_process.get_data_by_city(df, '成都')
beijing = data_process.get_data_by_city(df, '北京')
shanghai = data_process.get_data_by_city(df, '上海')
shenzhen = data_process.get_data_by_city(df, '深圳')
targets = ['pm25', 'pm10', 'so2', 'no2', 'co', 'o3']
chongqing_targets = [
    [round(chongqing[name2index[target]].values.mean(), 4) for target in targets]]
chengdu_targets = [
    [round(chengdu[name2index[target]].values.mean(), 4) for target in targets]]
beijing_targets = [
    [round(beijing[name2index[target]].values.mean(), 4) for target in targets]]
shanghai_targets = [
    [round(shanghai[name2index[target]].values.mean(), 4) for target in targets]]
shenzhen_targets = [
    [round(shenzhen[name2index[target]].values.mean(), 4) for target in targets]]

target_radar = (
    Radar(init_opts=opts.InitOpts(theme=ThemeType.WHITE, width="450px", bg_color='rgba(6,48,109,.2)', height="350px",
                                  chart_id='target_radar'))
    .add_schema(  # 添加schema架构
        schema=[
            opts.RadarIndicatorItem(name='pm2.5', max_=40),  # 设置指示器名称和最大值
            opts.RadarIndicatorItem(name='pm10', max_=60),
            opts.RadarIndicatorItem(name='so2', max_=10),
            opts.RadarIndicatorItem(name='no2', max_=30),
            opts.RadarIndicatorItem(name='co', max_=1),
            opts.RadarIndicatorItem(name='o3', max_=90),
        ]
    )
    .add('重庆', chongqing_targets, color="#fab27b")
    .add('成都', chengdu_targets, color="#f47920")
    .add('北京', beijing_targets, color="#f05b72")
    .add('上海', shanghai_targets, color="#1d953f")
    .add('深圳', shenzhen_targets, color="#6950a1")
    .set_global_opts(title_opts=opts.TitleOpts(title='各城市空气质量'),)
)

## 重庆13-22年的气温情况
monthes = ['01', '02', '03', '04', '05',
           '05', '07', '08', '09', '10', '11', '12']
chongqing_temp_index = [str(year)+"." + month
                        for year in range(13, 23) for month in monthes][:-3][10:]
chongqing_min_temp_value = []
chongqing_max_temp_value = []
chongqing_avg_temp_value = []

for index in chongqing_temp_index:
    df_temp = df_chongqing[(df_chongqing[name2index['date']].str[2:4]
                           == index[:2])]
    df_temp = df_chongqing[(df_chongqing[name2index['date']].str[5:7]
                           == index[3:])]
    min_temp = df_temp[name2index['min_temp']].values.min()
    max_temp = df_temp[name2index['max_temp']].values.max()
    chongqing_min_temp_value.append(round(min_temp))
    chongqing_max_temp_value.append(round(max_temp))
    chongqing_avg_temp_value.append((max_temp + min_temp) / 2)

cq_temp_line = Line(init_opts=opts.InitOpts(theme=ThemeType.WHITE, width="450px", bg_color='rgba(6,48,109,.2)', height="350px",
                                            chart_id='chongqing_temp')).add_xaxis(xaxis_data=chongqing_temp_index)  # 添加x轴
cq_temp_line.add_yaxis(  # 第一条曲线
    series_name='最高温度',
    y_axis=chongqing_max_temp_value,
    label_opts=opts.LabelOpts(is_show=False),
    yaxis_index=0,  # 设置y轴
    is_smooth=True,
)
cq_temp_line.add_yaxis(  # 添加第二条曲线
    series_name='最低温度',
    y_axis=chongqing_min_temp_value,
    label_opts=opts.LabelOpts(is_show=False),
    yaxis_index=0,  # 设置y轴
    is_smooth=True,
)
cq_temp_line.add_yaxis(  # 添加第三条曲线
    series_name='平均温度',
    y_axis=chongqing_avg_temp_value,
    label_opts=opts.LabelOpts(is_show=False),
    yaxis_index=0,  # 设置y轴
    is_smooth=True,
)

## 全国城市aqi
cities = {'河北': ['石家庄', '唐山', '秦皇岛', '保定', '张家口', '邯郸', '邢台', '承德', '沧州', '廊坊', '衡水'],
          '山西': ['太原', '大同', '阳泉', '长治', '临汾', '晋城', '朔州', '运城', '忻州', '吕梁', '晋中'],
          '内蒙古': ['呼和浩特', '包头', '鄂尔多斯', '乌海', '赤峰', '通辽', '巴彦淖尔', '兴安盟', '阿拉善盟', '呼伦贝尔', '二连浩特', '锡林郭勒'],
          '辽宁': ['沈阳', '大连', '丹东', '营口', '盘锦', '葫芦岛', '鞍山', '锦州', '本溪', '瓦房店', '抚顺', '辽阳', '阜新', '朝阳', '铁岭'],
          '吉林': ['长春', '吉林', '四平', '辽源', '白山', '松原', '白城', '延边', '通化'],
          '黑龙江': ['哈尔滨', '齐齐哈尔', '鸡西', '鹤岗', '双鸭山', '大庆', '佳木斯', '七台河', '牡丹江', '黑河', '绥化', '大兴安岭', '伊春', '甘南'],
          '江苏': ['南京', '无锡', '徐州', '常州', '苏州', '南通', '连云港', '淮安', '盐城', '扬州', '镇江', '泰州', '宿迁', '昆山', '海门', '太仓', '江阴', '溧阳', '金坛', '宜兴', '句容', '常熟', '吴江', '张家港'],
          '浙江': ['杭州', '宁波', '温州', '嘉兴', '湖州', '金华', '衢州', '舟山', '台州', '丽水', '绍兴', '义乌', '富阳', '临安'],
          '安徽': ['合肥', '芜湖', '蚌埠', '淮南', '马鞍山', '淮北', '铜陵', '安庆', '黄山', '滁州', '阜阳', '宿州', '巢湖', '六安', '亳州', '池州', '宣城'],
          '福建': ['福州', '厦门', '泉州', '莆田', '三明', '漳州', '南平', '龙岩', '宁德'],
          '江西': ['南昌', '景德镇', '萍乡', '新余', '鹰潭', '赣州', '宜春', '抚州', '九江', '上饶', '吉安'],
          '山东': ['济南', '青岛', '淄博', '枣庄', '东营', '烟台', '潍坊', '济宁', '泰安', '威海', '日照', '莱芜', '临沂', '德州', '聊城', '滨州', '菏泽', '乳山', '荣成', '文登', '章丘', '平度', '莱州', '招远', '莱西', '胶州', '蓬莱', '胶南', '寿光', '即墨'],
          '河南': ['郑州', '洛阳', '平顶山', '鹤壁', '焦作', '漯河', '三门峡', '南阳', '商丘', '信阳', '周口', '驻马店', '安阳', '开封', '濮阳', '许昌', '新乡'],
          '湖北': ['武汉', '十堰', '宜昌', '鄂州', '荆门', '孝感', '黄冈', '咸宁', '黄石', '恩施', '襄阳', '随州', '荆州'],
          '湖南': ['长沙', '株洲', '湘潭', '常德', '张家界', '益阳', '郴州', '永州', '怀化', '娄底', '邵阳', '岳阳', '湘西', '衡阳'],
          '广东': ['广州', '韶关', '深圳', '珠海', '汕头', '佛山', '江门', '肇庆', '惠州', '河源', '清远', '东莞', '中山', '湛江', '茂名', '梅州', '汕尾', '阳江', '潮州', '揭阳', '云浮'],
          '广西': ['南宁', '柳州', '北海', '桂林', '梧州', '防城港', '钦州', '贵港', '玉林', '百色', '贺州', '河池', '来宾', '崇左'],
          '海南': ['海口', '三亚'],
          '四川': ['成都', '自贡', '攀枝花', '泸州', '德阳', '绵阳', '广元', '遂宁', '乐山', '南充', '眉山', '达州', '雅安', '巴中', '资阳', '甘孜', '内江', '宜宾', '广安', '阿坝', '凉山'],
          '贵州': ['贵阳', '六盘水', '遵义', '安顺', '毕节', '铜仁', '黔西南', '黔南', '黔东南'],
          '云南': ['昆明', '玉溪', '保山', '昭通', '丽江', '临沧', '西双版纳', '德宏', '怒江', '大理', '曲靖', '楚雄', '红河', '思茅', '文山', '普洱', '迪庆'],
          '西藏': ['拉萨', '林芝', '山南', '昌都', '日喀则', '阿里', '那曲'],
          '陕西': ['西安', '铜川', '宝鸡', '咸阳', '渭南', '延安', '汉中', '榆林', '安康', '商洛'],
          '甘肃': ['兰州', '嘉峪关', '天水', '武威', '张掖', '平凉', '酒泉', '庆阳', '定西', '甘南', '临夏', '白银', '金昌', '陇南'],
          '青海': ['西宁', '海东', '果洛', '海北', '海南', '海西', '玉树', '黄南'],
          '宁夏': ['银川', '石嘴山', '吴忠', '固原', '中卫'],
          '新疆': ['乌鲁木齐', '伊犁哈萨克州', '克拉玛依', '哈密', '石河子', '和田', '五家渠', '阿克苏', '阿勒泰', '喀什', '库尔勒', '吐鲁番', '塔城', '博州', '昌吉', '克州'],
          '北京': ['北京'],
          '上海': ['上海'],
          '天津': ['天津'],
          '重庆': ['重庆']
          }

aqi_city_list = [data_process.get_data_by_city(
    df, city)[name2index['aqi']].values.mean() for city in city_names]
aqi_city_pair = [[city_name, city_aqi]
                 for city_name, city_aqi in zip(city_names, aqi_city_list)]
cities_names = list(cities.keys())
cities_aqi = [[] for i in range(len(cities_names))]

for pair in aqi_city_pair:
    i = 0
    for k, v in cities.items():
        if pair[0] in v:
            cities_aqi[i].append(pair[1])
        i += 1

cities_aqi = [np.array(aqi_list).mean() for aqi_list in cities_aqi]
aqi_cities_pair = [[name, float(val)]
                   for name, val in zip(cities_names, cities_aqi)]

aqi_map = (
    Map(init_opts=opts.InitOpts(theme=ThemeType.WHITE, width="450px", bg_color='rgba(6,48,109,.2)', height="350px",
                                chart_id='aqi_map'))
    .add("AQI", aqi_cities_pair, "china")
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(title="全国AQI"),
        visualmap_opts=opts.VisualMapOpts(max_=100),
    )
)

## 模型预测概率
model_name = ['SVM', 'DecisionTree', 'RandomForest', 'MLP', 'Transformer']
train = [98.04, 97.58, 97.88, 97.12, 96.49]
test = [96.18, 95.34, 96.45, 96.90, 97.54]

model_bar = (
    Bar(init_opts=opts.InitOpts(theme=ThemeType.WHITE, width="500px", bg_color='rgba(6,48,109,.2)', height="400px",
                                chart_id='bar1'))
    .add_xaxis(model_name)
    .add_yaxis("train", train)
    .add_yaxis("test", test)
    .set_global_opts(title_opts=opts.TitleOpts(title="模型预测比赛胜率"),
                     toolbox_opts=opts.BrushOpts(),
                     xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=30)))
)

# 拖动大屏
page = Page(
    page_title="基于Python的英雄联盟数据分析",
    layout=Page.DraggablePageLayout,  # 拖拽方式
)
page.add(
    temp_bar,
    wind_bar,
    quality_pie,
    target_radar,
    cq_temp_line,
    aqi_map,
    model_bar
)
page.render('大屏_临时.html')

# ## 生成最终大屏
# Page.save_resize_html(
#     source='大屏_临时.html',
#     cfg_file='./chart_config.json',
#     dest="final.html"
# )