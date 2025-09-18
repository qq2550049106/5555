import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from tqdm import tqdm
from bazi import *

# 1. 读数据
df = pd.read_csv('2025.csv')

# 2. 基础特征构造
cat_cols = ['特肖','特五行','特波色','日主五行','用神五行','本卦','之卦','世支','应支','日冲肖','日合肖']
def make_features(row):
    feats = {}
    # 原统计特征
    codes = [row[f'平码{i}'] for i in range(1,7)]
    feats['平码_max'] = max(codes)
    feats['平码_min'] = min(codes)
    feats['平码_sum'] = sum(codes)
    feats['平码_std'] = np.std(codes)
    feats['特尾'] = row['特尾']
    feats['特合'] = row['特合']
    feats['头']  = row['头']
    # 命理特征
    dt = pd.to_datetime(row['日期'])
    rizu, hgz = gz(dt.year, dt.month, dt.day, dt.hour)
    wux = riyuan_gz(rizu)
    stren = riyuan_strength(rizu, dt.month)
    yong = yong_shen(rizu, dt.month)
    bg, zg, shi, ying = qigua(dt)
    chong, liuhe = chonghe(rizu[1])
    feats['日主五行'] = wux
    feats['日元强弱'] = stren
    feats['用神五行'] = yong
    feats['本卦'] = bg
    feats['之卦'] = zg
    feats['世支'] = shi
    feats['应支'] = ying
    feats['日冲肖'] = chong
    feats['日合肖'] = liuhe
    feats['月'] = dt.month
    feats['日'] = dt.day
    feats['星期'] = dt.weekday()
    feats['时辰'] = dt.hour//2
    # 原类别特征
    for c in ['特肖','特五行','特波色']:
        feats[c] = row[c]
    return feats

X = pd.DataFrame([make_features(row) for _, row in tqdm(df.iterrows(), total=len(df))])
y = df['特码']

# 3. 类别编码
for c in cat_cols:
    X[c] = LabelEncoder().fit_transform(X[c])

# 4. 划分训练 / 测试（最后 10 期做回测）
train_x, train_y = X[:-10], y[:-10]
test_x,  test_y  = X[-10:], y[-10:]

# 5. 训练 LightGBM 多分类
model = LGBMClassifier(
    objective='multiclass',
    num_class=49,
    learning_rate=0.05,
    n_estimators=500,
    random_state=42
)
model.fit(train_x, train_y-1)

# 6. 回测：Top-5 命中率
proba = model.predict_proba(test_x)
top5_pred = np.argsort(proba, axis=1)[:, -5:] + 1
hit = [test_y.iloc[i] in top5_pred[i] for i in range(10)]
print('最近 10 期 Top-5 命中情况：', hit)
print('Top-5 命中率：{}%'.format(sum(hit)*10))

# 保存模型
import joblib
joblib.dump(model, 'lgb_model.pkl')