import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('SIRTUIN6.csv')

X = df.drop('Class', axis=1)
y = df['Class']

def scaleData(data, scaler):
    dataScaled = scaler.transform(data)
    return dataScaled

def handle_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1 
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data[column] = data[column].apply(lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x))
    return data

for col in df.select_dtypes(include=['int', 'float']).columns:
    df = handle_outliers(df, col)

scaler = StandardScaler()
indices = np.arange(X.shape[0])

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, indices, test_size=0.2, random_state=42
)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_scaled = scaler.transform(X)

X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, test_size=0.2, random_state=42)

# normal tree
normal_tree = DecisionTreeClassifier(random_state=42)
normal_tree.fit(X1_train, y1_train)

# prune tree
prePrune_tree = DecisionTreeClassifier(max_depth=3, max_features=6, max_leaf_nodes=10, min_samples_leaf=3, min_samples_split=2, random_state=42)
prePrune_tree = prePrune_tree.fit(X1_train, y1_train)

# model KNN dengan scaled data
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# model KNN dengan scaled data + tuning parameter
KNNtuned = KNeighborsClassifier(leaf_size=5,metric='euclidean',n_neighbors=9,weights='uniform')
KNNtuned.fit(X_train, y_train)

# model KNN dengan unscaled data
knn1 = KNeighborsClassifier()
knn1.fit(X1_train, y1_train)

# model KNN dengan unscaled data + tuning parameter
KNNtuned1 = KNeighborsClassifier(leaf_size=5,metric='manhattan',n_neighbors=3,weights='uniform')
KNNtuned1.fit(X1_train, y1_train)

st.title('Prediction Sirtuin6 Molecules')

SC5 = st.number_input('Masukkan nilai SC-5: ', value=0.0, step=0.00005, format='%.5f')
SP6 = st.number_input('Masukkan nilai SP-6: ', value=0.0, step=0.00005, format='%.5f')
SHBd = st.number_input('Masukkan nilai SHBd: ', value=0.0, step=0.00005, format='%.5f')
minHaaCH = st.number_input('Masukkan nilai minHaaCH: ', value=0.0, step=0.00005, format='%.5f')
maxwHBa = st.number_input('Masukkan nilai maxwHBa: ', value=0.0, step=0.00005, format='%.5f')
FMF = st.number_input('Masukkan nilai FMF: ', value=0.0, step=0.00005, format='%.5f')

if st.button('Predict Class'):
    data = pd.DataFrame({
    'SC-5': [SC5],
    'SP-6': [SP6],
    'SHBd': [SHBd],
    'minHaaCH': [minHaaCH],
    'maxwHBa': [maxwHBa],
    'FMF' : [FMF]
    })
    data_Scaled = scaleData(data, scaler)

    y_pred_normal_tree = normal_tree.predict(data)
    y_pred_prune_tree = prePrune_tree.predict(data)
    y_pred_knn_scaled = knn.predict(data)
    y_pred_tuned_knn_scaled = KNNtuned.predict(data)
    y_pred_knn_unscaled = knn1.predict(data)
    y_pred_tuned_knn_unscaled = KNNtuned1.predict(data)

    st.write(f'**Decision Tree:** ')
    st.write(f'Prediksi Class menggunakan Decision Tree: **{y_pred_normal_tree[0].replace('_', ' ')}**')
    st.write(f'Prediksi Class menggunakan Pruned Decision Tree: **{y_pred_prune_tree[0].replace('_', ' ')}**')
    st.write('')
    st.write(f'**KNN menggunakan Unscaled Data:** ')
    st.write(f'Prediksi Class menggunakan KNN: **{y_pred_knn_unscaled[0].replace('_', ' ')}**')
    st.write(f'Prediksi Class menggunakan Tuned KNN: **{y_pred_tuned_knn_unscaled[0].replace('_', ' ')}**')
    st.write('')
    st.write(f'**KNN menggunakan Scaled Data:** ')
    st.write(f'Prediksi Class menggunakan KNN: **{y_pred_knn_scaled[0].replace('_', ' ')}**')
    st.write(f'Prediksi Class menggunakan Tuned KNN: **{y_pred_tuned_knn_scaled[0].replace('_', ' ')}**')
