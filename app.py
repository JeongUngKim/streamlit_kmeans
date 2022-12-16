import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

def main() :
    st.title('K-means 클러스터링')

    # 1. csv 파일을 업로드 할 수 있다.
    file = st.file_uploader('CSV 파일 업로드',type=['csv'])

    # 2. 데이터 프레임으로 읽어오기

    if file is not None :
        df=pd.read_csv(file)
        st.dataframe(df)
    # 3. X로 설정할 데이터를 선택해주세요.
        st.dataframe(df.isna().sum())
        # df의 비어있는 데이터 삭제
        df.dropna(inplace=True)
        selected_x = st.multiselect('X로 설정할 데이터를 선택해주세요',df.columns)
        X = df.loc[:,selected_x]
        if len(X) > 0 : 
            st.dataframe(X)
            # X의 컬럼 데이터가 문자가 있는지 확인
            st.text('컬럼 데이터의 문자가 있는지 확인(object : 문자)')
            st.dataframe(X.dtypes)
            X_object_column=X.columns.values[X.dtypes == 'object']
            if X[X_object_column].nunique().values == 2 :
                # label 실행
                encoder = LabelEncoder()
                st.dataframe(X[X_object_column])
                st.text(X_object_column)
                X[X_object_column] = pd.DataFrame(encoder.fit_transform( X[X_object_column] ))
                st.dataframe(X)
            elif X[X_object_column].nunique().values > 2:
                # one-hot 실행
                X_list = X.columns.values.tolist()
                X_list=X_list.index(X_object_column)
                ct = ColumnTransformer( [ ( 'encoder', OneHotEncoder() , [X_list]  ) ] , 
                  remainder= 'passthrough' )
                X = ct.fit_transform(X)
                st.dataframe(X)
        # 4. wcss 를 위한 그룹의 갯수를 정해달라고 하자.
            st.subheader('wcss를 위한 클러스터링 갯수를 선택')
            slider_size = st.slider('최대 그룹 선택',min_value=2,max_value=20,value=10)
            wcss = []
            if slider_size < len(X) :
                for k in np.arange(1,slider_size+1) :
                    kmeans=KMeans(n_clusters=k , random_state=5)
                    kmeans.fit(X)
                    wcss.append(kmeans.inertia_)
            # 5. 엘보우 메소드를 차트로 나타내시오.
                fig = plt.figure()
                plt.plot(np.arange(1,slider_size+1) , wcss)
                plt.title('The Elbow Method')
                plt.xlabel('Number of Clusters')
                plt.ylabel('WCSS')
                st.pyplot(fig)
            # 7. 실제로 그룹핑할 갯수를 선택 .
                # cluster_size = st.slider('그룹 갯수 결정',min_value=1,max_value=slider_size)
                cluster_size = st.number_input('그룹 갯수 결정',1,slider_size)
            # 8. 실제로 kmeas 작동하기
                kmeans = KMeans(n_clusters=cluster_size,random_state=5)
                y_pred = kmeans.fit_predict(X)
                df['Group'] = y_pred
                st.dataframe(df.sort_values('Group'))

            # 9. df 저장
                df.to_csv('result.csv')

if __name__ == '__main__' :
    main()