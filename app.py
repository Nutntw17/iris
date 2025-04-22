import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ตั้งค่าหน้าเพจ
st.set_page_config(
    page_title="การแสดงผลข้อมูลดอกไอริส",
    page_icon="🌸",
    layout="wide"
)

# โหลดข้อมูล
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('iris.csv')
        return data
    except:
        st.error("ไม่สามารถโหลดไฟล์ iris.csv ได้")
        return None

df = load_data()

if df is not None:
    # หัวข้อหลัก
    st.title("🌸 การแสดงผลข้อมูลดอกไอริส")
    st.markdown("### แผนภาพคู่ของคุณลักษณะดอกไอริสแบ่งตามสายพันธุ์")
    
    # สร้าง pairplot แบบพื้นฐาน
    try:
        # วิธีที่ 1: ใช้ seaborn pairplot ปกติ
        fig = plt.figure(figsize=(10, 10))
        g = sns.pairplot(df, hue='species')
        st.pyplot(g.fig)
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการสร้างกราฟแบบที่ 1: {str(e)}")
        
        try:
            # วิธีที่ 2: ลองใช้การสร้าง matplotlib แบบตรงๆ
            fig, axes = plt.subplots(4, 4, figsize=(15, 15))
            
            features = df.columns[:-1]
            
            for i, feature_x in enumerate(features):
                for j, feature_y in enumerate(features):
                    ax = axes[i, j]
                    
                    if i == j:  # แสดงฮิสโตแกรมบนแนวทแยง
                        for species, color in zip(['setosa', 'versicolor', 'virginica'], ['blue', 'orange', 'green']):
                            data = df[df['species'] == species]
                            ax.hist(data[feature_x], alpha=0.5, color=color, label=species)
                    else:  # แสดงกราฟกระจายสำหรับคู่คุณลักษณะ
                        for species, color in zip(['setosa', 'versicolor', 'virginica'], ['blue', 'orange', 'green']):
                            data = df[df['species'] == species]
                            ax.scatter(data[feature_x], data[feature_y], alpha=0.5, color=color, label=species)
                    
                    if i == 3:
                        ax.set_xlabel(feature_x)
                    if j == 0:
                        ax.set_ylabel(feature_y)
            
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right')
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการสร้างกราฟแบบที่ 2: {str(e)}")
            
            # วิธีที่ 3: แสดงคู่กราฟแต่ละคู่แยกกัน
            st.write("แสดงกราฟแยกเป็นคู่ๆ:")
            
            features = df.columns[:-1]
            for i, feature_x in enumerate(features):
                for j, feature_y in enumerate(features):
                    if i < j:  # แสดงเฉพาะครึ่งบนของเมทริกซ์
                        fig, ax = plt.subplots(figsize=(5, 5))
                        for species, color in zip(['setosa', 'versicolor', 'virginica'], ['blue', 'orange', 'green']):
                            data = df[df['species'] == species]
                            ax.scatter(data[feature_x], data[feature_y], alpha=0.7, color=color, label=species)
                        ax.set_xlabel(feature_x)
                        ax.set_ylabel(feature_y)
                        ax.legend()
                        st.pyplot(fig)
else:
    st.error("ไม่สามารถดำเนินการต่อเนื่องจากไม่พบข้อมูล")
