import streamlit as st
import joblib
import re
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ==========================================
# 0. 網頁基本設定
# ==========================================
st.set_page_config(page_title="新聞真偽雙模型偵測系統", page_icon="📰", layout="wide")

# 🌟 全域字體設定：確保 Matplotlib 在網頁端繪圖時中文不會變成「口口口」
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'simhei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False 

# ==========================================
# 1. 載入模型與資料 (使用快取加速網頁)
# ==========================================
@st.cache_resource
def load_ml_components():
    vectorizer = joblib.load('tfidf_vectorizer_cleaned.pkl')
    dt_model = joblib.load('fake_news_dt_model_cleaned.pkl')
    rf_model = joblib.load('fake_news_rf_model_cleaned.pkl')
    return vectorizer, dt_model, rf_model

@st.cache_data
def load_sample_data():
    # 讀取部分資料供 EDA 分頁展示
    try:
        df_true = pd.read_csv('True.csv', nrows=1000)
        df_fake = pd.read_csv('Fake.csv', nrows=1000)
        df_true['Label'] = '真新聞 (True)'
        df_fake['Label'] = '假新聞 (Fake)'
        df = pd.concat([df_true, df_fake], ignore_index=True)
        df['Word_Count'] = df['text'].apply(lambda x: len(str(x).split()))
        return df
    except:
        # 若找不到原始 CSV 檔，建立虛擬資料避免網頁崩潰
        return pd.DataFrame({
            'title': ['Sample News Title'] * 10,
            'text': ['Sample content...'] * 10,
            'subject': ['politics'] * 10,
            'Label': ['真新聞 (True)'] * 5 + ['假新聞 (Fake)'] * 5,
            'Word_Count': [400, 500, 300, 600, 450, 200, 150, 180, 220, 250]
        })

vectorizer, dt_model, rf_model = load_ml_components()
df_sample = load_sample_data()

st.title("📰 AI 假新聞與內容農場雙模型偵測系統")
st.markdown("本系統整合了自然語言處理 (NLP) 與機器學習技術。請透過下方頁籤切換不同功能：")

# ==========================================
# 2. 建立三大分頁 (Tabs)
# ==========================================
tab1, tab2, tab3 = st.tabs(["🔍 系統實測區", "📊 資料集探索 (EDA)", "🏆 模型績效評估"])

# ------------------------------------------
# 分頁 1: 系統實測區
# ------------------------------------------
with tab1:
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'^.*?(reuters|associated_press|ap).*?-', '', text)
        text = text.replace('reuters', '').replace('getty images', '')
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        return text

    st.markdown("### 貼上新聞，即時打假！")
    user_input = st.text_area("請在此貼上欲鑑定的新聞內容 (英文)：", height=200, placeholder="請貼上完整的英文新聞內文或標題...")

    if st.button("開始雙模型交叉鑑定 🔍", type="primary"):
        if not user_input.strip():
            st.warning("⚠️ 請輸入新聞內容！")
        else:
            cleaned_input = clean_text(user_input)
            input_vec = vectorizer.transform([cleaned_input])
            
            col1, col2 = st.columns(2)
            
            # 決策樹預測結果
            with col1:
                st.markdown("#### 🌳 單一決策樹 (Decision Tree)")
                dt_pred = dt_model.predict(input_vec)[0]
                dt_proba = dt_model.predict_proba(input_vec)[0]
                dt_confidence = max(dt_proba) * 100
                if dt_pred == 1:
                    st.success("✅ 鑑定結果：真實新聞 (True News)")
                else:
                    st.error("❌ 鑑定結果：假新聞/農場文 (Fake News)")
                st.metric(label="模型信心指數", value=f"{dt_confidence:.2f}%")
                st.progress(int(dt_proba[1] * 100), text=f"真新聞機率: {dt_proba[1]*100:.1f}%")
            
            # 隨機森林預測結果
            with col2:
                st.markdown("#### 🌲 隨機森林 (Random Forest)")
                rf_pred = rf_model.predict(input_vec)[0]
                rf_proba = rf_model.predict_proba(input_vec)[0]
                rf_confidence = max(rf_proba) * 100
                if rf_pred == 1:
                    st.success("✅ 鑑定結果：真實新聞 (True News)")
                else:
                    st.error("❌ 鑑定結果：假新聞/農場文 (Fake News)")
                st.metric(label="模型信心指數", value=f"{rf_confidence:.2f}%")
                st.progress(int(rf_proba[1] * 100), text=f"真新聞機率: {rf_proba[1]*100:.1f}%")

# ------------------------------------------
# 分頁 2: 資料集探索 (EDA)
# ------------------------------------------
with tab2:
    st.markdown("### 1. 原始資料預覽")
    st.caption("展示真假新聞資料集整合後之結構預覽。")
    st.dataframe(df_sample[['title', 'text', 'subject', 'Label', 'Word_Count']].head(100), use_container_width=True)

    st.markdown("---")
    st.markdown("### 2. 互動式圖表分析：文章字數分佈")
    
    fig = px.box(df_sample, x="Label", y="Word_Count", color="Label", 
                 title="真假新聞內文字數 (Word Count) 分佈比較",
                 labels={"Word_Count": "文章字數 (Words)", "Label": "新聞真實性"},
                 color_discrete_map={'真新聞 (True)': '#1f77b4', '假新聞 (Fake)': '#2ca02c'})
    
    fig.update_yaxes(range=[0, 1500]) 
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# 分頁 3: 模型績效評估 (🌟 真實數據與雙矩陣更新區)
# ------------------------------------------
with tab3:
    st.markdown("### 🏆 雙模型真實績效大比拼 (測試集驗證)")
    st.markdown("以下指標與混淆矩陣皆為卸下通訊社作弊特徵（Data Leakage）後，模型對 **8,980 篇全新測試新聞**的真實表現：")
    
    # 建立左右兩大欄位對比成效
    metric_col1, metric_col2 = st.columns(2)
    
    with metric_col1:
        st.markdown("#### 🌲 隨機森林 評估指標")
        c1, c2, c3 = st.columns(3)
        c1.metric("準確率 (Accuracy)", "95.59%")
        c2.metric("精準度 (Precision)", "95.18%")
        c3.metric("召回率 (Recall)", "95.70%")
        
    with metric_col2:
        st.markdown("#### 🌳 單一決策樹 評估指標")
        c4, c5, c6 = st.columns(3)
        c4.metric("準確率 (Accuracy)", "92.49%")
        c5.metric("精準度 (Precision)", "89.88%")
        c6.metric("召回率 (Recall)", "95.15%")
        
    st.markdown("---")
    st.markdown("### 📊 混淆矩陣交叉對比 (Confusion Matrix Comparison)")
    st.caption("觀察重點：隨機森林（藍圖）相較於單一決策樹（綠圖），顯著降低了右上角『誤放假新聞 (FP)』與左下角『誤殺真新聞 (FN)』的數量。")
    
    # 並排繪製兩張混淆矩陣圖
    plot_col1, plot_col2 = st.columns(2)
    
    # 1. 隨機森林混淆矩陣 (藍色)
    with plot_col1:
        rf_cm_data = [[4440, 210],   # 實際假/預測假 (TN), 實際假/預測真 (FP)
                      [186, 4144]]   # 實際真/預測假 (FN), 實際真/預測真 (TP)
        
        fig_rf, ax_rf = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(rf_cm_data, annot=True, fmt='d', cmap='Blues', ax=ax_rf,
                    xticklabels=['預測: 假新聞', '預測: 真新聞'], 
                    yticklabels=['實際: 假新聞', '實際: 真新聞'],
                    annot_kws={"size": 13})
        ax_rf.set_title('隨機森林 混淆矩陣 (Random Forest)', fontsize=14, pad=10)
        ax_rf.set_ylabel('真實標籤 (Actual)', fontsize=11)
        ax_rf.set_xlabel('預測標籤 (Predicted)', fontsize=11)
        st.pyplot(fig_rf)
        
    # 2. 決策樹混淆矩陣 (綠色)
    with plot_col2:
        dt_cm_data = [[4186, 464],   # 實際假/預測假 (TN), 實際假/預測真 (FP)
                      [210, 4120]]   # 實際真/預測假 (FN), 實際真/預測真 (TP)
        
        fig_dt, ax_dt = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(dt_cm_data, annot=True, fmt='d', cmap='Greens', ax=ax_dt,
                    xticklabels=['預測: 假新聞', '預測: 真新聞'], 
                    yticklabels=['實際: 假新聞', '實際: 真新聞'],
                    annot_kws={"size": 13})
        ax_dt.set_title('單一決策樹 混淆矩陣 (Decision Tree)', fontsize=14, pad=10)
        ax_dt.set_ylabel('真實標籤 (Actual)', fontsize=11)
        ax_dt.set_xlabel('預測標籤 (Predicted)', fontsize=11)
        st.pyplot(fig_dt)

    # 結論總結
    st.info("""
    💡 **數據洞察結論：** 雖然單一決策樹已具備 92.49% 的不錯表現，但其最大的缺點在於**誤放率高 (FP = 464 筆)**。
    當改用隨機森林群體投票後，**錯誤放行的假新聞成功被縮減了一半以上 (從 464 筆降至 210 筆)**，且精準度大幅拉高至 95.18%，有效證明了集成學習（Ensemble Learning）在處理大眾傳播社群文本時的強健度與實用價值。
    """)