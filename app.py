import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from textblob import TextBlob
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

st.set_page_config(page_title="WhatsApp NLP Dashboard", layout="wide")
st.title("📊 WhatsApp Chat Analysis Dashboard")

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload WhatsApp CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, on_bad_lines="skip", encoding="utf-8")

    # =========================
    # BASIC CLEANING
    # =========================
    df = df.dropna(subset=["date", "time", "name", "chat"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["chat"] = df["chat"].astype(str)

    # Normalize flirt_label
    df["Flirt_Label"] = df["Flirt_Label"].astype(str).str.lower()
    df["is_flirt"] = df["Flirt_Label"].isin(["1", "flirt", "yes", "true"]).astype(int)

    # =========================
    # DATE-WISE COUNT CHART
    # =========================
    st.subheader("📅 Date-wise Message Count")
    
    plt.figure(figsize=(8,4))
    ax1=sns.countplot(x='date',hue='name', data=df)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()
    st.pyplot(plt,clear_figure=True)
   # =========================
    # DASHBOARD COLUMNS
    # =========================
    col1, col2, col3 = st.columns(3)

    # =========================
    # FLIRT ENCOUNTERS
    # =========================
    flirt_stats = df.groupby("name").agg(
        Total_Messages=("chat", "count"),
        Flirt_Messages=("is_flirt", "sum")
    )

    flirt_stats["Flirt %"] = round(
        (flirt_stats["Flirt_Messages"] / flirt_stats["Total_Messages"]) * 100, 2
    )

    most_talkative = flirt_stats["Total_Messages"].idxmax()
    least_talkative = flirt_stats["Total_Messages"].idxmin()

    with col1:
        st.subheader("💬 Flirt Encounters")
        st.write("**Most Talkative:**", most_talkative)
        st.write("**Least Talkative:**", least_talkative)
        st.dataframe(flirt_stats[["Flirt %"]])

    # =========================
    # TIME ENCOUNTER
    # =========================
    df["Hour"] = pd.to_datetime(df["time"], errors="coerce").dt.hour

    most_active_date = df["date"].dt.date.value_counts().idxmax()
    most_active_hour = df["Hour"].value_counts().idxmax()
    avg_msg_per_day = round(df.groupby(df["date"].dt.date).size().mean(), 2)

    with col2:
        st.subheader("⏰ Time Encounter")
        st.write("**Most Active Date:**", most_active_date)
        st.write("**Most Active Time:**", f"{most_active_hour}:00")
        st.write("**Average Messages / Day:**", avg_msg_per_day)

    # =========================
    # MEDIA ENCOUNTER
    # =========================
    media_df = pd.DataFrame({
        "Metric": [
            "Media Count",
            "Message Deleted",
            "Missed Audio Call",
            "Missed Video Call"
        ],
        "Count": [
            df["chat"].str.contains("media omitted", case=False).sum(),
            df["chat"].str.contains("deleted", case=False).sum(),
            df["chat"].str.contains("missed voice call", case=False).sum(),
            df["chat"].str.contains("missed video call", case=False).sum()
        ]
    })

    with col3:
        st.subheader("📎 Media Encounter")
        st.dataframe(media_df)

    # =========================
    # SENTIMENT ANALYSIS
    # =========================
    st.subheader("😊 Sentiment Analysis")
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia=SentimentIntensityAnalyzer()
    df['scores']=df['chat'].apply(lambda x:sia.polarity_scores(str(x)))
    df['compound']=df['scores'].apply(lambda x:x['compound'])
    df['comp_score']=df['compound'].apply(lambda x: 'pos' if x>0 else 'neg')
    plt.figure(figsize=(4,3))
    sns.countplot(x='comp_score',hue='name',data=df,palette='magma')
    plt.tight_layout()
    st.pyplot(plt, clear_figure=True)
    

    # =========================
    # TOPIC MODELING
    # =========================
    st.subheader("🧠 Topic Modeling")

    vectorizer = CountVectorizer(
        stop_words="english",
        max_df=0.9,
        min_df=5
    )
    dtm = vectorizer.fit_transform(df["chat"])

    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(dtm)

    feature_names = vectorizer.get_feature_names_out()

    for idx, topic in enumerate(lda.components_):
        st.write(
            f"**Topic {idx+1}:**",
            ", ".join([feature_names[i] for i in topic.argsort()[-8:]])
        )

    # =========================
    # WORD CLOUD
    # =========================
    st.subheader("☁️ Word Cloud")

    wc = WordCloud(
        width=800,
        height=400,
        max_font_size=100,
        background_color="white"
    ).generate(" ".join(df["chat"]))
    
    st.image(wc.to_array())
