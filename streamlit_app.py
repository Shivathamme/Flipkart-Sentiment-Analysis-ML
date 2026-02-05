import streamlit as st
import requests

st.set_page_config(page_title="Flipkart Sentiment Analysis")

st.title("🛒 Flipkart Review Sentiment Analysis")
st.write("Enter a product review to predict its sentiment.")

review_text = st.text_area("Enter Review Text")

if st.button("Predict Sentiment"):
    if review_text.strip() == "":
        st.warning("Please enter a review.")
    else:
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                params={"review": review_text}
            )

            if response.status_code == 200:
                result = response.json()
                st.success(f"Predicted Sentiment: **{result['sentiment']}**")
                st.write(f"Confidence: {result['confidence']}")
            else:
                st.error("Error from API")

        except Exception as e:
            st.error("FastAPI server is not running")
