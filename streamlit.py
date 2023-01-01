import streamlit as st

from predict import get_prediction

import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

if __name__ == "__main__":
    add_bg_from_local("artifacts/WordCloud.png")
    new_title = '<p style="font-family:sans-serif; color:White; font-size: 42px;">Sentiment Analysis</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    txt = st.text_area('Enter the text')
    if st.button("Predict"):
        if len(txt) == 0:
            st.error("Please! Enter the Text")
        else:
            response = get_prediction(txt)

            if(response[0]==-1):
                st.error(response[1])
            else:
                pred = response[1]
                sentiment= 'Negatvie'
                if pred>0.5:
                    sentiment = 'Positive'

                result = f'<p style="font-family:sans-serif; color:White; font-size: 16px;">\
                        Positive = {round(pred, 2)}<br>Negative = {round(1-pred, 2)}\
                            <br>Sentiment = {sentiment}</p>'
                st.markdown(result, unsafe_allow_html=True)

