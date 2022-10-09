import streamlit as st
import os
from PIL import Image
from app_funcs import *


st.set_page_config(
    page_title="ISR using EDSR & SwinR",
    page_icon="🦈",
    layout="centered",
    initial_sidebar_state="auto",
)

upload_path = os.path.join(os.getcwd(), "streamlit", "uploads/")
download_path = os.path.join(os.getcwd(), "streamlit", "downloads/")

st.title("ISR using EDSR & SwinR")

st.info('Support image formats - PNG')
uploaded_file = st.file_uploader("Upload Image", type="png")

if uploaded_file is not None:
    with open(os.path.join(upload_path, uploaded_file.name), "wb") as f:
        f.write((uploaded_file).getbuffer())
    with st.spinner(f"Working..."):
        uploaded_image = os.path.join(upload_path, uploaded_file.name)
        downloaded_image = os.path.join(
            download_path, str("output_" + uploaded_file.name))
        input_image = Image.open(uploaded_image)
        print("Opening ", input_image)
        st.markdown("---")
        st.image(
            input_image, caption=f'Input Image / width : {input_image.size[0]}, height : {input_image.size[1]}')
        model = instantiate_model()
        image_super_resolution(uploaded_image, downloaded_image, model)
        print("Output Image: ", downloaded_image)
        final_image = Image.open(downloaded_image)
        print("Opening ", final_image)
        st.markdown("---")
        st.image(
            final_image, caption=f'Final Image / width : {final_image.size[0]}, height : {final_image.size[1]}')
        with open(downloaded_image, "rb") as file:
            if uploaded_file.name.endswith('.png') or uploaded_file.name.endswith('.PNG'):
                if st.download_button(
                        label="Download Output Image",
                        data=file,
                        file_name=str("output_"+uploaded_file.name),
                        mime='image/png'):
                    downloaded_success()
            else:
                st.warning('Only PNG file can be uploaded')
else:
    st.warning('Please upload your Image file')

st.markdown("<br><hr><center>Made with Team D of DSL Modeling Proejct <br> <a href='https://github.com/MyeongheonChoi/Image_Super_Resolution-22_Fall_ModelingProject'><strong>Github</strong></a></center><hr>", unsafe_allow_html=True)
