import os
import shutil
import zipfile

import streamlit as st

from backend.helpers import Trainer

st.set_page_config(
    page_title='Dataset Tool | Samsus Superbrain',
    page_icon='🧊',
    initial_sidebar_state='auto',
)

st.title('Dataset Uploader')

st.divider()

st.header('Upload Dataset')

uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True, type=['zip', 'docx', 'txt', 'csv', 'md', 'pdf',])

st.caption('Only .zip, .txt, .csv, .docx, .md and .pdf files are allowed.')

train_button = st.button('Train', key='train_button')

if train_button and uploaded_files:

    st.divider()
    st.header('Uploading files')
    progress_text = "Uploading Files."
    my_bar = st.progress(0, text=progress_text)

    files_uploaded_amount = 0
    files_uploaded = []

    with st.spinner('Uploading...'):
        for uploaded_file in uploaded_files:
            store_path = os.path.join('training', 'datasets')

            if not os.path.exists(store_path):
                os.makedirs(store_path)

            if uploaded_file.name.endswith('.zip'):
                with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                        zip_ref.extractall(store_path)

                for root, dirs, files in os.walk(store_path):
                    for dir in dirs:
                        shutil.rmtree(os.path.join(root, dir))
            else:
                bytes_data = uploaded_file.read()
                with open(os.path.join(store_path, uploaded_file.name), "wb") as f:
                        f.write(bytes_data)

            for root, dirs, files in os.walk(store_path):
                for file in files:
                    if not file.endswith('.csv') and not file.endswith('.txt') and not file.endswith('.md') and not file.endswith('.pdf'):
                        os.remove(os.path.join(root, file))

            files_uploaded.append(uploaded_file.name)
            files_uploaded_amount += 1
            my_bar.progress(files_uploaded_amount / len(uploaded_files), text=progress_text)

            print("File uploaded: " + uploaded_file.name,)

        st.subheader('Files uploaded:')
        st.code('\n'.join('{}: {}'.format(*k) for k in enumerate(uploaded_files)))


        st.divider()
        st.header('Training')

        files_uploaded_amount = 0
        files_uploaded = []

        trainer = Trainer()

        with st.spinner(text='Training...'):
            result = trainer.train()

        st.write(result)

