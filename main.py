import streamlit as st
import numpy as np
import requests
from PIL import Image
import matplotlib.pyplot as plt
import io
from openai import OpenAI
import base64
from PIL import Image
from streamlit_cropper import st_cropper

import tiktoken
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
    
def main():

    st.title("💧 _KGT Chatbot_ ")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    cont=st.container()
    with st.sidebar:
        openai_api_key = st.secret['api_key']
        genre = st.radio(
            "Chat Type",
            ["***Text Generation***","***Document based***", "***Imaga Generation***", "***Vision***", "***Image Edit***"],
            index=0,
        )
        st.divider()
        if genre == "***Text Generation***":
            st.write('')
        elif genre == "***Document based***":
            uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx'], accept_multiple_files=True)
            if uploaded_files:
                with st.spinner("Thinking..."):
                    files_text = get_text(uploaded_files)
                    text_chunks = get_text_chunks(files_text)
                    vetorestore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vetorestore, openai_api_key)
                    st.session_state.processComplete = True
        elif genre == "***Imaga Generation***":       
            st.write('')     
        elif genre == "***Vision***":  
            uploaded_image = st.file_uploader("Upload your Image", type=['png', 'jpg', 'jpeg', 'gif'], accept_multiple_files=False)            
            # st.write(uploaded_image)
            if uploaded_image:
                st.image(uploaded_image)
                base64_image=base64.b64encode(uploaded_image.read()).decode('utf-8')
                headers = {
                  "Content-Type": "application/json",
                  "Authorization": f"Bearer {openai_api_key}"
                }
        elif genre == "***Image Edit***": 
            img_file = st.file_uploader("Upload your Image", type=['png'], accept_multiple_files=False) 
            if img_file:
                img = Image.open(img_file)
                img_crop = Image.open(img_file)
                with cont:                    
                    rect = st_cropper(
                        img_crop,
                        realtime_update=True,
                        # box_color=box_color,
                        # aspect_ratio=aspect_ratio,
                        return_type='box',
                        # stroke_width=stroke_width
                    )    
                left, top, width, height = tuple(map(int, rect.values()))
                
                alpha = Image.new("L", img.size)
                alpha.paste(255 ,(0,0,alpha.size[0],alpha.size[1]))
                alpha.paste(0, (left,top,left+width,top+height))
                img_crop.putalpha(alpha)
                # st.image(img_crop)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "안녕하세요? 궁금한 것이 있으면 물어보세요"}]

    for msg in st.session_state.messages:
        if msg["content"][0:4]=='http':
            st.chat_message(msg["role"]).image(msg["content"])
        else:
            st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        client = OpenAI(api_key=openai_api_key)
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.spinner("Thinking..."):
            if genre == "***Imaga Generation***":
                response=client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    n=1,
                    size="1024x1024")
                st.session_state.messages.append({"role": "assistant", "content": response.data[0].url})
                st.chat_message("assistant").image(response.data[0].url)
            elif genre == "***Vision***":
                payload = {
                  "model": "gpt-4-vision-preview",
                  "messages": [
                    {
                      "role": "user",
                      "content": [
                        {
                          "type": "text",
                          "text": prompt
                        },
                        {
                          "type": "image_url",
                          "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                          }
                        }
                      ]
                    }
                  ],
                  "max_tokens": 1500
                }
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                st.session_state.messages.append({"role": "assistant", "content": response.json()['choices'][0]['message']['content']})
                st.chat_message("assistant").write(response.json()['choices'][0]['message']['content'])
            elif genre == "***Document based***":
                # response = client.chat.completions.create(model="gpt-4", messages=st.session_state.messages)
                # msg = response.choices[0].message.content
                # st.session_state.messages.append({"role": "assistant", "content": msg})
                # st.chat_message("assistant").write(msg)

                chain = st.session_state.conversation
                with st.spinner("Thinking..."):
                    result = chain({"question": prompt})
                    # with get_openai_callback() as cb:
                    #     st.session_state.chat_history = result['chat_history']
                    response = result['answer']
                    source_documents = result['source_documents']

                    # st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.chat_message("assistant").write(response)
                with st.expander("참고 문서 확인"):
                    st.markdown(source_documents[0].metadata['source'], help=source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help=source_documents[1].page_content)
            elif genre == "***Image Edit***":
                img.save('img.png')
                img_crop.save('img_crop.png')
                img=open('img.png','rb')
                img_crop=open('img_crop.png','rb')
                response=client.images.edit(
                  image=img,
                  mask=img_crop,
                  prompt=prompt,
                  n=1,
                )
                st.session_state.messages.append({"role": "assistant", "content": response.data[0].url})
                st.chat_message("assistant").image(response.data[0].url)
            else:                    
                response = client.chat.completions.create(model="gpt-4", messages=st.session_state.messages)
                msg = response.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.chat_message("assistant").write(msg)
                
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


def get_text(docs):
    doc_list = []

    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb


def get_conversation_chain(vetorestore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-4', temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vetorestore.as_retriever(search_type='mmr', vervose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )

    return conversation_chain

if __name__ == '__main__':
    main()
