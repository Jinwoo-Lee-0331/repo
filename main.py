import streamlit as st
import numpy as np
import requests
from PIL import Image
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

from audio_recorder_streamlit import audio_recorder
from pydub import AudioSegment
from io import BytesIO
    
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
        openai_api_key = st.secrets["api_key"]
        gr, tg = st.columns([3,2])
        with gr:
            genre = st.radio(
                "Chat Type",
                ["***Text Generation***","***Document based***", "***Vision***", "***Imaga Generation***", "***Image Edit***", "***Text to Speech***", "***Speech to Text***"],
                captions=['일반적인 TEXT 답변','문서 기반 답변', '이미지 기반 답변', '이미지 생성(DALL-E-3)', '이미지 편집(DALL-E-2)', '음성 생성', '음성 인식'],    
                index=0,
            )            
                
        with tg:
            st.markdown(" ")
            st.markdown(" ")
            togg=st.toggle("***Record Chat***")
            st.markdown("", help="회사망 사용 불가")
        st.divider()
    
        if genre == "***Text Generation***":
            st.write('')
        elif genre == "***Imaga Generation***":       
            st.write('')     
        elif genre == "***Text to Speech***":     
            voice = st.radio("Voice",['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'],index=0)
        elif genre == "***Document based***":
            uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx'], accept_multiple_files=True)
            if uploaded_files:
                with st.spinner("Thinking..."):
                    files_text = get_text(uploaded_files)
                    text_chunks = get_text_chunks(files_text)
                    vetorestore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vetorestore, openai_api_key)
                    st.session_state.processComplete = True
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
                # st.write(f"{img.size[0]}x{img.size[1]}")
        elif genre == "***Speech to Text***": 
            audio_bytes = audio_recorder(
                    text="Click to Record",
                    recording_color="#e8b62c",
                    neutral_color="#6aa36f",
                    # icon_name="user",
                    icon_size="2x",
                )
            uploaded_audio = st.file_uploader("Upload your Audio", type=['mp3','wav','mpeg','webm','mp4','m4a'], accept_multiple_files=False)
            
    st.chat_message("assistant").write("안녕하세요? 궁금한 것이 있으면 물어보세요")
    
    if genre == "***Speech to Text***": 
        if audio_bytes:
            st.chat_message("user").audio(audio_bytes)
            audio_bytes = AudioSegment(data=audio_bytes).export('./test.wav',format='wav')
    
            client = OpenAI(api_key=openai_api_key)                
            transcript = client.audio.transcriptions.create(
              model="whisper-1",
              file=audio_bytes,
                language='ko',
                temperature=0.1
            )
            msg = transcript.text
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)        
        
        if uploaded_audio:
            st.chat_message("user").audio(uploaded_audio, format="audio/wav") 
    
            client = OpenAI(api_key=openai_api_key)                
            transcript = client.audio.transcriptions.create(
              model="whisper-1",
              file=uploaded_audio
            )
            msg = transcript.text
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)      

    ############# session_state ###############
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "안녕하세요? 궁금한 것이 있으면 물어보세요"}]
    if togg:
        if "messages" in st.session_state:
            for messages in st.session_state.messages:
                if len(messages["content"]) > 5000:
                    st.chat_message(messages["role"]).audio(messages["content"])   
                elif messages["content"][0:4]=='http':
                    st.chat_message(messages["role"]).image(messages["content"])
                else:
                    st.chat_message(messages["role"]).write(messages["content"])

    ############# chat ###############
    
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
                
                # img.save('img.png')
                # img_crop.save('img_crop.png')
                # img=open(img,'rb')
                # img_crop=open(img_crop,'rb')
                byte_img = io.BytesIO()
                img.save(byte_img, format='png')
                byte_img_crop = io.BytesIO()
                img_crop.save(byte_img_crop, format='png')
                response=client.images.edit(
                  image=byte_img,
                  mask=byte_img_crop,
                  prompt=prompt,
                    # size = f"{img.size[0]}x{img.size[1]}",
                  n=1,
                )
                st.session_state.messages.append({"role": "assistant", "content": response.data[0].url})
                st.chat_message("assistant").image(response.data[0].url)
                
            elif genre == "***Text to Speech***":
                response = client.audio.speech.create(
                    model="tts-1-hd",
                    voice=voice,
                    input=prompt,
                )
                msg = response.read()
                st.session_state.messages.append({"role": "assistant", "content": msg, "genre": 'tts'})
                st.chat_message("assistant").audio(msg)

            elif genre == "***Speech to Text***":
                transcript = client.audio.transcriptions.create(
                  model="whisper-1",
                  file=audio_bytes.read()
                )
                msg = transcript.text
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.chat_message("assistant").write(msg)
                
            else:
                response = client.chat.completions.create(model="gpt-4", messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}])
                msg = response.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.chat_message("assistant").write(msg)
                

############# document based def ###############

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
