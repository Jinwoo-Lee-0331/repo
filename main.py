import streamlit as st
import numpy as np
import requests
from PIL import Image
import io
from openai import OpenAI
import base64
from PIL import Image
from streamlit_cropper import st_cropper
import pandas as pd

from audio_recorder_streamlit import audio_recorder
from pydub import AudioSegment
from io import BytesIO
from io import StringIO
    
def main():
    st.set_page_config(
        page_title="CHATKGT",  # String or None. Strings get appended with "• Streamlit".
        # layout="wide",
        # page_icon="💧",  # Can be "centered" or "wide". In the future also "dashboard", etc.
        # initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
        page_icon="./data/ci.png",  # String, anything supported by st.image, or None.
    )
    st.markdown(
        """
        <style>
            .block-container {
                    padding-top: 0rem;
                }
            .sidebar .sidebar-content {
            background-image: linear-gradient(#2e7bcf,#2e7bcf);
            color: white;
            }              
            .css-1aumxhk {
            background-color: #011839;
            background-image: none;
            color: #ffffff
            }
            section[data-testid="stSidebar"] {
                width: 420px !important; # Set the width to your desired value
                background-color: white;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.title("💧 _KGT Chatbot_ ")
    if "thread" not in st.session_state:
        st.session_state.thread = None
        
    if "asst_list" not in st.session_state:
        st.session_state.asst_list = None

    if "asst" not in st.session_state:
        st.session_state.asst = None

    if "client" not in st.session_state:
        st.session_state.client = None
        
    if "key_tab" not in st.session_state:
        st.session_state.key_tab = None

    cont=st.container()
    with st.sidebar:                        
        gr, tg = st.columns([3,2])
        with gr:
            genre = st.radio(
                "Chat Type",
                ["***Text Generation***","***Document based***", "***Vision***", "***Imaga Generation***", "***Image Edit***", "***Text to Speech***", "***Speech to Text***"],
                captions=['일반적인 TEXT 답변','문서 기반 답변', '이미지 기반 답변', '이미지 생성(DALL-E-3)', '이미지 편집(DALL-E-2)', '음성 생성', '음성 인식'],
                index=0,
            )                
        with tg:
            st.session_state.key_tab=pd.read_csv('./data/api_key_table.csv')  
            select_key = st.selectbox("Key Index",st.session_state.key_tab.key_index.to_list())
            pw = st.text_input("password",'',type='password')
            # togg=st.toggle("***Record Chat***", help="⚠️ 회사 네트워크 보안 제한 - 외부 네트워크에서 사용해주세요")
        st.divider()
        
        if st.session_state.key_tab.loc[st.session_state.key_tab['key_index'] == select_key,'password'].reset_index(drop=True)[0].astype('str') == pw:
            openai_api_key = st.secrets[select_key]
        else:
            openai_api_key = None
            st.error("key index or password is wrong")

        if openai_api_key is not None:
            st.session_state.client = OpenAI(api_key=openai_api_key)
            thrd= st.session_state.key_tab.loc[st.session_state.key_tab['key_index'] == select_key,'thread'].reset_index(drop=True)[0]
            # st.write(thrd)
            st.session_state.thread = st.session_state.client.beta.threads.retrieve(thread_id=thrd)
            # st.session_state.thread = st.session_state.client.beta.threads.create()
            my_assistants = st.session_state.client.beta.assistants.list()
            st.session_state.asst_list={}
            for i in range(len(my_assistants.data)):
                st.session_state.asst_list[my_assistants.data[i].name]=my_assistants.data[i].id
        
    ###############################################################
        
        if genre == "***Text Generation***":
            mdl = st.radio("Model",['gpt-3.5-turbo', 'gpt-4', 'gpt-4o'],index=0)
            system_prompt = st.text_area(
            "System Prompt",
            "당신은 유능한 조수입니다. 당신은 구글에서 정보를 찾을 수 있고 그 정보를 바탕으로 답변을 해줄 수 있습니다. 관련 정보를 찾으면 해당 url과 사진 등을 제시해주세요.",
                height = 50,
                help = "CHATGPT의 특성 및 대답 형식을 설정"
            )
            temp = st.slider("Temperature",0.0,1.0,0.5, help="수치가 작을수록 정형화된 대답, 수치가 커질수록 창의적인 대답이 나옵니다.")
            st.markdown("""ℹ️
            [프롬프트 작성 Tip](https://platform.openai.com/docs/guides/prompt-engineering)
            """)
            
        elif genre == "***Imaga Generation***":       
            # st.caption("⚠️ 회사 네트워크에서 보안 제한 - 외부 네트워크에서 사용해주세요")     
            st.caption("원하는 이미지에 대한 설명을 해주세요")     
        elif genre == "***Text to Speech***":     
            voice = st.radio("Voice",['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'],index=0)
        elif genre == "***Document based***":
            system_prompt = st.text_area(
            "System Prompt",
            "당신은 유능한 조수입니다. 당신은 질문에 답변하기 위해 파일에 액세스 할 수 있습니다. 항상 파일들의 정보를 기반으로 응답하세요. 관련 정보를 찾으면 해당 정보(해당 내용의 페이지 위치나 문단)와 사진 등을 제시하면 좋습니다.",
                height = 50,
                help = "CHATGPT의 특성 및 대답 형식을 설정"
            )
            temp = st.slider("Temperature",0.0,1.0,0.5, help="수치가 작을수록 정형화된 대답, 수치가 커질수록 창의적인 대답이 나옵니다.")
            st.divider()
            dbc1, dbc2 = st.columns([1,1])
            with dbc2:
                asst_name = st.text_input('Your Assistant ID', '')
                db_button = st.button('Create Assistant!',use_container_width=True, help="2134")
            with dbc1:
                uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx'], accept_multiple_files=True)
            
            if db_button:
                if uploaded_files:
                    file_id=[]
                    for doc in uploaded_files:
                        openai_file = st.session_state.client.files.create(
                            file=doc,
                            purpose="assistants"
                        )
                        file_id.append(openai_file.id)
                        
                    if asst_name in st.session_state.asst_list:
                        st.session_state.asst = st.session_state.client.beta.assistants.update(
                          st.session_state.asst_list[asst_name],
                          instructions=system_prompt,
                          name=asst_name,
                          tools=[{"type": "retrieval"}],
                          model="gpt-4-turbo-preview",
                          file_ids=file_id,
                        )
                    else:
                        st.session_state.asst = st.session_state.client.beta.assistants.create(
                            instructions=system_prompt,
                            name=asst_name,
                            tools=[{"type": "retrieval"}],
                            model="gpt-4-turbo-preview",
                          file_ids=file_id,
                        )                    
                else:
                    st.warning("파일을 업로드하고 어시스턴트 id를 입력해주세요")
        elif genre == "***Vision***":            
            vision_radio=st.radio("How to take your image",['Upload','Camera'])
            if vision_radio=='Camera':
                uploaded_image = st.camera_input("Take your Image")
            elif vision_radio=='Upload':
                uploaded_image = st.file_uploader("Upload your Image", type=['png', 'jpg', 'jpeg', 'gif'], accept_multiple_files=False)  
            if uploaded_image:
                st.image(uploaded_image)
                base64_image=base64.b64encode(uploaded_image.read()).decode('utf-8')
                headers = {
                  "Content-Type": "application/json",
                  "Authorization": f"Bearer {openai_api_key}"
                }
        elif genre == "***Image Edit***":            
            ie_radio=st.radio("How to take your image",['Upload','Camera'])
            if ie_radio=='Camera':
                img_file = st.camera_input("Take your Image")
            elif ie_radio=='Upload':
                img_file = st.file_uploader("Upload your Image", type=['png'], accept_multiple_files=False) 
            # img_file = st.file_uploader("Upload your Image", type=['png'], accept_multiple_files=False) 
            # img_file = st.camera_input("Upload your Image")
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
            stt_radio=st.radio("How to take your audio",['Upload','Record'])
            audio_bytes=[]
            uploaded_audio=[]
            if stt_radio=='Record':
                audio_bytes = audio_recorder(
                    text="Click to Record",
                    recording_color="#e8b62c",
                    neutral_color="#6aa36f",
                    # icon_name="user",
                    icon_size="2x",
                )
            elif stt_radio=='Upload':
                uploaded_audio = st.file_uploader("Upload your Audio", type=['mp3','wav','mpeg','webm','mp4','m4a'], accept_multiple_files=False)
            
    # st.chat_message("assistant").write("안녕하세요? 궁금한 것이 있으면 물어보세요")
    
    if genre == "***Speech to Text***":
        if audio_bytes:
            st.chat_message("user").audio(audio_bytes)
            audio_bytes = AudioSegment(data=audio_bytes).export('./test.wav',format='wav')
            transcript = st.session_state.client.audio.transcriptions.create(
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
            transcript = st.session_state.client.audio.transcriptions.create(
              model="whisper-1",
              file=uploaded_audio
            )
            msg = transcript.text
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)

    ############# session_state ###############
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "안녕하세요? 궁금한 것이 있으면 물어보세요"}]
    # if togg:
    if "messages" in st.session_state:
        for messages in st.session_state.messages:
            if messages["content"][0]==255:
                st.chat_message(messages["role"]).audio(messages["content"])
            elif messages["content"][0:4]=='http':
                st.chat_message(messages["role"]).image(messages["content"])
            else:
                st.chat_message(messages["role"]).write(messages["content"])

    ############# chat ###############
    
    if prompt := st.chat_input():
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        with st.spinner("Thinking..."):
            if genre == "***Imaga Generation***":
                response=st.session_state.client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    n=1,
                    size="1024x1024")
                st.session_state.messages.append({"role": "assistant", "content": response.data[0].url})
                st.chat_message("assistant").image(response.data[0].url)
                
            elif genre == "***Vision***":
                payload = {
                  "model": "gpt-4o",
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
                if st.session_state.asst is not None:
                    thread_message = st.session_state.client.beta.threads.messages.create(
                      st.session_state.thread.id,
                      role="user",
                      content=prompt,
                    )
    
                    run = st.session_state.client.beta.threads.runs.create(
                        thread_id=st.session_state.thread.id,
                        assistant_id=st.session_state.asst.id
                    )                
                    import time
                    
                    while True:
                        run = st.session_state.client.beta.threads.runs.retrieve(
                            thread_id=st.session_state.thread.id,
                            run_id=run.id
                        )
                        if run.status == "completed":
                            break
                        else:
                            time.sleep(1)
    
                    thread_messages = st.session_state.client.beta.threads.messages.list(st.session_state.thread.id)
                    st.session_state.messages.append({"role": "assistant", "content": thread_messages.data[0].content[0].text.value})
                    st.chat_message("assistant").write(thread_messages.data[0].content[0].text.value)
                else:
                    st.warning("파일을 업로드 및 Assistant ID를 입력하고 'Creat Assistant' 버튼을 누르세요")
                    
            elif genre == "***Image Edit***":    
                byte_img = io.BytesIO()
                img.save(byte_img, format='png')
                byte_img_crop = io.BytesIO()
                img_crop.save(byte_img_crop, format='png')
                response=st.session_state.client.images.edit(
                  image=byte_img,
                  mask=byte_img_crop,
                  prompt=prompt,
                    # size = f"{img.size[0]}x{img.size[1]}",
                  n=1,
                )
                st.session_state.messages.append({"role": "assistant", "content": response.data[0].url})
                st.chat_message("assistant").image(response.data[0].url)
                
            elif genre == "***Text to Speech***":
                response = st.session_state.client.audio.speech.create(
                    model="tts-1-hd",
                    voice=voice,
                    input=prompt,
                )
                msg = response.read()
                st.session_state.messages.append({"role": "assistant", "content": msg, "genre": 'tts'})
                st.chat_message("assistant").audio(msg)

            elif genre == "***Speech to Text***":
                transcript = st.session_state.client.audio.transcriptions.create(
                  model="whisper-1",
                  file=audio_bytes.read()
                )
                msg = transcript.text
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.chat_message("assistant").write(msg)
                
            else:                
                st.session_state.asst = st.session_state.client.beta.assistants.update(
                          'asst_tT5FZJESlwnysflUHlBwNC0x',
                          instructions=system_prompt,
                          name='kgt',
                          # tools=[{"type": "retrieval"}],
                          model = 'gpt-4o',
                          # file_ids=file_id,
                        )
                thread_message = st.session_state.client.beta.threads.messages.create(
                      st.session_state.thread.id,
                      role="user",
                      content=prompt,
                )

                run = st.session_state.client.beta.threads.runs.create(
                    thread_id=st.session_state.thread.id,
                    assistant_id=st.session_state.asst.id
                )                
                import time
                
                while True:
                    run = st.session_state.client.beta.threads.runs.retrieve(
                        thread_id=st.session_state.thread.id,
                        run_id=run.id
                    )
                    if run.status == "completed":
                        break
                    else:
                        time.sleep(1)

                thread_messages = st.session_state.client.beta.threads.messages.list(st.session_state.thread.id)
                st.session_state.messages.append({"role": "assistant", "content": thread_messages.data[0].content[0].text.value})
                st.chat_message("assistant").write(thread_messages.data[0].content[0].text.value)
                
                # response = st.session_state.client.chat.completions.create(model="gpt-4", messages=[
                #     {"role": "system", "content":system_prompt},
                #     {"role": "user", "content": prompt}],
                #     temperature = temp)
                # msg = response.choices[0].message.content
                # st.session_state.messages.append({"role": "assistant", "content": msg})
                # st.chat_message("assistant").write(msg)

if __name__ == '__main__':
    main()
