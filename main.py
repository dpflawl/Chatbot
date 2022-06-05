import streamlit as st
import transformers
#from transformers import pipeline
from transformers import PreTrainedTokenizerFast
import pandas as pd
#from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2Config, GPT2LMHeadModel
import torch
from streamlit_chat import message
import os
import requests
from pathlib import Path
from pytorch_lightning import LightningModule

import tensorflow as tf
import konlpy

                
#st.write(os.getcwd())

st.title("'위로봇👾'과 대화를 나눠보세요.")
st.caption("'위로봇'은 대화에서 감정을 분석하고 이모티콘을 함께 사용하여 감정적인 공감과 위로를 해주는 챗봇입니다.")

def get_text():
    input_text = st.text_input("You: ","안녕?", key="input")
    return input_text 

#input = st.text_input('입력:')
user_input = get_text()

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
    st.session_state.chat_history_ids = None
    
if 'past' not in st.session_state:
    st.session_state['past'] = []

    
U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'   
            
import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

                
@st.cache
def get_obj_det_model_Drive():
    cloud_model_location = "1-EqYjXiygYvJkT6_4peMEN77apMODYA7"
    f_checkpoint = Path("KoGPT2Chatbot.pth")        
    # cloud_model_location = "17jWPP5pbhj67RqIuzIbuBnOznTgG2STy"
    # f_checkpoint = Path("model_-last.ckpt")
    emo_model = tf.keras.models.load_model('model/emotion_model.h5')
    if not f_checkpoint.exists():
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            download_file_from_google_drive(cloud_model_location, f_checkpoint)
    
    #config = GPT2Config()
    #config.pad_token_id = tokenizer.token_to_id('<pad>')

    ##model = GPT2LMHeadModel(config)
    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')   
    ###model_state_dict = model.state_dict()
    checkpoint = torch.load(f_checkpoint)
    for key in list(checkpoint.keys()):
      if 'kogpt2.' in key:
          checkpoint[key.replace('kogpt2.', '')] = checkpoint[key]
          del checkpoint[key]
    #for key in list(checkpoint.keys()):     
    #  torch.reshape(checkpoint[key], (model_state_dict[key].shape[0], model_state_dict[key].shape[1]))     

    model.load_state_dict(checkpoint)
    #model.load_state_dict(f_checkpoint, strict=False)
    #model = GPT2LMHeadModel.load_state_dict(torch.load(f_checkpoint))
    
    model.eval()
    #emo_model.eval()
    return model, emo_model


if user_input:
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
      bos_token='</s>', eos_token='</s>', unk_token='<unk>',
      pad_token='<pad>', mask_token='<mask>')
    #model = GPT2LMHeadModel.load_state_dict(torch.load("/app/chatbot/KoGPT2Chatbot.pth"))
    model, emo_model = get_obj_det_model_Drive()

    with torch.no_grad():
        #new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        new_user_input_ids = tokenizer.encode(tokenizer.unk_token + user_input + '<unused1>' + str(0) + S_TKN + '', return_tensors='pt')
        #tok.encode(U_TKN + q + SENT + sent + S_TKN + a)
        bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_user_input_ids], dim=-1) if 'past' not in st.session_state else new_user_input_ids
        st.session_state.chat_history_ids = model.generate(bot_input_ids,
                                                            max_length=32,
                                                            repetition_penalty=2.0,
                                                            pad_token_id=tokenizer.pad_token_id,
                                                            eos_token_id=tokenizer.eos_token_id,
                                                            bos_token_id=tokenizer.bos_token_id,
                                                            use_cache=True)
        encoded = [tokenizer.encode(user_input)] # 정수 인코딩
        pad_new = tf.keras.preprocessing.sequence.pad_sequences(encoded, maxlen = len(encoded)) # 패딩
        score = float(emo_model.predict(pad_new))
        if (score > 0.9):
          emoji = "😁"
        elif (score > 0.8):
          emoji = "😀"
        elif (score > 0.7):
          emoji = "😊"
        elif (score > 0.5):
          emoji = "🙂"
        elif (score > 0.4):
          emoji = "😶"
        elif (score > 0.3):
          emoji = "😔"
        else:
          emoji = "😢"
        response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)       
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response+emoji)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

#st.subheader('챗봇 답변')
#st.write(f"Chatbot: {response}")
#st.session_state.old_response = response 
  
  
#st.subheader('감정 분석 결과')
#df = pd.DataFrame([[r[0]["label"], r[0]["score"]]], columns=['Label', 'Score'])

# st.write(r[0])
#st.table(df.style.background_gradient(cmap='RdYlGn', subset='Score', vmin=0., vmax=1.))
