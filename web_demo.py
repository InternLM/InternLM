"""
This script refers to the dialogue example of streamlit, the interactive generation code of chatglm2 and transformers. We mainly modified part of the code logic to adapt to the generation of our model.
Please refer to these links below for more information:
    1. streamlit chat example: https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
    2. chatglm2: https://github.com/THUDM/ChatGLM2-6B
    3. transformers: https://github.com/huggingface/transformers
"""

import argparse
import streamlit as st
import torch
from dataclasses import dataclass, asdict
from typing import List, Optional, Callable, Optional
import copy
import warnings
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList

from tools.transformers.interface import generate_interactive, GenerationConfig

logger = logging.get_logger(__name__)


def on_btn_click():
    del st.session_state.messages

@st.cache_resource
def load_model(model_path, max_position_embeddings):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, max_position_embeddings=max_position_embeddings).to(torch.bfloat16).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


def prepare_generation_config(max_value):
    with st.sidebar:
        max_length = st.slider("Max Length", min_value=32, max_value=max_value, value=2048)
        top_p = st.slider(
            'Top P', 0.0, 1.0, 0.8, step=0.01
        )
        temperature = st.slider(
            'Temperature', 0.0, 1.0, 0.7, step=0.01
        )
        st.button("Clear Chat History", on_click=on_btn_click)
    
    generation_config = GenerationConfig(
        max_length=max_length,
        top_p=top_p,
        temperature=temperature
    )
    
    return generation_config


user_prompt = "<|User|>:{user}<eoh>\n"
robot_prompt = "<|Bot|>:{robot}<eoa>\n"
cur_query_prompt = "<|User|>:{user}<eoh>\n<|Bot|>:"


def combine_history(prompt):
    messages = st.session_state.messages
    total_prompt = ""
    for message in messages:
        cur_content = message["content"]
        if message["role"] == "user":
            cur_prompt = user_prompt.replace("{user}", cur_content)
        elif message["role"] == "robot":
            cur_prompt = robot_prompt.replace("{robot}", cur_content)
        else:
            raise RuntimeError
        total_prompt += cur_prompt
    total_prompt = total_prompt + cur_query_prompt.replace("{user}", prompt)
    return total_prompt

def parse_args():
    parser = argparse.ArgumentParser(description='Chat with the model, please add "--" between web_demo.py and the args in terminal')
    parser.add_argument('--model', default='internlm/internlm-chat-7b', help='Type of InternLM')
    parser.add_argument('--max_value', default=2048, type=int, help='The max length of the generated text')
    return parser.parse_args()

  
def main(args):
    torch.cuda.empty_cache()
   
    print("load model begin.")
    model, tokenizer = load_model(args.model, max_position_embeddings=args.max_value)
    print("load model end.")
    
    user_avator = "doc/imgs/user.png"
    robot_avator = "doc/imgs/robot.png"
    
    title = args.model.split("/")[-1] if 'internlm' in args.model.split("/")[-1] else 'internlm-7b'
    st.title(title[0].upper() + title[1:])
    
    generation_config = prepare_generation_config(args.max_value)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user", avatar=user_avator):
            st.markdown(prompt)
        real_prompt = combine_history(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt, "avatar": user_avator})

        with st.chat_message("robot", avatar=robot_avator):
            message_placeholder = st.empty()
            for cur_response in generate_interactive(model=model, tokenizer=tokenizer, prompt=real_prompt, additional_eos_token_id=103028, **asdict(generation_config)):
                # Display robot response in chat message container
                message_placeholder.markdown(cur_response + "▌")
            message_placeholder.markdown(cur_response)
        # Add robot response to chat history
        st.session_state.messages.append({"role": "robot", "content": cur_response, "avatar": robot_avator})
        torch.cuda.empty_cache()
        

if __name__ == "__main__":
    args = parse_args()
    main(args)
