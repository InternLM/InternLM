"""
Directly load models in internlm format for interactive dialogue.
"""
import logging

import streamlit as st
import torch
from sentencepiece import SentencePieceProcessor

from tools.load_internlm_model import (
    initialize_internlm_model,
    internlm_interactive_generation,
)
from tools.transformers.interface import GenerationConfig

logger = logging.getLogger(__file__)


MODEL_CONFIG_MAP = {
    "internlm-chat-7b": dict(
        checkpoint=False,
        num_attention_heads=32,
        embed_split_hidden=True,
        vocab_size=103168,
        embed_grad_scale=1,
        parallel_output=False,
        hidden_size=4096,
        num_layers=32,
        mlp_ratio=8 / 3,
        apply_post_layer_norm=False,
        dtype="torch.bfloat16",
        norm_type="rmsnorm",
        layer_norm_epsilon=1e-5,
        use_flash_attn=True,
        num_chunks=1,
        use_dynamic_ntk_rope=True,
    ),
    "internlm-chat-7b-v1.1": dict(
        checkpoint=False,
        num_attention_heads=32,
        embed_split_hidden=True,
        vocab_size=103168,
        embed_grad_scale=1,
        parallel_output=False,
        hidden_size=4096,
        num_layers=32,
        mlp_ratio=8 / 3,
        apply_post_layer_norm=False,
        dtype="torch.bfloat16",
        norm_type="rmsnorm",
        layer_norm_epsilon=1e-5,
        use_flash_attn=True,
        num_chunks=1,
        use_dynamic_ntk_rope=True,
    ),
}


def on_btn_click():
    del st.session_state.messages


@st.cache_resource
def load_model():
    model = initialize_internlm_model(
        model_type="INTERNLM",
        ckpt_dir="[Please replace this with the directory where the internlm model weights are stored]",
        # Please change the model here to other models supported by internlm according to your own usage needs.
        model_config=MODEL_CONFIG_MAP["internlm-chat-7b-v1.1"],
        del_model_prefix=True,
    )
    tokenizer = SentencePieceProcessor("tools/V7_sft.model")  # pylint: disable=E1121
    return model, tokenizer


def prepare_generation_config():
    with st.sidebar:
        max_length = st.slider("Max Length", min_value=32, max_value=16000, value=8000)
        top_p = st.slider("Top P", 0.0, 1.0, 0.8, step=0.01)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.01)
        st.button("Clear Chat History", on_click=on_btn_click)

    generation_config = GenerationConfig(max_length=max_length, top_p=top_p, temperature=temperature)

    return generation_config


system_meta_instruction = (
    """<|System|>:You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). """
    + """It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""
)
user_prompt = "<|User|>:{user}\n"
robot_prompt = "<|Bot|>:{robot}<eoa>\n"
cur_query_prompt = "<|User|>:{user}\n<|Bot|>:"


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
    total_prompt = system_meta_instruction + total_prompt + cur_query_prompt.replace("{user}", prompt)
    print(total_prompt)
    return total_prompt


def main():
    # torch.cuda.empty_cache()
    print("load model begin.")
    model, tokenizer = load_model()
    print("load model end.")

    user_avator = "doc/imgs/user.png"
    robot_avator = "doc/imgs/robot.png"

    st.title("InternLM-Chat-7B")

    generation_config = prepare_generation_config()

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
            for cur_response in internlm_interactive_generation(
                model=model,
                tokenizer=tokenizer,
                prompt=real_prompt,
                generation_config=generation_config,
                additional_eos_token_list=[103028],
            ):
                # Display robot response in chat message container
                message_placeholder.markdown(cur_response + "▌")
            message_placeholder.markdown(cur_response)  # pylint: disable=W0631
        # Add robot response to chat history
        st.session_state.messages.append(
            {"role": "robot", "content": cur_response, "avatar": robot_avator}  # pylint: disable=W0631
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
