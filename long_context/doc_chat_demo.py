import argparse
import logging
from dataclasses import dataclass

import streamlit as st
from magic_doc.docconv import DocConverter
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class GenerationConfig:
    # this config is used for chat to provide more diversity
    max_tokens: int = 1024
    top_p: float = 1.0
    temperature: float = 0.1
    repetition_penalty: float = 1.005


def generate(
    client,
    messages,
    generation_config,
):
    stream = client.chat.completions.create(
        model=st.session_state['model_name'],
        messages=messages,
        stream=True,
        temperature=generation_config.temperature,
        top_p=generation_config.top_p,
        max_tokens=generation_config.max_tokens,
        frequency_penalty=generation_config.repetition_penalty,
    )
    return stream


def prepare_generation_config():
    with st.sidebar:
        max_tokens = st.number_input('Max Tokens',
                                     min_value=100,
                                     max_value=4096,
                                     value=1024)
        top_p = st.number_input('Top P', 0.0, 1.0, 1.0, step=0.01)
        temperature = st.number_input('Temperature', 0.0, 1.0, 0.05, step=0.01)
        repetition_penalty = st.number_input('Repetition Penalty',
                                             0.8,
                                             1.2,
                                             1.02,
                                             step=0.001,
                                             format='%0.3f')
        st.button('Clear Chat History', on_click=on_btn_click)

    generation_config = GenerationConfig(max_tokens=max_tokens,
                                         top_p=top_p,
                                         temperature=temperature,
                                         repetition_penalty=repetition_penalty)

    return generation_config


def on_btn_click():
    del st.session_state.messages
    st.session_state.file_content_found = False
    st.session_state.file_content_used = False


user_avator = 'assets/user.png'
robot_avator = 'assets/robot.png'

st.title('InternLM2.5 File Chat 📝')


def main(base_url):
    # Initialize the client for the model
    client = OpenAI(base_url=base_url, timeout=12000)

    # Get the model ID
    model_name = client.models.list().data[0].id
    st.session_state['model_name'] = model_name

    # Get the generation config
    generation_config = prepare_generation_config()

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'file_content_found' not in st.session_state:
        st.session_state.file_content_found = False
        st.session_state.file_content_used = False
        st.session_state.file_name = ''

    # Handle file upload
    if not st.session_state.file_content_found:
        uploaded_file = st.file_uploader('Upload an article',
                                         type=('txt', 'md', 'pdf'))
        file_content = ''
        if uploaded_file is not None:
            if uploaded_file.type == 'application/pdf':
                with open('uploaded_file.pdf', 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                converter = DocConverter(s3_config=None)
                file_content, time_cost = converter.convert(
                    'uploaded_file.pdf', conv_timeout=300)
                # Reset flag when a new file is uploaded
                st.session_state.file_content_found = True
                # Store the file content in session state
                st.session_state.file_content = file_content
                # Store the file name in session state
                st.session_state.file_name = uploaded_file.name
            else:
                file_content = uploaded_file.read().decode('utf-8')
                # Reset flag when a new file is uploaded
                st.session_state.file_content_found = True
                # Store the file content in session state
                st.session_state.file_content = file_content
                # Store the file name in session state
                st.session_state.file_name = uploaded_file.name

    if st.session_state.file_content_found:
        st.success(f"File '{st.session_state.file_name}' "
                   'has been successfully uploaded!')

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message['role'], avatar=message.get('avatar')):
            st.markdown(message['content'])

    # Handle user input and response generation
    if prompt := st.chat_input("What's up?"):
        turn = {'role': 'user', 'content': prompt, 'avatar': user_avator}
        if (st.session_state.file_content_found
                and not st.session_state.file_content_used):
            assert st.session_state.file_content is not None
            merged_prompt = f'{st.session_state.file_content}\n\n{prompt}'
            # Set flag to indicate file content has been used
            st.session_state.file_content_used = True
            turn['merged_content'] = merged_prompt

        st.session_state.messages.append(turn)
        with st.chat_message('user', avatar=user_avator):
            st.markdown(prompt)

        with st.chat_message('assistant', avatar=robot_avator):
            messages = [{
                'role':
                m['role'],
                'content':
                m['merged_content'] if 'merged_content' in m else m['content'],
            } for m in st.session_state.messages]
            # Log messages to the terminal
            for m in messages:
                logging.info(
                    f"\n\n*** [{m['role']}] ***\n\n\t{m['content']}\n\n")
            stream = generate(client, messages, generation_config)
            response = st.write_stream(stream)
        st.session_state.messages.append({
            'role': 'assistant',
            'content': response,
            'avatar': robot_avator
        })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run Streamlit app with OpenAI client.')
    parser.add_argument('--base_url',
                        type=str,
                        required=True,
                        help='Base URL for the OpenAI client')
    args = parser.parse_args()
    main(args.base_url)
