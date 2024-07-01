import pytest
import torch
from auto_gptq.modeling import BaseGPTQForCausalLM
from lmdeploy import TurbomindEngineConfig, pipeline
from PIL import Image
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

prompts = ['你好', "what's your name"]


def assert_model(response):
    assert len(response) != 0
    assert 'UNUSED_TOKEN' not in response
    assert 'Mynameis' not in response
    assert 'Iama' not in response


class TestChat:
    """Test cases for chat model."""

    @pytest.mark.parametrize(
        'model_name',
        [
            'internlm/internlm2_5-7b-chat', 'internlm/internlm2_5-7b-chat-1m',
            'internlm/internlm2-chat-7b', 'internlm/internlm2-chat-7b-sft',
            'internlm/internlm2-chat-20b', 'internlm/internlm2-chat-20b-sft',
            'internlm/internlm2-chat-1_8b', 'internlm/internlm2-chat-1_8b-sft'
        ],
    )
    @pytest.mark.parametrize(
        'usefast',
        [
            True,
            False,
        ],
    )
    def test_demo_default(self, model_name, usefast):
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  trust_remote_code=True,
                                                  use_fast=usefast)
        # Set `torch_dtype=torch.float16` to load model in float16, otherwise
        # it will be loaded as float32 and might cause OOM Error.
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16,
            trust_remote_code=True).cuda()
        model = model.eval()
        history = []
        for prompt in prompts:
            response, history = model.chat(tokenizer, prompt, history=history)
            print(response)
            assert_model(response)

        history = []
        for prompt in prompts:
            length = 0
            for response, history in model.stream_chat(tokenizer,
                                                       prompt,
                                                       history=[]):
                print(response[length:], flush=True, end='')
                length = len(response)
            assert_model(response)


class TestChatAwq:
    """Test cases for chat model."""

    @pytest.mark.parametrize(
        'model_name',
        ['internlm/internlm2-chat-20b-4bits'],
    )
    def test_demo_default(self, model_name):
        engine_config = TurbomindEngineConfig(model_format='awq')
        pipe = pipeline('internlm/internlm2-chat-20b-4bits',
                        backend_config=engine_config)
        responses = pipe(['Hi, pls intro yourself', 'Shanghai is'])
        print(responses)
        for response in responses:
            assert_model(response.text)


class TestBase:
    """Test cases for base model."""

    @pytest.mark.parametrize(
        'model_name',
        [
            'internlm/internlm2_5-7b', 'internlm/internlm2-7b',
            'internlm/internlm2-base-7b', 'internlm/internlm2-20b',
            'internlm/internlm2-base-20b', 'internlm/internlm2-1_8b'
        ],
    )
    @pytest.mark.parametrize(
        'usefast',
        [
            True,
            False,
        ],
    )
    def test_demo_default(self, model_name, usefast):
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  trust_remote_code=True,
                                                  use_fast=usefast)
        # Set `torch_dtype=torch.float16` to load model in float16, otherwise
        # it will be loaded as float32 and might cause OOM Error.
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16,
            trust_remote_code=True).cuda()
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors='pt')
            for k, v in inputs.items():
                inputs[k] = v.cuda()
            gen_kwargs = {
                'max_length': 128,
                'top_p': 10,
                'temperature': 1.0,
                'do_sample': True,
                'repetition_penalty': 1.0,
            }
            output = model.generate(**inputs, **gen_kwargs)
            output = tokenizer.decode(output[0].tolist(),
                                      skip_special_tokens=True)
            print(output)
            assert_model(output)


class TestMath:
    """Test cases for base model."""

    @pytest.mark.parametrize(
        'model_name',
        [
            'internlm/internlm2-math-7b', 'internlm/internlm2-math-base-7b',
            'internlm/internlm2-math-plus-1_8b',
            'internlm/internlm2-math-plus-7b'
        ],
    )
    @pytest.mark.parametrize(
        'usefast',
        [
            True,
            False,
        ],
    )
    def test_demo_default(self, model_name, usefast):
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  trust_remote_code=True,
                                                  use_fast=usefast)
        # Set `torch_dtype=torch.float16` to load model in float16, otherwise
        # it will be loaded as float32 and might cause OOM Error.
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True,
            torch_dtype=torch.float16).cuda()
        model = model.eval()
        model = model.eval()
        response, history = model.chat(tokenizer,
                                       '1+1=',
                                       history=[],
                                       meta_instruction='')
        print(response)
        assert_model(response)
        assert '2' in response

class TestReward:
    """Test cases for base model."""

    @pytest.mark.parametrize(
        'model_name',
        [
            'internlm/internlm-reward-1_8b', 'internlm/internlm-reward-7b',
            'internlm/internlm-reward-20b'
        ],
    )
    @pytest.mark.parametrize(
        'usefast',
        [
            True,
            False,
        ],
    )
    def test_demo_default(self, model_name, usefast):
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  trust_remote_code=True,
                                                  use_fast=usefast)
        model = AutoModel.from_pretrained(model_name, device_map="cuda", 
                                          torch_dtype=torch.float16, 
                                          trust_remote_code=True,)
        tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                                  trust_remote_code=True)

        chat_1 = [
            {"role": "user", "content": "Hello! What's your name?"},
            {"role": "assistant", "content": "My name is InternLM2! A helpful AI assistant. What can I do for you?"}
        ]
        chat_2 = [
            {"role": "user", "content": "Hello! What's your name?"}, 
            {"role": "assistant", "content": "I have no idea."}
        ]

        # get reward score for a single chat
        score1 = model.get_score(tokenizer, chat_1)
        score2 = model.get_score(tokenizer, chat_2)
        print("score1: ", score1)
        print("score2: ", score2)
        assert score1 > 0.5 && score1 < 1 && score2 < 0
        
        # batch inference, get multiple scores at once
        scores = model.get_scores(tokenizer, [chat_1, chat_2])
        print("scores: ", scores)
        assert scores[0] > 0.5 && scores[0] < 1 && scores[1] < 0        
        
        # compare whether chat_1 is better than chat_2
        compare_res = model.compare(tokenizer, chat_1, chat_2)
        print("compare_res: ", compare_res)
        assert compare_res
        # >>> compare_res:  True
        
        # rank multiple chats, it will return the ranking index of each chat
        # the chat with the highest score will have ranking index as 0 
        rank_res = model.rank(tokenizer, [chat_1, chat_2])
        print("rank_res: ", rank_res)  # lower index means higher score
        # >>> rank_res:  [0, 1]  
        assert rank_res[0] == 0 && rank_res[1] == 1
)

class TestMMModel:
    """Test cases for base model."""

    @pytest.mark.parametrize(
        'model_name',
        [
            'internlm/internlm-xcomposer2-7b',
            'internlm/internlm-xcomposer2-7b-4bit'
        ],
    )
    def test_demo_default(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  trust_remote_code=True)
        # Set `torch_dtype=torch.float16` to load model in float16, otherwise
        # it will be loaded as float32 and might cause OOM Error.

        if '4bit' in model_name:
            model = InternLMXComposer2QForCausalLM.from_quantized(
                model_name, trust_remote_code=True, device='cuda:0').eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float32,
                trust_remote_code=True).cuda()

        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  trust_remote_code=True)
        model = model.eval()
        img_path_list = [
            'tests/panda.jpg',
            'tests/bamboo.jpeg',
        ]
        images = []
        for img_path in img_path_list:
            image = Image.open(img_path).convert('RGB')
            image = model.vis_processor(image)
            images.append(image)
        image = torch.stack(images)
        query = '<ImageHere> <ImageHere>please write an article ' \
            + 'based on the images. Title: my favorite animal.'
        with torch.cuda.amp.autocast():
            response, history = model.chat(tokenizer,
                                           query=query,
                                           image=image,
                                           history=[],
                                           do_sample=False)
        print(response)
        assert len(response) != 0
        assert ' panda' in response

        query = '<ImageHere> <ImageHere>请根据图片写一篇作文：我最喜欢的小动物。' \
            + '要求：选准角度，确定立意，明确文体，自拟标题。'
        with torch.cuda.amp.autocast():
            response, history = model.chat(tokenizer,
                                           query=query,
                                           image=image,
                                           history=[],
                                           do_sample=False)
        print(response)
        assert len(response) != 0
        assert '熊猫' in response


class TestMMVlModel:
    """Test cases for base model."""

    @pytest.mark.parametrize(
        'model_name',
        [
            'internlm/internlm-xcomposer2-vl-7b',
            'internlm/internlm-xcomposer2-vl-7b-4bit'
        ],
    )
    def test_demo_default(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  trust_remote_code=True)

        torch.set_grad_enabled(False)

        # init model and tokenizer
        if '4bit' in model_name:
            model = InternLMXComposer2QForCausalLM.from_quantized(
                model_name, trust_remote_code=True, device='cuda:0').eval()
        else:
            model = AutoModel.from_pretrained(
                model_name, trust_remote_code=True).cuda().eval()

        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  trust_remote_code=True)

        query = '<ImageHere>Please describe this image in detail.'
        image = 'tests/image.webp'
        with torch.cuda.amp.autocast():
            response, _ = model.chat(tokenizer,
                                     query=query,
                                     image=image,
                                     history=[],
                                     do_sample=False)
        print(response)
        assert len(response) != 0
        assert 'Oscar Wilde' in response
        assert 'Live life with no excuses, travel with no regret' in response


class InternLMXComposer2QForCausalLM(BaseGPTQForCausalLM):
    layers_block_name = 'model.layers'
    outside_layer_modules = [
        'vit',
        'vision_proj',
        'model.tok_embeddings',
        'model.norm',
        'output',
    ]
    inside_layer_modules = [
        ['attention.wqkv.linear'],
        ['attention.wo.linear'],
        ['feed_forward.w1.linear', 'feed_forward.w3.linear'],
        ['feed_forward.w2.linear'],
    ]
