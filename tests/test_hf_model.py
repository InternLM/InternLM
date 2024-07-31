import pytest
import torch
from auto_gptq.modeling import BaseGPTQForCausalLM
from bs4 import BeautifulSoup
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
            'internlm/internlm2_5-20b-chat', 'internlm/internlm2_5-1_8b-chat',
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


class TestBase:
    """Test cases for base model."""

    @pytest.mark.parametrize(
        'model_name',
        [
            'internlm/internlm2_5-7b', 'internlm/internlm2-7b',
            'internlm/internlm2-base-7b', 'internlm/internlm2-20b',
            'internlm/internlm2-base-20b', 'internlm/internlm2-1_8b',
            'internlm/internlm2_5-20b',
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


class TestReward:
    """Test cases for base model."""

    @pytest.mark.parametrize(
        'model_name',
        [
            'internlm/internlm2-1_8b-reward', 'internlm/internlm2-7b-reward',
            'internlm/internlm2-20b-reward'
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
        model = AutoModel.from_pretrained(
            model_name,
            device_map='cuda',
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  trust_remote_code=True)

        chat_1 = [{
            'role': 'user',
            'content': "Hello! What's your name?"
        }, {
            'role':
            'assistant',
            'content':
            'I am InternLM2! A helpful AI assistant. What can I do for you?'
        }]
        chat_2 = [{
            'role': 'user',
            'content': "Hello! What's your name?"
        }, {
            'role': 'assistant',
            'content': 'I have no idea.'
        }]

        # get reward score for a single chat
        score1 = model.get_score(tokenizer, chat_1)
        score2 = model.get_score(tokenizer, chat_2)
        print('score1: ', score1)
        print('score2: ', score2)
        assert score1 > 0
        assert score2 < 0

        # batch inference, get multiple scores at once
        scores = model.get_scores(tokenizer, [chat_1, chat_2])
        print('scores: ', scores)
        assert scores[0] > 0
        assert scores[1] < 0

        # compare whether chat_1 is better than chat_2
        compare_res = model.compare(tokenizer, chat_1, chat_2)
        print('compare_res: ', compare_res)
        assert compare_res
        # >>> compare_res:  True

        # rank multiple chats, it will return the ranking index of each chat
        # the chat with the highest score will have ranking index as 0
        rank_res = model.rank(tokenizer, [chat_1, chat_2])
        print('rank_res: ', rank_res)  # lower index means higher score
        # >>> rank_res:  [0, 1]
        assert rank_res[0] == 0
        assert rank_res[1] == 1


class TestXcomposer2d5Model:
    """Test cases for base model."""

    @pytest.mark.parametrize(
        'model_name',
        [
            'internlm/internlm-xcomposer2d5-7b',
        ],
    )
    def test_video_understanding(self, model_name):
        torch.set_grad_enabled(False)

        # init model and tokenizer
        model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            trust_remote_code=True).cuda().eval().half()
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  trust_remote_code=True)
        model.tokenizer = tokenizer

        query = 'Here are some frames of a video. Describe this video in detail'  # noqa: F401, E501
        image = [
            '/mnt/petrelfs/qa-caif-cicd/github_runner/examples/liuxiang.mp4',
        ]

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            response, his = model.chat(tokenizer,
                                       query,
                                       image,
                                       do_sample=False,
                                       num_beams=3,
                                       use_meta=True)
        print(response)
        assert len(response) > 100
        assert 'athlete' in response.lower()

        query = 'tell me the athlete code of Liu Xiang'
        image = [
            '/mnt/petrelfs/qa-caif-cicd/github_runner/examples/liuxiang.mp4',
        ]
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            response, _ = model.chat(tokenizer,
                                     query,
                                     image,
                                     history=his,
                                     do_sample=False,
                                     num_beams=3,
                                     use_meta=True)
        print(response)
        assert len(response) > 10
        assert '1363' in response.lower()

    @pytest.mark.parametrize(
        'model_name',
        [
            'internlm/internlm-xcomposer2d5-7b',
        ],
    )
    def test_multi_image_understanding(self, model_name):
        torch.set_grad_enabled(False)

        # init model and tokenizer
        model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            trust_remote_code=True).cuda().eval().half()
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  trust_remote_code=True)
        model.tokenizer = tokenizer

        query = 'Image1 <ImageHere>; Image2 <ImageHere>; Image3 <ImageHere>; I want to buy a car from the three given cars, analyze their advantages and weaknesses one by one'  # noqa: F401, E501
        image = [
            '/mnt/petrelfs/qa-caif-cicd/github_runner/examples/cars1.jpg',
            '/mnt/petrelfs/qa-caif-cicd/github_runner/examples/cars2.jpg',
            '/mnt/petrelfs/qa-caif-cicd/github_runner/examples/cars3.jpg',
        ]
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            response, his = model.chat(tokenizer,
                                       query,
                                       image,
                                       do_sample=False,
                                       num_beams=3,
                                       use_meta=True)
        print(response)
        assert len(response) > 100
        assert 'car' in response.lower()

        query = 'Image4 <ImageHere>; How about the car in Image4'
        image.append(
            '/mnt/petrelfs/qa-caif-cicd/github_runner/examples/cars4.jpg')
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            response, _ = model.chat(tokenizer,
                                     query,
                                     image,
                                     do_sample=False,
                                     num_beams=3,
                                     history=his,
                                     use_meta=True)
        print(response)
        assert len(response) > 10
        assert 'ferrari' in response.lower()

    @pytest.mark.parametrize(
        'model_name',
        [
            'internlm/internlm-xcomposer2d5-7b',
        ],
    )
    def test_high_resolution_default(self, model_name):
        torch.set_grad_enabled(False)

        # init model and tokenizer
        model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            trust_remote_code=True).cuda().eval().half()
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  trust_remote_code=True)
        model.tokenizer = tokenizer

        query = 'Analyze the given image in a detail manner'
        image = ['/mnt/petrelfs/qa-caif-cicd/github_runner/examples/dubai.png']
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            response, _ = model.chat(tokenizer,
                                     query,
                                     image,
                                     do_sample=False,
                                     num_beams=3,
                                     use_meta=True)
        print(response)
        assert len(response) > 100
        assert 'dubai' in response.lower()

    @pytest.mark.parametrize(
        'model_name',
        [
            'internlm/internlm-xcomposer2d5-7b',
        ],
    )
    def test_introduce_web_default(self, model_name):
        torch.set_grad_enabled(False)
        # init model and tokenizer
        model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            trust_remote_code=True).cuda().eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  trust_remote_code=True)
        model.tokenizer = tokenizer

        query = '''A website for Research institutions. The name is Shanghai
        AI lab. Top Navigation Bar is blue.Below left, an image shows the
        logo of the lab. In the right, there is a passage of text below that
        describes the mission of the laboratory.There are several images to
        show the research projects of Shanghai AI lab.'''
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            response = model.write_webpage(
                query,
                seed=202,
                task='Instruction-aware Webpage Generation',
                repetition_penalty=3.0)
        print(response)
        assert len(response) > 100
        assert is_html_code(response)
        assert 'href' in response.lower()

    @pytest.mark.parametrize(
        'model_name',
        [
            'internlm/internlm-xcomposer2d5-7b',
        ],
    )
    def test_resume_to_webset_default(self, model_name):
        torch.set_grad_enabled(False)

        # init model and tokenizer
        model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            trust_remote_code=True).cuda().eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  trust_remote_code=True)
        model.tokenizer = tokenizer

        # the input should be a resume in markdown format
        query = '/mnt/petrelfs/qa-caif-cicd/github_runner/examples/resume.md'
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            response = model.resume_2_webpage(query,
                                              seed=202,
                                              repetition_penalty=3.0)
        print(response)
        assert len(response) > 100
        assert is_html_code(response)
        assert 'href' in response.lower()

    @pytest.mark.parametrize(
        'model_name',
        [
            'internlm/internlm-xcomposer2d5-7b',
        ],
    )
    def test_screen_to_webset_default(self, model_name):
        torch.set_grad_enabled(False)

        # init model and tokenizer
        model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            trust_remote_code=True).cuda().eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  trust_remote_code=True)
        model.tokenizer = tokenizer

        query = 'Generate the HTML code of this web image with Tailwind CSS.'
        image = [
            '/mnt/petrelfs/qa-caif-cicd/github_runner/examples/screenshot.jpg'
        ]
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            response = model.screen_2_webpage(query,
                                              image,
                                              seed=202,
                                              repetition_penalty=3.0)
        print(response)
        assert len(response) > 100
        assert is_html_code(response)
        assert 'href' in response.lower()

    @pytest.mark.parametrize(
        'model_name',
        [
            'internlm/internlm-xcomposer2d5-7b',
        ],
    )
    def test_write_artical_default(self, model_name):
        torch.set_grad_enabled(False)

        # init model and tokenizer
        model = AutoModel.from_pretrained(
            'internlm/internlm-xcomposer2d5-7b',
            torch_dtype=torch.bfloat16,
            trust_remote_code=True).cuda().eval()
        tokenizer = AutoTokenizer.from_pretrained(
            'internlm/internlm-xcomposer2d5-7b', trust_remote_code=True)
        model.tokenizer = tokenizer

        query = '''阅读下面的材料，根据要求写作。 电影《长安三万里》的出现让人感慨，影片并未将重点全落在大唐风华上，
        也展现了恢弘气象的阴暗面，即旧门阀的资源垄断、朝政的日益衰败与青年才俊的壮志难酬。高适仕进无门，只能回乡>沉潜修行。
        李白虽得玉真公主举荐，擢入翰林，但他只是成为唐玄宗的御用文人，不能真正实现有益于朝政的志意。然而，片中高潮部分《将进酒》一节，
        人至中年、挂着肚腩的李白引众人乘仙鹤上天，一路从水面、瀑布飞升至银河进入仙>宫，李白狂奔着与仙人们碰杯，最后大家纵身飞向漩涡般的九重天。
        肉身的微贱、世路的“天生我材必有用，坎坷，拘不住精神的高蹈。“天生我材必有用，千金散尽还复来。” 古往今来，身处闲顿、遭受挫折、被病痛折磨，
        很多人都曾经历>了人生的“失意”，却反而成就了他们“诗意”的人生。对正在追求人生价值的当代青年来说，如何对待人生中的缺憾和困顿?诗意人生中又
        有怎样的自我坚守和自我认同?请结合“失意”与“诗意”这两个关键词写一篇文章。 要求:选准角度，确定>立意，明确文体，自拟标题;不要套作，不得抄
        袭;不得泄露个人信息;不少于 800 字。'''
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            response = model.write_artical(query, seed=8192)
        print(response)
        assert len(response) > 100
        assert '。' in response and '诗' in response

        query = '''Please write a blog based on the title: French Pastries:
        A Sweet Indulgence'''
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            response = model.write_artical(query, seed=8192)
        print(response)
        assert len(response) > 100
        assert ' ' in response and 'a' in response


def is_html_code(html_code):
    try:
        soup = BeautifulSoup(html_code, 'lxml')
        if soup.find('html'):
            print('HTML appears to be well-formed.')
            return True
        else:
            print('There was an issue with the HTML structure.')
            return False
    except Exception as e:
        print('Error parsing HTML:', str(e))
        return False


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
