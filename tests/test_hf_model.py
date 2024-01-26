import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

prompts = ['你好', "what's your name"]


def assert_model(response):
    assert len(response) != 0
    assert 'UNUSED_TOKEN' not in response


class TestChat:
    """Test cases for chat model."""

    @pytest.mark.parametrize(
        'model_name',
        [
            'internlm/internlm2-chat-7b',
            'internlm/internlm2-chat-7b-sft',
        ],
    )
    def test_demo_default(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  trust_remote_code=True)
        # Set `torch_dtype=torch.float16` to load model in float16, otherwise
        # it will be loaded as float32 and might cause OOM Error.
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16,
            trust_remote_code=True).cuda()
        model = model.eval()
        for prompt in prompts:
            response, history = model.chat(tokenizer, prompt, history=[])
            print(response)
            assert_model(response)

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
            'internlm/internlm2-7b',
            'internlm/internlm2-base-7b',
        ],
    )
    def test_demo_default(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  trust_remote_code=True)
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
