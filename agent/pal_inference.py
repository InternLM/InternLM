# flake8: noqa

# This file is modified from:
# hhttps://github.com/reasoning-machines/pal/blob/main/pal/core/interface.py
#
# Copyright 2022 PAL Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# isort: skip_file
import argparse
import copy
import json
import os
import signal
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
import tqdm
from datasets import load_dataset
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import (LogitsProcessorList,
                                           StoppingCriteriaList)
from transformers.utils import logging

logger = logging.get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='PAL Inference')
    parser.add_argument('model',
                        type=str,
                        help='Path to the pre-trained LLM used for inference.')
    parser.add_argument(
        'out_dir',
        type=str,
        help=
        'Name of the output folder where generated code snippets will be saved.'
    )
    parser.add_argument('--dataset',
                        default='gsm8k',
                        type=str,
                        help='Name of the dataset used for code generation.')
    parser.add_argument(
        '--max_length',
        default=2048,
        type=int,
        help='Maximum input token length for the natural language description.',
    )
    parser.add_argument(
        '--top_p',
        default=0.8,
        type=float,
        help='Probability threshold to choose sample tokens during generation.',
    )
    parser.add_argument(
        '--eoh',
        default='',
        type=str,
        help='End of human (user) token.',
    )
    parser.add_argument(
        '--eoa',
        default='',
        type=str,
        help='End of assistant (bot) token.',
    )
    parser.add_argument(
        '--eos',
        default='',
        type=str,
        help='End of system token.',
    )
    parser.add_argument(
        '--temperature',
        '-t',
        default=1.0,
        type=float,
        help='Temperature of token sampling during generation.')
    parser.add_argument(
        '--time_out',
        default=100,
        type=float,
        help='Maximum time allowed for executing generated code.')
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help=
        'Print code error information when executing generated code (optional).',
    )
    parser.add_argument(
        '--append',
        '-a',
        action='store_true',
        help='Append output to the history results (optional).')
    args = parser.parse_args()
    return args


class Timeout:
    """Timer to execute code.

    Adapted from https://github.com/reasoning-machines/pal

    Args:
        seconds (float): The maximum seconds to execute code
        error_message (str)
    """

    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def timeout_handler(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(self.seconds)

    def __exit__(self, error_type, value, traceback):
        signal.alarm(0)


@dataclass
class GenerationConfig:
    max_length: int = 64
    top_p: float = 0.8
    temperature: float = 0.8
    do_sample: bool = True
    repetition_penalty: float = 1.0


@torch.inference_mode()
def generate_interactive(
    model,
    tokenizer,
    prompt,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor],
                                                List[int]]] = None,
    additional_eos_token_id: Optional[int] = None,
    **kwargs,
):
    inputs = tokenizer([prompt], padding=True, return_tensors='pt')
    input_length = len(inputs['input_ids'][0])
    for k, v in inputs.items():
        inputs[k] = v.cuda()
    input_ids = inputs['input_ids']
    batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]  # noqa: F841  # pylint: disable=W0612
    if generation_config is None:
        generation_config = model.generation_config
    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)
    bos_token_id, eos_token_id = (  # noqa: F841  # pylint: disable=W0612
        generation_config.bos_token_id,
        generation_config.eos_token_id,
    )
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    if additional_eos_token_id is not None:
        eos_token_id.append(additional_eos_token_id)
    has_default_max_length = kwargs.get(
        'max_length') is None and generation_config.max_length is not None
    if has_default_max_length and generation_config.max_new_tokens is None:
        warnings.warn(
            f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
            'This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we'
            ' recommend using `max_new_tokens` to control the maximum length of the generation.',
            UserWarning,
        )
    elif generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
        if not has_default_max_length:
            logger.warn(  # pylint: disable=W4902
                f'Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(='
                f'{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. '
                'Please refer to the documentation for more information. '
                '(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)',
                UserWarning,
            )

    if input_ids_seq_length >= generation_config.max_length:
        input_ids_string = 'input_ids'
        logger.warning(
            f'Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to'
            f' {generation_config.max_length}. This can lead to unexpected behavior. You should consider'
            ' increasing `max_new_tokens`.')

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList(
    )
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList(
    )

    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config,
        stopping_criteria=stopping_criteria)
    logits_warper = model._get_logits_warper(generation_config)

    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    scores = None
    while True:
        model_inputs = model.prepare_inputs_for_generation(
            input_ids, **model_kwargs)
        # forward pass to get next token
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # sample
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        if generation_config.do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=False)
        unfinished_sequences = unfinished_sequences.mul(
            (min(next_tokens != i for i in eos_token_id)).long())

        output_token_ids = input_ids[0].cpu().tolist()
        output_token_ids = output_token_ids[input_length:]
        for each_eos_token_id in eos_token_id:
            if output_token_ids[-1] == each_eos_token_id:
                output_token_ids = output_token_ids[:-1]
        response = tokenizer.decode(output_token_ids)

        yield response
        # stop when each sentence is finished, or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(
                input_ids, scores):
            break


class GenericRuntime:
    """Adapted from https://github.com/reasoning-machines/pal."""

    GLOBAL_DICT: dict = {}
    LOCAL_DICT = None
    HEADERS: List = []

    def __init__(self):
        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(
            self.LOCAL_DICT) if self.LOCAL_DICT else None

        for c in self.HEADERS:
            self.exec_code(c)

    def exec_code(self, code_piece: str) -> None:
        exec(code_piece, self._global_vars)

    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars)

    def inject(self, var_dict: Dict[str, Any]) -> None:
        for k, v in var_dict.items():
            self._global_vars[k] = v

    @property
    def answer(self):
        return self._global_vars['answer']


class PALInterface:
    """PAL interface wrap fun:`generate_interactive` to extract and execute
    generated code.

    Adapted from https://github.com/reasoning-machines/pal

    Args:
        model (AutoModelForCausalLM)
        tokenizer (AutoTokenizer)
        generation_config (GenerationConfig): Decode strategies
        additional_eos_token_id (int): End of sentence token id, default: 103028
        get_answer_expr (str): The function name of generated code, default: "solution()"
        verbose (bool): Print error information
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        generation_config: GenerationConfig,
        additional_eos_token_id: int = 103028,
        get_answer_expr: str = 'solution()',
        verbose: bool = False,
    ):
        self.runtime = GenericRuntime()
        self.history: List = []
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.additional_eos_token_id = additional_eos_token_id
        self.answer_expr = get_answer_expr
        self.verbose = verbose

    def generate(self, prompt):
        # The api will generate response word by word
        # we only need the last generation as the final results
        for cur_gen in generate_interactive(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                additional_eos_token_id=self.additional_eos_token_id,
                **asdict(self.generation_config),
        ):
            continue
        # Get final response
        self.history.append(cur_gen)
        # Extract code block
        code = self.process_generation_to_code(cur_gen)
        return code

    def process_generation_to_code(self, gens: str):
        if '```python' in gens:
            gens = gens.split('```python')[1].split('```')[0]
        elif '```' in gens:
            gens = gens.split('```')[1].split('```')[0]
        code = gens.split('\n')
        return code

    def run(self, prompt, time_out: float = 100):
        code = self.generate(prompt)
        with Timeout(time_out):
            try:
                exec_result = self.execute(code)
            except Exception as e:
                if self.verbose:
                    print(e)
        return exec_result

    def execute(self, code: List[str]):
        self.runtime.exec_code('\n'.join(code))
        return self.runtime.eval_code(self.answer_expr)

    def clear_history(self):
        self.history = []


def load_model(args):
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 trust_remote_code=True).to(
                                                     torch.bfloat16).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              trust_remote_code=True)
    return model, tokenizer


def load_data(args):
    # Load data from huggingface dataset
    if args.dataset == 'gsm8k':
        gsm8k = load_dataset(path=args.dataset, name='main')
        test_set = gsm8k['test']
        input_data = []
        for data in test_set:
            question = data['question']
            target = float(data['answer'].split('#')[-1].replace(',', ''))
            input_data.append({'question': question, 'target': target})
    else:
        raise NotImplementedError
    return input_data


PROMPT = """<|System|>:You are a helpful assistant which use tools to solve mathematical reasoning questions. The tools you can use are:
PythonExecutor: It can execute Python code. The code must be a function, and the function name must be 'solution'. The example format is as follows:
```python
def solution():
    variable_names_with_real_meaning = func(variable)
    return variable_names_with_real_meaning
```{eos}
<|User|>:Olivia has $23. She bought five bagels for $3 each. How much money does she have left?{eoh}
<|Bot|>:
```python
def solution():
    money_initial = 23
    bagels = 5
    bagel_cost = 3
    money_spent = bagels * bagel_cost
    money_left = money_initial - money_spent
    result = money_left
    return result
```{eoa}
<|User|>:Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?{eoh}
<|Bot|>:
```python
def solution():
    golf_balls_initial = 58
    golf_balls_lost_tuesday = 23
    golf_balls_lost_wednesday = 2
    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
    result = golf_balls_left
    return result
```{eoa}
<|User|>:There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?{eoh}
<|Bot|>:
```python
def solution():
    computers_initial = 9
    computers_per_day = 5
    num_days = 4  # 4 days between monday and thursday
    computers_added = computers_per_day * num_days
    computers_total = computers_initial + computers_added
    result = computers_total
    return result
```{eoa}
<|System|>:How about this question?{eos}
<|User|>:{question}{eoh}
<|Bot|>:""".strip()


def main():

    args = parse_args()

    print('load model begin.')
    model, tokenizer = load_model(args)
    print('load model end.')

    generation_config = GenerationConfig(max_length=args.max_length,
                                         top_p=args.top_p,
                                         temperature=args.temperature)

    verbose = args.verbose
    interface = PALInterface(model=model,
                             tokenizer=tokenizer,
                             generation_config=generation_config,
                             verbose=verbose)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    savepath = os.path.join(args.out_dir, args.dataset + '.json')

    # Load from history results
    if args.append and os.path.exists(savepath):
        lines = open(savepath).readlines()
        num_skip_exps = len(lines)
        scores = [x['score'] for x in map(json.loads, lines)]
    else:
        num_skip_exps = 0
        scores = []

    examples = load_data(args)
    with open(savepath, 'a' if args.append else 'w') as f:
        pbar = tqdm.tqdm(examples[num_skip_exps:],
                         initial=num_skip_exps,
                         total=len(examples))
        for x in pbar:
            question = x['question']
            result = copy.copy(x)

            try:
                answer = interface.run(
                    prompt=PROMPT.format(question=question,
                                         eoh=args.eoh,
                                         eoa=args.eoa,
                                         eos=args.eos),
                    time_out=args.time_out,
                )
                answer = float(answer)
                score = 1 if abs(answer - x['target']) < 1e-3 else 0
            except Exception as e:
                if verbose:
                    print(e)
                answer = ''
                score = 0
            scores.append(score)
            result['answer'] = answer
            result['score'] = score
            result['generation'] = interface.history
            f.write(json.dumps(result) + '\n')

            interface.clear_history()
            f.flush()

    print(f'{args.model}: Accuracy - {sum(scores) / len(scores)}')
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
