# flake8: noqa
# isort: skip_file

# This logic is modified from ToRA:
#   - https://github.com/microsoft/ToRA
#
# Copyright (c) Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

import argparse
import multiprocessing
import os
import re
import sys
import traceback
from math import isclose, ceil
from typing import Union

import jsonlines
import numpy as np
from datasets import load_dataset
from lagent import (INTERNLM2_META, ActionExecutor, HFTransformer,
                    Internlm2Agent, Internlm2Protocol, LMDeployPipeline,
                    IPythonInteractiveManager)
from pebble import ProcessPool
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from tqdm import tqdm

# --------------------- modify the system prompt as needed ---------------------
DEFAULT_PROMPT = (
    'Integrate step-by-step reasoning and Python code to solve math problems '
    'using the following guidelines:\n'
    '- Analyze the question and write jupyter code to solve the problem;\n'
    r"- Present the final result in LaTeX using a '\boxed{{}}' without any "
    'units. \n')
# ------------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description='Math Code Interpreter')
    parser.add_argument('--backend',
                        type=str,
                        default='lmdeploy',
                        help='Which inference framework to use.',
                        choices=['lmdeploy', 'hf'])
    parser.add_argument(
        '--model_path',
        type=str,
        default='internlm/internlm2-chat-7b',
        help='Path or name to the model, could be HuggingFace model specifier.'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to save inference results to, should be a `jsonl` file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=100,
                        help='Agent inference batch size')
    parser.add_argument(
        '--max_turn',
        type=int,
        default=5,
        help=
        'Maximum number of interaction rounds between the agent and environment'
    )
    parser.add_argument(
        '--tp',
        type=int,
        default=1,
        help='Number of tensor parallelism. It may be required in LMDelpoy.')
    parser.add_argument('--temperature',
                        type=float,
                        default=0.1,
                        help='Temperature in next token prediction')
    parser.add_argument('--top_p',
                        type=float,
                        default=0.8,
                        help='Parameter for Top-P Sampling.')
    parser.add_argument('--top_k',
                        type=int,
                        default=40,
                        help='Parameter for Top-K Sampling.')
    parser.add_argument('--stop_words',
                        type=str,
                        default=['<|action_end|>', '<|im_end|>'],
                        action='append',
                        help='Stop words')
    parser.add_argument('--max_new_tokens',
                        type=int,
                        default=512,
                        help='Number of maximum generated tokens.')
    parser.add_argument(
        '--do_infer',
        default=True,
        action=argparse.BooleanOptionalAction,  # python > 3.8
        help='Whether to launch model inference.')
    # parser.add_argument(
    #     '--no-do_infer',
    #     dest='do_infer',
    #     action='store_false',
    #     help='Disable the inference.'
    # )
    parser.add_argument('--do_eval',
                        default=False,
                        action='store_true',
                        help='Whether to evaluate the inference results.')
    parser.add_argument('--overwrite',
                        default=False,
                        action='store_true',
                        help='Whether to overwrite the existing result file')
    return parser.parse_args()


def _fix_fracs(string):
    substrs = string.split('\\frac')
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += '\\frac'
            if len(substr) > 0 and substr[0] == '{':
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != '{':
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}{' + b + '}' + post_substr
                    else:
                        new_str += '{' + a + '}{' + b + '}'
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}' + b + post_substr
                    else:
                        new_str += '{' + a + '}' + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split('/')) != 2:
        return string
    a = string.split('/')[0]
    b = string.split('/')[1]
    try:
        if 'sqrt' not in a:
            a = int(a)
        if 'sqrt' not in b:
            b = int(b)
        assert string == '{}/{}'.format(a, b)
        new_string = '\\frac{' + str(a) + '}{' + str(b) + '}'
        return new_string
    except Exception:
        return string


def _fix_sqrt(string):
    _string = re.sub(r'\\sqrt(\w+)', r'\\sqrt{\1}', string)
    return _string


def strip_string(string):
    string = str(string).strip()
    # linebreaks
    string = string.replace('\n', '')

    # right "."
    string = string.rstrip('.')

    # remove inverse spaces
    string = string.replace('\\!', '')
    string = string.replace('\\ ', '')

    # replace \\ with \
    string = string.replace('\\\\', '\\')
    string = string.replace('\\\\', '\\')

    # replace tfrac and dfrac with frac
    string = string.replace('tfrac', 'frac')
    string = string.replace('dfrac', 'frac')

    # remove \left and \right
    string = string.replace('\\left', '')
    string = string.replace('\\right', '')

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r'\\text{.*?}$', '', string).strip()
    if _string != '' and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    # Remove circ (degrees)
    string = string.replace('^{\\circ}', '')
    string = string.replace('^\\circ', '')

    # remove dollar signs
    string = string.replace('\\$', '')
    string = string.replace('$', '')

    string = string.replace('\\text', '')
    string = string.replace('x\\in', '')

    # remove percentage
    string = string.replace('\\%', '')
    string = string.replace('\%', '')
    string = string.replace('%', '')

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(' .', ' 0.')
    string = string.replace('{.', '{0.')

    # cdot
    string = string.replace('\\cdot', '')

    # inf
    string = string.replace('infinity', '\\infty')
    if '\\infty' not in string:
        string = string.replace('inf', '\\infty')
    string = string.replace('+\\inity', '\\infty')

    # and
    string = string.replace('and', '')
    string = string.replace('\\mathbf', '')

    # use regex to remove \mbox{...}
    string = re.sub(r'\\mbox{.*?}', '', string)

    # quote
    string.replace("'", '')
    string.replace('"', '')

    # i, j
    if 'j' in string and 'i' not in string:
        string = string.replace('j', 'i')

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r'(\d+)\.0+([^\d])', r'\1\2', string)
    string = re.sub(r'(\d+)\.0+$', r'\1', string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == '.':
        string = '0' + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split('=')) == 2:
        if len(string.split('=')[0]) <= 2:
            string = string.split('=')[1]

    string = _fix_sqrt(string)
    string = string.replace(' ', '')

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def last_boxed_only_string(string):
    idx = string.rfind('\\boxed')
    if idx < 0:
        idx = string.rfind('\\fbox')
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == '{':
            num_left_braces_open += 1
        if string[i] == '}':
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def extract_answer(pred_str: str, execute: bool = False) -> str:
    if re.search('\boxed|boxed', pred_str):
        answer = re.split('\boxed|boxed', pred_str)[-1]
        if len(answer) == 0:
            return ''
        elif (answer[0] == '{'):
            stack = 1
            a = ''
            for c in answer[1:]:
                if (c == '{'):
                    stack += 1
                    a += c
                elif (c == '}'):
                    stack -= 1
                    if (stack == 0): break
                    a += c
                else:
                    a += c
        else:
            a = answer.split('$')[0].strip()
    elif re.search('[Tt]he (final )?answer is:?', pred_str):
        a = re.split('[Tt]he (final )?answer is:?',
                     pred_str)[-1].strip().rstrip('.')
    elif pred_str.startswith('```python') and execute:
        # fall back to program
        from lagent import get_tool

        a = get_tool('IPythonInteractive').exec(pred_str).value or ''
    else:  # use the last number
        pred = re.findall(r'-?\d*\.?\d+', pred_str.replace(',', ''))
        if len(pred) >= 1:
            a = pred[-1]
        else:
            a = ''
    # multiple lines
    pred = a.split('\n')[0]
    if pred != '' and pred[0] == ':':
        pred = pred[1:]
    if pred != '' and pred[-1] == '.':
        pred = pred[:-1]
    if pred != '' and pred[-1] == '/':
        pred = pred[:-1]
    pred = strip_string(pred)
    return pred


def is_digit(s):
    try:
        float(str(s).replace(',', ''))
        return True
    except ValueError:
        return False


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    tolerance: float = 1e-4,
    timeout: bool = False,
) -> bool:
    """Exact match of math if and only if:

    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    try:  # 1. numerical equal
        if is_digit(prediction) and is_digit(reference):
            prediction = float(str(prediction).replace(',', ''))
            reference = float(str(reference).replace(',', ''))
            # number questions
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if isclose(item, prediction, rel_tol=tolerance):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except Exception:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    ## deal with [], (), {}
    pred_str, ref_str = prediction, reference
    if (prediction.startswith('[') and prediction.endswith(']')
            and not reference.startswith('(')) or (
                prediction.startswith('(') and prediction.endswith(')')
                and not reference.startswith('[')):
        pred_str = pred_str.strip('[]()')
        ref_str = ref_str.strip('[]()')
    for s in ['{', '}', '(', ')']:
        ref_str = ref_str.replace(s, '')
        pred_str = pred_str.replace(s, '')
    if pred_str == ref_str:
        return True

    ## [a, b] vs. [c, d], return a==c and b==d
    if ((prediction.startswith('[') and prediction.endswith(']')) and
        (reference.startswith('[') and reference.endswith(']'))
            or (prediction.startswith('(') and prediction.endswith(')')) and
        (reference.startswith('(') and reference.endswith(')'))):
        pred_parts = prediction[1:-1].split(',')
        ref_parts = reference[1:-1].split(',')
        if len(pred_parts) == len(ref_parts):
            if all([
                    math_equal(pred_parts[i], ref_parts[i], include_percentage,
                               is_close) for i in range(len(pred_parts))
            ]):
                return True

    # symbolic equal with sympy
    if timeout:
        if call_with_timeout(symbolic_equal_process, prediction, reference):
            return True
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False


def math_equal_process(param):
    return math_equal(param[-2], param[-1])


def symbolic_equal(a, b):

    def _parse(s):
        for f in [parse_latex, parse_expr]:
            try:
                return f(s)
            except Exception:
                pass
        return s

    a = _parse(a)
    b = _parse(b)

    try:
        if simplify(a - b) == 0:
            return True
    except Exception:
        pass

    try:
        if isclose(N(a), N(b), rel_tol=1e-3):
            return True
    except Exception:
        pass
    return False


def symbolic_equal_process(a, b, output_queue):
    result = symbolic_equal(a, b)
    output_queue.put(result)


def call_with_timeout(func, *args, timeout=1, **kwargs):
    output_queue = multiprocessing.Queue()
    process_args = args + (output_queue, )
    process = multiprocessing.Process(target=func,
                                      args=process_args,
                                      kwargs=kwargs)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return False

    return output_queue.get()


def init_agent(backend: str, max_turn: int, model_path: str, tp: int,
               **kwargs):
    if backend == 'lmdeploy':
        from lmdeploy import TurbomindEngineConfig
        model = LMDeployPipeline(
            path=model_path,
            model_name='internlm2-chat',
            meta_template=INTERNLM2_META,
            pipeline_cfg=dict(backend_config=TurbomindEngineConfig(tp=tp)),
            **kwargs)
    elif backend == 'hf':
        model = HFTransformer(path=model_path,
                              meta_template=INTERNLM2_META,
                              **kwargs)
    else:
        raise NotImplementedError

    agent = Internlm2Agent(
        llm=model,
        protocol=Internlm2Protocol(meta_prompt=None,
                                   interpreter_prompt=DEFAULT_PROMPT),
        interpreter_executor=ActionExecutor(actions=[
            IPythonInteractiveManager(max_workers=200,
                                      ci_lock=os.path.join(
                                          os.path.dirname(__file__),
                                          '.ipython.lock'))
        ]),
        max_turn=max_turn)
    return agent


def predict(args):

    def process(d, k):
        d['idx'] = k
        d['query'] = d['problem']
        gt = extract_answer(d['solution'])
        if '\\boxed{90\\text{ square\nunits}}' in d['solution']:
            gt = '90'
        elif '$6$ is our answer' in d['solution']:
            gt = '6'
        elif gt.startswith('x\\in'):
            gt = gt[len('x\\in'):]
        gt = strip_string(gt)
        d['gt'] = gt
        d['pred'], d['steps'] = [], []
        d['error'] = None
        return d

    dataset = load_dataset('lighteval/MATH', split='test').map(process, True)
    agent = init_agent(
        backend=args.backend,
        max_turn=args.max_turn,
        model_path=args.model_path,
        tp=args.tp,
        temperature=args.temperature,
        stop_words=args.stop_words,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
    )
    num_batches = ceil(len(dataset) / args.batch_size)
    with jsonlines.open(args.output_path, 'w', flush=True) as f:
        for i in tqdm(range(num_batches)):
            batch = dataset.select(
                range(i * args.batch_size,
                      min((i + 1) * args.batch_size, len(dataset))))
            try:
                rets = agent.batch_chat(batch['query'])
                for item, ret in zip(batch, rets):
                    item['steps'] = ret.inner_steps
                    last = item['steps'][-1]
                    item['pred'].append(
                        extract_answer(last['content']) if last['role'] ==
                        'language' else '😭')
                    f.write(item)
            except Exception as e:
                err = str(traceback.format_exc())
                print(f'Processing batch data error: {e}\n{err}')
                for item in batch:
                    item['error'] = err
                    f.write(item)
            finally:
                agent._interpreter_executor.actions[
                    'IPythonInteractiveManager'].reset()


def evaluate(args):
    samples = [sample for sample in jsonlines.open(args.output_path)]
    scores = []
    timeout_cnt = 0
    with ProcessPool() as pool:
        future = pool.map(
            math_equal_process,
            [(idx, pred, sample['gt']) for idx, sample in enumerate(samples)
             for pred in sample['pred']],
            timeout=20,
        )
        iterator = future.result()
        with tqdm(total=len(samples), desc='Evaluate') as progress_bar:
            while True:
                try:
                    result = next(iterator)
                    scores.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                    scores.append(False)
                    timeout_cnt += 1
                except Exception as error:
                    print(error.__traceback__)
                    scores.append(False)
                    # sys.exit()
                progress_bar.update(1)

    idx = 0
    score_mat = []
    for sample in samples:
        sample['score'] = scores[idx:idx + len(sample['pred'])]
        assert len(sample['score']) == len(sample['pred'])
        score_mat.append(sample['score'])
        idx += len(sample['pred'])

    max_len = max([len(s) for s in score_mat])

    for i, s in enumerate(score_mat):
        if len(s) < max_len:
            score_mat[i] = s + [s[-1]] * (max_len - len(s))  # pad

    # output mean of each column of scores
    col_means = np.array(score_mat).mean(axis=0)
    mean_score = list(np.round(col_means * 100, decimals=1))

    result_str = f'Num samples: {len(samples)}\n' \
        f'Num scores: {len(scores)}\n' \
        f'Sum scores: {sum(scores)}\n' \
        f'Timeout samples: {timeout_cnt}\n' \
        f"Empty samples: {len([s for s in samples if not s['pred'][-1]])}\n" \
        f'Mean score: {mean_score}\n'

    # each type score
    if 'type' in samples[0]:
        type_scores = {}
        for sample in samples:
            if sample['type'] not in type_scores:
                type_scores[sample['type']] = []
            type_scores[sample['type']].append(sample['score'][-1])
        type_scores = {
            k: np.round(np.array(v).mean() * 100, decimals=1)
            for k, v in type_scores.items()
        }
        type_scores = {
            k: v
            for k, v in sorted(type_scores.items(), key=lambda item: item[0])
        }
        result_str += f'Type scores: {type_scores}\n'

    print(result_str)


if __name__ == '__main__':
    args = parse_args()
    if args.do_infer and os.path.exists(
            args.output_path) and not args.overwrite:
        args.do_infer = False
        print(f'File {args.output_path} already exists. '
              f'Please add the `--overwrite` flag if needed.')
    if args.do_infer:
        predict(args)
    if args.do_eval:
        if not args.do_infer:
            evaluate(args)
        else:
            import subprocess

            res = subprocess.run(
                [
                    sys.executable, __file__, '--output_path',
                    args.output_path, '--no-do_infer', '--do_eval'
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            print(res.stdout)
