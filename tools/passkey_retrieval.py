import argparse
import random
from numpy import random
from transformers import AutoTokenizer, AutoModelForCausalLM
import pdb

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--max_tokens', type=int, default=4000, help='maximum token length for evaluation')
    parser.add_argument('--interval', type=int, default=1000, help='interval for evaluation')
    parser.add_argument('--num_tests', type=int, default=20, help='number of repeat testing for each length')

    args = parser.parse_args()
    return args

# copy from https://github.com/dvlab-research/LongLoRA/blob/main/passkey_retrivial.py
def generate_prompt_landmark(n_garbage=60000, seed=666):
    """Generates a text file and inserts an passkey at a random position."""
    rnd_state = random.get_state()
    random.seed(seed)
    n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 5000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key? The pass key is"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question,
    ]
    random.set_state(rnd_state)
    return "\n".join(lines), str(pass_key)


def main(args):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("internlm/internlm-20b-chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("internlm/internlm-20b-chat", trust_remote_code=True, device_map="auto")

    total_test_points = args.max_tokens // args.interval
    all_accuries = {}
    for i in range(total_test_points):
        # This is a rough ratio to control the number of texts and tokens
        n_garbage = int(3.75 * (i + 1) * args.interval // 1024 * 1024)
        passed_tests = 0
        total_tokens = 0

        for j in range(args.num_tests):
            prompt, pass_key = generate_prompt_landmark(n_garbage=n_garbage, seed=j)
            response, _ = model.chat(tokenizer, prompt, history=[])
            if pass_key in response:
                passed_tests += 1
            total_tokens += len(tokenizer(prompt).input_ids)
        avg_tokens = total_tokens//args.num_tests
        accuracy = passed_tests/args.num_tests
        print("accuracy on the token length %d is %f"%(avg_tokens, accuracy))
        all_accuries[str(avg_tokens)] = accuracy
    print("accuries over tokens", all_accuries)


if __name__ == "__main__":
    args = parse_config()
    main(args)
