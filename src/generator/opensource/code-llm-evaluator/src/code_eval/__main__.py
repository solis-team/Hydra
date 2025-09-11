import os
import sys
import argparse
import pkg_resources

def get_args():
    parser = argparse.ArgumentParser(description=f"{20*'='} Code Evaluator {20*'='}")
    
    parser.add_argument("-V", "--version", action="version", help="Get version",
                        version=pkg_resources.get_distribution("code_eval").version)
    
    parser.add_argument("--task",
                        help='Select pre-defined task')
    parser.add_argument("--model_name", type=str,
                        help='Local path or Huggingface Hub link to load model')
    parser.add_argument("--peft_model", default=None, type=str,
                        help='Lora config')
    parser.add_argument("--cache_dir", default=None, type=str,
                        help='Cache for save model download checkpoint and dataset')
    parser.add_argument("--save_dir", default="./output", type=str,
                        help='Save generation and result path')
    
    parser.add_argument("--backend", default="tf", type=str,
                        help="Select between ``vllm`` or Huggingface's transformers ``tf`` backend")
    parser.add_argument("--max_tokens", default=128, type=int,
                        help='Number of max new tokens')
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--inst_token", default="", type=str)
    parser.add_argument("--assist_token", default="", type=str)
    parser.add_argument("--temperature", default=0.9, type=float)
    parser.add_argument("--repetition_penalty", default=1.2, type=float)
    parser.add_argument("--num_return_sequences", default=1, type=int)
    
    args = parser.parse_args()
    
    if not args.cache_dir:
        TRANSFORMER_CACHE = os.getenv("TRANSFORMER_CACHE")
        HF_HOME = os.getenv("HF_HOME")
        if TRANSFORMER_CACHE:
            args.cache_dir = TRANSFORMER_CACHE
        else:
            args.cache_dir = HF_HOME
        
    return args, parser


def main():
    args, parsre = get_args()
    if args.task:
        generate(args=args)
    else:
        parsre.print_help()
    

def generate(args):
    print(args)
    from code_eval.evaluator import Evaluator
    from code_eval.tasks import HumanEval, MBPP

    if args.task == "humaneval":
        task_loader = HumanEval(inst_token=args.inst_token,
                                assist_token=args.assist_token)
    elif args.task == "instruct-humaneval":
        task_loader = HumanEval(inst_token=args.inst_token,
                                assist_token=args.assist_token,
                                mode="instruct")
    elif args.task == "instruct-humaneval-no-context":
        task_loader = HumanEval(inst_token=args.inst_token,
                                assist_token=args.assist_token,
                                mode="instruct-no-context")
    elif args.task == "mbpp":
        task_loader = MBPP(inst_token=args.inst_token,
                                assist_token=args.assist_token)
    elif args.task == "mbpp-no-context":
        task_loader = MBPP(inst_token=args.inst_token,
                                assist_token=args.assist_token,
                                mode="no-context")
    else:
        raise NotImplementedError("Not support task `{}` yet".format(args.task))
    
    evaluator = Evaluator(task=task_loader,
                        model_name=args.model_name,
                        peft_model=args.peft_model,
                        batch_size=args.batch_size,
                        cache_dir=args.cache_dir,
                        save_dir=args.save_dir)

    output = evaluator.generate(
        backend=args.backend,
        num_return_sequences=args.num_return_sequences,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty
    )
    
    print("======= Finish generated =======")


if __name__ == '__main__':
    main()