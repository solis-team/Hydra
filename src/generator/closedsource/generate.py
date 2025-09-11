import openai
import argparse
import json
import os
from tqdm import tqdm
import logging


class Chatter:
    def __init__(self, system_message: str, retries: int = 3, timeout: int = 3):
        self.__retries = retries
        self.__timeout = timeout
        self.__system_message = system_message
        self.__configure_openai()

    @property
    def system_message(self):
        return self.__system_message

    def __configure_openai(self):
        try:
            openai.api_key=os.getenv("OPENAI_API_KEY")
            openai.api_base=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        except AttributeError as e:
            logging.error(f"[Chatter] - Configuration error: {e}")
            raise ValueError("Missing OpenAI API key in settings")

    def call_openai_api(self, query: str) -> dict:
        for attempt in range(self.__retries):
            try:
                res = openai.ChatCompletion.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4"), 
                    temperature=0.2,
                    timeout=self.__timeout,
                    messages=[
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": query},
                    ],
                )
                return {"message": str(res["choices"][0]["message"]["content"])}
            except Exception as e:
                logging.error(f"[Chatter] - An exception has occurred. {e}")
        return {"message": ""}

    def postprocess_code_completion(self, completion: str, lan: str = "python") -> str:
        if f"```{lan}" in completion:
            completion = completion[completion.find(f"```{lan}") + len(f"```{lan}") :]
            completion = completion[: completion.find("```")]
        else:
            logging.error("Error: No code block found")
        return completion

    def chat(self, query: str) -> str:
        res = self.call_openai_api(query)["message"]
        return res


chatter = Chatter(
    system_message="You are a helpful coding assistant."
)

def main():
    parser = argparse.ArgumentParser(description="Generate code completion using OpenAI API")
    parser.add_argument("--benchmark", choices=["RepoExec", "DevEval"], required=True, help="Benchmark to process")
    parser.add_argument("--candidate_count", type=int, default=5, help="Number of candidates to generate")
    args = parser.parse_args()
    
    if args.benchmark == "RepoExec":
        prompt_file = "data/prompt/RepoExec_prompt.jsonl"
        output_dir = "data/generation/RepoExec"
        output_file = os.path.join(output_dir, "repoexec.final.generated.jsonl")
    else:
        prompt_file = "data/prompt/DevEval_prompt.jsonl"
        output_dir = "data/generation/DevEval"
        output_file = os.path.join(output_dir, "deveval.final.generated.jsonl")
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(prompt_file, "r", encoding="utf-8") as f_in:
        for line in tqdm(f_in, desc=f"Processing {args.benchmark}"):
            line = line.strip()
            if not line:
                continue
            
            sample = json.loads(line)
            with open(output_file, "a", encoding="utf-8") as f_out:
                response_texts = []
                for i in range(args.candidate_count):
                    res = chatter.chat(sample["prompt"])
                    response_texts.append(res)
                
                output_line = {
                    "task_id": sample["id"],
                    "prompt": sample["prompt"],
                    "response": response_texts
                }
                f_out.write(json.dumps(output_line) + "\n")
    
    print(f"Done generating. Output saved to {output_file}")


if __name__ == "__main__":
    main()


