import openai
import argparse
import json
import os
from tqdm import tqdm
import logging


class Chatter:
    def __init__(self, system_message: str, model_name: str, retries: int = 3, timeout: int = 3):
        """Chatter wraps OpenAI ChatCompletion calls.

        Args:
            system_message: the system message passed to the model
            model_name: name of the OpenAI model to use (e.g. 'gpt-4')
            retries: number of retry attempts on error
            timeout: request timeout in seconds
        """
        self.__retries = retries
        self.__timeout = timeout
        self.__system_message = system_message
        self.__model_name = model_name
        self.__configure_openai()

    @property
    def system_message(self):
        return self.__system_message

    def __configure_openai(self):
        # Configure API key and base URL from environment. Model is supplied via constructor.
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logging.error("[Chatter] - Missing OPENAI_API_KEY environment variable")
            raise ValueError("Missing OpenAI API key in environment (OPENAI_API_KEY)")
        openai.api_key = api_key
        openai.api_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    def call_openai_api(self, query: str) -> dict:
        for attempt in range(self.__retries):
            try:
                res = openai.ChatCompletion.create(
                    model=self.__model_name,
                    temperature=0.2,
                    timeout=self.__timeout,
                    messages=[
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": query},
                    ],
                )
                return {"message": str(res["choices"][0]["message"]["content"]) }
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


def main():
    parser = argparse.ArgumentParser(description="Generate code completion using OpenAI API")
    parser.add_argument("--benchmark", choices=["RepoExec", "DevEval"], required=True, help="Benchmark to process")
    parser.add_argument("--candidate_count", type=int, default=5, help="Number of candidates to generate")
    parser.add_argument("--model", required=True, help="OpenAI model name to use (e.g. 'gpt-4')")
    args = parser.parse_args()

    # instantiate chatter with model passed as CLI argument instead of reading from env
    chatter = Chatter(system_message="You are a helpful coding assistant.", model_name=args.model)

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


