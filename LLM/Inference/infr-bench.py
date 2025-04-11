import argparse
import numpy as np
import time
import csv 

from openai import OpenAI
import os


openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"


def write_csv(args,latency, tokens_per_second):
    list_1 = ["Model Name", "precision", "throughput", "latency", "batch size", "tensor_parallel", "input length", "output length","context_length"]
    list_2 = [args.model, "float16", tokens_per_second, latency, args.batch_size, args.tensor_parallel, args.input_length, args.output_length,args.context_length]

    csv_file = "results.csv"
    file_exists = os.path.exists(csv_file)

    with open(csv_file, 'a', newline = '') as csvfile:
        writer = csv.writer(csvfile)
        
        if not file_exists:
            writer.writerow(list_1)
        
        writer.writerow(list_2) 
        
    csvfile.close()

def generate_input(args):
    random_words = ["France" for _ in range(args.input_length)]

    input_id = ""

    for word in random_words:
        input_id = input_id + word + " "

    input_id = input_id[:-1]
    
    ip = {}
    ip["role"] = "user"
    ip["content"] = input_id
    # print(ip)

    input_list = []

    for batch_size in range(args.batch_size):
        input_list.append(ip)  

    return input_list



def Benchmark(args):

    # print(os.environ.get("model_name"))
    args.model=os.environ.get("model_name")
    args.tensor_parallel=os.environ.get("tp_size")
    args.context_length=os.environ.get("context_length")
    print(args)
    
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    gen_input = generate_input(args)
 

    start_time = time.perf_counter()
    for _ in range (args.itr):
        completion = client.chat.completions.create(
            model=args.model,
            messages=gen_input,
            max_completion_tokens=args.output_length,
            max_tokens=args.output_length,
        )
    end_time = time.perf_counter()
    latency = (end_time - start_time) / args.itr

    # Extract values
    total_tokens = completion.usage.total_tokens
    tokens_per_second= total_tokens / latency
    

    # Calculate tokens per second    
    # print("prompt tokens:", completion.usage.prompt_tokens)
    # print("competion_tokens: ", completion.usage.completion_tokens)
    print("tokens_per_second: ", tokens_per_second)
    

    write_csv(args,latency, tokens_per_second)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cerebras Inference Benchmark')
    
    parser.add_argument('--input-length', type=int, help='Input Length')
    parser.add_argument('--output-length', type=int, help='Output Length')
    # parser.add_argument('--tensor-parallel', type=int, help='Tensor Parallel')
    parser.add_argument('--batch-size', type=int, help='Batch Size')
    parser.add_argument('--itr', default=10, type=int, help='Number of iterations')


    # parser.add_argument('--model', type=str, help='Model Name')
    args = parser.parse_args()
    Benchmark(args)