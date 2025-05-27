import csv
from collections import defaultdict
from datasets import Dataset
import pandas as pd

def read_csv(filename, types):
    return pd.read_csv(
            filename,
            encoding='utf8',
            quoting=csv.QUOTE_ALL,
            dtype=types,
            names=types.keys(), 
        )

PROMPT = "БЕЗ ВТРАТИ змісту та НАДМІРНОГО скорочення (зберігай все важливе). відповідь надавай виключно українською мовою. ці дані для моделі детекції фейків, вони ВЖЕ позначені як неправдиві, тому не потрібно пояснювати чому. підсумуй кожну зі стрічок з максимальним обмеженням символів у 128 літер (найкраще розмір близький до цього розміру або повні речення) без втрати змісту та поверни відповідь без лапок та завершальної крапки, кожна стрічка у новому рядку:\n" 


directory = 'fake_news_dataset/'
datasets = {}
dfs = {}
NAMES = ['test', 'train', 'val']
for name in NAMES:
    df = read_csv(f'{directory}{name}_bad.csv', {'headline': str})
    dfs[name] = df
    datasets[name] = Dataset.from_pandas(df)

def list_to_csv(lines: list[str], filename: str):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        for line in lines:
            writer.writerow([line])

def save_summarized_to_csv(summarized: defaultdict[list[str]], suffix):
    for key, lines in summarized.items():
        filename = f"{directory}{key}_{suffix}.csv"
        list_to_csv(lines, filename)

def summarize_with_chatgpt():
    from UnlimitedGPT import ChatGPT
    from tenacity import retry, stop_after_attempt, wait_fixed
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time
    import tqdm

    def print_error(retry_state):
        print(f"Error: {retry_state.outcome.exception()}")

    @retry(stop=stop_after_attempt(20), wait=wait_fixed(60), retry_error_callback=print_error)
    def get_response(bot: ChatGPT, prompt: str):
        return bot.send_message(prompt).response

    def summarize_rows(chunk: list[str], bot: ChatGPT, prompt: str):
        for row in chunk:
            prompt += row + "\n"
        
        response = get_response(bot, prompt)
        print(response)
        summarized_lines = response.split("\n")

        return summarized_lines
    
    ACCESS_TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6Ik1UaEVOVUpHTkVNMVFURTRNMEZCTWpkQ05UZzVNRFUxUlRVd1FVSkRNRU13UmtGRVFrRXpSZyJ9.eyJwd2RfYXV0aF90aW1lIjoxNzI4NzMxNjIzNDAwLCJzZXNzaW9uX2lkIjoiZml1VDVERWoyZktMbjhpSFJ2RVZ5SW9HOHhPU1hhR3MiLCJodHRwczovL2FwaS5vcGVuYWkuY29tL3Byb2ZpbGUiOnsiZW1haWwiOiIwNG11bmEwNEBnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZX0sImh0dHBzOi8vYXBpLm9wZW5haS5jb20vYXV0aCI6eyJwb2lkIjoib3JnLWpyclNyYTdOSFlOMHA1ZzdDWVZJaVB4SCIsInVzZXJfaWQiOiJ1c2VyLWw0TUQ2ZUdXYk9yTnBHbzNzbFJ2UGQ1USJ9LCJpc3MiOiJodHRwczovL2F1dGgwLm9wZW5haS5jb20vIiwic3ViIjoiZ29vZ2xlLW9hdXRoMnwxMTc3MTk1ODM4ODExNzA1MzI0MzkiLCJhdWQiOlsiaHR0cHM6Ly9hcGkub3BlbmFpLmNvbS92MSIsImh0dHBzOi8vb3BlbmFpLm9wZW5haS5hdXRoMGFwcC5jb20vdXNlcmluZm8iXSwiaWF0IjoxNzI4NzMxNjI0LCJleHAiOjE3Mjk1OTU2MjQsInNjb3BlIjoib3BlbmlkIHByb2ZpbGUgZW1haWwgbW9kZWwucmVhZCBtb2RlbC5yZXF1ZXN0IG9yZ2FuaXphdGlvbi5yZWFkIG9yZ2FuaXphdGlvbi53cml0ZSBvZmZsaW5lX2FjY2VzcyIsImF6cCI6IlRkSkljYmUxNldvVEh0Tjk1bnl5d2g1RTR5T282SXRHIn0.Yma4GMaLYPyVh8ltonTM-KuAPZH23C26zLBGRZT-eU9hNAwNnrO5uFysGyNUI6sJe4-ZBkj8NxO8XkbFNqQmjKMUhLzn1ap_3NgEEjX1qjxWL9aY_ClUUVIIevygvPwg0kx-qAVZyhHoZxIGfzXyEnHTe67C-v4RtyXDc6EqsAydH-5a9OPxJDW86ExuqxiYhlJifY8qgltbRbB2S5VEUpQqqQtEfn5CjIp54vl4HjxTsNoSCunMd_WBooIq56-3lXX9QiLLLu6fvptRaOr4Ro0s2wf5zPSeR93iF4xeA4s4OUp1hy2nWqXcs6wvjX-TePsa4e9M6IePwLH6laW4_A"
    RATE_LIMIT = 20
    MAX_CHAR_LIMIT = 4096 - len(PROMPT)
    BOT = ChatGPT(ACCESS_TOKEN)
    start_idx = 0
    summarized = defaultdict(list)
    # Continue adding rows until we reach the character limit

    def process_dataframe(name, df):
        start_idx = 0
        while start_idx < len(df):
            chunk = []
            total_chars = 0
            while start_idx < len(df) and total_chars + len(df['headline'].iloc[start_idx]) < MAX_CHAR_LIMIT:
                row_text = df['headline'].iloc[start_idx]
                chunk.append('* ' + row_text)
                total_chars += len(row_text)
                start_idx += 1
            summarized_chunk = summarize_rows(chunk, BOT, PROMPT)
            summarized[name].extend(summarized_chunk)
            time.sleep(60 / RATE_LIMIT)
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_dataframe, name, df) for name, df in dfs.items()]
        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()
    save_summarized_to_csv(summarized, 'chatgpt')

def summarize_with_t5_model():
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration, pipeline

    tokenizer = AutoTokenizer.from_pretrained('ukr-models/uk-summarizer')
    model = T5ForConditionalGeneration.from_pretrained('ukr-models/uk-summarizer')
    device = 0 if torch.cuda.is_available() else -1

    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, max_length=128, num_beams=4, no_repeat_ngram_size=2, clean_up_tokenization_spaces=True, device=device)

    # Function to shorten text using AI summarization
    def shorten_text(batch, max_len=128):
        headlines = batch['headline']
        summaries = summarizer(headlines, max_length=max_len//2, min_length=30, do_sample=False)
        summary_texts = [summary['summary_text'] for summary in summaries]
        print(summary_texts)
        return {'headline': summary_texts}

    for name, dataset in datasets.items():
        dataset = dataset.map(shorten_text, batched=True, batch_size=8)
        dataset.to_csv(f'{directory}{name}_good.csv', index=False, header=False, quoting=csv.QUOTE_ALL, escapechar='\\')

def split_into_chunks():
    results = defaultdict(list)

    # Step 3: Function to split dataframe into chunks of size 63
    def split_into_chunks(data, chunk_size=16):
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]

    # Step 4: Function to process a chunk (prepend * to each item and join with \n)
    def process_chunk(chunk):
        return '\n'.join([f"*{item}" for item in chunk])

    # Step 5: Process each dataframe and save the result in `results` dict
    for name, df in dfs.items():
        # Here I'm assuming you're processing the first column (adapt as needed)
        column_data = df[df.columns[0]].astype(str).tolist()  # Convert the first column to a list of strings
        chunks = split_into_chunks(column_data)
        
        for chunk in chunks:
            processed_chunk = process_chunk(chunk)
            results[name].append(PROMPT + processed_chunk)
    
    save_summarized_to_csv(results, 'split')

def summarize_chunks_with_gemini():
    from gemini import Gemini
    import os
    os.environ["GEMINI_LANGUAGE"] = "UK" 

    results = defaultdict(list)

    split_dfs = {
        name : read_csv(f'{directory}{name}_split.csv', {'headline': str})
        for name in NAMES
    }
    
    proxy_url = "http://BKQ7DzV8q0ayF3ecmjfnlw:@smartproxy.crawlbase.com:8012" 
    proxies = {"http": proxy_url, "https": proxy_url}
    client = Gemini(auto_cookies=True, target_cookies = ["__Secure-1PSIDCC", " __Secure-1PSID", "__Secure-1PSIDTS", "NID"], proxies=proxies, timeout=30, verify=False) 
    for name, df in split_dfs.items():
        for row in df['headline']:
            response = client.generate_content(row)
            payload = response.payload
            print(payload)
            results[name].append(payload)
        save_summarized_to_csv(results[name], f"{directory}{name}_gemini.csv")

# summarize_chunks_with_gemini()

def read(file_path, types):
    df = pd.read_csv(
        file_path,
        encoding='utf8',
        # quoting=QUOTE_ALL,
        # dtype=types,
        names=types.keys(), 
    )
    return df

from sklearn.model_selection import train_test_split

types={'headline': str, 'label': int}
dir = 'fake_news_dataset/'
train_df = read(f'{dir}train.csv', types)
val_df = read(f'{dir}val.csv', types)
test_df = read(f'{dir}test.csv', types)
val_df, temp_val_df = train_test_split(val_df, test_size=0.4, random_state=42)
test_df, temp_test_df = train_test_split(test_df, test_size=0.4, random_state=42)

train_df = pd.concat([train_df, temp_val_df, temp_test_df])

train_df.to_csv(f'{dir}train.csv', index=False, header=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
val_df.to_csv(f'{dir}val.csv', index=False, header=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
test_df.to_csv(f'{dir}test.csv', index=False, header=False, quoting=csv.QUOTE_ALL, encoding='utf-8')