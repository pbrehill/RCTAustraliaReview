import pandas as pd
import openai
from tqdm import tqdm
from config import OPENAI_API_KEY
import logging

openai.api_key = OPENAI_API_KEY

logging.basicConfig(filename='api_costs.log', level=logging.INFO)
pricing = 5.00 / 1000000

def calculate_cost(response):
    token_count = response.usage.total_tokens
    cost = token_count * pricing
    return cost, token_count

def classify_papers(csv_file):
    # Load the CSV file
    data = pd.read_csv(
        csv_file,
        dtype=str,
    )

    data = data.fillna('')

    # Concatenate the Title, Abstract, and Keywords
    prompt_bodies = "Title: " + data['Title'] + "\n\n" + \
                   "Abstract: " + data['Abstract'] + "\n\n" + \
                   "Keywords: " + data['Author Keywords'] + data['Index Keywords']

    # Iterate over each row and send requests to OpenAI
    results = []
    total_cost = 0
    for text in tqdm(prompt_bodies):
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "Classify the following description"},
                      {"role": "user", "content": text + "\n\nIs this an Australian randomised controlled trial or field experiment? Please start your answer with 'Yes' or 'No'"}],
            max_tokens=30,
            temperature=0
        )
        cost, token_count = calculate_cost(response)
        logging.info(f"Prompt: {prompt_bodies}, Tokens used: {token_count}, Cost: ${cost:.4f}")
        total_cost += cost
        print(total_cost)

        results.append(response.choices[0].message.content)

    # Add the results to the dataframe
    data['Classification'] = results
    return data


# Example usage
classified_data = classify_papers('input.csv')
classified_data.to_excel('classified_data1.xlsx', index=False)
print(classified_data)
