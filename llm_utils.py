from xml.parsers.expat import model
import requests
from typing import Union, List
from dotenv import load_dotenv 
import os
from openai import OpenAI
import csv
import json
import re

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "llama2-7b-chat")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
TIMEOUT = int(os.getenv("TIMEOUT", 60))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))


def fd_classify_prompt(lhs: Union[str, List[str]], rhs: str, table_name: str = None, lhs_desc: str = None, rhs_desc: str = None) -> str:
    if isinstance(lhs, str):
        lhs = [lhs]

    lhs_str = ", ".join(lhs)

    # Build prompt with conditional information sections
    prompt = f"""
You are an expert in relational database theory and data profiling.

You are given a candidate functional dependency (FD) of the form:

    LHS → RHS

where:
- LHS is a set of column names from a table
- RHS is a single column name from the same table

A functional dependency holds if, for every pair of rows in the table, whenever the values of all columns in the LHS are equal, the value of the RHS is also equal.

Task: Determine whether the following candidate functional dependency is genuine and holds for typical real-world data. A genuine functional dependency is one where the LHS columns meaningfully and logically determine the RHS based on domain knowledge and business rules. It represents a stable, expected relationship that should hold across real-world instances of the data, not merely a statistical coincidence of the current dataset.


Candidate functional dependency:
    {{{lhs_str}}} → {rhs}
"""
    if table_name or lhs_desc or rhs_desc:
        prompt += "\nYou are also provided with the following information about the table to help you make your decision:"
        if table_name:
            prompt += f"\n- Table name: {table_name}"
        if lhs_desc:
            prompt += f"\n- LHS Description: {lhs_desc}"
        if rhs_desc:
            prompt += f"\n- RHS Description: {rhs_desc}"

    prompt += """

Rules:
- Answer strictly with **'1'** if the LHS functionally determines the RHS (genuine).
- Answer strictly with **'0'** if it does NOT (non-genuine).
- Do NOT explain your reasoning.
- Do NOT output anything other than a single character: '1' or '0'.
- If you are uncertain, answer **'0'**.

Answer:
"""
    return prompt.strip()

def call_ollama(prompt: str, model: str | None = None) -> str:
    url = f"{OLLAMA_API_URL}/api/chat"

    payload = {
        "model": model or LLM_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": TEMPERATURE,
        "stream": False
    }


    print(f"Calling Ollama API")
    response = requests.post(
        url,
        json=payload,
        timeout=TIMEOUT
    )
    response.raise_for_status()
    print(f"Response received from Ollama API")

    return response.json()["message"]["content"]

def call_perplexity(prompt: str) -> str:
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise ValueError("PERPLEXITY_API_KEY is not set in environment variables.") 
    
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.perplexity.ai"
    )

    resp = client.chat.completions.create(
        model="sonar-pro",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=TEMPERATURE,
        stream=False
    )

    return resp.choices[0].message.content
        
def generate_column_desc(input_csv_path: str) -> str:
    """
    Generate column descriptions from a CSV file using an LLM.
    """
    from pathlib import Path
    
    table_name = Path(input_csv_path).stem

    with open(input_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        
        columns = next(reader)

        examples = []
        for i, row in enumerate(reader):
            if i >= 10:
                break
            examples.append(row)
    
    examples_table = "Column Names: " + ", ".join(columns) + "\n\n"
    examples_table += "Example Data:\n"
    for i, row in enumerate(examples, 1):
        examples_table += f"Row {i}: {dict(zip(columns, row))}\n"
    
    prompt = f"""You are a data analyst expert. Analyze the following dataset column names and example data to infer what each column represents.

Table Name: {table_name}

{examples_table}

Based on the table name, column names, and example values provided, generate a concise description for each column. The description should explain what data the column contains and be appropriate for the {table_name} domain.

Return ONLY valid JSON (no markdown, no extra text) with the following format:
{{
    "column_name_1": "description of column 1",
    "column_name_2": "description of column 2",
    ...
}}
"""
    
    response = call_perplexity(prompt)
    
    try:
        json_obj = json.loads(response)
        return json.dumps(json_obj, indent=2)
    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                json_obj = json.loads(json_match.group())
                return json.dumps(json_obj, indent=2)
            except json.JSONDecodeError:
                return response
        return response

def generate_all_column_descriptions(data_dir: str, dest_path: str = None) -> str:
    """
    Generate column descriptions for all CSV files in a directory and combine them into one JSON.
    """
    from pathlib import Path
    
    combined_descriptions = {}
    data_path = Path(data_dir)

    csv_files = list(data_path.glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return json.dumps({})
    
    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...")
        try:
            # Generate descriptions for this CSV
            descriptions = generate_column_desc(str(csv_file))
            descriptions_dict = json.loads(descriptions)
            
            # Add to combined object with filename as key
            combined_descriptions[csv_file.stem] = descriptions_dict
            print(f"✓ Completed {csv_file.name}")
        except Exception as e:
            print(f"✗ Error processing {csv_file.name}: {str(e)}")
            continue
    
    result = json.dumps(combined_descriptions, indent=2)
    
    # Determine save path
    if dest_path is None:
        dest_path = os.path.join(os.getcwd(), "column_descriptions.json")
    elif os.path.isdir(dest_path):
        # If dest_path is a directory, append the filename
        dest_path = os.path.join(dest_path, "column_descriptions.json")
    
    # Save to file
    with open(dest_path, 'w', encoding='utf-8') as f:
        f.write(result)
    print(f"✓ Saved combined descriptions to {dest_path}")
    
    return result
    
