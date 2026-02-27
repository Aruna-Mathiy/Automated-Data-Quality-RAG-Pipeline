import pandas as pd
import numpy as np
import logging
from openai import OpenAI
import os

# LOGGING INFO

logging.basicConfig(
    filename="validation_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Pipeline started.")

client = OpenAI("")

RULES = [
    "Name must not be empty",
    "Email must contain '@'",
    "Age must be greater than 18"
]

# DATA CLEANING
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Starting data cleaning...")

    try:
        df = df.copy()

        # Trim whitespace from all string columns
        string_cols = df.select_dtypes(include="object").columns
        df[string_cols] = df[string_cols].apply(lambda col: col.str.strip())

        # Replace empty strings with NA
        df.replace("", pd.NA, inplace=True)

        # Standardize name formatting
        if "name" in df.columns:
            df["name"] = df["name"].str.title()

        # Clean email formatting (remove internal spaces)
        if "email" in df.columns:
            df["email"] = df["email"].str.replace(" ", "", regex=False)

        #  Convert age to numeric
        if "age" in df.columns:
            df["age"] = pd.to_numeric(df["age"], errors="coerce")

        # Fill missing job titles
        if "job_title" in df.columns:
            df["job_title"] = df["job_title"].fillna("Unknown")

        # Drop rows where all fields are missing
        df.dropna(how="all", inplace=True)

        logging.info("Data cleaning completed.")
        return df

    except Exception as e:
        logging.error(f"Data cleaning failed: {e}")
        raise


# EMBEDDING INFO

def get_embedding(text):
    try:
        logging.info(f"Generating embedding for: {text}")
        resp = client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return resp.data[0].embedding
    except Exception as e:
        logging.error(f"Embedding error for '{text}': {e}")
        return None

def embed_rules(rules):
    logging.info("Embedding rules...")
    return [get_embedding(r) for r in rules]

# RULE EMBEDDING

def initialize_rules():
    rule_emb = embed_rules(RULES)

    if any(r is None for r in rule_emb):
        logging.error("Rule embedding failed. Exiting.")
        raise Exception("Rule embedding failed")

    return rule_emb

RULE_EMB = initialize_rules()


# COSINE SIMILARITY

def cosine_similarity(a, b):
    try:
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    except Exception as e:
        logging.error(f"Cosine similarity error: {e}")
        return -1

def best_matching_rule(text):
    logging.info(f"Selecting best rule for: {text}")
    q_emb = get_embedding(text)

    if q_emb is None:
        logging.error("Query embedding failed. Defaulting to first rule.")
        return RULES[0]

    #scores = [cosine_similarity(q_emb, r_emb) for r_emb in RULE_EMB]
    scores = []
    for r_emb in RULE_EMB:
        if r_emb is None:
            scores.append(-1)
        else:
            scores.append(cosine_similarity(q_emb, r_emb))
    best_rule = RULES[int(np.argmax(scores))]

    logging.info(f"Best rule selected: {best_rule}")
    return best_rule

# LLM VALIDATION

def validate_row(row):
    row_text = f"Name: {row['name']}, Email: {row['email']}, Age: {row['age']}"
    logging.info(f"Validating row: {row_text}")

    rule = best_matching_rule(row_text)

    prompt = f"""
    Validate this data:
    {row_text}

    Based on rule:
    {rule}

    Reply only: VALID or INVALID with reason.
    """

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        result = resp.choices[0].message.content
        logging.info(f"Validation result: {result}")
        return result

    except Exception as e:
        logging.error(f"LLM validation failed for row '{row_text}': {e}")
        return "ERROR: LLM validation failed"

# PIPELINE
def run_pipeline(df):
    logging.info("Running full validation pipeline...")
    df = clean_data(df)
    df["validation_result"] = df.apply(validate_row, axis=1)
    logging.info("Pipeline completed successfully.")
    return df

# SAMPLE DATA

data = [
    {"name": "John", "email": "john@email.com", "age": 25, "job_title": "Data Analyst"},
    {"name": "Sarah", "email": "", "age": 17, "job_title": ""},
    {"name": "Mike", "email": "mikeemail.com", "age": 30, "job_title": "Developer"},
    {"name": "Alice", "email": "alice@email.com", "age": 22, "job_title": ""},
    {"name": "Bob", "email": "bob@email", "age": 16, "job_title": "Intern"},
]

df = pd.DataFrame(data)
df = run_pipeline(df)
print(df)
