from datasets import load_dataset
import pandas as pd

def prepare_data():
    dataset = load_dataset("Abirate/english_quotes")
    df = pd.DataFrame(dataset["train"])

    # Handle missing values
    df = df.dropna(subset=["quote", "author", "tags"])

    # Convert tags list to string
    df["tags"] = df["tags"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else x
    )

    # Create embedding text
    df["embedding_text"] = (
        "Quote: " + df["quote"] +
        " Author: " + df["author"] +
        " Tags: " + df["tags"]
    )

    df.to_csv("data/processed_quotes.csv", index=False)
    print("Data saved to data/processed_quotes.csv")

if __name__ == "__main__":
    prepare_data()
