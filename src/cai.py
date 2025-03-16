import yfinance as yf
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import re
import nltk
import string
from nltk.tokenize import word_tokenize

nltk.download("punkt")


# Function to download and preprocess financial data for a given company ticker
def download_financials(ticker, start_date, end_date):
    company = yf.Ticker(ticker)
    data = company.history(start=start_date, end=end_date)
    data = data[["Open", "High", "Low", "Close", "Volume"]]
    data.ffill(inplace=True)
    return data


# Function to structure and clean data for RAG model retrieval
def structure_data_for_retrieval(data):
    # Convert the index to datetime if not already in that format
    data.index = pd.to_datetime(
        data.index, errors="coerce"
    )  # Coerce errors to NaT (Not a Time)
    # Handle any missing dates or columns by forward-filling or backward-filling
    data.ffill(inplace=True)

    # Extract Year and Quarter from the datetime index
    data["Year"] = data.index.year
    data["Quarter"] = data.index.quarter

    # Group data by Year and Quarter, then apply aggregation functions
    data_grouped = data.groupby(["Year", "Quarter"]).agg(
        {
            "Open": "mean",  # Average opening price
            "High": "mean",  # Average highest price
            "Low": "mean",  # Average lowest price
            "Close": "mean",  # Average closing price
            "Volume": "sum",  # Total volume of shares traded
        }
    )

    # Optionally: You can reset index if needed for easier manipulation
    data_grouped.reset_index(inplace=True)
    return data_grouped


def advanced_multi_stage_rag_model(financial_data, chunk_size="sentence"):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Generate structured financial reports from the data
    financial_reports = []
    for (year, quarter), group in financial_data.groupby(["Year", "Quarter"]):
        report = {
            "year": year,
            "quarter": quarter,
            "open": group["Open"].mean(),
            "close": group["Close"].mean(),
            "high": group["High"].max(),
            "low": group["Low"].min(),
            "volume": group["Volume"].sum(),
        }
        financial_reports.append(report)

    # Group text chunks for each financial metric
    text_chunks = []
    for report in financial_reports:
        text_chunks.append(
            [
                f"For {report['year']} Q{report['quarter']} the company had:",
                f" - Average Opening Price: {report['open']:.2f}",
                f" - Average Closing Price: {report['close']:.2f}",
                f" - Highest Price: {report['high']:.2f}",
                f" - Lowest Price: {report['low']:.2f}",
                f" - Total Trading Volume: {report['volume']}",
            ]
        )

    # Flatten grouped chunks for tokenization
    grouped_text_chunks = [" ".join(chunk) for chunk in text_chunks]

    # Initialize BM25
    tokenized_chunks = [
        word_tokenize(chunk)
        for chunk in grouped_text_chunks
        if chunk not in string.punctuation
    ]
    bm25 = BM25Okapi(tokenized_chunks)

    # Embed each chunk using the pre-trained model
    embeddings = model.encode(grouped_text_chunks, convert_to_tensor=True)

    # Create a FAISS index to store and retrieve the embeddings
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(embeddings.cpu().numpy())  # Add embeddings to FAISS index

    return faiss_index, bm25, grouped_text_chunks, model


def extract_year_and_quarter(query):
    """
    Extracts the year and quarter from the user's query using regex.
    Supports formats like 'Q1 2023', '2023 Q1',
    'What is the volume for Q1 2023?', etc.
    """
    match = re.search(r"(Q[1-4])\s*(\d{4})", query)
    # Match Q1 2023, Q2 2023, etc.

    if not match:
        # Try the reverse: 2023 Q1
        match = re.search(r"(\d{4})\s*(Q[1-4])", query)
        # Match 2023 Q1, 2024 Q2, etc.

    if match:
        if "q".lower() in match.group(2).lower():
            return match.group(2), match.group(1)
        # (quarter, year) to match "Q1 2023"
        else:
            return match.group(1), match.group(2)
    else:
        return None, None  # No match found


# Multi-Stage Retrieval with BM25 + FAISS + Re-ranking
def multi_stage_retrieve_and_rerank(query, faiss_index, bm25, text_chunks, model, k=1):
    # Stage 1: BM25-based Retrieval
    tokenized_query = [
        token for token in word_tokenize(query) if token not in string.punctuation
    ]

    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_k_indices = np.argsort(bm25_scores)[-k:][::-1]
    # Get top-k BM25 indices
    # Retrieve the top-k BM25 results (keyword-based)
    bm25_top_k_chunks = [text_chunks[i] for i in bm25_top_k_indices]

    # Stage 2: FAISS-based Retrieval
    query_embedding = model.encode([query], convert_to_tensor=True)
    D, T = faiss_index.search(query_embedding.cpu().numpy(), k)
    # Retrieve using FAISS

    # Retrieve the top-k FAISS results (semantic-based)
    faiss_top_k_chunks = [text_chunks[i] for i in T[0]]

    # Stage 3: Combine results from BM25 and FAISS
    combined_chunks = list(set(bm25_top_k_chunks + faiss_top_k_chunks))

    # Stage 4: Re-ranking by cosine similarity
    combined_embeddings = model.encode(combined_chunks, convert_to_tensor=True)
    cosine_similarities = cosine_similarity(
        query_embedding.cpu().numpy(), combined_embeddings.cpu().numpy()
    )

    # Re-rank the combined chunks based on cosine similarity
    reranked_chunks = [
        combined_chunks[i] for i in np.argsort(cosine_similarities[0])[::-1]
    ]
    # Extract year and quarter from the query to filter the relevant chunks
    quarter, year = extract_year_and_quarter(query)
    filtered_chunks = []
    filtered_chunks = []
    for chunk in reranked_chunks:
        if f"{year}" in chunk and f"{quarter}" in chunk:
            filtered_chunks.append(chunk)
        elif f"{year}" in chunk and quarter is None:
            filtered_chunks.append(chunk)
        elif f"{quarter}" in chunk and year is None:
            filtered_chunks.append(chunk)

    # Return safely with a fallback if no match is found
    return filtered_chunks, cosine_similarities[0]


# Return top-k re-ranked chunks


def validate_user_query(query):
    # Input-side guardrail: Remove offensive,
    # irrelevant, or non-financial queries
    if not query or len(query.split()) < 2:
        return (
            False,
            """Query is too short or empty.
    Please enter a more specific query.""",
        )

    # Example of checking for harmful content
    harmful_keywords = ["hack", "malware", "scam", "fraud"]
    if any(keyword in query.lower() for keyword in harmful_keywords):
        return (
            False,
            """ Query contains harmful content.
            Please ask a legitimate question.""",
        )

    # Example of checking if the query is related to financial topics
    financial_keywords = [
        "revenue",
        "profit",
        "volume",
        "quarter",
        "earnings",
        "sales",
        "stock price",
        "opening price",
        "closing price",
        "trading volume",
        "high price",
        "low price",
        "average",
        "increase",
        "decrease",
    ]
    if not any(keyword.lower() in query.lower() for keyword in financial_keywords):
        return (
            False,
            """Query does not seem to relate to financial topics.
            Please ask a relevant financial question like
            'What was the average stock price for Q1 2023?'
            or 'What is the EPS for Q2 2023?'""",
        )

    # Check if the query asks for data that
    # can be computed from available financial data
    valid_question_keywords = [
        "average",
        "increase",
        "decrease",
        "what",
        "how",
        "stock price",
        "volume",
        "trading volume",
        "quarter",
        "year",
    ]
    if not any(keyword.lower() in query.lower() for keyword in valid_question_keywords):
        return (
            False,
            """Query is not formulated in a way that can be
            answered based on the financial data we have.
            Please ask something like
            'What was the average opening stock price for Q1 2023?'
            or 'How did the trading volume change in Q2 2023?'""",
        )

    return True, ""


def extract_metric_from_query(query):
    # Use regex to extract the financial metric from the query
    # Example: 'What was the volume for Q4 2023'
    match = re.search(r"(volume|open|close|high|low)", query.lower())
    print(match)
    if match:
        return match.group(1)  # Return the matched metric
    return None


def extract_metric_from_chunk(chunk, metric):
    if metric.lower() == "volume":
        match = re.search(r"Total Trading Volume: ([\d,]+)", chunk)
    elif metric.lower() == "open":
        match = re.search(r"Average Opening Price: ([\d\.]+)", chunk)
    elif metric.lower() == "close":
        match = re.search(r"Average Closing Price: ([\d\.]+)", chunk)
    elif metric.lower() == "high":
        match = re.search(r"Highest Price: ([\d\.]+)", chunk)
    elif metric.lower() == "low":
        match = re.search(r"Lowest Price: ([\d\.]+)", chunk)
    else:
        return "Metric not found"

    return match.group(1) if match else "Metric not found"


# Guardrail for output validation
def filter_output_answer(answer, query, model, threshold=0.3):
    metric = extract_metric_from_query(query)
    # If no specific metric is found in the query, return an error
    if not metric:
        return "No relevant metric found in the query."

    # Output-side guardrail: Ensure answer is not hallucinated or misleading
    answer_embedding = model.encode([answer], convert_to_tensor=True)
    query_embedding = model.encode([query], convert_to_tensor=True)
    cosine_sim = cosine_similarity(
        query_embedding.cpu().numpy(), answer_embedding.cpu().numpy()
    )[0][0]
    if cosine_sim < threshold:
        return """The answer seems irrelevant or hallucinated.
            Please ask a different query."""
    # If the metric matches, return the answer; else, return an error
    final_ans = extract_metric_from_chunk(answer, metric)
    return final_ans


# Test function for running pre-defined questions (Updated for Streamlit)
def run_test_cases(faiss_index, bm25, text_chunks, model):
    # Define test questions based on the financial reports and data
    test_questions = {
        "High Confidence Relevant Financial Question": {
            "question": "What was the Volume for Q4 2023?",
            "expected": "Volume for Q4 2023 is 200350.",
            # Replace with actual data
        },
        "Low Confidence Relevant Financial Question": {
            "question": "What was the revenue for Q3 2023?",
            "expected": "Revenue for Q3 2023 is $4.5 billion.",
            # Replace with actual data
        },
        "Irrelevant Question": {
            "question": "What is the capital of France?",
            "expected": "Irrelevant, should return no relevant answers.",
        },
        "Quarterly Profit Check": {
            "question": "What was the profit for Q1 2023?",
            "expected": """No direct profit data available,
            but you can check EPS or revenue for Q1 2023.""",
        },
        "Company Debt Query": {
            "question": "What is the total debt for the company in Q2 2023?",
            "expected": """ Debt information not explicitly available,
            but operational cost information is available for Q2.""",
        },
        "General Revenue Query": {
            "question": "What was the total revenue for the company?",
            "expected": """ Revenue information is available for
            specific quarters (e.g., Q1 2023, Q4 2023).""",
        },
    }

    st.write("### Test Questions & Results")
    results = []

    for test_name, test_case in test_questions.items():
        query = test_case["question"]

        reranked_chunks = multi_stage_retrieve_and_rerank(
            query, faiss_index, bm25, text_chunks, model, k=3
        )

        # Filter the results
        if reranked_chunks:
            filtered_answer = filter_output_answer(reranked_chunks[0], query, model)
            results.append((query, filtered_answer))
        else:
            results.append((query, "No relevant answers found."))

    # Display the test results
    for query, filtered_answer in results:
        st.write(f"**Question:** {query}")
        st.write(f"**Answer:** {filtered_answer}")
        st.write("---")


# Main UI with Streamlit
def main():
    # st.title("Multi-Stage RAG Model for Financial
    # Data Retrieval with Guardrails and Testing")

    # Inputs for the UI
    ticker = st.text_input("Enter the Stock Ticker:", "TGT")
    start_date = st.date_input("Start Date:", pd.to_datetime("2023-03-15"))
    end_date = st.date_input("End Date:", pd.to_datetime("2025-03-15"))

    # Download and preprocess the data
    financial_data = download_financials(ticker, start_date, end_date)
    financial_data.head()
    # Structure the data for easier retrieval
    structured_data = structure_data_for_retrieval(financial_data)

    # Initialize and run the multi-stage RAG model
    faiss_index, bm25, text_chunks, model = advanced_multi_stage_rag_model(
        structured_data
    )

    # Accept user query
    query = st.text_input("Enter your query:", "")

    if query:
        # Validate the query (Input-side guardrail)
        is_valid, error_message = validate_user_query(query)
        if not is_valid:
            st.error(error_message)
        else:
            reranked_chunks, reranked_scores = multi_stage_retrieve_and_rerank(
                query, faiss_index, bm25, text_chunks, model, k=3
            )

            # Display the relevant chunks and confidence score
            st.write("### Top Retrieved Answers:")
            for i, chunk in enumerate(reranked_chunks, 1):
                # Filter the output (Output-side guardrail)
                filtered_answer = filter_output_answer(chunk, query, model)
                st.write(
                    f"**Answer {i}:** {filtered_answer}, confidence label: {reranked_scores[i-1]:.2f}"
                )

    # Add a button to test only test cases
    if st.button("Run Test Cases"):
        st.write("### Running Test Cases")
        run_test_cases(faiss_index, bm25, text_chunks, model)


# Run the app
if __name__ == "__main__":
    main()
