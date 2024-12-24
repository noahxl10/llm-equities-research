from src._types import Model

models = {
    "OpenAI": Model(
        owner="OpenAI",
        qualified_api_name="gpt-4o",
        model_type="Chatbot",
        guid="6d2fd171-1022-46ea-9fc5-62e50d6f29f0",
        max_tokens=0,
        token_limit=0,
        requests_per_batch=0,
    ),
    "Gemini": Model(
        owner="Google",
        qualified_api_name="gemini-1.5-pro",
        model_type="Chatbot",
        guid="de3d6d3d-f9c6-450a-a2b6-f0213940b54f",
        max_tokens=0,
        token_limit=0,
        requests_per_batch=0,
    ),
}


# PATHS
# Removed for privacy
static_path = ""

financial_model_path = ""
financial_model_file_name = ""
financial_model_file_name_only = ""

one_pager_path = ""
one_pager_file_name = ""
one_pager_file_name_only = ""

evidence_synthesis_path = ""
evidence_synthesis_file_name = ""
evidence_synthesis_file_name_only = ""

engine_blob_path = ""
financial_model_blob_path = ""

binaried_output_base_path = ""

openai_financial_model_config = {
    "BURL": {
        "assistant_id": "",
        "vector_store_id": "",
    }
}

ALLOWED_EXTENSIONS = {"csv"}


PROXIES = {
   # REMOVED FOR PRIVACY
}

SENTIMENT_DICTIONARY = {
    "Positive": ["yes ", "yes,", "yes.", "yes"],
    "Negative": ["no ", "no,", "no.", "no"],
    "Neutral": [
        "as an AI" "I don't have real-time access to specific financial data",
        "I'm unable to assist you with that.",
        "I'm a text-based AI and can't assist with that.",
        "I'm not programmed to assist with that.",
    ],
}
