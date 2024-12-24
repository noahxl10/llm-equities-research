import os
import asyncio
import pandas as pd

# Local imports (adjust as necessary depending on your project structure)
# Assumes that 'Gemini' is accessible via model_wizards
from ..wizard import wizard, model_wizards

###############################################################################
# Global paths / constants (replace with actual file paths as needed)
###############################################################################
RESPONSE_FILE_PATH = ""                # TODO: Set valid path
QUESTION_AND_ANSWERS_FILE_PATH = ""    # TODO: Set valid path
RED_FLAGS_OUTPUT_FILE = ""             # TODO: Set valid path for remove_red_flags_extra_cols()

###############################################################################
# Data Retrieval & Construction
###############################################################################
def build_red_flags():
    """
    Reads in response data and question/answer data, merges where result_number < 0
    to identify 'red flags' from the dataset.
    """
    responses = pd.read_csv(RESPONSE_FILE_PATH)
    qa_df = pd.read_csv(QUESTION_AND_ANSWERS_FILE_PATH)

    red_flag_responses = responses[responses["result_number"] < 0]
    red_flags_df = red_flag_responses.merge(qa_df, how="left", on="question")
    return red_flags_df

###############################################################################
# Async Processing
###############################################################################
async def fetch_responses(gemini_pro_model, df_index, red_flag_text, company_name):
    """
    Sends prompts to 'gemini_pro_model' for the specified row (index=df_index).
    Returns the row index, attribute response, engagement response, and any
    failure details.
    """
    try:
        # Prompts up to 65 characters (including spaces).
        attribute_prompt = (
            f"In no more than 65 characters (including spaces), please answer the following question: "
            f"What is the clearest way to explain what specifically makes the following statement "
            f"true of {company_name}? {red_flag_text}"
        )
        engagement_prompt = (
            "In no more than 65 characters (including spaces), please answer the following question: "
            "What is the clearest way to explain what action this company’s board might hypothetically "
            "be able to take in order to neutralize the negative implications of the above statement "
            "you explained, for the benefit of the company’s shareholders?"
        )

        # Send both prompts in one call
        attribute_response, engagement_response = await gemini_pro_model.attr_eng_chat_context(
            attribute_prompt,
            engagement_prompt
        )

        return (df_index, attribute_response, engagement_response)

    except Exception as e:
        # Return index and mark failure
        return (df_index, e, None, 1, str(e))

async def run_red_flags(red_flags_df):
    """
    Iterates over 'red_flags_df' in batches of 10 rows at a time, fetching
    attribute and engagement responses asynchronously from the 'Gemini' model.
    Updates the DataFrame with the results and any failures.
    """
    # Instantiate the model from model_wizards
    gemini_pro_model = model_wizards.Gemini()

    # Ensure necessary columns exist
    red_flags_df["attribute_response"] = ""
    red_flags_df["engagement_response"] = ""
    red_flags_df["failures"] = 0
    red_flags_df["failure_message"] = ""

    batch_size = 10
    total_rows = len(red_flags_df)

    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)
        tasks = []

        # Create async tasks for the current batch
        for index in range(start, end):
            current_row = red_flags_df.iloc[index]
            red_flag_text = current_row["n_answer"]
            company_name = current_row["company"]

            task = fetch_responses(gemini_pro_model, index, red_flag_text, company_name)
            tasks.append(asyncio.ensure_future(task))

        # Gather all results for this batch
        results = await asyncio.gather(*tasks)

        # Update the DataFrame based on the results
        for result in results:
            df_index, attribute_resp, engagement_resp, *extra = result
            failure_flag = extra[0] if extra else None
            failure_msg = extra[1] if len(extra) > 1 else None

            if attribute_resp and engagement_resp:
                red_flags_df.at[df_index, "attribute_response"] = attribute_resp
                red_flags_df.at[df_index, "engagement_response"] = engagement_resp
            if failure_flag:
                red_flags_df.at[df_index, "failures"] = failure_flag
                red_flags_df.at[df_index, "failure_message"] = failure_msg

    return red_flags_df

###############################################################################
# Post-Processing
###############################################################################
def remove_red_flags_extra_cols():
    """
    Reads a CSV of red flags, filters specific columns, measures response length,
    and writes out a final CSV.
    """
    df = pd.read_csv(RED_FLAGS_OUTPUT_FILE)

    # Keep only the columns we need
    df = df[
        [
            "company",
            "question",
            "response",
            "result_number",
            "y_answer",
            "n_answer",
            "attribute_response",
            "engagement_response",
        ]
    ]

    # Track character length of final responses
    df["length_of_attribute_response"] = df["attribute_response"].apply(
        lambda x: len(x) if x else 0
    )
    df["length_of_engagement_response"] = df["engagement_response"].apply(
        lambda x: len(x) if x else 0
    )

    # Example output path
    output_path = "/src/rsrcs/red_flags_output_final.csv"
    df.to_csv(output_path, index=False)