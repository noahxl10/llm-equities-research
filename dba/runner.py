"""
Utility script for creating database tables and seeding them with initial data.
"""

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from sqlalchemy.sql import update

from db import Engine, db_session
from data_models import (
    Base,
    Companies,
    AppUser,
    FinancialModelMeasurePromptsBase,
    OnePagerPromptsBase,
    QuestionRepositoryResearch,
)

def create_tables():
    """
    Creates all tables defined in the Base metadata on the provided Engine.
    """
    Base.metadata.create_all(Engine)

def insert_companies_from_csv():
    """
    Reads company data from a CSV file and inserts it into the Companies table.
    Updates any blank tickers to None.
    """
    csv_file_path = ""
    df = pd.read_csv(csv_file_path)
    df.fillna("", inplace=True)

    for _, row in df.iterrows():
        company = Companies(
            name=row["name"],
            ticker=row["ticker"],
            gp_ticker=row["gp_ticker"],
            strategies=row["strategies"],
        )
        db_session().add(company)

    db_session().commit()
    db_session().execute(
        update(Companies).where(Companies.ticker == "").values(ticker=None)
    )
    db_session().commit()
    db_session().close()

def insert_users():
    """
    Inserts example user accounts into the AppUser table.
    """
    # User 1
    user_1 = AppUser(email="", username="bbarth")
    user_1.set_password("")
    db_session().add(user_1)
    db_session().commit()

    # User 2
    user_2 = AppUser(email="", username="nalex")
    user_2.set_password("")
    db_session().add(user_2)
    db_session().commit()

    db_session().close()

def insert_fm_prompts_into_table():
    """
    Reads financial model measure prompts from a CSV and inserts them 
    into the FinancialModelMeasurePromptsBase table.
    """
    csv_file_path = ""
    df = pd.read_csv(csv_file_path)
    df.fillna("", inplace=True)

    for _, row in df.iterrows():
        prompt = FinancialModelMeasurePromptsBase(
            prompt_type=row["prompt_type"],
            base_column=row["excel_base_column"],
            is_yoy=row["is_yoy"],
            beginning_of_prompt=row["beginning_of_prompt"],
            core_measure_prompt=row["core_measure_prompt"],
            override_core_measure_prompt=row["override_core_measure_prompt"],
            end_of_prompt=row["end_of_prompt"],
            applicable_financial_statement=row["applicable_financial_statement"],
            example_prompt_unit=row["example_prompt_unit"],
            projection_or_historical=row["projection_or_historical"],
            time_period_type=row["time_period_type"],
            number_format_type=row["number_format_type"],
        )
        db_session().add(prompt)

    db_session().commit()
    db_session().close()

def update_prompts_base():
    """
    Reads updates to core_measure_prompt from a CSV 
    and applies them to existing FinancialModelMeasurePromptsBase records.
    """
    file_path = ""
    df = pd.read_csv(file_path)

    for _, row in df.iterrows():
        guid = row["guid"]
        core_measure_prompt = row["core_measure_prompt"]

        record = (
            db_session()
            .query(FinancialModelMeasurePromptsBase)
            .filter_by(guid=guid)
            .first()
        )

        if record:
            record.core_measure_prompt = core_measure_prompt
            db_session().commit()

    db_session().close()

def insert_one_pager_prompts_into_table():
    """
    Reads one-pager prompts from a CSV file and inserts them 
    into the OnePagerPromptsBase table.
    """
    csv_file_path = (
    )
    df = pd.read_csv(csv_file_path)
    df.fillna("", inplace=True)

    for _, row in df.iterrows():
        prompt = OnePagerPromptsBase(
            cell=row["cell"],
            beginning_of_prompt=row["beginning_of_prompt"],
            core_prompt=row["core_prompt"],
            override_core_prompt=row["override_core_prompt"],
        )
        db_session().add(prompt)

    db_session().commit()
    db_session().close()

def insert_one_pager_questions_into_r_and_d():
    """
    Reads question data from a CSV and inserts it into the QuestionRepositoryResearch table.
    Each row represents a question with optional yes/no point values and 
    a default list of applicable question years.
    """
    csv_file_path = (
    )
    df = pd.read_csv(csv_file_path)
    df = df.replace({float("nan"): None})

    for _, row in df.iterrows():
        question_record = QuestionRepositoryResearch(
            question=row["question"],
            yes_points=row["y"],
            no_points=row["n"],
            applicable_question_years=["2019", "2020", "2021", "2022"],
        )
        db_session().add(question_record)
        db_session().commit()

    db_session().close()

# Example: automatically create tables on script execution
create_tables()
