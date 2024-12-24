from flask import Blueprint, request, jsonify
from sqlalchemy import text
import sys
from db.db_session import db_session
from db.data_models import (
    Companies,
    FinancialModelMeasurePrompts,
    AsisstantBlobs,
    EngineBuildStatus,
)
from services.utils import now

financial_model_blueprint = Blueprint('financial_model_blueprint', __name__)

@financial_model_blueprint.route("/fetch_financial_model_data", methods=["GET"])
def fetch_financial_model_data():
    """
    Retrieves prompts, completed build status, and any assistant blob data for 
    a company's financial model.
    """
    ticker = request.args.get("ticker")
    with db_session() as session:
        company_guid = Companies.get_company_guid(session, ticker)

    # Check if a completed build for 'financial_model' exists
    with db_session() as session:
        record = (
            session.query(EngineBuildStatus)
            .filter_by(
                company_guid=str(company_guid),
                engine="financial_model",
                status="completed",
            )
            .first()
        )
    model_available = 1 if record else 0

    # Insert default prompts if none exist
    if ticker:
        with db_session() as session:
            insert_query = text(f"""
                WITH check_value AS (
                    SELECT EXISTS (
                      SELECT 1 
                      FROM llm.financial_model_measure_prompts
                      WHERE company_guid = '{company_guid}'
                    ) AS company_exists
                )
                INSERT INTO llm.financial_model_measure_prompts (
                    measure_guid, company_guid,
                    prompt_type, base_column, is_yoy,
                    beginning_of_prompt, core_measure_prompt, override_core_measure_prompt,
                    end_of_prompt, applicable_financial_statement, example_prompt_unit,
                    projection_or_historical, time_period_type, number_format_type,
                    created_at, modified_at
                )
                SELECT 
                    guid, '{company_guid}',
                    prompt_type, base_column, is_yoy,
                    beginning_of_prompt, core_measure_prompt, override_core_measure_prompt,
                    end_of_prompt, applicable_financial_statement, example_prompt_unit,
                    projection_or_historical, time_period_type, number_format_type,
                    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                FROM llm.financial_model_measure_prompts_base
                WHERE NOT (SELECT company_exists FROM check_value)
            """)
            session.execute(insert_query)
            session.commit()

    # Fetch the actual prompt data
    with db_session() as session:
        prompts = (
            session.query(FinancialModelMeasurePrompts)
            .filter(
                FinancialModelMeasurePrompts.company_guid == company_guid,
                FinancialModelMeasurePrompts.projection_or_historical != "projection",
            )
            .order_by(FinancialModelMeasurePrompts.prompt_type.desc())
        )

    prompt_data = [
        {
            "ticker": ticker,
            "measure_guid": p.measure_guid,
            "prompt_type": p.prompt_type,
            "base_column": p.base_column,
            "default_prompt": p.core_measure_prompt,
            "override_prompt": p.override_core_measure_prompt,
            "last_updated": p.modified_at,
        }
        for p in prompts
    ]

    # Fetch associated file data
    with db_session() as session:
        file_records = session.query(AsisstantBlobs).filter(
            AsisstantBlobs.company_guid == company_guid
        )
    file_data = []
    for f in file_records:
        try:
            # e.g. uploads/<company_guid>/<subtype>/filename
            subtype = f.blob_path.split(f"{company_guid}/")[1].split("/")[0]
        except:
            subtype = "unknown"
        file_data.append({
            "type": subtype,
            "file_name": f.file_name,
            "uploaded_at": f.created_at,
        })

    return {
        "modelAvailable": model_available,
        "fileData": file_data,
        "promptData": prompt_data,
    }
