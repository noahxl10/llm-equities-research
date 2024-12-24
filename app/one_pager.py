from flask import Blueprint, request
from sqlalchemy import text
import sys
from db.db_session import db_session
from db.data_models import (
    Companies,
    OnePagerPrompts,
    AsisstantBlobs,
    EngineBuildStatus,
    OaiAssistants,
)
from services.utils import now
from services.wizard_service import WizardService
from config import config

one_pager_blueprint = Blueprint('one_pager_blueprint', __name__)
wizard_service = WizardService()

@one_pager_blueprint.route("/fetch_one_pager_data", methods=["GET"])
def fetch_one_pager_data():
    """
    Fetches all relevant one-pager prompts, build status, and uploaded files for a given ticker.
    If none exist, inserts default prompts from one_pager_prompts_base.
    """
    ticker = request.args.get("ticker")

    with db_session() as session:
        company_guid = Companies.get_company_guid(session, ticker)

    # Check if a completed build for 'one_pager' exists
    with db_session() as session:
        build_status = (
            session.query(EngineBuildStatus)
            .filter_by(
                company_guid=str(company_guid),
                engine="one_pager",
                status="completed",
            )
            .first()
        )
    model_available = 1 if build_status else 0

    if ticker != "":
        with db_session() as session:
            insert_query = text(f"""
                WITH check_value AS (
                    SELECT EXISTS (
                      SELECT 1 FROM llm.one_pager_prompts
                      WHERE company_guid = '{company_guid}'
                    ) AS company_exists
                )
                INSERT INTO llm.one_pager_prompts (
                    prompt_guid, company_guid,
                    cell, beginning_of_prompt,
                    core_prompt, override_core_prompt,
                    created_at, modified_at
                )
                SELECT
                    guid, '{company_guid}',
                    cell, beginning_of_prompt,
                    core_prompt, override_core_prompt,
                    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                FROM llm.one_pager_prompts_base
                WHERE NOT (SELECT company_exists FROM check_value)
            """)
            session.execute(insert_query)
            session.commit()

        # Ensure an OAI assistant is created for this company if needed
        model = config.models["OpenAI"]
        wizard_service.get_assistant(company_guid, model.qualified_api_name)

    # Fetch prompts
    with db_session() as session:
        prompts = session.query(OnePagerPrompts).filter(
            OnePagerPrompts.company_guid == company_guid
        )

    prompt_data = [
        {
            "ticker": ticker,
            "prompt_guid": p.prompt_guid,
            "cell": p.cell,
            "default_prompt": p.core_prompt,
            "override_prompt": p.override_core_prompt,
            "last_updated": p.modified_at,
        }
        for p in prompts
    ]

    # Fetch associated files
    with db_session() as session:
        file_records = session.query(AsisstantBlobs).filter(
            AsisstantBlobs.company_guid == company_guid
        )
    file_data = []
    for f in file_records:
        try:
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
