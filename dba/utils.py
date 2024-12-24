import os
import uuid
import pandas as pd
import traceback
from sqlalchemy import text, insert

from src.utils import now
from dba.db import db_session, Engine as engine
import dba.data_models as data_models


def get_company_guid(ticker: str) -> str:
    """
    Retrieves the GUID for a company by its ticker (or gp_ticker).
    :param ticker: The ticker (or gp_ticker) string to look up.
    :return: The GUID of the matching company, if found.
    """
    with db_session() as session:
        company = (
            session.query(data_models.Companies)
            .filter(
                (data_models.Companies.ticker == ticker)
                | (data_models.Companies.gp_ticker == ticker)
            )
            .first()
        )
    return company.guid if company else None


def update_status(guid: str, new_status: str) -> None:
    """
    Updates the build status (e.g., 'pending', 'completed', 'failed') for a given GUID.
    :param guid: The primary key or unique identifier for the EngineBuildStatus record.
    :param new_status: The new status value.
    """
    with db_session() as session:
        record = session.query(data_models.EngineBuildStatus).filter_by(guid=guid).first()
        if record:
            record.status = new_status
            print(f"Status updated to {new_status} for GUID {guid}")
        else:
            print(f"No record found for GUID {guid}")


def upsert_companies(companies_csv: str, new_strategy: str) -> None:
    """
    Inserts or updates companies from a CSV file, adding a new strategy to their 'strategies' field if not present.
    :param companies_csv: The path to the CSV file. Must contain 'company_name' and 'gp_ticker' columns.
    :param new_strategy: The new strategy string to add.
    """
    df = pd.read_csv(companies_csv)

    for _, row in df.iterrows():
        company_name = row["company_name"]
        gp_ticker = row["gp_ticker"].upper()

        query_check = text("""
            SELECT strategies FROM llm.companies 
            WHERE gp_ticker = :gp_ticker
        """)

        with engine.connect() as conn:
            result = conn.execute(query_check, {"gp_ticker": gp_ticker}).fetchone()

        if result:
            current_strategies = result[0] if result[0] else ""

            # Decide whether to append the new strategy (simple logic; you could refine if needed)
            update_strategy = True  # Currently always True; refine if you want a condition
            if update_strategy:
                # Merge the current strategies with the new strategy
                if "," in current_strategies:
                    strats = current_strategies.split(",")
                    strats.append(new_strategy)
                    updated_strategies = ",".join(strats)
                else:
                    updated_strategies = ",".join([current_strategies, new_strategy])

                query_update = text(f"""
                    UPDATE llm.companies
                    SET strategies = :updated_strategies,
                        modified_at = CURRENT_TIMESTAMP
                    WHERE gp_ticker = :gp_ticker
                """)

                with engine.begin() as conn:
                    conn.execute(
                        query_update,
                        {"updated_strategies": updated_strategies, "gp_ticker": gp_ticker}
                    )
        else:
            # Create a new company record
            with db_session() as session:
                company = data_models.Companies(
                    name=company_name,
                    gp_ticker=gp_ticker,
                    ticker=gp_ticker,
                    strategies=new_strategy,
                )
                session.add(company)
                session.commit()


def upsert_portfolio(name: str) -> bool:
    """
    Inserts or updates a portfolio record based on its name.
    If a conflict occurs (same name), modifies 'modified_at'.
    :param name: The name of the portfolio.
    :return: True if successful.
    """
    guid = uuid.uuid4()
    with db_session() as session:
        stmt = (
            insert(data_models.Portfolio)
            .values(guid=guid, name=name)
            .on_conflict_do_update(
                index_elements=[data_models.Portfolio.name],
                set_={"modified_at": now()},
            )
        )
        session.execute(stmt)
        session.commit()
    return True


def build_requests(
    group_key: str,
    strategy: str,
    category: str = None,
    portfolio: str = "all_companies",
) -> bool:
    """
    Builds response_base entries for companies in a given portfolio plus the selected questions/quarters.

    :param group_key: A grouping key for these requests.
    :param strategy: The strategy string (e.g., "exposure_categories_v1", "one_pager_v1").
    :param category: An optional category name to filter question set.
    :param portfolio: The name of the portfolio to use; default is 'all_companies'.
    :return: True if insertion is successful.
    """
    with db_session() as session:
        portfolio_guid = data_models.Portfolio.get_guid(session, portfolio)

    if category:
        sub_query = f"""
            SELECT
                guid AS question_guid,
                NULL AS year
            FROM llm.exposure_categories
            WHERE name = '{category}'
        """
    else:
        sub_query = """
            SELECT
                question_guid,
                NULL AS year
            FROM llm.question_repository_research
        """

    main_query = text(f"""
        INSERT INTO llm.response_base (
            group_key,
            company_guid,
            question_guid,
            response_guid,
            parsed_response,
            engine_build_guid,
            engine,
            year_asked_for,
            success,
            created_at,
            modified_at
        )
        WITH companies AS (
            SELECT company_guid 
            FROM llm.company_portfolios
            WHERE portfolio_guid = :portfolio_guid
        ),
        questions AS (
            {sub_query}
        )
        SELECT
            :group_key AS group_key,
            c.company_guid,
            q.question_guid,
            NULL AS response_guid,
            NULL AS parsed_response,
            NULL AS engine_build_guid,
            :strategy AS engine,
            q.year AS year_asked_for,
            NULL AS success,
            CURRENT_TIMESTAMP AS created_at,
            CURRENT_TIMESTAMP AS modified_at
        FROM companies c
        CROSS JOIN questions q
    """)

    print(main_query)
    with db_session() as session:
        session.execute(
            main_query,
            {"portfolio_guid": portfolio_guid, "group_key": group_key, "strategy": strategy}
        )
        session.commit()

    return True


def insert_portfolio(name: str) -> None:
    """
    Creates a new Portfolio record if it doesn't exist; does not handle duplicates.
    :param name: The portfolio name.
    """
    with db_session() as session:
        session.add(data_models.Portfolio(name=name))
        session.commit()


def build_file(company_guid: str, build_type: str, eng, send_email: bool = True, executor=None) -> None:
    """
    Schedules building a file for a given company. Creates an 'EngineBuildStatus'
    record, sets it to 'pending', and executes the 'run' method of the given engine in a separate thread.

    :param company_guid: The company's unique GUID.
    :param build_type: A string describing the build process.
    :param eng: The engine object with a 'run' method.
    :param send_email: Whether to send an email upon completion.
    :param executor: Executor to run the task asynchronously (e.g., ThreadPoolExecutor).
    """
    with db_session() as session:
        current_build_status = (
            session.query(data_models.EngineBuildStatus)
            .filter(
                data_models.EngineBuildStatus.company_guid == company_guid,
                data_models.EngineBuildStatus.status == "completed",
            )
            .order_by(data_models.EngineBuildStatus.created_at.desc())
            .first()
        )

    with db_session() as session:
        build_status = data_models.EngineBuildStatus(
            company_guid=company_guid,
            status="pending",
            engine=str(build_type),
        )
        session.add(build_status)

        try:
            if send_email:
                recipients = os.getenv("FILE_RECIPIENTS").split(",")
            else:
                recipients = None

            # Using an executor to run in the background
            executor.submit(eng.run(recipients=recipients, build_status_guid=build_status.guid))
            session.commit()

        except Exception:
            build_status.status = "failed"
            build_status.failure_reason = traceback.format_exc()
            session.commit()


def build_file_local(company_guid: str, build_type: str, eng, send_email: bool = True, executor=None) -> None:
    """
    Similar to 'build_file' but designed for local usage. Creates an EngineBuildStatus,
    sets it to 'pending', and either executes 'run' directly or via an executor.

    :param company_guid: The company's unique GUID.
    :param build_type: A string describing the build process.
    :param eng: The engine object with a 'run' method.
    :param send_email: Whether to send an email upon completion.
    :param executor: Executor to run the task asynchronously.
    """
    with db_session() as session:
        current_build_status = (
            session.query(data_models.EngineBuildStatus)
            .filter(
                data_models.EngineBuildStatus.company_guid == company_guid,
                data_models.EngineBuildStatus.status == "completed",
            )
            .order_by(data_models.EngineBuildStatus.created_at.desc())
            .first()
        )

    with db_session() as session:
        build_status = data_models.EngineBuildStatus(
            company_guid=company_guid,
            status="pending",
            engine=str(build_type),
        )
        session.add(build_status)
        session.commit()
        build_status_guid = build_status.guid

    try:
        recipients = ["noahxl10@gmail.com"] if send_email else None
        print(f"Recipients: {recipients}")

        if executor:
            executor.submit(eng.run(recipients=recipients, build_status_guid=build_status_guid))
        else:
            print("Running locally...")
            eng.run(recipients=recipients, build_status_guid=build_status_guid)
            print("Run completed locally.")

    except Exception:
        with db_session() as session:
            record = session.query(data_models.EngineBuildStatus).filter_by(guid=build_status_guid).first()
            if record:
                record.status = "failed"
                record.failure_reason = traceback.format_exc()
                session.commit()
            print(f"Build failed: {traceback.format_exc()}")
