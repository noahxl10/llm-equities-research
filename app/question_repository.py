from flask import Blueprint, request, jsonify
from sqlalchemy import func, update, and_
from db.db_session import db_session
from db.data_models import (
    QuestionRepositoryResearch,
    QuestionGroups,
    AppUser,
    Companies,
    ResponseBase,
    WizardResponse,
)
from services.utils import now

question_repository_blueprint = Blueprint('question_repository_blueprint', __name__)

@question_repository_blueprint.route("/fetch_question_repository_data", methods=["GET"])
def fetch_question_repository_data():
    """
    Fetches the main question repository, question groups, and optionally user or 
    prediction market data if needed.
    """
    with db_session() as session:
        queries = session.query(QuestionRepositoryResearch).order_by(
            QuestionRepositoryResearch.created_at.desc()
        )
        results = queries.all()

    def attempt_round(x):
        try:
            return round(x, 2)
        except:
            return None

    def remove_nulls(lst):
        return [item for item in lst if item is not None]

    r_d_data = [
        {
            "question_guid": r.question_guid,
            "question": r.question,
            "question_groups": remove_nulls(r.question_groups),
            "applicable_question_years": r.applicable_question_years,
            "no_points": r.no_points,
            "yes_points": r.yes_points,
            "materiality": attempt_round(r.materiality),
            "created_at": r.created_at.strftime("%m-%d-%y"),
        }
        for r in results
    ]

    with db_session() as session:
        group_results = session.query(QuestionGroups).order_by(QuestionGroups.created_at.desc()).all()

    question_groups = [{"name": g.group} for g in group_results]

    return {
        "questionRepository": r_d_data,
        "questionGroups": question_groups,
    }


@question_repository_blueprint.route("/add_row", methods=["POST"])
def add_row():
    """
    An example of how you might add rows for multiple table types:
    repository_research, question_group, etc.
    """
    # 'tableType' determines which logic to execute
    table_type = request.json.get("tableType")
    ticker = request.json.get("ticker")

    # Data to be inserted
    data_rows = request.json.get("data", [])

    if table_type == "repository_research":
        for row in data_rows:
            question_groups = [i["name"] for i in row.get("question_groups", []) if i]
            question_years = sorted(int(i["name"]) for i in row.get("years", []))

            with db_session() as session:
                new_question = QuestionRepositoryResearch(
                    question=row.get("question"),
                    question_groups=question_groups if question_groups else None,
                    yes_points=row.get("yes_value"),
                    no_points=row.get("no_value"),
                    applicable_question_years=question_years,
                    created_at=now(),
                )
                session.add(new_question)
                session.commit()

    elif table_type == "question_group":
        # Example: updating existing records to add question_groups
        for row in data_rows:
            group_name = row.get("group")
            for question in row.get("questions", []):
                with db_session() as session:
                    stmt = (
                        update(QuestionRepositoryResearch)
                        .where(QuestionRepositoryResearch.question == question["name"])
                        .values(
                            question_groups=func.array_append(
                                QuestionRepositoryResearch.question_groups,
                                group_name,
                            )
                        )
                    )
                    session.execute(stmt)
                    session.commit()

            # Insert a new row into QuestionGroups
            with db_session() as session:
                new_group = QuestionGroups(group=group_name, created_at=now())
                session.add(new_group)
                session.commit()

    # Expand with more conditions (e.g. update_question_group, etc.) as needed

    return "Success", 200
