import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sqlalchemy import (
    func,
    Column,
    String,
    Integer,
    DateTime,
    Text,
    Float,
    Boolean,
    ARRAY,
    JSON,
    UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import validates
import uuid
import bcrypt
from src.utils import now

Base = declarative_base()


class AppRequestLog(Base):
    __tablename__ = "request_logs"
    __table_args__ = {"schema": "app"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    guid = Column(UUID(as_uuid=True), default=uuid.uuid4)
    ip_address = Column(String(45), nullable=False)
    user = Column(String, nullable=False)
    endpoint = Column(String, nullable=False)
    request_body = Column(Text, nullable=True)
    request_headers = Column(Text, nullable=False)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())

    def __repr__(self):
        return f"<RequestLog {self.ip_address}>"


class AppUser(Base):
    __tablename__ = "users"
    __table_args__ = {"schema": "app"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    guid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True)
    email = Column(String, nullable=False)
    username = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now())
    modified_at = Column(DateTime, default=func.now(), onupdate=func.now())

    @classmethod
    def get_user_by_username(cls, username):
        return cls.query.filter_by(username=username).first()

    @classmethod
    def get_user_by_email(cls, email):
        return cls.query.filter(func.lower(cls.email) == email.lower()).first().guid

    def delete(self, session):
        session.delete(self)
        session.commit()

    def save(self, session):
        session.add(self)
        session.commit()

    def set_password(self, password):
        self.hashed_password = bcrypt.hashpw(
            password.encode("utf-8"), bcrypt.gensalt()
        ).decode("utf-8")

    def check_password(self, password):
        return bcrypt.checkpw(
            password.encode("utf-8"), self.hashed_password.encode("utf-8")
        )


class AppToken(Base):
    __tablename__ = "tokens"
    __table_args__ = {"schema": "app"}

    id = Column(Integer, primary_key=True)
    guid = Column(UUID(as_uuid=True), default=uuid.uuid4)
    jti = Column(String, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=func.now())

    def __repr__(self):
        return f"<Token {self.jti}>"


class Companies(Base):
    __tablename__ = "companies"
    __table_args__ = {"schema": "llm"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    guid = Column(UUID(as_uuid=True), default=uuid.uuid4, nullable=False)
    name = Column(String, nullable=False)
    ticker = Column(String)  # what the market uses
    gp_ticker = Column(String, nullable=False)  # What grandeur peak uses/gives me
    strategies = Column(String, default="")  # comma delimeter
    created_at = Column(DateTime, default=func.now())
    modified_at = Column(DateTime, default=func.now(), onupdate=func.now())
    @classmethod
    def get_company_guid(cls, session, ticker):
        """Class method to get the company GUID by ticker or gp_ticker"""
        company = (
            session.query(cls)
            .filter((cls.ticker == ticker) | (cls.gp_ticker == ticker))
            .first()
        )
        if company:
            return company.guid
        else:
            return None


class Portfolio(Base):
    __tablename__ = "portfolios"
    __table_args__ = (
        UniqueConstraint('name', name='uq_portfolio_name'),  # Unique constraint on name
        {"schema": "llm"}  # Schema name
    )

    guid = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now())
    modified_at = Column(DateTime, default=func.now(), onupdate=func.now())

    @classmethod
    def get_guid(cls, session, portfolio_name):
        """Class method to get the portfolio GUID by its name"""
        portfolio = session.query(cls).filter_by(name=portfolio_name).first()
        if portfolio:
            return portfolio.guid
        else:
            return None


class CompanyPortfolio(Base):
    __tablename__ = "company_portfolios"
    __table_args__ = {"schema": "llm"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    company_guid = Column(UUID(as_uuid=True), default=uuid.uuid4, nullable=False)
    portfolio_guid = Column(UUID(as_uuid=True), default=uuid.uuid4, nullable=False)


class OaiAssistants(Base):
    __tablename__ = "oai_assistants"
    __table_args__ = {"schema": "llm"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    company_guid = Column(UUID(as_uuid=True), nullable=False)
    assistant_id = Column(String, nullable=False)
    vector_store_id = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now())
    modified_at = Column(DateTime, default=func.now(), onupdate=func.now())


class AsisstantBlobs(Base):
    __tablename__ = "assistant_blobs"
    __table_args__ = {"schema": "llm"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    company_guid = Column(UUID(as_uuid=True), nullable=False)
    assistant_id = Column(String, nullable=False)
    file_name = Column(String, nullable=False)
    blob_path = Column(String, nullable=False)
    is_uploaded = Column(Boolean, nullable=False)
    created_at = Column(DateTime, default=func.now())
    modified_at = Column(DateTime, default=func.now(), onupdate=func.now())


class EngineBuildStatus(Base):
    __tablename__ = "engine_build_status"
    __table_args__ = {"schema": "llm"}
    id = Column(Integer, primary_key=True, autoincrement=True)
    guid = Column(UUID(as_uuid=True), default=uuid.uuid4)
    company_guid = Column(UUID, nullable=False)

    engine = Column(String, nullable=False)
    status = Column(String, nullable=False)  # in_progress, completed, failed

    failure_reason = Column(String, nullable=True)
    completed_at = Column(DateTime)  #
    blob_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=func.now())
    modified_at = Column(DateTime, default=func.now(), onupdate=func.now())


class FinancialModelResponses(Base):
    __tablename__ = "financial_model_responses"
    __table_args__ = {"schema": "llm"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    guid = Column(UUID(as_uuid=True), default=uuid.uuid4(), nullable=False)
    company_guid = Column(String, nullable=False)
    response = Column(Text, nullable=False)
    created_at = Column(DateTime, default=func.now())
    modified_at = Column(DateTime, default=func.now(), onupdate=func.now())


class FinancialModelMeasurePromptsBase(Base):
    __tablename__ = "financial_model_measure_prompts_base"
    __table_args__ = {"schema": "llm"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    guid = Column(UUID(as_uuid=True), default=uuid.uuid4, nullable=False)

    prompt_type = Column(String, nullable=False)
    base_column = Column(String, nullable=True)
    is_yoy = Column(String, nullable=True)

    beginning_of_prompt = Column(String, nullable=True)
    core_measure_prompt = Column(String, nullable=True)
    override_core_measure_prompt = Column(String, nullable=True)
    end_of_prompt = Column(String, nullable=True)

    applicable_financial_statement = Column(String, nullable=True)
    example_prompt_unit = Column(String, nullable=True)
    projection_or_historical = Column(String, nullable=True)
    time_period_type = Column(String, nullable=True)
    number_format_type = Column(String, nullable=True)

    created_at = Column(DateTime, default=func.now())
    modified_at = Column(DateTime, default=func.now(), onupdate=func.now())

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.is_yoy == "":
            self.is_yoy = None


class Segments(Base):
    __tablename__ = "segments"
    __table_args__ = {"schema": "llm"}

    company_guid = Column(UUID(as_uuid=True), primary_key=True)
    segments = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now())


class FinancialModelMeasurePrompts(Base):
    __tablename__ = "financial_model_measure_prompts"
    __table_args__ = {"schema": "llm"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    measure_guid = Column(UUID(as_uuid=True))
    company_guid = Column(UUID(as_uuid=True), nullable=False)

    prompt_type = Column(String, nullable=False)
    base_column = Column(String, nullable=True)
    is_yoy = Column(String, nullable=True)

    beginning_of_prompt = Column(String, nullable=True)
    core_measure_prompt = Column(String, nullable=True)
    override_core_measure_prompt = Column(String, nullable=True)
    end_of_prompt = Column(String, nullable=True)

    applicable_financial_statement = Column(String, nullable=True)
    example_prompt_unit = Column(String, nullable=True)
    projection_or_historical = Column(String, nullable=True)
    time_period_type = Column(String, nullable=True)
    number_format_type = Column(String, nullable=True)

    response_guid = Column(UUID(as_uuid=True), nullable=True)

    created_at = Column(DateTime, default=func.now())
    modified_at = Column(DateTime, default=func.now(), onupdate=func.now())

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.is_yoy == "":
            self.is_yoy = None


class OnePagerPromptsBase(Base):
    __tablename__ = "one_pager_prompts_base"
    __table_args__ = {"schema": "llm"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    guid = Column(UUID(as_uuid=True), default=uuid.uuid4, nullable=False)

    cell = Column(String, nullable=True)

    beginning_of_prompt = Column(String, nullable=True)
    core_prompt = Column(String, nullable=True)
    override_core_prompt = Column(String, nullable=True)

    created_at = Column(DateTime, default=func.now())
    modified_at = Column(DateTime, default=func.now(), onupdate=func.now())

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class OnePagerPrompts(Base):
    __tablename__ = "one_pager_prompts"
    __table_args__ = {"schema": "llm"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt_guid = Column(UUID(as_uuid=True))
    company_guid = Column(UUID(as_uuid=True), nullable=False)
    cell = Column(String, nullable=True)

    beginning_of_prompt = Column(String, nullable=True)
    core_prompt = Column(String, nullable=True)
    override_core_prompt = Column(String, nullable=True)

    created_at = Column(DateTime, default=func.now())
    modified_at = Column(DateTime, default=func.now(), onupdate=func.now())


class WizardRequest(Base):
    __tablename__ = "requests"
    __table_args__ = {"schema": "llm"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    guid = Column(UUID(as_uuid=True), default=uuid.uuid4)
    company_guid = Column(UUID(as_uuid=True))

    model_guid = Column(UUID(as_uuid=True))
    model_qualified_api_name = Column(String, nullable=False)

    external_id = Column(String, nullable=True)

    internal_parameters = Column(String, nullable=True)  # Store as JSON string

    request_type = Column(String, nullable=False)
    request = Column(String, nullable=False)  # Store as JSON string
    request_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now())

    created_at = Column(DateTime, default=func.now())

    def __repr__(self):
        return f"<ModelRequest(id='{self.id}', model_id='{self.model_guid}', request_type='{self.request_type}')>"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.request_hash = hash(f"{self.model_guid}{self.request_type}{self.request}")


class WizardResponse(Base):
    __tablename__ = "responses"
    __table_args__ = {"schema": "llm"}

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    guid = Column(UUID(as_uuid=True), default=uuid.uuid4)

    request_guid = Column(UUID(as_uuid=True))

    status = Column(String, nullable=False)

    raw_api_response = Column(String, nullable=True)  # Store as JSON string
    response = Column(String, nullable=True)

    is_success = Column(Integer, default=0)
    error = Column(String, nullable=True)

    internal_parameters = Column(String, nullable=True)  # Store as JSON string

    time_to_complete_request = Column(Float)  # seconds

    created_at = Column(DateTime, default=now())  # response recieved at
    modified_at = Column(DateTime, default=now(), onupdate=now())

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.id = uuid.uuid4()
        self.created_at = now()


class ResponseBase(Base):
    __tablename__ = "response_base"
    __table_args__ = {"schema": "llm"}

    id = Column(Integer, primary_key=True, autoincrement=True)

    # An optional key to make it easier to prioritize and sort requests/responses
    group_key = Column(String)

    company_guid = Column(UUID(as_uuid=True), default=uuid.uuid4, nullable=False)
    question_guid = Column(UUID(as_uuid=True), default=uuid.uuid4, nullable=False)
    response_guid = Column(UUID(as_uuid=True), default=uuid.uuid4)
    parsed_response = Column(String)
    engine_build_guid = Column(UUID(as_uuid=True), default=uuid.uuid4)
    engine = Column(String)
    year_asked_for = Column(String)
    success = Column(Integer)
    created_at = Column(DateTime, default=func.now())
    modified_at = Column(DateTime, default=func.now(), onupdate=func.now())


class QuestionRepositoryResearch(Base):
    __tablename__ = "question_repository_research"
    __table_args__ = {"schema": "llm"}

    id = Column(Integer, primary_key=True, autoincrement=True)

    question_guid = Column(UUID(as_uuid=True), default=uuid.uuid4, nullable=False)
    question = Column(String, nullable=False)
    question_groups = Column(ARRAY(String))
    applicable_question_years = Column(ARRAY(String))
    no_points = Column(Integer, nullable=True)
    yes_points = Column(Integer, nullable=True)
    materiality = Column(Float, nullable=True)
    created_at = Column(DateTime, default=func.now())
    modified_at = Column(DateTime, default=func.now(), onupdate=func.now())


class QuestionGroups(Base):
    __tablename__ = "question_groups"
    __table_args__ = {"schema": "llm"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    guid = Column(UUID(as_uuid=True), default=uuid.uuid4)
    group = Column(String)
    created_at = Column(DateTime, default=func.now())
    modified_at = Column(DateTime, default=func.now(), onupdate=func.now())


class ExposureCategories(Base):
    __tablename__ = "exposure_categories"
    __table_args__ = {"schema": "llm"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    guid = Column(UUID(as_uuid=True), default=uuid.uuid4, nullable=False)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now())
    modified_at = Column(DateTime, default=func.now(), onupdate=func.now())


class ExposureCategoriesCompanies(Base):
    __tablename__ = "exposure_categories_companies"
    __table_args__ = {"schema": "llm"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    company_guid = Column(UUID(as_uuid=True), default=uuid.uuid4, nullable=False)
    exposure_cateogry_guid = Column(UUID(as_uuid=True), default=uuid.uuid4, nullable=False)
    is_active = Column(Integer)
    created_at = Column(DateTime, default=func.now())


class PredictionMarket(Base):
    __tablename__ = "prediction_markets"
    __table_args__ = {"schema": "llm"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    guid = Column(UUID(as_uuid=True), default=uuid.uuid4, nullable=False)
    exposure_cateogry_guid = Column(UUID(as_uuid=True), default=uuid.uuid4, nullable=False)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now())
    modified_at = Column(DateTime, default=func.now(), onupdate=func.now())


class PredictionMarketOrder(Base):
    __tablename__ = "prediction_market_orders"
    __table_args__ = {"schema": "llm"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_market_guid = Column(UUID(as_uuid=True), default=uuid.uuid4, nullable=False)
    user_guid = Column(UUID(as_uuid=True), default=uuid.uuid4, nullable=False)
    order = Column(JSON) # {"type": "buy", "units":10}
    created_at = Column(DateTime, default=func.now())


class EvidenceSynthesisConditions(Base):
    __tablename__ = "evidence_synthesis_conditions"
    __table_args__ = {"schema": "llm"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    guid = Column(UUID(as_uuid=True), default=uuid.uuid4)
    company_guid = Column(UUID(as_uuid=True), nullable=False)
    condition = Column(String, nullable=False)
    response_guid = Column(UUID(as_uuid=True), nullable=True)
    output = Column(String, nullable=True)
    created_at = Column(DateTime, default=func.now())
    modified_at = Column(DateTime, default=func.now(), onupdate=func.now())


class InvestmentThesis(Base):
    __tablename__ = "investment_thesis"
    __table_args__ = {"schema": "llm"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    guid = Column(UUID(as_uuid=True), default=uuid.uuid4)
    company_guid = Column(UUID(as_uuid=True), nullable=False)
    thesis = Column(String, nullable=False)
    question = Column(String, nullable=False)
    condition = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now())
    modified_at = Column(DateTime, default=func.now(), onupdate=func.now())


class PortfolioManagementParameters(Base):
    __tablename__ = "portfolio_management_parameters"
    __table_args__ = {"schema": "llm"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    guid = Column(UUID(as_uuid=True), default=uuid.uuid4)
    company_guid = Column(UUID(as_uuid=True), nullable=False)
    parameter = Column(String, nullable=False)
    value = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now())
    modified_at = Column(DateTime, default=func.now(), onupdate=func.now())


class SynthesisOutput(Base):
    __tablename__ = "synthesis_output"
    __table_args__ = {"schema": "llm"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    guid = Column(UUID(as_uuid=True), default=uuid.uuid4)
    company_guid = Column(UUID(as_uuid=True), nullable=False)
    output = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now())
    modified_at = Column(DateTime, default=func.now(), onupdate=func.now())


def is_ip_addr(ip):
    # IP validation logic here
    import re

    ip_pattern = re.compile(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$")
    return bool(ip_pattern.match(ip))


class IPEntry(Base):
    __tablename__ = "ip_entries"
    __table_args__ = {"schema": "app"}

    id = Column(Integer, primary_key=True)
    ip_address = Column(String(16), unique=True)
    first_seen = Column(DateTime, default=func.now())
    last_seen = Column(DateTime, default=func.now())

    @validates("ip_address")
    def validate_ip(self, key, ip_address):
        assert is_ip_addr(ip_address), f"Invalid IP address: {ip_address}"
        return ip_address

    def __repr__(self):
        return f"<IPEntry(ip_address='{self.ip_address}', first_seen='{self.first_seen}', last_seen='{self.last_seen}')>"
