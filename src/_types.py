from dataclasses import dataclass, field
import uuid
from datetime import datetime
from typing import List
from .utils import now

# @dataclass
# class Error:
#     error_id: str = field(default_factory=str)
#     level: str = field(default_factory=str)
#     error_message: str = field(default_factory=str)


# @dataclass
# class ResponseError(Response):
#     error: Error = field(default_factory=Error)


@dataclass
class Model:
    guid: str = field(default_factory=str) # UUID
    model_hash: str = field(default_factory=str)
    owner: str = field(default_factory=str) # Company that developed the model
    qualified_api_name: str = field(default_factory=str) # Model Name as used in API calls IMPORTANT
    model_type: str = field(default_factory=str) # Model Type
    max_tokens: int = field(default_factory=int) # Max tokens
    token_limit: int = field(default_factory=int) # Token Limit
    requests_per_batch: int = field(default_factory=int)
    created_at: str = field(default_factory=str)

    def __post_init__(self):
        self.id = uuid.uuid4(),
        self.model_hash = hash(f'{self.owner}{self.qualified_api_name}{self.model_type}')
        self.created_at = now()


@dataclass
class Strategy:
    id: str = field(default_factory=str)
    name: str = field(default_factory=str)
    description: str = field(default_factory=str)
    created_at: str = field(default_factory=str)
    modified_at: str = field(default_factory=str)

    def __post_init__(self):
        self.id = uuid.uuid4(),
        self.created_at = now()


@dataclass
class Company:
    id: str = field(default_factory=str)
    guid: str = field(default_factory=str)
    name: str = field(default_factory=str)
    ticker: str = field(default_factory=str)
    gp_ticker: str = field(default_factory=str)
    strategies: list[Strategy] = field(default_factory=list[Strategy])
    created_at: str = field(default_factory=str)
    modified_at: str = field(default_factory=str)


@dataclass
class Question:
    id: str = field(default_factory=str)
    question: str = field(default_factory=str)
    strategies: list[Strategy] = field(default_factory=list[Strategy])
    created_at: str = field(default_factory=str)
    modified_at: str = field(default_factory=str)

    def __post_init__(self):
        self.id = uuid.uuid4(),
        self.created_at = now()


@dataclass
class ExcelMap:
    cell: str = field(default_factory=str)
    value: str = field(default_factory=str)


@dataclass 
class Strategy:
    id: str = field(default_factory=str)
    name: str = field(default_factory=str)
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.id = uuid.uuid4()


@dataclass
class ModelRequest:
    id: str = field(default_factory=str)
    company: Company = field(default_factory=Company)
    question: Question = field(default_factory=Question)
    model: Model = field(default_factory=Model)
    strategy: Strategy = field(default_factory=Strategy)
    request: dict = field(default_factory=dict)
    request_hash: str = field(default_factory=str)
    is_success: int = field(default_factory=int)
    created_at: str = field(default_factory=str)
    modified_at: str = field(default_factory=str)

    def __post_init__(self):
        self.id = uuid.uuid4(),
        self.request_hash = hash(f'{self.model.id}{self.strategy}{self.request}')
        self.created_at = now()


@dataclass
class ModelResponse:
    id: str = field(default_factory=str)
    model_id: str = field(default_factory=str)
    status: str = field(default_factory=str)
    raw_api_response: str = field(default_factory=dict)
    response: str = field(default_factory=str)
    request: ModelRequest = field(default_factory=ModelRequest)
    error: str = field(default_factory=str)
    success: str = field(default_factory=str)
    seconds_to_complete_request: str = field(default_factory=str)
    created_at: datetime.now = field(default_factory=datetime.now)

    def __post_init__(self):
        self.id = uuid.uuid4(),
        self.created_at = now()



@dataclass
class AppScope:
    id: str = field(default_factory=str)
    name: str = field(default_factory=str)
    description: str = field(default_factory=str)
    created_at: str = field(default_factory=str)
    modified_at: str = field(default_factory=str)

    def __post_init__(self):
        self.id = uuid.uuid4(),
        self.created_at = datetime.now()


@dataclass
class AppRole:
    id: str = field(default_factory=str)
    name: str = field(default_factory=str)
    scopes: list[AppScope] = field(default_factory=list[AppScope])
    created_at: str = field(default_factory=str)
    modified_at: str = field(default_factory=str)

    def __post_init__(self):
        self.id = uuid.uuid4(),
        self.created_at = now()


@dataclass
class AppUser:
    id: str = field(default_factory=str)
    email: str = field(default_factory=str)
    password: str = field(default_factory=str)
    role: AppRole = field(default_factory=AppRole)
    created_at: str = field(default_factory=str)
    modified_at: str = field(default_factory=str)

    def __post_init__(self):
        self.id = uuid.uuid4(),
        self.created_at = now()


@dataclass
class AppRequest:
    id: str = field(default_factory=str)
    requesting_user: AppUser = field(default_factory=AppUser)
    endpoint: str = field(default_factory=str)
    created_at: str = field(default_factory=str)
    success: str = field(default_factory=str)
    log_id: str = field(default_factory=str)
    
    def __post_init__(self):
        self.id = uuid.uuid4(),
        self.created_at = now()


@dataclass
class AppLog:
    id: str = field(default_factory=str)
    log_message: str = field(default_factory=str)
    created_at: str = field(default_factory=str)
    level: str = field(default_factory=str)


    def __post_init__(self):
        self.id = uuid.uuid4(),
        self.created_at = now()
