import asyncio
import time
from functools import wraps
from flask import request
import json
from dba.data_models import AppRequestLog
import traceback


def timeout(seconds):
    """
    Decorator to timeout async function after given seconds.
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            future = asyncio.ensure_future(func(*args, **kwargs))
            done, pending = await asyncio.wait({future}, timeout=seconds)
            for task in pending:
                task.cancel()  # Cancel the task if it's still running
            if future in done:
                return future.result()
            raise asyncio.TimeoutError(
                f"{func.__name__} timed out after {seconds} seconds"
            )

        return wrapper

    return decorator


def time_async_request(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        return runtime, result

    return wrapper


def time_request(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        response, request = func(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        return runtime, response, request

    return wrapper


def log_requests_to_db(db_session):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            error_log = AppRequestLog(
                route=request.path,
                ip_address=request.remote_addr,
                method=request.method,
            )
            db_session.add(error_log)
            db_session.commit()  # Log the request before executing the route

            try:
                return f(*args, **kwargs)
            except Exception as e:
                error_log.exception = (
                    traceback.format_exc()
                )  # Log the exception if one occurs
                db_session.commit()  # Update the existing log with the exception
                raise e  # Re-raise the exception after logging

        return decorated_function

    return decorator
