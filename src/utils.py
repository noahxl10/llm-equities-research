
from datetime import datetime, timedelta, timezone
import datetime as dt
import uuid
import json
import csv


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


def get_binary(answer):
    """Get binary sentiment from answer"""

    try:
        answer = answer.lower()
        if any(sub in answer for sub in SENTIMENT_DICTIONARY["Positive"]):
            return 1
        elif any(sub in answer for sub in SENTIMENT_DICTIONARY["Negative"]):
            return 0
        else:
            return "NULL"
    except:
        return -99


def sort_by_byte_size(list_of_lists):
    """Sort list of lists by byte size

    Args:

        list_of_lists: [[]]

    Returns:

        ordered list_of_lists

    """

    def sort_func(data):
        return all(isinstance(item, str) for item in list_of_lists)

    sorted_data = sorted(list_of_lists, key=sort_func, reverse=False)
    return sorted_data


def parse_config_file(config_file: str) -> str:
    """Parse config file

    Args:

        config_file (str): path to config file

    Returns:

        dict: config dict
    """

    config = {}
    with open(config_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            key, value = line.split("=")
            config[key.strip()] = value.strip()
    return config


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in constants.ALLOWED_EXTENSIONS
    )


def now(return_type: str = "string", time_add: dt.timedelta = None):
    """Get current time

    Args:
        return_type (str): 'string', 'datetime', or 'epoch', default 'string'
        time_add (datetime.timedelta):  add time to current time, default None

    Returns:
        time (str, datetime, or int): current time
    """

    time = datetime.now(timezone.utc)
    if time_add is not None:
        time = time + time_add

    if return_type == "datetime":
        return time
    elif return_type == "epoch":
        return int(time.timestamp())
    else:
        time = time.strftime("%Y-%m-%d %H:%M:%S")
        return time


def get_current_date(format: str = "%Y-%m-%d") -> str:
    """Get current date

    Returns:

        date (str): current date
    """

    date = datetime.now()
    date = date.strftime(format)

    return date


def generate_random_uuid(length: int = None) -> str:
    """Generate random uuid

    Args:

        length (int): length of uuid, default None

    Returns:

        random_uuid (str): random uuid
    """

    random_uuid = uuid.uuid4()
    if length:
        random_uuid = random_uuid.hex[:length]

    return random_uuid


def is_current_time_between(
    start_hour: int, end_hour: int, start_minute: int = 0, end_minute: int = 0
) -> bool:
    """Check if current time is between start and end time

    Args:

        start_hour (int): 24 hour format, 0-23
        end_hour (int): 24 hour format, 0-23
        start_minute (int): 0-59, default 0
        end_minute (int): 0-59, default 0

    Returns:

        bool: True if current time is between start and end time, else False
    """

    current_time = dt.datetime.now().time()
    start_time = dt.time(start_hour, start_minute)
    end_time = dt.time(end_hour, end_minute)

    if start_time <= current_time <= end_time:
        return True
    else:
        return False


def csv_to_json(self, csv_file_path, json_file_path):
    data = []
    try:
        with open(csv_file_path, mode="r", newline="") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                data.append(row)
    except FileNotFoundError:
        return
    except Exception as e:
        return

    try:
        with open(json_file_path, mode="w") as json_file:
            json.dump(data, json_file, indent=4)
    except Exception as e:
        pass
