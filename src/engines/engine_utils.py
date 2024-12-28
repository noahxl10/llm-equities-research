def get_one_pager_beginning_of_prompt(ticker):
    f""


def build_one_pager_request_string(
    ticker, beg_of_prompt, override_core_prompt, core_prompt
):
    return f"""
    """


def post_process_func(y_n, y, n):
    if y_n == 1:
        return y
    if y_n == 0:
        return n


def coalesce(*args):
    return next((arg for arg in args if (arg is not None and arg != "")), None)


def get_y_n(response):
    yes_responses = {"yes.", "yes,", "yes "}  #  'y', 'yep'}
    no_responses = {"no.", "no,", "no "}  # 'n', 'nope'}
    response_lower = response.lower()
    response_lower = response_lower[0:10]
    if response_lower in yes_responses or "yes" in response_lower:
        return 1
    elif response_lower in no_responses or "no" in response_lower:
        return 0
    else:
        return -1
