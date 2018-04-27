def _which_decade(date):
    if date.day <= 10:
        return 1
    elif date.day <= 20:
        return 2
    else:
        return 3


def decade_grouper(date):
    return f"{date.year}-{date.month:02}-D{_which_decade(date)}"


def month_grouper(date):
    return f"{date.year}-{date.month:02}"
