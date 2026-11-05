"""Reject ambiguous JSON objects at raw log ingestion boundaries."""


class DuplicateJSONMember(ValueError):
    """A raw object repeats a decoded member name, even with equal values."""


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            # Neither member names nor values are safe diagnostic content.
            raise DuplicateJSONMember('duplicate JSON object member')
        result[key] = value
    return result
