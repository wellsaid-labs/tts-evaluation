import urllib.request


# Check the URL requested is valid
def url_first_side_effect(url, *args, **kwargs):
    # TODO: Fix failure case if internet does not work
    assert urllib.request.urlopen(url).getcode() == 200
    return None


# NOTE: Consumes the first argument
url_second_side_effect = lambda _, *args, **kwargs: url_first_side_effect(*args, **kwargs)
