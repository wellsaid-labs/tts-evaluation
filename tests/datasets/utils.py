import urllib.request


# Check the URL requested is valid
def url_first_side_effect(url, *args, **kwargs):
    # TODO: Fix failure case if internet does not work
    assert urllib.request.urlopen(url).getcode() == 200
    return None


# Check the URL requested is valid
def url_second_side_effect(_, url, *args, **kwargs):
    # TODO: Fix failure case if internet does not work
    assert urllib.request.urlopen(url).getcode() == 200
    return None
