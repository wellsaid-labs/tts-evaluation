import urllib.request


# Check the URL requested is valid
def urlretrieve_side_effect(url, *args, **kwargs):
    # TODO: Fix failure case if internet does not work
    assert urllib.request.urlopen(url).getcode() == 200


# Check the URL requested is valid
def _download_file_from_drive_side_effect(_, url, **kwargs):
    # TODO: Fix failure case if internet does not work
    assert urllib.request.urlopen(url).getcode() == 200
