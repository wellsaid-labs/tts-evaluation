""" Print the "Frequency of interruption" from https://aws.amazon.com/ec2/spot/instance-advisor/
for `machine_type` / `os` across all regions.

Example:

    python docs/train_model_aws_spot_interruption.py \
        --machine_type g4dn.12xlarge \
        --os Linux

"""
import argparse
import urllib.request
import json


def main(machine_type, os):
    """
    Print the "Frequency of interruption" from https://aws.amazon.com/ec2/spot/instance-advisor/
    for `machine_type` / `os` across all regions.

    Args:
        machine_type (str): The machine type to fetch data for.
        os (str): The OS to fetch data for.
    """

    with urllib.request.urlopen(
            'https://spot-bid-advisor.s3.amazonaws.com/spot-advisor-data.json') as url:
        data = json.loads(url.read().decode())

    filtered = []
    for region, rates in data['spot_advisor'].items():
        if machine_type in rates[os]:
            filtered.append((region, data['ranges'][rates[os][machine_type]['r']]))

    filtered.sort(key=lambda k: k[1]['index'])

    print('Frequency of interruption | Region')
    for region, rate in filtered:
        print(rate['label'], ' | ', region)


if __name__ == '__main__':  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--machine_type', type=str, required=True, help='The machine type to fetch data for.')
    parser.add_argument('--os', default='Linux', type=str, help='The OS to fetch data for.')
    args = parser.parse_args()

    main(args.machine_type, args.os)
