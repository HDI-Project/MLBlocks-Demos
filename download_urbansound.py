import argparse
import os

import boto3


def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def download_folder_contents(bucket, prefix, local_dir):
    for obj in bucket.objects.filter(Prefix=prefix):
        if obj.key != prefix:
            filename = os.path.join(local_dir, os.path.split(obj.key)[-1])
            print('Downloading S3:{}/{} as {}'.format(bucket.name, obj.key, filename))
            bucket.download_file(obj.key, filename)


def download(bucket, remote_dir, local_dir):
    ensure_directory(directory=local_dir)
    download_folder_contents(bucket=bucket, prefix=remote_dir, local_dir=local_dir)


if __name__ == '__main__':
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('hdi-demos')
    download(bucket, 'mlblocks-demo/data/UrbanSound', 'UrbanSound')
