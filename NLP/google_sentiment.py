#!/usr/bin/env python

# Copyright 2016 Google, Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Analyzes text using the Google Cloud Natural Language API."""

import argparse
import json
import sys
import pandas as pd

import googleapiclient.discovery

def analyze_sentiment(text, encoding='UTF32'):
    body = {
        'document': {
            'type': 'PLAIN_TEXT',
            'content': text,
            'language': 'EN',
        },
        'encoding_type': encoding
    }

    service = googleapiclient.discovery.build('language', 'v1')

    request = service.documents().analyzeSentiment(body=body)
    response = request.execute()

    return response

## My code from here to apply sentiment analysis and store it

data = pd.read_csv('data 2.csv')

counter = 0
def sentiment(article): 
    sentiment.counter += 1 
    print(sentiment.counter)
    return analyze_sentiment(article)['documentSentiment']['score']
sentiment.counter = 0 


data['sentiment'] = data['headline'].apply(sentiment)

data.to_csv('new_data_2.csv')

