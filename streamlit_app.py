import streamlit as st

import os
import re
import json
import torch
import helper
import numpy as np
from torch import nn
from torch.nn import functional as F
import pickle

import model

net = model.net
vocab = model.vocab

premise = "Saya lapar sekali."
hypothesis = "Saya tidak tidur."

st.set_page_config(layout="wide")

st.title('Natural Language Inference for Bahasa Indonesia')
st.text('Analyzing the logical relationship between premise and hypothesis.')
premise = st.text_input('Premise (end sentence with .)')
hypothesis = st.text_input('Hypothesis (end sentence with .)')

if st.button('Predict'):
    if(premise != "" and hypothesis != ""):
        st.subheader('Output')
        result = model.predict_inli(net, vocab, premise.split(), hypothesis.split())
        st.write(result)

st.sidebar.header('About')
st.sidebar.write('Project on NLP task known as Natural Language Inference (NLI). NLI involves determining the relationship between pairs of sentences, typically categorized as entailment, contradiction, or neutral.')
st.sidebar.header('Source')
st.sidebar.markdown("+ [Github](https://github.com/chukbert/naturalLanguageInference)")
