#!/user/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import sklearn
from .preprocess import load

train_explicit, train_implicit, train_tag = load('train')
# predict symptom
