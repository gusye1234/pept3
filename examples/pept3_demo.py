import logging
import sys

import torch

sys.path.append('../')
from pept3 import utils

utils.LOGGING_LEVEL = logging.INFO
from pept3.features import generate_prosit_feature_set
from pept3.finetune import pept3_nfold_finetune
from pept3.models import Model_Factories, Model_Weights_Factories

model = Model_Factories('prosit')
model.load_state_dict(torch.load(Model_Weights_Factories('prosit'), map_location='cpu'))
print('Running', model.comment())
table_file = './demo_data/demo_input.tab'
model = model.eval()

fmodels, id2select = pept3_nfold_finetune(model, table_file)


generate_prosit_feature_set(
    fmodels,
    table_file,
    id2select,
    save2='./demo_data/demo_out.tab',
    tensor_need=True,
    tensor_path='./demo_data/tensor.hdf5',
)
