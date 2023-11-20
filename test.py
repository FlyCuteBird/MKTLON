''' Testing '''

import evaluation
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
model_path = "./runs/runX/checkpoint/t2i/model_best.pth.tar"
#model_path = "./runs/runX/checkpoint/i2t/model_best.pth.tar"
data_path = "./data/"
evaluation.evalrank(model_path, data_path=data_path, split="test")
