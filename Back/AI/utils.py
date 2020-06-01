from datetime import datetime
import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import config as cfg
import torch
import pickle
import datetime


def save_model(model,name,root_path):
	path = root_path+name+".pth"
	torch.save(model.state_dict(),path)

def save_config(config,name,root_path):
	save_path = root_path+name+".json"
	with open(save_path, 'w', encoding='utf-8') as f:
		json.dump(config, f)

def save_loss(loss,name,root_path):
	save_path = root_path+name+".pkl"
	with open(save_path, 'wb') as f:
		pickle.dump(loss, f)

def date2str():
	return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
