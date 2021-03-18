#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/17 10:01 PM
# @Author  : Gear

import torch
import torch.nn as nn


class MLoss(nn.Module):
	def __init__(self, alpah):
		super(MLoss, self).__init__()
		self.alpah = alpah
		self.fsa = torch.nn.CrossEntropyLoss()
		self.merry = torch.nn.CrossEntropyLoss()
	
	def forward(self, scores,retags,label):
		merryup_loss = self.merry(scores, label)
		fsa_loss = self.fsa(retags,label)
		loss = merryup_loss + self.alpah * fsa_loss #loss融合
		return loss