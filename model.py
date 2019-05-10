# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
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
# ==============================================================================

"""This file contains code to build and run the tensorflow graph for the sequence-to-sequence model"""

import os
import time
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from attention_decoder import attention_decoder
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.util import nest
from tensorflow.python.ops import rnn_cell_impl as rnc
import horovod.tensorflow as hvd

#_state_size_with_prefix = rnc._state_size_with_prefix  # will need a workaround with higher versions

FLAGS = tf.app.flags.FLAGS
#reuse = 


def get_initial_cell_state(cell, initializer, batch_size, dtype):
	"""Return state tensor(s), initialized with initializer.
  Args:
	cell: RNNCell.
	batch_size: int, float, or unit Tensor representing the batch size.
	initializer: function with two arguments, shape and dtype, that
		determines how the state is initialized.
	dtype: the data type to use for the state.
  Returns:
	If `state_size` is an int or TensorShape, then the return value is a
	`N-D` tensor of shape `[batch_size x state_size]` initialized
	according to the initializer.
	If `state_size` is a nested list or tuple, then the return value is
	a nested list or tuple (of the same structure) of `2-D` tensors with
  the shapes `[batch_size x s]` for each s in `state_size`.
  Snippet from : https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
  """
	state_size = cell.state_size  # starting state. returns the size of individual states
	if nest.is_sequence(state_size):
		state_size_flat = nest.flatten(state_size)
		init_state_flat = [  # this part for multi-layered RNN
			initializer(_state_size_with_prefix(s), batch_size, dtype, i)
			for i, s in enumerate(state_size_flat)]
		init_state = nest.pack_sequence_as(structure=state_size,
										   flat_sequence=init_state_flat)
	else:
		init_state_size = _state_size_with_prefix(state_size)
		init_state = initializer(init_state_size, batch_size, dtype, None)

	return init_state


def make_variable_state_initializer(**kwargs):
	def variable_state_initializer(shape, batch_size, dtype, index):
		"""
	shape : shape of the cell of the RNNCell
	batch_size : int, float, or unit Tensor representing the batch size.
	dtype: the data type to use for the state. Typically float32
	index : not sure

	"""
		args = kwargs.copy()

		if args.get('name'):
			args['name'] = args['name'] + '_' + str(index)  # naming the variable ?
		else:
			args['name'] = 'init_state_' + str(index)

		args['shape'] = shape
		args['dtype'] = dtype

		var = tf.get_variable(**args)  # name, shape, dtype
		var = tf.expand_dims(var, 0)  # 1 * shape
		var = tf.tile(var, tf.stack([batch_size] + [1] * len(shape)))
		var.set_shape(_state_size_with_prefix(shape, prefix=[None]))
		return var

	return variable_state_initializer


class SummarizationModel(object):
	"""A class to represent a sequence-to-sequence model for text summarization. Supports both baseline mode, pointer-generator mode, and coverage"""

	def __init__(self, hps, vocab, elmo=None):
		self._hps = hps
		self._vocab = vocab
		self.use_glove = hps.use_glove.value
		if hps.mode.value == 'train':
			if hps.use_glove.value:
				self._vocab.set_glove_embedding(hps.glove_path.value, hps.emb_dim.value)

		if hps.use_regularizer.value:
			self.beta_l2 = hps.beta_l2.value
		else:
			self.beta_l2 = 0.0

		self._regularizer = tf.contrib.layers.l2_regularizer(scale=self.beta_l2)
		self._reuse = hvd.rank() > 0
		self.elmo = elmo

	def _add_placeholders(self):
		"""Add placeholders to the graph. These are entry points for any input data."""
		hps = self._hps
		self._epoch_num = tf.placeholder(tf.int32, shape=(), name='epoch_num')
		# encoder part
		self._enc_batch = tf.placeholder(tf.int32, [hps.batch_size.value, None], name='enc_batch')
		self._enc_lens = tf.placeholder(tf.int32, [hps.batch_size.value], name='enc_lens')
		self._enc_padding_mask = tf.placeholder(tf.float32, [hps.batch_size.value, None], name='enc_padding_mask')
		
		if FLAGS.use_elmo:
			self._enc_batch_raw = tf.placeholder(tf.string, [hps.batch_size.value,None], name='enc_batch_raw')


		if FLAGS.query_encoder:
			self._query_batch = tf.placeholder(tf.int32, [hps.batch_size.value, None], name='query_batch')
			self._query_lens = tf.placeholder(tf.int32, [hps.batch_size.value], name='query_lens')
			self._query_padding_mask = tf.placeholder(tf.float32, [hps.batch_size.value, None], name='query_padding_mask')
			if FLAGS.use_query_elmo:
				self._query_batch_raw = tf.placeholder(tf.string, [hps.batch_size.value, None], name='query_batch_raw')


		if FLAGS.pointer_gen:
			self._enc_batch_extend_vocab = tf.placeholder(tf.int32, [hps.batch_size.value, None],
														  name='enc_batch_extend_vocab')
			self._max_art_oovs = tf.placeholder(tf.int32, [], name='max_art_oovs')

		if FLAGS.word_gcn:
			self._word_adj_in = [
				{lbl: tf.sparse_placeholder(tf.float32, shape=[None, None], name='word_adj_in_{}'.format(lbl)) for lbl
				 in range(hps.num_word_dependency_labels)} for _ in range(hps.batch_size.value)]
			self._word_adj_out = [
				{lbl: tf.sparse_placeholder(tf.float32, shape=[None, None], name='word_adj_out_{}'.format(lbl)) for lbl
				 in range(hps.num_word_dependency_labels)} for _ in range(hps.batch_size.value)]
			if hps.mode.value == 'train':
				self._word_gcn_dropout = tf.placeholder_with_default(hps.word_gcn_dropout.value, shape=(), name='dropout')
			else:
				self._word_gcn_dropout = tf.placeholder_with_default(1.0, shape=(), name='dropout')
			
			if FLAGS.use_coref_graph:
				self._word_adj_in_coref = [tf.sparse_placeholder(tf.float32, shape=[None, None], name='word_adj_in_coref') for _ in range(hps.batch_size.value)]
				self._word_adj_out_coref = [tf.sparse_placeholder(tf.float32, shape=[None, None], name='word_adj_out_coref') for _ in range(hps.batch_size.value)]
			if FLAGS.use_entity_graph:
				self._word_adj_entity = [tf.sparse_placeholder(tf.float32, shape=[None, None], name='word_adj_entity') for _ in range(hps.batch_size.value)]
			if FLAGS.use_lexical_graph:
				self._word_adj_lexical = [tf.sparse_placeholder(tf.float32, shape=[None, None], name='word_adj_lexical') for _ in range(hps.batch_size.value)]


		self._max_word_seq_len = tf.placeholder(tf.int32, shape=(), name='max_word_seq_len')
	
		if FLAGS.query_gcn:
			self._query_adj_in = [
				{lbl: tf.sparse_placeholder(tf.float32, shape=[None, None], name='query_adj_in_{}'.format(lbl)) for lbl
				 in range(hps.num_word_dependency_labels)} for _ in range(hps.batch_size.value)]
			self._query_adj_out = [
				{lbl: tf.sparse_placeholder(tf.float32, shape=[None, None], name='query_adj_out_{}'.format(lbl)) for lbl
				 in range(hps.num_word_dependency_labels)} for _ in range(hps.batch_size.value)]
			if hps.mode.value == 'train':
				self._query_gcn_dropout = tf.placeholder_with_default(hps.query_gcn_dropout.value, shape=(),
																	  name='query_dropout')
			else:
				self._query_gcn_dropout = tf.placeholder_with_default(1.0, shape=(), name='query_dropout')

		self._max_query_seq_len = tf.placeholder(tf.int32, shape=(), name='max_query_seq_len')
	
		# decoder part
		if hps.mode.value == "decode"  or hps.mode.value == "decode_by_val" :
			self._dec_batch = tf.placeholder(tf.int32, [hps.batch_size.value, hps.max_dec_steps], name='dec_batch')
			self._target_batch = tf.placeholder(tf.int32, [hps.batch_size.value, hps.max_dec_steps], name='target_batch')
			self._dec_padding_mask = tf.placeholder(tf.float32, [hps.batch_size.value, hps.max_dec_steps],
												name='dec_padding_mask')
		else:
			self._dec_batch = tf.placeholder(tf.int32, [hps.batch_size.value, hps.max_dec_steps.value], name = 'dec_batch')
			self._target_batch = tf.placeholder(tf.int32, [hps.batch_size.value, hps.max_dec_steps.value], name = 'target_batch')
			self._dec_padding_mask = tf.placeholder(tf.float32, [hps.batch_size.value, hps.max_dec_steps.value], name='dec_padding_mask')

		if hps.mode.value == "decode"   or hps.mode.value == "decode_by_val" and hps.coverage.value:
			self.prev_coverage = tf.placeholder(tf.float32, [hps.batch_size.value, None], name='prev_coverage')

	def _make_feed_dict(self, batch, just_enc=False):
		"""Make a feed dictionary mapping parts of the batch to the appropriate placeholders.

		Args:
		  batch: Batch object
		  just_enc: Boolean. If True, only feed the parts needed for the encoder.
		"""
		hps = self._hps
		feed_dict = {}
		feed_dict[self._enc_batch] = batch.enc_batch
		feed_dict[self._enc_lens] = batch.enc_lens
		feed_dict[self._enc_padding_mask] = batch.enc_padding_mask
		feed_dict[self._epoch_num] = batch.epoch_num


		if FLAGS.use_elmo:
			feed_dict[self._enc_batch_raw] = batch.enc_batch_raw

		if FLAGS.query_encoder:
			feed_dict[self._query_batch] = batch.query_batch
			feed_dict[self._query_lens] = batch.query_lens
			feed_dict[self._query_padding_mask] = batch.query_padding_mask
			if FLAGS.use_query_elmo:
				feed_dict[self._query_batch_raw] = batch.query_batch_raw


		if FLAGS.pointer_gen:
			feed_dict[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
			feed_dict[self._max_art_oovs] = batch.max_art_oovs

		if FLAGS.word_gcn:
			feed_dict[self._max_word_seq_len] = batch.max_word_len
			word_adj_in = batch.word_adj_in
			word_adj_out = batch.word_adj_out
			for i in range(hps.batch_size.value):
				for lbl in range(hps.num_word_dependency_labels):
					feed_dict[self._word_adj_in[i][lbl]] = tf.SparseTensorValue(
						indices=np.array([word_adj_in[i][lbl].row, word_adj_in[i][lbl].col]).T,
						values=word_adj_in[i][lbl].data,
						dense_shape=word_adj_in[i][lbl].shape)

					feed_dict[self._word_adj_out[i][lbl]] = tf.SparseTensorValue(
						indices=np.array([word_adj_out[i][lbl].row, word_adj_out[i][lbl].col]).T,
						values=word_adj_out[i][lbl].data,
						dense_shape=word_adj_out[i][lbl].shape)
			
			if FLAGS.use_coref_graph:
				word_adj_out_coref = batch.word_adj_out_coref
				word_adj_in_coref = batch.word_adj_in_coref
			
				for i in range(hps.batch_size.value):
					feed_dict[self._word_adj_out_coref[i]] = tf.SparseTensorValue(
						indices=np.array([word_adj_out_coref[i].row, word_adj_out_coref[i].col]).T,
						values=word_adj_out_coref[i].data,
						dense_shape=word_adj_out_coref[i].shape)
			
				for i in range(hps.batch_size.value):
					feed_dict[self._word_adj_in_coref[i]] = tf.SparseTensorValue(
						indices=np.array([word_adj_in_coref[i].row, word_adj_in_coref[i].col]).T,
						values=word_adj_in_coref[i].data,
						dense_shape=word_adj_in_coref[i].shape)
			
			if FLAGS.use_entity_graph:
				word_adj_entity = batch.word_adj_entity
				for i in range(hps.batch_size.value):
					feed_dict[self._word_adj_entity[i]] = tf.SparseTensorValue(
						indices=np.array([word_adj_entity[i].row, word_adj_entity[i].col]).T,
						values=word_adj_entity[i].data,
						dense_shape=word_adj_entity[i].shape)

			if FLAGS.use_lexical_graph:
				word_adj_lexical = batch.word_adj_lexical
				for i in range(hps.batch_size.value):
					feed_dict[self._word_adj_lexical[i]] = tf.SparseTensorValue(
						indices=np.array([word_adj_lexical[i].row, word_adj_lexical[i].col]).T,
						values=word_adj_lexical[i].data,
						dense_shape=word_adj_lexical[i].shape)
				

		if FLAGS.query_gcn:
			feed_dict[self._max_query_seq_len] = batch.max_query_len
			query_adj_in = batch.query_adj_in
			query_adj_out = batch.query_adj_out
			for i in range(hps.batch_size.value):
				for lbl in range(hps.num_word_dependency_labels):
					feed_dict[self._query_adj_in[i][lbl]] = tf.SparseTensorValue(
						indices=np.array([query_adj_in[i][lbl].row, query_adj_in[i][lbl].col]).T,
						values=query_adj_in[i][lbl].data,
						dense_shape=query_adj_in[i][lbl].shape)

					feed_dict[self._query_adj_out[i][lbl]] = tf.SparseTensorValue(
						indices=np.array([query_adj_out[i][lbl].row, query_adj_out[i][lbl].col]).T,
						values=query_adj_out[i][lbl].data,
						dense_shape=query_adj_out[i][lbl].shape)

		if not just_enc:
			feed_dict[self._dec_batch] = batch.dec_batch
			feed_dict[self._target_batch] = batch.target_batch
			feed_dict[self._dec_padding_mask] = batch.dec_padding_mask

		return feed_dict

	def _add_elmo_encoder(self, encoder_inputs, seq_len, trainable=True, layer_name='word_emb', name='elmo_encoder'):

		#with tf.variable_scope(name):
			
		encoder_outputs = self.elmo(inputs={ "tokens": encoder_inputs,"sequence_len": seq_len},signature="tokens",as_dict=True)[layer_name]

		return encoder_outputs	
	
	def _add_encoder(self, encoder_inputs, seq_len, num_layers=1, name='encoder', keep_prob=0.7):
		"""Add a single-layer bidirectional LSTM encoder to the graph.

		Args:
		  encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
		  seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

		Returns:
		  encoder_outputs:
			A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
		  fw_state, bw_state:
			Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
		"""
		with tf.variable_scope(name):

			if self._hps.use_lstm.value:
				cell_fw = []
				cell_bw = []
				if self._hps.lstm_type.value == 'layer_norm':
					for _ in range(num_layers):
						cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self._hps.hidden_dim.value)
						if num_layers > 1:
							cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob, input_keep_prob=keep_prob) 
						cell_fw.append(cell)

					for _ in range(num_layers):
						cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self._hps.hidden_dim.value)
						if num_layers > 1:
							cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob, input_keep_prob=keep_prob) 
						cell_bw.append(cell)
				else:
					for _ in range(num_layers):
						cell= tf.contrib.rnn.LSTMCell(self._hps.hidden_dim.value, initializer=tf.contrib.layers.xavier_initializer(seed=1), state_is_tuple=True)
						if num_layers > 1:
							cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob, input_keep_prob=keep_prob) 
						cell_fw.append(cell)

					for _ in range(num_layers):
						cell= tf.contrib.rnn.LSTMCell(self._hps.hidden_dim.value, initializer=tf.contrib.layers.xavier_initializer(seed=1), state_is_tuple=True)
						if num_layers > 1:
							cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob, input_keep_prob=keep_prob) 
						cell_fw.append(cell)

			elif self._hps.use_gru.value:
				cell_fw = [tf.contrib.rnn.GRUCell(self._hps.hidden_dim.value) for _ in range(num_layers)]
				cell_bw = [tf.contrib.rnn.GRUCell(self._hps.hidden_dim.value) for _ in range(num_layers)]

	
			else:
				cell_fw = [tf.contrib.rnn.BasicRNNCell(self._hps.hidden_dim.value) for _ in range(num_layers)]
				cell_bw = [tf.contrib.rnn.BasicRNNCell(self._hps.hidden_dim.value) for _ in range(num_layers)]

			if self._hps.lstm_type.value == 'basic':
				cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim.value, initializer=tf.contrib.layers.xavier_initializer(seed=1),
											  state_is_tuple=True)
				cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim.value, initializer=tf.contrib.layers.xavier_initializer(seed=1),
											  state_is_tuple=True)	
				
				(encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs, dtype=tf.float32, sequence_length=seq_len,  swap_memory=True)
				encoder_outputs = tf.concat(axis=2, values=encoder_outputs)
			

			else:
				
				temp_outputs = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs, sequence_length=seq_len, dtype=tf.float32)
				
				encoder_outputs, encoder_fw_state, encoder_bw_state = temp_outputs	

				fw_st = encoder_fw_state[-1] #last states 
				bw_st = encoder_bw_state[-1]	

		return encoder_outputs, fw_st, bw_st

	   
	   
	def _add_gcn_layer(self, gcn_in, in_dim, gcn_dim, batch_size, max_nodes, max_labels, adj_in, adj_out, word_only=True,
					   num_layers=1,
					   use_gating=False, use_skip=True, use_normalization=True, dropout=1.0, name="GCN",
					   use_label_information=False, loop_dropout=1.0, use_fusion=False):

		if not self._hps.use_label_information:
			max_labels = 1


		# construct single adjacency matrix
		# max_words = tf.cast(tf.shape(adj_in[0][0].dense_shape[0]), dtype=tf.int64)
		max_words = tf.cast(max_nodes, dtype=tf.int64)
		indices = []
		b_data = []
		for b in range(batch_size):
			for l in range(max_labels):
				t_indices = adj_in[b][l].indices
				t_indices += max_words * b
				indices.append(t_indices)
				b_data.append(tf.ones([tf.shape(t_indices)[0]], dtype=tf.int32) * l)
		
		indices = tf.concat(indices, axis=0)
		b_data = tf.concat(b_data, axis=0)
		adj_in = tf.SparseTensor(indices=indices, values=tf.ones([tf.shape(indices)[0]]),
								 dense_shape=[batch_size * max_words, batch_size * max_words])
		labels_in = tf.SparseTensor(indices=indices, values=b_data,
									dense_shape=[batch_size * max_words, batch_size * max_words])

		indices = []
		b_data = []
		for b in range(batch_size):
			for l in range(max_labels):
				t_indices = adj_out[b][l].indices
				t_indices += max_words * b
				indices.append(t_indices)
				b_data.append(tf.ones([tf.shape(t_indices)[0]], dtype=tf.int32) * l)
		indices = tf.concat(indices, axis=0)
		b_data = tf.concat(b_data, axis=0)
		adj_out = tf.SparseTensor(indices=indices, values=tf.ones([tf.shape(indices)[0]]),
								  dense_shape=[batch_size * max_words, batch_size * max_words])
		labels_out = tf.SparseTensor(indices=indices, values=b_data,
									 dense_shape=[batch_size * max_words, batch_size * max_words])

		if self._hps.use_coref_graph.value and word_only:
			indices_in = []
			indices_out = []
			
			for b in range(batch_size):
				t_indices = self._word_adj_in_coref[b].indices
				t_indices += max_words*b
				u_indices = self._word_adj_out_coref[b].indices
				u_indices += max_words*b
				indices_in.append(t_indices)
				indices_out.append(u_indices)
			
			indices_in = tf.concat(indices_in,axis=0)
			indices_out = tf.concat(indices_out, axis=0)
			
			adj_in_coref = tf.SparseTensor(indices=indices_in, values=tf.ones([tf.shape(indices_in)[0]]), dense_shape=[batch_size * max_words, batch_size * max_words])
			adj_out_coref = tf.SparseTensor(indices=indices_out, values=tf.ones([tf.shape(indices_out)[0]]), dense_shape=[batch_size * max_words, batch_size * max_words])

		if self._hps.use_entity_graph.value and word_only:
			indices = []
			
			for b in range(batch_size):
				t_indices = self._word_adj_entity[b].indices
				indices.append(t_indices)
			
			indices = tf.concat(indices, axis=0)
			adj_entity = tf.SparseTensor(indices=indices, values=tf.ones([tf.shape(indices)[0]]), dense_shape=[batch_size * max_words, batch_size * max_words])
	

		if self._hps.use_lexical_graph.value and word_only:
			indices = []
			
			for b in range(batch_size):
				t_indices = self._word_adj_lexical[b].indices
				indices.append(t_indices)
			
			indices = tf.concat(indices, axis=0)
			adj_lexical = tf.SparseTensor(indices=indices, values=tf.ones([tf.shape(indices)[0]]), dense_shape=[batch_size * max_words, batch_size * max_words])

		

		out = [gcn_in]
		if use_fusion:
			in_dims = [in_dim] + [gcn_dim]*num_layers
			fusion_weights = []
			for layer in range(num_layers+1):
				fusion_weights.append(tf.get_variable("weights_fusion_"+str(layer)+"_" + name, [in_dims[layer], gcn_dim],
										   initializer=tf.random_normal_initializer(stddev=0.01, seed=2)))


		for layer in range(num_layers):
			gcn_in = out[-1]
			if len(out) > 1:
				in_dim = gcn_dim

			gcn_in_2d = tf.reshape(gcn_in, [-1, in_dim])

			with tf.variable_scope('%s-%d' % (name, layer)):

				w_in = tf.get_variable("weights", [in_dim, gcn_dim],
									   initializer=tf.random_normal_initializer(stddev=0.01, seed=2))
				w_out = tf.get_variable("weights_inv", [in_dim, gcn_dim],
										initializer=tf.random_normal_initializer(stddev=0.01, seed=3))
				w_loop = tf.get_variable("weights_loop", [in_dim, gcn_dim],
										 initializer=tf.random_normal_initializer(stddev=0.01, seed=4))

				# Layer biases
				b_in = tf.get_variable("bias_labels", [max_labels, gcn_dim],
									   initializer=tf.random_normal_initializer(stddev=0.01, seed=5))
				b_out = tf.get_variable("bias_labels_inv", [max_labels, gcn_dim],
										initializer=tf.random_normal_initializer(stddev=0.01, seed=6))
				b_loop = tf.get_variable("bias_loop", [gcn_dim],
										 initializer=tf.random_normal_initializer(stddev=0.01, seed=7))

				if self._hps.use_coref_graph.value and word_only:
					w_in_coref = tf.get_variable("weights_coref", [in_dim, gcn_dim], initializer=tf.random_normal_initializer(stddev=0.01, seed=20))
					w_out_coref = tf.get_variable("weights_inv_coref", [in_dim, gcn_dim], initializer=tf.random_normal_initializer(stddev=0.01, seed=20))
					b_in_coref = tf.get_variable("bias_coref", [gcn_dim], initializer=tf.random_normal_initializer(stddev=0.01, seed=70))
					b_out_coref = tf.get_variable("bias_inv_coref", [gcn_dim], initializer=tf.random_normal_initializer(stddev=0.01, seed=70))
					gates_in_coref = 1.
					gates_out_coref = 1.

				if self._hps.use_lexical_graph.value and word_only:
					w_lexical = tf.get_variable("weights_lexical", [in_dim, gcn_dim], initializer=tf.random_normal_initializer(stddev=0.01, seed=20))
					b_lexical = tf.get_variable("bias_lexical", [gcn_dim], initializer=tf.random_normal_initializer(stddev=0.01, seed=70))
					gates_lexical = 1.

				if self._hps.use_entity_graph.value and word_only:
					w_entity = tf.get_variable("weights_entity", [in_dim, gcn_dim], initializer=tf.random_normal_initializer(stddev=0.01, seed=20))
					b_entity = tf.get_variable("bias_entity", [gcn_dim], initializer=tf.random_normal_initializer(stddev=0.01, seed=70))
					gates_entity = 1.


				gates_loop = 1.

				if use_gating:
					w_gate_in = tf.get_variable("weights_gate", [in_dim, 1],
												initializer=tf.random_normal_initializer(stddev=0.01, seed=8))
					w_gate_out = tf.get_variable("weights_gate_inv", [in_dim, 1],
												 initializer=tf.random_normal_initializer(stddev=0.01, seed=9))
					w_gate_loop = tf.get_variable("weights_gate_loop", [in_dim, 1],
												  initializer=tf.random_normal_initializer(stddev=0.01, seed=10))

					b_gate_in = tf.get_variable("bias_gate", [max_labels],
												initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01,
																						 seed=11))
					b_gate_out = tf.get_variable("bias_gate_inv", [max_labels],
												 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01,
																						  seed=12))
					b_gate_loop = tf.get_variable("bias_gate_loop", [1], initializer=tf.constant_initializer(1.))


					# compute gates_in
					gates_in = tf.matmul(gcn_in_2d, w_gate_in)
					adj_in *= tf.transpose(gates_in)
					gates_bias = tf.squeeze(tf.nn.embedding_lookup(b_gate_in, labels_in.values, name='gates_lab'))
					values = tf.nn.sigmoid(adj_in.values + gates_bias)
					adj_in = tf.SparseTensor(indices=adj_in.indices, values=values, dense_shape=adj_in.dense_shape)

					# compute gates_out
					gates_out = tf.matmul(gcn_in_2d, w_gate_out)
					adj_out *= tf.transpose(gates_out)
					gates_bias = tf.squeeze(tf.nn.embedding_lookup(b_gate_out, labels_out.values, name='gates_lab'))
					values = tf.nn.sigmoid(adj_out.values + gates_bias)
					adj_out = tf.SparseTensor(indices=adj_out.indices, values=values, dense_shape=adj_out.dense_shape)

					# compute gates_loop
					gates_loop = tf.nn.sigmoid(tf.matmul(gcn_in_2d, w_gate_loop) + b_gate_loop)
					
					if self._hps.use_coref_graph.value and word_only:
						w_gate_in_coref = tf.get_variable("weights_gate_coref", [in_dim, 1],
													initializer=tf.random_normal_initializer(stddev=0.01, seed=8))
						w_gate_out_coref = tf.get_variable("weights_gate_inv_coref", [in_dim, 1],
													 initializer=tf.random_normal_initializer(stddev=0.01, seed=9))

						b_gate_in_coref = tf.get_variable("bias_gate_coref", [1],
													initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01,
																							 seed=11))
						b_gate_out_coref = tf.get_variable("bias_gate_inv_coref", [1],
													 initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01,
																							  seed=12))
	
						# compute gates_in
						gates_in_coref = tf.matmul(gcn_in_2d, w_gate_in_coref)
						adj_in_coref *= tf.transpose(gates_in_coref)
						values = tf.nn.sigmoid(adj_in_coref.values + b_gate_in_coref)
						adj_in_coref = tf.SparseTensor(indices=adj_in_coref.indices, values=values, dense_shape=adj_in_coref.dense_shape)

						gates_out_coref = tf.matmul(gcn_in_2d, w_gate_out_coref)
						adj_out_coref *= tf.transpose(gates_out_coref)
						values = tf.nn.sigmoid(adj_out_coref.values + b_gate_out_coref)
						adj_out_coref = tf.SparseTensor(indices=adj_out_coref.indices, values=values, dense_shape=adj_out_coref.dense_shape)

					if self._hps.use_entity_graph.value and word_only:
						w_gate_entity = tf.get_variable("weights_gate_entity", [in_dim, 1],
													 initializer=tf.random_normal_initializer(stddev=0.01, seed=9))

						b_gate_entity = tf.get_variable("bias_gate_entity", [1],
													initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01,
																							 seed=11))
						gates_entity = tf.matmul(gcn_in_2d, w_gate_entity)
						adj_entity *= tf.transpose(gates_entity)
						values = tf.nn.sigmoid(adj_entity.values + b_gate_entity)
						adj_entity = tf.SparseTensor(indices=adj_entity.indices, values=values, dense_shape=adj_entity.dense_shape)

					
					if self._hps.use_lexical_graph.value and word_only:	
						w_gate_lexical = tf.get_variable("weights_gate_lexical", [in_dim, 1],
													 initializer=tf.random_normal_initializer(stddev=0.01, seed=9))

						b_gate_lexical = tf.get_variable("bias_gate_lexical", [1],
													initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01,
																							 seed=11))
						gates_lexical = tf.matmul(gcn_in_2d, w_gate_lexical)
						adj_lexical *= tf.transpose(gates_lexical)
						values = tf.nn.sigmoid(adj_lexical.values + b_gate_lexical)
						adj_lexical = tf.SparseTensor(indices=adj_lexical.indices, values=values, dense_shape=adj_lexical.dense_shape)

				else:
					# Ideally have to normalize (sum or softmax) for gating too !
					adj_in = adj_in.__mul__(tf.sparse_reduce_sum(adj_in, axis=1))

					adj_out = adj_out.__mul__(tf.sparse_reduce_sum(adj_out, axis=1))

					if self._hps.use_coref_graph.value and word_only:
						adj_in_coref = adj_in_coref.__mul__(tf.sparse_reduce_sum(adj_in_coref, axis=1))
						adj_out_coref = adj_out_coref.__mul__(tf.sparse_reduce_sum(adj_out_coref, axis=1))

					if self._hps.use_entity_graph.value and word_only:
						adj_entity = adj_entity.__mul__(tf.sparse_reduce_sum(adj_entity, axis=1))

					if self._hps.use_lexical_graph.value and word_only:
						adj_lexical = adj_lexical.__mul__(tf.sparse_reduce_sum(adj_lexical, axis=1))


				# Do convolution for adj_in
				h_in = tf.matmul(gcn_in_2d, w_in)
				h_in = tf.sparse_tensor_dense_matmul(adj_in, h_in)
				labels_pad, _ = tf.sparse_fill_empty_rows(labels_in, 0)
				labels_weights, _ = tf.sparse_fill_empty_rows(adj_in, 0.)
				labels_in_embed = tf.nn.embedding_lookup_sparse(b_in, labels_pad, labels_weights, combiner='sum')

				h_in = h_in + labels_in_embed
				# h_in = tf.reshape(h_in, [batch_size, max_nodes, gcn_dim])

				if dropout != 1.0: h_in = tf.nn.dropout(h_in, keep_prob=dropout)  # this is normal dropout

				# Do convolution for adj_out
				h_out = tf.matmul(gcn_in_2d, w_out)
				h_out = tf.sparse_tensor_dense_matmul(adj_out, h_out)
				labels_out_pad, _ = tf.sparse_fill_empty_rows(labels_out, 0)
				labels_out_weights, _ = tf.sparse_fill_empty_rows(adj_out, 0.)
				labels_out_embed = tf.nn.embedding_lookup_sparse(b_out, labels_out_pad, labels_out_weights,
																 combiner='sum')
				h_out = h_out + labels_out_embed
				# h_out = tf.reshape(h_out, [batch_size, max_nodes, gcn_dim])

				if dropout != 1.0: h_out = tf.nn.dropout(h_out, keep_prob=dropout)  # this is normal dropout

				# graph convolution, loops
				h_loop = tf.matmul(gcn_in_2d, w_loop) + b_loop
				h_loop = h_loop * gates_loop
				# h_loop = tf.reshape(h_loop, [batch_size, max_nodes, gcn_dim])

				# loop dropout. consider self as a neighbour loop_probability times only

				if dropout != 1.0: h_loop = tf.nn.dropout(h_loop, keep_prob=dropout, seed=13)  # this is normal dropout

				# final result is the sum of those (with residual connection to inputs)
				
				if self._hps.use_default_graph:
					h_final = h_in + h_out + h_loop
				
				if self._hps.use_coref_graph.value and word_only:
					h_in_coref = tf.matmul(gcn_in_2d, w_in_coref)
					h_in_coref = tf.sparse_tensor_dense_matmul(adj_in_coref, h_in_coref) + b_in_coref
					h_out_coref = tf.matmul(gcn_in_2d, w_out_coref)
					h_out_coref = tf.sparse_tensor_dense_matmul(adj_out_coref, h_out_coref) + b_out_coref
					h_coref = h_in_coref + h_out_coref
					h_final = h_final + h_coref

				if self._hps.use_entity_graph.value and word_only:
					h_entity = tf.matmul(gcn_in_2d, w_entity)
					h_entity = tf.sparse_tensor_dense_matmul(adj_entity, h_entity) + b_entity
					#h_entity = h_entity * gates_loop
					h_final = h_final + h_entity

				if self._hps.use_lexical_graph.value and word_only:
					h_lexical = tf.matmul(gcn_in_2d, w_lexical)
					h_lexical = tf.sparse_tensor_dense_matmul(adj_lexical, h_lexical) + b_lexical
					#h_lexical = h_lexical * gates_loop
					h_final = h_final + h_lexical

				h = tf.nn.relu(h_final)
				
				h = tf.reshape(h, [batch_size, max_nodes, gcn_dim])

				if use_skip:
					b_skip = tf.get_variable('b_skip', [1], initializer=tf.constant_initializer(0.0))
					if in_dim != gcn_dim:
						w_adjust = tf.get_variable('w_adjust', [in_dim, gcn_dim],
												   initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01,
																							seed=14),
												   regularizer=self._regularizer)
						gcn_in = tf.tensordot(gcn_in, w_adjust, axes=[[2], [0]])
						#gcn_in = tf.matmul(gcn_in, w_adjust)

					h = (1 - b_skip) * h + b_skip * (gcn_in)

				
				out.append(h)

		if use_fusion:
			h = tf.tensordot(out[0], fusion_weights[0],axes=[[2],[0]])
			for layer in range(1, num_layers + 1):
				h += tf.tensordot(out[layer], fusion_weights[layer], axes=[[2], [0]])

		return h  # batch_size * max_enc_len * gcn_dim

	
	def _reduce_states(self, fw_st, bw_st):
		"""Add to the graph a linear layer to reduce the encoder's final FW and BW state into a single initial state for the decoder. This is needed because the encoder is bidirectional but the decoder is not.

	Args:
	  fw_st: LSTMStateTuple with hidden_dim units.
	  bw_st: LSTMStateTuple with hidden_dim units.

	Returns:
	  state: LSTMStateTuple with hidden_dim units.
	"""
		hidden_dim = self._hps.hidden_dim.value
		with tf.variable_scope('reduce_final_st'):
			if self._hps.use_lstm.value:
				# Define weights and biases to reduce the cell and reduce the state
				w_reduce_c = tf.get_variable('w_reduce_c', [hidden_dim * 2, hidden_dim], dtype=tf.float32,
											 initializer=self.rand_unif_init, regularizer=self._regularizer)
				bias_reduce_c = tf.get_variable('bias_reduce_c', [hidden_dim], dtype=tf.float32,
												initializer=self.rand_unif_init, regularizer=self._regularizer)

			bias_reduce_h = tf.get_variable('bias_reduce_h', [hidden_dim], dtype=tf.float32,
											initializer=self.rand_unif_init, regularizer=self._regularizer)

			w_reduce_h = tf.get_variable('w_reduce_h', [hidden_dim * 2, hidden_dim], dtype=tf.float32,
										 initializer=self.rand_unif_init, regularizer=self._regularizer)

			# Apply linear layer
			if self._hps.use_lstm.value:
				old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c])  # Concatenation of fw and bw cell
				new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c)  # Get new cell from old cell

				old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h])  # Concatenation of fw and bw state
				new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)  # Get new state from old state
			else:
				old_h = tf.concat(axis=1, values=[fw_st, bw_st])  # Concatenation of fw and bw state
				new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)  # Get new state from old state

			if self._hps.use_lstm.value:
				return tf.contrib.rnn.LSTMStateTuple(new_c, new_h)  # Return new cell and state
			else:
				return new_h

	def _add_decoder(self, inputs):
		"""Add attention decoder to the graph. In train or eval mode, you call this once to get output on ALL steps. In decode (beam search) mode, you call this once for EACH decoder step.

	Args:
	  inputs: inputs to the decoder (word embeddings). A list of tensors shape (batch_size, emb_dim)

	Returns:
	  outputs: List of tensors; the outputs of the decoder
	  out_state: The final state of the decoder
	  attn_dists: A list of tensors; the attention distributions
	  p_gens: A list of tensors shape (batch_size, 1); the generation probabilities
	  coverage: A tensor, the current coverage vector
	"""
		hps = self._hps
		if hps.use_lstm.value:
			cell = tf.contrib.rnn.LSTMCell(hps.hidden_dim.value, state_is_tuple=True, initializer=self.rand_unif_init)
		elif hps.use_gru.value:
			cell = tf.contrib.rnn.GRUCell(hps.hidden_dim.value)

		else:
			cell = tf.contrib.rnn.BasicRNNCell(hps.hidden_dim.value)

		if hps.no_lstm_encoder.value :
			#self._dec_in_state = get_initial_cell_state(cell, make_variable_state_initializer(), hps.batch_size.value,
			#self._dec_in_state = rnc._zero_state_tensors(cell.size, hps.batch_size.value, float32)
		# TODO Feed the averaged gcn word vectors
			self._dec_in_state = cell.zero_state(hps.batch_size.value, tf.float32)
		prev_coverage = self.prev_coverage if hps.mode.value == "decode"   or hps.mode.value == "decode_by_val" and hps.coverage.value else None  # In decode mode, we run attention_decoder one step at a time and so need to pass in the previous step's coverage vector each time

		if hps.query_encoder.value:
			outputs, out_state, attn_dists, p_gens, coverage = attention_decoder(inputs, self._dec_in_state,
																				 self._enc_states,
																				 self._enc_padding_mask,
																				 cell, use_query=True,
																				 query_states=self._query_states,
																				 query_padding_mask=self._query_padding_mask,
																				 initial_state_attention=(
																							 hps.mode.value == "decode"  or hps.mode.value == "decode_by_val" ),
																				 use_lstm=hps.use_lstm.value,
																				 pointer_gen=hps.pointer_gen.value,
																				 use_coverage=hps.coverage.value,
																				 prev_coverage=prev_coverage)
		else:
			outputs, out_state, attn_dists, p_gens, coverage = attention_decoder(inputs, self._dec_in_state,
																				 self._enc_states,
																				 self._enc_padding_mask,
																				 cell, initial_state_attention=(
							hps.mode.value == "decode"  or hps.mode.value == "decode_by_val" ), use_lstm=hps.use_lstm.value,  pointer_gen=hps.pointer_gen.value,
																				 use_coverage=hps.coverage.value,
																				 prev_coverage=prev_coverage)

		return outputs, out_state, attn_dists, p_gens, coverage

	def _calc_final_dist(self, vocab_dists, attn_dists):
		"""Calculate the final distribution, for the pointer-generator model

	Args:
	  vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
	  attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays

	Returns:
	  final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
	"""
		with tf.variable_scope('final_distribution'):
			# Multiply vocab dists by p_gen and attention dists by (1-p_gen)
			vocab_dists = [p_gen * dist for (p_gen, dist) in zip(self.p_gens, vocab_dists)]
			attn_dists = [(1 - p_gen) * dist for (p_gen, dist) in zip(self.p_gens, attn_dists)]

			# Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
			extended_vsize = self._vocab.size() + self._max_art_oovs  # the maximum (over the batch) size of the extended vocabulary
			extra_zeros = tf.zeros((self._hps.batch_size.value, self._max_art_oovs))
			vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in
									vocab_dists]  # list length max_dec_steps of shape (batch_size, extended_vsize)

			# Project the values in the attention distributions onto the appropriate entries in the final distributions
			# This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
			# This is done for each decoder timestep.
			# This is fiddly; we use tf.scatter_nd to do the projection
			batch_nums = tf.range(0, limit=self._hps.batch_size.value)  # shape (batch_size)
			batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
			attn_len = tf.shape(self._enc_batch_extend_vocab)[1]  # number of states we attend over
			batch_nums = tf.tile(batch_nums, [1, attn_len])  # shape (batch_size, attn_len)
			indices = tf.stack((batch_nums, self._enc_batch_extend_vocab), axis=2)  # shape (batch_size, enc_t, 2)
			shape = [self._hps.batch_size.value, extended_vsize]
			attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in
									attn_dists]  # list length max_dec_steps (batch_size, extended_vsize)

			# Add the vocab distributions and the copy distributions together to get the final distributions
			# final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving the final distribution for that decoder timestep
			# Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
			final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in
						   zip(vocab_dists_extended, attn_dists_projected)]

			return final_dists

	def _add_emb_vis(self, embedding_var):
		"""Do setup so that we can view word embedding visualization in Tensorboard, as described here:
	https://www.tensorflow.org/get_started/embedding_viz
	Make the vocab metadata file, then make the projector config file pointing to it."""
		train_dir = os.path.join(FLAGS.log_root, "train")
		vocab_metadata_path = os.path.join(train_dir, "vocab_metadata.tsv")
		self._vocab.write_metadata(vocab_metadata_path)  # write metadata file
		summary_writer = tf.summary.FileWriter(train_dir)
		config = projector.ProjectorConfig()
		embedding = config.embeddings.add()
		embedding.tensor_name = embedding_var.name
		embedding.metadata_path = vocab_metadata_path
		projector.visualize_embeddings(summary_writer, config)

	def _add_seq2seq(self):
		"""Add the whole sequence-to-sequence model to the graph."""
		hps = self._hps
		vsize = self._vocab.size()  # size of the vocabulary


		with tf.variable_scope('seq2seq'):
			# Some initializers
			self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag.value, hps.rand_unif_init_mag.value,seed=123)
			self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std.value, seed=123)
			self.gcn_weight_init = tf.random_normal_initializer(stddev=0.01, seed=123)
			self.gcn_bias_init = tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=123)
			# Add embedding matrix (shared by the encoder and decoder inputs)
			with tf.variable_scope('embedding'):
				if hps.mode.value == "train":
					if self.use_glove:
						embedding = tf.get_variable('embedding', dtype=tf.float32,
													initializer=tf.cast(self._vocab.glove_emb, tf.float32),
													trainable=hps.emb_trainable.value, regularizer=self._regularizer)
						
					else:
						embedding = tf.get_variable('embedding', [vsize, hps.emb_dim.value], dtype=tf.float32,
													initializer=self.trunc_norm_init, trainable=hps.emb_trainable.value,
													regularizer=self._regularizer)

				else:
					embedding = tf.get_variable('embedding', [vsize, hps.emb_dim.value], dtype=tf.float32)

				if hps.mode.value == "train": self._add_emb_vis(embedding)  # add to tensorboard
				emb_enc_inputs = tf.nn.embedding_lookup(embedding, self._enc_batch)  # tensor with shape (batch_size, max_enc_steps, emb_size)
				if hps.query_encoder.value:
					emb_query_inputs = tf.nn.embedding_lookup(embedding, self._query_batch)  # tensor with shape (batch_size, max_query_steps, emb_size)

				emb_dec_inputs = [tf.nn.embedding_lookup(embedding, x) for x in tf.unstack(self._dec_batch, axis=1)]  # list length max_dec_steps containing shape (batch_size, emb_size)
				if self._hps.use_elmo.value:
					self.elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=self._hps.elmo_trainable.value)

					enc_elmo_states = self._add_elmo_encoder(self._enc_batch_raw, self._enc_lens,trainable=self._hps.elmo_trainable.value, layer_name=self._hps.elmo_embedding_layer.value, name='elmo_encoder')
					if self._hps.use_query_elmo.value:
						enc_query_elmo_states = self._add_elmo_encoder(self._query_batch_raw, self._query_lens, trainable=self._hps.elmo_trainable.value, layer_name=self._hps.elmo_embedding_layer.value, name='elmo_encoder_query')
	
					if self._hps.use_elmo_glove.value:
						emb_enc_inputs = tf.concat([emb_enc_inputs,enc_elmo_states],axis=2) #batch_size, max_enc_steps, emb_size + 1024
						if self._hps.use_query_elmo.value:
							emb_query_inputs = tf.concat([emb_query_inputs,enc_query_elmo_states], axis=2)

					else:
						emb_enc_inputs = enc_elmo_states
						if self._hps.use_query_elmo.value:
							emb_query_inputs = enc_query_elmo_states

			if self._hps.use_gcn_before_lstm.value:
				
################################################## G-LSTM ###############################################

				################ GCN LAYER #######################	
				gcn_in = emb_enc_inputs
				in_dim = tf.shape(emb_enc_inputs)[2]
				gcn_dim = hps.word_gcn_dim.value

				gcn_outputs = self._add_gcn_layer(gcn_in=gcn_in, in_dim=in_dim, gcn_dim=hps.word_gcn_dim.value,
												  batch_size=hps.batch_size.value, max_nodes=self._max_word_seq_len,
												  max_labels=hps.num_word_dependency_labels, adj_in=self._word_adj_in,
												  adj_out=self._word_adj_out, 
												  num_layers=hps.word_gcn_layers.value,
												  use_gating=hps.word_gcn_gating.value, use_skip=hps.word_gcn_skip.value,
												  dropout=self._word_gcn_dropout,
												  name="gcn_word",
												  loop_dropout= hps.word_loop_dropout.value,
												  use_fusion=hps.word_gcn_fusion.value)

				######## INTERM CONCAT ##########
				if hps.concat_with_word_embedding.value:  #interm concat
					b_interm_word = tf.get_variable('b_interm_word', [1], initializer=tf.constant_initializer(0.0))
					
					if hps.word_gcn_dim.value!= hps.emb_dim.value:
						w_interm_word = tf.get_variable('w_interm_word', [hps.emb_dim.value, hps.word_gcn_dim.value], initializer=tf.contrib.layers.xavier_initializer(), regularizer=self._regularizer)
						emb_enc_inputs = tf.tensordot(emb_enc_inputs, w_interm_word, axes=[[2], [0]])

					gcn_outputs = ( 1.0 - b_interm_word) * gcn_outputs + b_interm_word * emb_enc_inputs
				
				
				########### LSTM LAYER ############	
				enc_outputs, fw_st, bw_st = self._add_encoder(gcn_outputs, self._enc_lens, num_layers=hps.encoder_lstm_layers.value, keep_prob=hps.lstm_dropout.value)
				self._dec_in_state = self._reduce_states(fw_st, bw_st)


				########### UPPER CONCAT ##########

				if self._hps.concat_gcn_lstm.value:
					b_upper_word = tf.get_variable('b_upper_word', [1], initializer=tf.constant_initializer(0.0))
					
					if hps.word_gcn_dim.value!= hps.hidden_dim.value * 2:
						w_interm_word = tf.get_variable('w_upper_word', [hps.word_gcn_dim.value, hps.hidden_dim.value*2], initializer=tf.contrib.layers.xavier_initializer(), regularizer=self._regularizer)
						gcn_outputs = tf.tensordot(gcn_outputs, w_interm_word, axes=[[2], [0]])

					self._enc_states = ( 1 - b_upper_word) * enc_outputs + b_upper_word * gcn_outputs
					
				else:
					self._enc_states = enc_outputs

				################QUERY ENCODER###########################
				if self._hps.query_encoder.value:

					q_gcn_in = emb_query_inputs
					
					q_in_dim = tf.shape(emb_query_inputs)[2]
					q_gcn_outputs = emb_query_inputs #if not used
					
					######### GCN LAYER #############

					if self._hps.query_gcn.value:  
						q_gcn_outputs = self._add_gcn_layer(gcn_in=q_gcn_in, in_dim=q_in_dim, gcn_dim=hps.query_gcn_dim.value,
															batch_size=hps.batch_size.value, max_nodes=self._max_query_seq_len,
															max_labels=hps.num_word_dependency_labels,
															adj_in=self._query_adj_in,
															adj_out=self._query_adj_out,
															num_layers=hps.query_gcn_layers.value,
															use_gating=hps.query_gcn_gating.value, use_skip=hps.query_gcn_skip.value,
															dropout=self._query_gcn_dropout,
															name="gcn_query",
															loop_dropout=hps.query_loop_dropout.value,
															use_fusion=hps.query_gcn_fusion.value, word_only=False)

						########## INTERM CONCAT ##############
						if hps.concat_with_word_embedding.value:
							b_interm_query = tf.get_variable('b_interm_query', [1], initializer=tf.constant_initializer(0.0))
					
							if hps.emb_dim.value!= hps.query_gcn_dim.value:
								w_interm_query = tf.get_variable('w_interm_query', [hps.emb_dim.value, hps.query_gcn_dim.value], initializer=tf.contrib.layers.xavier_initializer(), regularizer=self._regularizer)
								emb_query_inputs = tf.tensordot(emb_query_inputs, w_interm_query, axes=[[2], [0]])

							q_gcn_outputs = ( 1 - b_interm_query) * q_gcn_outputs + b_upper_query * emb_query_inputs

					
					######## LSTM LAYER #############		
					
					query_outputs, fw_st_q, bw_st_q = self._add_encoder(q_gcn_outputs, self._query_lens, num_layers=hps.query_encoder_lstm_layers.value, name='query_encoder',keep_prob=hps.lstm_dropout.value)
					self._query_states = query_outputs
					
					######### UPPER CONCAT ############
					if self._hps.concat_gcn_lstm.value and self._hps.query_gcn.value:
						b_upper_query = tf.get_variable('b_upper_query', [1], initializer=tf.constant_initializer(0.0))
					
						if hps.query_gcn_dim.value!= hps.hidden_dim.value * 2:
							w_interm_query = tf.get_variable('w_upper_query', [hps.query_gcn_dim.value, hps.hidden_dim.value*2], initializer=tf.contrib.layers.xavier_initializer(), regularizer=self._regularizer)
							q_gcn_outputs = tf.tensordot(q_gcn_outputs, w_interm_query, axes=[[2], [0]])

						self._query_states = ( 1 - b_upper_query) * query_outputs + b_upper_query * q_gcn_outputs
						
					else:
						self._query_states = query_outputs

######################################### TLSTM and PA-LSTM ###################################################

			else:
								
				if not self._hps.no_lstm_encoder.value:

				#####LSTM LAYER ########
					enc_outputs, fw_st, bw_st = self._add_encoder(emb_enc_inputs, self._enc_lens, num_layers=hps.encoder_lstm_layers.value, keep_prob=hps.lstm_dropout.value)

					if self._hps.stacked_lstm.value:  # lstm over lstm
						enc_outputs, fw_st, bw_st = self._add_encoder(enc_outputs, self._enc_lens, name='stacked_encoder',keep_prob=hps.lstm_dropout.value)
					
					# Our encoder is bidirectional and our decoder is unidirectional so we need to reduce the final encoder hidden state to the right size to be the initial decoder hidden state
					self._dec_in_state = self._reduce_states(fw_st, bw_st)	
					self._enc_states = enc_outputs	
			



				if self._hps.word_gcn.value:

					if self._hps.use_gcn_lstm_parallel.value or self._hps.no_lstm_encoder.value:
						gcn_in = emb_enc_inputs
						in_dim = tf.shape(emb_enc_inputs)[2]
					
					else:
						######### INTERM CONCAT ########
						in_dim = self._hps.hidden_dim.value * 2

						if self._hps.concat_with_word_embedding.value:  # interm concat
							b_interm_word = tf.get_variable('b_interm_word', [1], initializer=tf.constant_initializer(0.0))

							if hps.emb_dim.value != hps.hidden_dim.value * 2:
								w_interm_word = tf.get_variable('w_interm_word', [hps.emb_dim.value, hps.hidden_dim.value * 2], initializer=tf.contrib.layers.xavier_initializer(),  regularizer=self._regularizer)
								emb_enc_inputs = tf.tensordot(emb_enc_inputs, w_interm_word, axes=[[2], [0]])
								
							gcn_in = b_interm_word * emb_enc_inputs + (1.0 - b_interm_word) * self._enc_states
							
						else:
							
							gcn_in = self._enc_states
							
					############# GCN LAYER ############	
					gcn_outputs = self._add_gcn_layer(gcn_in=gcn_in, in_dim=in_dim, gcn_dim=hps.word_gcn_dim.value,
												  batch_size=hps.batch_size.value, max_nodes=self._max_word_seq_len,
												  max_labels=hps.num_word_dependency_labels, adj_in=self._word_adj_in,
												  adj_out=self._word_adj_out,
												  num_layers=hps.word_gcn_layers.value,
												  use_gating=hps.word_gcn_gating.value, use_skip=hps.word_gcn_skip.value,
												  dropout=self._word_gcn_dropout,
												  name="gcn_word",
												  loop_dropout=hps.word_loop_dropout.value,
												  use_fusion=hps.word_gcn_fusion.value)

					
					############## UPPPER CONCAT ###############
						
					if self._hps.concat_gcn_lstm.value:  # upper concatenate
						b_upper_word = tf.get_variable('b_upper_word', [1], initializer=tf.constant_initializer(0.0))

						if hps.word_gcn_dim.value != hps.hidden_dim.value * 2:
							w_upper_word = tf.get_variable('w_upper_word',[hps.word_gcn_dim.value, hps.hidden_dim.value * 2], initializer=tf.contrib.layers.xavier_initializer(), regularizer=self._regularizer)
							gcn_outputs = tf.tensordot(gcn_outputs, w_upper_word, axes=[[2], [0]])

						self._enc_states = b_upper_word * enc_outputs + (1.0 - b_upper_word) * gcn_outputs

					else:
						self._enc_states = gcn_outputs  # note we return the last output from the gcn directly instead of all the outputs outputs

				
				##################### QUERY ENCODER	###################	
				if self._hps.query_encoder.value:
					
					if not self._hps.no_lstm_query_encoder.value:
						query_outputs, fw_st_q, bw_st_q = self._add_encoder(emb_query_inputs, self._query_lens, num_layers= hps.query_encoder_lstm_layers.value, name='query_encoder',keep_prob=hps.lstm_dropout.value)
						self._query_states = query_outputs
						q_in_dim = self._hps.hidden_dim.value * 2

				
					if self._hps.query_gcn.value:
						if self._hps.use_gcn_lstm_parallel.value or self._hps.no_lstm_query_encoder.value:
							q_gcn_in = emb_query_inputs
							q_in_dim = tf.shape(emb_query_inputs)[2]
						else:
							q_in_dim = self._hps.hidden_dim.value * 2

							######### INTERM CONCAT ############
							if self._hps.concat_with_word_embedding.value:  # interm concat
								b_interm_query = tf.get_variable('b_interm_query', [1], initializer=tf.constant_initializer(0.0))

								if hps.emb_dim.value != hps.hidden_dim.value * 2:
									w_interm_query = tf.get_variable('w_interm_query', [hps.emb_dim.value, hps.hidden_dim.value * 2], initializer=tf.contrib.layers.xavier_initializer(),  regularizer=self._regularizer)
									emb_query_inputs = tf.tensordot(emb_query_inputs, w_interm_query, axes=[[2], [0]])
								
								q_gcn_in = b_interm_query * emb_query_inputs + (1.0 - b_interm_query) * self._query_states
							
							else:
								q_gcn_in = self._query_states 



						q_gcn_outputs = self._add_gcn_layer(gcn_in=q_gcn_in, in_dim=q_in_dim, gcn_dim=hps.query_gcn_dim.value,
															batch_size=hps.batch_size.value, max_nodes=self._max_query_seq_len,
															max_labels=hps.num_word_dependency_labels,
															adj_in=self._query_adj_in,
															adj_out=self._query_adj_out,
															num_layers=hps.query_gcn_layers.value,
															use_gating=hps.query_gcn_gating.value, use_skip=hps.query_gcn_skip.value,
															dropout=self._query_gcn_dropout,
															name="gcn_query",
															loop_dropout=hps.query_loop_dropout.value,
															use_fusion=hps.query_gcn_fusion.value, word_only=False)
						

						############ UPPER CONCAT ############

						if self._hps.concat_gcn_lstm.value: 
							b_upper_query = tf.get_variable('b_upper_query', [1], initializer=tf.constant_initializer(0.0))

							if hps.query_gcn_dim.value != hps.hidden_dim.value * 2:
								w_upper_query = tf.get_variable('w_upper_query',[hps.query_gcn_dim.value, hps.hidden_dim.value * 2], initializer=tf.contrib.layers.xavier_initializer(), regularizer=self._regularizer)
								q_gcn_outputs = tf.tensordot(q_gcn_outputs, w_upper_query, axes=[[2], [0]])

							self._query_states = b_upper_query * query_outputs + (1.0 - b_upper_query) * q_gcn_outputs
						
						else:
							self._query_states = q_gcn_outputs  



################################ DECODER ######################################################

			# Add the decoder.
			with tf.variable_scope('decoder'):
				decoder_outputs, self._dec_out_state, self.attn_dists, self.p_gens, self.coverage = self._add_decoder(emb_dec_inputs)

			# Add the output projection to obtain the vocabulary distribution
			with tf.variable_scope('output_projection'):
				w = tf.get_variable('w', [hps.hidden_dim.value, vsize], dtype=tf.float32, initializer=self.trunc_norm_init,
									regularizer=self._regularizer)
				w_t = tf.transpose(w)
				v = tf.get_variable('v', [vsize], dtype=tf.float32, initializer=self.trunc_norm_init,
									regularizer=self._regularizer)
				vocab_scores = []  # vocab_scores is the vocabulary distribution before applying softmax. Each entry on the list corresponds to one decoder step
				for i, output in enumerate(decoder_outputs):
					if i > 0:
						tf.get_variable_scope().reuse_variables()
					vocab_scores.append(tf.nn.xw_plus_b(output, w, v))  # apply the linear layer

				vocab_dists = [tf.nn.softmax(s) for s in
							   vocab_scores]  # The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.

			# For pointer-generator model, calc final distribution from copy distribution and vocabulary distribution
			if FLAGS.pointer_gen:
				final_dists = self._calc_final_dist(vocab_dists, self.attn_dists)
			else:  # final distribution is just vocabulary distribution
				final_dists = vocab_dists

			if hps.mode.value in ['train', 'eval']:
				# Calculate the loss
				with tf.variable_scope('loss'):
					if FLAGS.pointer_gen:
						# Calculate the loss per step
						# This is fiddly; we use tf.gather_nd to pick out the probabilities of the gold target words
						loss_per_step = []  # will be list length max_dec_steps containing shape (batch_size)
						batch_nums = tf.range(0, limit=hps.batch_size.value)  # shape (batch_size)
						for dec_step, dist in enumerate(final_dists):
							targets = self._target_batch[:,
									  dec_step]  # The indices of the target words. shape (batch_size)
							indices = tf.stack((batch_nums, targets), axis=1)  # shape (batch_size, 2)
							gold_probs = tf.gather_nd(dist,
													  indices)  # shape (batch_size). prob of correct words on this step
							losses = -tf.log(gold_probs + 1e-10)
							loss_per_step.append(losses)

						# Apply dec_padding_mask and get loss
						self._loss = _mask_and_avg(loss_per_step, self._dec_padding_mask, hps.max_dec_steps.value)

					else:  # baseline model
						self._loss = tf.contrib.seq2seq.sequence_loss(tf.stack(vocab_scores, axis=1),
																	  self._target_batch,
																	  self._dec_padding_mask)  # this applies softmax internally
					if hps.use_regularizer.value:
						self._loss += tf.contrib.layers.apply_regularization(self._regularizer, tf.get_collection(
							tf.GraphKeys.REGULARIZATION_LOSSES))

					tf.summary.scalar('loss', self._loss)

					# Calculate coverage loss from the attention distributions
					if hps.coverage.value:
						with tf.variable_scope('coverage_loss'):
							self._coverage_loss = _coverage_loss(self.attn_dists, self._dec_padding_mask)
							tf.summary.scalar('coverage_loss', self._coverage_loss)
						self._total_loss = self._loss + hps.cov_loss_wt.value * self._coverage_loss
						tf.summary.scalar('total_loss', self._total_loss)

		if hps.mode.value == "decode" or hps.mode.value == "decode_by_val":
			# We run decode beam search mode one decoder step at a time
			assert len(
				final_dists) == 1  # final_dists is a singleton list containing shape (batch_size, extended_vsize)
			final_dists = final_dists[0]
			topk_probs, self._topk_ids = tf.nn.top_k(final_dists,
													 hps.batch_size.value * 2)  # take the k largest probs. note batch_size=beam_size in decode mode
			self._topk_log_probs = tf.log(topk_probs)

	
	def _add_train_op(self):
		"""Sets self._train_op, the op to run for training."""
		# Take gradients of the trainable variables w.r.t. the loss function to minimize
		loss_to_minimize = self._total_loss if self._hps.coverage.value else self._loss

		# Apply adagrad optimizer
		if self._hps.optimizer.value == 'adagrad':
			optimizer = tf.train.AdagradOptimizer(self._hps.lr.value, initial_accumulator_value=self._hps.adagrad_init_acc.value)
		if self._hps.optimizer.value == 'adam':
			optimizer = tf.train.AdamOptimizer(learning_rate=self._hps.adam_lr.value)
		if self._hps.optimizer.value == 'momentum':
			def f2():
				return  tf.train.exponential_decay(learning_rate=self._hps.lr.value, global_step=self.global_step, decay_steps=self._hps.learning_rate_change_interval.value*self._hps.save_steps.value, decay_rate=0.5 , staircase=True)
			def f1():
				return self._hps.lr.value
			learning_rate = tf.cond(self.global_step <= self._hps.learning_rate_change_after.value * self._hps.save_steps.value, lambda: f1(), lambda: f2())

			optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)

		optimizer = hvd.DistributedOptimizer(optimizer)
		tvars = tf.trainable_variables()
		grads_and_vars=optimizer.compute_gradients(loss_to_minimize, tvars)
		grads = [grad for grad,var in grads_and_vars]
		tvars = [var for grad,var in grads_and_vars]
		grads, global_norm = tf.clip_by_global_norm(grads, self._hps.max_grad_norm.value)
		
		self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')

	def build_graph(self):
		"""Add the placeholders, model, global step, train_op and summaries to the graph"""
		tf.logging.info('Building graph...')
		t0 = time.time()
    		with tf.device('/gpu:0'):
			self._add_placeholders()
                
			self._add_seq2seq()

		self.global_step = tf.Variable(0, name='global_step', trainable=False)
		if self._hps.mode.value == 'train':
			self._add_train_op()
		self._summaries = tf.summary.merge_all()
		t1 = time.time()
		tf.logging.info('Time to build graph: %i seconds', t1 - t0)

	def run_train_step(self, sess, batch):
		"""Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step and (optionally) coverage loss."""
		feed_dict = self._make_feed_dict(batch)
		to_return = {
			'train_op': self._train_op,
			'summaries': self._summaries,
			'loss': self._loss,
			'global_step': self.global_step,
			'epoch_num': self._epoch_num
		}
		if self._hps.coverage.value:
			to_return['coverage_loss'] = self._coverage_loss
		return sess.run(to_return, feed_dict)

	def run_eval_step(self, sess, batch):
		"""Runs one evaluation iteration. Returns a dictionary containing summaries, loss, global_step and (optionally) coverage loss."""
		feed_dict = self._make_feed_dict(batch)
		to_return = {
			'summaries': self._summaries,
			'loss': self._loss,
			'global_step': self.global_step,
		}
		if self._hps.coverage.value:
			to_return['coverage_loss'] = self._coverage_loss
		return sess.run(to_return, feed_dict)

	def run_encoder(self, sess, batch, use_query=False):
		"""For beam search decoding. Run the encoder on the batch and return the encoder states and decoder initial state.

	Args:
	  sess: Tensorflow session.
	  batch: Batch object that is the same example repeated across the batch (for beam search)

	Returns:
	  enc_states: The encoder states. A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim].
	  dec_in_state: A LSTMStateTuple of shape ([1,hidden_dim],[1,hidden_dim])
	"""
		feed_dict = self._make_feed_dict(batch, just_enc=True)  # feed the batch into the placeholders
		if use_query:

			(enc_states, query_states, dec_in_state, global_step) = sess.run(
				[self._enc_states, self._query_states, self._dec_in_state, self.global_step],
				feed_dict)  # run the encoder
		else:
			(enc_states, dec_in_state, global_step) = sess.run([self._enc_states, self._dec_in_state, self.global_step],
															   feed_dict)

		# dec_in_state is LSTMStateTuple shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
		# Given that the batch is a single example repeated, dec_in_state is identical across the batch so we just take the top row.
		if self._hps.use_lstm.value:
			dec_in_state = tf.contrib.rnn.LSTMStateTuple(dec_in_state.c[0], dec_in_state.h[0])
		else:
			dec_in_state = dec_in_state[0]  # verify ?
		if use_query:
			return enc_states, dec_in_state, query_states
		else:
			return enc_states, dec_in_state

	def decode_onestep(self, sess, batch, latest_tokens, enc_states, dec_init_states, prev_coverage, query_states=None):
		"""For beam search decoding. Run the decoder for one step.
	Args:
	  sess: Tensorflow session.
	  batch: Batch object containing single example repeated across the batch
	  latest_tokens: Tokens to be fed as input into the decoder for this timestep
	  enc_states: The encoder states.
	  dec_init_states: List of beam_size LSTMStateTuples; the decoder states from the previous timestep
	  prev_coverage: List of np arrays. The coverage vectors from the previous timestep. List of None if not using coverage.
	  query_states : The query states
	Returns:
	  ids: top 2k ids. shape [beam_size, 2*beam_size]
	  probs: top 2k log probabilities. shape [beam_size, 2*beam_size]
	  new_states: new states of the decoder. a list length beam_size containing
		LSTMStateTuples each of shape ([hidden_dim,],[hidden_dim,])
	  attn_dists: List length beam_size containing lists length attn_length.
	  p_gens: Generation probabilities for this step. A list length beam_size. List of None if in baseline mode.
	  new_coverage: Coverage vectors for this step. A list of arrays. List of None if coverage is not turned on.
	"""
		
		beam_size = len(dec_init_states)
		
		if FLAGS.use_lstm:

		# Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
			cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
			hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]
			new_c = np.concatenate(cells, axis=0)  # shape [batch_size,hidden_dim]
			new_h = np.concatenate(hiddens, axis=0)  # shape [batch_size,hidden_dim]
			new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
			
		else:
			cells = [np.expand_dims(state, axis=0)  for state in dec_init_states]
			new_dec_in_state = np.concatenate(cells, axis=0)
			
		feed = {
			self._enc_states: enc_states,
			self._enc_padding_mask: batch.enc_padding_mask,
			self._dec_in_state: new_dec_in_state,
			self._dec_batch: np.transpose(np.array([latest_tokens])),
		}

		to_return = {
			"ids": self._topk_ids,
			"probs": self._topk_log_probs,
			"states": self._dec_out_state,
			"attn_dists": self.attn_dists
		}

		if FLAGS.pointer_gen:
			feed[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
			feed[self._max_art_oovs] = batch.max_art_oovs
			to_return['p_gens'] = self.p_gens

		if FLAGS.word_gcn:
			feed[self._max_word_seq_len] = batch.max_word_len

		if FLAGS.query_encoder:
			feed[self._query_states] = query_states
			feed[self._query_padding_mask] = batch.query_padding_mask

		if FLAGS.query_gcn:
			feed[self._max_query_seq_len] = batch.max_query_len

		if self._hps.coverage.value:
			feed[self.prev_coverage] = np.stack(prev_coverage, axis=0)
			to_return['coverage'] = self.coverage

		results = sess.run(to_return, feed_dict=feed)  # run the decoder step
		

		if FLAGS.use_lstm: 
		# Convert results['states'] (a single LSTMStateTuple) into a list of LSTMStateTuple -- one for each hypothesis
			new_states = [tf.contrib.rnn.LSTMStateTuple(results['states'].c[i, :], results['states'].h[i, :]) for i in
					  xrange(beam_size)]
		else:
			new_states = [results['states'][i,:] for i in  xrange(beam_size)]

		# Convert singleton list containing a tensor to a list of k arrays
		assert len(results['attn_dists']) == 1
		attn_dists = results['attn_dists'][0].tolist()

		if FLAGS.pointer_gen:
			# Convert singleton list containing a tensor to a list of k arrays
			assert len(results['p_gens']) == 1
			p_gens = results['p_gens'][0].tolist()
		else:
			p_gens = [None for _ in xrange(beam_size)]

		# Convert the coverage tensor to a list length k containing the coverage vector for each hypothesis
		if FLAGS.coverage:
			new_coverage = results['coverage'].tolist()
			assert len(new_coverage) == beam_size
		else:
			new_coverage = [None for _ in xrange(beam_size)]

		return results['ids'], results['probs'], new_states, attn_dists, p_gens, new_coverage



def reduce_sum_lossop(x, max_dec_steps):
	return tf.squeeze(tf.matmul(x, tf.ones([max_dec_steps, 1])))

def reduce_mean_op(x):
	divide_by = tf.shape(x)
	w = tf.squeeze(tf.matmul(tf.reshape(x,[1,divide_by[0]]), tf.reshape(tf.ones(divide_by),[divide_by[0],1])))		
	return w/tf.to_float(divide_by[0])

def _mask_and_avg(values, padding_mask, max_dec_steps):
	"""Applies mask to values then returns overall average (a scalar)
  Args:
	values: a list length max_dec_steps containing arrays shape (batch_size).
	padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.
  Returns:
	a scalar
  """

	#deterministic reduce_sum
	#batch_size = tf.shape(padding_mask)[0]
	batch_s = tf.reshape(tf.shape(padding_mask)[0],[])
	dec_lens = reduce_sum_lossop(padding_mask, max_dec_steps)  # shape batch_size. float32
	
	values_per_step = [v * padding_mask[:, dec_step] for dec_step, v in enumerate(values)]
	values_per_ex = sum(values_per_step) / dec_lens  # shape (batch_size); normalized value for each batch member
	#return tf.reduce_mean(values_per_ex)  # overall average
	return reduce_mean_op(values_per_ex)

def _coverage_loss(attn_dists, padding_mask):
	"""Calculates the coverage loss from the attention distributions.
  Args:
	attn_dists: The attention distributions for each decoder timestep. A list length max_dec_steps containing shape (batch_size, attn_length)
	padding_mask: shape (batch_size, max_dec_steps).
  Returns:
	coverage_loss: scalar
  """
	coverage = tf.zeros_like(attn_dists[0])  # shape (batch_size, attn_length). Initial coverage is zero.
	covlosses = []  # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
	for a in attn_dists:
		covloss = tf.reduce_sum(tf.minimum(a, coverage), [1])  # calculate the coverage loss for this step
		covlosses.append(covloss)
		coverage += a  # update the coverage vector
	coverage_loss = _mask_and_avg(covlosses, padding_mask)
	return coverage_loss
