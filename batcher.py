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

"""This file contains code to process data into batches"""

import Queue
from random import shuffle
from threading import Thread
import time
import numpy as np
import tensorflow as tf
import data
import pickle
import threading
import random
import ast






class Example(object):
	"""Class representing a train/val/test example for text summarization."""

	def __init__(self, article, abstract_sentences, vocab, hps, word_edge_list=None, query=None, query_edge_list=None, epoch_num=None, bert_vocab=None):
		"""Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.

		Args:
			article: source text; a string. each token is separated by a single space.
			abstract_sentences: list of strings, one per abstract sentence. In each sentence, each token is separated by a single space.
			vocab: Vocabulary object
			hps: hyperparameters
		"""
		self.hps = hps
		# Get ids of special tokens
		start_decoding = vocab.word2id(data.START_DECODING)
		stop_decoding = vocab.word2id(data.STOP_DECODING)
		self.bert_vocab = bert_vocab
		self.epoch_num = epoch_num #deprecated
		self.enc_pos_offset = None
		self.query_pos_offset = None
		# Process the article
		article_words = article.split()
		if len(article_words) > hps.max_enc_steps.value:
			article_words = article_words[:hps.max_enc_steps.value]
		self.enc_len = len(article_words)  # store the length after truncation but before padding
		self.enc_input = [vocab.word2id(w) for w in
						  article_words]  # list of word ids; OOVs are represented by the id for UNK token
		#tf.logging.info(self.enc_len)
		if self.hps.use_elmo.value:
			self.enc_input_raw = article_words 
		# Process the abstract
		abstract = ' '.join(abstract_sentences)  # string
		abstract_words = abstract.split()  # list of strings
		abs_ids = [vocab.word2id(w) for w in
				   abstract_words]  # list of word ids; OOVs are represented by the id for UNK token

		# Process the query 
		if hps.query_encoder.value:
			query_words = query.split()
			#query_words = word_features.get_tokens(query)
			if len(query_words) > hps.max_query_steps.value:
				#tf.logging.info('Before_query: %d Hps: %d'%(len(query_words),hps.max_query_steps.value))
				query_words = query_words[len(query_words)- hps.max_query_steps.value:]
				#tf.logging.info('Big_query : %d'%(len(query_words)))
				query = " ".join(q for q in query_words)
			self.query_len = len(query_words) # store the length after truncation but before padding
			
			self.query_input = [vocab.word2id(w) for w in query_words] # list of word ids; OOVs are represented by the id for UNK token
			if self.hps.use_query_elmo.value:
				self.query_input_raw = query_words #tensorflow_hub requires raw text
				
		# Get the decoder input sequence and target sequence
		self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, hps.max_dec_steps.value, start_decoding,
																 stop_decoding)
		self.dec_len = len(self.dec_input)

		# If using pointer-generator mode, we need to store some extra info
		if hps.pointer_gen.value:
			# Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id; also store the in-article OOVs words themselves
			self.enc_input_extend_vocab, self.article_oovs = data.article2ids(article_words, vocab)

			# Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
			abs_ids_extend_vocab = data.abstract2ids(abstract_words, vocab, self.article_oovs)

			# Overwrite decoder target sequence so it uses the temp article OOV ids
			_, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, hps.max_dec_steps.value, start_decoding,
														stop_decoding)

		if hps.word_gcn.value:
			self.word_edge_list = word_edge_list

		if hps.query_gcn.value:
			self.query_edge_list = query_edge_list

		if hps.use_bert.value:
			self.enc_input, self.enc_pos_offset = bert_vocab.convert_glove_to_bert_indices(self.enc_input)	
			self.enc_len = len(self.enc_input)
			if hps.use_query_bert.value:
				self.query_input, self.query_pos_offset = bert_vocab.convert_glove_to_bert_indices(self.query_input)	
 				self.query_len = len(self.query_input)
		
		# Store the original strings
		self.original_article = article
		self.original_abstract = abstract
		self.original_abstract_sents = abstract_sentences
		#if hps.query_encoder:
		self.original_query = query

	def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
		"""Given the reference summary as a sequence of tokens, return the input sequence for the decoder, and the target sequence which we will use to calculate loss. The sequence will be truncated if it is longer than max_len. The input sequence must start with the start_id and the target sequence must end with the stop_id (but not if it's been truncated).

		Args:
			sequence: List of ids (integers)
			max_len: integer
			start_id: integer
			stop_id: integer

		Returns:
			inp: sequence length <=max_len starting with start_id
			target: sequence same length as input, ending with stop_id only if there was no truncation
		"""
		inp = [start_id] + sequence[:]
		target = sequence[:]
		if len(inp) > max_len:  # truncate
			inp = inp[:max_len]
			target = target[:max_len]  # no end_token
		else:  # no truncation
			target.append(stop_id)  # end token
		assert len(inp) == len(target)
		return inp, target

	def pad_decoder_inp_targ(self, max_len, pad_id):
		"""Pad decoder input and target sequences with pad_id up to max_len."""
		while len(self.dec_input) < max_len:
			self.dec_input.append(pad_id)
		while len(self.target) < max_len:
			self.target.append(pad_id)

	def pad_encoder_input(self, max_len, pad_id):
		"""Pad the encoder input sequence with pad_id up to max_len."""
		while len(self.enc_input) < max_len:
			self.enc_input.append(pad_id)
		if self.hps.pointer_gen.value:
			while len(self.enc_input_extend_vocab) < max_len:
				self.enc_input_extend_vocab.append(pad_id)
	def pad_query_input(self, max_len, pad_id):
		"""Pad the query input sequence with pad_id up to max_len."""
		while len(self.query_input) < max_len:
			self.query_input.append(pad_id)

	def pad_encoder_input_raw(self, max_len):
		while len(self.enc_input_raw) < max_len:
			self.enc_input_raw.append("")

	def pad_query_input_raw(self,max_len):
		while len(self.query_input_raw) < max_len:
			self.query_input_raw.append("")
		

class Batch(object):
	"""Class representing a minibatch of train/val/test examples for text summarization."""

	def __init__(self, example_list, hps, vocab):
		"""Turns the example_list into a Batch object.

		Args:
			 example_list: List of Example objects
			 hps: hyperparameters
			 vocab: Vocabulary object
		"""
		self.pad_id = vocab.word2id(data.PAD_TOKEN)  # id of the PAD token used to pad sequences
		# self.word_adj_in = None
		# self.word_adj_out = None
		self.init_encoder_seq(example_list, hps)  # initialize the input to the encoder
		if hps.query_encoder:
			self.init_query_seq(example_list, hps) #initialize the input to query_encoder
		self.init_decoder_seq(example_list, hps)  # initialize the input and targets for the decoder
		self.store_orig_strings(example_list)  # store the original strings
		self.epoch_num = example_list[0].epoch_num
		
#		self.query_encoder = hps.query_encoder

	# self.max_word_len = 400

	def init_encoder_seq(self, example_list, hps):
		"""Initializes the following:
				self.enc_batch:
					numpy array of shape (batch_size, <=max_enc_steps) containing integer ids (all OOVs represented by UNK id), padded to length of longest sequence in the batch
				self.enc_lens:
					numpy array of shape (batch_size) containing integers. The (truncated) length of each encoder input sequence (pre-padding).
				self.enc_padding_mask:
					numpy array of shape (batch_size, <=max_enc_steps), containing 1s and 0s. 1s correspond to real tokens in enc_batch and target_batch; 0s correspond to padding.

			If hps.pointer_gen.value, additionally initializes the following:
				self.max_art_oovs:
					maximum number of in-article OOVs in the batch
				self.art_oovs:
					list of list of in-article OOVs (strings), for each example in the batch
				self.enc_batch_extend_vocab:
					Same as self.enc_batch, but in-article OOVs are represented by their temporary article OOV number.
		"""
		# Determine the maximum length of the encoder input sequence in this batch
		encoder_lengths = [ex.enc_len for ex in example_list]
		max_enc_seq_len = max(encoder_lengths)
		self.max_word_len = max_enc_seq_len
		# Pad the encoder input sequences up to the length of the longest sequence
		
		for ex in example_list:

			ex.pad_encoder_input(max_enc_seq_len, self.pad_id)
			if hps.use_elmo.value:
				ex.pad_encoder_input_raw(max_enc_seq_len)

		# Initialize the numpy arrays
		# Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
		self.enc_batch = np.zeros((hps.batch_size.value, max_enc_seq_len), dtype=np.int32)
		self.enc_lens = np.zeros((hps.batch_size.value), dtype=np.int32)
		self.enc_padding_mask = np.zeros((hps.batch_size.value, max_enc_seq_len), dtype=np.float32)
		self.enc_segment_id = [[0] * max_enc_seq_len for i in range(hps.batch_size.value)]
		self.enc_bert_mask_id = [[0] * max_enc_seq_len for i in range(hps.batch_size.value)]
		if hps.use_elmo.value:
			self.enc_batch_raw = [ex.enc_input_raw for ex in example_list]
			#tf.logging.info(self.enc_batch_raw)


		# Fill in the numpy arrays
		for i, ex in enumerate(example_list):
			self.enc_batch[i, :] = ex.enc_input[:]
			self.enc_lens[i] = ex.enc_len
			for j in xrange(ex.enc_len):
				self.enc_padding_mask[i][j] = 1
				self.enc_bert_mask_id[i][j] = 1

		# For pointer-generator mode, need to store some extra info
		if hps.pointer_gen.value:
			# Determine the max number of in-article OOVs in this batch
			self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
			# Store the in-article OOVs themselves
			self.art_oovs = [ex.article_oovs for ex in example_list]
			# Store the version of the enc_batch that uses the article OOV ids
			self.enc_batch_extend_vocab = np.zeros((hps.batch_size.value, max_enc_seq_len), dtype=np.int32)
			for i, ex in enumerate(example_list):
				self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]

		if hps.word_gcn.value:
			edge_list = []

			for ex in example_list:
				edge_list.append(ex.word_edge_list)
			if hps.use_bert.value:
				offset_list = []
				for ex in example_list:
					offset_list.append(ex.enc_pos_offset)

			self.word_adj_in, self.word_adj_out = data.get_adj(edge_list, hps.batch_size.value, max_enc_seq_len, use_label_information=hps.use_label_information.value, flow_alone=hps.flow_alone.value, flow_combined=hps.flow_combined.value, keep_prob=hps.word_gcn_edge_dropout.value, 
				use_bert=hps.use_bert.value, bert_mapping=offset_list, max_length=hps.max_enc_steps.value)

			if hps.use_coref_graph.value:
				self.word_adj_in_coref, self.word_adj_out_coref = data.get_specific_adj(edge_list, hps.batch_size.value, max_enc_seq_len, 'coref', encoder_lengths, keep_prob=hps.word_gcn_edge_dropout.value,use_bert=hps.use_bert.value, bert_mapping=offset_list, max_length=hps.max_enc_steps.value)
			
			if hps.use_entity_graph.value:
				_, self.word_adj_entity = data.get_specific_adj(edge_list, hps.batch_size.value, max_enc_seq_len, 'entity', encoder_lengths, use_both=False, keep_prob=hps.word_gcn_edge_dropout.value,use_bert=hps.use_bert.value, bert_mapping=offset_list, max_length=hps.max_enc_steps.value)

			if hps.use_lexical_graph.value:
				_, self.word_adj_lexical = data.get_specific_adj(edge_list, hps.batch_size.value, max_enc_seq_len, 'lexical', encoder_lengths, use_both=False, keep_prob=hps.word_gcn_edge_dropout.value, use_bert=hps.use_bert.value, bert_mapping=offset_list, max_length=hps.max_enc_steps.value)


	def init_query_seq(self, example_list, hps):
		
	#	"""Initializes the following:
	#			self.query_batch:
	#				numpy array of shape (batch_size, <=max_query_steps) containing integer ids (all OOVs represented by UNK id), padded to length of longest sequence in the batch
	#			self.query_lens:
	#				numpy array of shape (batch_size) containing integers. The (truncated) length of each encoder input sequence (pre-padding).
	#			self.query_padding_mask:
	#				numpy array of shape (batch_size, <=max_query_steps), containing 1s and 0s. 1s correspond to real tokens in enc_batch and target_batch; 0s correspond to padding.
	#
	#		
	#	"""
		# Determine the maximum length of the encoder input sequence in this batch
			max_query_seq_len = max([ex.query_len for ex in example_list])
			self.max_query_len = max_query_seq_len
			#tf.logging.info("QUe : %d"%(max_query_seq_len))
			# Pad the encoder input sequences up to the length of the longest sequence
			for ex in example_list:
				ex.pad_query_input(max_query_seq_len, self.pad_id)
				if hps.use_query_elmo.value :
					ex.pad_query_input_raw(max_query_seq_len)

			# Initialize the numpy arrays
			# Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
			self.query_batch = np.zeros((hps.batch_size.value, max_query_seq_len), dtype=np.int32)
			self.query_lens = np.zeros((hps.batch_size.value), dtype=np.int32)
			self.query_padding_mask = np.zeros((hps.batch_size.value, max_query_seq_len), dtype=np.float32)
			self.query_segment_id = [[0] * max_query_seq_len for i in range(hps.batch_size.value)]
			self.query_bert_mask_id = [[0]* max_query_seq_len for i in range(hps.batch_size.value)]
			if hps.use_query_elmo.value:
				self.query_batch_raw = [ex.query_input_raw for ex in example_list]

			# Fill in the numpy arrays
			for i, ex in enumerate(example_list):
				self.query_batch[i, :] = ex.query_input[:]
				self.query_lens[i] = ex.query_len
				for j in xrange(ex.query_len):
					self.query_padding_mask[i][j] = 1
					self.query_bert_mask_id[i][j] = 1

			if hps.query_gcn.value:
				query_edge_list = []
				for ex in example_list:
					query_edge_list.append(ex.query_edge_list)
				if hps.use_bert.value:
					offset_list = []
					for ex in example_list:
						offset_list.append(ex.query_pos_offset)

				#note query_edge_list is list of query edge lists. The length is equal to the batch size
				self.query_adj_in, self.query_adj_out = data.get_adj(query_edge_list, hps.batch_size.value, max_query_seq_len,use_label_information=hps.use_label_information.value,																   flow_alone=hps.flow_alone.value, flow_combined=hps.flow_combined.value, keep_prob=hps.query_gcn_edge_dropout.value, use_bert=hps.use_bert.value, bert_mapping=offset_list, max_length=hps.max_query_steps.value)


	def init_decoder_seq(self, example_list, hps):
		"""Initializes the following:
				self.dec_batch:
					numpy array of shape (batch_size, max_dec_steps), containing integer ids as input for the decoder, padded to max_dec_steps length.
				self.target_batch:
					numpy array of shape (batch_size, max_dec_steps), containing integer ids for the target sequence, padded to max_dec_steps length.
				self.dec_padding_mask:
					numpy array of shape (batch_size, max_dec_steps), containing 1s and 0s. 1s correspond to real tokens in dec_batch and target_batch; 0s correspond to padding.
				"""
		# Pad the inputs and targets
		for ex in example_list:
			ex.pad_decoder_inp_targ(hps.max_dec_steps.value, self.pad_id)

		# Initialize the numpy arrays.
		# Note: our decoder inputs and targets must be the same length for each batch (second dimension = max_dec_steps) because we do not use a dynamic_rnn for decoding. However I believe this is possible, or will soon be possible, with Tensorflow 1.0, in which case it may be best to upgrade to that.
		self.dec_batch = np.zeros((hps.batch_size.value, hps.max_dec_steps.value), dtype=np.int32)
		self.target_batch = np.zeros((hps.batch_size.value, hps.max_dec_steps.value), dtype=np.int32)
		self.dec_padding_mask = np.zeros((hps.batch_size.value, hps.max_dec_steps.value), dtype=np.float32)

		# Fill in the numpy arrays
		for i, ex in enumerate(example_list):
			self.dec_batch[i, :] = ex.dec_input[:]
			self.target_batch[i, :] = ex.target[:]
			for j in xrange(ex.dec_len):
				self.dec_padding_mask[i][j] = 1

	def store_orig_strings(self, example_list):
		"""Store the original article and abstract strings in the Batch object"""
		self.original_articles = [ex.original_article for ex in example_list]  # list of lists
		self.original_abstracts = [ex.original_abstract for ex in example_list]  # list of lists
		self.original_abstracts_sents = [ex.original_abstract_sents for ex in example_list]  # list of list of lists
 #               if self.query_encoder:
#		 	self.original_queries = [ex.original_query for ex in example_list] # list of lists


class Batcher(object):
	"""A class to generate minibatches of data. Buckets examples together based on length of the encoder sequence."""

	BATCH_QUEUE_MAX = 100  # max number of batches the batch_queue can hold

	def __init__(self, data_, vocab, bert_vocab, hps, device_id, single_pass,data_format):
		"""Initialize the batcher. Start threads that process the data into batches.
	
		Args:
			data_path: tf.Example filepattern.
			vocab: Vocabulary objects
			hps: hyperparameters
			single_pass: If True, run through the dataset exactly once (useful for when you want to run evaluation on the dev or test set). Otherwise generate random batches indefinitely (useful for training).
		"""
		self._data = data_
		self._vocab = vocab
		self._hps = hps
		self._device_id = device_id
		self._single_pass = single_pass
		self._data_as_tf_example = data_format
		self.bert_vocab = bert_vocab
		# Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
		self._batch_queue = Queue.Queue(self.BATCH_QUEUE_MAX)
		self._example_queue = Queue.Queue(self.BATCH_QUEUE_MAX * self._hps.batch_size.value)
		#	self._data = pickle.load(open(data_path,'rb'))
		#	tf.logging.info(len(self._data))

		# Different settings depending on whether we're in single_pass mode or not
		if single_pass:
			self._num_example_q_threads = 1  # just one thread, so we read through the dataset just once
			self._num_batch_q_threads = 1  # just one thread to batch examples
			self._bucketing_cache_size = 1  # only load one batch's worth of examples before bucketing; this essentially means no bucketing
			self._finished_reading = False  # this will tell us when we're finished reading the dataset
		else:
			self._num_example_q_threads = 1  # num threads to fill example queue
			self._num_batch_q_threads = 1  # num threads to fill batch queue
			self._bucketing_cache_size = 1  # how many batches-worth of examples to load into cache before bucketing

		# Start the threads that load the queues
		self._example_q_threads = []
		for k in xrange(self._num_example_q_threads):
			self._example_q_threads.append(Thread(name=str(k), target=self.fill_example_queue))
			self._example_q_threads[-1].daemon = True
			self._example_q_threads[-1].start()

		self._batch_q_threads = []
		for _ in xrange(self._num_batch_q_threads):
			self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
			self._batch_q_threads[-1].daemon = True
			self._batch_q_threads[-1].start()

		# Start a thread that watches the other threads and restarts them if they're dead
		if not single_pass:  # We don't want a watcher in single_pass mode because the threads shouldn't run forever
			self._watch_thread = Thread(target=self.watch_threads)
			self._watch_thread.daemon = True
			self._watch_thread.start()

	def next_batch(self):
		"""Return a Batch from the batch queue.

		If mode='decode' then each batch contains a single example repeated beam_size-many times; this is necessary for beam search.

		Returns:
			batch: a Batch object, or None if we're in single_pass mode and we've exhausted the dataset.
		"""
		# If the batch queue is empty, print a warning
		if self._batch_queue.qsize() == 0:
			tf.logging.warning(
				'Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i',
				self._batch_queue.qsize(), self._example_queue.qsize())
			if self._single_pass and self._finished_reading:
				tf.logging.info("Finished reading dataset in single_pass mode.")
				return None

		batch = self._batch_queue.get()  # get the next Batch
		return batch

	def fill_example_queue(self):
		"""Reads data from file and processes into Examples which are then placed into the example queue."""
		input_gen = self.text_generator(data.example_generator(self._data, self._single_pass,self._device_id, data_as_tf_example=self._data_as_tf_example))
		count = 0
		query = None
		word_edge_list = None
		query_edge_list = None
		if self._data_as_tf_example:
			while True:
				try:
					 article, abstract, word_edge_list, query, query_edge_list, epoch_num = input_gen.next() # read the next example from file. article and abstract are both strings.
					 #tf.logging.info(random.randint(1,101))
				except StopIteration: # if there are no more examples:
					tf.logging.info("The example generator for this example queue filling thread has exhausted data.")
					if self._single_pass:
						tf.logging.info("single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
						self._finished_reading = True
						break
					else:
						raise Exception("single_pass mode is off but the example generator is out of data; error.")
				abstract_sentences = [sent.strip() for sent in data.abstract2sents(abstract)] # Use the <s> and </s> tags in abstract to get a list of sentences.
				example = Example(article, abstract_sentences, self._vocab, self._hps, word_edge_list=word_edge_list, query=query, query_edge_list=query_edge_list, epoch_num=epoch_num, bert_vocab=self.bert_vocab)
				self._example_queue.put(example)
		else:

			while True:
				try:
					curr_data = input_gen.next()
					count = count + 1
					article = curr_data['article']
					abstract = curr_data['abstract'].strip()
					if self._hps.word_gcn.value:
						word_edge_list = curr_data['word_edge_list']
					if self._hps.query_encoder.value:
						query = curr_data['query']
					if self._hps.query_gcn.value:
						query_edge_list = curr_data['query_edge_list']
				except Exception as e:  # if there are no more examples:
					tf.logging.info("The example generator for this example queue filling thread has exhausted data.")
					if self._single_pass:
						tf.logging.info(
							"single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
						self._finished_reading = True
						break
					else:
						tf.logging.info(e)
						raise Exception("single_pass mode is off but the example generator is out of data; error.")

				abstract_sentences = [sent.strip() for sent in data.abstract2sents(abstract)]  # Use the <s> and </s> tags in abstract to get a list of sentences.
				example = Example(article, abstract_sentences, self._vocab, self._hps, word_edge_list=word_edge_list, query=query, query_edge_list=query_edge_list, epoch_num=epoch_num)
				self._example_queue.put(example)  # place the Example in the example queue.

	def fill_batch_queue(self):
		"""Takes Examples out of example queue, sorts them by encoder sequence length, processes into Batches and places them in the batch queue.

		In decode mode, makes batches that each contain a single example repeated.
		"""
		while True:
			if self._hps.mode.value != 'decode':
				# Get bucketing_cache_size-many batches of Examples into a list, then sort
				inputs = []
				for _ in xrange(self._hps.batch_size.value * self._bucketing_cache_size):
					inputs.append(self._example_queue.get())
				inputs = sorted(inputs, key=lambda inp: inp.enc_len)  # sort by length of encoder sequence

				# Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
				batches = []
				for i in xrange(0, len(inputs), self._hps.batch_size.value):
					batches.append(inputs[i:i + self._hps.batch_size.value])
				
				for b in batches:  # each b is a list of Example objects
					self._batch_queue.put(Batch(b, self._hps, self._vocab))

			else:  # beam search decode mode
				ex = self._example_queue.get()
				b = [ex for _ in xrange(self._hps.batch_size.value)]
				self._batch_queue.put(Batch(b, self._hps, self._vocab))

	def watch_threads(self):
		"""Watch example queue and batch queue threads and restart if dead."""
		while True:
			time.sleep(60)
			for idx, t in enumerate(self._example_q_threads):
				if not t.is_alive():  # if the thread is dead
					tf.logging.error('Found example queue thread dead. Restarting.')
					new_t = Thread(target=self.fill_example_queue)
					self._example_q_threads[idx] = new_t
					new_t.daemon = True
					new_t.start()
			for idx, t in enumerate(self._batch_q_threads):
				if not t.is_alive():  # if the thread is dead
					tf.logging.error('Found batch queue thread dead. Restarting.')
					new_t = Thread(target=self.fill_batch_queue)
					self._batch_q_threads[idx] = new_t
					new_t.daemon = True
					new_t.start()

	def text_generator(self, example_generator):
		"""Generates article and abstract text from tf.Example.

		Args:
			example_generator: a generator of tf.Examples from file. See data.example_generator"""
		if self._data_as_tf_example:
			query_text = None
			query_edge_list = None
			word_edge_list = None
			
			while True:
				e, epoch_num = example_generator.next() # e is a tf.Example
				try:
					article_text = e.features.feature['article'].bytes_list.value[0] # document text
					abstract_text = e.features.feature['abstract'].bytes_list.value[0] # response text
					if self._hps.query_encoder.value:
						try:
							query_text = e.features.feature['query'].bytes_list.value[0] # context text
						except:
							query_text = ''
					if self._hps.word_gcn.value:
						word_edge_list = []
						if self._hps.use_default_graph.value:
							word_edge_list = word_edge_list + ast.literal_eval(e.features.feature['word_edge_list'].bytes_list.value[0])
						#tf.logging.info((word_edge_list[0]))
						if self._hps.use_coref_graph.value:
							word_edge_list = word_edge_list + ast.literal_eval(e.features.feature['word_coref_edge_list'].bytes_list.value[0])
						if self._hps.use_entity_graph.value:
							word_edge_list = word_edge_list + ast.literal_eval(e.features.feature['word_entity_edge_list'].bytes_list.value[0])
						if self._hps.use_lexical_graph.value:
							word_edge_list = word_edge_list + ast.literal_eval(e.features.feature['word_lexical_edge_list'].bytes_list.value[0])
					#	print(word_edge_list)
					

					if self._hps.query_gcn.value:
						query_edge_list = []
						if self._hps.use_default_graph.value:
							query_edge_list = query_edge_list + ast.literal_eval(e.features.feature['query_edge_list'].bytes_list.value[0])
						
						'''
						These are inter-sentence graph and may not be applicable
						if self._hps.use_coref_graph.value:
							query_edge_list = query_edge_list + ast.literal_eval(e.features.feature['query_coref_edge_list'].bytes_list.value[0])
						if self._hps.use_entity_graph.value:
							query_edge_list = query_edge_list + ast.literal_eval(e.features.feature['query_entity_edge_list'].bytes_list.value[0])
						if self._hps.use_lexical_graph.value:
							query_edge_list = query_edge_list + ast.literal_eval(e.features.feature['query_lexical_edge_list'].bytes_list.value[0])
						'''



				except ValueError:
					tf.logging.error('Failed to get article or abstract from example')
					continue
				if len(article_text)==0: # See https://github.com/abisee/pointer-generator/issues/1
					tf.logging.warning('Found an example with empty article text. Skipping it.')
				else:
					#tf.logging.info(abstract_text)
					yield (article_text, abstract_text, word_edge_list, query_text, query_edge_list, epoch_num)
			
		else:

			while True:
				e = example_generator.next()
				yield e
			
				

