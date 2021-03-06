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

"""This file contains code to read the train/eval/test data from file and process it, and read the vocab data from file and process it"""

import glob
import random
import struct
import csv
from tensorflow.core.example import example_pb2
from collections import defaultdict as ddict
import pickle
import scipy.sparse as sp
import tensorflow as tf
import numpy as np
import horovod.tensorflow as hvd 
import collections
import six
import unicodedata

# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]'  # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]'  # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]'  # This has a vocab id, which is used at the end of untruncated target sequences


# Note: none of <s>, </s>, [PAD], [UNK], [START], [STOP] should appear in the vocab file.

def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


class Vocab(object):
	"""Vocabulary class for mapping between words and ids (integers)"""

	def __init__(self, vocab_file, max_size):
		"""Creates a vocab of up to max_size words, reading from the vocab_file. If max_size is 0, reads the entire vocab file.

	Args:
	  vocab_file: path to the vocab file, which is assumed to contain "<word> <frequency>" on each line, sorted with most frequent word first. This code doesn't actually use the frequencies, though.
	  max_size: integer. The maximum size of the resulting Vocabulary."""
		self._word_to_id = {}
		self._id_to_word = {}
		self._count = 0  # keeps track of total number of words in the Vocab

		# [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
		for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
			self._word_to_id[w] = self._count
			self._id_to_word[self._count] = w
			self._count += 1

		# Read the vocab file and add words up to max_size
		with open(vocab_file, 'r') as vocab_f:
			for line in vocab_f:
				pieces = line.split()
				if len(pieces) != 2:
					print ('Warning: incorrectly formatted line in vocabulary file: %s\n' % line)
					continue
				w = pieces[0]
				if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
					raise Exception(
						'<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
				if w in self._word_to_id:
					raise Exception('Duplicated word in vocabulary file: %s' % w)
				self._word_to_id[w] = self._count
				self._id_to_word[self._count] = w
				self._count += 1
				if max_size != 0 and self._count >= max_size:
					print ("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (
					max_size, self._count))
					break

		print ("Finished constructing vocabulary of %i total words. Last word added: %s" % (
		self._count, self._id_to_word[self._count - 1]))

	def word2id(self, word):
		"""Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
		if word not in self._word_to_id:
			return self._word_to_id[UNKNOWN_TOKEN]
		return self._word_to_id[word]

	def id2word(self, word_id):
		"""Returns the word (string) corresponding to an id (integer)."""
		if word_id not in self._id_to_word:
			raise ValueError('Id not found in vocab: %d' % word_id)
		return self._id_to_word[word_id]

	def size(self):
		"""Returns the total size of the vocabulary"""
		return self._count

	def write_metadata(self, fpath):
		"""Writes metadata file for Tensorboard word embedding visualizer as described here:
	  https://www.tensorflow.org/get_started/embedding_viz

	Args:
	  fpath: place to write the metadata file
	"""
		print "Writing word embedding metadata file to %s..." % (fpath)
		with open(fpath, "w") as f:
			fieldnames = ['word']
			writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
			for i in xrange(self.size()):
				writer.writerow({"word": self._id_to_word[i]})
	
	def set_glove_embedding(self,fpath,embedding_dim):
		""" Creates glove embedding_matrix from file path"""
		emb = np.random.randn(self._count,embedding_dim)
#	tf.logging.info(emb[0])
		with open(fpath) as f: #python 3.x support 
			for k,line in enumerate(f):
				fields = line.split()
				if len(fields) - 1 != embedding_dim:
					# Sometimes there are funny unicode parsing problems that lead to different
					# fields lengths (e.g., a word with a unicode space character that splits
					# into more than one colum      n).  We skip those lines.  Note that if you have
					# some kind of long header, this could result in all of your lines getting
					# skipped.  It's hard to check for that here; you just have to look in the
					# embedding_misses_file and at the model summary to make sure things look
					# like they are supposed to.
					#logger.warning("Found line with wrong number of dimensions (expected %d, was %d): %s",
							#                  embedding_dim, len(fields) - 1, line)
					raise Exception("Found line with wrong number of dimensions (expected %d, was %d): %s",
											   embedding_dim, len(fields) - 1, line)
					continue
				word = fields[0]
				if word in self._word_to_id:
					vector = np.asarray(fields[1:], dtype='float32')
					emb[self._word_to_id[word]] = vector
#		if k%1000 == 0:
#		   tf.logging.info('glove : %d',k)
		self.glove_emb = emb


class BertVocab(object):
	"""
	While glove_vocab has been used as default. The term glove is misnomer. Glove_vocab represents normal vocab in this file
	This function converts individual tokens to their respective word piece tokens
	"""
	
	def __init__(self, glove_vocab, bert_vocab_file_path):
		self.bert_vocab = collections.OrderedDict()
		self.glove_vocab = glove_vocab
		index = 0
		with tf.gfile.GFile(bert_vocab_file_path, "r") as reader: #obtain bert vocab
			while True:
				token = convert_to_unicode(reader.readline())
				if not token:
					break
				token = token.strip()
				self.bert_vocab[token] = index
				index += 1
		not_found = 0 
		self.index_map_glove_to_bert = {}
		
		for i in range(glove_vocab._count):
			if glove_vocab._id_to_word[i] in self.bert_vocab:
				self.index_map_glove_to_bert[i] = [self.bert_vocab[glove_vocab._id_to_word[i]]]
			else: #Word Piece Tokenizer
				not_found = not_found + 1
				new_tokens = [] 
				token = glove_vocab._id_to_word[i]
				chars = list(token)
				is_bad = False
				start = 0
				sub_tokens = []
				while start < len(chars):
					end = len(chars)
					cur_substr = None
					while start < end:
						substr = "".join(chars[start:end])
						if start > 0:
							substr = "##" + substr
						if substr in self.bert_vocab:
							cur_substr = substr
							break
						end -= 1
					if cur_substr is None:
						is_bad = True
						break
					sub_tokens.append(cur_substr)
					start = end

				if is_bad:
					new_tokens.append(self.bert_vocab['[UNK]'])
				else:
					sub_tokens_bert = [self.bert_vocab[s] for s in sub_tokens]
					new_tokens = new_tokens + sub_tokens_bert

				self.index_map_glove_to_bert[i] = new_tokens


		tf.logging.info(not_found)		


	def convert_glove_to_bert_indices(self, token_ids):
		"""
		Converts words to their respective word-piece tokenized indices
		token_ids : ids from the word 
		"""
		
		new_tokens = [self.bert_vocab['[CLS]']] #As pert the bert repo instructions
		offset = 1
		pos_offset = []
		for token_id in token_ids:
			pos_offset.append(offset) #wordpiece tokenizer can return more than one index hence we maintain an offset array. This is useful for the BERT + GCN experiments. 
			if token_id in self.index_map_glove_to_bert:
				bert_tokens = self.index_map_glove_to_bert[token_id]
				offset = offset + len(bert_tokens) - 1 
				#new_tokens.append(self.index_map_glove_to_bert[token_id])
				new_tokens = new_tokens + bert_tokens
			else:
				#wordpiece might be redundant for training data. Keep for unseen instances
				token = glove_vocab._id_to_word[token_id]
				chars = list(token)

				is_bad = False
				start = 0
				sub_tokens = []
				while start < len(chars):
					end = len(chars)
					cur_substr = None
					while start < end:
						substr = "".join(chars[start:end])
						if start > 0:
							substr = "##" + substr
						if substr in self.vocab:
							cur_substr = substr
							break
						end -= 1
					if cur_substr is None:
						is_bad = True
						break
					sub_tokens.append(cur_substr)
					start = end

				if is_bad:
					#new_tokens.append(self.index_map_glove_to_bert['[UNK]'])
					new_token = new_token + self.index_map_glove_to_bert['[UNK]']
				else:
					sub_tokens_bert = [self.bert_vocab[s] for s in sub_tokens]
					new_tokens = new_tokens + sub_tokens_bert
					offset = offset + len(sub_tokens_bert) - 1

			
		new_tokens.append(self.bert_vocab['[SEP]'])
		return new_tokens, pos_offset







def convert_by_vocab(vocab, items):
  """Converts a sequence of [tokens|ids] using the vocab."""
  output = []
  for item in items:
	if item in vocab:
	  output.append(vocab[item])
	else:
	  output.append(vocab['[UNK]'])
  return output


def convert_tokens_to_ids(vocab, tokens):
  return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
  return convert_by_vocab(inv_vocab, ids)


def example_generator(data_path, single_pass, device_rank,data_as_tf_example=True):
	"""Generates tf.Examples from data files.

	Binary data format: <length><blob>. <length> represents the byte size
	of <blob>. <blob> is serialized tf.Example proto. The tf.Example contains
	the tokenized article text and summary.

  Args:
	data_path:
	  Path to tf.Example data files. Can include wildcards, e.g. if you have several training data chunk files train_001.bin, train_002.bin, etc, then pass data_path=train_* to access them all.
	single_pass:
	  Boolean. If True, go through the dataset exactly once, generating examples in the order they appear, then return. Otherwise, generate random examples indefinitely.

  Yields:
	Deserialized tf.Example.
  """
		
	random.seed(device_rank+1)
	if data_as_tf_example:
		epoch = 0
		while True:
			filelist = glob.glob(data_path) # get the list of datafiles
			assert filelist, ('Error: Empty filelist at %s' % data_path) # check filelist isn't empty
			if single_pass:
				filelist = sorted(filelist)
			else:
				random.shuffle(filelist)
			#tf.logging.info(filelist)				
			for file_no, f in enumerate(filelist):
				reader = open(f, 'rb')
				all_examples = []
				while True:
					len_bytes = reader.read(8)
					if not len_bytes: 
						if not single_pass:
							random.shuffle(all_examples)
						for k in all_examples:
							yield example_pb2.Example.FromString(k), epoch
						break # finished reading this file
					
					str_len = struct.unpack('q', len_bytes)[0]
					example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
					all_examples.append(example_str)

			if single_pass:
				print "example_generator completed reading all datafiles. No more data."
				break

		
	else:
		#pickle format
		while True:
			if single_pass:
				for data_ in data_path:
					for i in data_:
						yield i
			else:
				random.shuffle(data_path)
				for data_ in data_path:
					new_data = data_
					x = np.arange(len(new_data))
					np.random.shuffle(x)
					# random.shuffle(new_data)
					for i in x:
						yield new_data[i]
			if single_pass:
				break
	 

def article2ids(article_words, vocab):
	"""Map the article words to their ids. Also return a list of OOVs in the article.

  Args:
	article_words: list of words (strings)
	vocab: Vocabulary object

  Returns:
	ids:
	  A list of word ids (integers); OOVs are represented by their temporary article OOV number. If the vocabulary size is 50k and the article has 3 OOVs, then these temporary OOV numbers will be 50000, 50001, 50002.
	oovs:
	  A list of the OOV words in the article (strings), in the order corresponding to their temporary article OOV numbers."""
	ids = []
	oovs = []
	unk_id = vocab.word2id(UNKNOWN_TOKEN)
	for w in article_words:
		i = vocab.word2id(w)
		if i == unk_id:  # If w is OOV
			if w not in oovs:  # Add to list of OOVs
				oovs.append(w)
			oov_num = oovs.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
			ids.append(vocab.size() + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
		else:
			ids.append(i)
	return ids, oovs


def abstract2ids(abstract_words, vocab, article_oovs):
	"""Map the abstract words to their ids. In-article OOVs are mapped to their temporary OOV numbers.

  Args:
	abstract_words: list of words (strings)
	vocab: Vocabulary object
	article_oovs: list of in-article OOV words (strings), in the order corresponding to their temporary article OOV numbers

  Returns:
	ids: List of ids (integers). In-article OOV words are mapped to their temporary OOV numbers. Out-of-article OOV words are mapped to the UNK token id."""
	ids = []
	unk_id = vocab.word2id(UNKNOWN_TOKEN)
	for w in abstract_words:
		i = vocab.word2id(w)
		if i == unk_id:  # If w is an OOV word
			if w in article_oovs:  # If w is an in-article OOV
				vocab_idx = vocab.size() + article_oovs.index(w)  # Map to its temporary article OOV number
				ids.append(vocab_idx)
			else:  # If w is an out-of-article OOV
				ids.append(unk_id)  # Map to the UNK token id
		else:
			ids.append(i)
	return ids



def outputids2words(id_list, vocab, article_oovs):
	"""Maps output ids to words, including mapping in-article OOVs from their temporary ids to the original OOV string (applicable in pointer-generator mode).

  Args:
	id_list: list of ids (integers)
	vocab: Vocabulary object
	article_oovs: list of OOV words (strings) in the order corresponding to their temporary article OOV ids (that have been assigned in pointer-generator mode), or None (in baseline mode)

  Returns:
	words: list of words (strings)
  """
	words = []
	for i in id_list:
		try:
			w = vocab.id2word(i)  # might be [UNK]
		except ValueError as e:  # w is OOV
			assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
			article_oov_idx = i - vocab.size()
			try:
				w = article_oovs[article_oov_idx]
			except ValueError as e:  # i doesn't correspond to an article oov
				raise ValueError('Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (i, article_oov_idx, len(article_oovs)))
		words.append(w)
	return words
		


def abstract2sents(abstract):
	"""Splits abstract text from datafile into list of sentences.

  Args:
	abstract: string containing <s> and </s> tags for starts and ends of sentences

  Returns:
	sents: List of sentence strings (no tags)"""
	cur = 0
	sents = []
	while True:
		try:
			start_p = abstract.index(SENTENCE_START, cur)
			end_p = abstract.index(SENTENCE_END, start_p + 1)
			cur = end_p + len(SENTENCE_END)
			sents.append(abstract[start_p + len(SENTENCE_START):end_p].strip())
		except ValueError as e:  # no more sentences
			return sents


def show_art_oovs(article, vocab):
	"""Returns the article string, highlighting the OOVs by placing __underscores__ around them"""
	unk_token = vocab.word2id(UNKNOWN_TOKEN)
	words = article.split(' ')
	words = [("__%s__" % w) if vocab.word2id(w) == unk_token else w for w in words]
	out_str = ' '.join(words)
	return out_str


def show_abs_oovs(abstract, vocab, article_oovs):
	"""Returns the abstract string, highlighting the article OOVs with __underscores__.

  If a list of article_oovs is provided, non-article OOVs are differentiated like !!__this__!!.

  Args:
	abstract: string
	vocab: Vocabulary object
	article_oovs: list of words (strings), or None (in baseline mode)
  """
	unk_token = vocab.word2id(UNKNOWN_TOKEN)
	words = abstract.split(' ')
	new_words = []
	for w in words:
		if vocab.word2id(w) == unk_token:  # w is oov
			if article_oovs is None:  # baseline mode
				new_words.append("__%s__" % w)
			else:  # pointer-generator mode
				if w in article_oovs:
					new_words.append("__%s__" % w)
				else:
					new_words.append("!!__%s__!!" % w)
		else:  # w is in-vocab word
			new_words.append(w)
	out_str = ' '.join(new_words)
	return out_str


dep_list = ['cc', 'agent', 'ccomp', 'prt', 'meta', 'nsubjpass', 'csubj', 'conj', 'amod', 'poss', 'neg', 'csubjpass',
			'mark', 'auxpass', 'advcl', 'aux', 'ROOT', 'prep', 'parataxis', 'xcomp', 'nsubj', 'nummod', 'advmod',
			'punct', 'quantmod', 'acomp', 'compound', 'pcomp', 'intj', 'relcl', 'npadvmod', 'case', 'attr', 'dep',
			'appos', 'det', 'nmod', 'dobj', 'dative', 'pobj', 'expl', 'predet', 'preconj', 'oprd', 'acl', 'flow']

dep_dict = {label: i for i, label in enumerate(dep_list)}


def get_specific_adj(batch_list, batch_size, max_nodes, label, encoder_lengths, use_both=True, keep_prob=1.0, use_bert=False, bert_mapping=None, max_length=300):
	

	adj_main_in = []
	adj_main_out = []
	
	if bert_mapping is None:
		bert_mapping = [[] for i in range(len(batch_list))] #empty array for allowing in the next loop


	for edge_list, enc_length, offset_list in zip(batch_list, encoder_lengths, bert_mapping):
		#print(edge_list)
		curr_adj_in = []
		curr_adj_out = []
		curr_data_in = []
		curr_data_out = []
		seen_nodes = []
		


		for s, d, lbl in edge_list:
			if s >=max_nodes or d >=max_nodes or s>=max_length or d>=max_length:
				continue
			if use_bert:
				src = s + offset_list[s]
				dest = d + offset_list[d]
			else:
				src = s
				dest = d
			seen_nodes.append(src)
			seen_nodes.append(dest)

			if lbl!=label:
				continue
			#if src >= max_nodes or dest >= max_nodes:
			#	continue
			x = np.random.uniform()
			if x<=keep_prob:
				curr_adj_out.append((src, dest))
				curr_data_out.append(1.0)
				if use_both:
					curr_adj_in.append((dest, src))
					curr_data_in.append(1.0)
				else:
					curr_adj_out.append((dest, src))    
					curr_data_out.append(1.0)
		
		'''		
		Use this snippet when you need to use the A + I condition (refer README)
		seen_nodes = list(set(seen_nodes))
		
		for src in range(enc_length): #A + I for entity and coref
			curr_adj_out.append((src, src))
			curr_data_out.append(1.0)
			if use_both:
				curr_adj_in.append((src, src))
				curr_data_in.append(1.0)
		'''		  		

		if len(curr_adj_in) == 0:
			adj_in = sp.coo_matrix((max_nodes, max_nodes))
		else:
			adj_in = sp.coo_matrix((curr_data_in, zip(*curr_adj_in)), shape=(max_nodes, max_nodes))
		if len(curr_adj_out) == 0:
			adj_out = sp.coo_matrix((max_nodes, max_nodes))
		else:
			adj_out = sp.coo_matrix((curr_data_out, zip(*curr_adj_out)), shape=(max_nodes, max_nodes))

		adj_main_in.append(adj_in)
		adj_main_out.append(adj_out)
		
	return adj_main_in, adj_main_out                

def get_adj(batch_list, batch_size, max_nodes, use_label_information=True, label_dict=dep_dict,flow_alone=False, flow_combined=False, keep_prob=1.0, use_bert=False, bert_mapping=None, max_length=300):
	adj_main_in, adj_main_out = [], []
	max_labels = 45

	if bert_mapping is None:
		bert_mapping = [[] for i in range(len(batch_list))] #empty array for allowing in the next loop

	for edge_list, offset_list in zip(batch_list, bert_mapping):
		adj_in, adj_out = {}, {}

		l_e = len(edge_list)
		in_ind, in_data = ddict(list), ddict(list)
		out_ind, out_data = ddict(list), ddict(list)
		count = 0
	  
		for s, d, lbl_ in edge_list:
			if s>=max_nodes or d >= max_nodes or s>=max_length or d>=max_length:
				continue
			if use_bert:
				try:
					src = s + offset_list[s]
				except:
					tf.logging.info(s)
					tf.logging.info(len(offset_list))
	
				dest = d + offset_list[d]
			else:
				src = s
				dest = d
			#if src >= max_nodes or dest >= max_nodes:
			#	continue
			
			if flow_alone:
				lbl = 0
				if src+1 < max_nodes:
					x = np.random.uniform()

					if x<= keep_prob:
						out_ind[lbl].append((src, src+1))
						out_data[lbl].append(1.0)
					
					x = np.random.uniform()
					
					if x<=keep_prob:
						in_ind[lbl].append((src+1, src))
						in_data[lbl].append(1.0)
					
			   
			else:    
				if lbl_ not in label_dict:
					continue

				lbl = label_dict[lbl_]
				
				if not use_label_information: #all assigned the same label information
					lbl = 0 
				
				x = np.random.uniform()
				if x<=keep_prob:
					out_ind[lbl].append((src, dest))
					out_data[lbl].append(1.0)

				x = np.random.uniform()
				if x<=keep_prob:    
					in_ind[lbl].append((dest, src))
					in_data[lbl].append(1.0)
					
				if flow_combined and dest!=src+1:
					if not use_label_information: #all assigned the same label information
						lbl = 0
					else:
						lbl = label_dict['flow']
					out_ind[lbl].append((src, src+1))
					out_data[lbl].append(1.0)

					in_ind[lbl].append((src+1, src))
					in_data[lbl].append(1.0)    
		

		count = count + 1

		if flow_combined:
			max_labels = max_labels + 1
		if not use_label_information:
			max_labels = 1

		for lbl in range(max_labels):
			if lbl not in out_ind:
				adj_out[lbl] = sp.coo_matrix((max_nodes, max_nodes))
			else:
				adj_out[lbl] = sp.coo_matrix((out_data[lbl], zip(*out_ind[lbl])), shape=(max_nodes, max_nodes))
			
			if lbl not in in_ind:
				adj_in[lbl] = sp.coo_matrix((max_nodes, max_nodes))
			else:
				adj_in[lbl] = sp.coo_matrix((in_data[lbl], zip(*in_ind[lbl])), shape=(max_nodes, max_nodes))


		adj_main_in.append(adj_in)
		adj_main_out.append(adj_out)
		# print(adj_main_in)

	return adj_main_in, adj_main_out


def create_glove_embedding_matrix (vocab,vocab_size,emb_dim,glove_path):
	emb = np.random.rand(vocab_size,emb_dim)
	count = 0
	with open(args['glove_path'],encoding='utf-8') as f: #python 3.x support 
		#all_lines = []
	#with codecs.open(args['glove_path'],'r',encoding='utf-8') as f: #python 2.x support
		for line in f:
			fields = line.split()
			if len(fields) - 1 != embedding_dim:
				# Sometimes there are funny unicode parsing problems that lead to different
				# fields lengths (e.g., a word with a unicode space character that splits
				# into more than one colum      n).  We skip those lines.  Note that if you have
				# some kind of long header, this could result in all of your lines getting
				# skipped.  It's hard to check for that here; you just have to look in the
				# embedding_misses_file and at the model summary to make sure things look
				# like they are supposed to.
				#logger.warning("Found line with wrong number of dimensions (expected %d, was %d): %s",
						#                  embedding_dim, len(fields) - 1, line)
				raise Exception("Found line with wrong number of dimensions (expected %d, was %d): %s",
										   embedding_dim, len(fields) - 1, line)
				continue
			word = fields[0]
			if word in w2id:
				vector = np.asarray(fields[1:], dtype='float32')

	
	return emb

