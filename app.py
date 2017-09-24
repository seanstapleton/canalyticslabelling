#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import urlopen
from urllib.parse import parse_qs

from flask import Flask, redirect, render_template, request, jsonify

import argparse
import sys

import numpy as np
import tensorflow as tf

app = Flask(__name__)

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  image = urlopen(file_name)
  file_reader = image.read()
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

@app.route('/', methods=['GET', 'POST'])
def homepage():
    print(request.method)
    if request.method == 'GET':
        return 'Hello, World!'
    else:
        file_name = request.form["image_Uri"]
        model_file = "retrained_graph.pb"
        label_file = "retrained_labels.txt"
        input_height = 299
        input_width = 299
        input_mean = 128
        input_std = 128
        input_layer = "Mul"
        output_layer = "final_result"

        graph = load_graph(model_file)
        t = read_tensor_from_image_file(file_name,
                                        input_height=input_height,
                                        input_width=input_width,
                                        input_mean=input_mean,
                                        input_std=input_std)

        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        input_operation = graph.get_operation_by_name(input_name);
        output_operation = graph.get_operation_by_name(output_name);

        with tf.Session(graph=graph) as sess:
          results = sess.run(output_operation.outputs[0],
                            {input_operation.outputs[0]: t})
        results = np.squeeze(results)

        top_k = results.argsort()[-5:][::-1]
        labels = load_labels(label_file)
        for i in top_k:
          print(labels[i], results[i])
        return jsonify({'labels': 0, 'pred': 1})

@app.errorhandler(500)
def server_error(e):
    return """
    An internal server error occurred
    """, 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8081, debug=True)
