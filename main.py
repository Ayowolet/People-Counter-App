"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

current_count=0
previous_count=0
total_count=0
start_time=0
duration=0
frame_count=0
wait_time=57
input_image= False
    
    
def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = None
    #client = mqtt.Client()
    #client.connect(MQTT_HOST,MQTT_PORT,MQTT_KEEPALIVE_INTERVAL)
    return client

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    model_input_shape = infer_network.get_input_shape()
    
    # Get the width and height of the model input
    sizes = (model_input_shape[3], model_input_shape[2])

    ### TODO: Handle the input stream ###
    # Check if the input is from a camera
    if args.input == 'CAM':
        input_stream = 0

    # Check if the input is an image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        input_image = True
        input_stream = args.input

    # Check if the input is a video
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "File does not exist, check again"
    
    
    ### TODO: Handle the input stream ###
    input_stream = cv2.VideoCapture(args.input)
    
    if input_stream:
        input_stream.open(args.input)
    if not input_stream:
        log.error("The video can't be opened")
    
    width = int(input_stream.get(3))
    height = int(input_stream.get(4))
    
    # Create a video output to see your result
    out = cv2.VideoWriter('out.mp4',0x00000021,30,(width,height))
    print(cv2.VideoWriter('out.mp4',0x00000021,30,(width,height)))
    
    ### TODO: Loop until stream is over ###
    while input_stream.isOpened() and not input_image:
        
        ### TODO: Read from the video capture ###
        flag, frame = input_stream.read()
        if not flag:
            break
        
        ### TODO: Pre-process the image as needed ###
        image = cv2.resize(frame, sizes)
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, *image.shape)
        
        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(frame)
        
        ### TODO: Wait for the result ###
        if infer_network.wait()==0:
        
            ### TODO: Get the results of the inference request ###
            results = infer_network.get_output()
            
            ### TODO: Extract any desired stats from the results ###
            current_count = 0
            for box in results[0][0]:# The shape of the output is 1x1x100x7
                conf = box[2]
                
                if conf >= prob_threshold:
                    xmin = int(box[3] * width)
                    ymin = int(box[4] * height)
                    xmax = int(box[5] * width)
                    ymax = int(box[6] * height)
                    
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 1)
                    current_count += 1
                
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            
            # Do this when a person is detected
            if current_count > previous_count:
                time_started = time.time()
                total_count += current_count - previous_count
                frame_count = 0
                
                total_count_payload = {
                    "total" : total_count
                }
                #client.publish("person", json.dumps(total_count_payload))
                print("person", json.dumps(total_count_payload))
            
            
            # when there is one less person
            if current_count < previous_count and frame_count < wait_time:
                current_count = previous_count
                frame_count += 1
            
            # when there is one less person for up to 30 frames
            if current_count < previous_count and frame_count == wait_time:
                duration = int(time.time() - time_started)
                
                duration_payload = {
                    "duration": duration
                }
                #client.publish("person/duration", json.dumps(duration_payload))
                print("person/duration", json.dumps(duration_payload))
                
            previous_count = current_count
            
            current_count_payload = {
                "count" : current_count
            }
            #client.publish("person", json.dumps(current_count_payload))
            print("person", json.dumps(current_count_payload))
            
        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
        if input_image:
            cv2.imwrite("output_image.jpg", frame)

    
    input_stream.release()
    cv2.destroyAllWindows()
    client.disconnect()
    infer_network.clean()
    
    
def main():
    """
    Load the network and parse the output.

    :return: None
    """
    print('We are here')
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
