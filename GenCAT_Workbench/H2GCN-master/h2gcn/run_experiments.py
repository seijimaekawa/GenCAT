import models
import datasets
import numpy as np
from modules import arguments, logger, monitor
from models import tf
tf.config.experimental_run_functions_eagerly(True)
# tf.get_logger().setLevel("ERROR")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

parser = arguments.create_parser()
parser.add_argument("--debug", action="store_true", help="Debug in VS Code")
parser.add_argument("--random_seed", type=int, default=123)
parser.add_argument("--use_full_gpu", action="store_true", dest="_use_full_gpu")
parser.add_argument("--interactive", "-i", action="store_true", dest="_interactive")
known_args, _ = parser.parse_known_args()
if known_args.debug:
    import ptvsd
    print("Waiting for debugger attach")
    ptvsd.enable_attach(address=('localhost', 5678), redirect_output=True)
    ptvsd.wait_for_attach()
    breakpoint()

if not known_args._use_full_gpu:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

if known_args.random_seed:
    tf.random.set_seed(known_args.random_seed)

import time
start = time.time()

subparser = parser.add_argument_group(
    "Experiment arguments (run_experiments.py)")
subparser.add_argument("--epochs", type=int, default=1000,
                       help="(default: %(default)s)")

models.add_subparsers(parser)
datasets.add_subparsers(parser)
logger.add_subparser_args(parser)
monitor.add_subparser_args(parser)

# Prepare configs and data
args = arguments.parse_args(parser)

for func in args.objects["pretrain_callbacks"]:
    func(**args.objects["tensors"])

args.current_epoch = 0
epoch_time = []
while args.current_epoch < args.epochs:
    t = time.time()
    args.current_epoch += 1
    for func in args.objects["pre_epoch_callbacks"]:
        func(args.current_epoch, args)
    args.objects["epoch_stats"] = dict()
    args.objects["epoch_stats"].update(
        args.objects["train_step"](**args.objects["tensors"]))
    args.objects["epoch_stats"].update(
        args.objects["test_step"](**args.objects["tensors"]))
    for func in args.objects["post_epoch_callbacks"]:
        func(args.current_epoch, args)
    epoch_time.append(time.time() - t)
    while args.current_epoch >= args.epochs and len(args.objects["post_train_callbacks"]) > 0:
        func = args.objects["post_train_callbacks"].popleft()
        test_acc = func(args)

elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
with open(args.root_path+"experimental_results/H2GCN_"+args.dataset+"_iter"+str(args.iter_count), 'w') as f:
    f.write(str(args.objects["epoch_stats"]['test_accuracy'].numpy()) + "\n" + str(elapsed_time) + "[sec]\n" + str(np.average(epoch_time)) + "\n" + str(args.current_epoch))
    # f.write(str(args.objects["best_val_stats"]['test_accuracy'].numpy()) + "\n" + str(elapsed_time) + "[sec]\n" + str(np.average(epoch_time)) + "\n" + str(args.objects["best_val_stats"]["epoch"]))

if args._interactive:
    import IPython
    IPython.embed()
