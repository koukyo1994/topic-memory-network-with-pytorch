import argparse

from pathlib import Path

TOPIC_NUM = 0
HIDDEN_NUM = [500, 500]
TOPIC_EMB_DIM = 150
MAX_SEQ_LEN = 24
BATCH_SIZE = 32
MAX_EPOCH = 800
MIN_EPOCH = 50
PATIENT = 10
PATIENT_GLOBAL = 60
PRE_TRAIN_EPOCHS = 50
ALTER_TRAIN_EPOCHS = 50
TARGET_SPARSITY = 0.75
KL_GROWING_EPOCH = 0
SHORTCUT = True
TRANSFORM = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", dafault="data/tmn_out")
    parser.add_argument("--embedding", default="data/embedding")
    parser.add_argument("--log_dir", default="log/")
    parser.add_argument("--output", defualt="out/")
    parser.add_argument("--n_topics", default=50, type=int)

    args = parser.parse_args()

    log_path = Path(args.log_dir)
    log_path.mkdir(exist_ok=True)

    emb_path = Path(args.embedding)
    emb_path.mkdir(exist_ok=True)

    data_path = Path(args.data)
    data_path.mkdir(exist_ok=True)

    out_path = Path(args.output)
    out_path.mkdir(exist_ok=True)
