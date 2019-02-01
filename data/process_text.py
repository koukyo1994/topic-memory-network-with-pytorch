import json
import pickle
import argparse

from pathlib import Path

from sklearn.model_selection import train_test_split

from util import get_logger
from loader import DataLoader
from filtering import get_wids, create_dictionary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="exp")
    parser.add_argument("--input")
    parser.add_argument("--output")

    args = parser.parse_args()
    logger = get_logger(tag=args.tag)

    if (not args.input) or (not args.output):
        print(f"Usage: python {__file__} <input_data_file> <output_dir>")
        exit(1)

    loader = DataLoader(args.input)
    msgs, labels, label_dict = loader.get_data()

    dictionary, bow_dictionary = create_dictionary(msgs)
    seq_title, bow_title, label_title = get_wids(
        msgs, dictionary, bow_dictionary, labels, logger)

    seq_title_train, seq_title_test, \
        bow_title_train, bow_title_test, \
        label_train, label_test = train_test_split(
            seq_title,
            bow_title,
            label_title,
            shuffle=True,
            test_size=0.2,
            random_state=42)

    logger.info("Save data")
    out = Path(args.output)
    out.mkdir(exist_ok=True)
    pickle.dump(seq_title, open(out / "dataMsg.pkl", "wb"))
    pickle.dump(seq_title_train, open(out / "dataMsgTrain.pkl", "wb"))
    pickle.dump(seq_title_test, open(out / "dataMsgTest.pkl", "wb"))
    pickle.dump(bow_title, open(out / "dataMsgBow.pkl", "wb"))
    pickle.dump(bow_title_train, open(out / "dataMsgBowTrain.pkl", "wb"))
    pickle.dump(bow_title_test, open(out / "dataMsgBowTest.pkl", "wb"))
    pickle.dump(label_title, open(out / "dataMsgLabel.pkl", "wb"))
    pickle.dump(label_train, open(out / "dataMsgLabelTrain.pkl", "wb"))
    pickle.dump(label_test, open(out / "dataMsgLabelTest.pkl", "wb"))
    dictionary.save(str(out / "dataDictSeq.npy"))
    bow_dictionary.save(str(out / "dataDictBow.npy"))
    json.dump(label_dict, open(out / "labelDict.json", "w"), indent=4)
    logger.info("done!")
