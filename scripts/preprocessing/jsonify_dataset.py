import argparse
import json

from m2t.preprocessing import _JSONIFIERS, get_jsonifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=list(_JSONIFIERS.keys()),
        help="dataset to use.",
        required=True,
    )
    parser.add_argument("--input-dir", required=True, help="input dir for the dataset.")
    parser.add_argument(
        "--output-dir",
        default="./tmp",
        help="where to write the json file(s) for the dataset.",
    )
    parser.add_argument("--split", choices=["train", "test", "validation", "eval", "all"])
    parser.add_argument(
        "--dataset-kwargs",
        default=None,
        help="json-formatted string of dataset-specific kwargs; "
        + 'example: `{"minimum_caption_length": 8}` ',
    )

    args = parser.parse_args()

    jsonify_kwargs = json.loads(args.dataset_kwargs) if args.dataset_kwargs else {}

    jsonifier = get_jsonifier(
        args.dataset, input_dir=args.input_dir, split=args.split, **jsonify_kwargs
    )
    jsonifier.load_raw_data()
    jsonifier.export_to_json(args.output_dir)

    return


if __name__ == "__main__":
    main()
