import os
import time
import pickle
from argparse import ArgumentParser


def combine_data(raw_data_dir="raw_data", suffix=".pickle",
                 output="combined_measurement_data.pickle", save_output_to_file=True):
    """Simply combine different RepchainDataFrameHolder objects with a common suffix without further processing.

    Parameters
    ----------
    raw_data_dir : str (optional)
        Directory containing the data (pickle files) to be combined.
    suffix : str (optional)
        Common ending of the files to be combined.
    output: str (optional)
        filename of combined file
    save_output_to_file : bool (optional)
        if True, saves output into a file (default) and returns the combined data
        if False, nothing is saved to a file and the combined data is only returned
    """
    start_time = time.time()

    # collect all data from folder, process and put together
    num_files = 0
    combined_data = None
    sf = len(suffix)
    if not os.path.exists(raw_data_dir):
        raise NotADirectoryError("No raw_data directory found!")
    for filename in os.listdir(raw_data_dir):
        if filename[-sf:] == suffix:
            if os.path.isfile("{}/{}".format(raw_data_dir, filename)):
                # read out RepChain Containers
                new_data = pickle.load(open("{}/{}".format(raw_data_dir, filename), "rb"))
                if combined_data is None:
                    combined_data = new_data
                else:
                    combined_data.combine(new_data, assert_equal_baseline_parameters=False)

                num_files += 1
    num_suc = len(combined_data.dataframe.index)

    if save_output_to_file is True:
        # save combined data as single pickle file
        pickle.dump(combined_data, open(output, "wb"))

    runtime = time.time() - start_time

    print("\n\nCombined {} files containing {} successes in {} seconds into output file {}."
          .format(num_files, num_suc, runtime, output))

    # returns the combined data
    return combined_data


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-d", "--raw_data_dir", required=False, type=str, default="raw_data",
                        help="Directory containing the raw data (i.e. pickled RepchainDataFrameHolders)"
                             "that should be combined.")
    parser.add_argument("-s", "--suffix", required=False, type=str, default=".pickle",
                        help="Common suffix of all the raw-data files in raw_data_dir that should be combined.")
    parser.add_argument("-o", "--output", required=False, type=str, default="combined_qkd_data.pickle",
                        help="File to store combined data in. This is a pickled RepchainDataFrameHolder including"
                             "all the results of all the pickled RepchainDataFrameHolders in raw_data_dir with the "
                             "right suffix.")

    args = parser.parse_args()
    combine_data(raw_data_dir=args.raw_data_dir, suffix=args.suffix, output=args.output, save_output_to_file=True)
