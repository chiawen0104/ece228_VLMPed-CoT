from pie_data import PIE
from jaad_data import JAAD
import os
import yaml
from data_generation_class import data_generation

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def generate_data(data_split, data_parameter_path, saved_files_path):
    """
    Generate trajectory-sequence data for a given dataset split (e.g., train/val/test)
    based on configuration options stored in a YAML file, and save the processed output.
    """
    # Load data/model configuration from configs.yaml under the given parameter directory
    with open(os.path.join(data_parameter_path, 'configs.yaml'), 'r') as yamlfile:
        opts = yaml.safe_load(yamlfile)


    # Extract model-related options and data-related options
    model_opts = opts['model_opts']
    data_opts = opts['data_opts']

    # Determine "time to event" (tte). If time_to_event is a list/tuple, use the 2nd value;
    # otherwise use it directly as an integer.
    tte = model_opts['time_to_event'] if isinstance(model_opts['time_to_event'], int) else \
                model_opts['time_to_event'][1]

    # Ensure the minimum track size is long enough to cover observation length + tte
    data_opts['min_track_size'] = model_opts['obs_length'] + tte

    # Initialize the dataset handler based on dataset name in model options
    if model_opts['dataset'] == 'pie':
        # PIE dataset: create dataset object with the specified local path
        imdb = PIE(data_path='/data/chl343/PIE')
        # Print dataset statistics (optional, but useful for sanity-check)
        imdb.get_data_stats()
    elif model_opts['dataset'] == 'jaad':
        # JAAD dataset: create dataset object with the specified local path
        imdb = JAAD(data_path="/data/chl343/JAAD")
    else:
        # If dataset name is not supported, stop with a clear error
        raise ValueError("{} dataset is incorrect".format(model_opts['dataset']))

    # Create the data generation/processing helper class
    method_class = data_generation()

    # Generate raw trajectory sequences for the given split using data options
    seq = imdb.generate_data_trajectory_sequence(data_split, **data_opts)

    # Convert / package sequences into the final training/testing format and save to disk
    method_class.get_data(data_split, seq, model_opts, saved_files_path)

    # Signal completion
    print('data generation complete')


if __name__ == '__main__':
    # Directory that contains configs.yaml and other parameter files for this data type
    data_parameter_path = 'data parameter/pie'

    # Root directory where the generated data will be saved
    data_save_path = 'data'

    # Subfolder name (data type) under the save root
    data_type = 'pie'

    # Which split to generate: 'train', 'val', or 'test'
    data_split = 'test'

    # Final output directory: <data_save_path>/<data_type>
    data_save_path = os.path.join(data_save_path, data_type)

    # Create the output directory if it does not exist
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)

    # Run data generation for the specified split
    generate_data(data_split, data_parameter_path, data_save_path)
