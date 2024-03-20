with open("hyperparms_sequential_broad.json") as json_data_file:
    hp_values = json.load(json_data_file)

with open("tuner_kwargs.json") as json_data_file:
    hyper_search_kwargs = json.load(json_data_file)

tuner_init_kwargs = hyper_search_kwargs['init']
tuner_search_kwargs = hyper_search_kwargs['search']
evaluate_kwargs = hyper_search_kwargs['evaluate']
predict_kwargs = hyper_search_kwargs['predict']

target_metric_props = hyper_search_kwargs['target_metric']
# This is a dirty hack to enable shell scripts to override the target metric
# given in the json file
if "oom_cli_target_metric_name" in globals():
    if oom_cli_target_metric_name is not None:
        print("Target metric name overriden by CLI argument to "
              + oom_cli_target_metric_name
              + " instead of " + target_metric_props["name"]
              + " given in tuner_kwargs.json")
        target_metric_props["name"] = oom_cli_target_metric_name

# Another hack to read a compound metric from a dict if needed

if target_metric_props["name"].startswith('{'):
    # use the ast package to perform safe evaluation
    from ast import literal_eval
    metrics_dict = literal_eval(target_metric_props["name"])
    comp_met_name = ""
    for key in metrics_dict:
        comp_met_name += key + str(metrics_dict[key]) + "_"
    comp_met_name = comp_met_name[0:-1]
    target_metric_props["name"] = comp_met_name
    target_metric_props["compound"] = True
    target_metric_props["metrics_dict"] = metrics_dict
    # delete variables to reduce workspace overload
    del metrics_dict, comp_met_name
else:
    target_metric_props["compound"] = False


model_dir_prefix = get_modelsdir() / "mimic-omop/sequentialNN_3"
model_dir_suffix = ""
models_dir = model_dir_prefix / target_metric_props['name'] / model_dir_suffix

from tensorflow.keras import mixed_precision
from tensorflow import python as tfpython

# Check if there is a GPU available
print("Getting Device information...")
if len(tf.config.list_physical_devices('GPU'))>0:
    min_compute_cap = 9999
    # Get device infos
    device_info = tfpython.client.device_lib.list_local_devices()
    for device in device_info:
        if device.device_type=='GPU':
            split_str = device.physical_device_desc.split(',')
            for s in split_str:
                if s.startswith(' compute capability:'):
                    min_compute_cap = min(min_compute_cap,
                                          float(s.split(':')[1]))
    print("Minimum GPU compute capability is " + str(min_compute_cap))
    if min_compute_cap>7.0:
        print("Setting precision to mixed float 16 for speedup")
        mixed_precision.set_global_policy('mixed_float16')
    else:
        print("Keeping default precision settings.")
else:
    print("No GPU found, keeping default precision settings.")

