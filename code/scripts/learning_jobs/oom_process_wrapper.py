exec(open("syspath_setup.py").read())
import multiprocessing as mp
import traceback
import warnings
import time

# Get the target script argument
try:
    script_name = sys.argv[1]
except IndexError as ie:
    raise RuntimeError("Missing subprocess script name")

try:
    if len(sys.argv[2]) > 0:
        oom_cli_target_metric_name = sys.argv[2]
    else:
        oom_cli_target_metric_name = None
except IndexError as ie:
    oom_cli_target_metric_name = None

try:
    if len(sys.argv[3]) > 0:
        oom_cli_dir_suffix = "_" + sys.argv[3]
    else:
        oom_cli_dir_suffix = ""
except IndexError as ie:
    oom_cli_dir_suffix = ""

# Define a Process class to launch child processes in a spawn context with
# exception handling
# The spawn context is necessary to trigger TF re initialization
ctx = mp.get_context('spawn')


class SpawnProcess(ctx.Process):
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


# Define a utility worker function executing the desired script
def work_func():
    exec(open(script_name).read())


# PARENT PROCESS ONLY
if __name__ == '__main__':
    from tensorflow import config as tfconfig

    gpus = tfconfig.list_physical_devices('GPU')
    if gpus:
        tfconfig.experimental.set_memory_growth(gpus[0], True)
        tfconfig.set_logical_device_configuration(
            gpus[0],
            [tfconfig.LogicalDeviceConfiguration(memory_limit=1)])
    from keras_utils import errors as kuerrors

    max_tries = 20
    max_same_trial_tries = 2

    current_try = 0
    same_trial_retries = 0
    relaunch = True

    while relaunch:
        current_try += 1
        relaunch = False
        try:
            print("Launching child process...")
            child_process = SpawnProcess(target=work_func)
            child_process.start()
            child_process.join()
            if child_process.exception:
                error, traceback = child_process.exception
                child_process.terminate()
                raise error

        except kuerrors.KerasTunerSearchOOM as oom:
            same_trial_retries = 0
            warn_str = ("Parent process received a KerasTunerSearchOOM "
                        "exception from the child process with the following "
                        "message: "
                        + oom.message)

            if current_try < max_tries:
                warn_str += ("\nNumber of tries so far: "
                             + str(current_try)
                             + "(max tries set to " + str(max_tries) + ")"
                             + "\nRelaunching child process...")
                time.sleep(30)
                relaunch = True
            else:
                warn_str += "\nMax number of tries reached, terminating."
                raise RuntimeError(warn_str)
            warnings.warn(warn_str, category=RuntimeWarning)

        except kuerrors.KerasTunerSearchOOMSameTrial as oom:
            warn_str = (
                        "Parent process received a KerasTunerSearchOOMSameTrial"
                        "exception from the child process with the following "
                        "message: "
                        + oom.message)
            if same_trial_retries < max_same_trial_tries:
                same_trial_retries += 1
                warn_str += ("\nNumber of tries so far: "
                             + str(current_try)
                             + "(max tries set to " + str(max_tries) + ")"
                             + "Number of retries on the same trial so far: "
                             + str(same_trial_retries)
                             + "(max retries on the same trial set to " +
                             str(max_same_trial_tries) + ")"
                             + "\nRelaunching child process...")
                time.sleep(30)
                relaunch = True
            else:
                warn_str += "\nMax number of tries on the same trial " \
                            "reached, terminating."
                raise RuntimeError(warn_str)
            warnings.warn(warn_str, category=RuntimeWarning)
        except Exception as e:
            print(traceback)
            raise e
        else:
            print("Script process ended succefully after "
                  + str(current_try - 1) + " retries.")

# Prevent infinite loop, handle evaluate OOM
