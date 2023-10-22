import torch.optim as optim
from os.path import dirname, join
from os import remove

_root_dir = dirname(dirname(__file__))

MODEL_PARAM = join(_root_dir, 'param')
MODEL_NAME = ("continue_train_sst_{:s}.ratio_{:f}.new_data_num_{:d}.base_{:s}.model_{:s}.epoch_{:d}.loss_{:f}")

print("MODEL_PARAM = ",MODEL_PARAM)

def build_optimizer(optim_name, model, **param):
    if optim_name == "SGD":
        return optim.SGD(model.parameters(), lr=param["learning_rate"], momentum=param["momentum"],
                         weight_decay=param["weight_decay"])
    elif optim_name == "adam":
        return optim.Adam(model.parameters(), lr=param["learning_rate"])

def save_training_state_gpus(accelerator,
                        data_file_name,
                        based,
                        model_name,
                        epoch_i,
                        loss,
                        model_state,
                        save_all,
                        is_best_loss,
                        prev_model_file_path):
    model_param_name = MODEL_NAME.format(
        data_file_name,
        based,
        model_name,
        epoch_i,
        loss
    )

    model_param_path = join(MODEL_PARAM, model_param_name)
    print("model_param_path = ",model_param_path)

    if accelerator.is_main_process:
        if save_all:
            accelerator.save(model_state, model_param_path)
            return None
        elif is_best_loss:
            if prev_model_file_path is not None:
                try:
                    if (epoch_i+1) % 10 != 0:
                        remove(prev_model_file_path)
                except FileNotFoundError:
                    accelerator.print("skip remove file")

            accelerator.save(model_state, model_param_path)
            return model_param_path
        else:
            return prev_model_file_path