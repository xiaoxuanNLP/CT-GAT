import pandas as pd
import re


def load_TCAB(train=True):
    """
        scenario: Domain, either abuse or sentiment.
        target_model_dataset: Dataset being attacked.
        target_model_train_dataset: Dataset the target model trained on.
        target_model: Type of victim model (e.g., bert, roberta, xlnet).
        attack_toolchain: Open-source attack toolchain, either TextAttack or OpenAttack.
        attack_name: Name of the attack method.
        original_text: Original input text.
        original_output: Prediction probabilities of the target model on the original text.
        ground_truth: Encoded label for the original task of the domain dataset. 1 and 0 means toxic and toxic for abuse datasets, respectively. 1 and 0 means positive and negative sentiment for sentiment datasets. If there is a neutral sentiment, then 2, 1, 0 means positive, neutral, and negative sentiment.
        status: Unperturbed example if "clean"; successful adversarial attack if "success".
        perturbed_text: Text after it has been perturbed by an attack.
        perturbed_output: Prediction probabilities of the target model on the perturbed text.
        attack_time: Time taken to execute the attack.
        num_queries: Number of queries performed while attacking.
        frac_words_changed: Fraction of words changed due to an attack.
        test_index: Index of each unique source example (original instance) (LEGACY - necessary for backwards compatibility).
        original_text_identifier: Index of each unique source example (original instance).
        unique_src_instance_identifier: Primary key to uniquely identify to every source instance; comprised of (target_model_dataset, test_index, original_text_identifier).
        pk: Primary key to uniquely identify every attack instance; comprised of (attack_name, attack_toolchain, original_text_identifier, scenario, target_model, target_model_dataset, test_index)
            :return:
        """
    if train == True:
        return pd.read_csv("./train.csv")
    else:
        return pd.read_csv("./val.csv")


def split_TCAB(train=True, scenario=None, target_model_dataset=None, target_model_train_dataset=None, target_model=None,
               status=None):
    pattern_no = re.compile(r'no.')
    df = load_TCAB(train)
    if (scenario == None) and (target_model_dataset == None) and (target_model_train_dataset == None) and (
            target_model == None) and (status == None):
        if train == True:
            df.to_csv("./train_split_" + str(scenario) + "_" + str(target_model_dataset) + "_" +
                      str(target_model_train_dataset) + "_" + str(target_model) + "_" + str(status))
        else:
            df.to_csv("./val_split_" + str(scenario) + "_" + str(target_model_dataset) + "_" +
                      str(target_model_train_dataset) + "_" + str(target_model) + "_" + str(status))
        return
    else:
        if scenario == None:
            df = df
        else:
            if "no." in scenario:
                scenario = re.sub(pattern_no, "", scenario)
                df = df[df['scenario'] != scenario]
                scenario = "no." + scenario
            else:
                df = df[df['scenario'] == scenario]
        if target_model_dataset != None:
            if "no." in target_model_dataset:
                target_model_dataset = re.sub(pattern_no, "", target_model_dataset)
                df = df[df['target_model_dataset'] != target_model_dataset]
                target_model_dataset = "no." + target_model_dataset
            else:
                df = df[df['target_model_dataset'] == target_model_dataset]
        if target_model_train_dataset != None:
            if "no." in target_model_train_dataset:
                target_model_train_dataset = re.sub(pattern_no, "", target_model_train_dataset)
                df = df[df['target_model_train_dataset'] != target_model_train_dataset]
                target_model_train_dataset = "no." + target_model_train_dataset
            else:
                df = df[df['target_model_train_dataset'] == target_model_train_dataset]
        if target_model != None:
            if "no." in target_model:
                target_model = re.sub(pattern_no, "", target_model)
                df = df[df['target_model'] != target_model]
                target_model = "no." + target_model
            else:
                df = df[df['target_model'] == target_model]
        if status != None:
            if "no." in status:
                status = re.sub(pattern_no, "", status)
                df = df[df['status'] != status]
                status = "no." + status
            else:
                df = df[df['status'] == status]
    if train == True:
        df.to_csv("./train_split_" + str(scenario) + "_" + str(target_model_dataset) + "_" +
                  str(target_model_train_dataset) + "_" + str(target_model) + "_" + str(status))
    else:
        df.to_csv("./val_split_" + str(scenario) + "_" + str(target_model_dataset) + "_" +
                  str(target_model_train_dataset) + "_" + str(target_model) + "_" + str(status))


if __name__ == "__main__":
    split_TCAB(train=True)
    split_TCAB(train=False)

