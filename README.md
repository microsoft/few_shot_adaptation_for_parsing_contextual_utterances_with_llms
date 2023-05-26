## Introduction
This repository contains code for the paper Few-Shot Adaptation for Parsing Contextual Utterances with LLMs.

## Data
The data is under the directory: `release_data`. The non-contextual utterance files with `*-find_event.*.jsonl` are the data used to train and evaluate the model before finetuning with contextual utterances. For contextual utterances, the files that are used are `*-find_event_revise.*.proportional_split.jsonl` for all paradigms in the paper except for the Parse-Then-Resolve paradigm which use `*-find_event_revise.*.proportional_split.edit_fragment_plan.resplit.jsonl`. Each line in the train/validation files contains one example.

## Environment Setup
All experiments are run with a modified version of the codebase from the BenchCLAMP codebase under `semantic_parsing_with_constrained_lm/`. To set up the environement, follow the instructions under `semantic_parsing_with_constrained_lm/README.md`.

## Grammar Files
To generate the grammar files, first convert the files into the original SMCalFlow format by running `scripts/convert_to_smcalflow_format.py`. An example of the arguments to apss int are in `scripts/convert_to_smcalflow_format.sh`.

With the original SMCalFlow formatted files, we can now generate the grammar files for the contextual utterances. To generate `python src/semantic_parsing_with_constrained_lm/domains/lispress_v2/create_benchclamp_data.py` in the `semantic_parsing_with_constrained_lm` directory.

## Fine-Tuning Experiments
For the finetuning experiments, we first train the LLMs on the non-contextual utterances. The `exp-name-pattern`
 argument controls the model, data, input format, and learning rate for a training run. As an example, to produc
e the base model trained on the non-contextual utterances, run the following command:
```
python -m semantic_parsing_with_constrained_lm.finetune.lm_finetune \
--config-name semantic_parsing_with_constrained_lm.configs.benchclamp_config \
--exp-name-pattern 't5-base-lm-adapt_calflowfindevent_no_context_all_0.0001'

```

Then, to evaluate the model run, `model-loc` passing in the model from the previous step.
```
python -m semantic_parsing_with_constrained_lm.run_exp \
        --config-name semantic_parsing_with_constrained_lm.configs.benchclamp_config \
        --exp-names 't5-base-lm-adapt_calflowfindevent_no_context_all_0.0001_10000_dev_eval' \
        --model-loc '/trained_models/1.0/t5-base-lm-adapt_calflowfindevent_no_context_all_0.0001/checkpoint-1000
0'
```

To finetune the model on the contextual utterances, change the `exp-name-pattern` to vary what data to train on,
 and what context is presented to the model. For example, to finetune the model with the `Parse-With-Reference-P
rogram` paradigm on the contextual utterances, run the following command:
```
python -m semantic_parsing_with_constrained_lm.finetune.lm_finetune \
--config-name semantic_parsing_with_constrained_lm.configs.benchclamp_config \
--exp-name-pattern 't5-base-lm-findevent_calflowfindeventrevise_last_plan_low_0_0.0001'
```
The reproduce all other paradigms in the fine-tuning section of Table 1, change the `last_plan` to `last_utteran
ce` for the `Parse-With-Last-Utterance-History`, and `rewritten_utterance` for the `Rewrite-Then-Parse` approarc
h. For the `Parse-Then-Resolve` paradigm, change the data from `calflowfindeventreviseeditfragment`. Evaluation
is done in the same way as the non-contextual utterances.

## In-context Learning Experiments
To run the in-context learning experiments in Table 1, first set the environemtal variables by `OPENAI_GPT3_ENGI
NE` to the model name in the OpenAI API. Our experiments use `text-davinci-003`. Also set `SM_OPENAI_API_KEY` to
 the OpenAI API key. Note: SM_OPENAI_KEY is an internal engine. Then, run the `semantic_parsing_with_constrained
_lm.run_exp` with the the GPT3 config. For example, to run the `Parse-With-Reference-Program` paradigm, run the
following command:
```
python -m semantic_parsing_with_constrained_lm.run_exp \
    --config-name semantic_parsing_with_constrained_lm.configs.benchclamp_gpt3_config \
    --exp-name-pattern 'text-davinci-003_calflowfindeventrevise_last_plan_low_0_2_dev_eval_constrained_bs_5'
```
## Binary Classifier
To get the results for Table 2, we additionally need to train a binary classifer for deciding when to run the co
ntextual parsing model vs non-contextual parsing model. To train finetune the model used in the paper,  run `scr
ipts/finetune_revision_classifier.py` finetunes`roberta-base` on examples a balanced set of examples from the `r
elease_data` contextual and non-contextual utterance.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
