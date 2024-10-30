# CorrectionLM: Self-Corrections with SLM for Dialogue State Tracking
This is the original implementation of "[CorrectionLM: Self-Corrections with SLM for Dialogue State Tracking](https://arxiv.org/abs/2410.18209)" by [Chia-Hsuan Lee](https://chiahsuanlee.github.io/), [Hao Cheng](https://sites.google.com/site/hcheng2site) and [Mari Ostendorf](https://people.ece.uw.edu/ostendorf/).

<p align="center">
  <img src="correctionlm.pdf" width="60%" height="60%">
</p>

The task is to track user intents predefined by a schema (ontology) in a multi-turn conversation with an agent. 
CorrectionLM is a novel correction framework that enables Small Language Models (e.g. Llama3-8B) to self-correct using in-context exemplars without LLM (e.g. GPT-4o) involvement. 

[**Installation**](#Installation) | [**Preprocess**](#Download-and-Preprocess-Data) | [**Training**](#Training) | [**Inference**](#Inference) | | [**Evaluation**](#Evaluation) | | [**Citation**](#Citation-and-Contact)

## Installation #TODO

Create a conda environment
```console
conda env create -f env.yml 
```

## Download and Preprocess Data

To download and create the [MultiWoz 2.4](https://github.com/smartyfh/MultiWOZ2.4/) 
```console
cd data
python create_data.py --main_dir mw24_src --mwz_ver 2.4 --target_path mw24
python sample.py --input_fn mw24/train_dials.json --target_fn mw21_5p_train.json --ratio 0.05 --seed 1
```


## Training
The first step of the training is to obtain SLM predictions using ICL in order to provide supervision signals for the correction training.
```console
python run_mwoz_ICL_5shot.py \
      --output_dir expts/llama3_on_train5p_zeroshot/  \
      --lm meta-llama/Meta-Llama-3-8B-Instruct \
      --test_fn data/mw21_5p_train.json \
      --mwz_ver 2.4
```
You can also use GPT-4o for comparisons
```console
python run_mwoz_ICL_5shot.py \
      --output_dir expts/gpt4o_on_train5p_zeroshot/  \
      --lm gpt4 \
      --test_fn data/mw21_5p_train.json \
      --mwz_ver 2.4
```

Then we create the in-context exemplars to finetune the SLM. Unlike traditional ICL methods that only consider the input and gold output, we also incoporate the model’s (erroneous) self predictions.
```console
python create_mwoz_llama_SFT_prompt.py \
      --train_fn expts/llama3_on_train5p_zeroshot/running_log.json \
      --retriever_dir retriever/expts/mw21_5p/ \
      --output_fn data/llama3_on_train5p_zeroshot_ICL_prompt.json  \
      --test_fn expts/llama3_on_train5p_zeroshot/running_log.json \
      --mwz_ver 2.4
```

The second step is to train the SLM. In order to be computation-efficient, we aadopt [QLoRA](https://arxiv.org/abs/2305.14314) for training, i.e. we quantize the SLM then insert LoRA adapaters.
```console
sh train.sh
```


## Inference
The first step of the inference is to get initial predictions by a non-finetuned SLM.
```console
python run_ICL_vanilla.py \
      --train_fn data/mw21_5p_train.json \
      --retriever_dir retriever/expts/mw21_5p/ \
      --lm meta-llama/Meta-Llama-3-8B-Instruct \
      --output_dir expts/llama3_train_5p_on_test100p/  \
      --test_fn data/mw24_100p_test.json \
      --mwz_ver 2.4 
```

We then prompt the correction-tuned SLM (correction SLM) to refine the initial predictions made in the first step.
```console
python run_correctionlm.py \
    --train_fn expts/llama3_on_train5p_zeroshot/running_log.json \
    --retriever_dir retriever/expts/mw21_5p/ \
    --output_dir expts/correction_outputs/llama_example_llama_inference_train5p_test100p/  \
    --test_fn expts/llama3_train_5p_on_test100p/running_log.json \
    --mwz_ver 2.4 \
    --model expts/correction_models/sft_llama3_on_train5p_zeroshot/
done
```

## Evaluation
Compute the JGA and F1 for both dialogue level (DST) and turn level (TLB). 
```console
python eval_result.py \
      --eval_fn expts/correction_outputs/llama_example_llama_inference_train5p_test100p/running_log.json \
      --eval_mode second_pass # first_pass to score on the results produced by non-finetuned LM
```


## Citation and Contact

If you find our code or paper useful, please cite the paper:
```bib
```

Please contact Chia-Hsuan Lee (chiahsuan.li[at]gmail.com) for questions and suggestions.