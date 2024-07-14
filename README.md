
# Build and Train Mistral 7B Model from Scratch

This project aims to provide a simple framework for training and inference of language models from scratch, with a specific focus on the **Mistral 7B** model tailored for **Algerian Darija 🇩🇿**. The repository includes tools for **model definition**, **training**, and **inference** .

The project Details can be found in the [This blog post]() .

## Running the train Script : 

To run the train script, you can use the command line to specify the training parameters.

#### Command-line Arguments

- `--epochs`: Number of epochs to train the model (default: 10)
- `--train_data`: Path to the training data (default: set in settings.py)
- `--eval_data`: Path to the evaluation data (default: set in settings.py)

#### Example : 

```bash
python train.py --epochs 100 --train_data src/train_eval_data/train.txt --eval_data src/train_eval_data/eval.txt
```

## Running the Inference Script : 

This will generate and print multiple sequences of text based on the provided prompt.

#### Command-line Arguments

- `--prompt`: Input prompt text in Algerian Darija .
- `--model_path`: Path to the saved model file .
- `--max_length`: Maximum length of generated text (default: 30) .
- `--num_return_sequences`: Number of sequences to generate (default: 5) .

#### Example : 
```bash
python inference.py --prompt "وحد نهار" --model_path "model_final.pt" --max_length 30 --num_return_sequences 5
```
