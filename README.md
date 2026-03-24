# Transformer Activity Pack

**Transformer Architectures Activity: From BERT to GPT to BART**

## Purpose

This activity is designed so that you start from a complete encoder-only model, inspect a complete encoder-decoder model, and then construct the decoder-only model by yourselves. The intention is not only to code, but to explain what architectural change turns one transformer family into another. All explanations must refer to the outputs produced by your own implementation.

## Files in the Pack

The pack is organized as Python fragments. The fragments intentionally do not contain import statements. They are meant to be loaded from a main Python file or notebook cell that already imported the shared dependencies.

| File                               | Purpose                                                                                                |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `00_setup.py`                    | Setup, toy corpus, vocabulary, train/validation split. Hyperparameters are intentionally left blank.   |
| `01_batching.py`                 | Batch helpers for the three tasks: language modeling, classification, and encoder-decoder training.    |
| `02_core_modules.py`             | Full FeedForward and AttentionHead. Skeletons for EncoderBlock, DecoderBlock, and EncoderDecoderBlock. |
| `03_models_bert_bart.py`         | Full TinyBERT and TinyBART.                                                                            |
| `04_model_gpt_skeleton.py`       | Skeleton of TinyGPT, including generation methods.                                                     |
| `05_training_utils_and_demos.py` | Full BERT and BART utilities and demos. GPT demo intentionally empty.                                  |
| `main.py`                        | Main driver that coordinates execution of all fragments.                                               |

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the program
python main.py
```

## Suggested Workflow

### [x] Task 0: Import Files and Modules

- [X] The import statements have been removed from all the provided Python files. Identify the necessary modules and make sure they are correctly imported so that the code can run.
- [X] Create a main file that coordinates the execution of the program. This file should import the provided fragments, instantiate the corresponding models, and run the desired demos. You can use `05_training_utils_and_demos.py` as a guide.

---

### [x] Task 1: Choose Hyperparameters

- [X] Fill in the hyperparameters in `00_setup.py`. Start with values that are small enough to train quickly.

**Questions:**

- [X] Which hyperparameters change model capacity, and which mainly change training behavior?
  the hyperparameters that change model capacity make up the size of the neural network and its ability(in theory) to learn more complex patterns. These ones include d_model(hidden dimensions), n_layers(number os layers and blocks), n_heads(number of attention heads), d_ff(feedforward network size) and vocab_size.
  the hyperparameters that change the training behaviour make up how the model would update the weights and how fast it can learn and generate unseen data without changing architectural things. these include learning rate, batch size, dropout, weight decay, warmup steps and epochs.
- [X] Why does context length matter differently for BERT-style classification and GPT-style generation?
  BERT has a small text size that it uses, whereas GPT generally uses a bigger context in order to make better assumptions and create better responses. BERT only needs enough context to classify a text, while GPT needs enough context to allow it to reason, in order to come back with a good and somewhat reasonable response.
- [X] What trade-off appears when increasing d_model or the number of layers?
  Both d_model and number of layers are hyperparameters that influence model capacity, as mentioned for the first question, so the model would benefit from increasing one of them as it would learn more complex patterns and lead to better performance. However, the tradeoff is that it would require more computation, time, and VRAM. Moreover, if the dataset is small and the dimensions or number of layers are too big, the model might overfit and generalize poorly. Also, deeper models can lead to vanishng gradients or exploding gradients, which can make training unstable.

---

### [x] Task 2: The Attention Mechanism

- [X] The AttentionHead is fully supplied in `02_core_modules.py`. Since this is the key architectural unit behind all three models, answer the question below.

**Questions:**

- [X] Temporarily modify the attention mechanism so that causal masking is disabled. Run the GPT model later in the activity (Task 6) with and without masking, using the same prompt. Compare the generated outputs and describe two concrete differences.

---

### [ ] Task 3: Complete the Three Block Types

Use the residual pattern to complete:

- [ ] EncoderBlock
- [ ] DecoderBlock
- [ ] EncoderDecoderBlock

**Questions:**

- [ ] Run your GPT model and then temporarily disable causal masking in the DecoderBlock. Generate text in both cases. Based on the outputs, explain which version behaves autoregressively and why.
- [ ] In the encoder-decoder block, where does information from the source sequence enter?
- [ ] Why is cross-attention not needed in BERT or GPT?

---

### [ ] Task 4: Build TinyGPT by Mirroring the Other Models

- [X] Complete TinyGPT in `04_model_gpt_skeleton.py`. You should reuse the architectural patterns that can be found in BERT and BART models.

**Questions:**

- [X] Which precise architectural change makes GPT decoder-only rather than encoder-only?
- [X] Why does GPT use a language-modeling head instead of a classification head?
- [X] Why is next-token prediction compatible with decoder masking but not with bidirectional self-attention?

---

### [ ] Task 5: Implement Different Decoding Strategies

Complete the GPT generation methods:

- [ ] temperature sampling
- [ ] top-k sampling

**Questions:**

- [ ] Generate text from the same prompt using temperature = 0.5, and temperature = 1.5. Include both outputs and describe how the structure, coherence, and variability of the text change. Explain why temperature produces this effect.
- [ ] What practical problem does top-k sampling try to reduce?
- [ ] Try to use the encoder-only (BERT-like) model to generate text in the same way as GPT. What happens in practice? Based on this experiment, explain why the GPT model is suitable for generation and the BERT model is not.

---

### [x] Task 6: Complete the GPT Demo

- [X] The BERT and BART demos are complete. The GPT demo in `05_training_utils_and_demos.py` is empty on purpose. Instantiate the model, train it, and compare generations under different decoding settings.

**Reflection Questions:**

- [X] Which decoding method produced the most coherent output? And the most diverse output?
  From the outputs in this run, top-k sampling produced the most coherent output overall, while temperature sampling produced the most diverse output. beam search was the most conservative and repeated patterns more often, so it was less diverse and in this tiny setup also less natural.
- [X] Did lower validation loss always imply more interesting generations?
  No. lower validation loss generally means better next-token prediction on average, but it does not guarantee that generations are more interesting. in this experiment, outputs could still be repetitive or dull even when validation loss improved.
- [X] What kinds of errors remained even after training?
  The main remaining errors were character repetition loops, broken or partial words, abrupt sentence transitions, and local incoherence after a few tokens. beam search especially showed repetitive continuations, and all methods still produced text that looked statistically plausible but not fully meaningful.

---

## [ ] Discussion

- [ ] After the coding tasks, go along the following questions:

### [ ] Using Your Own Implementations of BERT, GPT, and BART

- [ ] Provide one concrete example of output from each model (or explain why one cannot produce output).
- [ ] Based on these results, explain how differences in masking and attention structure lead to different behaviors. Your answer must refer to the outputs you obtained.

### [ ] Masking Comparison Experiment

- [ ] Run your model once with causal masking enabled and once with it disabled (you may temporarily modify the code). Generate text in both cases using the same prompt.
- [ ] Describe two concrete differences you observe in the generated outputs.
- [ ] Explain why these differences occur.
