
"""
LLMs suffer from being overly dependent on exact wording

It seems to me that training LLMs on next-token prediction loss offers too sparse a gradient: 
take the training sample "Tracy had a bright green sweater".
During training, an LLM may predict a number of colours instead of green, say, blue.
To us humans, we don't really care the exact colour it chooses, a LLM that suggest red is as good as one that suggest blue or green.
The perplexity loss, however, treats the model's answer blue' as if it were as incorrect as if it had predicted that 1 + 1 = 3.
I posit that what matters is general concepts, rather than exact words - if we train the LLM to instead spit out rough embeddings
of what *concept* would be predicted next, then if it had answered 'blue' instead of 'green', the loss would identify that this is in the same ballpark as green - the soft mbedding space would
allow it to identify that although it is not the exact answer we are looking for, it is nearby the area corresponding to colours, including 'blue', and the weights could be updated in a much more direct and informative way.

I posit that LLMs already do this to some extent when we tie the weights - a currently popular method to speed up LLM training. Doing this forces the LLM's last hidden layer to predict the word/token embedding corresponding
to the correct word. This is a much denser signal, and may be permitting much more informative gradients than from softmax.
However, this is not taken to the extent I am suggesting - because embedding are given at the sub-word level, a single subword may be part of many different wrods which may not be semantically related. This means that the different sementics meaning must be shared across this word, so when it appear as a token, activations are taken up for irrelevant information - this is fine, sparsity is great, but this issue is the elvel of sparsity is not learnable - a subword level token shared with 10 words may be forced to have one of its semantic meanings be 10 times as sparse as a token that doesnt have competing neighbours.
learnable = better.
(except int the case of learnable positional embeddings, which which case rotary/alibi > learnt. actually that not in teh case of genrealisation - NoPE extrapolates better but RoPE has that interpolation ability but thats a really specific case).

The initial idea to make LLMs invariant to exact tokens, was to transform the dataset into embedding space - create a text autoencoder ( encoder is an encoder that produces an embedding every N tokens, the decoder is a causal decoder only LLM. This way embedding can contain future information if it is relevant. the final LLM is no trained with these as input dw, it remains causal) and train the LLM take in standard tokenized tokens and predict the next embedding.
(side note: Its a bad idea to autoregressively predict latents and take those as input, you will likely make a prediction with out-of-distribution activation and when taht enters your context you will only get much worse (current LLMs already struggle to predict the next token and fail with OOD tokens which are obviously way more in distribution than weird self-generated activations), so I would have adapted this to take in actual tokens, predict the next concept-level token, feed this to the decoder of the text autoencoder, then put its predicted next few subword/char level tokens back into context. Also obviously we cant train on embedding from the encoder in teh autoencoder, they are not causal.)
However, I realised that:

training:
1) encoder -> embeddings -> decoder -> characters
2) decoder -> embeddings^

is the same as training:
1) decoder -> embeddings -> decoder -> characters

(there is actually the difference that, in the first, the autoencoder encoder is non-causal, so it may have embeddings which each ahve future info, so it trains the decoder LLM to look further haed. ofc, its not guaranteed the encoder even does, it wil if it helps improve the mebdding quality so the decoder can estimate the tokens, but that depends on, e.g.,  info in tokens 8+ helping tokens 7-.).
As another way to encourage longer span sequences from each embedding the LLM, we could train the decoder to output longer sequences.


The intuition still remains that bottlenecking the information between the LLM and the tokens should result in the LLM having to output higher level concepts and gradients accumulating across the sparse subword/char/word-level actual output into dense representations for the LLM - the gradients entering into the LLM should be dense

The LLM in disincentivised from being optmized for token-level: every output it gives must help the char/subword-decoder to maximise its logpobs for many characters in a row - its output representations MUST be dense and invariant to exact words so that all the words are represented.

We probably want multiple stages of heirarchy - a character level one, and a word level one so that exact, various characters can be merged into words and multiple words can be merged into concepts without the two very separate abstration levels taking bandwidth from the other

We probably want to nerf the char/word/subword-level decoder so it cant just do language modelling and must depend on the concepts from the LLM


There is a side benefit to applying this kind of heirarchy not just on the output side, but on the input side - it fixes all the weird issues caused by tokenization, probably allowing much faster learning of specifics like numeracy, word/character counting, ASCII art and other figures, etymology stuff that you could get from looking at the character components of words but which the tokenizer cant break down cleanly



skip transformer decoder -> every N embedding -> LLM -> every token conditions text with summation as before

differenr, rough idea - different brain sections are specialised neurons with specialised loss functions.
brain works by linking them to appropriate data sources so they becaome useful

here we could try: at lower level (near input) create compression loss NN (maximise information relayed per neuron), at higher level (working from the sparse/compressed neural representations) do sequence prediction
# TODO: concept level input will also be good: we can see that non-fixed input embeddings are fine in transformers e.g. ViTs.
# + The reason char-level models are bad is that you cant make a useful emebdding for a single character which allows for finding other good tokens
#  --> make a autoencoder which turn char level into concept level, do next concept prediction, at inference decode those into chars



 --- FUTURE: it would be good to separate concepts by something mroe emaningful than every N characters. e.g., break it down into discrete, sequential packets of info, with a single chunk of contiguous low perplexity tokens (and the single preceeding high perplexity token) being a chunk of information. information is directly related to perplexity, so this is a good measure.
# --- separate dataset into chunks based on ppx # load pretrained 128M parameter RNN - an RNN will be used as the final decoder, probably best to use smth ith similar behaviour. also necessary for efficient char level language modelling.





!!! KEY: DO NOT MEASURE PERFORMANCE BY PERPLEXITY - THAT IS THE WHOLE POINT: WE ARE TRYING TO MAKE AN ARCHITEURE THAT BETTER LEARNS OVERALL CONCEPTS, NOT EXACT TOKENS.
Evaluation using tokenizer-independent metrics - BLEU, HumanEval, GPT-4 as judge, MMLU, AGIEval, ...
"""

# The main motivation for this is making the output more informative to allow for more useful gradients that can point the majority of parameters more directly towards the global minimum we are looking for (high level concept generation)









# ----- HYPERPARAMETERS
LLM_context_length = 1024
concept_to_token_ratio = 10 #8 words=10.7 tokens # 32 for char-rnn each hidden state should predict about 8 words ahead, as the human brain does: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10038805/. 8 words = 8 * 4.7 characters ~ 38. we'll cut to 32.
seq_length = LLM_context_length*concept_to_token_ratio # we could randomise this every sample, i.e. for each sample, select which embeddings from the LLM get added to the rnn_decoder input, so the deocder is train to decode for a variety of lengths - encourages decoder LLM embedding providing longer range data. 
LLM_layers =   4
LLM_n_heads =  16
LLM_hid =      1024

# --- char rnn parameters
rnn_hid =     512
rnn_n_heads = 4

# --- training
batch_size = 8






# ----- dependencies
!pip install transformers
import torch
from transformers import LlamaModel, LlamaConfig, AutoTokenizer, LlamaForCausalLM, RwkvConfig, RwkvModel
from huggingface_hub import login
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
access_token_read = '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'
login(token = access_token_read)






# --- make dataset
file = '/content/drive/MyDrive/PythonQAStrings.txt'
from torch.utils.data import Dataset, DataLoader
class TextDataset(Dataset):
    def __init__(self, filepath):
        self.texts = []
        with open(filepath, 'r') as f:
            f=f.read()
            self.texts = f.split('<|endoftext|>')
        print('num documents=',len(self.texts))
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = tokenizer.encode_plus(text, max_length=seq_length, truncation=True, padding='max_length', return_tensors='pt')
        tokens.input_ids = tokens.input_ids.squeeze(0)
        return tokens.to(device)
dataset = TextDataset(file)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)





# --- subword tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# --- char tokenizer:
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode_char(text, max_length, pad_token):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    encoded_text = [stoi.get(c, stoi[pad_token]) for c in text]
    encoded_text = encoded_text[:max_length] + [stoi[pad_token]] * max(0, max_length - len(encoded_text))
    return {
        'input_ids': encoded_text,
        'attention_mask': [1] * len(encoded_text)
    }
def decode_char(tokens):
    itos = {i: ch for ch, i in stoi.items()}
    decoded_text = ''.join([itos.get(i, '') for i in tokens['input_ids']])
    return decoded_text.strip()






# --- initialize output char-RNN
RWKV_config = RwkvConfig()
RWKV_config.tie_word_embeddings = True # irrelevant for the char-to-concept decoder since it does output softmax. relevant for concept-to-char model.
RWKV_config.is_decoder = True
RWKV_config.intermediate_size = rnn_hid*4
RWKV_config.hidden_size = rnn_hid
RWKV_config.attention_hidden_size = rnn_hid//rnn_n_heads
RWKV_config.context_length = LLM_context_length*concept_to_token_ratio # max tokens per text sample * average num chars per token
RWKV_config.num_hidden_layers = 2 # TinyStories showed you can get decent grammar performance with this size with good data. well see how good ours is I guess
RWKV_config.output_hidden_states = True

char_rnn_dec = RwkvModel(RWKV_config).to(torch.bfloat16).to(device)
char_rnn_dec.eval()





# --- Create non-causal concept encoder (C) (tokens to concepts)
# the final autorecressive decoder-only model will use itself as the encoder - note that for char-level or removing tokenizer dependency we will need a separate encoder so that decoder take in only high level concepts
from transformers import LlamaModel, LlamaConfig, AutoTokenizer, LlamaForCausalLM
half_config = LlamaConfig(
    vocab_size=32000,
    hidden_size=LLM_hid,
    intermediate_size=LLM_hid*4,
    num_hidden_layers=LLM_layers,
    num_attention_heads=LLM_n_heads,
    num_key_value_heads=None,
    hidden_act='silu',
    max_position_embeddings=seq_length,
    initializer_range=0.02,
    rms_norm_eps=1e-06,
    use_cache=True,
    pad_token_id=None,
    bos_token_id=1,
    eos_token_id=2,
    pretraining_tp=1,
    tie_word_embeddings=True,
    rope_theta=10000.0,
    rope_scaling=None)

# --- Create causal concept decoder (D) (concepts to concepts)
LLM_decoder = torch.compile(LlamaForCausalLM(half_config).to(torch.bfloat16).cuda())
LLM_decoder.eval()




# --- Define the optimizer and loss function
optimizer = optim.AdamW( list(LLM_decoder.parameters())+list(char_rnn_dec.parameters()), lr=1e-4, betas=(0.9,0.95) )
import math
warmup_iters = 0 # pre-ln transformers dont need warmup # # learning rate decay scheduler (cosine with warmup) from karpathy's nano-gpt
lr_decay_iters = 1000
learning_rate = 1e-4
min_lr = 1e-5
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)




losses = []
# Iterate over the batches
for b, batch in enumerate(dataloader):
    batch.input_ids = batch.input_ids.squeeze(1)
    labels = batch.input_ids

    # update learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = get_lr(b)


    # FOR CHAR RNN INPUT:
    # Step 1: compress input chars with char_rnn_enc
    #LLM_input_embeds = char_rnn_enc(batch.input_ids, output_hidden_states=True).hidden_states[-1][:,::concept_to_token_ratio] # get all samples, get every Nth token
    
    # Step 2: LLM takes in (compressed?) tokens, predicts next concept
    LLM_hidden_states = LLM_decoder(input_ids=batch.input_ids, output_hidden_states=True).hidden_states[-1]

    # Step 3.1: pass char tokens to char_rnn_dec
    char_rnn_dec_input_embeds = char_rnn_dec.embeddings(batch.input_ids)

    # Step 3.2: spread the LLM output latents to fit char_rnn_dec, add them to char_rnn_dec input embeds
    num_tokens_to_add = char_rnn_dec_input_embeds.size(1) - LLM_hidden_states.size(1)
    if num_tokens_to_add > 0:
        LLM_hidden_states = torch.cat([LLM_hidden_states, torch.zeros_like(LLM_hidden_states)[:, :num_tokens_to_add, :]], dim=1)
    elif num_tokens_to_add < 0:
        LLM_hidden_states = LLM_hidden_states[:, :char_rnn_dec_input_embeds.size(1), :]
    char_rnn_dec_input_embeds += LLM_hidden_states


    # Step 4: pass char_rnn_dec input embeds to char_rnn_dec
    # at this point, every token rnn_dec sees is either: a multiple of N and has the LLMs output added to it (this includes the first token. this allows future tokens until the next embeding adddition to condition on just that latent before the next)
    outputs = char_rnn_dec(inputs_embeds=char_rnn_dec_input_embeds.to(torch.bfloat16).cuda(), labels=labels.cuda())


    # Step 5: Train D on the modified inputs and token labels
    outputs.loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    print(b, outputs.loss.item(), 'lr',get_lr(b))
    losses.append(outputs.loss.item())

    # GET D-DECODER GENERATION SAMPLES (not optimized w/ cache):
    if b%200==0:

        LLM_decoder.eval()
        char_rnn_dec.eval()
        generated_tokens = [1]
        with torch.no_grad():

            # give <s> token to char_rnn_dec to start decoding
            charnnd_prompt = char_rnn_dec.model.embed_tokens(torch.tensor([generated_tokens]).to(device)).to(device).reshape([1,1,labels.shape[-1]])

            # Each iteration, the LLM takes in the previous chunk of tokens (or <s>), and predicts the next concept - this is summed with the last token of the generated text and given to the char rnn decoder input. the char rnn decoder takes in this concept + previuous generation (and previous concepts added in) and outputs the next chunk of tokens.
            for c in range(64//concept_to_token_ratio):

                # FOR CHAR RNN INPUT:
                # LLM_input_embeds = char_rnn_enc(batch.input_ids, output_hidden_states=True).hidden_states[-1][:,::concept_to_token_ratio] # get all samples, get every Nth token

                # get LLM_concept to predict concepts from token inputs
                concept = LLM_decoder(input_ids=torch.tensor([generated_tokens]).to(device), output_hidden_states=True).hidden_states[-1][0,-1,:]  # get last token output of last hidden layer

                # from predicted concept, predict next tok_per_enc_out tokens with D_decoder
                charnnd_prompt[0,-1,:] += concept # add predicted concept to D_decoder's first input token for this chun - this way all its predicted token are onditioned on teh predicted concept
                
                # now that a concept has been predicted by the LLM, and it is added to the char rnn decoder prompt, we get teh char rnn decoder to autoregressivly predict the nxt N tokens (it conditions from teh new predicted concept from the LLM)
                for i in range(concept_to_token_ratio): # given starting token, predict the remaining tokens in this chunk, then predict the next one - this is what the Concept LLM predicts from and its predicted concept will be added to it
                    probs_ = char_rnn_dec(inputs_embeds=charnnd_prompt).logits[0,-1]
                    predicted_token = torch.argmax(probs_, dim=-1).reshape([1,1]).to(device)
                    # feed predicted tokens back into self
                    generated_tokens.append(predicted_token)
                    D_prompt = torch.cat([D_prompt, char_rnn_dec.model.embed_tokens(predicted_token)], dim=1)

            generated_text = tokenizer.decode(torch.tensor(generated_tokens))
            print('generated: #######',generated_text,'#######')

    torch.cuda.empty_cache()

    if b == 5000:
        break



print(losses)

torch.save(LLM_decoder.state_dict(),'/content/drive/MyDrive/LLM_decoder')
torch.save(char_rnn_dec.state_dict(),'/content/drive/MyDrive/char_rnn_dec')

import matplotlib.pyplot as plt
plt.plot(losses)