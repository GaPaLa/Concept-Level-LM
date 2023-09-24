# Concept Level Lanauge Modelling
 We want LLMs to be invariant to exact phrasing for faster learning of the higher level concepts we are more interested in.

# Motivation
In language modelling, say we take the prompt "the cat is" with the correct next token being "white"; the next token prediction loss (softmax) will treat a model confidently predicting "black" and a model confidently predicting "frog" as equally incorrect. In my opinion, this approach hinders the learning process significantly - the loss should produce a rich gradient pointing the model to the correct area of outputs - a model saying frog should be strongly correctd in the direction of cat colours, while a model outputting 'black' should hardly be corrected. In other words, from my understanding, it would speed up training to have a loss function that evaluates how closely the model's output aligns with the true answer in a semantic embedding space, considering the underlying meaning of the words, not just their exact wording.

From my understanding, this is the reason why tying the embedding and output projection weights of a language model is so effective. However, this weight tying these layers only addresses this issue at the token level, while the problem extends to longer sequences as well. For instance, if the prompt is "tell me something funny," and the correct answer is "why did the chicken cross the road?" but the model predicts "why was the skeleton sad at the dance?", then, even with weight-tying, the token-level prediction loss will only be useful up to the point where the words "skeleton"/"chicken". While these words are distinctly different, the semantics of the total phrases share a common intent and conceptual similarity (both are classic jokes). Nevertheless, the model that outputs "skeleton" will be penalized as severely as if it had generated something completely unrelated. This is because token-level word embeddings do not capture the sentence-level semantics that the model intended.

(Although the example above example isn't a good representation of token-level teacher forcing, this is partially because that is what I am criticizing. I feel that it gets across the idea I intend though: for a prompt requesting a joke, a model saying the skeleto joke should not face a significant weight adjustment compared to a model saying the chicken joke).



# attention is very useful for extracting important information from earlier but the brain is much more like recurrent neural network. someone on twitter mentioned that RNN language models get higher human-brain-similarity scores than transformers [i dont have a source  for this - will have to look up]

# we also know recurrent neural networks have strong inductive baises that allow solving problems that transfomers cant (see deepmind chomsky heirarchy / tape-RNN paper)

# maybe the heirarchy can reflect this - the model is majority RNN but has a self attention layer near the highest level to get distance relevant past hidden states - we can probably inject retrieval here too.