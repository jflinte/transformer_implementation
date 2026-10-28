# Transformer Implementation

Ideas:
- One simple way to have structured data could be in the case of a board game, such as chess or checkers
- Positional embedding might be interesting (in the sense that state of board is temporal, but no position on board has precedent over another)
- Has verifiable reward (win or loose / correct or incorrect move / etc)

resources:
- [Attention is All You Need](https://arxiv.org/pdf/1706.03762)
- [Transformers with Structured Data](https://www.next.gr/ai/image-classification/using-transformers-with-structured-data)
- [transformer walkthough](https://jalammar.github.io/illustrated-transformer/)
- [Lukasz Kaiser's explanation](https://www.youtube.com/watch?v=rBCqOTEfxvg)
- [Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/#background)
- [Brandon's Transformers from Scratch](https://brandonrohrer.com/transformers.html)
- [Peter's Transformers from Scratch](https://peterbloem.nl/blog/transformers)
- Datasets
    - [Nerual Crystal Checkers](https://huggingface.co/neuralcrystal#:~:text=neuralcrystal%2Fcheckers%2Dpositions.%20Viewer%20%E2%80%A2%20Updated%20about%2016%20hours,17%20hours%20ago%20%E2%80%A2%202.5M%20%E2%80%A2%2028.)
    - KingsRow 8-Piece & 10-Piece WLD Tablebases
    - Chinook Endgame Database
    - The American Checker Federation (ACF) Match Archive
    - Kaggle Checkers Gameplay Dataset
    - Martin Fierz / CheckerBoard Game Archive


Transformer:
- follows encoder / decoder architecture (encoder maps input sequence to continuous representation, decoder auto-regressively (i.e. uses previously generated symbols as input) generates output)
- uses only attention, rather than recursive or convolutional layers
- encoder sublayers: multi-head self attention -> fully connected feed-forward network -> layer normalization + residual connection
- encoder sublayers: (causal/masked) multi-head self attention -> fully connected feed-forward network -> layer normalization + residual connection


Attention
- mapping query, set of key-value pairs to output (everything is a vector) 
- output is weighted sum of values, weight computed by compatibility function of query with corresponding key
- scaled dot product attention
    - inputs: queries and keys (of dimension d_k) and values (d_v)
    - compute dot products of all queries and keys, then divide by square root of dimension, d_k
    - apply softmax function, to obtain weights on the values
    - this is very efficient, as it is simplyl performing matrix multiplications
    - scaling keeps dot products in reasonable magnitude
- multi-head attention
    - linearly project queries, keys, and values h times with learned weights
    - each projection has scaled dot product attention applied to it
    - the output values are concatonated and then linearly projected
    - This lets model view information from different perspectives (i.e. different representation subspaces at different positions)
- uses of attention
    - encoder-decoder: queries = previous decoder layer, keys&values = output of encoder (every position in decoder can attend every position in input sequence)
    - encoder: self attention (keys, values, queries come from same place - output of previous layer in encoder, every psoition in encoder can attent to all position in the previous layer)
    - decoder: self attention in decoder lets every position in decoder attend to every position up to and including that position. Leftward information flow (noncausal?) is prevented via masking in softmax
- position wise feed forward networks
    - each layer in encoder/decoder have fully connected feed-forward network (2 linear transformation w/ ReLU activation function inbetween)
    - Stay same horizontally (across sequence)
    - change vertically (deeper into layers)
- positional encoding
    - because no recurrence (makes sense) or convolution (why does this give order? causal masking?), need way to store relative/absolute position of tokens in sequence
    - positional encodings they use: sin and cos
    - learned positional embeddings performed nearly identically, but using fixed encoding allows model to operate with sequence lengths longer than ones encountered during training
    - the vector is of size d_model, a different sinusoid for each vector entry from [0, d_model-1] is added to final embedding
    - with this construction, the positional encoding ensures that relative position, rather than absolute position, is important. 
        - $$\vec{v}_p \cdot \vec{v}_{p+k} = \sin(p)\sin(p+k) + \cos(p)\cos(p+k)$$
        - $$\sin(p)\sin(p+k) + \cos(p)\cos(p+k) = \cos\big((p+k) - p\big) = \cos(k)$$
- Reasons for using Attention
    - total computational complexity per layer
    - parallelization of computation
    - path length between long-range dependencies in network

vggt uses vision transformer ViT: https://arxiv.org/html/2010.11929v2
- reshape image into series of flattened 2D patches 
- patches are flattened and mapped to Dimension D with trainable linear projection
- og utilizes 1d positional embeddings
- encoder = alternating layers multi-head attention, MLP blocks, layer normalization, residual connections (MLP contains GELU activation function)


from Transformer interview with Kaiser:
attention
- make a query with a vector, then look for similar things in the past
- attention looks at everything, looks for things that are similar

transformer
- you have some inputs (say an English sentence)
- attention -> feed-forward
- recurrent neural network, very obvious that you go from one word to the next
- to produce output, you want to attend to only things before, if RNN you have to look at every step, but in transformers, you can do it in parallel
- SoftMax gives you probability distribution of keys that are similar to queries
- there are multiple attention heads looking at different words, these heads look at different things
- multimodal multitasking: process a modality and compress it into something that the models can interpret
- mixture of experts (100 matrices, maybe 1000, and you have a learnable gate that selects only maybe 4), these give you a lot of parameters, but still speed (I think)



gemma4 (https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4)
- interweave with global attention

Transformer architecture
- input (AxB matrix)
- input embedding (AxC matrix)
	- the onehot encoding basically just selects one of the weights
	- AxB x BxC = AxC where A is the number of input tokens, B is the vocabulary size, and C is the embedding dimension (d_model)
- positional encoding (AxC matrix)
	- Each token is a vector of length B (d_model). For each of these dimensions, there are separate sinusoids created, each with their own unique frequency, chosen based on their index position
	- each position is a point on all of these sinusoids
	- Because the frequencies very so drastically, early positions will have less changer than later ones, just like in a binary clock
	- Because all of the weights are learned and because they are much larger than the positional embeddings, certain positions might encode semantic meaning while others encode positional meaning. Additionally, it can create a sort of star orbit situation, where slight variations leave no doubt as to the original semantic meaning
- multi-head attention
	- dividing by square root of d_k
        - Each term has a variance of 1, so adding dk of them gives a variance of d_k or a standard deviation of square root of d_k. By dividing by this number, you give them a standard deviation of 1
        - It also prevents vanishing gradients
- encoder
    - Input (Ax1) matrix (i.e. input tokens)
    - Input embedding turns into (Ax512) 
    - Projection matrices (weights) are (512x64)
    - (Ax512)(512x64) = (Ax64)
    - Attention score is Q * K transpose or (Ax64)(64xA) = (AxA)
    - Attention output is A_weights * V or (AxA)(Ax64) = (Ax64)
    - 8 attention heads, so now (Ax512)
    - feedforward layers or (Ax512)*(512x2048)=(Ax2048)
- decoder