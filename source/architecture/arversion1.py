numlayer_enc = 1
numlayer_dec = 1
d_model = 640
d_ff = 2560
num_of_heads = 8
max_len = 256 
vocab_size = 40000

embedding_dropout = 0.0
encoder_dropout = [0.1 + i * 0 for i in range(numlayer_enc)]
decoder_dropout = [0.1 + i * 0 for i in range(numlayer_dec)]
output_projection_bias = True
encoder_bias = [True for i in range(numlayer_enc)]
decoder_bias = [True for i in range(numlayer_dec)]