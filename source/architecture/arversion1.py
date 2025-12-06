numlayer_enc = 8
numlayer_dec = 8
d_model  = 640
d_ff = 4096
num_of_heads = 8
max_len = 2048
vocab_size = 40000

embedding_dropout = 0.3
encoder_dropout = [0. + i * 0 for i in range(numlayer_enc)]
decoder_dropout = [0. + i * 0 for i in range(numlayer_dec)]
output_projection_bias = True
encoder_bias = [True for i in range(numlayer_enc)]
decoder_bias = [True for i in range(numlayer_dec)]