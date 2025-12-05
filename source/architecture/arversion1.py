numlayer_enc = 6
numlayer_dec = 6
d_model  = 512
d_ff = 2048
num_of_heads = 8
max_len = 512
vocab_size = 32000

embedding_dropout = 0.3
encoder_dropout = [0. + i * 0 for i in range(numlayer_enc)]
decoder_dropout = [0. + i * 0 for i in range(numlayer_dec)]
output_projection_bias = True
encoder_bias = [True for i in range(numlayer_enc)]
decoder_bias = [True for i in range(numlayer_dec)]