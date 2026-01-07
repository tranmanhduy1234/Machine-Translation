numlayer_enc = 6
numlayer_dec = 6
d_model = 640
d_ff = 2560
num_of_heads = 8
max_len = 256 
vocab_size = 40000

embedding_dropout = 0.0
encoder_dropout = [0.1 + i * 0.001 for i in range(numlayer_enc)]
decoder_dropout = [0.1 + i * 0.001 for i in range(numlayer_dec)]
output_projection_bias = False
encoder_bias = [False, False, False, False, False, False]
decoder_bias = [False, False, False, False, False, False]