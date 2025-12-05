import sentencepiece as spm

# Tải mô hình .model của bạn
sp = spm.SentencePieceProcessor()
sp.load(r'D:\chuyen_nganh\Machine Translation version2\pre-handle\unigram_32000.model') # Thay bằng đường dẫn file của bạn

# Kiểm tra các ID đặc biệt
print(f"Token <unk>: ID = {sp.unk_id()}")
print(f"Token <s> (BOS): ID = {sp.bos_id()}")
print(f"Token </s> (EOS): ID = {sp.eos_id()}")

# Nếu bạn có định nghĩa token <pad>
if sp.pad_id() != -1: # Mặc định là -1 nếu không được set
    print(f"Token <pad>: ID = {sp.pad_id()}")
else:
    print("Token <pad> không được định nghĩa riêng biệt.")

# Kiểm tra tổng số từ vựng
print(f"Tổng số từ vựng (Vocab size): {sp.get_piece_size()}")

print("\n--- Kết quả Tokenization ---")
s = """
Trong cuộc sống, con người không chỉ cần vật chất mà còn cần tình yêu thương để tồn tại và phát triển. Tình yêu thương là thứ quý giá nhất giúp con người gắn kết với nhau, cùng sẻ chia niềm vui, nỗi buồn và giúp nhau vượt qua những khó khăn. Nếu thế giới này thiếu vắng tình yêu thương, mọi thứ sẽ trở nên vô nghĩa. Chính vì thế, tình yêu thương có một sức mạnh vô cùng to lớn, có thể thay đổi con người và xã hội theo hướng tốt đẹp hơn. Trước hết, tình yêu thương là gì? Đó là sự quan tâm, sẻ chia, giúp đỡ lẫn nhau bằng cả tấm lòng chân thành, không toan tính. Tình yêu thương không chỉ giới hạn trong phạm vi gia đình mà còn lan tỏa trong xã hội. Đó có thể là tình cảm giữa cha mẹ và con cái, giữa anh chị em trong gia đình, giữa thầy cô và học trò, giữa bạn bè hay thậm chí là giữa những người xa lạ với nhau.
Tình yêu thương không chỉ mang lại niềm hạnh phúc cho người nhận mà còn làm giàu đẹp tâm hồn của người trao đi. Sức mạnh của tình yêu thương thể hiện rõ nhất trong những lúc con người gặp khó khăn, hoạn nạn. Khi ai đó gặp biến cố, tình yêu thương sẽ là nguồn động viên to lớn giúp họ có thêm nghị lực để vượt qua. Chẳng hạn, trong thiên tai, dịch bệnh, chiến tranh, sự đoàn kết, giúp đỡ lẫn nhau chính là minh chứng rõ ràng cho sức mạnh của tình yêu thương. Không chỉ có tác động đến từng cá nhân, tình yêu thương còn làm thay đổi cả xã hội. Một xã hội mà con người biết yêu thương nhau sẽ tràn đầy hạnh phúc, văn minh và phát triển. Khi mọi người biết quan tâm đến nhau, cuộc sống sẽ trở nên ấm áp hơn, tệ nạn xã hội cũng sẽ giảm đi. Trái lại, nếu thiếu tình yêu thương, con người sẽ trở nên ích kỷ, vô cảm, làm mất đi giá trị nhân văn tốt đẹp của cuộc sống.
Tuy nhiên, trong xã hội hiện đại, không ít người đang sống thờ ơ, lạnh lùng với nhau. Họ chỉ quan tâm đến bản thân mà quên đi trách nhiệm đối với cộng đồng. Sự vô cảm này có thể khiến con người dần đánh mất đi giá trị tốt đẹp của chính mình. Vì vậy, chúng ta cần phải lan tỏa tình yêu thương, bắt đầu từ những hành động nhỏ bé như giúp đỡ người già yếu, sẻ chia với những người có hoàn cảnh khó khăn, hay đơn giản chỉ là một lời động viên dành cho người khác khi họ gặp chuyện buồn.
Tình yêu thương là một sức mạnh to lớn giúp con người xích lại gần nhau hơn, làm cho cuộc sống trở nên tốt đẹp hơn. Mỗi người hãy biết yêu thương và chia sẻ để góp phần tạo dựng một thế giới hòa bình, nhân ái. Hãy nhớ rằng, khi trao đi yêu thương, chúng ta cũng sẽ nhận lại hạnh phúc.
"""
pieces = sp.EncodeAsPieces(s)
ids = sp.EncodeAsIds(s)
print(f"Tokens: {pieces}")
print(len(ids))