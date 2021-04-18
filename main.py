from dataset import get_dataloader
from encoder import Encoder
from decoder import Decoder

train_dataloader = get_dataloader()
encoder = Encoder()
decoder = Decoder()
print(encoder)
print(decoder)
for feature, target, feature_length, target_length in train_dataloader:
    output, encoder_hidden, _ = encoder(feature, feature_length)
    result = decoder(encoder_hidden)
    print(result.size())
    break
