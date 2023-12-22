def decode_label(encoded_label, label_encoder):
    return label_encoder.inverse_transform([encoded_label])[0]
