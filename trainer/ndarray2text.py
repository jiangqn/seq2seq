from src.module.utils.constants import EOS_INDEX, PAD_INDEX

def ndarray2text(ndarray, index2word):
    texts = []
    for piece in ndarray:
        text = ''
        for x in piece.tolist():
            if x == EOS_INDEX or x == PAD_INDEX:
                break
            text += index2word[x] + ' '
        text = text.strip()
        texts.append(text)
    return texts