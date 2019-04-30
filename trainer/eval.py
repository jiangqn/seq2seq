import torch
from trainer.masked_cross_entropy import masked_cross_entropy
from trainer.ndarray2text import ndarray2text

def nmt_eval(model, data_loader, max_len, criterion, index2word):
    total_samples = 0
    total_loss = 0
    model.eval()
    texts = []
    with torch.no_grad():
        for data in data_loader:
            src, trg = data
            src, trg = src.cuda(), trg.cuda()
            logit = model.translate(src, max_len)
            logit = logit[:, 0:trg.size(1), :].contiguous()
            loss = masked_cross_entropy(logit, trg, criterion)
            total_loss += loss.item() * src.size(0)
            total_samples += src.size(0)
            hyp = logit.argmax(dim=1, keepdim=False)
            hyp = hyp.cpu().numpy()
            texts.extend(ndarray2text(hyp, index2word))
    for i in range(10):
        print(texts[i])
    avg_loss = total_loss / total_samples
    return avg_loss