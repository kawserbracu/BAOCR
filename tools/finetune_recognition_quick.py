import argparse
import json
import torch
from pathlib import Path
from typing import List
import sys

# Project import path so we can reuse in-repo modules
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from training.train_recognition import RecDataset, collate_fn
from models.recognition import CRNN
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

def freeze_all_but_head(m: CRNN, train_conv_proj: bool = False, train_rnn: bool = False, cnn_tail_layers: int = 0):
    for p in m.parameters():
        p.requires_grad = False
    # Train final fc (and optionally conv_proj)
    for p in m.fc.parameters():
        p.requires_grad = True
    if train_conv_proj and hasattr(m, 'conv_proj') and m.conv_proj is not None:
        for p in m.conv_proj.parameters():
            p.requires_grad = True
    params = list(m.fc.parameters())
    if train_conv_proj and hasattr(m, 'conv_proj') and m.conv_proj is not None:
        params += list(m.conv_proj.parameters())
    if train_rnn and hasattr(m, 'rnn') and m.rnn is not None:
        for p in m.rnn.parameters():
            p.requires_grad = True
        params += list(m.rnn.parameters())
    # Optionally unfreeze last N conv layers of the CNN backbone
    if hasattr(m, 'cnn') and m.cnn is not None and cnn_tail_layers > 0:
        # collect conv layer parameters in order
        conv_param_groups = []
        for mod in m.cnn.modules():
            if isinstance(mod, nn.Conv2d):
                conv_param_groups.append(list(mod.parameters()))
        # flatten and take tail
        flat = [p for group in conv_param_groups for p in group]
        tail = flat[-min(len(flat), cnn_tail_layers):]
        for p in tail:
            p.requires_grad = True
        params += tail
    return params

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True, type=str, help='variant dir: merged_all[_clahe|_highboost]')
    ap.add_argument('--vocab_path', required=True, type=str)
    ap.add_argument('--ckpt_path', required=True, type=str, help='existing rec model_best.pt')
    ap.add_argument('--output_name', required=True, type=str)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--max_steps', type=int, default=600, help='stop early after N steps')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--train_conv_proj', action='store_true', help='also train conv_proj')
    ap.add_argument('--train_rnn', action='store_true', help='also unfreeze and train BiLSTM')
    ap.add_argument('--train_cnn_tail', type=int, default=0, help='unfreeze last N Conv2d params of CNN backbone')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = data_dir / 'runs' / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    train_json = data_dir / 'recognition_train.json'
    val_json = data_dir / 'recognition_val.json'
    train_ds = RecDataset(train_json, vocab_path=Path(args.vocab_path), augment=True)
    val_ds = RecDataset(val_json, vocab_path=Path(args.vocab_path), augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # Model
    from data.tokenizer import BengaliWordOCRTokenizer
    tok = BengaliWordOCRTokenizer(); tok.load_vocab(Path(args.vocab_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CRNN(vocab_size=tok.vocab_size(), pretrained_backbone=False)
    state = torch.load(args.ckpt_path, map_location='cpu')
    # Handle checkpoints with different vocab/class counts by dropping the old head
    sd = state if 'state_dict' not in state else state['state_dict']
    # Remove classifier weights that may mismatch (fc.*). Keep conv_proj from checkpoint.
    keys_to_drop = [k for k in list(sd.keys()) if k.startswith('fc.') or k.startswith('module.fc.')]
    for k in keys_to_drop:
        sd.pop(k, None)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if keys_to_drop:
        print(f"[load] Dropped head params due to vocab mismatch: {len(keys_to_drop)} keys. Missing: {missing}, Unexpected: {unexpected}")
    model.to(device)

    # Freeze
    head_params = freeze_all_but_head(model, train_conv_proj=args.train_conv_proj, train_rnn=args.train_rnn, cnn_tail_layers=args.train_cnn_tail)
    # Slightly lower LR if RNN/CNN are being trained
    lr = 2.5e-4 if (args.train_rnn or args.train_cnn_tail>0) else 5e-4
    optimizer = Adam(head_params, lr=lr)
    ctc = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    def step_batch(batch):
        model.train()
        images, flat_targets, input_lengths, target_lengths = batch
        images = images.to(device)
        flat_targets = flat_targets.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = target_lengths.to(device)
        logits = model(images)  # (T,N,V)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        T, N, V = log_probs.shape
        maxW = input_lengths.max().clamp(min=1)
        scaled_ilens = torch.clamp((input_lengths.float() / maxW.float()) * float(T), min=1.0, max=float(T)).round().long()
        loss = ctc(log_probs, flat_targets, scaled_ilens, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss.detach().cpu())

    @torch.no_grad()
    def eval_epoch():
        model.eval()
        losses: List[float] = []
        for batch in val_loader:
            images, flat_targets, input_lengths, target_lengths = batch
            images = images.to(device)
            flat_targets = flat_targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)
            logits = model(images)
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            T, N, V = log_probs.shape
            maxW = input_lengths.max().clamp(min=1)
            scaled_ilens = torch.clamp((input_lengths.float() / maxW.float()) * float(T), min=1.0, max=float(T)).round().long()
            loss = ctc(log_probs, flat_targets, scaled_ilens, target_lengths)
            losses.append(float(loss.detach().cpu()))
        return sum(losses)/max(1,len(losses))

    steps = 0
    best = None
    for ep in range(1, args.epochs + 1):
        ep_losses: List[float] = []
        for batch in train_loader:
            l = step_batch(batch)
            ep_losses.append(l)
            steps += 1
            if steps % 50 == 0:
                print(f"ep={ep} step={steps} train_ctc={sum(ep_losses)/len(ep_losses):.4f}")
            if steps >= args.max_steps:
                break
        val_loss = eval_epoch()
        print(f"[VAL] ep={ep} steps={steps} val_ctc={val_loss:.4f}")
        # save best
        if best is None or val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), out_dir / 'model_best_headonly.pt')
            with open(out_dir / 'best.txt','w',encoding='utf-8') as f:
                f.write(f"val_ctc={val_loss:.6f}\nsteps={steps}\nepoch={ep}\n")
        if steps >= args.max_steps:
            break

    print("Finished quick finetune. Best val_ctc:", best, "Saved to", out_dir)
if __name__ == "__main__":
    main()
